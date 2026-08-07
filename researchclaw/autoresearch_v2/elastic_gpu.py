"""Elastic ClusterBridge allocation and hot GPU-broker lifecycle."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from researchclaw.cluster import (
    ClusterBridgePoolConfig,
    ClusterNode,
    RayPoolConfig,
)
from researchclaw.experiment.clusterbridge_pool import ClusterBridgePool

from .config import GPUConfig
from .gpu import GPUBroker, build_clusterbridge_broker_from_pool


class ResourceManagerError(RuntimeError):
    """Raised when the central ClusterBridge resource manager fails."""


class ClusterBridgeResourceClient:
    """JSON client around the authoritative ``cb`` resource commands."""

    def __init__(
        self,
        command: str,
        *,
        owner: str,
        timeout_sec: float = 180.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.command = str(Path(command).expanduser())
        self.owner = str(owner).strip()
        self.timeout_sec = float(timeout_sec)
        self._runner = runner

    def snapshot(self) -> dict[str, Any]:
        value = self._run(
            ["resource-status", "--owner", self.owner, "--json"],
            json_output=True,
        )
        snapshot = value.get("snapshot", {})
        if not isinstance(snapshot, Mapping):
            raise ResourceManagerError(
                "resource-status response has no snapshot"
            )
        return dict(snapshot)

    def request(
        self,
        *,
        project: str,
        purpose: str,
        gpus: int,
        duration_min: int,
        allow_cross_cluster: bool,
        gpu_type: str,
        priority: str,
    ) -> None:
        args = [
            "request",
            "--gpus",
            str(gpus),
            "--project",
            project,
            "--purpose",
            purpose,
            "--duration",
            str(duration_min),
            "--priority",
            priority,
        ]
        if allow_cross_cluster:
            args.append("--allow-cross-cluster")
        if gpu_type:
            args.extend(["--gpu-type", gpu_type])
        self._run(args)

    def renew(self, allocation_id: str, *, ttl_min: int) -> None:
        self._run(
            [
                "alloc-renew",
                allocation_id,
                "--ttl",
                str(ttl_min),
            ]
        )

    def release(self, allocation_id: str) -> None:
        self._run(["alloc-release", allocation_id])

    def _run(
        self,
        args: list[str],
        *,
        json_output: bool = False,
    ) -> Any:
        env = {**os.environ, "CB_SID": self.owner}
        completed = self._runner(
            ["bash", self.command, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_sec,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "no output"
            )
            raise ResourceManagerError(
                "ClusterBridge resource command failed with "
                f"exit {completed.returncode}: {detail}"
            )
        if not json_output:
            return completed.stdout
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ResourceManagerError(
                "ClusterBridge resource command returned invalid JSON"
            ) from exc


class ResourceManagedGPUManager:
    """Maintain one elastic allocation and expose a hot-swappable broker."""

    def __init__(
        self,
        config: GPUConfig,
        *,
        task_env: Mapping[str, str] | None = None,
        cache_dir: str = "",
        cache_archive: str = "",
        client: ClusterBridgeResourceClient | None = None,
        pool_factory: Callable[..., ClusterBridgePool] = ClusterBridgePool,
        broker_factory: Callable[..., GPUBroker] = (
            build_clusterbridge_broker_from_pool
        ),
        monotonic: Callable[[], float] = time.monotonic,
        prepare_async: bool = True,
        task_namespace: str = "",
    ) -> None:
        self.config = config
        self.elastic = config.resource_manager
        self.task_env = {
            str(key): str(value)
            for key, value in (task_env or {}).items()
        }
        self.cache_dir = str(cache_dir or "").strip()
        self.cache_archive = str(cache_archive or "").strip()
        self.task_namespace = str(task_namespace or "").strip()
        self.client = client or ClusterBridgeResourceClient(
            self.elastic.cb_command,
            owner=self.elastic.owner,
            timeout_sec=self.elastic.command_timeout_sec,
        )
        self.pool_factory = pool_factory
        self.broker_factory = broker_factory
        self._monotonic = monotonic
        self._prepare_async = bool(prepare_async)
        self._lock = threading.RLock()
        self._broker: GPUBroker | None = None
        self._pool: ClusterBridgePool | None = None
        self._allocation: dict[str, Any] | None = None
        self._request_pending = False
        self._next_reconcile = 0.0
        self._next_renew = 0.0
        self._last_error = ""
        self._state = "starting"
        self._closed = False
        self._spin_paused_at = 0.0
        self._prepare_thread: threading.Thread | None = None
        self._renew_thread: threading.Thread | None = None
        self._reconcile_thread: threading.Thread | None = None
        atexit.register(self._shutdown_hook)

    @property
    def broker(self) -> GPUBroker | None:
        # Pointer reads are atomic under CPython. Keep this non-blocking because
        # the background resource-manager RPC may be waiting on the shared
        # control plane while the Controller still needs to heartbeat.
        return self._broker

    @property
    def configured_capacity(self) -> int:
        allocation = self._allocation
        if allocation is not None:
            return int(
                allocation.get(
                    "gpu_count",
                    self.elastic.desired_gpus,
                )
                or self.elastic.desired_gpus
            )
        return self.elastic.desired_gpus

    def bootstrap(self) -> None:
        """Try once immediately; failure remains retryable on later ticks."""

        self.reconcile(force=True)

    def reconcile(self, *, force: bool = False) -> bool:
        """Reconcile allocation, lease renewal, pool readiness and broker."""

        if not self._prepare_async:
            return self._reconcile_sync(force=force)
        with self._lock:
            if self._closed:
                return False
            now = self._monotonic()
            if not force and now < self._next_reconcile:
                return False
            thread = self._reconcile_thread
            if thread is not None and thread.is_alive():
                return False
            self._next_reconcile = (
                now + self.elastic.reconcile_interval_sec
            )
            self._reconcile_thread = threading.Thread(
                target=self._reconcile_worker,
                name="autoresearch-v2-gpu-reconcile",
                daemon=True,
            )
            self._reconcile_thread.start()
        return False

    def _reconcile_worker(self) -> None:
        self._reconcile_sync(force=True)

    def _reconcile_sync(self, *, force: bool = False) -> bool:
        """Blocking implementation used by tests and the background worker."""

        to_close: GPUBroker | None = None
        early_result: bool | None = None
        with self._lock:
            if self._closed:
                return False
            now = self._monotonic()
            if not force and now < self._next_reconcile:
                return False
            self._next_reconcile = (
                now + self.elastic.reconcile_interval_sec
            )
            previous = self._broker
            try:
                snapshot = self.client.snapshot()
                allocation = self._select_allocation(snapshot)
                if allocation is None:
                    if self._broker_has_running_tasks():
                        raise ResourceManagerError(
                            "allocation unavailable while GPU tasks are "
                            "running; retaining the old broker until tasks "
                            "finish"
                        )
                    to_close = self._take_broker()
                    self._allocation = None
                    self._state = (
                        "waiting_allocation"
                        if self._request_pending
                        else "requesting"
                    )
                    if not self._request_pending and not self._has_request(
                        snapshot
                    ):
                        self.client.request(
                            project=self.elastic.project,
                            purpose=self.elastic.purpose,
                            gpus=self.elastic.desired_gpus,
                            duration_min=self.elastic.duration_min,
                            allow_cross_cluster=(
                                self.elastic.allow_cross_cluster
                            ),
                            gpu_type=self.elastic.gpu_type,
                            priority=self.elastic.priority,
                        )
                        self._request_pending = True
                        # The manager may grant synchronously. Pick it up now
                        # so an otherwise idle controller does not wait a full
                        # reconcile interval.
                        snapshot = self.client.snapshot()
                        allocation = self._select_allocation(snapshot)
                    if allocation is None:
                        self._request_pending = self._has_request(snapshot)
                        self._last_error = ""
                        early_result = previous is not self._broker

                if allocation is not None:
                    self._request_pending = False
                    allocation_id = str(allocation.get("id", ""))
                    current_id = str(
                        (self._allocation or {}).get("id", "")
                    )
                    if current_id and current_id != allocation_id:
                        if self._broker_has_running_tasks():
                            raise ResourceManagerError(
                                "allocation changed while GPU tasks are "
                                "running; waiting for the old broker to drain"
                            )
                        to_close = self._take_broker()
                    self._allocation = dict(allocation)
                    if now >= self._next_renew:
                        self._start_renew(allocation_id)
                    if self._broker is None:
                        self._start_attach(allocation)
                    elif now >= self._next_spin_pause_at():
                        self._start_spin_pause()
                    if self._broker is not None:
                        self._state = "ready"
                    self._last_error = ""
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._state = (
                    "degraded" if self._broker is not None else "unavailable"
                )
            changed = previous is not self._broker
        if to_close is not None:
            to_close.close()
        return changed if early_result is None else early_result

    def close(self) -> None:
        broker: GPUBroker | None = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            allocation_id = str(
                (self._allocation or {}).get("id", "")
            )
            broker = self._take_broker()
            if self.elastic.release_on_shutdown and allocation_id:
                try:
                    self.client.release(allocation_id)
                except Exception as exc:  # noqa: BLE001
                    self._last_error = f"{type(exc).__name__}: {exc}"
            self._state = "closed"
        if broker is not None:
            broker.close()
        thread = self._prepare_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        renew_thread = self._renew_thread
        if (
            renew_thread is not None
            and renew_thread is not threading.current_thread()
        ):
            renew_thread.join(timeout=5.0)
        reconcile_thread = self._reconcile_thread
        if (
            reconcile_thread is not None
            and reconcile_thread is not threading.current_thread()
        ):
            reconcile_thread.join(timeout=5.0)

    def _shutdown_hook(self) -> None:
        """Stop pool renewal on abrupt interpreter exit without releasing."""

        with self._lock:
            broker = self._broker
            pool = self._pool
            self._closed = True
        if broker is not None:
            try:
                broker.close()
            except Exception:  # noqa: BLE001, S110
                pass
        elif pool is not None:
            try:
                pool.stop_keepalive()
            except Exception:  # noqa: BLE001, S110
                pass

    def snapshot(self) -> dict[str, Any]:
        allocation = dict(self._allocation or {})
        return {
            "mode": "resource_manager",
            "state": self._state,
            "desired_gpus": self.elastic.desired_gpus,
            "min_gpus": self.elastic.min_gpus,
            "max_gpus": self.elastic.max_gpus,
            "allocation_id": allocation.get("id", ""),
            "allocated_gpus": int(
                allocation.get("gpu_count", 0) or 0
            ),
            "nodes": list(allocation.get("nodes", ()) or ()),
            "last_error": self._last_error,
            "request_pending": self._request_pending,
        }

    def _start_renew(self, allocation_id: str) -> None:
        """Renew the allocation off the controller heartbeat thread."""

        thread = getattr(self, "_renew_thread", None)
        if thread is not None and thread.is_alive():
            return
        # Schedule the next attempt immediately. The worker records any
        # failure and pulls the next attempt forward to the reconcile cadence.
        self._next_renew = (
            self._monotonic() + self.elastic.renew_interval_sec
        )
        self._renew_thread = threading.Thread(
            target=self._renew_worker,
            args=(allocation_id,),
            name="autoresearch-v2-gpu-renew",
            daemon=True,
        )
        self._renew_thread.start()

    def _renew_worker(self, allocation_id: str) -> None:
        try:
            self.client.renew(
                allocation_id,
                ttl_min=self.elastic.renew_ttl_min,
            )
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._next_renew = min(
                    self._next_renew,
                    self._monotonic()
                    + self.elastic.reconcile_interval_sec,
                )

    def _next_spin_pause_at(self) -> float:
        """Return when the running Ray pool needs its keepalive paused again."""

        if self._spin_paused_at <= 0.0:
            return 0.0
        # ``node_spin.sh pause`` currently suppresses the daemon for 60
        # minutes. Refresh comfortably before that deadline while the
        # allocation remains attached to a live Ray broker.
        return self._spin_paused_at + min(
            45.0 * 60.0,
            max(
                60.0,
                float(self.elastic.renew_interval_sec) * 3.0,
            ),
        )

    def _start_spin_pause(self) -> None:
        """Pause the GPU keepalive while this manager owns a prepared pool."""

        pool = self._pool
        if pool is None:
            return
        try:
            pool.pause_spin()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            return
        self._spin_paused_at = self._monotonic()

    def _select_allocation(
        self,
        snapshot: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        allocations = snapshot.get("allocations", ())
        if not isinstance(allocations, list):
            return None
        owned = [
            dict(item)
            for item in allocations
            if isinstance(item, Mapping)
            and str(item.get("owner", "")) == self.elastic.owner
            and str(item.get("project", "")) == self.elastic.project
            and str(item.get("status", "active")) == "active"
        ]
        if not owned:
            return None
        preferred = self.elastic.preferred_allocation_id.strip()
        if preferred:
            return next(
                (
                    item
                    for item in owned
                    if str(item.get("id", "")) == preferred
                ),
                None,
            )
        owned.sort(
            key=lambda item: (
                -int(item.get("gpu_count", 0) or 0),
                str(item.get("issued_at", "")),
            )
        )
        allocation = owned[0]
        if int(allocation.get("gpu_count", 0) or 0) < (
            self.elastic.min_gpus
        ):
            return None
        return allocation

    def _has_request(self, snapshot: Mapping[str, Any]) -> bool:
        queue = snapshot.get("queue", ())
        if not isinstance(queue, list):
            return False
        return any(
            isinstance(item, Mapping)
            and str(item.get("owner", "")) == self.elastic.owner
            and str(item.get("project", "")) == self.elastic.project
            and str(item.get("status", "queued")) == "queued"
            for item in queue
        )

    def _attach_allocation(self, allocation: Mapping[str, Any]) -> None:
        node_details = allocation.get("node_details", ())
        if not isinstance(node_details, list) or not node_details:
            raise ResourceManagerError(
                "allocation has no node_details for pool materialization"
            )
        nodes: list[ClusterNode] = []
        for item in node_details:
            if not isinstance(item, Mapping):
                continue
            gpu_count = int(item.get("gpus", 0) or 0)
            address = str(item.get("ip", "") or "")
            if not address or gpu_count <= 0:
                continue
            nodes.append(
                ClusterNode(
                    address=address,
                    ray_ip=address,
                    gpu_ids=tuple(range(gpu_count)),
                )
            )
        total = sum(node.gpu_count for node in nodes)
        if total < self.elastic.min_gpus:
            raise ResourceManagerError(
                f"allocation exposes {total} GPUs, below min_gpus "
                f"{self.elastic.min_gpus}"
            )
        pool_id = (
            f"autoresearch-v2-{str(allocation['id']).replace('_', '-')}"
        )
        pool_config = ClusterBridgePoolConfig(
            nodes=tuple(nodes),
            cb_command=self.elastic.cb_command,
            purpose=self.elastic.purpose,
            pool_id=pool_id,
            log_root=self.elastic.log_root,
            claim_ttl_min=self.elastic.renew_ttl_min,
            renew_interval_sec=self.elastic.renew_interval_sec,
            max_renew_failures=3,
            command_timeout_sec=self.elastic.command_timeout_sec,
            parallelism=max(1, len(nodes)),
            expected_total_gpus=total,
            allow_force_claim=False,
            expected_claim_owner=self.elastic.owner,
            prepare_cache_dir=self.cache_dir,
            prepare_cache_archive=self.cache_archive,
            ray=RayPoolConfig(
                command=self.elastic.ray_command,
                python=self.elastic.ray_python,
                head_node=nodes[0].address,
                port=self.elastic.ray_port,
                start_timeout_sec=min(
                    300.0,
                    self.elastic.prepare_timeout_sec,
                ),
                resource_timeout_sec=self.elastic.prepare_timeout_sec,
                poll_interval_sec=3.0,
                stop_timeout_sec=60.0,
            ),
        )
        state_path = pool_config.state_dir / "state.json"
        pool = self.pool_factory(
            pool_config,
            initialize_state=not state_path.is_file(),
        )
        restored = False
        if state_path.is_file():
            try:
                # The allocation is already authoritative lease evidence. Fast
                # restore the durable prepared/Ray flags without re-reading
                # every CephFS claim file; the resource-manager snapshot above
                # has already validated owner, allocation id, nodes and status.
                restored = self._restore_allocated_pool_state(
                    pool,
                    state_path,
                    pool_id=pool_id,
                    node_addresses=[node.address for node in nodes],
                )
            except Exception:  # noqa: BLE001
                restored = False
        if not restored:
            # The central allocation has already projected authoritative claim
            # records. Adopt those records rather than issuing a second request.
            pool.adopt_claimed_lease()
            pool.prepare()
        else:
            # A restored Ray cluster may have outlived the 60-minute pause
            # issued during its original preparation. Reassert the pause before
            # exposing the broker so daemon_gpu cannot consume otherwise idle
            # cards in the shared research pool.
            # Older controllers may also have persisted a terminal legacy
            # LeaseKeepalive failure. Stop it here so pool task probes/submits
            # no longer consult stale per-node claim renewal health; the
            # ResourceManagedGPUManager owns the authoritative allocation
            # renewal in this mode.
            pool.stop_keepalive()
            pool.pause_spin()
        broker = self.broker_factory(
            pool,
            total_gpus=total,
            reserved_gpus=self.config.reserved_gpus,
            max_share_per_idea=self.config.max_share_per_idea,
            target_utilization=self.config.target_utilization,
            probe_failure_threshold=self.config.probe_failure_threshold,
            task_env=self.task_env,
            task_namespace=self.task_namespace,
            manage_pool_keepalive=False,
            lease_registry_path=(
                Path("/root/.local/state/autoresearch-v2/gpu-leases")
                / f"{pool_id}.sqlite3"
            ),
        )
        with self._lock:
            if self._closed:
                broker.close()
                return
            self._pool = pool
            self._broker = broker
            self._spin_paused_at = self._monotonic()

    @staticmethod
    def _restore_allocated_pool_state(
        pool: ClusterBridgePool,
        state_path: Path,
        *,
        pool_id: str,
        node_addresses: list[str],
    ) -> bool:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if str(data.get("pool_id", "")) != pool_id:
            return False
        state_nodes = [
            str(item.get("address", ""))
            for item in data.get("nodes", ())
            if isinstance(item, Mapping)
        ]
        if state_nodes != node_addresses:
            return False
        if not (
            bool(data.get("claimed"))
            and bool(data.get("prepared"))
            and bool(data.get("ray_started"))
        ):
            return False
        restore_allocated = getattr(
            pool,
            "restore_allocated_state",
            None,
        )
        if callable(restore_allocated):
            restore_allocated(data)
        else:
            pool.restore_state()
        return True

    def _start_attach(self, allocation: Mapping[str, Any]) -> None:
        thread = self._prepare_thread
        if thread is not None and thread.is_alive():
            self._state = "preparing"
            return
        if not self._prepare_async:
            self._attach_allocation(allocation)
            return
        self._state = "preparing"
        allocation_copy = dict(allocation)
        self._prepare_thread = threading.Thread(
            target=self._attach_worker,
            args=(allocation_copy,),
            name="autoresearch-v2-gpu-prepare",
            daemon=True,
        )
        self._prepare_thread.start()

    def _attach_worker(self, allocation: Mapping[str, Any]) -> None:
        try:
            with self._lock:
                if self._closed:
                    return
            # Pool preparation can take minutes. Keep it outside the manager
            # lock so controller ticks, status, and shutdown remain responsive.
            self._attach_allocation(allocation)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._state = "unavailable"
        else:
            with self._lock:
                self._last_error = ""
                self._state = "ready"

    def _take_broker(self) -> GPUBroker | None:
        broker = self._broker
        self._broker = None
        self._pool = None
        self._spin_paused_at = 0.0
        return broker

    def _broker_has_running_tasks(self) -> bool:
        broker = self._broker
        return bool(broker is not None and broker.leases)
