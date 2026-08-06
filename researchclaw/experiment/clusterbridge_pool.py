"""Multi-node ClusterBridge / Ray execution pool.

The pool owns the complete operational lifecycle for a set of explicitly
configured nodes:

``claim -> cleanup -> pause spin -> start Ray -> run task -> stop/cleanup
-> resume spin -> release``.

All node operations are routed through the authoritative
``clusterbridge.sh`` transport.  No SSH implementation or fallback exists.
The module is intentionally standalone so the main ResearchClaw sandbox
factory can wire it in later without changing the existing single-node
sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from researchclaw.cluster import (
    BridgeResult,
    ClusterBridgeClient,
    ClusterBridgePoolConfig,
    ClusterNode,
    KeepaliveSnapshot,
    LeaseKeepalive,
)
from researchclaw.factory.io import append_jsonl

_REMOTE_RESULT_PREFIX = "__RESEARCHCLAW_POOL_RESULT__="
_REMOTE_TASK_ROOT = Path("/tmp/researchclaw-autoresearch-v2")


class ClusterPoolError(RuntimeError):
    """Base class for pool lifecycle and task errors."""


class PoolNotClaimedError(ClusterPoolError):
    """Raised when a node command is attempted without an owned lease."""


class PoolNotReadyError(ClusterPoolError):
    """Raised when Ray has not been prepared and validated."""


class PoolLeaseOwnershipError(ClusterPoolError):
    """Raised when configured nodes are not actively leased by this pool."""


class RayResourceError(ClusterPoolError):
    """Raised when Ray does not expose the configured GPU resources."""


class PoolTaskTimeout(ClusterPoolError):
    """Raised when a submitted task exceeds its wall-clock timeout."""


class PoolTaskNotFinished(ClusterPoolError):
    """Raised when collection is requested for a task that is still running."""


class PoolTaskConflict(ClusterPoolError):
    """Raised when a durable task_id is reused for a different request."""


@dataclass(frozen=True, slots=True)
class RayResources:
    """Ray cluster resources observed on the head node."""

    total_cpu: float
    total_gpu: float
    available_cpu: float
    available_gpu: float
    alive_nodes: int
    raw_cluster_resources: dict[str, float]
    raw_available_resources: dict[str, float]
    nodes: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class PoolTaskResult:
    """Result and log locations for one background head-node task."""

    task_id: str
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float
    timed_out: bool
    remote_dir: str
    stdout_path: str
    stderr_path: str
    result_path: str
    pid: int | None = None
    trusted_gpu_evidence: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PoolTaskHandle:
    """Durable identity returned immediately after asynchronous submission."""

    task_id: str
    pid: int | None
    submitted_at: str
    timeout_sec: float
    remote_dir: str
    request_fingerprint: str


@dataclass(frozen=True, slots=True)
class PoolTaskProbe:
    """Current detached-task state observed on the head node."""

    task_id: str
    state: str
    pid: int | None
    returncode: int | None
    elapsed_sec: float
    timed_out: bool
    stdout: str
    stderr: str
    remote_dir: str


class ClusterBridgePool:
    """Manage claimed nodes and a Ray cluster over ClusterBridge."""

    def __init__(
        self,
        config: ClusterBridgePoolConfig,
        *,
        client: ClusterBridgeClient | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        initialize_state: bool = True,
    ) -> None:
        self.config = config
        self.client = client or ClusterBridgeClient(
            config.cb_command,
            command_timeout_sec=config.command_timeout_sec,
            allow_force_claim=config.allow_force_claim,
        )
        self._sleep = sleep
        self._monotonic = monotonic
        self._claimed = False
        self._prepared = False
        self._ray_started = False
        self._keepalive: LeaseKeepalive | None = None
        self._state_lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._state_dir = config.state_dir
        self._state_dir.mkdir(parents=True, exist_ok=True)
        (self._state_dir / "tasks").mkdir(parents=True, exist_ok=True)
        if initialize_state:
            self._write_state()

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        restore_state: bool = False,
        **kwargs: Any,
    ) -> ClusterBridgePool:
        if restore_state:
            kwargs.setdefault("initialize_state", False)
        pool = cls(ClusterBridgePoolConfig.from_file(path), **kwargs)
        if restore_state:
            pool.restore_state()
        return pool

    @property
    def claimed(self) -> bool:
        with self._state_lock:
            return self._claimed

    @property
    def prepared(self) -> bool:
        with self._state_lock:
            return self._prepared

    @property
    def ray_started(self) -> bool:
        with self._state_lock:
            return self._ray_started

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def status(self, *, probe: bool = False) -> dict[str, Any]:
        """Return durable local state and optionally probe claimed nodes/Ray."""
        status: dict[str, Any] = self._state_payload()
        if probe:
            bridge = self.client.list_nodes()
            status["clusterbridge_list"] = bridge.stdout
            if self.claimed:
                try:
                    status["nodes"] = {
                        node.address: self._node_probe(node)
                        for node in self.config.nodes
                    }
                except Exception as exc:  # noqa: BLE001
                    status["node_probe_error"] = f"{type(exc).__name__}: {exc}"
                if self.ray_started:
                    try:
                        status["ray_resources"] = asdict(
                            self.query_ray_resources()
                        )
                    except Exception as exc:  # noqa: BLE001
                        status["ray_probe_error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
        return status

    def restore_state(self) -> dict[str, Any]:
        """Restore lifecycle flags written by a prior control process.

        This is for a trusted supervisor or CLI continuation.  A state file is
        not itself lease evidence: before exposing ``claimed=True`` this method
        verifies every configured node still has an active claim matching the
        configured owner/purpose.  This prevents a stale controller from
        recreating or releasing a lease that has already expired or changed
        hands.
        """
        state_path = self._state_dir / "state.json"
        if not state_path.is_file():
            raise ClusterPoolError(f"pool state file not found: {state_path}")
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if data.get("pool_id") != self.config.pool_id:
            raise ClusterPoolError("pool state belongs to a different pool_id")
        state_nodes = [
            str(item.get("address"))
            for item in data.get("nodes", ())
            if isinstance(item, Mapping)
        ]
        if state_nodes != self.node_addresses:
            raise ClusterPoolError(
                "pool state node list does not match current configuration"
            )
        durable_claimed = bool(data.get("claimed"))
        if durable_claimed:
            try:
                self.verify_claim_ownership()
            except Exception:
                with self._state_lock:
                    self._claimed = False
                    self._prepared = False
                    self._ray_started = False
                raise
        with self._state_lock:
            self._claimed = durable_claimed
            self._prepared = durable_claimed and bool(data.get("prepared"))
            self._ray_started = durable_claimed and bool(data.get("ray_started"))
        return data

    def restore_allocated_state(
        self,
        data: Mapping[str, Any],
    ) -> None:
        """Restore trusted flags after resource-manager allocation validation.

        Unlike ``restore_state()``, this method does not read per-node claim
        files. It is only for callers that already validated an authoritative
        central allocation containing the exact owner and node set.
        """

        with self._state_lock:
            self._claimed = bool(data.get("claimed"))
            self._prepared = self._claimed and bool(data.get("prepared"))
            self._ray_started = self._prepared and bool(
                data.get("ray_started")
            )

    def verify_claim_ownership(self) -> dict[str, dict[str, object]]:
        """Fail unless every configured node has this pool's active lease."""

        records = self.client.claim_records(self.config.nodes)
        failures: list[str] = []
        expected_owner = self.config.expected_claim_owner
        expected_purpose = self.config.purpose.strip()
        for node in self.config.nodes:
            record = records.get(node.address, {})
            owner = str(record.get("owner", "")).strip()
            purpose = str(record.get("purpose", "")).strip()
            if not bool(record.get("active")):
                failures.append(f"{node.address}: inactive or missing lease")
                continue
            if expected_owner and owner != expected_owner:
                failures.append(
                    f"{node.address}: owner {owner!r} != {expected_owner!r}"
                )
            if expected_purpose and purpose != expected_purpose:
                failures.append(
                    f"{node.address}: purpose {purpose!r} != "
                    f"{expected_purpose!r}"
                )
        if failures:
            raise PoolLeaseOwnershipError(
                "ClusterBridge lease ownership verification failed: "
                + "; ".join(failures)
            )
        return records

    def adopt_claimed_lease(self) -> None:
        """Adopt externally allocated nodes after authoritative verification.

        The central resource manager projects the same ``claim.json`` records
        used by the legacy pool lifecycle.  Elastic controllers therefore do
        not issue a second claim; they verify owner/purpose and durably mark
        the pool claimed before preparing Ray.
        """

        self.verify_claim_ownership()
        with self._state_lock:
            self._claimed = True
            self._prepared = False
            self._ray_started = False
        self._event("lease_adopted", nodes=self.node_addresses)
        self._write_state()

    def claim(self, *, force: bool = False, start_keepalive: bool = True) -> None:
        """Atomically claim all configured nodes.

        ``force`` is rejected by :class:`ClusterBridgeClient` unless the pool
        configuration explicitly enables it.  The default can therefore never
        steal another user's node.
        """
        with self._state_lock:
            if self._claimed:
                if start_keepalive:
                    self.start_keepalive()
                return

        self._event("claim_started", force=force)
        try:
            self.client.claim(
                self.config.nodes,
                purpose=self.config.purpose,
                ttl_min=self.config.claim_ttl_min,
                force=force,
            )
        except BaseException as exc:
            # ClusterBridge's current multi-node claim is not transactional:
            # an unknown/racing node can fail after earlier claim files were
            # written.  Roll back any claims owned by this controller.
            rollback_error: BaseException | None = None
            try:
                self.client.release(self.config.nodes, force=False)
            except BaseException as release_exc:  # noqa: BLE001
                rollback_error = release_exc
            self._event(
                "claim_failed",
                error=f"{type(exc).__name__}: {exc}",
                rollback_error=(
                    f"{type(rollback_error).__name__}: {rollback_error}"
                    if rollback_error is not None
                    else None
                ),
            )
            if rollback_error is not None:
                raise ClusterPoolError(
                    _format_errors("pool claim rollback", [exc, rollback_error])
                ) from exc
            raise
        with self._state_lock:
            self._claimed = True
        self._event("claim_succeeded", nodes=self.node_addresses)
        self._write_state()
        if start_keepalive:
            self.start_keepalive()

    def renew(self) -> None:
        self._require_claimed()
        self.verify_claim_ownership()
        self.client.renew(
            self.config.nodes,
            ttl_min=self.config.claim_ttl_min,
        )
        # A racing writer could replace an advisory claim between the precheck
        # and renew. Re-read before recording success.
        self.verify_claim_ownership()
        self._event("lease_renewed")
        self._write_state()

    def release(
        self,
        *,
        force: bool = False,
        restore_spin: bool = True,
        cleanup: bool = True,
    ) -> None:
        """Best-effort teardown followed by lease release."""
        self.stop_keepalive()
        if not self.claimed:
            return

        # Never clean, spin, or release nodes merely because an old local state
        # file says they were ours. A changed/expired lease is a hard stop.
        self.verify_claim_ownership()

        errors: list[BaseException] = []
        if cleanup or self.ray_started:
            try:
                self.stop_ray(cleanup=cleanup)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        if restore_spin:
            try:
                self.spin()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        try:
            # Once teardown reaches release, relinquish every owned lease even
            # if an earlier cleanup/spin step failed.  The failures are still
            # surfaced below, but the nodes are not left claimed indefinitely.
            self.client.release(self.config.nodes, force=force)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        else:
            with self._state_lock:
                self._claimed = False
                self._prepared = False
                self._ray_started = False
            self._event("lease_released")
            self._write_state()

        if errors:
            raise ClusterPoolError(_format_errors("pool release", errors))

    def start_keepalive(self) -> None:
        self._require_claimed()
        with self._state_lock:
            if self._keepalive is None:
                self._keepalive = LeaseKeepalive(
                    self.renew,
                    interval_sec=self.config.renew_interval_sec,
                    max_failures=self.config.max_renew_failures,
                    on_update=self._keepalive_updated,
                )
            keepalive = self._keepalive
        keepalive.start()
        self._event("keepalive_started")
        self._write_state()

    def stop_keepalive(self) -> None:
        keepalive = self._keepalive
        if keepalive is not None:
            keepalive.stop()
            self._event("keepalive_stopped")
            self._write_state()

    def assert_lease_healthy(self) -> None:
        keepalive = self._keepalive
        if keepalive is not None:
            keepalive.assert_healthy()

    def cleanup_nodes(self) -> dict[str, BridgeResult]:
        self._require_claimed()
        command = (
            "set -euo pipefail; "
            f"bash {shlex.quote(self.config.node_cleanup_script)}"
        )
        results = self._run_all(command, operation="node_cleanup")
        with self._state_lock:
            self._prepared = False
            self._ray_started = False
        self._write_state()
        return results

    def pause_spin(self) -> dict[str, BridgeResult]:
        self._require_claimed()
        command = (
            "set -euo pipefail; "
            f"bash {shlex.quote(self.config.node_spin_script)} pause"
        )
        return self._run_all(command, operation="spin_pause")

    def spin(self) -> dict[str, BridgeResult]:
        self._require_claimed()
        command = (
            "set -euo pipefail; "
            f"bash {shlex.quote(self.config.node_spin_script)} spin"
        )
        return self._run_all(command, operation="spin_resume")

    def prepare(self) -> RayResources:
        """Prepare all nodes, start Ray, and validate the full GPU allocation."""
        self._require_claimed()
        self.assert_lease_healthy()
        self._event("prepare_started")
        try:
            self.cleanup_nodes()
            self.pause_spin()
            self.prepare_cache()
            self.validate_node_gpus()
            self.start_ray()
            resources = self.wait_for_ray_resources(
                timeout_sec=self.config.ray.resource_timeout_sec
            )
        except BaseException:
            self._event("prepare_failed")
            try:
                self.stop_ray(cleanup=True, best_effort=True)
            except Exception as cleanup_error:  # noqa: BLE001
                self._event(
                    "prepare_rollback_error",
                    operation="stop_ray",
                    error=f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
            try:
                self.spin()
            except Exception as spin_error:  # noqa: BLE001
                self._event(
                    "prepare_rollback_error",
                    operation="spin",
                    error=f"{type(spin_error).__name__}: {spin_error}",
                )
            raise

        with self._state_lock:
            self._prepared = True
        self._event("prepare_succeeded", resources=asdict(resources))
        self._write_state()
        return resources

    def prepare_cache(self) -> dict[str, BridgeResult]:
        """Materialize an optional immutable dependency cache per node."""

        self._require_claimed()
        cache_dir = self.config.prepare_cache_dir.strip()
        archive = self.config.prepare_cache_archive.strip()
        if not cache_dir or not archive:
            return {}
        command = (
            "set -euo pipefail; "
            f"archive={shlex.quote(archive)}; "
            f"target={shlex.quote(cache_dir)}; "
            "[ -s \"$archive\" ]; "
            "digest=$(sha256sum \"$archive\" | awk '{print $1}'); "
            "marker=\"$target/.autoresearch-cache-sha256\"; "
            "if [ -f \"$marker\" ] && "
            "[ \"$(cat \"$marker\")\" = \"$digest\" ]; then "
            "printf 'cache-ready %s\\n' \"$digest\"; exit 0; fi; "
            "parent=$(dirname \"$target\"); base=$(basename \"$target\"); "
            "mkdir -p \"$parent\"; "
            "stage=\"$parent/.${base}.stage.$digest\"; "
            "previous=\"$parent/.${base}.previous\"; "
            "if [ -d \"$stage\" ]; then "
            "find \"$stage\" -mindepth 1 -delete; "
            "else mkdir -p \"$stage\"; fi; "
            "if [ -d \"$previous\" ]; then "
            "find \"$previous\" -mindepth 1 -delete; "
            "rmdir \"$previous\"; fi; "
            "tar -xf \"$archive\" -C \"$stage\"; "
            "printf '%s\\n' \"$digest\" > "
            "\"$stage/.autoresearch-cache-sha256\"; "
            "if [ -d \"$target\" ]; then "
            "mv \"$target\" \"$previous\"; fi; "
            "mv \"$stage\" \"$target\"; "
            "if [ -d \"$previous\" ]; then "
            "find \"$previous\" -mindepth 1 -delete; "
            "rmdir \"$previous\"; fi; "
            "printf 'cache-prepared %s\\n' \"$digest\""
        )
        results = self._run_all(
            command,
            operation="cache_prepare",
            timeout_sec=max(
                self.config.command_timeout_sec,
                self.config.ray.resource_timeout_sec,
            ),
        )
        self._event(
            "cache_prepare_succeeded",
            cache_dir=cache_dir,
            archive=archive,
        )
        return results

    def validate_node_gpus(self) -> dict[str, dict[str, Any]]:
        """Verify every configured GPU ID is visible on its assigned node."""
        self._require_claimed()

        def probe(node: ClusterNode) -> tuple[str, dict[str, Any]]:
            configured = ",".join(str(item) for item in node.gpu_ids)
            command = (
                "set -euo pipefail; "
                "command -v nvidia-smi >/dev/null; "
                "visible=$(nvidia-smi --query-gpu=index "
                "--format=csv,noheader,nounits | tr -d ' ' | paste -sd, -); "
                f"configured={shlex.quote(configured)}; "
                "python3 - \"$visible\" \"$configured\" <<'PY'\n"
                "import json, sys\n"
                "visible=[int(x) for x in sys.argv[1].split(',') if x]\n"
                "configured=[int(x) for x in sys.argv[2].split(',') if x]\n"
                "missing=[x for x in configured if x not in visible]\n"
                "print(json.dumps({'visible_gpu_ids': visible, "
                "'configured_gpu_ids': configured, 'missing_gpu_ids': missing}, "
                "sort_keys=True))\n"
                "raise SystemExit(4 if missing else 0)\n"
                "PY"
            )
            result = self.client.run_node(
                node,
                command,
                timeout_sec=self.config.command_timeout_sec,
            )
            payload = _last_json_object(result.stdout)
            return node.address, payload

        validated = self._parallel_nodes(probe, operation="gpu_validation")
        total = sum(
            len(value.get("configured_gpu_ids", ()))
            for value in validated.values()
        )
        if total != self.config.expected_total_gpus:
            raise RayResourceError(
                "configured GPU validation returned "
                f"{total}, expected {self.config.expected_total_gpus}"
            )
        self._event("gpu_validation_succeeded", nodes=validated, total=total)
        return validated

    def start_ray(self) -> None:
        """Start the Ray head and workers as background processes."""
        self._require_claimed()
        self.assert_lease_healthy()
        head = self.config.head_node
        ray_address = f"{head.ray_ip}:{self.config.ray.port}"

        self._event("ray_start_started", head=head.address, address=ray_address)
        # Cleanup should already have stopped Ray, but make this idempotent when
        # callers use start_ray directly.
        self._run_all(
            "if command -v "
            f"{shlex.quote(self.config.ray.command)} >/dev/null 2>&1; then "
            f"{shlex.quote(self.config.ray.command)} stop --force "
            ">/dev/null 2>&1 || true; fi",
            operation="ray_pre_stop",
        )

        head_log = self._state_dir / "ray-head.log"
        head_command = self._ray_start_command(
            head,
            is_head=True,
            address=ray_address,
            log_path=head_log,
        )
        self.client.run_node(
            head,
            head_command,
            timeout_sec=self.config.command_timeout_sec,
        )
        try:
            self._wait_for_ray_head(
                timeout_sec=self.config.ray.start_timeout_sec
            )
            if self.config.worker_nodes:
                self._parallel_nodes(
                    lambda node: (
                        node.address,
                        self.client.run_node(
                            node,
                            self._ray_start_command(
                                node,
                                is_head=False,
                                address=ray_address,
                                log_path=(
                                    self._state_dir
                                    / f"ray-worker-{node.slug}.log"
                                ),
                            ),
                            timeout_sec=self.config.command_timeout_sec,
                        ),
                    ),
                    nodes=self.config.worker_nodes,
                    operation="ray_worker_start",
                )
        except BaseException:
            try:
                self.stop_ray(cleanup=False)
            except Exception as stop_error:  # noqa: BLE001
                self._event(
                    "ray_start_rollback_error",
                    error=f"{type(stop_error).__name__}: {stop_error}",
                )
            raise

        with self._state_lock:
            self._ray_started = True
        self._event("ray_start_succeeded")
        self._write_state()

    def stop_ray(
        self,
        *,
        cleanup: bool = False,
        best_effort: bool = False,
    ) -> None:
        """Stop Ray on all nodes and optionally run the standard cleanup."""
        self._require_claimed()
        errors: list[BaseException] = []
        command = (
            "if command -v "
            f"{shlex.quote(self.config.ray.command)} >/dev/null 2>&1; then "
            f"{shlex.quote(self.config.ray.command)} stop --force "
            ">/dev/null 2>&1 || true; fi"
        )
        try:
            self._run_all(command, operation="ray_stop")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        if cleanup:
            try:
                self.cleanup_nodes()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
        with self._state_lock:
            self._ray_started = False
            if cleanup:
                self._prepared = False
        self._event("ray_stopped", cleanup=cleanup)
        self._write_state()
        if errors and not best_effort:
            raise ClusterPoolError(_format_errors("stop Ray", errors))
        if errors:
            self._event(
                "ray_stop_best_effort_errors",
                errors=[str(error) for error in errors],
            )

    def query_ray_resources(self) -> RayResources:
        """Read cluster and available resources from the Ray head."""
        self._require_claimed()
        head = self.config.head_node
        address = f"{head.ray_ip}:{self.config.ray.port}"
        py = shlex.quote(self.config.ray.python)
        command = (
            "set -euo pipefail; "
            f"RAY_ADDRESS={shlex.quote(address)} {py} - <<'PY'\n"
            "import json, ray\n"
            "ray.init(address='auto', logging_level='ERROR')\n"
            "cluster={str(k): float(v) for k, v in "
            "ray.cluster_resources().items()}\n"
            "available={str(k): float(v) for k, v in "
            "ray.available_resources().items()}\n"
            "nodes=[]\n"
            "for node in ray.nodes():\n"
            "    if not node.get('Alive'): continue\n"
            "    resources=node.get('Resources') or {}\n"
            "    nodes.append({'node_id': str(node.get('NodeID','')), "
            "'node_ip': str(node.get('NodeManagerAddress','')), "
            "'gpu': float(resources.get('GPU',0.0)), "
            "'cpu': float(resources.get('CPU',0.0))})\n"
            "print(json.dumps({'cluster': cluster, 'available': available, "
            "'alive_nodes': len(nodes), 'nodes': nodes}, sort_keys=True))\n"
            "ray.shutdown()\n"
            "PY"
        )
        result = self.client.run_node(
            head,
            command,
            timeout_sec=self.config.command_timeout_sec,
        )
        payload = _last_json_object(result.stdout)
        cluster = _numeric_mapping(payload.get("cluster"))
        available = _numeric_mapping(payload.get("available"))
        return RayResources(
            total_cpu=float(cluster.get("CPU", 0.0)),
            total_gpu=float(cluster.get("GPU", 0.0)),
            available_cpu=float(available.get("CPU", 0.0)),
            available_gpu=float(available.get("GPU", 0.0)),
            alive_nodes=int(payload.get("alive_nodes", 0)),
            raw_cluster_resources=cluster,
            raw_available_resources=available,
            nodes=tuple(
                dict(item)
                for item in payload.get("nodes", ())
                if isinstance(item, Mapping)
            ),
        )

    def wait_for_ray_resources(
        self,
        *,
        timeout_sec: float | None = None,
    ) -> RayResources:
        """Wait until Ray reports every node and every configured GPU."""
        timeout = float(
            timeout_sec
            if timeout_sec is not None
            else self.config.ray.resource_timeout_sec
        )
        deadline = self._monotonic() + timeout
        last: RayResources | None = None
        last_error: BaseException | None = None

        while self._monotonic() < deadline:
            self.assert_lease_healthy()
            try:
                last = self.query_ray_resources()
                if (
                    int(last.total_gpu) == self.config.expected_total_gpus
                    and last.alive_nodes == len(self.config.nodes)
                    and self._ray_nodes_match_config(last)
                ):
                    self._event(
                        "ray_resources_validated",
                        resources=asdict(last),
                    )
                    return last
            except BaseException as exc:  # noqa: BLE001
                last_error = exc
            self._sleep(self.config.ray.poll_interval_sec)

        detail = (
            f"last resources={asdict(last)}"
            if last is not None
            else f"last error={last_error}"
        )
        raise RayResourceError(
            "Ray did not expose the required resources within "
            f"{timeout:.1f}s: expected {self.config.expected_total_gpus} GPUs "
            f"and {len(self.config.nodes)} alive nodes; {detail}"
        )

    def _ray_nodes_match_config(self, resources: RayResources) -> bool:
        """Require the live Ray nodes to be exactly the configured IP/GPU set."""

        if not resources.nodes:
            # Compatibility for injected test doubles and older saved probes.
            return True
        expected = {
            str(node.ray_ip): node.gpu_count for node in self.config.nodes
        }
        observed: dict[str, int] = {}
        for item in resources.nodes:
            node_ip = str(item.get("node_ip", "")).strip()
            try:
                gpu_count = int(float(item.get("gpu", 0)))
            except (TypeError, ValueError):
                return False
            if not node_ip or node_ip in observed:
                return False
            observed[node_ip] = gpu_count
        return observed == expected

    def run_task(
        self,
        command: str,
        *,
        timeout_sec: float,
        env: Mapping[str, str] | None = None,
        task_id: str | None = None,
        require_ready: bool = True,
        poll_interval_sec: float = 2.0,
        num_gpus: int = 0,
        num_cpus: int = 1,
    ) -> PoolTaskResult:
        """Backward-compatible blocking wrapper over the asynchronous API."""
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        handle = self.submit_task(
            command,
            timeout_sec=timeout_sec,
            env=env,
            task_id=task_id,
            require_ready=require_ready,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
        )
        while True:
            probe = self.probe_task(handle.task_id)
            if probe.state == "finished":
                return self.collect_task(handle.task_id)
            if probe.state == "lost":
                result = self.collect_task(handle.task_id)
                raise ClusterPoolError(
                    f"task {handle.task_id} exited without writing result "
                    f"metadata; stderr: {result.stderr[-2000:]}"
                )
            if probe.state == "timed_out":
                result = self.collect_task(handle.task_id)
                raise PoolTaskTimeout(
                    f"task {handle.task_id} exceeded {handle.timeout_sec:.1f}s; "
                    f"logs: {result.stdout_path}, {result.stderr_path}"
                )
            self._sleep(poll_interval_sec)

    def submit_task(
        self,
        command: str,
        *,
        timeout_sec: float,
        env: Mapping[str, str] | None = None,
        task_id: str | None = None,
        require_ready: bool = True,
        num_gpus: int = 0,
        num_cpus: int = 1,
    ) -> PoolTaskHandle:
        """Submit or idempotently adopt one detached task.

        ``task_id`` is a durable idempotency key. Reusing it for the exact
        same normalized request returns the existing handle; changing command,
        environment, or timeout fails closed.
        """
        self._require_claimed()
        if require_ready and not self.prepared:
            raise PoolNotReadyError("pool must be prepared before running a task")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if num_gpus < 0 or num_cpus < 0:
            raise ValueError("Ray resource requests cannot be negative")
        self.assert_lease_healthy()

        task_id = task_id or f"task-{uuid.uuid4().hex[:12]}"
        if not _safe_task_id(task_id):
            raise ValueError(
                "task_id must contain only letters, digits, '.', '_' or '-'"
            )
        task_dir = self._state_dir / "tasks" / task_id
        result_path = task_dir / "result.json"
        pid_path = task_dir / "pid"

        head = self.config.head_node
        ray_address = f"{head.ray_ip}:{self.config.ray.port}"
        env_values = {
            "PYTHONUNBUFFERED": "1",
            "RAY_ADDRESS": ray_address,
            **{str(key): str(value) for key, value in (env or {}).items()},
        }
        invalid_env_names = [
            key for key in env_values if not _safe_env_name(key)
        ]
        if invalid_env_names:
            raise ValueError(
                "invalid environment variable names: "
                + ", ".join(sorted(invalid_env_names))
            )
        request = {
            "schema_version": 1,
            "task_id": task_id,
            "command": command,
            "timeout_sec": float(timeout_sec),
            "env": dict(sorted(env_values.items())),
            "num_gpus": int(num_gpus),
            "num_cpus": int(num_cpus),
        }
        fingerprint = _request_fingerprint(request)
        request["fingerprint"] = fingerprint
        request_path = task_dir / "request.json"
        handle_path = task_dir / "handle.json"
        if task_dir.exists():
            prior = _read_json_mapping(request_path)
            prior_fingerprint = str(prior.get("fingerprint", ""))
            if not prior_fingerprint:
                prior_fingerprint = _request_fingerprint(
                    {
                        key: prior.get(key)
                        for key in (
                            "schema_version",
                            "task_id",
                            "command",
                            "timeout_sec",
                            "env",
                            "num_gpus",
                            "num_cpus",
                        )
                    }
                )
            if prior_fingerprint != fingerprint:
                raise PoolTaskConflict(
                    f"task_id {task_id!r} already belongs to a different request"
                )
            handle_data = _read_json_mapping(handle_path)
            handle = PoolTaskHandle(
                task_id=task_id,
                pid=_optional_int(handle_data.get("pid")),
                submitted_at=str(
                    handle_data.get("submitted_at")
                    or prior.get("submitted_at")
                    or _utc_now()
                ),
                timeout_sec=float(
                    handle_data.get("timeout_sec", timeout_sec)
                ),
                remote_dir=str(task_dir),
                request_fingerprint=fingerprint,
            )
            self._event("task_adopted", task_id=task_id)
            return handle

        task_dir.mkdir(parents=True, exist_ok=False)
        submitted_at = _utc_now()
        request["submitted_at"] = submitted_at
        _atomic_json_write(request_path, request)
        env_text = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in sorted(env_values.items())
        )
        payload_path = task_dir / "ray_task.json"
        task_script = task_dir / "ray_task.py"
        ray_wrapper = (
            "import json, os, pathlib, subprocess, sys, threading, time\n"
            "import ray\n"
            "payload=json.load(open(sys.argv[1], encoding='utf-8'))\n"
            "ray.init(address=os.environ.get('RAY_ADDRESS', 'auto'))\n"
            "@ray.remote(num_gpus=payload['num_gpus'], "
            "num_cpus=payload['num_cpus'])\n"
            "def run(command, env):\n"
            "    child_env=os.environ.copy(); child_env.update(env)\n"
            "    task_context=ray.get_runtime_context()\n"
            "    visible=child_env.get('CUDA_VISIBLE_DEVICES', '')\n"
            "    samples=[]; stop=threading.Event()\n"
            "    query=['nvidia-smi','--query-gpu=index,uuid,name,memory.used,"
            "utilization.gpu','--format=csv,noheader,nounits']\n"
            "    def sample():\n"
            "        while not stop.is_set():\n"
            "            try:\n"
            "                result=subprocess.run(query, text=True, "
            "capture_output=True, timeout=5, check=False)\n"
            "                rows=[]\n"
            "                if result.returncode == 0:\n"
            "                    for line in result.stdout.splitlines():\n"
            "                        parts=[part.strip() for part in "
            "line.split(',', 4)]\n"
            "                        if len(parts) == 5:\n"
            "                            index,uuid,name,memory,utilization=parts\n"
            "                            if visible and index not in "
            "visible.split(',') and uuid not in visible.split(','):\n"
            "                                continue\n"
            "                            rows.append({'index': index, "
            "'uuid': uuid, 'name': name, "
            "'memory_used_mb': float(memory), "
            "'utilization_gpu_percent': float(utilization)})\n"
            "                samples.append({'timestamp_unix': time.time(), "
            "'gpus': rows})\n"
            "            except Exception as exc:\n"
            "                samples.append({'timestamp_unix': time.time(), "
            "'error': f'{type(exc).__name__}: {exc}', 'gpus': []})\n"
            "            stop.wait(0.25)\n"
            "    sampler=threading.Thread(target=sample, daemon=True)\n"
            "    sampler.start()\n"
            "    started=time.time()\n"
            "    try:\n"
            "        completed=subprocess.run(command, shell=True, "
            "executable='/bin/bash', env=child_env)\n"
            "        returncode=int(completed.returncode)\n"
            "    finally:\n"
            "        stop.set(); sampler.join(timeout=6)\n"
            "    if not samples:\n"
            "        try:\n"
            "            result=subprocess.run(query, text=True, "
            "capture_output=True, timeout=5, check=False)\n"
            "            rows=[]\n"
            "            if result.returncode == 0:\n"
            "                for line in result.stdout.splitlines():\n"
            "                    parts=[part.strip() for part in "
            "line.split(',', 4)]\n"
            "                    if len(parts) == 5:\n"
            "                        index,uuid,name,memory,utilization=parts\n"
            "                        if visible and index not in "
            "visible.split(',') and uuid not in visible.split(','):\n"
            "                            continue\n"
            "                        rows.append({'index': index, "
            "'uuid': uuid, 'name': name, "
            "'memory_used_mb': float(memory), "
            "'utilization_gpu_percent': float(utilization)})\n"
            "            samples.append({'timestamp_unix': time.time(), "
            "'gpus': rows})\n"
            "        except Exception as exc:\n"
            "            samples.append({'timestamp_unix': time.time(), "
            "'error': f'{type(exc).__name__}: {exc}', 'gpus': []})\n"
            "    rows=[gpu for item in samples for gpu in item.get('gpus', [])]\n"
            "    evidence={'schema': 'autoresearch_v2.trusted_gpu_evidence', "
            "'version': 1, 'task_id': payload['task_id'], "
            "'ray_task_id': str(task_context.get_task_id()), "
            "'ray_node_id': str(task_context.get_node_id()), "
            "'ray_actor_id': str(task_context.get_actor_id()), "
            "'hostname': __import__('socket').gethostname(), "
            "'cuda_visible_devices': visible, "
            "'allocated_gpus': int(payload['num_gpus']), "
            "'started_at_unix': started, 'ended_at_unix': time.time(), "
            "'returncode': returncode, 'samples': samples, "
            "'gpu_uuids': sorted({row['uuid'] for row in rows}), "
            "'gpu_names': sorted({row['name'] for row in rows}), "
            "'peak_gpu_memory_mb': max([row['memory_used_mb'] "
            "for row in rows] or [0.0]), "
            "'peak_gpu_utilization_percent': max(["
            "row['utilization_gpu_percent'] for row in rows] or [0.0])}\n"
            "    return {'returncode': returncode, 'evidence': evidence}\n"
            "result=ray.get(run.remote(payload['command'], payload['env']))\n"
            "evidence_path=pathlib.Path(payload['evidence_path'])\n"
            "evidence_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "temporary=evidence_path.with_name(evidence_path.name+'.tmp')\n"
            "temporary.write_text(json.dumps(result['evidence'], "
            "sort_keys=True), encoding='utf-8')\n"
            "os.replace(temporary, evidence_path)\n"
            "sys.exit(int(result['returncode']))\n"
        )
        if num_gpus or num_cpus != 1:
            remote_task_dir = (
                _REMOTE_TASK_ROOT / self.config.pool_id / "tasks" / task_id
            )
            remote_payload_path = remote_task_dir / "ray_task.json"
            remote_task_script = remote_task_dir / "ray_task.py"
            remote_trusted_gpu_evidence_path = (
                remote_task_dir / "trusted_gpu_evidence.json"
            )
            _atomic_json_write(
                payload_path,
                {
                    "task_id": task_id,
                    "command": command,
                    "env": dict(sorted(env_values.items())),
                    "num_gpus": int(num_gpus),
                    "num_cpus": int(num_cpus),
                    "evidence_path": str(
                        remote_trusted_gpu_evidence_path
                    ),
                },
            )
            task_script.write_text(ray_wrapper, encoding="utf-8")
            # ``state_dir`` may be controller-local for low-latency pool
            # bookkeeping. Stage the tiny Ray launcher and JSON payload into
            # the head node's local task directory before starting it; do not
            # assume the controller's absolute path is shared with workers.
            staged_files = (
                "mkdir -p "
                f"{shlex.quote(str(remote_task_dir))}; "
                f"cat > {shlex.quote(str(remote_task_script))} <<"
                "'RESEARCHCLAW_RAY_TASK_PY'\n"
                f"{ray_wrapper}"
                "RESEARCHCLAW_RAY_TASK_PY\n"
                f"cat > {shlex.quote(str(remote_payload_path))} <<"
                "'RESEARCHCLAW_RAY_TASK_JSON'\n"
                f"{payload_path.read_text(encoding='utf-8').rstrip()}\n"
                "RESEARCHCLAW_RAY_TASK_JSON\n"
            )
            execution = (
                "{\n"
                f"{staged_files}"
                f"{shlex.quote(self.config.ray.python)} "
                f"{shlex.quote(str(remote_task_script))} "
                f"{shlex.quote(str(remote_payload_path))}\n"
                "}"
            )
            execution_env = ""
            execution_root = remote_task_dir
        else:
            execution = f"bash -lc {shlex.quote(command)}"
            execution_env = f"env {env_text} "
            execution_root = task_dir
        execution_stdout_path = execution_root / "stdout.log"
        execution_stderr_path = execution_root / "stderr.log"
        execution_result_path = execution_root / "result.json"
        inner = (
            "set +e; "
            "trap '' HUP; "
            f"{execution_env}{execution} "
            f"> {shlex.quote(str(execution_stdout_path))} "
            f"2> {shlex.quote(str(execution_stderr_path))}; "
            "rc=$?; "
            f"python3 - \"$rc\" "
            f"{shlex.quote(str(execution_result_path))} <<'PY'\n"
            "import json, os, sys, time\n"
            "path=sys.argv[2]\n"
            "tmp=path+'.tmp.'+str(os.getpid())\n"
            "with open(tmp, 'w', encoding='utf-8') as f:\n"
            "    json.dump({'returncode': int(sys.argv[1]), "
            "'finished_at': time.time()}, f, sort_keys=True)\n"
            "os.replace(tmp, path)\n"
            "PY\n"
            "exit \"$rc\""
        )
        execution_root_setup = (
            f"mkdir -p {shlex.quote(str(execution_root))}; "
            if execution_root != task_dir
            else ""
        )
        start_command = (
            "set -euo pipefail; "
            f"mkdir -p {shlex.quote(str(task_dir))}; "
            f"{execution_root_setup}"
            f"rm -f {shlex.quote(str(result_path))} "
            f"{shlex.quote(str(pid_path))}; "
            # GNU setsid exits as soon as the launched program forks unless
            # --wait is requested. Ray's Python client may daemonize after it
            # connects, so tracking plain `setsid` can make a healthy GPU task
            # look lost before the remote result/evidence files are written.
            f"nohup setsid --wait bash -lc {shlex.quote(inner)} </dev/null "
            f"> {shlex.quote(str(task_dir / 'launcher.log'))} 2>&1 & "
            "pid=$!; "
            f"printf '%s\\n' \"$pid\" > {shlex.quote(str(pid_path))}; "
            "printf '%s\\n' \"$pid\""
        )

        self._event(
            "task_started",
            task_id=task_id,
            head=head.address,
            timeout_sec=timeout_sec,
            command=command,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
        )
        launched = self.client.run_node(
            head,
            start_command,
            timeout_sec=self.config.command_timeout_sec,
        )
        pid = _last_int(launched.stdout)
        handle = PoolTaskHandle(
            task_id=task_id,
            pid=pid,
            submitted_at=submitted_at,
            timeout_sec=float(timeout_sec),
            remote_dir=str(task_dir),
            request_fingerprint=fingerprint,
        )
        _atomic_json_write(handle_path, asdict(handle))
        return handle

    def probe_task(self, task_id: str) -> PoolTaskProbe:
        """Probe a detached task without waiting for completion."""
        self._require_claimed()
        self.assert_lease_healthy()
        if not _safe_task_id(task_id):
            raise ValueError(
                "task_id must contain only letters, digits, '.', '_' or '-'"
            )
        task_dir = self._state_dir / "tasks" / task_id
        if not task_dir.is_dir():
            raise ClusterPoolError(f"pool task does not exist: {task_id}")
        handle = _read_json_mapping(task_dir / "handle.json")
        request = _read_json_mapping(task_dir / "request.json")
        timeout_sec = float(
            handle.get("timeout_sec", request.get("timeout_sec", 0.0))
        )
        submitted_at = str(
            handle.get("submitted_at") or request.get("submitted_at") or ""
        )
        elapsed = _elapsed_since(submitted_at)
        head = self.config.head_node
        remote = self.client.run_node(
            head,
            self._task_probe_command(task_dir),
            timeout_sec=self._task_probe_timeout_sec(),
        )
        payload = _extract_prefixed_json(remote.stdout)
        state = str(payload.get("state", "unknown"))
        timed_out = (
            state == "running"
            and timeout_sec > 0
            and elapsed >= timeout_sec
        )
        if timed_out:
            self._terminate_task(head, task_dir)
            remote = self.client.run_node(
                head,
                self._task_probe_command(task_dir),
                timeout_sec=self._task_probe_timeout_sec(),
            )
            payload = _extract_prefixed_json(remote.stdout)
            state = "timed_out"
            payload["returncode"] = 124
            self._finish_task_result(
                task_id=task_id,
                task_dir=task_dir,
                payload=payload,
                elapsed_sec=elapsed,
                timed_out=True,
                pid=_optional_int(handle.get("pid") or payload.get("pid")),
            )
            self._event("task_timed_out", task_id=task_id)
        elif state in {"finished", "lost"}:
            self._finish_task_result(
                task_id=task_id,
                task_dir=task_dir,
                payload=(
                    payload
                    if state == "finished"
                    else {**payload, "returncode": -1}
                ),
                elapsed_sec=elapsed,
                timed_out=False,
                pid=_optional_int(handle.get("pid") or payload.get("pid")),
            )
        return PoolTaskProbe(
            task_id=task_id,
            state=state,
            pid=_optional_int(handle.get("pid") or payload.get("pid")),
            returncode=_optional_int(payload.get("returncode")),
            elapsed_sec=elapsed,
            timed_out=timed_out,
            stdout=str(payload.get("stdout.log", "")),
            stderr=str(payload.get("stderr.log", "")),
            remote_dir=str(task_dir),
        )

    def collect_task(self, task_id: str) -> PoolTaskResult:
        """Collect a terminal task result without blocking."""
        if not _safe_task_id(task_id):
            raise ValueError(
                "task_id must contain only letters, digits, '.', '_' or '-'"
            )
        task_dir = self._state_dir / "tasks" / task_id
        summary = _read_json_mapping(task_dir / "summary.json")
        if summary:
            return PoolTaskResult(**summary)
        probe = self.probe_task(task_id)
        if probe.state == "running":
            raise PoolTaskNotFinished(f"task {task_id} is still running")
        summary = _read_json_mapping(task_dir / "summary.json")
        if not summary:
            raise ClusterPoolError(
                f"task {task_id} has terminal state {probe.state!r} "
                "without a durable summary"
            )
        return PoolTaskResult(**summary)

    def cancel_task(self, task_id: str) -> PoolTaskResult:
        """Idempotently terminate a detached pool task and persist a summary."""

        self._require_claimed()
        self.verify_claim_ownership()
        if not _safe_task_id(task_id):
            raise ValueError(
                "task_id must contain only letters, digits, '.', '_' or '-'"
            )
        task_dir = self._state_dir / "tasks" / task_id
        if not task_dir.is_dir():
            raise ClusterPoolError(f"pool task does not exist: {task_id}")
        summary = _read_json_mapping(task_dir / "summary.json")
        if summary:
            return PoolTaskResult(**summary)
        head = self.config.head_node
        probe = self.client.run_node(
            head,
            self._task_probe_command(task_dir),
            timeout_sec=self._task_probe_timeout_sec(),
        )
        payload = _extract_prefixed_json(probe.stdout)
        if str(payload.get("state")) == "running":
            self._terminate_task(head, task_dir)
            probe = self.client.run_node(
                head,
                self._task_probe_command(task_dir),
                timeout_sec=self._task_probe_timeout_sec(),
            )
            payload = _extract_prefixed_json(probe.stdout)
        result = self._finish_task_result(
            task_id=task_id,
            task_dir=task_dir,
            payload={**payload, "returncode": 130},
            elapsed_sec=0.0,
            timed_out=False,
            pid=_last_int(str(payload.get("pid", ""))),
        )
        self._event("task_cancelled", task_id=task_id)
        return result

    def close(self) -> None:
        self.release()

    def __enter__(self) -> Self:
        self.claim(start_keepalive=True)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()

    @property
    def node_addresses(self) -> list[str]:
        return [node.address for node in self.config.nodes]

    def _run_all(
        self,
        command: str,
        *,
        operation: str,
        timeout_sec: float | None = None,
        nodes: Iterable[ClusterNode] | None = None,
    ) -> dict[str, BridgeResult]:
        self._require_claimed()

        def run(node: ClusterNode) -> tuple[str, BridgeResult]:
            return (
                node.address,
                self.client.run_node(
                    node,
                    command,
                    timeout_sec=timeout_sec or self.config.command_timeout_sec,
                ),
            )

        return self._parallel_nodes(run, nodes=nodes, operation=operation)

    def _parallel_nodes(
        self,
        fn: Callable[[ClusterNode], tuple[str, Any]],
        *,
        nodes: Iterable[ClusterNode] | None = None,
        operation: str,
    ) -> dict[str, Any]:
        node_list = tuple(nodes or self.config.nodes)
        if not node_list:
            return {}
        results: dict[str, Any] = {}
        errors: list[BaseException] = []
        workers = min(len(node_list), self.config.parallelism)
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=f"cluster-pool-{operation}",
        ) as executor:
            future_map = {executor.submit(fn, node): node for node in node_list}
            for future in as_completed(future_map):
                node = future_map[future]
                try:
                    key, value = future.result()
                    results[key] = value
                    if isinstance(value, BridgeResult):
                        self._write_operation_log(operation, node, value)
                except BaseException as exc:  # noqa: BLE001
                    errors.append(
                        ClusterPoolError(
                            f"{operation} failed on {node.address}: {exc}"
                        )
                    )
        if errors:
            self._event(
                "node_operation_failed",
                operation=operation,
                errors=[str(error) for error in errors],
            )
            raise ClusterPoolError(_format_errors(operation, errors))
        self._event(
            "node_operation_succeeded",
            operation=operation,
            nodes=sorted(results),
        )
        return results

    def _ray_start_command(
        self,
        node: ClusterNode,
        *,
        is_head: bool,
        address: str,
        log_path: Path,
    ) -> str:
        ray = shlex.quote(self.config.ray.command)
        gpu_ids = ",".join(str(item) for item in node.gpu_ids)
        if is_head:
            ray_args = (
                f"--head --node-ip-address={shlex.quote(str(node.ray_ip))} "
                f"--port={self.config.ray.port} "
                f"--num-gpus={node.gpu_count}"
            )
        else:
            ray_args = (
                f"--address={shlex.quote(address)} "
                f"--node-ip-address={shlex.quote(str(node.ray_ip))} "
                f"--num-gpus={node.gpu_count}"
            )
        return (
            "set -euo pipefail; "
            f"mkdir -p {shlex.quote(str(log_path.parent))}; "
            f"rm -f {shlex.quote(str(log_path))}; "
            f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_ids)} "
            f"nohup setsid {ray} start --block {ray_args} "
            f"> {shlex.quote(str(log_path))} 2>&1 < /dev/null & "
            "pid=$!; sleep 1; "
            "if ! kill -0 \"$pid\" 2>/dev/null; then "
            f"cat {shlex.quote(str(log_path))} >&2 || true; exit 1; fi; "
            "printf '%s\\n' \"$pid\""
        )

    def _wait_for_ray_head(self, *, timeout_sec: float) -> None:
        deadline = self._monotonic() + timeout_sec
        last_error: BaseException | None = None
        head = self.config.head_node
        address = f"{head.ray_ip}:{self.config.ray.port}"
        ray = shlex.quote(self.config.ray.command)
        while self._monotonic() < deadline:
            self.assert_lease_healthy()
            try:
                result = self.client.run_node(
                    head,
                    "set -euo pipefail; "
                    f"{ray} status --address={shlex.quote(address)} "
                    ">/dev/null",
                    timeout_sec=self.config.command_timeout_sec,
                )
                if result.returncode == 0:
                    return
            except BaseException as exc:  # noqa: BLE001
                last_error = exc
            self._sleep(self.config.ray.poll_interval_sec)
        raise RayResourceError(
            f"Ray head did not become healthy within {timeout_sec:.1f}s: "
            f"{last_error}"
        )

    def _node_probe(self, node: ClusterNode) -> dict[str, Any]:
        command = (
            "set -euo pipefail; "
            "python3 - <<'PY'\n"
            "import json, socket\n"
            "print(json.dumps({'hostname': socket.gethostname()}))\n"
            "PY\n"
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,"
            "utilization.gpu --format=csv,noheader,nounits"
        )
        result = self.client.run_node(
            node,
            command,
            timeout_sec=self.config.command_timeout_sec,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        metadata: dict[str, Any] = {}
        if lines:
            try:
                metadata = json.loads(lines[0])
            except json.JSONDecodeError:
                metadata = {}
        metadata["gpus"] = lines[1:]
        metadata["configured_gpu_ids"] = list(node.gpu_ids)
        return metadata

    def _task_probe_command(self, task_dir: Path) -> str:
        remote_task_dir = (
            _REMOTE_TASK_ROOT
            / self.config.pool_id
            / "tasks"
            / task_dir.name
        )
        return (
            "set -euo pipefail; "
            f"python3 - {shlex.quote(str(remote_task_dir))} "
            f"{shlex.quote(_REMOTE_RESULT_PREFIX)} <<'PY'\n"
            "import json, os, pathlib, sys\n"
            "root=pathlib.Path(sys.argv[1]); prefix=sys.argv[2]\n"
            "result=root/'result.json'; pidfile=root/'pid'\n"
            "payload={}\n"
            "if result.is_file():\n"
            "    try: payload=json.loads(result.read_text(encoding='utf-8'))\n"
            "    except Exception as exc: payload={'metadata_error': str(exc)}\n"
            "    payload['state']='finished'\n"
            "else:\n"
            "    pid=None\n"
            "    try: pid=int(pidfile.read_text().strip())\n"
            "    except Exception: pass\n"
            "    alive=False\n"
            "    if pid:\n"
            "        try: os.kill(pid, 0); alive=True\n"
            "        except OSError: pass\n"
            "    payload={'state': 'running' if alive else 'lost', 'pid': pid}\n"
            "for name in ('stdout.log','stderr.log'):\n"
            "    path=root/name\n"
            "    try: payload[name]=path.read_text(encoding='utf-8', "
            "errors='replace')\n"
            "    except FileNotFoundError: payload[name]=''\n"
            "evidence=root/'trusted_gpu_evidence.json'\n"
            "if evidence.is_file():\n"
            "    try:\n"
            "        value=json.loads(evidence.read_text(encoding='utf-8'))\n"
            "        if isinstance(value, dict): "
            "payload['trusted_gpu_evidence']=value\n"
            "        else: payload['trusted_gpu_evidence_error']="
            "'evidence is not a JSON object'\n"
            "    except Exception as exc:\n"
            "        payload['trusted_gpu_evidence_error']=str(exc)\n"
            "print(prefix+json.dumps(payload, sort_keys=True))\n"
            "PY"
        )

    def _task_probe_timeout_sec(self) -> float:
        """Keep controller heartbeats responsive when ClusterBridge is busy.

        A task probe only reads a PID/result file and tails two logs.  It must
        not inherit the long lifecycle timeout used for Ray startup or node
        cleanup; otherwise probing several jobs serially can stall one
        controller tick for minutes.
        """

        return max(3.0, min(10.0, float(self.config.command_timeout_sec)))

    def _terminate_task(self, node: ClusterNode, task_dir: Path) -> None:
        grace = self.config.task_kill_grace_sec
        remote_task_dir = (
            _REMOTE_TASK_ROOT
            / self.config.pool_id
            / "tasks"
            / task_dir.name
        )
        command = (
            "set -euo pipefail; "
            f"pid=$(cat {shlex.quote(str(remote_task_dir / 'pid'))} "
            "2>/dev/null || true); "
            "if [ -n \"$pid\" ] && kill -0 \"$pid\" 2>/dev/null; then "
            "kill -TERM -- \"-$pid\" 2>/dev/null || "
            "kill -TERM \"$pid\" 2>/dev/null || true; "
            f"deadline=$((SECONDS+{grace})); "
            "while kill -0 \"$pid\" 2>/dev/null && "
            "[ \"$SECONDS\" -lt \"$deadline\" ]; do sleep 1; done; "
            "if kill -0 \"$pid\" 2>/dev/null; then "
            "kill -KILL -- \"-$pid\" 2>/dev/null || "
            "kill -KILL \"$pid\" 2>/dev/null || true; fi; "
            "fi"
        )
        self.client.run_node(
            node,
            command,
            timeout_sec=grace + self.config.command_timeout_sec,
        )

    def _finish_task_result(
        self,
        *,
        task_id: str,
        task_dir: Path,
        payload: Mapping[str, Any],
        elapsed_sec: float,
        timed_out: bool,
        pid: int | None,
    ) -> PoolTaskResult:
        trusted_gpu_evidence = payload.get("trusted_gpu_evidence")
        if not isinstance(trusted_gpu_evidence, Mapping):
            trusted_gpu_evidence = (
                _read_json_mapping(task_dir / "trusted_gpu_evidence.json")
                or None
            )
        result = PoolTaskResult(
            task_id=task_id,
            returncode=int(payload.get("returncode", 124 if timed_out else -1)),
            stdout=str(payload.get("stdout.log", "")),
            stderr=str(payload.get("stderr.log", "")),
            elapsed_sec=elapsed_sec,
            timed_out=timed_out,
            remote_dir=str(task_dir),
            stdout_path=str(task_dir / "stdout.log"),
            stderr_path=str(task_dir / "stderr.log"),
            result_path=str(task_dir / "result.json"),
            pid=pid,
            trusted_gpu_evidence=(
                dict(trusted_gpu_evidence)
                if isinstance(trusted_gpu_evidence, Mapping)
                else None
            ),
        )
        summary_path = task_dir / "summary.json"
        _atomic_json_write(summary_path, asdict(result))
        self._event(
            "task_finished",
            task_id=task_id,
            returncode=result.returncode,
            elapsed_sec=elapsed_sec,
            timed_out=timed_out,
        )
        return result

    def _require_claimed(self) -> None:
        if not self.claimed:
            raise PoolNotClaimedError(
                "cluster nodes are not claimed; call claim() first"
            )

    def _keepalive_updated(self, snapshot: KeepaliveSnapshot) -> None:
        self._write_state(keepalive=snapshot)

    def _state_payload(
        self,
        *,
        keepalive: KeepaliveSnapshot | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            snapshot = keepalive
            if snapshot is None and self._keepalive is not None:
                snapshot = self._keepalive.snapshot()
            return {
                "pool_id": self.config.pool_id,
                "claimed": self._claimed,
                "prepared": self._prepared,
                "ray_started": self._ray_started,
                "head_node": self.config.head_node.address,
                "nodes": [
                    {
                        "address": node.address,
                        "ray_ip": node.ray_ip,
                        "gpu_ids": list(node.gpu_ids),
                    }
                    for node in self.config.nodes
                ],
                "configured_gpu_count": self.config.configured_gpu_count,
                "expected_total_gpus": self.config.expected_total_gpus,
                "keepalive": asdict(snapshot) if snapshot else None,
                "updated_at": _utc_now(),
            }

    def _write_state(
        self,
        *,
        keepalive: KeepaliveSnapshot | None = None,
    ) -> None:
        _atomic_json_write(
            self._state_dir / "state.json",
            self._state_payload(keepalive=keepalive),
        )

    def _event(self, event: str, **payload: Any) -> None:
        record = {
            "time": _utc_now(),
            "event": event,
            **payload,
        }
        path = self._state_dir / "events.jsonl"
        with self._event_lock:
            append_jsonl(path, record)

    def _write_operation_log(
        self,
        operation: str,
        node: ClusterNode,
        result: BridgeResult,
    ) -> None:
        path = self._state_dir / f"{operation}-{node.slug}.log"
        text = (
            f"$ {' '.join(shlex.quote(part) for part in result.argv)}\n"
            f"exit={result.returncode} elapsed_sec={result.elapsed_sec:.3f}\n"
            "--- stdout ---\n"
            f"{result.stdout}\n"
            "--- stderr ---\n"
            f"{result.stderr}\n"
        )
        path.write_text(text, encoding="utf-8")


def _numeric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            result[str(key)] = float(item)
    return result


def _last_json_object(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ClusterPoolError("remote command did not return a JSON object")


def _extract_prefixed_json(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        if line.startswith(_REMOTE_RESULT_PREFIX):
            value = json.loads(line[len(_REMOTE_RESULT_PREFIX) :])
            if isinstance(value, dict):
                return value
    raise ClusterPoolError("task probe did not return result metadata")


def _last_int(text: str) -> int | None:
    for line in reversed(text.splitlines()):
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _request_fingerprint(value: Mapping[str, Any]) -> str:
    payload = {
        key: value.get(key)
        for key in (
            "schema_version",
            "task_id",
            "command",
            "timeout_sec",
            "env",
            "num_gpus",
            "num_cpus",
        )
    }
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _elapsed_since(timestamp: str) -> float:
    if not timestamp:
        return 0.0
    try:
        started = datetime.fromisoformat(timestamp)
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - started).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def _safe_env_name(value: str) -> bool:
    return bool(value) and (
        value[0].isalpha() or value[0] == "_"
    ) and all(char.isalnum() or char == "_" for char in value)


def _safe_task_id(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= 128
        and value[0].isalnum()
        and all(char.isalnum() or char in "._-" for char in value)
    )


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _format_errors(operation: str, errors: Iterable[BaseException]) -> str:
    details = "; ".join(str(error) for error in errors)
    return f"{operation} encountered errors: {details}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
