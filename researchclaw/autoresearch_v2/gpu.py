"""Logical GPU scheduling over one existing asynchronous pool owner."""

from __future__ import annotations

import math
import os
import socket
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .gpu_lease import (
    SharedGPULeaseRegistry,
    shared_registry_path,
    stable_pool_identity,
)
from .models import JobRecord


class AsyncGPUPool(Protocol):
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
    ) -> Any: ...

    def probe_task(self, task_id: str) -> Any: ...

    def collect_task(self, task_id: str) -> Any: ...

    def cancel_task(self, task_id: str) -> Any: ...


@dataclass(slots=True)
class GPULease:
    task_id: str
    idea_id: str
    job_id: str
    allocated_gpus: int
    state: str = "running"
    probe_failures: int = 0


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    admitted: bool
    allocated_gpus: int = 0
    reason: str = ""
    task_id: str = ""


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


class AdaptiveGPUScheduler:
    """Fair-share scheduler with malleable allocation and short-job backfill."""

    def __init__(
        self,
        *,
        total_gpus: int,
        reserved_gpus: int = 0,
        max_share_per_idea: float = 0.5,
        target_utilization: float = 0.9,
    ) -> None:
        self.total_gpus = max(0, int(total_gpus))
        self.reserved_gpus = max(0, int(reserved_gpus))
        self.max_share_per_idea = float(max_share_per_idea)
        self.target_utilization = float(target_utilization)

    @property
    def usable_gpus(self) -> int:
        return max(0, self.total_gpus - self.reserved_gpus)

    def used(self, leases: Iterable[GPULease]) -> int:
        return sum(
            lease.allocated_gpus
            for lease in leases
            if lease.state
            in {
                "reserved",
                "submitted",
                "running",
                "probe_degraded",
                "orphaned",
            }
        )

    def available(self, leases: Iterable[GPULease]) -> int:
        return max(0, self.usable_gpus - self.used(leases))

    def target_gpus(self, *, pending: bool) -> int:
        if not pending:
            return 0
        return min(
            self.usable_gpus,
            max(1, math.ceil(self.usable_gpus * self.target_utilization)),
        )

    def allocate(
        self,
        job: JobRecord,
        *,
        leases: Iterable[GPULease],
    ) -> AllocationDecision:
        if not job.requires_gpu:
            return AllocationDecision(True, 0, "non_gpu")
        current = list(leases)
        available = self.available(current)
        idea_cap = max(
            job.min_gpus,
            int(self.usable_gpus * self.max_share_per_idea),
        )
        used_by_idea = sum(
            lease.allocated_gpus
            for lease in current
            if lease.idea_id == job.idea_id
            and lease.state in {"submitted", "running"}
        )
        cap = min(
            available,
            max(0, idea_cap - used_by_idea),
            job.max_gpus or job.preferred_gpus or job.min_gpus,
        )
        if cap < job.min_gpus:
            return AllocationDecision(False, 0, "insufficient_capacity")
        preferred = job.preferred_gpus or job.min_gpus
        allocated = max(job.min_gpus, min(preferred, cap))
        return AllocationDecision(True, allocated, "admitted")

    @staticmethod
    def order(
        jobs: Iterable[JobRecord],
        priorities: Mapping[str, float],
    ) -> list[JobRecord]:
        """Information-value priority plus shortest-job backfill."""

        return sorted(
            jobs,
            key=lambda job: (
                -float(priorities.get(job.idea_id, 0.0)),
                job.preferred_gpus or job.min_gpus,
                job.timeout_sec,
                job.created_at,
                job.job_id,
            ),
        )


class GPUBroker:
    """Single logical owner for every physical asynchronous GPU task."""

    def __init__(
        self,
        *,
        pool: AsyncGPUPool,
        scheduler: AdaptiveGPUScheduler,
        probe_failure_threshold: int = 3,
        lease_registry: SharedGPULeaseRegistry | None = None,
        owner_id: str | None = None,
        lease_heartbeat_interval_sec: float | None = None,
    ) -> None:
        self.pool = pool
        self.scheduler = scheduler
        self.probe_failure_threshold = max(
            1,
            int(probe_failure_threshold),
        )
        self.leases: dict[str, GPULease] = {}
        self.lease_registry = lease_registry
        self.owner_id = owner_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        )
        self._lease_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None
        if self.lease_registry is not None:
            self.lease_registry.heartbeat(self.owner_id)
            interval = lease_heartbeat_interval_sec
            if interval is None:
                interval = min(
                    30.0,
                    max(1.0, self.lease_registry.owner_ttl_sec / 3.0),
                )
            if interval > 0:
                self._lease_thread = threading.Thread(
                    target=self._lease_heartbeat_loop,
                    args=(float(interval),),
                    name=f"gpu-lease-{self.owner_id[-12:]}",
                    daemon=True,
                )
                self._lease_thread.start()
        if hasattr(self.pool, "start_keepalive"):
            self.pool.start_keepalive()

    def submit(
        self,
        job: JobRecord,
        *,
        priorities: Mapping[str, float],
    ) -> AllocationDecision:
        del priorities
        if not job.command:
            return AllocationDecision(False, 0, "command_missing")
        existing = self.leases.get(job.job_id)
        if existing is not None:
            return AllocationDecision(
                True,
                existing.allocated_gpus,
                "existing_lease",
                existing.task_id,
            )
        task_id = (
            job.submitted_task_id
            or f"{job.job_id}-attempt-{job.attempt + 1:02d}"
        )
        if self.lease_registry is not None:
            self._refresh_global_leases()
            reservation = self.lease_registry.reserve(
                owner_id=self.owner_id,
                task_id=task_id,
                idea_id=job.idea_id,
                job_id=job.job_id,
                min_gpus=job.min_gpus,
                preferred_gpus=job.preferred_gpus,
                max_gpus=job.max_gpus,
            )
            decision = AllocationDecision(
                reservation.admitted,
                reservation.allocated_gpus,
                reservation.reason,
                task_id if reservation.admitted else "",
            )
        else:
            decision = self.scheduler.allocate(
                job,
                leases=self.leases.values(),
            )
        if not decision.admitted:
            return decision
        try:
            self.pool.submit_task(
                job.command,
                timeout_sec=job.timeout_sec,
                task_id=task_id,
                num_gpus=decision.allocated_gpus,
                num_cpus=2,
                env={
                    "AUTORESEARCH_V2_IDEA_ID": job.idea_id,
                    "AUTORESEARCH_V2_JOB_ID": job.job_id,
                    "AUTORESEARCH_V2_ATTEMPT_ID": job.attempt_id,
                    "AUTORESEARCH_V2_GPU_COUNT": str(
                        decision.allocated_gpus
                    ),
                    "AUTORESEARCH_V2_OUTPUT_DIR": job.expected_output_dir,
                },
            )
        except Exception as submit_error:
            # submit_task uses deterministic IDs. A transport error may occur
            # after durable creation, so probe before permitting a duplicate.
            try:
                probe = self.pool.probe_task(task_id)
            except Exception:  # noqa: BLE001
                if self.lease_registry is not None:
                    self.lease_registry.detach(self.owner_id, task_id)
                raise submit_error
            state = str(_value(probe, "state", "unknown"))
            if state not in {
                "submitted",
                "running",
                "finished",
                "timed_out",
                "lost",
            }:
                if self.lease_registry is not None:
                    self.lease_registry.release(self.owner_id, task_id)
                raise
        if self.lease_registry is not None:
            self.lease_registry.mark_state(
                self.owner_id,
                task_id,
                "submitted",
            )
        self.leases[job.job_id] = GPULease(
            task_id=task_id,
            idea_id=job.idea_id,
            job_id=job.job_id,
            allocated_gpus=decision.allocated_gpus,
            state="submitted",
        )
        return AllocationDecision(
            True,
            decision.allocated_gpus,
            "admitted",
            task_id,
        )

    def adopt(
        self,
        job: JobRecord,
        *,
        task_id: str,
        allocated_gpus: int,
    ) -> None:
        if not task_id:
            raise ValueError("cannot adopt a GPU job without a task id")
        if self.lease_registry is not None:
            self.lease_registry.adopt(
                owner_id=self.owner_id,
                task_id=task_id,
                idea_id=job.idea_id,
                job_id=job.job_id,
                allocated_gpus=allocated_gpus,
            )
        self.leases[job.job_id] = GPULease(
            task_id=task_id,
            idea_id=job.idea_id,
            job_id=job.job_id,
            allocated_gpus=max(1, int(allocated_gpus)),
            state="running",
        )

    def reconcile(self) -> list[tuple[str, dict[str, Any]]]:
        self._refresh_global_leases()
        completed: list[tuple[str, dict[str, Any]]] = []
        for job_id, lease in list(self.leases.items()):
            try:
                probe = self.pool.probe_task(lease.task_id)
            except Exception as exc:  # noqa: BLE001
                lease.probe_failures += 1
                lease.state = "probe_degraded"
                if lease.probe_failures < self.probe_failure_threshold:
                    continue
                if self.lease_registry is not None:
                    self.lease_registry.detach(
                        self.owner_id,
                        lease.task_id,
                    )
                completed.append(
                    (
                        job_id,
                        {
                            "returncode": -1,
                            "elapsed_sec": 0.0,
                            "pool_state": "probe_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "task_id": lease.task_id,
                        },
                    )
                )
                del self.leases[job_id]
                continue
            state = str(_value(probe, "state", "unknown"))
            lease.state = state
            lease.probe_failures = 0
            if state in {"submitted", "running"}:
                if self.lease_registry is not None:
                    self.lease_registry.mark_state(
                        self.owner_id,
                        lease.task_id,
                        state,
                    )
                continue
            try:
                result = self.pool.collect_task(lease.task_id)
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "returncode": -1,
                    "elapsed_sec": float(
                        _value(probe, "elapsed_sec", 0.0) or 0.0
                    ),
                    "pool_state": state,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                payload = {
                    "returncode": int(_value(result, "returncode", -1)),
                    "elapsed_sec": float(
                        _value(result, "elapsed_sec", 0.0) or 0.0
                    ),
                    "stdout": str(_value(result, "stdout", "") or ""),
                    "stderr": str(_value(result, "stderr", "") or ""),
                    "stdout_path": str(
                        _value(result, "stdout_path", "") or ""
                    ),
                    "stderr_path": str(
                        _value(result, "stderr_path", "") or ""
                    ),
                    "result_path": str(
                        _value(result, "result_path", "") or ""
                    ),
                    "remote_dir": str(
                        _value(result, "remote_dir", "") or ""
                    ),
                    "timed_out": bool(
                        _value(result, "timed_out", False)
                    ),
                }
            payload["task_id"] = lease.task_id
            payload["allocated_gpus"] = lease.allocated_gpus
            payload["pool_state"] = payload.get("pool_state", state)
            completed.append((job_id, payload))
            if self.lease_registry is not None:
                self.lease_registry.release(
                    self.owner_id,
                    lease.task_id,
                )
            del self.leases[job_id]
        return completed

    def close(self) -> None:
        self._lease_stop.set()
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=5.0)
        if self.lease_registry is not None:
            for lease in self.leases.values():
                try:
                    state = str(
                        _value(
                            self.pool.probe_task(lease.task_id),
                            "state",
                            "unknown",
                        )
                    )
                except Exception:  # noqa: BLE001
                    state = "unknown"
                if state in {"submitted", "running"}:
                    self.lease_registry.detach(
                        self.owner_id,
                        lease.task_id,
                    )
                elif state in {"finished", "timed_out", "lost", "cancelled"}:
                    self.lease_registry.release(
                        self.owner_id,
                        lease.task_id,
                    )
                else:
                    self.lease_registry.detach(
                        self.owner_id,
                        lease.task_id,
                    )
            self.lease_registry.close_owner(self.owner_id)
        # v2 adopts but does not release the physical pool. It does own the
        # keepalive thread it started for the lifetime of this controller.
        if hasattr(self.pool, "stop_keepalive"):
            self.pool.stop_keepalive()

    def cancel(self, job_id: str) -> None:
        lease = self.leases.pop(job_id, None)
        if lease is not None:
            try:
                self.pool.cancel_task(lease.task_id)
            except Exception:
                if self.lease_registry is not None:
                    self.lease_registry.detach(
                        self.owner_id,
                        lease.task_id,
                    )
                raise
            if self.lease_registry is not None:
                self.lease_registry.release(
                    self.owner_id,
                    lease.task_id,
                )

    def snapshot(self, *, pending_jobs: int = 0) -> dict[str, Any]:
        local_leases = list(self.leases.values())
        global_leases = (
            self.lease_registry.list_leases()
            if self.lease_registry is not None
            else []
        )
        leases = (
            [
                GPULease(
                    lease.task_id,
                    lease.idea_id,
                    lease.job_id,
                    lease.allocated_gpus,
                    lease.state,
                )
                for lease in global_leases
            ]
            if global_leases
            else local_leases
        )
        used = self.scheduler.used(leases)
        usable = self.scheduler.usable_gpus
        return {
            "total_gpus": self.scheduler.total_gpus,
            "reserved_gpus": self.scheduler.reserved_gpus,
            "usable_gpus": usable,
            "allocated_gpus": used,
            "available_gpus": max(0, usable - used),
            "utilization": (used / usable) if usable else 0.0,
            "target_utilization": self.scheduler.target_utilization,
            "target_gpus": self.scheduler.target_gpus(
                pending=bool(pending_jobs)
            ),
            "pending_jobs": max(0, int(pending_jobs)),
            "leases": [
                {
                    "task_id": lease.task_id,
                    "idea_id": lease.idea_id,
                    "job_id": lease.job_id,
                    "allocated_gpus": lease.allocated_gpus,
                    "state": lease.state,
                    "probe_failures": lease.probe_failures,
                }
                for lease in leases
            ],
        }

    def _refresh_global_leases(self) -> None:
        if self.lease_registry is None:
            return
        self.lease_registry.heartbeat(self.owner_id)
        self.lease_registry.reap_stale(self.pool.probe_task)

    def _lease_heartbeat_loop(self, interval_sec: float) -> None:
        while not self._lease_stop.wait(interval_sec):
            try:
                if self.lease_registry is not None:
                    self.lease_registry.heartbeat(self.owner_id)
            except Exception:  # noqa: BLE001, S112
                # Scheduling paths retry the heartbeat synchronously and fail
                # closed if the shared registry remains unavailable.
                continue


def build_clusterbridge_broker(
    pool_config: str,
    *,
    reserved_gpus: int,
    max_share_per_idea: float,
    target_utilization: float,
    probe_failure_threshold: int = 3,
    restore_state: bool = True,
) -> GPUBroker:
    """Adopt a prepared ClusterBridge/Ray pool without owning its lifecycle."""

    from researchclaw.experiment.clusterbridge_pool import (
        ClusterBridgePool,
        PoolNotClaimedError,
        PoolNotReadyError,
    )
    from researchclaw.factory.pool_config import PoolConfigSummary

    summary = PoolConfigSummary.from_file(pool_config)
    pool = ClusterBridgePool.from_file(
        summary.config_path,
        restore_state=restore_state,
    )
    if not pool.claimed:
        raise PoolNotClaimedError(
            "v2 requires an already claimed and prepared shared GPU pool; "
            "it will not claim, prepare, or release physical nodes"
        )
    if not pool.prepared:
        raise PoolNotReadyError(
            "v2 requires an already claimed and prepared shared GPU pool; "
            "it will not claim, prepare, or release physical nodes"
        )
    config = getattr(pool, "config", None)
    nodes = [
        {
            "address": node.address,
            "ray_ip": node.ray_ip,
            "gpu_ids": list(node.gpu_ids),
        }
        for node in getattr(config, "nodes", ())
    ]
    pool_id = stable_pool_identity(
        nodes=nodes,
        expected_total_gpus=summary.expected_total_gpus,
    )
    state_dir = Path(
        getattr(pool, "state_dir", summary.config_path.parent)
    ).resolve()
    lease_registry = SharedGPULeaseRegistry(
        shared_registry_path(state_dir, pool_id),
        pool_id=pool_id,
        total_gpus=summary.expected_total_gpus,
        reserved_gpus=reserved_gpus,
        max_share_per_idea=max_share_per_idea,
        owner_ttl_sec=max(
            60.0,
            float(getattr(config, "renew_interval_sec", 120.0)) * 2.0,
        ),
    )
    return GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(
            total_gpus=summary.expected_total_gpus,
            reserved_gpus=reserved_gpus,
            max_share_per_idea=max_share_per_idea,
            target_utilization=target_utilization,
        ),
        probe_failure_threshold=probe_failure_threshold,
        lease_registry=lease_registry,
    )


def clusterbridge_capacity(pool_config: str) -> int:
    """Read configured GPU capacity without adopting or probing a live pool."""

    from researchclaw.factory.pool_config import PoolConfigSummary

    return PoolConfigSummary.from_file(pool_config).expected_total_gpus
