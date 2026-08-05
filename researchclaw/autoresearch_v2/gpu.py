"""Logical GPU scheduling over one existing asynchronous pool owner."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

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
            if lease.state in {"submitted", "running"}
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
    ) -> None:
        self.pool = pool
        self.scheduler = scheduler
        self.probe_failure_threshold = max(
            1,
            int(probe_failure_threshold),
        )
        self.leases: dict[str, GPULease] = {}
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
        decision = self.scheduler.allocate(
            job,
            leases=self.leases.values(),
        )
        if not decision.admitted:
            return decision
        task_id = (
            job.submitted_task_id
            or f"{job.job_id}-attempt-{job.attempt + 1:02d}"
        )
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
                raise submit_error
            state = str(_value(probe, "state", "unknown"))
            if state not in {
                "submitted",
                "running",
                "finished",
                "timed_out",
                "lost",
            }:
                raise
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
        self.leases[job.job_id] = GPULease(
            task_id=task_id,
            idea_id=job.idea_id,
            job_id=job.job_id,
            allocated_gpus=max(1, int(allocated_gpus)),
            state="running",
        )

    def reconcile(self) -> list[tuple[str, dict[str, Any]]]:
        completed: list[tuple[str, dict[str, Any]]] = []
        for job_id, lease in list(self.leases.items()):
            try:
                probe = self.pool.probe_task(lease.task_id)
            except Exception as exc:  # noqa: BLE001
                lease.probe_failures += 1
                lease.state = "probe_degraded"
                if lease.probe_failures < self.probe_failure_threshold:
                    continue
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
            del self.leases[job_id]
        return completed

    def close(self) -> None:
        # v2 adopts but does not release the physical pool. It does own the
        # keepalive thread it started for the lifetime of this controller.
        if hasattr(self.pool, "stop_keepalive"):
            self.pool.stop_keepalive()

    def cancel(self, job_id: str) -> None:
        lease = self.leases.pop(job_id, None)
        if lease is not None:
            self.pool.cancel_task(lease.task_id)

    def snapshot(self, *, pending_jobs: int = 0) -> dict[str, Any]:
        leases = list(self.leases.values())
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

    from researchclaw.experiment.clusterbridge_pool import ClusterBridgePool
    from researchclaw.factory.pool_config import PoolConfigSummary

    summary = PoolConfigSummary.from_file(pool_config)
    pool = ClusterBridgePool.from_file(
        summary.config_path,
        restore_state=restore_state,
    )
    if not pool.claimed or not pool.prepared:
        raise RuntimeError(
            "v2 requires an already claimed and prepared shared GPU pool; "
            "it will not claim, prepare, or release physical nodes"
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
    )
