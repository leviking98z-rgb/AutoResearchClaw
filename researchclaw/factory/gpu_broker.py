"""Single-owner logical GPU Broker over a prepared ClusterBridge/Ray pool."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from .models import LeaseStatus, ResourceLease, WorkItem, WorkItemStatus, utc_now
from .scheduler import FactoryScheduler
from .store import FactoryStore


class AsyncPool(Protocol):
    def submit_task(
        self,
        command: str,
        *,
        timeout_sec: float,
        env: dict[str, str] | None = None,
        task_id: str | None = None,
        require_ready: bool = True,
        num_gpus: int = 0,
        num_cpus: int = 1,
    ) -> Any: ...

    def probe_task(self, task_id: str) -> Any: ...

    def collect_task(self, task_id: str) -> Any: ...

    def cancel_task(self, task_id: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class BrokerSubmission:
    admitted: bool
    reason: str
    lease: ResourceLease | None = None


def _probe_state(probe: Any) -> str:
    if isinstance(probe, dict):
        return str(probe.get("state", "unknown"))
    return str(getattr(probe, "state", "unknown"))


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _elapsed_seconds(started_at: str, released_at: str) -> float:
    try:
        return max(
            0.0,
            (
                datetime.fromisoformat(released_at)
                - datetime.fromisoformat(started_at)
            ).total_seconds(),
        )
    except (TypeError, ValueError):
        return 0.0


class GPUBroker:
    """The only Factory component allowed to call the physical pool."""

    def __init__(
        self,
        *,
        pool: AsyncPool,
        store: FactoryStore,
        scheduler: FactoryScheduler,
    ) -> None:
        self.pool = pool
        self.store = store
        self.scheduler = scheduler

    def submit(self, item: WorkItem) -> BrokerSubmission:
        leases = self.store.list_leases()
        if not item.command:
            return BrokerSubmission(False, "GPU_COMMAND_MISSING")

        lease = next(
            (
                current
                for current in leases
                if current.item_id == item.item_id
                and current.status
                not in {LeaseStatus.RELEASED, LeaseStatus.EXPIRED}
            ),
            None,
        )
        if lease is None:
            decision = self.scheduler.allocate_gpu(item, leases=leases)
            if not decision.admitted:
                return BrokerSubmission(False, decision.reason)
            lease = self.scheduler.lease_for(item, decision.allocated_gpus)
            leases.append(lease)
            item.attempt += 1
        elif lease.status is LeaseStatus.RUNNING:
            return BrokerSubmission(True, "EXISTING_RUNNING_LEASE", lease)
        elif lease.status is LeaseStatus.ADMITTED:
            # A durable ADMITTED lease means the attempt number and
            # deterministic task ID were allocated before a crash. Adopt that
            # same attempt rather than advancing to a duplicate retry.
            task_suffix = lease.pool_task_id.rsplit("-attempt-", 1)
            try:
                admitted_attempt = int(task_suffix[1])
            except (IndexError, TypeError, ValueError):
                admitted_attempt = 0
            if admitted_attempt > 0:
                item.attempt = max(item.attempt, admitted_attempt)
        else:
            return BrokerSubmission(
                False,
                "EXISTING_LEASE_NOT_ADMITTABLE",
                lease,
            )
        lease.status = LeaseStatus.ADMITTED
        self.store.save_leases(leases)
        self.store.event(
            "gpu_lease_admitted",
            idea_id=item.idea_id,
            item_id=item.item_id,
            lease_id=lease.lease_id,
            allocated_gpus=lease.allocated_gpus,
        )
        self.store.idea_event(
            item.idea_id,
            "gpu_lease_admitted",
            item_id=item.item_id,
            lease_id=lease.lease_id,
            allocated_gpus=lease.allocated_gpus,
            pool_task_id=lease.pool_task_id,
        )

        try:
            self.pool.submit_task(
                item.command,
                timeout_sec=item.resources.timeout_sec,
                task_id=lease.pool_task_id,
                num_gpus=lease.allocated_gpus,
                num_cpus=item.resources.cpus,
                env={
                    "RESEARCHCLAW_FACTORY_ID": self.store.factory_id,
                    "RESEARCHCLAW_IDEA_ID": item.idea_id,
                    "RESEARCHCLAW_WORK_ITEM_ATTEMPT": str(item.attempt),
                    "RESEARCHCLAW_WORK_ITEM_ID": item.item_id,
                    "RESEARCHCLAW_GPU_REQUEST": str(lease.allocated_gpus),
                },
            )
        except Exception as exc:
            # Submission transport can fail after the durable pool task was
            # created. Probe the deterministic task id before declaring the
            # lease unused and allowing a duplicate retry.
            try:
                probe = self.pool.probe_task(lease.pool_task_id)
            except Exception:  # noqa: BLE001
                probe = None
            if probe is not None and _probe_state(probe) in {
                "submitted",
                "running",
                "finished",
                "timed_out",
                "lost",
            }:
                lease.status = LeaseStatus.RUNNING
                lease.started_at = lease.started_at or utc_now()
                item.status = WorkItemStatus.RUNNING
                item.metadata["allocated_gpus"] = lease.allocated_gpus
                self.store.save_leases(leases)
                self.store.save_work_item(
                    item,
                    event_type="gpu_submission_adopted",
                )
                return BrokerSubmission(
                    True,
                    "SUBMISSION_ADOPTED_AFTER_ERROR",
                    lease,
                )
            lease.status = LeaseStatus.RELEASED
            lease.released_at = utc_now()
            self.store.save_leases(leases)
            self.store.event(
                "gpu_submission_rolled_back",
                idea_id=item.idea_id,
                item_id=item.item_id,
                lease_id=lease.lease_id,
                attempt=item.attempt,
                error=f"{type(exc).__name__}: {exc}",
            )
            self.store.idea_event(
                item.idea_id,
                "gpu_submission_rolled_back",
                item_id=item.item_id,
                lease_id=lease.lease_id,
                attempt=item.attempt,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        lease.status = LeaseStatus.RUNNING
        lease.started_at = utc_now()
        item.status = WorkItemStatus.RUNNING
        item.metadata["allocated_gpus"] = lease.allocated_gpus
        self.store.save_leases(leases)
        self.store.save_work_item(item, event_type="gpu_work_item_started")
        return BrokerSubmission(True, "SUBMITTED", lease)

    def can_admit_driver(
        self,
        item: WorkItem,
    ) -> tuple[bool, int, str]:
        """Reserve logical GPU capacity for a pipeline-owned Stage 12 driver."""

        leases = self.store.list_leases()
        existing = next(
            (
                lease
                for lease in leases
                if lease.item_id == item.item_id
                and lease.status
                not in {LeaseStatus.RELEASED, LeaseStatus.EXPIRED}
            ),
            None,
        )
        if existing is not None:
            return True, existing.allocated_gpus, "EXISTING_DRIVER_LEASE"
        decision = self.scheduler.allocate_gpu(item, leases=leases)
        if not decision.admitted:
            return False, 0, decision.reason
        lease = self.scheduler.lease_for(item, decision.allocated_gpus)
        lease.status = LeaseStatus.RUNNING
        lease.started_at = utc_now()
        lease.pool_task_id = ""
        leases.append(lease)
        self.store.save_leases(leases)
        self.store.event(
            "gpu_driver_lease_admitted",
            idea_id=item.idea_id,
            item_id=item.item_id,
            lease_id=lease.lease_id,
            allocated_gpus=lease.allocated_gpus,
        )
        self.store.idea_event(
            item.idea_id,
            "gpu_driver_lease_admitted",
            item_id=item.item_id,
            lease_id=lease.lease_id,
            allocated_gpus=lease.allocated_gpus,
        )
        return True, lease.allocated_gpus, "DRIVER_LEASE_ADMITTED"

    def release_driver(self, item_id: str) -> float:
        leases = self.store.list_leases()
        changed = False
        for lease in leases:
            if lease.item_id != item_id:
                continue
            if lease.status in {LeaseStatus.RELEASED, LeaseStatus.EXPIRED}:
                continue
            lease.status = LeaseStatus.RELEASED
            lease.released_at = utc_now()
            changed = True
        if changed:
            self.store.save_leases(leases)
            released = next(
                (lease for lease in leases if lease.item_id == item_id),
                None,
            )
            elapsed_sec = (
                _elapsed_seconds(released.started_at, released.released_at)
                if released is not None
                else 0.0
            )
            self.store.event(
                "gpu_driver_lease_released",
                item_id=item_id,
                elapsed_sec=elapsed_sec,
            )
            item = self.store.get_work_item(item_id)
            if item is not None:
                item.result.setdefault("elapsed_sec", elapsed_sec)
                self.store.idea_event(
                    item.idea_id,
                    "gpu_driver_lease_released",
                    item_id=item_id,
                    elapsed_sec=elapsed_sec,
                )
            return elapsed_sec
        return 0.0

    def reconcile(self) -> list[WorkItem]:
        changed: list[WorkItem] = []
        leases = self.store.list_leases()
        items = {item.item_id: item for item in self.store.list_work_items()}
        for lease in leases:
            if lease.status is not LeaseStatus.RUNNING:
                continue
            if not lease.pool_task_id:
                # Pipeline-owned Stage 12 drivers are reconciled by the Idea
                # Actor; this Broker still owns their logical accounting.
                continue
            item = items.get(lease.item_id)
            if item is None:
                # Unknown durable lease is fail-closed: cancel its known pool
                # task rather than allowing an unaccounted orphan to continue.
                try:
                    self.pool.cancel_task(lease.pool_task_id)
                except Exception as exc:  # noqa: BLE001
                    self.store.event(
                        "gpu_orphan_cancel_failed",
                        idea_id=lease.idea_id,
                        item_id=lease.item_id,
                        pool_task_id=lease.pool_task_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    continue
                lease.status = LeaseStatus.EXPIRED
                lease.released_at = utc_now()
                self.store.event(
                    "gpu_orphan_cancelled",
                    idea_id=lease.idea_id,
                    item_id=lease.item_id,
                    pool_task_id=lease.pool_task_id,
                )
                continue
            try:
                probe = self.pool.probe_task(lease.pool_task_id)
            except Exception as exc:  # noqa: BLE001
                self.store.event(
                    "gpu_task_probe_failed",
                    idea_id=lease.idea_id,
                    item_id=lease.item_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.store.idea_event(
                    lease.idea_id,
                    "gpu_task_probe_failed",
                    item_id=lease.item_id,
                    pool_task_id=lease.pool_task_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            state = _probe_state(probe)
            if state in {"running", "submitted"}:
                continue
            if state == "finished":
                try:
                    result = self.pool.collect_task(lease.pool_task_id)
                except Exception as exc:  # noqa: BLE001
                    item.status = WorkItemStatus.FAILED
                    item.result = {
                        "failure_reason": "GPU_COLLECT_FAILED",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    self.store.event(
                        "gpu_task_collect_failed",
                        idea_id=lease.idea_id,
                        item_id=lease.item_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    self.store.idea_event(
                        lease.idea_id,
                        "gpu_task_collect_failed",
                        item_id=lease.item_id,
                        pool_task_id=lease.pool_task_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    returncode = int(_result_value(result, "returncode", -1))
                    item.status = (
                        WorkItemStatus.SUCCEEDED
                        if returncode == 0
                        else WorkItemStatus.FAILED
                    )
                    item.result = {
                        "returncode": returncode,
                        "stdout_path": _result_value(
                            result, "stdout_path", ""
                        ),
                        "stderr_path": _result_value(
                            result, "stderr_path", ""
                        ),
                        "elapsed_sec": _result_value(
                            result, "elapsed_sec", 0.0
                        ),
                        "timed_out": bool(
                            _result_value(result, "timed_out", False)
                        ),
                    }
            else:
                item.status = WorkItemStatus.FAILED
                item.result = {"pool_state": state}
            lease.status = LeaseStatus.RELEASED
            lease.released_at = utc_now()
            self.store.save_work_item(item, event_type="gpu_work_item_finished")
            changed.append(item)
        self.store.save_leases(leases)
        return changed

    def cancel_item(self, item: WorkItem) -> None:
        leases = self.store.list_leases()
        elapsed_sec = 0.0
        for lease in leases:
            if lease.item_id != item.item_id:
                continue
            if lease.status in {LeaseStatus.RELEASED, LeaseStatus.EXPIRED}:
                continue
            result = self.pool.cancel_task(lease.pool_task_id)
            try:
                elapsed_sec = max(
                    elapsed_sec,
                    float(_result_value(result, "elapsed_sec", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                pass
            lease.status = LeaseStatus.RELEASED
            lease.released_at = utc_now()
        item.status = WorkItemStatus.CANCELLED
        if elapsed_sec > 0:
            item.result["elapsed_sec"] = elapsed_sec
        self.store.save_leases(leases)
        self.store.save_work_item(item, event_type="gpu_work_item_cancelled")
