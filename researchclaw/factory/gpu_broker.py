"""Single-owner logical GPU Broker over a prepared ClusterBridge/Ray pool."""

from __future__ import annotations

from dataclasses import dataclass
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
        decision = self.scheduler.allocate_gpu(item, leases=leases)
        if not decision.admitted:
            return BrokerSubmission(False, decision.reason)
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
            lease = self.scheduler.lease_for(item, decision.allocated_gpus)
            leases.append(lease)
        lease.status = LeaseStatus.ADMITTED
        lease.allocated_gpus = decision.allocated_gpus
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

        self.pool.submit_task(
            item.command,
            timeout_sec=item.resources.timeout_sec,
            task_id=lease.pool_task_id,
            env={
                "RESEARCHCLAW_IDEA_ID": item.idea_id,
                "RESEARCHCLAW_WORK_ITEM_ID": item.item_id,
                "RESEARCHCLAW_GPU_REQUEST": str(lease.allocated_gpus),
            },
        )
        lease.status = LeaseStatus.RUNNING
        lease.started_at = utc_now()
        item.status = WorkItemStatus.RUNNING
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

    def release_driver(self, item_id: str) -> None:
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
            self.store.event("gpu_driver_lease_released", item_id=item_id)
            item = self.store.get_work_item(item_id)
            if item is not None:
                self.store.idea_event(
                    item.idea_id,
                    "gpu_driver_lease_released",
                    item_id=item_id,
                )

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
                self.pool.cancel_task(lease.pool_task_id)
                lease.status = LeaseStatus.EXPIRED
                lease.released_at = utc_now()
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
                result = self.pool.collect_task(lease.pool_task_id)
                returncode = int(_result_value(result, "returncode", -1))
                item.status = (
                    WorkItemStatus.SUCCEEDED
                    if returncode == 0
                    else WorkItemStatus.FAILED
                )
                item.result = {
                    "returncode": returncode,
                    "stdout_path": _result_value(result, "stdout_path", ""),
                    "stderr_path": _result_value(result, "stderr_path", ""),
                    "elapsed_sec": _result_value(result, "elapsed_sec", 0.0),
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
        for lease in leases:
            if lease.item_id != item.item_id:
                continue
            if lease.status in {LeaseStatus.RELEASED, LeaseStatus.EXPIRED}:
                continue
            self.pool.cancel_task(lease.pool_task_id)
            lease.status = LeaseStatus.RELEASED
            lease.released_at = utc_now()
        item.status = WorkItemStatus.CANCELLED
        self.store.save_leases(leases)
        self.store.save_work_item(item, event_type="gpu_work_item_cancelled")
