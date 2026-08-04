"""Deterministic global scheduling for LLM/CPU work and logical GPU leases."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from .config import FactoryConfig
from .models import (
    ACTIVE_IDEA_STATUSES,
    Idea,
    IdeaStatus,
    LeaseStatus,
    ResourceLease,
    WorkItem,
    WorkKind,
    stable_key,
)


@dataclass(frozen=True, slots=True)
class SchedulingDecision:
    item_id: str
    admitted: bool
    reason: str
    allocated_gpus: int = 0


def _parse_time(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return 0.0


def active_status_count(ideas: Iterable[Idea]) -> Counter[IdeaStatus]:
    return Counter(
        idea.status for idea in ideas if idea.status in ACTIVE_IDEA_STATUSES
    )


class FactoryScheduler:
    """Admission scheduler with fairness, aging, and short-job backfill."""

    def __init__(self, config: FactoryConfig, *, total_gpus: int = 0) -> None:
        self.config = config
        self.total_gpus = max(0, int(total_gpus))
        self.max_gpus_per_node = self.total_gpus

    def active_slots_available(self, ideas: Iterable[Idea]) -> int:
        active = sum(idea.status in ACTIVE_IDEA_STATUSES for idea in ideas)
        return max(0, self.config.population.max_active_ideas - active)

    def status_slot_available(
        self,
        status: IdeaStatus,
        ideas: Iterable[Idea],
    ) -> bool:
        counts = active_status_count(ideas)
        limits = {
            IdeaStatus.SCREENING: self.config.population.max_screening_ideas,
            IdeaStatus.PILOT: self.config.population.max_pilot_ideas,
            IdeaStatus.SMOKE: self.config.population.max_pilot_ideas,
            IdeaStatus.VALIDATING: self.config.population.max_validation_ideas,
            IdeaStatus.PAPER: self.config.population.max_paper_ideas,
        }
        limit = limits.get(status)
        return limit is None or counts[status] < limit

    def order(self, items: Iterable[WorkItem], ideas: Iterable[Idea]) -> list[WorkItem]:
        priorities = {idea.idea_id: idea.priority for idea in ideas}
        return sorted(
            items,
            key=lambda item: (
                -priorities.get(item.idea_id, 0.0),
                item.resources.preferred_gpus
                if self.config.scheduler.backfill_enabled
                else 0,
                _parse_time(item.created_at),
                item.item_id,
            ),
        )

    def gpu_capacity(self, leases: Iterable[ResourceLease]) -> int:
        used = sum(
            lease.allocated_gpus
            for lease in leases
            if (
                lease.status.value
                if isinstance(lease.status, LeaseStatus)
                else str(lease.status)
            )
            in {"admitted", "running"}
        )
        return max(
            0,
            self.total_gpus
            - self.config.scheduler.reserved_gpus
            - used,
        )

    def gpu_share_cap(self) -> int:
        usable = max(0, self.total_gpus - self.config.scheduler.reserved_gpus)
        return max(
            1 if usable else 0,
            int(usable * self.config.scheduler.max_gpu_share_per_idea),
        )

    def allocate_gpu(
        self,
        item: WorkItem,
        *,
        leases: Iterable[ResourceLease],
    ) -> SchedulingDecision:
        if item.kind is not WorkKind.GPU_EXPERIMENT:
            return SchedulingDecision(item.item_id, True, "NON_GPU", 0)
        capacity = self.gpu_capacity(leases)
        current_for_idea = sum(
            lease.allocated_gpus
            for lease in leases
            if lease.idea_id == item.idea_id
            and (
                lease.status.value
                if isinstance(lease.status, LeaseStatus)
                else str(lease.status)
            )
            in {"admitted", "running"}
        )
        share_remaining = max(0, self.gpu_share_cap() - current_for_idea)
        profile_cap = (
            self.config.scheduler.validation_max_gpus_per_idea
            if item.profile == "validation"
            else self.config.scheduler.pilot_max_gpus_per_idea
        )
        placement_cap = (
            self.max_gpus_per_node
            if item.resources.placement == "single_node"
            else self.total_gpus
        )
        available = min(
            capacity,
            share_remaining,
            profile_cap,
            placement_cap,
            item.resources.max_gpus,
        )
        if available < item.resources.min_gpus:
            return SchedulingDecision(
                item.item_id,
                False,
                "INSUFFICIENT_GPU_CAPACITY",
            )
        allocated = min(
            item.resources.preferred_gpus or item.resources.min_gpus,
            available,
        )
        return SchedulingDecision(
            item.item_id,
            True,
            "GPU_LEASE_ADMITTED",
            allocated,
        )

    @staticmethod
    def lease_for(item: WorkItem, allocated_gpus: int) -> ResourceLease:
        return ResourceLease(
            lease_id=f"lease-{stable_key(item.deterministic_task_id(), length=12)}",
            idea_id=item.idea_id,
            item_id=item.item_id,
            requested_gpus=item.resources.preferred_gpus,
            allocated_gpus=allocated_gpus,
            priority=float(item.metadata.get("priority", 0.0) or 0.0),
            pool_task_id=item.deterministic_task_id(),
        )
