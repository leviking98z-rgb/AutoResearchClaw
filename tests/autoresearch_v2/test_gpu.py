from __future__ import annotations

from researchclaw.autoresearch_v2.gpu import (
    AdaptiveGPUScheduler,
    GPULease,
)
from researchclaw.autoresearch_v2.models import JobKind, JobRecord


def _job(
    idea: str,
    gpus: tuple[int, int, int],
    *,
    timeout: float = 100,
) -> JobRecord:
    return JobRecord(
        job_id=f"{idea}-pilot",
        idea_id=idea,
        kind=JobKind.PILOT,
        requires_gpu=True,
        min_gpus=gpus[0],
        preferred_gpus=gpus[1],
        max_gpus=gpus[2],
        timeout_sec=timeout,
    )


def test_scheduler_malleably_backfills_available_capacity() -> None:
    scheduler = AdaptiveGPUScheduler(
        total_gpus=8,
        reserved_gpus=1,
        max_share_per_idea=0.5,
    )
    leases = [
        GPULease("task-a", "idea-a", "job-a", 3),
        GPULease("task-b", "idea-b", "job-b", 2),
    ]
    decision = scheduler.allocate(
        _job("idea-c", (1, 4, 4)),
        leases=leases,
    )
    assert decision.admitted
    assert decision.allocated_gpus == 2


def test_scheduler_enforces_per_idea_fair_share() -> None:
    scheduler = AdaptiveGPUScheduler(
        total_gpus=8,
        max_share_per_idea=0.5,
    )
    leases = [GPULease("task-a", "idea-a", "job-a", 4)]
    decision = scheduler.allocate(
        _job("idea-a", (1, 2, 2)),
        leases=leases,
    )
    assert not decision.admitted
    assert decision.reason == "insufficient_capacity"


def test_short_job_backfill_breaks_equal_priority_ties() -> None:
    scheduler = AdaptiveGPUScheduler(total_gpus=8)
    long = _job("long", (1, 4, 4), timeout=1000)
    short = _job("short", (1, 1, 1), timeout=10)
    ordered = scheduler.order(
        [long, short],
        priorities={"long": 0.8, "short": 0.8},
    )
    assert [job.idea_id for job in ordered] == ["short", "long"]
