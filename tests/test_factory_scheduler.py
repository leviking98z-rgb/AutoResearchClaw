from __future__ import annotations

from researchclaw.factory.config import FactoryConfig
from researchclaw.factory.models import (
    Idea,
    ResourceLease,
    ResourceRequest,
    WorkItem,
    WorkKind,
)
from researchclaw.factory.scheduler import FactoryScheduler


def _idea(identifier: str, priority: float) -> Idea:
    return Idea(
        idea_id=identifier,
        title=identifier,
        research_question="question",
        falsifiable_hypothesis="hypothesis",
        primary_metric="metric",
        priority=priority,
    )


def _item(identifier: str, idea_id: str, gpus: int) -> WorkItem:
    return WorkItem(
        item_id=identifier,
        idea_id=idea_id,
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=gpus,
            max_gpus=gpus,
        ),
    )


def test_scheduler_fair_share_and_backfill() -> None:
    config = FactoryConfig.from_mapping(
        {
            "factory": {
                "scheduler": {
                    "reserved_gpus": 2,
                    "max_gpu_share_per_idea": 0.5,
                }
            }
        }
    )
    scheduler = FactoryScheduler(config, total_gpus=10)
    ideas = [_idea("a", 0.8), _idea("b", 0.8)]
    small = _item("small", "a", 1)
    large = _item("large", "b", 4)
    assert scheduler.order([large, small], ideas)[0] is small

    existing = ResourceLease(
        lease_id="lease-a",
        idea_id="a",
        item_id="other",
        requested_gpus=4,
        allocated_gpus=4,
        status="running",  # type: ignore[arg-type]
    )
    assert scheduler.allocate_gpu(small, leases=[existing]).admitted is False
    assert scheduler.allocate_gpu(large, leases=[existing]).allocated_gpus == 4


def test_scheduler_enforces_profile_and_single_node_caps() -> None:
    config = FactoryConfig.from_mapping(
        {
            "factory": {
                "scheduler": {
                    "reserved_gpus": 0,
                    "pilot_max_gpus_per_idea": 4,
                    "validation_max_gpus_per_idea": 8,
                    "max_gpu_share_per_idea": 1.0,
                }
            }
        }
    )
    scheduler = FactoryScheduler(config, total_gpus=32)
    scheduler.max_gpus_per_node = 8
    pilot = _item("pilot-large", "a", 16)
    pilot.resources.min_gpus = 1
    assert scheduler.allocate_gpu(pilot, leases=[]).allocated_gpus == 4

    validation = _item("validation-large", "b", 16)
    validation.profile = "validation"
    validation.resources.min_gpus = 9
    assert scheduler.allocate_gpu(validation, leases=[]).admitted is False
