from __future__ import annotations

import pytest

from researchclaw.factory.models import (
    BudgetTier,
    GateAction,
    GateDecision,
    Idea,
    IdeaStatus,
    ResourceRequest,
    WorkItem,
    WorkKind,
)


def _idea() -> Idea:
    return Idea(
        idea_id="idea-one",
        title="A Falsifiable RSI Study",
        research_question="Does the gate improve held-out performance?",
        falsifiable_hypothesis="The gate improves accuracy by at least 3%.",
        primary_metric="accuracy",
        priority=0.8,
    )


def test_factory_models_round_trip() -> None:
    idea = _idea()
    assert Idea.from_dict(idea.to_dict()) == idea

    item = WorkItem(
        item_id="idea-one-pilot",
        idea_id=idea.idea_id,
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=2,
            max_gpus=4,
        ),
    )
    restored = WorkItem.from_dict(item.to_dict())
    assert restored.resources.preferred_gpus == 2
    assert restored.deterministic_task_id() == "idea-one-pilot-attempt-01"

    decision = GateDecision(
        decision=GateAction.PROMOTE,
        reason_code="PILOT_SIGNAL_VALID",
        current_tier=BudgetTier.PILOT,
        next_tier=BudgetTier.VALIDATION,
        next_status=IdeaStatus.VALIDATING,
    )
    assert GateDecision.from_dict(decision.to_dict()) == decision


def test_resource_request_rejects_inverted_gpu_bounds() -> None:
    with pytest.raises(ValueError, match="min <= preferred <= max"):
        ResourceRequest(min_gpus=2, preferred_gpus=1, max_gpus=4)


def test_idea_priority_is_normalized_probability() -> None:
    with pytest.raises(ValueError, match="priority"):
        data = _idea().to_dict()
        data["priority"] = 1.5
        Idea.from_dict(data)
