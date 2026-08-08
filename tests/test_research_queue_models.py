from __future__ import annotations

from researchclaw.research_queue.config import ResearchQueueConfig
from researchclaw.research_queue.models import (
    BudgetLevel,
    Conclusion,
    GenerationBatch,
    IdeaProposal,
    IdeaRecord,
    IdeaStatus,
)
from researchclaw.research_queue.workers import StaticIdeaProducer


def test_budget_levels_advance_in_one_direction() -> None:
    assert BudgetLevel.B0.next() is BudgetLevel.B1
    assert BudgetLevel.B1.next() is BudgetLevel.B2
    assert BudgetLevel.B2.next() is None


def test_idea_round_trip_has_only_four_statuses() -> None:
    proposal = IdeaProposal(
        title="Test",
        question="Does it work?",
        hypothesis="It works.",
        treatment="Treatment",
        control="Control",
        primary_metric="score",
    )
    idea = IdeaRecord.from_proposal(proposal)
    idea.status = IdeaStatus.CONCLUDED
    idea.conclusion = Conclusion.POSITIVE
    restored = IdeaRecord.from_mapping(idea.to_dict())
    assert restored == idea
    assert {item.value for item in IdeaStatus} == {
        "candidate",
        "active",
        "concluded",
        "quarantined",
    }


def test_config_resolves_relative_paths(tmp_path) -> None:
    config = ResearchQueueConfig.from_mapping(
        {
            "research_queue": {
                "enabled": True,
                "state_dir": "state",
                "artifact_dir": "artifacts",
                "models": {"researchclaw_config": "models.yaml"},
                "gpu": {
                    "max_total_gpus": 4,
                    "max_gpus_per_run": 2,
                },
            }
        },
        base_dir=tmp_path,
    )
    assert config.root == tmp_path / "state"
    assert config.artifact_root == tmp_path / "artifacts"
    assert config.models.researchclaw_config == str(tmp_path / "models.yaml")


def test_static_idea_producer_reports_exhaustion() -> None:
    proposal = IdeaProposal(
        title="Finite",
        question="Question?",
        hypothesis="Hypothesis",
        treatment="A",
        control="B",
        primary_metric="score",
    )
    producer = StaticIdeaProducer([proposal])

    batch = producer.generate(2, existing=[])

    assert isinstance(batch, GenerationBatch)
    assert [idea.title for idea in batch.ideas] == ["Finite"]
    assert batch.exhausted is True
    assert producer.generate(1, existing=[]).exhausted is True


def test_cycling_static_idea_titles_remain_unique_across_batches() -> None:
    proposals = [
        IdeaProposal(
            title=f"Cycle {index}",
            question="Question?",
            hypothesis="Hypothesis",
            treatment="A",
            control="B",
            primary_metric="score",
        )
        for index in (1, 2)
    ]
    producer = StaticIdeaProducer(proposals, cycle=True)

    first = producer.generate(3, existing=[])
    second = producer.generate(3, existing=[])
    titles = [idea.title for idea in [*first.ideas, *second.ideas]]

    assert titles == [
        "Cycle 1 #1",
        "Cycle 2 #2",
        "Cycle 1 #3",
        "Cycle 2 #4",
        "Cycle 1 #5",
        "Cycle 2 #6",
    ]
    assert first.exhausted is False
    assert second.exhausted is False
