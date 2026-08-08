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
from researchclaw.research_queue.workers import LLMPreparationWorker, StaticIdeaProducer


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


def test_config_rejects_negative_total_token_limit() -> None:
    try:
        ResearchQueueConfig.from_mapping(
            {
                "research_queue": {
                    "limits": {"max_total_tokens": -1},
                }
            }
        )
    except ValueError as exc:
        assert "max_total_tokens" in str(exc)
    else:
        raise AssertionError("negative max_total_tokens should fail validation")


def test_config_rejects_negative_generation_interval() -> None:
    try:
        ResearchQueueConfig.from_mapping(
            {
                "research_queue": {
                    "limits": {"generation_interval_sec": -0.1},
                }
            }
        )
    except ValueError as exc:
        assert "generation_interval_sec" in str(exc)
    else:
        raise AssertionError("negative generation_interval_sec should fail")


def test_config_rejects_negative_generation_max_batches() -> None:
    try:
        ResearchQueueConfig.from_mapping(
            {
                "research_queue": {
                    "limits": {"generation_max_batches": -1},
                }
            }
        )
    except ValueError as exc:
        assert "generation_max_batches" in str(exc)
    else:
        raise AssertionError("negative generation_max_batches should fail")


def test_config_rejects_nonpositive_llm_call_timeout() -> None:
    try:
        ResearchQueueConfig.from_mapping(
            {
                "research_queue": {
                    "concurrency": {"llm_call_timeout_sec": 0},
                }
            }
        )
    except ValueError as exc:
        assert "llm_call_timeout_sec" in str(exc)
    else:
        raise AssertionError("nonpositive llm_call_timeout_sec should fail")


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


def test_llm_preparer_accepts_no_previous_revision(tmp_path) -> None:
    class FakeClient:
        def chat(self, messages, **kwargs):
            del messages, kwargs
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"method_summary":"test","treatment":"A",'
                        '"control":"B","primary_metric":"score",'
                        '"requested_gpus":0,"timeout_sec":10,'
                        '"command":["python","experiment.py"],'
                        '"source_files":{"experiment.py":'
                        '"import os\\nprint(os.environ['
                        '\\"RESEARCH_QUEUE_BUDGET_JSON\\"])\\n"},'
                        '"plan":{"cheap_test":"test"}}'
                    ),
                    "model": "fake",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            )()

    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Prepare",
            question="Question?",
            hypothesis="Hypothesis",
            treatment="A",
            control="B",
            primary_metric="score",
        )
    )
    worker = LLMPreparationWorker(
        client=FakeClient(),
        python_executable="python",
        max_gpus_per_run=0,
        max_tokens=4321,
    )

    prepared = worker.prepare(
        idea,
        revision=1,
        budget=ResearchQueueConfig().budget(BudgetLevel.B0),
        previous_revision=None,
        feedback="",
    )

    assert prepared.revision == 1
    assert prepared.plan["method_summary"] == "test"


def test_llm_preparer_rejects_hardcoded_budget_parameters() -> None:
    class FakeClient:
        def chat(self, messages, **kwargs):
            del messages, kwargs
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"method_summary":"test","treatment":"A",'
                        '"control":"B","primary_metric":"score",'
                        '"requested_gpus":0,"timeout_sec":10,'
                        '"command":["python","experiment.py"],'
                        '"source_files":{"experiment.py":"print(1)"},'
                        '"plan":{"cheap_test":"test"}}'
                    ),
                    "model": "fake",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            )()

    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Hardcoded",
            question="Question?",
            hypothesis="Hypothesis",
            treatment="A",
            control="B",
            primary_metric="score",
        )
    )
    worker = LLMPreparationWorker(
        client=FakeClient(),
        python_executable="python",
        max_gpus_per_run=0,
    )

    try:
        worker.prepare(
            idea,
            revision=1,
            budget=ResearchQueueConfig().budget(BudgetLevel.B0),
            previous_revision=None,
            feedback="",
        )
    except ValueError as exc:
        assert "RESEARCH_QUEUE_BUDGET_JSON" in str(exc)
    else:
        raise AssertionError("hardcoded budget parameters should fail")
