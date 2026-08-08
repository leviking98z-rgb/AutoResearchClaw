from __future__ import annotations

from researchclaw.research_queue.config import ResearchQueueConfig
from researchclaw.research_queue.models import (
    BudgetLevel,
    Conclusion,
    GenerationBatch,
    IdeaProposal,
    IdeaRecord,
    IdeaStatus,
    RunRecord,
    RunStatus,
)
from researchclaw.research_queue.workers import (
    LLMPreparationWorker,
    LLMReviewWorker,
    StaticIdeaProducer,
    validate_python_sources,
)


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


def test_config_requires_scientific_gate_for_promotion(tmp_path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text("benchmark: {}\n")
    try:
        ResearchQueueConfig.from_mapping(
            {
                "research_queue": {
                    "promotion": {
                        "enabled": True,
                        "benchmark_config": str(benchmark),
                    }
                }
            }
        )
    except ValueError as exc:
        assert "scientific_gate" in str(exc)
    else:
        raise AssertionError("promotion without scientific gate should fail")


def test_config_rejects_single_seed_confirmatory_benchmark(tmp_path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text("benchmark:\n  seeds: [17]\n")
    try:
        ResearchQueueConfig.from_mapping(
            {
                "research_queue": {
                    "scientific_gate": {"enabled": True},
                    "promotion": {
                        "enabled": True,
                        "benchmark_id": "cifar10_calibration",
                        "benchmark_config": str(benchmark),
                    },
                }
            }
        )
    except ValueError as exc:
        assert "at least 2 benchmark seeds" in str(exc)
    else:
        raise AssertionError("single-seed benchmark should fail validation")


def test_progressive_config_requires_disjoint_seed_partitions(tmp_path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        """
benchmark:
  pairing_seeds: [101, 103, 107, 17, 29]
  seeds: [17, 29]
""".lstrip()
    )
    base = {
        "scientific_gate": {"enabled": True},
        "promotion": {
            "enabled": True,
            "benchmark_id": "cifar10_calibration",
            "benchmark_config": str(benchmark),
            "progressive_pilot": True,
        },
        "limits": {
            "max_revisions_per_idea": 1,
            "max_runs_per_budget": 1,
        },
        "budgets": {
            "B0": {"parameters": {"seeds": [101]}},
            "B1": {"parameters": {"seeds": [103, 107]}},
            "B2": {"parameters": {"seeds": [17, 29]}},
        },
    }

    config = ResearchQueueConfig.from_mapping({"research_queue": base})

    assert config.promotion.progressive_pilot is True
    assert config.budgets[BudgetLevel.B2].parameters["seeds"] == [17, 29]

    overlapping = {
        **base,
        "budgets": {
            "B0": {"parameters": {"seeds": [101]}},
            "B1": {"parameters": {"seeds": [101, 107]}},
            "B2": {"parameters": {"seeds": [17, 29]}},
        },
    }
    try:
        ResearchQueueConfig.from_mapping({"research_queue": overlapping})
    except ValueError as exc:
        assert "partitions overlap" in str(exc)
    else:
        raise AssertionError("overlapping progressive partitions should fail")


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


def test_llm_reviewer_compacts_history_and_omits_latest_duplicate() -> None:
    class FakeClient:
        prompt = ""

        def chat(self, messages, **kwargs):
            del kwargs
            self.prompt = messages[0]["content"]
            return type(
                "Response",
                (),
                {
                    "content": (
                        '{"action":"conclude","reason":"bounded evidence",'
                        '"next_budget":null,"conclusion":"negative"}'
                    ),
                    "model": "fake",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            )()

    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Compact review",
            question="Does it improve?",
            hypothesis="It improves.",
            treatment="A",
            control="B",
            primary_metric="score",
        )
    )
    prior = RunRecord(
        run_id="run-prior",
        idea_id=idea.idea_id,
        revision=1,
        budget=BudgetLevel.B0,
        requested_gpus=1,
        timeout_sec=10,
        command=("python", "experiment.py"),
        output_dir="prior",
        status=RunStatus.SUCCEEDED,
        result={
            "ok": True,
            "metrics": {"effect": 0.01},
            "artifacts": [],
        },
    )
    latest = RunRecord(
        run_id="run-latest-unique",
        idea_id=idea.idea_id,
        revision=1,
        budget=BudgetLevel.B1,
        requested_gpus=1,
        timeout_sec=20,
        command=("python", "experiment.py"),
        output_dir="latest",
        status=RunStatus.SUCCEEDED,
        result={
            "ok": True,
            "metrics": {
                "effect": 0.02,
                "effect_ci": [0.01, 0.03],
                "per_seed": [
                    {
                        "seed": index,
                        "effect": index / 1000,
                        "raw_blob": f"private-row-{index:03d}",
                    }
                    for index in range(100)
                ],
            },
            "artifacts": [f"/large/artifact-{index}.json" for index in range(30)],
            "usage": {"budget_parameters": {"examples": 512, "seeds": 100}},
        },
    )
    client = FakeClient()
    decision = LLMReviewWorker(client=client).review(
        idea,
        run=latest,
        history=[prior, latest],
        limits={"remaining_steps_after_review": 0},
    )

    assert decision.conclusion is Conclusion.NEGATIVE
    assert client.prompt.count("run-latest-unique") == 1
    assert "run-prior" in client.prompt
    assert '"row_count": 100' in client.prompt
    assert '"effect_ci": [' in client.prompt
    assert "private-row-099" not in client.prompt
    assert "artifact-29.json" not in client.prompt
    assert len(client.prompt) < 15_000


def test_python_source_gate_allows_stdlib_numpy_and_local_modules() -> None:
    errors = validate_python_sources(
        {
            "helper.py": "VALUE = 1\n",
            "experiment.py": (
                "import json\n"
                "import numpy as np\n"
                "from helper import VALUE\n"
                "print(json.dumps([np.mean([VALUE])]))\n"
            ),
        },
        allowed_imports=("numpy",),
    )

    assert errors == []


def test_python_source_gate_rejects_sklearn_before_run() -> None:
    errors = validate_python_sources(
        {"experiment.py": ("from sklearn.linear_model import LogisticRegression\n")},
        allowed_imports=("numpy",),
    )

    assert any("sklearn" in error for error in errors)


def test_python_source_gate_rejects_syntax_and_dynamic_imports() -> None:
    errors = validate_python_sources(
        {
            "broken.py": "if True print('broken')\n",
            "dynamic.py": "__import__('sklearn')\n",
        },
        allowed_imports=("numpy",),
    )

    assert any("invalid Python syntax" in error for error in errors)
    assert any("dynamic imports" in error for error in errors)
