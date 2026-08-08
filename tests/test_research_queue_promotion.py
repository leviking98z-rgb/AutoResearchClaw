from __future__ import annotations

import asyncio
import json

from researchclaw.research_queue.benchmark_profile import TREATMENT_API
from researchclaw.research_queue.config import BenchmarkPromotionConfig
from researchclaw.research_queue.models import (
    IdeaProposal,
    IdeaRecord,
    MetricDirection,
    MetricGuardrail,
    MetricRelation,
    ResearchSpec,
    RunResult,
)
from researchclaw.research_queue.promotion import (
    BenchmarkPromotionBridge,
    re_review_artifacts,
    review_benchmark_result,
)
from researchclaw.research_queue.store import ResearchQueueStore


class StaticTreatmentWorker:
    def build(self, idea, *, spec, feedback):
        del idea, spec, feedback
        return (
            """
class Identity:
    def fit(self, calibration_logits, calibration_labels):
        return {}

    def transform(self, logits, state):
        return logits


def build_treatment():
    return Identity()
""".lstrip(),
            {"total_tokens": 7, "model": "fake"},
        )


class RecordingBenchmarkBackend:
    def __init__(self) -> None:
        self.runs = []

    async def run(self, run, *, revision_dir, output_dir, env):
        del revision_dir, env
        self.runs.append(run)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "ok",
            "metrics": {
                "baseline_ece": 0.05,
                "treatment_ece": 0.1,
                "baseline_accuracy": 0.7,
                "treatment_accuracy": 0.7,
                "baseline_nll": 1.0,
                "treatment_nll": 0.9,
            },
            "usage": {"gpu_count": run.requested_gpus},
            "per_seed": [
                {
                    "baseline": {
                        "ece": 0.05,
                        "accuracy": 0.7,
                        "nll": 1.0,
                    },
                    "treatment": {
                        "ece": 0.1,
                        "accuracy": 0.7,
                        "nll": 0.9,
                    },
                    "evidence": {
                        "argmax_preserved": True,
                        "argmax_changed_count": 0,
                        "baseline_predictions_sha256": "a",
                        "treatment_predictions_sha256": "a",
                    },
                },
                {
                    "baseline": {
                        "ece": 0.05,
                        "accuracy": 0.7,
                        "nll": 1.0,
                    },
                    "treatment": {
                        "ece": 0.1,
                        "accuracy": 0.7,
                        "nll": 0.9,
                    },
                    "evidence": {
                        "argmax_preserved": True,
                        "argmax_changed_count": 0,
                        "baseline_predictions_sha256": "b",
                        "treatment_predictions_sha256": "b",
                    },
                },
            ],
            "evidence": {
                "protocol": {
                    "calibration_split": "clean",
                    "evaluation_split": "corrupted",
                    "pairing_strategy": "disjoint_example_blocks",
                },
                "argmax": {
                    "argmax_preserved": True,
                    "argmax_changed_count": 0,
                    "per_example_prediction_hashes": True,
                },
                "compute_accounting": {
                    "matched_dimensions": [
                        "calibration_examples",
                        "evaluation_examples",
                        "model_forward_examples",
                    ],
                    "all_declared_dimensions_matched": True,
                },
            },
        }
        (output_dir / "result.json").write_text(json.dumps(result))
        return RunResult(
            ok=True,
            metrics=result["metrics"],
            usage={"gpu_count": run.requested_gpus, "gpu_seconds": 1.0},
        )

    async def close(self):
        return None

    def snapshot(self):
        return {}


class FailingTreatmentWorker:
    def build(self, idea, *, spec, feedback):
        del idea, spec, feedback
        raise ValueError("structured treatment invalid")


class NeverTreatmentWorker:
    def build(self, idea, *, spec, feedback):
        del idea, spec, feedback
        raise AssertionError("incompatible ResearchSpec reached treatment generation")


def test_promotion_bridge_generates_treatment_and_separates_outcome(
    tmp_path,
) -> None:
    template = tmp_path / "benchmark.yaml"
    template.write_text(
        """
benchmark:
  cache_dir: cache
  output_dir: old-output
  treatment_path: old.py
  device: cuda
  require_cuda: true
""".lstrip()
    )
    store = ResearchQueueStore(tmp_path / "state", artifact_root=tmp_path / "artifacts")
    store.initialize()
    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Promote",
            question="Does it improve calibration?",
            hypothesis="It lowers ECE.",
            treatment="New calibration.",
            control="Temperature scaling.",
            primary_metric="ECE",
        )
    )
    store.upsert_idea(idea)
    spec = ResearchSpec(
        question=idea.question,
        hypothesis=idea.hypothesis,
        treatment=idea.treatment,
        control=idea.control,
        primary_metric="ECE",
        metric_direction=MetricDirection.MINIMIZE,
        guardrails=("accuracy must not decrease", "report NLL"),
        validity_conditions=("same held-out split",),
        compute_matching=("same logits and labels",),
        stopping_rules=("reject when ECE does not improve",),
        benchmark_id="cifar10_calibration",
        treatment_api=TREATMENT_API,
        primary_requires_effect_ci=True,
        guardrail_metrics=(
            MetricGuardrail(
                metric="accuracy",
                direction=MetricDirection.MAXIMIZE,
                relation=MetricRelation.EQUAL,
                per_pair=True,
            ),
            MetricGuardrail(
                metric="nll",
                direction=MetricDirection.MINIMIZE,
                require_effect_ci=True,
            ),
        ),
        minimum_pairs=2,
        calibration_split="clean",
        evaluation_split="corrupted",
        pairing_strategy="disjoint_example_blocks",
        require_per_example_argmax=True,
        required_compute_accounting=(
            "calibration_examples",
            "evaluation_examples",
            "model_forward_examples",
        ),
    )
    backend = RecordingBenchmarkBackend()
    bridge = BenchmarkPromotionBridge(
        config=BenchmarkPromotionConfig(
            enabled=True,
            benchmark_config=str(template),
            preflight_timeout_sec=5,
        ),
        store=store,
        treatment_worker=StaticTreatmentWorker(),
        run_backend=backend,
        max_gpus_per_run=1,
    )

    outcome = asyncio.run(bridge.promote(idea, spec=spec))

    assert outcome.execution_passed is True
    assert outcome.scientific_valid is True
    assert outcome.hypothesis_supported is False
    assert outcome.promotion_decision == "reject"
    assert backend.runs[0].requested_gpus == 1
    benchmark_root = store.idea_dir(idea.idea_id) / "benchmark"
    assert (benchmark_root / "treatment.py").is_file()
    assert (benchmark_root / "research_spec.json").is_file()
    assert (benchmark_root / "final_review.json").is_file()


def test_promotion_bridge_turns_treatment_generation_error_into_outcome(
    tmp_path,
) -> None:
    template = tmp_path / "benchmark.yaml"
    template.write_text(
        """
benchmark:
  cache_dir: cache
  output_dir: old-output
  treatment_path: old.py
  device: cpu
  require_cuda: false
""".lstrip()
    )
    store = ResearchQueueStore(tmp_path / "state", artifact_root=tmp_path / "artifacts")
    store.initialize()
    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Invalid treatment",
            question="Does it work?",
            hypothesis="It lowers ECE.",
            treatment="New calibration.",
            control="Temperature scaling.",
            primary_metric="ECE",
        )
    )
    store.upsert_idea(idea)
    spec = ResearchSpec(
        question=idea.question,
        hypothesis=idea.hypothesis,
        treatment=idea.treatment,
        control=idea.control,
        primary_metric="ECE",
        metric_direction=MetricDirection.MINIMIZE,
        guardrails=("accuracy must not decrease", "report NLL"),
        validity_conditions=("same held-out split",),
        compute_matching=("same logits and labels",),
        stopping_rules=("reject when ECE does not improve",),
        benchmark_id="cifar10_calibration",
        treatment_api=TREATMENT_API,
        primary_requires_effect_ci=True,
        guardrail_metrics=(
            MetricGuardrail(
                metric="accuracy",
                direction=MetricDirection.MAXIMIZE,
                relation=MetricRelation.EQUAL,
                per_pair=True,
            ),
            MetricGuardrail(
                metric="nll",
                direction=MetricDirection.MINIMIZE,
                require_effect_ci=True,
            ),
        ),
        minimum_pairs=2,
        calibration_split="clean",
        evaluation_split="corrupted",
        pairing_strategy="disjoint_example_blocks",
        require_per_example_argmax=True,
        required_compute_accounting=(
            "calibration_examples",
            "evaluation_examples",
            "model_forward_examples",
        ),
    )
    backend = RecordingBenchmarkBackend()
    bridge = BenchmarkPromotionBridge(
        config=BenchmarkPromotionConfig(
            enabled=True,
            benchmark_config=str(template),
            max_treatment_repairs=1,
        ),
        store=store,
        treatment_worker=FailingTreatmentWorker(),
        run_backend=backend,
        max_gpus_per_run=1,
    )

    outcome = asyncio.run(bridge.promote(idea, spec=spec))

    assert outcome.execution_passed is False
    assert outcome.scientific_valid is False
    assert outcome.promotion_decision == "reject"
    assert "preflight" in outcome.reason
    assert backend.runs == []
    preflight = json.loads(
        (store.idea_dir(idea.idea_id) / "benchmark" / "preflight-01.json").read_text()
    )
    assert "structured treatment invalid" in preflight["errors"][0]


def test_offline_review_rejects_successful_execution_with_insufficient_evidence() -> (
    None
):
    value = {
        **ResearchSpec(
            question="Does it improve calibration?",
            hypothesis="It lowers ECE without worse NLL.",
            treatment="Margin-adaptive scaling.",
            control="Scalar temperature scaling.",
            primary_metric="ece",
            metric_direction=MetricDirection.MINIMIZE,
            guardrails=("accuracy equal", "NLL no worse"),
            validity_conditions=("same held-out split",),
            compute_matching=("same logits",),
            stopping_rules=("inconclusive below five pairs",),
            benchmark_id="cifar10_calibration",
            treatment_api="fit/transform",
        ).to_dict(),
        "minimum_pairs": 5,
        "primary_requires_effect_ci": True,
        "guardrail_metrics": [
            {
                "metric": "accuracy",
                "direction": "maximize",
                "relation": "equal",
                "per_pair": True,
            },
            {
                "metric": "nll",
                "direction": "minimize",
                "relation": "no_worse",
                "require_effect_ci": True,
            },
        ],
    }
    spec = ResearchSpec.from_mapping(value)
    benchmark_result = {
        "status": "ok",
        "metrics": {
            "baseline_ece": 0.0624,
            "treatment_ece": 0.0571,
            "baseline_accuracy": 0.704,
            "treatment_accuracy": 0.704,
            "baseline_nll": 0.9349,
            "treatment_nll": 0.9365,
        },
        "per_seed": [
            {
                "baseline": {"ece": 0.0773, "accuracy": 0.694, "nll": 0.9803},
                "treatment": {"ece": 0.0649, "accuracy": 0.694, "nll": 0.9821},
            },
            {
                "baseline": {"ece": 0.0476, "accuracy": 0.714, "nll": 0.8896},
                "treatment": {"ece": 0.0494, "accuracy": 0.714, "nll": 0.8908},
            },
        ],
    }

    outcome = review_benchmark_result(
        spec=spec,
        benchmark_result=benchmark_result,
        execution_passed=True,
    )

    assert outcome.execution_passed is True
    assert outcome.scientific_valid is False
    assert outcome.hypothesis_supported is None
    assert outcome.promotion_decision == "reject"
    assert "insufficient independent pairs" in outcome.reason


def test_re_review_artifacts_rewrites_both_final_reviews(tmp_path) -> None:
    idea_dir = tmp_path / "ideas" / "idea-test"
    benchmark_dir = idea_dir / "benchmark"
    benchmark_dir.mkdir(parents=True)
    spec = ResearchSpec(
        question="Does it improve?",
        hypothesis="It improves ECE.",
        treatment="Treatment",
        control="Control",
        primary_metric="ece",
        metric_direction=MetricDirection.MINIMIZE,
        guardrails=("accuracy equal", "NLL no worse"),
        validity_conditions=("same split",),
        compute_matching=("same logits",),
        stopping_rules=("inconclusive below five pairs",),
        benchmark_id="cifar10_calibration",
        treatment_api="fit/transform",
        primary_requires_effect_ci=True,
        guardrail_metrics=(
            MetricGuardrail(
                metric="accuracy",
                direction=MetricDirection.MAXIMIZE,
                relation=MetricRelation.EQUAL,
                per_pair=True,
            ),
            MetricGuardrail(
                metric="nll",
                direction=MetricDirection.MINIMIZE,
                require_effect_ci=True,
            ),
        ),
        minimum_pairs=5,
    )
    benchmark_result = {
        "status": "ok",
        "metrics": {
            "baseline_ece": 0.08,
            "treatment_ece": 0.04,
            "baseline_accuracy": 0.7,
            "treatment_accuracy": 0.7,
            "baseline_nll": 1.0,
            "treatment_nll": 1.1,
        },
        "per_seed": [
            {
                "baseline": {"ece": 0.08, "accuracy": 0.7, "nll": 1.0},
                "treatment": {"ece": 0.04, "accuracy": 0.7, "nll": 1.1},
            },
            {
                "baseline": {"ece": 0.08, "accuracy": 0.7, "nll": 1.0},
                "treatment": {"ece": 0.04, "accuracy": 0.7, "nll": 1.1},
            },
        ],
    }
    (benchmark_dir / "research_spec.json").write_text(json.dumps(spec.to_dict()))
    (benchmark_dir / "result.json").write_text(json.dumps(benchmark_result))
    (idea_dir / "final_review.json").write_text(
        json.dumps(
            {
                "scientific_valid": True,
                "promotion_decision": "accept",
                "usage": {"total_tokens": 10},
                "provenance": {"idea_id": "idea-test"},
            }
        )
    )

    outcome = re_review_artifacts(idea_dir=idea_dir)

    assert outcome.scientific_valid is False
    assert outcome.usage["total_tokens"] == 10
    for path in (
        idea_dir / "final_review.json",
        benchmark_dir / "final_review.json",
    ):
        saved = json.loads(path.read_text())
        assert saved["promotion_decision"] == "reject"


def test_promotion_rejects_incompatible_spec_before_treatment_or_gpu(
    tmp_path,
) -> None:
    template = tmp_path / "benchmark.yaml"
    template.write_text(
        """
benchmark:
  cache_dir: cache
  output_dir: output
  treatment_path: treatment.py
  seeds: [17, 29]
  device: cuda
  require_cuda: true
""".lstrip()
    )
    store = ResearchQueueStore(tmp_path / "state")
    store.initialize()
    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Impossible pair count",
            question="Does it improve?",
            hypothesis="It lowers ECE.",
            treatment="Adaptive calibration.",
            control="Temperature scaling.",
            primary_metric="ECE",
        )
    )
    store.upsert_idea(idea)
    spec = ResearchSpec(
        question=idea.question,
        hypothesis=idea.hypothesis,
        treatment=idea.treatment,
        control=idea.control,
        primary_metric="ece",
        metric_direction=MetricDirection.MINIMIZE,
        guardrails=("accuracy unchanged", "NLL no worse"),
        validity_conditions=("frozen held-out split",),
        compute_matching=("same model logits",),
        stopping_rules=("reject invalid evidence",),
        benchmark_id="cifar10_calibration",
        treatment_api=TREATMENT_API,
        minimum_pairs=5,
        primary_requires_effect_ci=True,
        guardrail_metrics=(
            MetricGuardrail(
                metric="nll",
                direction=MetricDirection.MINIMIZE,
                require_effect_ci=True,
            ),
        ),
        calibration_split="clean",
        evaluation_split="corrupted",
        pairing_strategy="disjoint_example_blocks",
        require_per_example_argmax=True,
        required_compute_accounting=(
            "calibration_examples",
            "evaluation_examples",
            "model_forward_examples",
        ),
    )
    backend = RecordingBenchmarkBackend()
    bridge = BenchmarkPromotionBridge(
        config=BenchmarkPromotionConfig(
            enabled=True,
            benchmark_config=str(template),
        ),
        store=store,
        treatment_worker=NeverTreatmentWorker(),
        run_backend=backend,
        max_gpus_per_run=1,
    )

    outcome = asyncio.run(bridge.promote(idea, spec=spec))

    assert outcome.execution_passed is False
    assert "requires 5 independent pairs" in outcome.reason
    assert backend.runs == []
    benchmark_root = store.idea_dir(idea.idea_id) / "benchmark"
    assert not (benchmark_root / "treatment.py").exists()
    compatibility = json.loads(
        (benchmark_root / "benchmark_compatibility.json").read_text()
    )
    assert compatibility["checks"]["minimum_pairs_available"] is False
