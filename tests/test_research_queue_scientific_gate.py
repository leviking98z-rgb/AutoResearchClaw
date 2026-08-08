from __future__ import annotations

from researchclaw.research_queue.models import (
    MetricDirection,
    MetricGuardrail,
    ResearchSpec,
)
from researchclaw.research_queue.promotion import validate_treatment_source
from researchclaw.research_queue.scientific_gate import (
    hypothesis_supported,
    validate_benchmark_result,
    validate_research_spec,
)
from researchclaw.research_queue.treatment_preflight import preflight_treatment
from researchclaw.research_queue.workers import _treatment_errors


def _spec() -> ResearchSpec:
    return ResearchSpec(
        question="Does calibration improve under shift?",
        hypothesis="The treatment lowers ECE without reducing accuracy.",
        treatment="Fit classwise affine logit calibration.",
        control="Scalar temperature scaling.",
        primary_metric="ECE",
        metric_direction=MetricDirection.MINIMIZE,
        guardrails=("accuracy must not decrease", "report NLL"),
        validity_conditions=("use held-out evaluation labels only for metrics",),
        compute_matching=("same logits, labels, split, and seeds",),
        stopping_rules=("reject when ECE does not improve",),
        benchmark_id="cifar10_calibration",
        treatment_api="fit(calibration_logits, labels); transform(logits, state)",
        guardrail_metrics=(
            MetricGuardrail(
                metric="accuracy",
                direction=MetricDirection.MAXIMIZE,
            ),
            MetricGuardrail(
                metric="nll",
                direction=MetricDirection.MINIMIZE,
            ),
        ),
        primary_requires_effect_ci=True,
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


def test_research_spec_round_trip_and_gate() -> None:
    spec = _spec()
    restored = ResearchSpec.from_mapping(spec.to_dict())

    assert restored == spec
    gate = validate_research_spec(
        restored,
        benchmark_id="cifar10_calibration",
    )
    assert gate.passed


def test_scientific_result_separates_execution_validity_and_support() -> None:
    spec = _spec()
    result = {
        "status": "ok",
        "metrics": {
            "baseline_ece": 0.08,
            "treatment_ece": 0.04,
            "baseline_accuracy": 0.7,
            "treatment_accuracy": 0.7,
            "baseline_nll": 1.0,
            "treatment_nll": 0.9,
        },
        "per_seed": [
            {
                "baseline": {"ece": 0.08, "accuracy": 0.7, "nll": 1.0},
                "treatment": {"ece": 0.04, "accuracy": 0.7, "nll": 0.9},
                "evidence": {
                    "argmax_preserved": True,
                    "argmax_changed_count": 0,
                    "baseline_predictions_sha256": "a",
                    "treatment_predictions_sha256": "a",
                },
            },
            {
                "baseline": {"ece": 0.08, "accuracy": 0.7, "nll": 1.0},
                "treatment": {"ece": 0.04, "accuracy": 0.7, "nll": 0.9},
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

    gate = validate_benchmark_result(spec, result)

    assert gate.passed
    assert hypothesis_supported(spec, result) is True


def test_hypothesis_support_rejects_floating_point_noop() -> None:
    spec = _spec()
    spec.minimum_effect = 0.0
    result = {
        "metrics": {
            "baseline_ece": 0.48481052937129565,
            "treatment_ece": 0.48481052937129554,
        },
        "uncertainty": {
            "effect_ece_ci": [
                1.1102230246251566e-17,
                7.771561172376095e-17,
            ],
        },
    }

    assert hypothesis_supported(spec, result) is False


def test_scientific_result_rejects_accuracy_degradation() -> None:
    spec = _spec()
    result = {
        "status": "ok",
        "metrics": {
            "baseline_ece": 0.08,
            "treatment_ece": 0.04,
            "baseline_accuracy": 0.7,
            "treatment_accuracy": 0.6,
            "baseline_nll": 1.0,
            "treatment_nll": 0.9,
        },
        "per_seed": [
            {
                "baseline": {"ece": 0.08, "accuracy": 0.7, "nll": 1.0},
                "treatment": {"ece": 0.04, "accuracy": 0.6, "nll": 0.9},
            },
            {
                "baseline": {"ece": 0.08, "accuracy": 0.7, "nll": 1.0},
                "treatment": {"ece": 0.04, "accuracy": 0.6, "nll": 0.9},
            },
        ],
    }

    gate = validate_benchmark_result(spec, result)

    assert not gate.passed
    assert any("accuracy" in item for item in gate.errors)


def test_scientific_result_enforces_pairs_ci_and_nll_guardrail() -> None:
    base = _spec()
    value = base.to_dict()
    value.update(
        {
            "minimum_pairs": 5,
            "primary_requires_effect_ci": True,
            "guardrail_metrics": [
                {
                    "metric": "accuracy",
                    "direction": "maximize",
                    "relation": "equal",
                    "tolerance": 0.0,
                    "per_pair": True,
                },
                {
                    "metric": "nll",
                    "direction": "minimize",
                    "relation": "no_worse",
                    "tolerance": 0.0,
                    "require_effect_ci": True,
                },
            ],
        }
    )
    spec = ResearchSpec.from_mapping(value)
    result = {
        "status": "ok",
        "metrics": {
            "baseline_ece": 0.0624754093,
            "treatment_ece": 0.0571526214,
            "baseline_accuracy": 0.704,
            "treatment_accuracy": 0.704,
            "baseline_nll": 0.9349099013,
            "treatment_nll": 0.9364806385,
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

    gate = validate_benchmark_result(spec, result)

    assert not gate.passed
    assert gate.checks["minimum_pairs_met"] is False
    assert gate.checks["primary_effect_ci_supports_hypothesis"] is False
    assert gate.checks["nll_guardrail_passed"] is False
    assert any("insufficient independent pairs" in item for item in gate.errors)
    assert any("nll" in item for item in gate.errors)


def test_scientific_result_accepts_complete_machine_contract() -> None:
    value = _spec().to_dict()
    value.update(
        {
            "minimum_pairs": 5,
            "primary_requires_effect_ci": True,
            "guardrail_metrics": [
                {
                    "metric": "accuracy",
                    "direction": "maximize",
                    "relation": "equal",
                    "tolerance": 0.0,
                    "per_pair": True,
                },
                {
                    "metric": "nll",
                    "direction": "minimize",
                    "relation": "no_worse",
                    "tolerance": 0.0,
                    "require_effect_ci": True,
                },
            ],
        }
    )
    spec = ResearchSpec.from_mapping(value)
    rows = [
        {
            "baseline": {"ece": 0.08, "accuracy": 0.7, "nll": 1.0},
            "treatment": {"ece": 0.04, "accuracy": 0.7, "nll": 0.9},
            "evidence": {
                "argmax_preserved": True,
                "argmax_changed_count": 0,
                "baseline_predictions_sha256": f"baseline-{index}",
                "treatment_predictions_sha256": f"treatment-{index}",
            },
        }
        for index in range(5)
    ]
    result = {
        "status": "ok",
        "metrics": {
            "baseline_ece": 0.08,
            "treatment_ece": 0.04,
            "baseline_accuracy": 0.7,
            "treatment_accuracy": 0.7,
            "baseline_nll": 1.0,
            "treatment_nll": 0.9,
        },
        "uncertainty": {
            "effect_ece_ci": [0.04, 0.04],
            "effect_nll_ci": [0.1, 0.1],
        },
        "per_seed": rows,
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

    gate = validate_benchmark_result(spec, result)

    assert gate.passed
    assert gate.checks["primary_effect_ci_supports_hypothesis"] is True
    assert gate.checks["nll_effect_ci_passed"] is True


def test_scientific_result_rejects_equal_accuracy_with_changed_argmax() -> None:
    spec = _spec()
    rows = [
        {
            "baseline": {"ece": 0.08, "accuracy": 0.5, "nll": 1.0},
            "treatment": {"ece": 0.04, "accuracy": 0.5, "nll": 0.9},
            "evidence": {
                "argmax_preserved": index == 0,
                "argmax_changed_count": 0 if index == 0 else 2,
                "baseline_predictions_sha256": f"base-{index}",
                "treatment_predictions_sha256": f"treatment-{index}",
            },
        }
        for index in range(2)
    ]
    result = {
        "status": "ok",
        "metrics": {
            "baseline_ece": 0.08,
            "treatment_ece": 0.04,
            "baseline_accuracy": 0.5,
            "treatment_accuracy": 0.5,
            "baseline_nll": 1.0,
            "treatment_nll": 0.9,
        },
        "uncertainty": {"effect_ece_ci": [0.04, 0.04]},
        "per_seed": rows,
        "evidence": {
            "protocol": {
                "calibration_split": "clean",
                "evaluation_split": "corrupted",
                "pairing_strategy": "disjoint_example_blocks",
            },
            "argmax": {
                "argmax_preserved": False,
                "argmax_changed_count": 2,
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

    gate = validate_benchmark_result(spec, result)

    assert gate.passed is False
    assert gate.checks["per_example_argmax_attested"] is False
    assert any("argmax" in error for error in gate.errors)


def test_generated_treatment_preflight_and_static_gate(tmp_path) -> None:
    path = tmp_path / "treatment.py"
    source = """
import numpy as np


class TemperatureTreatment:
    def fit(self, calibration_logits, calibration_labels):
        del calibration_labels
        scale = float(np.std(calibration_logits))
        return {"scale": max(scale, 1e-6)}

    def transform(self, logits, state):
        return logits / state["scale"]


def build_treatment():
    return TemperatureTreatment()
""".lstrip()
    path.write_text(source, encoding="utf-8")

    assert validate_treatment_source(source) == []
    result = preflight_treatment(path, examples=16, classes=3, timeout_sec=5)
    assert result["passed"] is True
    assert result["output_shape"] == [16, 3]


def test_generated_treatment_static_gate_rejects_file_access() -> None:
    source = """
def build_treatment():
    open("/tmp/leak", "w").write("bad")
""".lstrip()

    errors = validate_treatment_source(source)

    assert any("forbidden" in item for item in errors)


def test_treatment_worker_gate_does_not_reject_identifiers_containing_os() -> None:
    source = """
import numpy as np


class PostHocTreatment:
    def fit(self, calibration_logits, calibration_labels):
        positives = np.maximum(np.asarray(calibration_logits), 0.0)
        return {"scale": float(positives.mean() + 1.0)}

    def transform(self, logits, state):
        return np.asarray(logits) / state["scale"]


def build_treatment():
    return PostHocTreatment()
""".lstrip()

    errors = _treatment_errors({"treatment_source": source})

    assert errors == []
