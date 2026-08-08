from __future__ import annotations

import pytest

from researchclaw.research_queue.benchmark_profile import (
    TREATMENT_API,
    build_benchmark_plan,
    load_benchmark_profile,
    validate_benchmark_compatibility,
)
from researchclaw.research_queue.config import ResearchQueueConfig
from researchclaw.research_queue.models import (
    MetricDirection,
    MetricGuardrail,
    ResearchSpec,
)


def _config(tmp_path, *, seeds: str = "[17, 29]"):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        f"""
benchmark:
  cache_dir: cache
  output_dir: output
  treatment_path: treatment.py
  seeds: {seeds}
  corruption: gaussian_noise
""".lstrip()
    )
    return path


def _spec(
    *,
    minimum_pairs: int = 2,
    minimum_effect: float = 0.001,
) -> ResearchSpec:
    return ResearchSpec(
        question="Does treatment improve calibration?",
        hypothesis="ECE improves without worse NLL or changed predictions.",
        treatment="Adaptive calibration.",
        control="Scalar temperature scaling.",
        primary_metric="ece",
        metric_direction=MetricDirection.MINIMIZE,
        guardrails=("NLL no worse", "argmax unchanged"),
        validity_conditions=("frozen protocol",),
        compute_matching=("same data and model logits",),
        stopping_rules=("reject unsupported effects",),
        benchmark_id="cifar10_calibration",
        treatment_api=TREATMENT_API,
        minimum_effect=minimum_effect,
        minimum_pairs=minimum_pairs,
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


def test_profile_rejects_impossible_pair_count_before_execution(tmp_path) -> None:
    profile = load_benchmark_profile(
        "cifar10_calibration",
        _config(tmp_path),
    )

    result = validate_benchmark_compatibility(
        _spec(minimum_pairs=5),
        profile,
    )

    assert result.passed is False
    assert result.checks["minimum_pairs_available"] is False
    assert "requires 5" in result.errors[0]


def test_profile_freezes_protocol_and_capabilities(tmp_path) -> None:
    config = _config(tmp_path, seeds="[17, 29, 43, 59, 71]")
    profile = load_benchmark_profile("cifar10_calibration", config)
    plan = build_benchmark_plan(profile=profile, config_path=config)

    assert profile.available_pairs == 5
    assert profile.calibration_split == "clean"
    assert profile.evaluation_split == "corrupted"
    assert profile.pairing_strategy == "disjoint_example_blocks"
    assert profile.minimum_effect_for("ECE") == pytest.approx(0.001)
    assert validate_benchmark_compatibility(_spec(minimum_pairs=5), profile).passed
    assert plan["protocol"]["seeds"] == [17, 29, 43, 59, 71]
    assert len(plan["config_sha256"]) == 64
    assert plan["benchmark_profile"]["minimum_effects"] == {"ece": 0.001}


def test_profile_rejects_zero_practical_effect_floor(tmp_path) -> None:
    profile = load_benchmark_profile(
        "cifar10_calibration",
        _config(tmp_path),
    )

    result = validate_benchmark_compatibility(
        _spec(minimum_effect=0.0),
        profile,
    )

    assert result.passed is False
    assert result.checks["minimum_effect_meets_profile"] is False
    assert "frozen cifar10_calibration floor" in result.errors[-1]


def test_profile_does_not_claim_unobservable_objective_evaluations(tmp_path) -> None:
    profile = load_benchmark_profile(
        "cifar10_calibration",
        _config(tmp_path),
    )
    value = _spec().to_dict()
    value["required_compute_accounting"].append("objective_evaluations")

    result = validate_benchmark_compatibility(
        ResearchSpec.from_mapping(value),
        profile,
    )

    assert result.passed is False
    assert result.checks["compute_accounting_supported"] is False


def test_required_action_paths_fail_fast_when_unreachable() -> None:
    with pytest.raises(ValueError, match="unreachable"):
        ResearchQueueConfig.from_mapping(
            {
                "limits": {
                    "max_revisions_per_idea": 1,
                    "max_steps_per_idea": 5,
                    "required_paths": ["revise", "b2"],
                }
            }
        )

    config = ResearchQueueConfig.from_mapping(
        {
            "limits": {
                "max_revisions_per_idea": 2,
                "max_steps_per_idea": 10,
                "required_paths": ["revise", "b2"],
            }
        }
    )
    assert config.path_reachability()["revise"] is True
    assert config.path_reachability()["b2"] is True
