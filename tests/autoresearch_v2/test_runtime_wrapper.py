from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.autoresearch_v2.runtime_wrapper import (
    RAW_DIRNAME,
    WRAPPER_SCHEMA,
    WRAPPER_VERSION,
    RuntimeArtifactError,
    normalize_runtime_artifacts,
)


def _plan() -> dict[str, object]:
    return {
        "datasets": [
            {
                "name": "GSM8K-development",
                "resource_id": "openai/gsm8k",
                "split_role": "development",
                "split_id": "gsm8k-dev-v1",
            },
            {
                "name": "GSM8K-screening",
                "resource_id": "openai/gsm8k",
                "split_role": "screening",
                "split_id": "gsm8k-screen-v1",
            },
        ],
        "gate_statistic": {"name": "paired_accuracy_difference"},
        "uncertainty": {
            "method": "paired_bootstrap",
            "confidence_level": 0.9,
            "resamples": 1000,
            "rng_seed": 1729,
            "undefined_resample_policy": "drop",
            "decision_role": "descriptive",
        },
        "validity_criteria": [
            {
                "id": "completed_examples",
                "metric": "completed_examples",
                "operator": ">=",
                "value": 4,
            }
        ],
        "promotion_criteria": [
            {
                "id": "primary_effect",
                "metric": "paired_accuracy_difference",
                "operator": ">=",
                "value": 0.15,
            }
        ],
        "call_ledger": {
            "components": [
                {"name": "adaptation", "total_calls": 8},
                {"name": "final_evaluation", "total_calls": 16},
            ],
            "total_model_calls": 24,
        },
    }


def _write_raw(
    output_dir: Path,
    *,
    metrics: dict[str, object] | None = None,
    runtime: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            metrics
            or {
                "completed_examples": 4,
                "paired_accuracy_difference": 0.20,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "runtime_evidence.json").write_text(
        json.dumps(
            runtime
            or {
                "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
                "datasets_loaded": ["GSM8K"],
                "examples_processed": 4,
                "seed": 7,
                "examples_by_role": {
                    "development": 2,
                    "screening": 4,
                },
                "call_counts": {
                    "adaptation": 8,
                    "final_evaluation": 16,
                },
            }
        ),
        encoding="utf-8",
    )


def test_runtime_wrapper_compiles_and_is_idempotent(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    _write_raw(
        output_dir,
        runtime={
            "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["GSM8K"],
            "examples_processed": 4,
            "seed": 7,
            "examples_by_role": {
                "development": 2,
                "screening": 4,
            },
            "call_counts": {
                "adaptation": 8,
                "final_evaluation": 16,
            },
            "uncertainty": {
                "available": True,
                "method": "paired_bootstrap",
                "confidence_level": 0.9,
                "resamples": 1000,
                "rng_seed": 1729,
                "metric": "paired_accuracy_difference",
                "observations": [1, 0, 0, -0.2],
                "lower": 999.0,
                "upper": 1000.0,
            },
        },
    )

    first = normalize_runtime_artifacts(
        output_dir=output_dir,
        plan=_plan(),
        mode="pilot",
        allocated_gpus=1,
        cwd=tmp_path,
    )
    raw_before = {
        name: (output_dir / RAW_DIRNAME / name).read_text(encoding="utf-8")
        for name in ("metrics.json", "runtime_evidence.json")
    }
    second = normalize_runtime_artifacts(
        output_dir=output_dir,
        plan=_plan(),
        mode="pilot",
        allocated_gpus=1,
        cwd=tmp_path,
    )

    runtime = first["runtime_evidence"]
    assert runtime["wrapper_schema"] == WRAPPER_SCHEMA
    assert runtime["wrapper_version"] == WRAPPER_VERSION
    assert runtime["seeds"] == [7]
    assert runtime["gpu_count"] == 1
    assert runtime["dataset_roles"]["GSM8K-development"]["resource_id"] == (
        "openai/gsm8k"
    )
    assert runtime["criterion_results"] == {
        "completed_examples": {"value": 4, "passed": True},
        "primary_effect": {"value": 0.20, "passed": True},
    }
    assert runtime["uncertainty"]["aggregation"] == "mean"
    assert runtime["uncertainty"]["point_estimate"] == pytest.approx(0.2)
    assert runtime["uncertainty"]["interval_bounds_source"] == (
        "controller_recomputed"
    )
    assert runtime["uncertainty"]["lower"] != 999.0
    assert runtime["uncertainty"]["upper"] != 1000.0
    assert runtime["gate_decision"] == "promote"
    assert second["already_compiled"] is True
    assert second["runtime_evidence"] == runtime
    assert raw_before == {
        name: (output_dir / RAW_DIRNAME / name).read_text(encoding="utf-8")
        for name in ("metrics.json", "runtime_evidence.json")
    }


def test_runtime_wrapper_migrates_boolean_loader_with_model_id(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "smoke"
    _write_raw(
        output_dir,
        metrics={
            "completed_examples": 1,
            "paired_accuracy_difference": 0.0,
        },
        runtime={
            "model_loaded": True,
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["openai/gsm8k"],
            "examples_processed": 1,
            "examples_by_role": {"screening": 1},
            "seeds": [7],
            "call_counts": {
                "adaptation": 1,
                "final_evaluation": 1,
            },
        },
    )

    value = normalize_runtime_artifacts(
        output_dir=output_dir,
        plan=_plan(),
        mode="smoke",
        allocated_gpus=1,
        cwd=tmp_path,
    )

    runtime = value["runtime_evidence"]
    assert runtime["model_loaded"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert runtime["model_metadata"] == {"loaded": True}
    assert runtime["gate_decision"] == "promote"


def test_runtime_wrapper_rejects_boolean_loader_without_model_id(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "smoke"
    _write_raw(
        output_dir,
        runtime={
            "model_loaded": True,
            "datasets_loaded": ["openai/gsm8k"],
            "examples_processed": 1,
            "examples_by_role": {"screening": 1},
            "seeds": [7],
        },
    )

    with pytest.raises(
        RuntimeArtifactError,
        match="model_loaded must be a non-empty model id",
    ):
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=_plan(),
            mode="smoke",
            allocated_gpus=1,
            cwd=tmp_path,
        )


def test_runtime_wrapper_rejects_missing_measured_criterion(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    _write_raw(
        output_dir,
        metrics={"paired_accuracy_difference": 0.20},
    )

    with pytest.raises(
        RuntimeArtifactError,
        match="missing measured criterion metric 'completed_examples'",
    ):
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=_plan(),
            mode="pilot",
            allocated_gpus=1,
            cwd=tmp_path,
        )


def test_runtime_wrapper_rejects_missing_call_ledger_component(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    _write_raw(
        output_dir,
        runtime={
            "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["GSM8K"],
            "examples_processed": 4,
            "seeds": [7],
            "call_counts": {"final_evaluation": 16},
        },
    )

    with pytest.raises(
        RuntimeArtifactError,
        match="call_counts missing compiled components: adaptation",
    ):
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=_plan(),
            mode="pilot",
            allocated_gpus=1,
            cwd=tmp_path,
        )


def test_runtime_wrapper_normalizes_legacy_all_role_example_total(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    _write_raw(
        output_dir,
        runtime={
            "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["GSM8K"],
            "examples_processed": 6,
            "seeds": [7],
            "examples_by_role": {
                "development": 2,
                "screening": 4,
            },
            "call_counts": {
                "adaptation": 8,
                "final_evaluation": 16,
            },
        },
    )

    value = normalize_runtime_artifacts(
        output_dir=output_dir,
        plan=_plan(),
        mode="pilot",
        allocated_gpus=1,
        cwd=tmp_path,
    )

    runtime = value["runtime_evidence"]
    assert runtime["examples_processed"] == 4
    assert runtime["examples_by_role"] == {
        "development": 2,
        "screening": 4,
    }
    assert runtime["example_diagnostics"] == {
        "reported_examples_processed": 6,
        "canonical_examples_processed": 4,
        "normalization": "legacy_all_roles_total_to_endpoint_count",
    }
    assert runtime["uncertainty"] == {
        "available": False,
        "method": "paired_bootstrap",
        "confidence_level": 0.9,
        "resamples": 1000,
        "rng_seed": 1729,
        "undefined_resample_policy": "drop",
        "decision_role": "descriptive",
        "metric": "paired_accuracy_difference",
        "point_estimate": 0.2,
        "reason": "item_level_paired_observations_not_emitted",
    }


def test_runtime_wrapper_requires_auditable_uncertainty_for_promotion(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    plan = _plan()
    plan["promotion_criteria"].append(
        {
            "id": "uncertainty_support",
            "metric": "primary_effect_ci_lower",
            "operator": ">",
            "value": 0.0,
        }
    )
    _write_raw(
        output_dir,
        metrics={
            "completed_examples": 4,
            "paired_accuracy_difference": 0.20,
            "primary_effect_ci_lower": 0.05,
        },
    )

    with pytest.raises(
        RuntimeArtifactError,
        match="auditable item-level or cluster-level observations",
    ):
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=plan,
            mode="pilot",
            allocated_gpus=1,
            cwd=tmp_path,
        )


def test_runtime_wrapper_recomputes_count_bootstrap_for_ci_gate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    plan = _plan()
    plan["gate_statistic"] = {"name": "paired_correct_count_difference"}
    plan["promotion_criteria"] = [
        {
            "id": "primary_effect",
            "metric": "paired_correct_count_difference",
            "operator": ">=",
            "value": 2,
        },
        {
            "id": "uncertainty_support",
            "metric": "primary_effect_ci_lower",
            "operator": ">",
            "value": 0.0,
        },
    ]
    _write_raw(
        output_dir,
        metrics={
            "completed_examples": 4,
            "paired_correct_count_difference": 2,
            "primary_effect_ci_lower": 999,
        },
        runtime={
            "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["GSM8K"],
            "examples_processed": 4,
            "seed": 7,
            "examples_by_role": {
                "development": 2,
                "screening": 4,
            },
            "call_counts": {
                "adaptation": 8,
                "final_evaluation": 16,
            },
            "uncertainty": {
                "available": True,
                "method": "paired_bootstrap",
                "confidence_level": 0.9,
                "resamples": 1000,
                "rng_seed": 1729,
                "metric": "paired_correct_count_difference",
                "observations": [1, 1, 0, 0],
                "lower": 999,
                "upper": 1000,
            },
        },
    )

    value = normalize_runtime_artifacts(
        output_dir=output_dir,
        plan=plan,
        mode="pilot",
        allocated_gpus=1,
        cwd=tmp_path,
    )

    runtime = value["runtime_evidence"]
    assert runtime["uncertainty"]["aggregation"] == "sum"
    assert runtime["uncertainty"]["point_estimate"] == 2
    assert runtime["uncertainty"]["lower"] == pytest.approx(0.0)
    assert runtime["uncertainty"]["upper"] == pytest.approx(4.0)
    assert runtime["metrics"]["primary_effect_ci_lower"] == pytest.approx(0.0)
    assert runtime["criterion_results"]["uncertainty_support"] == {
        "value": pytest.approx(0.0),
        "passed": False,
    }
    assert runtime["gate_decision"] == "reject"


def test_runtime_wrapper_recomputes_proportion_bootstrap_for_ci_gate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    plan = _plan()
    plan["promotion_criteria"].append(
        {
            "id": "uncertainty_support",
            "metric": "primary_effect_ci_lower",
            "operator": ">",
            "value": 0.0,
        }
    )
    _write_raw(
        output_dir,
        metrics={
            "completed_examples": 4,
            "paired_accuracy_difference": 0.5,
            "primary_effect_ci_lower": 999,
        },
        runtime={
            "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["GSM8K"],
            "examples_processed": 4,
            "seed": 7,
            "examples_by_role": {
                "development": 2,
                "screening": 4,
            },
            "call_counts": {
                "adaptation": 8,
                "final_evaluation": 16,
            },
            "uncertainty": {
                "available": True,
                "method": "paired_bootstrap",
                "confidence_level": 0.9,
                "resamples": 1000,
                "rng_seed": 1729,
                "metric": "paired_accuracy_difference",
                "observations": [1, 1, 0, 0],
                "lower": 999,
                "upper": 1000,
            },
        },
    )

    value = normalize_runtime_artifacts(
        output_dir=output_dir,
        plan=plan,
        mode="pilot",
        allocated_gpus=1,
        cwd=tmp_path,
    )

    runtime = value["runtime_evidence"]
    assert runtime["uncertainty"]["aggregation"] == "mean"
    assert runtime["uncertainty"]["point_estimate"] == pytest.approx(0.5)
    assert runtime["uncertainty"]["lower"] == pytest.approx(0.0)
    assert runtime["uncertainty"]["upper"] == pytest.approx(1.0)
    assert runtime["metrics"]["primary_effect_ci_lower"] == pytest.approx(0.0)
    assert runtime["criterion_results"]["uncertainty_support"] == {
        "value": pytest.approx(0.0),
        "passed": False,
    }
    assert runtime["gate_decision"] == "reject"


def test_runtime_wrapper_rejects_observations_that_do_not_match_gate(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    plan = _plan()
    plan["promotion_criteria"].append(
        {
            "id": "uncertainty_support",
            "metric": "primary_effect_ci_lower",
            "operator": ">",
            "value": 0.0,
        }
    )
    _write_raw(
        output_dir,
        metrics={
            "completed_examples": 4,
            "paired_accuracy_difference": 0.2,
            "primary_effect_ci_lower": 0.1,
        },
        runtime={
            "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["GSM8K"],
            "examples_processed": 4,
            "seed": 7,
            "examples_by_role": {
                "development": 2,
                "screening": 4,
            },
            "call_counts": {
                "adaptation": 8,
                "final_evaluation": 16,
            },
            "uncertainty": {
                "available": True,
                "method": "paired_bootstrap",
                "confidence_level": 0.9,
                "resamples": 1000,
                "rng_seed": 1729,
                "metric": "paired_accuracy_difference",
                "observations": [1, 1, 0, 0],
            },
        },
    )

    with pytest.raises(
        RuntimeArtifactError,
        match="reproduce neither the sum nor the mean",
    ):
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=plan,
            mode="pilot",
            allocated_gpus=1,
            cwd=tmp_path,
        )


def test_runtime_wrapper_normalizes_structured_resource_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    _write_raw(
        output_dir,
        runtime={
            "model_loaded": {
                "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "device": "cuda",
                "num_parameters": 1_543_714_304,
            },
            "datasets_loaded": [
                {
                    "resource_id": "openai/gsm8k",
                    "split": "train",
                },
                {
                    "resource_id": "openai/gsm8k",
                    "split": "test",
                },
            ],
            "examples_processed": 4,
            "seeds": [7],
            "examples_by_role": {
                "development": 2,
                "screening": 4,
            },
            "call_counts": {
                "adaptation": 8,
                "final_evaluation": 16,
            },
        },
    )

    value = normalize_runtime_artifacts(
        output_dir=output_dir,
        plan=_plan(),
        mode="pilot",
        allocated_gpus=1,
        cwd=tmp_path,
    )

    runtime = value["runtime_evidence"]
    assert runtime["model_loaded"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert runtime["model_metadata"] == {
        "device": "cuda",
        "num_parameters": 1_543_714_304,
    }
    assert runtime["datasets_loaded"] == ["openai/gsm8k"]


def test_runtime_wrapper_rejects_inconsistent_example_accounting(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    _write_raw(
        output_dir,
        runtime={
            "model_loaded": "Qwen/Qwen2.5-1.5B-Instruct",
            "datasets_loaded": ["GSM8K"],
            "examples_processed": 5,
            "seeds": [7],
            "examples_by_role": {
                "development": 2,
                "screening": 4,
            },
            "call_counts": {
                "adaptation": 8,
                "final_evaluation": 16,
            },
        },
    )

    with pytest.raises(
        RuntimeArtifactError,
        match="examples_processed must equal",
    ):
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=_plan(),
            mode="pilot",
            allocated_gpus=1,
            cwd=tmp_path,
        )


def test_runtime_wrapper_rejects_partial_canonical_markers(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "artifacts" / "pilot"
    _write_raw(
        output_dir,
        metrics={
            "result_valid": True,
            "metrics": {"paired_accuracy_difference": 0.20},
            "wrapper_schema": WRAPPER_SCHEMA,
            "wrapper_version": WRAPPER_VERSION,
        },
    )

    with pytest.raises(
        RuntimeArtifactError,
        match="inconsistent wrapper markers",
    ):
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=_plan(),
            mode="pilot",
            allocated_gpus=1,
            cwd=tmp_path,
        )
