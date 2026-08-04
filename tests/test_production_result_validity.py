"""Regression tests for fail-closed production result validation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from researchclaw.pipeline.experiment_diagnosis import diagnose_experiment
from researchclaw.pipeline.experiment_repair import (
    _build_experiment_summary_from_run,
    _summary_quality_score,
    run_repair_loop,
)
from researchclaw.pipeline.result_validity import assess_production_result
from researchclaw.pipeline.stages import StageStatus


def test_dataset_missing_placeholder_metrics_are_invalid() -> None:
    validity = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={
            "ece": 1.0,
            "accuracy": 0.0,
            "collapse_ratio": 1.0,
            "success_rate": 0.0,
        },
        stdout=(
            "DatasetNotFoundError: Dataset 'missing_seed_dataset' "
            "doesn't exist on the Hub"
        ),
    )

    assert not validity.valid
    assert not validity.successful
    assert any("dataset" in reason for reason in validity.reasons)
    assert "success_rate is 0" in validity.reasons


def test_traceback_with_finite_metrics_is_invalid() -> None:
    validity = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={"accuracy": 0.75},
        stdout=(
            "Traceback (most recent call last):\n"
            "  File 'main.py', line 10, in main\n"
            "RuntimeError: failed after writing placeholders"
        ),
    )

    assert not validity.successful
    assert any("traceback" in reason for reason in validity.reasons)


def test_synthetic_fallback_with_finite_metrics_is_invalid() -> None:
    validity = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={"baseline/0/accuracy": 0.99},
        stdout=(
            "Dataset load failed; falling back to synthetic data instead.\n"
            "condition=baseline seed=0 accuracy: 0.99"
        ),
    )

    assert not validity.valid
    assert not validity.successful
    assert any("synthetic" in reason for reason in validity.reasons)


def test_zero_success_rate_or_steps_completed_are_invalid() -> None:
    zero_success = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={"accuracy": 0.4, "success_rate": 0.0},
    )
    zero_steps = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={"accuracy": 0.4, "steps_completed": 0},
    )

    assert not zero_success.valid
    assert "success_rate is 0" in zero_success.reasons
    assert not zero_steps.valid
    assert "steps_completed is 0" in zero_steps.reasons


def test_no_finite_metrics_or_successful_seed_is_invalid() -> None:
    validity = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={"accuracy": float("nan")},
        stdout=(
            "condition=baseline seed=0 failed: dataset missing\n"
            "condition=proposed seed=0 failed: dataset missing"
        ),
    )

    assert not validity.valid
    assert validity.successful_seed_count == 0
    assert any("no finite" in reason for reason in validity.reasons)


def test_timeout_with_real_metrics_is_partial_not_success() -> None:
    validity = assess_production_result(
        returncode=-1,
        timed_out=True,
        metrics={"best_loss": 0.5},
        stdout="best_loss: 0.5",
    )

    assert validity.valid
    assert not validity.successful
    assert any("timed out" in reason for reason in validity.reasons)


def test_real_negative_result_remains_valid() -> None:
    validity = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={
            "baseline/0/accuracy": 0.0,
            "baseline/accuracy": 0.0,
            "baseline/effect_size": 0.0,
            "baseline/success_rate": 1.0,
            "baseline/steps_completed": 12.0,
        },
        stdout=(
            "condition=baseline seed=0 accuracy: 0.0\n"
            "condition=baseline success_rate: 1/1\n"
            "condition=baseline steps_completed: 12"
        ),
    )

    assert validity.valid
    assert validity.successful
    assert validity.valid_conditions == {"baseline"}
    assert validity.successful_seed_count == 1


def test_stage12_rejects_rc_zero_placeholder_result(tmp_path) -> None:
    from researchclaw.pipeline.stage_impls._execution import (
        _execute_experiment_run,
    )

    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-12"
    stage_dir.mkdir(parents=True)

    config = MagicMock()
    config.experiment.mode = "sandbox"
    config.experiment.time_budget_sec = 300
    config.experiment.metric_key = "accuracy"
    config.experiment.sandbox.python_path = "python"

    sandbox_result = MagicMock()
    sandbox_result.returncode = 0
    sandbox_result.timed_out = False
    sandbox_result.metrics = {
        "ece": 1.0,
        "accuracy": 0.0,
        "collapse_ratio": 1.0,
        "success_rate": 0.0,
    }
    sandbox_result.stdout = "DatasetNotFoundError: all seed datasets missing"
    sandbox_result.stderr = ""
    sandbox_result.elapsed_sec = 2.0

    sandbox = MagicMock()
    sandbox.run.return_value = sandbox_result
    sandbox.run_project.return_value = sandbox_result

    with (
        patch(
            "researchclaw.experiment.factory.create_sandbox",
            return_value=sandbox,
        ),
        patch(
            "researchclaw.pipeline.stage_impls._execution."
            "_ensure_sandbox_deps"
        ),
        patch(
            "researchclaw.pipeline.stage_impls._execution."
            "_read_prior_artifact",
            return_value="",
        ),
    ):
        result = _execute_experiment_run(
            stage_dir,
            run_dir,
            config,
            MagicMock(),
        )

    payload = json.loads(
        (stage_dir / "runs" / "run-1.json").read_text(encoding="utf-8")
    )
    assert result.status is StageStatus.FAILED
    assert payload["status"] == "failed"
    assert payload["result_valid"] is False


def test_repair_summary_rc_zero_invalid_cannot_be_ranked() -> None:
    run_result = {
        "stdout": "DatasetNotFoundError: dataset missing",
        "stderr": "",
        "returncode": 0,
        "metrics": {
            "A/accuracy": 0.0,
            "A/success_rate": 0.0,
        },
        "elapsed_sec": 1.0,
        "timed_out": False,
    }

    summary = _build_experiment_summary_from_run(run_result, {})

    assert summary["result_valid"] is False
    assert summary["best_run"]["status"] == "failed"
    assert _summary_quality_score(summary) < 0


def test_repair_loop_rc_zero_invalid_result_is_not_success(tmp_path) -> None:
    stage14 = tmp_path / "stage-14"
    stage14.mkdir()
    initial_summary = {
        "condition_summaries": {
            "baseline": {"metrics": {"accuracy": 0.5}},
        },
        "best_run": {
            "status": "completed",
            "metrics": {"baseline/0/accuracy": 0.5},
        },
    }
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(initial_summary),
        encoding="utf-8",
    )
    experiment_dir = tmp_path / "stage-10" / "experiment"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "main.py").write_text("print('old')", encoding="utf-8")

    repair_cfg = MagicMock()
    repair_cfg.min_conditions = 3
    repair_cfg.max_cycles = 1
    repair_cfg.timeout_sec_per_cycle = 30
    repair_cfg.use_opencode = False

    config = MagicMock()
    config.experiment.repair = repair_cfg
    config.experiment.mode = "sandbox"
    config.experiment.time_budget_sec = 30
    config.experiment.opencode.enabled = False
    config.experiment.metric_key = "accuracy"

    invalid_run = {
        "stdout": "DatasetNotFoundError: dataset missing",
        "stderr": "",
        "returncode": 0,
        "metrics": {
            "A/accuracy": 0.0,
            "A/success_rate": 0.0,
        },
        "elapsed_sec": 1.0,
        "timed_out": False,
    }

    with (
        patch(
            "researchclaw.llm.create_llm_client",
            return_value=MagicMock(),
        ),
        patch(
            "researchclaw.pipeline.experiment_repair._get_repaired_code",
            return_value={"main.py": "print('fixed')"},
        ),
        patch(
            "researchclaw.pipeline.experiment_repair."
            "_run_experiment_in_sandbox",
            return_value=invalid_run,
        ),
    ):
        result = run_repair_loop(tmp_path, config, "validity-test")

    assert result.success is False
    assert result.total_cycles == 1
    assert result.cycle_history[0].error.startswith(
        "Repair rerun invalid:"
    )
    assert result.best_experiment_summary is initial_summary or (
        result.best_experiment_summary
        and result.best_experiment_summary.get("result_valid") is not False
    )


def test_diagnosis_placeholder_conditions_do_not_complete() -> None:
    summary = {
        "condition_summaries": {
            f"C{i}": {
                "metrics": {
                    "ece": 1.0,
                    "accuracy": 0.0,
                    "collapse_ratio": 1.0,
                },
                "success_rate": 0.0,
            }
            for i in range(5)
        },
        "best_run": {
            "status": "completed",
            "metrics": {
                **{f"C{i}/accuracy": 0.0 for i in range(5)},
                **{f"C{i}/success_rate": 0.0 for i in range(5)},
            },
            "stdout": "DatasetNotFoundError: all seed datasets missing",
        },
    }

    diagnosis = diagnose_experiment(
        experiment_summary=summary,
        experiment_plan={"conditions": []},
    )

    assert diagnosis.conditions_completed == []
    assert diagnosis.conditions_failed == [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    assert diagnosis.total_planned == 5
    assert diagnosis.completion_rate == 0.0


def test_diagnosis_rate_never_exceeds_one_when_observed_exceeds_plan() -> None:
    summary = {
        "condition_summaries": {
            "A": {"metrics": {"accuracy": 0.1}},
            "B": {"metrics": {"accuracy": 0.2}},
            "C": {"metrics": {"accuracy": 0.3}},
        },
        "best_run": {
            "status": "completed",
            "metrics": {
                "A/0/accuracy": 0.1,
                "B/0/accuracy": 0.2,
                "C/0/accuracy": 0.3,
            },
        },
    }

    diagnosis = diagnose_experiment(
        experiment_summary=summary,
        experiment_plan={"conditions": [{"name": "A"}]},
    )

    assert diagnosis.conditions_completed == ["A", "B", "C"]
    assert diagnosis.total_planned == 3
    assert diagnosis.completion_rate == 1.0


def test_partial_conditions_count_only_real_successes() -> None:
    summary = {
        "condition_summaries": {
            "good": {
                "metrics": {"accuracy": 0.0, "effect_size": 0.0},
                "success_rate": 1.0,
                "steps_completed": 5,
            },
            "missing": {
                "metrics": {
                    "accuracy": 0.0,
                    "ece": 1.0,
                    "collapse_ratio": 1.0,
                },
                "success_rate": 0.0,
                "steps_completed": 0,
            },
        },
        "best_run": {
            "status": "completed",
            "metrics": {
                "good/0/accuracy": 0.0,
                "good/effect_size": 0.0,
                "good/success_rate": 1.0,
                "good/steps_completed": 5.0,
                "missing/accuracy": 0.0,
                "missing/ece": 1.0,
                "missing/collapse_ratio": 1.0,
                "missing/success_rate": 0.0,
                "missing/steps_completed": 0.0,
            },
        },
    }

    diagnosis = diagnose_experiment(
        experiment_summary=summary,
        experiment_plan={
            "conditions": [{"name": "good"}, {"name": "missing"}]
        },
    )

    assert diagnosis.conditions_completed == ["good"]
    assert diagnosis.conditions_failed == ["missing"]
    assert diagnosis.completion_rate == 0.5


def test_condition_scoped_dataset_failure_does_not_poison_other_condition() -> None:
    validity = assess_production_result(
        returncode=0,
        timed_out=False,
        metrics={
            "good/0/accuracy": 0.6,
            "good/success_rate": 1.0,
            "missing/accuracy": 0.0,
            "missing/success_rate": 0.0,
        },
        stdout=(
            "condition=good seed=0 accuracy: 0.6\n"
            "condition=missing DatasetNotFoundError: dataset missing\n"
            "condition=missing success_rate: 0/1"
        ),
    )

    assert validity.valid
    assert validity.successful
    assert validity.valid_conditions == {"good"}
    assert "missing" in validity.invalid_conditions
    assert set(validity.valid_metrics) == {
        "good/0/accuracy",
        "good/success_rate",
    }
