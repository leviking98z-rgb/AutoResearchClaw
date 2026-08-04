from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stage_impls._experiment_design import (
    _execute_experiment_design,
)
from researchclaw.pipeline.stage_impls._paper_writing import (
    _collect_raw_experiment_metrics,
    _execute_paper_draft,
)
from researchclaw.pipeline.stage_impls._review_publish import (
    _execute_export_publish,
    _execute_quality_gate,
)
from researchclaw.pipeline.stages import StageStatus


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict[str, str]]] = []

    def chat(
        self, messages: list[dict[str, str]], **_: object
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.content, model="fake")


def _config(tmp_path: Path, *, topic: str) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "production-scientific-gates", "mode": "docs-first"},
            "research": {
                "topic": topic,
                "domains": ["ml", "llm-agents"],
                "quality_threshold": 5.0,
                "graceful_degradation": True,
            },
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {
                "backend": "markdown",
                "root": str(tmp_path / "kb"),
            },
            "openclaw_bridge": {},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline-test-key",
                "primary_model": "fake",
            },
            "experiment": {
                "mode": "sandbox",
                "metric_key": "calibration_error",
                "metric_direction": "minimize",
                "time_budget_sec": 600,
                "benchmark_agent": {"enabled": False},
            },
            "web_search": {"enabled": False},
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _write_selected_topic(run_dir: Path) -> None:
    selected = {
        "id": "calibration-aware-acceptance",
        "title": "Calibration-Aware Acceptance Gates for LLM Self-Improvement",
        "research_question": (
            "Can calibrated acceptance gates prevent regressions during "
            "self-iterative optimization of LLM agents?"
        ),
        "falsifiable_hypothesis": (
            "A calibration-aware gate lowers accepted-regression rate relative "
            "to unconditional and confidence-only controls."
        ),
        "datasets": ["held-out LLM agent task traces", "accept/reject calibration set"],
        "models": ["API-backed LLM agent"],
        "primary_metric": "accepted-regression rate and expected calibration error",
        "baselines": ["no-self-improvement control", "confidence-only gate"],
        "ablations": ["without calibration", "without rollback"],
    }
    stage1 = run_dir / "stage-01"
    stage1.mkdir(parents=True, exist_ok=True)
    (stage1 / "selected_topic.json").write_text(
        json.dumps(selected), encoding="utf-8"
    )


def _write_failed_placeholder_evidence(run_dir: Path) -> None:
    runs_dir = run_dir / "stage-12" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "run-1.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "failed",
                "metrics": {
                    "accuracy": 0.91,
                    "domain_adaptation_loss": 0.12,
                },
                "stdout": "accuracy: 0.91\ndomain_adaptation_loss: 0.12\n",
                "stderr": "Traceback: training failed",
                "timed_out": False,
            }
        ),
        encoding="utf-8",
    )
    stage14 = run_dir / "stage-14"
    stage14.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_run": {
            "status": "failed",
            "metrics": {"accuracy": 0.91},
        },
        "metrics_summary": {
            "accuracy": {"min": 0.91, "max": 0.91, "mean": 0.91, "count": 1}
        },
        "condition_summaries": {
            "DANN_CIFAR10": {
                "status": "failed",
                "metrics": {"accuracy_mean": 0.91},
            }
        },
    }
    (stage14 / "experiment_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def test_stage9_rejects_rsi_plan_that_drifts_to_vision_domain_adaptation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-09"
    stage_dir.mkdir(parents=True)
    _write_selected_topic(run_dir)
    (run_dir / "stage-08").mkdir()
    (run_dir / "stage-08" / "hypotheses.md").write_text(
        "# Hypothesis\nCalibration-aware acceptance reduces LLM regressions.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "researchclaw.domains.detector.detect_domain",
        lambda **_: (_ for _ in ()).throw(RuntimeError("not needed")),
    )
    llm = _FakeLLM(
        yaml.safe_dump(
            {
                "objectives": ["Improve target-domain image accuracy"],
                "datasets": ["CIFAR-10", "Office-Home"],
                "models": ["ResNet-50"],
                "baselines": ["DANN", "CORAL"],
                "proposed_methods": ["adversarial feature alignment"],
                "ablations": ["without domain discriminator"],
                "metrics": ["top-1 accuracy", "domain adaptation loss"],
                "risks": ["negative transfer"],
                "compute_budget": {"max_gpu": 1, "max_hours": 1},
            }
        )
    )

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(tmp_path, topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement"),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.PAUSED
    assert result.decision == "semantic_misalignment"
    assert not (stage_dir / "exp_plan.yaml").exists()
    report = json.loads(
        (stage_dir / "semantic_alignment.json").read_text(encoding="utf-8")
    )
    assert report["aligned"] is False
    assert any(
        token in " ".join(report["forbidden_drift_terms"]).lower()
        for token in ("cifar", "dann", "coral", "domain adaptation")
    )


def test_stage9_accepts_generic_topic_plan_without_rsi_special_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-09"
    stage_dir.mkdir(parents=True)
    (run_dir / "stage-08").mkdir()
    (run_dir / "stage-08" / "hypotheses.md").write_text(
        "# Hypothesis\nA compact CNN improves CIFAR-10 accuracy.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "researchclaw.domains.detector.detect_domain",
        lambda **_: (_ for _ in ()).throw(RuntimeError("not needed")),
    )
    llm = _FakeLLM(
        yaml.safe_dump(
            {
                "objectives": ["Compare compact image classifiers"],
                "datasets": ["CIFAR-10"],
                "models": ["compact CNN"],
                "baselines": ["ResNet-18"],
                "proposed_methods": ["compact CNN with calibrated augmentation"],
                "ablations": ["without augmentation"],
                "metrics": ["top-1 accuracy"],
                "risks": ["overfitting"],
                "compute_budget": {"max_gpu": 1, "max_hours": 1},
            }
        )
    )

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(tmp_path, topic="Compact image classification on CIFAR-10"),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.DONE
    assert (stage_dir / "exp_plan.yaml").exists()


def test_raw_metric_collection_rejects_failed_placeholder_metrics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_failed_placeholder_evidence(run_dir)

    block, has_evidence = _collect_raw_experiment_metrics(run_dir)

    assert block == ""
    assert has_evidence is False


def test_paper_gate_rejects_synthetic_fallback_run(
    tmp_path: Path,
) -> None:
    from researchclaw.pipeline.stage_impls._paper_writing import (
        _paper_evidence_gate,
    )

    run_dir = tmp_path / "run"
    runs_dir = run_dir / "stage-12" / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "run-1.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "status": "completed",
                "returncode": 0,
                "metrics": {"baseline/0/accuracy": 0.99},
                "stdout": (
                    "Dataset load failed; falling back to synthetic data.\n"
                    "condition=baseline seed=0 accuracy: 0.99"
                ),
                "stderr": "",
                "timed_out": False,
            }
        ),
        encoding="utf-8",
    )

    report = _paper_evidence_gate(
        run_dir,
        _config(tmp_path, topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement"),
    )

    assert report["eligible"] is False
    assert report["valid_runs"] == []
    assert any(
        "synthetic" in reason.lower()
        for record in report["rejected_runs"]
        for reason in record["reasons"]
    )


def test_stage17_blocks_failed_placeholder_evidence_before_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-17"
    stage_dir.mkdir(parents=True)
    _write_selected_topic(run_dir)
    _write_failed_placeholder_evidence(run_dir)
    (run_dir / "stage-16").mkdir()
    (run_dir / "stage-16" / "outline.md").write_text(
        "# Outline\n## Results\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "researchclaw.pipeline.stage_impls._paper_writing._detect_domain",
        lambda *_: ("ml", "machine learning", "NeurIPS"),
    )
    llm = _FakeLLM("# Fabricated paper\nAccuracy is 91%.")

    result = _execute_paper_draft(
        stage_dir,
        run_dir,
        _config(tmp_path, topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement"),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.PAUSED
    assert result.decision == "blocked_invalid_evidence"
    assert llm.calls == []
    meta = json.loads((stage_dir / "paper_meta.json").read_text(encoding="utf-8"))
    assert meta["outcome"] == "blocked_invalid_evidence"
    assert "failed" in " ".join(meta["reasons"]).lower()


def test_stage20_and_stage22_reject_failed_placeholder_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _write_selected_topic(run_dir)
    _write_failed_placeholder_evidence(run_dir)
    stage19 = run_dir / "stage-19"
    stage19.mkdir(parents=True)
    (stage19 / "paper_revised.md").write_text(
        "# Paper\n\n## Results\nCIFAR-10 accuracy was 91%.",
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
    )

    stage20 = run_dir / "stage-20"
    stage20.mkdir()
    quality_result = _execute_quality_gate(
        stage20, run_dir, config, AdapterBundle(), llm=None
    )
    assert quality_result.status == StageStatus.FAILED
    assert quality_result.decision == "blocked_invalid_evidence"
    assert (stage20 / "evidence_gate.json").exists()

    stage22 = run_dir / "stage-22"
    stage22.mkdir()
    export_result = _execute_export_publish(
        stage22, run_dir, config, AdapterBundle(), llm=None
    )
    assert export_result.status == StageStatus.FAILED
    assert export_result.decision == "blocked_invalid_evidence"
    assert not (stage22 / "paper_final.md").exists()
