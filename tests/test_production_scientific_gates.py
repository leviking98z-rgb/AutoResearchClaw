from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stage_impls._experiment_design import (
    _apply_selected_topic_contract,
    _execute_experiment_design,
    _selected_topic_prompt_contract,
)
from researchclaw.pipeline.stage_impls._code_generation import (
    _assess_pilot_envelope,
    _assess_scientific_code_alignment,
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


class _RetryLLM(_FakeLLM):
    def __init__(self, first: str, second: str) -> None:
        super().__init__(first)
        self._responses = iter((first, second))

    def chat(
        self, messages: list[dict[str, str]], **_: object
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=next(self._responses), model="fake")


def _config(
    tmp_path: Path,
    *,
    topic: str,
    benchmark_agent_enabled: bool = False,
) -> RCConfig:
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
                "benchmark_agent": {"enabled": benchmark_agent_enabled},
            },
            "web_search": {"enabled": False},
        },
        project_root=tmp_path,
        check_paths=False,
    )


def _selected_topic() -> dict[str, object]:
    return {
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
        "datasets": ["GSM8K", "MATH", "MBPP", "HumanEval"],
        "models": ["API-backed LLM agent"],
        "primary_metric": (
            "held-out accuracy at selected iteration and tokens per solved task"
        ),
        "baselines": ["no-self-improvement control", "confidence-only gate"],
        "ablations": ["without calibration", "without rollback"],
        "failure_safety_tests": ["accepted regression", "reward hacking"],
        "cheap_pilot": "Run 100 held-out tasks for three self-improvement rounds.",
        "compute": {
            "gpu_count": 1,
            "wall_clock_hours": 1,
            "notes": "cheap pilot before scaling",
        },
        "pivot_policy": "Pivot if the held-out signal is absent after the pilot.",
    }


def _write_selected_topic(run_dir: Path) -> None:
    stage1 = run_dir / "stage-01"
    stage1.mkdir(parents=True, exist_ok=True)
    (stage1 / "selected_topic.json").write_text(
        json.dumps(_selected_topic()), encoding="utf-8"
    )


def _aligned_rsi_plan() -> dict[str, object]:
    selected = _selected_topic()
    return {
        "research_question": selected["research_question"],
        "falsifiable_hypothesis": selected["falsifiable_hypothesis"],
        "objectives": [
            "Measure held-out regression across recursive LLM-agent updates"
        ],
        "datasets": selected["datasets"],
        "models": selected["models"],
        "baselines": selected["baselines"],
        "proposed_methods": ["calibration-aware acceptance and rollback gate"],
        "ablations": selected["ablations"],
        "primary_metric": selected["primary_metric"],
        "metrics": [
            selected["primary_metric"],
            "accepted-regression rate",
            "expected calibration error",
        ],
        "cheap_pilot": selected["cheap_pilot"],
        "pivot_policy": selected["pivot_policy"],
        "risks": ["benchmark leakage", "selection bias"],
        "compute_budget": {"max_gpu": 1, "max_hours": 1},
    }


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


def test_stage9_rejects_raw_squad_bert_top1_replacement_before_anchoring(
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
                "research_question": (
                    "Does fine-tuning BERT improve SQuAD question answering?"
                ),
                "falsifiable_hypothesis": (
                    "BERT fine-tuning increases top-1 answer accuracy on SQuAD."
                ),
                "primary_metric": "top-1 accuracy",
                "objectives": ["Improve extractive question answering"],
                "datasets": ["SQuAD"],
                "models": ["BERT-base"],
                "baselines": ["frozen BERT encoder"],
                "proposed_methods": ["end-to-end BERT fine-tuning"],
                "ablations": ["without span loss"],
                "metrics": ["top-1 accuracy"],
                "risks": ["answer-span leakage"],
                "compute_budget": {"max_gpu": 1, "max_hours": 1},
            }
        )
    )

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(
            tmp_path,
            topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
        ),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.PAUSED
    assert result.decision == "semantic_misalignment"
    assert not (stage_dir / "exp_plan.yaml").exists()
    report = json.loads(
        (stage_dir / "semantic_alignment.json").read_text(encoding="utf-8")
    )
    assert report["phase"] == "pre_benchmark"
    assert report["raw_plan_check"] is True
    assert set(report["explicit_drift_fields"]) == {
        "research_question",
        "falsifiable_hypothesis",
        "primary_metric",
        "datasets",
        "models",
    }


def test_stage9_accepts_compatible_raw_contract_enrichments(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-09"
    stage_dir.mkdir(parents=True)
    selected = _selected_topic()
    _write_selected_topic(run_dir)
    (run_dir / "stage-08").mkdir()
    (run_dir / "stage-08" / "hypotheses.md").write_text(
        "# Hypothesis\nCalibration-aware acceptance reduces LLM regressions.",
        encoding="utf-8",
    )
    llm = _FakeLLM(
        yaml.safe_dump(
            {
                "research_question": (
                    f"{selected['research_question']} Analyze three random "
                    "seeds and stratify results by benchmark."
                ),
                "falsifiable_hypothesis": (
                    f"{selected['falsifiable_hypothesis']} The effect should "
                    "also survive a rollback-policy ablation."
                ),
                "primary_metric": (
                    f"{selected['primary_metric']}, reported with bootstrap "
                    "confidence intervals"
                ),
                "objectives": [
                    "Measure accepted regressions across recursive LLM updates"
                ],
                "datasets": ["GSM8K held-out split", "MBPP test split"],
                "models": [
                    "API-backed LLM agent with deterministic decoding controls"
                ],
                "baselines": selected["baselines"],
                "proposed_methods": [
                    "calibration-aware acceptance and rollback gate"
                ],
                "ablations": selected["ablations"],
                "metrics": [
                    selected["primary_metric"],
                    "accepted-regression rate",
                ],
                "risks": ["benchmark leakage"],
                "compute_budget": {"max_gpu": 1, "max_hours": 1},
            }
        )
    )

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(
            tmp_path,
            topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
        ),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.DONE
    report = json.loads(
        (
            stage_dir / "semantic_alignment_pre_benchmark.json"
        ).read_text(encoding="utf-8")
    )
    assert report["aligned"] is True
    assert report["explicit_drift_fields"] == []


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


def test_stage9_rsi_prompt_uses_selected_datasets_not_cifar_tier(
    tmp_path: Path,
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
    llm = _FakeLLM(yaml.safe_dump(_aligned_rsi_plan()))

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(
            tmp_path,
            topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
        ),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.DONE
    prompt = llm.calls[0][0]["content"]
    assert "MANDATORY SELECTED-TOPIC CONTRACT" in prompt
    assert "GSM8K, MATH, MBPP, HumanEval" in prompt
    assert "CIFAR-10, CIFAR-100, MNIST" not in prompt


def test_stage9_strict_retry_accepts_selected_contract_fields_when_raw_yaml_fails(
    tmp_path: Path,
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
    llm = _RetryLLM(
        "```yaml\ninvalid: [unterminated",
        yaml.safe_dump(
            {
                "objectives": [
                    "Measure regression across recursive LLM-agent updates"
                ],
                "proposed_methods": [
                    "calibration-aware acceptance and rollback gate"
                ],
                "risks": ["benchmark leakage"],
            }
        ),
    )

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(
            tmp_path,
            topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
        ),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.DONE
    plan = yaml.safe_load(
        (stage_dir / "exp_plan.yaml").read_text(encoding="utf-8")
    )
    assert plan["datasets"] == _selected_topic()["datasets"]
    assert plan["models"] == _selected_topic()["models"]
    assert plan["baselines"][:2] == _selected_topic()["baselines"]


def test_stage9_skips_benchmark_agent_for_authoritative_benchmarks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    called = False

    class _ExplodingBenchmarkOrchestrator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            nonlocal called
            called = True
            raise AssertionError("BenchmarkAgent should have been skipped")

    monkeypatch.setattr(
        "researchclaw.agents.benchmark_agent.BenchmarkOrchestrator",
        _ExplodingBenchmarkOrchestrator,
    )
    llm = _FakeLLM(yaml.safe_dump(_aligned_rsi_plan()))

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(
            tmp_path,
            topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
            benchmark_agent_enabled=True,
        ),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.DONE
    assert called is False
    assert not (stage_dir / "benchmark_plan.json").exists()


def test_selected_topic_contract_rejects_benchmark_agent_cifar_override() -> None:
    selected = _selected_topic()
    contaminated = {
        **_aligned_rsi_plan(),
        "datasets": ["CIFAR-10", "STL-10"],
        "models": ["ResNet-50"],
        "baselines": ["DANN", "CORAL"],
    }

    anchored = _apply_selected_topic_contract(
        contaminated,
        selected,
        str(selected["title"]),
    )

    assert anchored["datasets"] == ["GSM8K", "MATH", "MBPP", "HumanEval"]
    assert anchored["models"] == ["API-backed LLM agent"]
    assert anchored["baselines"][:2] == [
        "no-self-improvement control",
        "confidence-only gate",
    ]
    assert anchored["research_question"] == selected["research_question"]
    assert anchored["falsifiable_hypothesis"] == selected["falsifiable_hypothesis"]
    assert anchored["primary_metric"] == selected["primary_metric"]


def test_stage9_discards_contaminated_benchmark_agent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-09"
    stage_dir.mkdir(parents=True)
    selected = _selected_topic()
    selected.pop("baselines")
    stage1 = run_dir / "stage-01"
    stage1.mkdir(parents=True)
    (stage1 / "selected_topic.json").write_text(
        json.dumps(selected),
        encoding="utf-8",
    )
    (run_dir / "stage-08").mkdir()
    (run_dir / "stage-08" / "hypotheses.md").write_text(
        "# Hypothesis\nCalibration-aware acceptance reduces LLM regressions.",
        encoding="utf-8",
    )
    orchestrated = False

    class _ContaminatedBenchmarkOrchestrator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def orchestrate(self, _context: object) -> SimpleNamespace:
            nonlocal orchestrated
            orchestrated = True
            return SimpleNamespace(
                selected_benchmarks=[{"name": "CIFAR-10"}],
                selected_baselines=[{"name": "DANN"}, {"name": "CORAL"}],
                total_llm_calls=4,
                elapsed_sec=120.0,
                to_dict=lambda: {"selected_benchmarks": [{"name": "CIFAR-10"}]},
            )

    monkeypatch.setattr(
        "researchclaw.agents.benchmark_agent.BenchmarkOrchestrator",
        _ContaminatedBenchmarkOrchestrator,
    )
    llm = _FakeLLM(yaml.safe_dump(_aligned_rsi_plan()))

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(
            tmp_path,
            topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
            benchmark_agent_enabled=True,
        ),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.DONE
    assert orchestrated is True
    assert not (stage_dir / "benchmark_plan.json").exists()
    plan = yaml.safe_load(
        (stage_dir / "exp_plan.yaml").read_text(encoding="utf-8")
    )
    assert plan["datasets"] == selected["datasets"]
    assert "CIFAR-10" not in json.dumps(plan)
    assert "DANN" not in json.dumps(plan)
    assert "CORAL" not in json.dumps(plan)


def test_stage9_preserves_full_selected_topic_contract_in_exp_plan(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stage-09"
    stage_dir.mkdir(parents=True)
    selected = _selected_topic()
    _write_selected_topic(run_dir)
    (run_dir / "stage-08").mkdir()
    (run_dir / "stage-08" / "hypotheses.md").write_text(
        "# Hypothesis\nCalibration-aware acceptance reduces LLM regressions.",
        encoding="utf-8",
    )
    llm = _FakeLLM(yaml.safe_dump(_aligned_rsi_plan()))

    result = _execute_experiment_design(
        stage_dir,
        run_dir,
        _config(
            tmp_path,
            topic="Calibration-Aware Acceptance Gates for LLM Self-Improvement",
        ),
        AdapterBundle(),
        llm=llm,
    )

    assert result.status == StageStatus.DONE
    plan = yaml.safe_load(
        (stage_dir / "exp_plan.yaml").read_text(encoding="utf-8")
    )
    assert plan["topic"] == selected["title"]
    assert plan["research_question"] == selected["research_question"]
    assert plan["falsifiable_hypothesis"] == selected["falsifiable_hypothesis"]
    assert plan["primary_metric"] == selected["primary_metric"]
    assert plan["datasets"] == selected["datasets"]
    assert plan["models"] == selected["models"]
    assert plan["baselines"][:2] == selected["baselines"]
    assert plan["ablations"][:2] == selected["ablations"]
    assert plan["cheap_pilot"] == selected["cheap_pilot"]
    assert plan["pivot_policy"] == selected["pivot_policy"]
    assert plan["selected_compute"] == selected["compute"]
    assert plan["selected_topic_contract"]["id"] == selected["id"]
    report = json.loads(
        (stage_dir / "semantic_alignment.json").read_text(encoding="utf-8")
    )
    assert report["aligned"] is True
    assert report["phase"] == "final"


def test_selected_topic_prompt_contract_includes_authoritative_fields() -> None:
    selected = _selected_topic()

    prompt = _selected_topic_prompt_contract(
        selected,
        str(selected["title"]),
    )

    assert str(selected["research_question"]) in prompt
    assert str(selected["falsifiable_hypothesis"]) in prompt
    assert str(selected["primary_metric"]) in prompt
    assert "GSM8K" in prompt
    assert "cheap_pilot" in prompt
    assert "pivot_policy" in prompt


def test_stage10_scientific_gate_rejects_simulated_llm_trajectory_proxy() -> None:
    selected = _selected_topic()
    files = {
        "main.py": """
\"\"\"Synthetic trajectory study.
In a production experiment this would be replaced with actual LLM inference
on Qwen2.5-7B-Instruct and GSM8K/MBPP.
\"\"\"
from models import generate_synthetic_trajectories

def main():
    rows = generate_synthetic_trajectories()
    print(f"calibration_error: {rows[0]['ece']}")

if __name__ == "__main__":
    main()
""",
        "models.py": """
import numpy as np

def generate_synthetic_trajectories():
    rng = np.random.RandomState(42)
    return [{"correctness": int(rng.rand() > 0.5), "ece": rng.rand()}]
""",
    }

    report = _assess_scientific_code_alignment(
        files,
        selected,
        str(selected["title"]),
    )

    assert report["aligned"] is False
    assert report["missing_model_execution"] is True
    assert report["missing_dataset_execution"] is True
    assert report["simulation_hits"]


def test_stage10_scientific_gate_rejects_synthetic_mlp_self_training_proxy() -> None:
    selected = _selected_topic()
    files = {
        "main.py": """
\"\"\"Calibration gate on a synthetic self-training loop with a small MLP.
The model is retrained on its own predictions, simulating self-improvement.
\"\"\"
from sklearn.datasets import make_moons
import torch.nn as nn

class MLP(nn.Module):
    pass

def generate_dataset(seed):
    return make_moons(n_samples=1200, noise=0.3, random_state=seed)

if __name__ == "__main__":
    print(generate_dataset(0))
""",
    }

    report = _assess_scientific_code_alignment(
        files,
        selected,
        str(selected["title"]),
    )

    assert report["aligned"] is False
    assert report["requires_llm_subject"] is True
    assert report["requires_named_benchmark"] is True
    assert report["missing_model_execution"] is True
    assert report["missing_dataset_execution"] is True
    assert {
        hit["kind"] for hit in report["simulation_hits"]
    } >= {
        "synthetic_research_subject",
        "synthetic_benchmark_generator",
    }


def test_stage10_scientific_gate_accepts_real_model_and_benchmark_paths() -> None:
    selected = _selected_topic()
    files = {
        "main.py": """
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct"
    )
    batch = tokenizer(gsm8k[0]["question"], return_tensors="pt")
    generated = model.generate(**batch, max_new_tokens=32)
    print("held_out_accuracy: 0.0")

if __name__ == "__main__":
    main()
""",
    }

    report = _assess_scientific_code_alignment(
        files,
        selected,
        str(selected["title"]),
    )

    assert report["aligned"] is True
    assert report["missing_model_execution"] is False
    assert report["missing_dataset_execution"] is False


def test_stage10_pilot_gate_rejects_full_scale_gpu_campaign() -> None:
    selected = _selected_topic()
    files = {
        "config.py": """
from dataclasses import dataclass, field

@dataclass
class Config:
    num_gpus_available: int = 4
    pilot_rounds: int = 10
    prompts_per_round: int = 512
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
""",
    }

    report = _assess_pilot_envelope(files, selected)

    assert report["aligned"] is False
    assert len(report["reasons"]) == 4


def test_stage10_pilot_gate_accepts_bounded_real_pilot() -> None:
    selected = _selected_topic()
    files = {
        "config.py": """
from dataclasses import dataclass, field

@dataclass
class Config:
    num_gpus_available: int = 1
    pilot_rounds: int = 3
    prompts_per_round: int = 100
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
""",
    }

    report = _assess_pilot_envelope(files, selected)

    assert report["aligned"] is True
    assert report["reasons"] == []


def test_stage10_gate_rejects_fashionmnist_proxy_for_llm_rsi() -> None:
    selected = _selected_topic()
    files = {
        "main.py": """
from torchvision import datasets

SEEDS = [0, 1, 2]

def main():
    dataset = datasets.FashionMNIST(
        root="/opt/datasets", train=True, download=False
    )
    print("Real FashionMNIST + a small MLP proxy", len(dataset))
""",
    }

    report = _assess_scientific_code_alignment(
        files,
        selected,
        str(selected["title"]),
    )

    assert report["aligned"] is False
    assert "fashionmnist" in report["proxy_markers_found"]
    assert report["llm_model_implementation_markers"] == []
    assert report["llm_benchmark_implementation_markers"] == []
    assert any("image/" in reason for reason in report["reasons"])


def test_pilot_gate_detects_uppercase_seeds_and_dict_example_counts() -> None:
    selected = _selected_topic()
    files = {
        "main.py": """
HYPERPARAMETERS = {
    "seed_size": 300,
    "pool_size": 512,
    "dev_size": 256,
    "test_size": 512,
    "iterations": 8,
}
SEEDS = [0, 1, 2, 3]
""",
    }

    report = _assess_pilot_envelope(files, selected)

    assert report["aligned"] is False
    assert report["observed"]["seed_count"] == 4
    assert max(report["observed"]["examples"]) == 512
    assert max(report["observed"]["iterations"]) == 8
    assert any("more than 3 seeds" in reason for reason in report["reasons"])
    assert any(
        "more than 200 examples" in reason for reason in report["reasons"]
    )


def test_stage10_scientific_gate_does_not_treat_generic_open_as_dataset_load() -> None:
    selected = _selected_topic()
    files = {
        "main.py": """
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct"
    )
    with open("notes.txt", "w", encoding="utf-8") as stream:
        stream.write("simulate the benchmark in a production experiment")
    generated = model.generate(
        **tokenizer("question", return_tensors="pt"),
        max_new_tokens=8,
    )
    print(generated)
""",
    }

    report = _assess_scientific_code_alignment(
        files,
        selected,
        str(selected["title"]),
    )

    assert report["aligned"] is False
    assert report["missing_model_execution"] is False
    assert report["missing_dataset_execution"] is True
    assert "open(" not in report["dataset_execution_markers"]


def test_stage10_scientific_gate_does_not_ban_random_seeds_alone() -> None:
    files = {
        "main.py": """
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    rng = np.random.RandomState(7)
    values = rng.normal(size=100)
    train, test = train_test_split(values, random_state=7)
    print(f"mean_shift: {float(test.mean() - train.mean())}")

if __name__ == "__main__":
    main()
""",
    }

    report = _assess_scientific_code_alignment(
        files,
        {},
        "Monte Carlo estimator variance",
    )

    assert report["aligned"] is True
    assert report["authoritative_contract"] is False


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
