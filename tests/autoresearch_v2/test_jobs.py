from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from researchclaw.autoresearch_v2.gates import GateVerdict
from researchclaw.autoresearch_v2.ideas import candidate_to_idea
from researchclaw.autoresearch_v2.jobs import (
    BuildJobExecutor,
    DesignJobExecutor,
)
from researchclaw.autoresearch_v2.models import (
    AttemptRecord,
    AttemptStatus,
    JobKind,
    JobRecord,
)
from researchclaw.autoresearch_v2.store import V2Store


def _idea():
    idea = candidate_to_idea(
        {
            "id": "repair-test",
            "title": "Repairable screening design",
            "family": "calibration",
            "research_question": "Does a calibrated gate help?",
            "falsifiable_hypothesis": "The gate improves paired accuracy.",
            "closest_prior_work": ["Grounded paper"],
            "novelty_gap": "A gap",
            "datasets": ["GSM8K"],
            "models": ["Qwen2.5-1.5B-Instruct"],
            "compute": {"gpu_count": 1, "wall_clock_hours": 1},
            "primary_metric": "paired accuracy difference",
            "baselines": ["no-self-improvement"],
            "ablations": ["remove calibration"],
            "failure_safety_tests": ["heldout isolation"],
            "implementation_feasibility": "public stack",
            "licensing_feasibility": "permissive",
            "information_gain_if_true": "useful",
            "information_gain_if_false": "rules it out",
            "cheap_pilot": "32 paired public benchmark examples",
            "scores": {
                "novelty": 8,
                "scientific_importance": 8,
                "falsifiability": 9,
                "compute_tractability": 9,
                "reproducibility": 9,
                "meaningful_result_likelihood": 8,
                "risk": 2,
            },
        }
    )
    idea.candidate["novelty_evidence"] = {
        "available": True,
        "closest_papers": [{"title": "Grounded paper"}],
    }
    return idea


class _Role:
    def __init__(self, value):
        self.value = value
        self.prompts: list[str] = []

    def call(self, prompt: str, **kwargs):
        del kwargs
        self.prompts.append(prompt)
        return SimpleNamespace(value=self.value, total_tokens=10)


class _Gate:
    def review_design(self, idea, plan):
        del idea, plan
        return GateVerdict("promote", "ok", 1.0)


def _plan():
    return {
        "study_phase": "screening_pilot",
        "pilot_objective": "screen feasibility",
        "pilot_claim_scope": "coarse signal only",
        "research_question": "Does it work?",
        "hypothesis": "It helps",
        "primary_metric": "paired accuracy difference",
        "metric_direction": "maximize",
        "unit_of_analysis": "paired example",
        "datasets": [
            {
                "name": "GSM8K",
                "split_role": "screening",
                "split_id": "gsm8k-screening-v1",
                "used_for_adaptation": False,
            },
            {
                "name": "GSM8K-confirmatory",
                "split_role": "heldout_confirmatory",
                "split_id": "gsm8k-confirmatory-v1",
                "used_for_adaptation": False,
            }
        ],
        "models": [{"name": "Qwen", "role": "subject"}],
        "baselines": ["no-self-improvement"],
        "ablations": ["remove mechanism"],
        "arms": [
            {"name": "treatment", "role": "treatment"},
            {"name": "control", "role": "control"},
        ],
        "pilot": {
            "max_gpus": 1,
            "max_examples": 32,
            "max_seeds": 1,
            "timeout_sec": 7200,
        },
        "sample_accounting": {
            "arms": 2,
            "examples_per_arm": 32,
            "seeds": 1,
            "calls_per_example": 1,
            "total_model_calls": 64,
        },
        "effect_threshold": {"value": 0.15, "scale": "proportion"},
        "promotion_rule": "paired effect >= 0.15",
        "early_stop_rule": "invalid protocol",
        "estimand": "paired mean difference",
        "sample_size_rationale": "screening resolution",
        "workload_budget": {
            "conditions": 2,
            "models": 1,
            "examples": 32,
            "seeds": 1,
            "max_new_tokens": 64,
            "estimated_model_calls": 64,
        },
        "decision_table": [
            {"condition": "invalid", "decision": "retry"},
            {"condition": "above threshold", "decision": "promote"},
            {"condition": "below threshold", "decision": "reject"},
        ],
        "confirmatory_followup": {
            "required": True,
            "changes": [
                "increase examples",
                "increase independent seeds",
                "use untouched confirmatory data",
            ],
            "claim": "Only Scale may support the stronger claim.",
            "examples": 100,
            "independent_seeds": [11, 22, 33],
            "split_id": "gsm8k-confirmatory-v1",
            "untouched": True,
        },
        "required_runtime_evidence": ["metrics"],
    }


def test_design_retry_edits_previous_plan_and_review(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="design-job",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
        attempt=1,
    )
    previous = AttemptRecord(
        attempt_id="design-job-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.REJECTED,
    )
    prior_dir = store.prepare_candidate(previous)
    (prior_dir / "plan.json").write_text(
        json.dumps({"research_question": "original question"}),
        encoding="utf-8",
    )
    (prior_dir / "design_review.json").write_text(
        json.dumps({"required_changes": ["fix arithmetic"]}),
        encoding="utf-8",
    )
    store.save_attempt(previous)
    current = AttemptRecord(
        attempt_id="design-job-attempt-02",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=2,
        status=AttemptStatus.RUNNING,
    )
    role = _Role(_plan())

    outcome = DesignJobExecutor(role, decision_gate=_Gate()).execute(
        idea=idea,
        job=job,
        attempt=current,
        store=store,
    )

    assert outcome.success
    assert "This is a REVISION attempt" in role.prompts[0]
    assert "original question" in role.prompts[0]
    assert "fix arithmetic" in role.prompts[0]
    assert "Do not design a different study" in role.prompts[0]


def test_first_design_attempt_has_no_revision_directive() -> None:
    prompt = DesignJobExecutor._prompt(
        _idea(),
        prior_failure={},
    )
    assert "This is a REVISION attempt" not in prompt
    assert '"study_phase": "screening_pilot"' in prompt
    assert '"confirmatory_followup": {' in prompt
    assert '"split_id": "confirmatory-v1"' in prompt
    assert '"split_role": "screening"' in prompt
    assert "must NOT claim" in prompt


def test_build_smoke_can_defer_to_controller_managed_gpu_environment(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    current = store.current_dir(idea.idea_id)
    current.mkdir(parents=True)
    (current / "plan.json").write_text(
        json.dumps(_plan()),
        encoding="utf-8",
    )
    job = JobRecord(
        job_id="build-job",
        idea_id=idea.idea_id,
        kind=JobKind.BUILD,
    )
    store.save_job(job)
    attempt = AttemptRecord(
        attempt_id="build-job-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    source = """
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--mode")
parser.add_argument("--output")
args = parser.parse_args()
output = Path(os.environ["AUTORESEARCH_V2_OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
dataset = load_dataset("gsm8k", split="test")
model = AutoModelForCausalLM.from_pretrained("Qwen/test-model")
Path("metrics.json").write_text(json.dumps({"result_valid": True}))
Path("runtime_evidence.json").write_text(
    json.dumps({"model_loaded": str(model), "datasets_loaded": [str(dataset)]})
)
if args.mode == "smoke":
    print("smoke-ok")
""".strip()
    role = _Role(
        {
            "files": {"main.py": source},
            "commands": {
                "smoke": [
                    "python",
                    "main.py",
                    "--mode",
                    "smoke",
                    "--output",
                    "artifacts/smoke",
                ],
                "pilot": [
                    "python",
                    "main.py",
                    "--mode",
                    "pilot",
                    "--output",
                    "artifacts/pilot",
                ],
                "scale": [
                    "python",
                    "main.py",
                    "--mode",
                    "scale",
                    "--output",
                    "artifacts/scale",
                ],
            },
            "dependencies": ["transformers", "datasets"],
            "expected_outputs": [
                "metrics.json",
                "runtime_evidence.json",
            ],
        }
    )

    outcome = BuildJobExecutor(
        role,
        execute_smoke_locally=False,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    smoke = outcome.result["validation"]["smoke"]
    assert smoke == {
        "ok": True,
        "executed": False,
        "environment": "gpu_pool",
        "reason": "deferred_to_controller_managed_remote_smoke",
    }


def test_build_smoke_uses_configured_local_interpreter(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    current = store.current_dir(idea.idea_id)
    current.mkdir(parents=True)
    (current / "plan.json").write_text(
        json.dumps(_plan()),
        encoding="utf-8",
    )
    job = JobRecord(
        job_id="build-local-smoke",
        idea_id=idea.idea_id,
        kind=JobKind.BUILD,
    )
    attempt = AttemptRecord(
        attempt_id="build-local-smoke-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    source = """
import argparse
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--mode")
args = parser.parse_args()
Path(os.environ["AUTORESEARCH_V2_OUTPUT_DIR"]).mkdir(
    parents=True,
    exist_ok=True,
)
output = Path(os.environ["AUTORESEARCH_V2_OUTPUT_DIR"])
dataset = load_dataset("gsm8k", split="test")
model = AutoModelForCausalLM.from_pretrained("Qwen/test-model")
Path("metrics.json").write_text(
    '{"result_valid": true}',
)
Path("runtime_evidence.json").write_text(
    '{"model_loaded": true, "datasets_loaded": ["gsm8k"]}',
)
""".strip()
    role = _Role(
        {
            "files": {"main.py": source},
            "commands": {
                "smoke": ["python", "main.py", "--mode", "smoke"],
                "pilot": ["python", "main.py", "--mode", "pilot"],
                "scale": ["python", "main.py", "--mode", "scale"],
            },
            "dependencies": ["transformers", "datasets"],
            "expected_outputs": [
                "metrics.json",
                "runtime_evidence.json",
            ],
        }
    )

    outcome = BuildJobExecutor(
        role,
        python_executable="/definitely/missing/python",
        execute_smoke_locally=True,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert not outcome.success
    smoke = outcome.result["validation"]["smoke"]
    assert smoke["argv"][0] == "/definitely/missing/python"
    assert smoke["returncode"] == -1
