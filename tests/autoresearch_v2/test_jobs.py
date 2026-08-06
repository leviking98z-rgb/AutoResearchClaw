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
from researchclaw.autoresearch_v2.llm import StructuredRole
from researchclaw.autoresearch_v2.models import (
    AttemptRecord,
    AttemptStatus,
    JobKind,
    JobRecord,
)
from researchclaw.autoresearch_v2.protocols import validate_protocol_draft
from researchclaw.autoresearch_v2.store import V2Store
from researchclaw.autoresearch_v2.validation import validate_plan


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


class _CompilerRepairRole(_Role):
    def __init__(self, values):
        self.values = list(values)
        self.prompts: list[str] = []
        self.repairs: list[tuple[dict, list[str]]] = []

    def call(self, prompt: str, **kwargs):
        del kwargs
        self.prompts.append(prompt)
        return SimpleNamespace(value=self.values.pop(0), total_tokens=10)

    def repair(
        self,
        previous_value,
        errors,
        *,
        retry_context,
        max_tokens,
        temperature,
    ):
        del retry_context, max_tokens, temperature
        self.repairs.append((dict(previous_value), list(errors)))
        return SimpleNamespace(value=self.values.pop(0), total_tokens=10)


class _Gate:
    def __init__(self):
        self.plan = None

    def review_design(self, idea, plan):
        del idea
        self.plan = plan
        return GateVerdict("promote", "ok", 1.0)


class _RepairingGate:
    def __init__(self):
        self.plans = []

    def review_design(self, idea, plan):
        del idea
        self.plans.append(plan)
        if len(self.plans) == 1:
            return GateVerdict(
                "retry",
                "make the endpoint implementation-ready",
                1.0,
                required_changes=("define the exact denominator",),
            )
        return GateVerdict("promote", "fixed", 1.0)


class _RetryClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[str] = []

    def chat(self, messages, **kwargs):
        del kwargs
        self.requests.append(messages[0]["content"])
        value = self.responses.pop(0)
        return SimpleNamespace(
            content=json.dumps(value),
            model="worker",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


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


def _typed_draft():
    plan = _plan()
    return {
        key: value
        for key, value in plan.items()
        if key
        not in {
            "study_phase",
            "datasets",
            "sample_accounting",
            "decision_table",
            "required_runtime_evidence",
        }
    } | {
        "protocol_template": "calibration_verifier",
        "dataset": "GSM8K",
        "screening_access_policy": {
            "input_access": True,
            "within_episode_feedback": False,
            "cross_example_adaptation": False,
            "hidden_labels_for_tuning": False,
            "threshold_tuning": False,
        },
        "gate_statistic": {
            "name": "paired_accuracy_difference",
            "definition": "mean paired treatment-minus-control accuracy",
            "direction": "maximize",
            "threshold": {"value": 0.15, "scale": "proportion"},
            "undefined_policy": "reject",
        },
        "uncertainty": {
            "method": "paired_bootstrap",
            "cluster_unit": "example",
            "confidence_level": 0.90,
            "resamples": 2000,
        },
        "validity_criteria": [
            {
                "id": "completed_examples",
                "metric": "completed_examples",
                "operator": ">=",
                "value": 30,
                "scale": "absolute",
                "description": "at least 30 paired examples complete",
            }
        ],
        "promotion_criteria": [
            {
                "id": "primary_effect",
                "metric": "paired_accuracy_difference",
                "operator": ">=",
                "value": 0.15,
                "scale": "proportion",
                "description": "coarse paired effect",
            }
        ],
        "call_ledger": {
            "components": [
                {
                    "name": "final_evaluation",
                    "scope": "per_arm_example_seed",
                    "dataset_role": "screening",
                    "calls_per_unit": 1,
                }
            ]
        },
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
    role = _Role(_typed_draft())

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


def test_design_executor_compiles_typed_draft_before_decision_gate(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="typed-design",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
    )
    attempt = AttemptRecord(
        attempt_id="typed-design-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    gate = _Gate()

    outcome = DesignJobExecutor(
        _Role(_typed_draft()),
        decision_gate=gate,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert gate.plan is not None
    assert gate.plan["compiler"]["version"] == 2
    assert gate.plan["sample_accounting"]["total_model_calls"] == 64
    assert validate_plan(gate.plan) == []
    assert (
        store.current_dir(idea.idea_id) / "plan.json"
    ).is_file()


def test_design_repairs_decision_retry_inside_one_attempt(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="typed-design-repair",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
        attempt_limit=1,
    )
    attempt = AttemptRecord(
        attempt_id="typed-design-repair-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    role = _Role(_typed_draft())
    gate = _RepairingGate()

    outcome = DesignJobExecutor(
        role,
        decision_gate=gate,
        max_revisions=2,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert outcome.decision == "promote"
    assert len(role.prompts) == 2
    assert "DECISION REVIEW" in role.prompts[1]
    assert "define the exact denominator" in role.prompts[1]
    assert len(gate.plans) == 2
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert [item["decision"] for item in durable.validation["design_revisions"]] == [
        "retry",
        "promote",
    ]


def test_design_exhausts_bounded_internal_revisions(
    tmp_path: Path,
) -> None:
    class _AlwaysRetryGate:
        def review_design(self, idea, plan):
            del idea, plan
            return GateVerdict(
                "retry",
                "still underspecified",
                1.0,
                required_changes=("define the algorithm",),
            )

    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="typed-design-repair-limit",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
        attempt_limit=1,
    )
    attempt = AttemptRecord(
        attempt_id="typed-design-repair-limit-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    role = _Role(_typed_draft())

    outcome = DesignJobExecutor(
        role,
        decision_gate=_AlwaysRetryGate(),
        max_revisions=1,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert not outcome.success
    assert outcome.decision == "retry"
    assert len(role.prompts) == 2
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert durable.status is AttemptStatus.REJECTED
    assert len(durable.validation["design_revisions"]) == 2


def test_design_repairs_compiler_error_inside_revision_budget(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="typed-design-compiler-repair",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
        attempt_limit=1,
    )
    attempt = AttemptRecord(
        attempt_id="typed-design-compiler-repair-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    invalid = _typed_draft()
    invalid["call_ledger"]["components"][0]["scope"] = "per_benchmark_item"
    role = _CompilerRepairRole([invalid, _typed_draft()])

    outcome = DesignJobExecutor(
        role,
        decision_gate=_Gate(),
        max_revisions=2,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert outcome.decision == "promote"
    assert len(role.prompts) == 1
    assert len(role.repairs) == 1
    assert "unsupported call_ledger scope" in role.repairs[0][1][0]
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert durable.validation["design_revisions"][0]["decision"] == (
        "compiler_retry"
    )


def test_design_repairs_compiled_plan_validation_before_decision_gate(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="typed-design-plan-validation-repair",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
        attempt_limit=1,
    )
    attempt = AttemptRecord(
        attempt_id="typed-design-plan-validation-repair-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    invalid = _typed_draft()
    invalid["gate_statistic"]["threshold"]["value"] = 0.01
    invalid["promotion_criteria"][0]["value"] = 0.01
    role = _CompilerRepairRole([invalid, _typed_draft()])
    gate = _Gate()

    outcome = DesignJobExecutor(
        role,
        decision_gate=gate,
        max_revisions=2,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert len(role.repairs) == 1
    assert "below pilot sample resolution" in role.repairs[0][1][0]
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert durable.validation["design_revisions"][0]["decision"] == (
        "plan_validation_retry"
    )


def test_design_continues_after_structured_role_exhausts_local_retries(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="typed-design-validation-repair",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
        attempt_limit=1,
    )
    attempt = AttemptRecord(
        attempt_id="typed-design-validation-repair-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    invalid = _typed_draft()
    invalid["pilot"]["max_examples"] = 50
    client = _RetryClient([invalid, invalid, _typed_draft()])
    role = StructuredRole(
        client=client,
        system="return json",
        validator=validate_protocol_draft,
        max_attempts=2,
    )

    outcome = DesignJobExecutor(
        role,
        decision_gate=_Gate(),
        max_revisions=2,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert outcome.decision == "promote"
    assert len(client.requests) == 3
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert durable.validation["design_revisions"][0]["decision"] == (
        "draft_validation_retry"
    )


def test_first_design_attempt_has_no_revision_directive() -> None:
    prompt = DesignJobExecutor._prompt(
        _idea(),
        prior_failure={},
    )
    assert "This is a REVISION attempt" not in prompt
    assert '"protocol_template": "calibration_verifier"' in prompt
    assert '"confirmatory_followup": {' in prompt
    assert '"call_ledger": {' in prompt
    assert '"gate_statistic": {' in prompt
    assert '"screening_access_policy": {' in prompt
    assert "Controller creates disjoint" in prompt
    assert "Mechanical fields that the Controller owns" in prompt


def test_design_structured_retry_repairs_prior_json_locally() -> None:
    invalid = _typed_draft()
    invalid["gate_statistic"]["threshold"] = {
        "value": 0.15,
        "scale": "proportion_points",
    }
    client = _RetryClient([invalid, _typed_draft()])
    role = StructuredRole(
        client=client,
        system="return json",
        validator=lambda value: (
            []
            if value["gate_statistic"]["threshold"]["scale"]
            == "proportion"
            else ["invalid gate_statistic.threshold.scale"]
        ),
    )

    result = role.call(
        "make a plan",
        max_tokens=100,
        temperature=0.2,
        retry_context=DesignJobExecutor._validation_repair_context,
    )

    assert result.attempts == 2
    assert "Repair the exact prior JSON below" in client.requests[1]
    assert '"scale": "proportion_points"' in client.requests[1]
    assert "do not redesign the study" in client.requests[1]
    assert "per_arm_example_seed" in client.requests[1]
    assert "Controller derives those fields" in client.requests[1]
    assert "valid but unfavorable" in client.requests[1]


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
