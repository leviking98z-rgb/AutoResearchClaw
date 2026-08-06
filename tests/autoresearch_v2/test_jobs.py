from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from researchclaw.autoresearch_v2.gates import GateVerdict
from researchclaw.autoresearch_v2.ideas import candidate_to_idea
from researchclaw.autoresearch_v2.jobs import (
    BuildJobExecutor,
    DesignJobExecutor,
    ReportJobExecutor,
    _collect_report_evidence,
    resolve_experiment_lifecycle_gate,
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


class _ReportGate:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.reports: list[dict] = []
        self.evidence: list[dict] = []

    def review_report(self, idea, report, evidence):
        del idea
        self.reports.append(dict(report))
        self.evidence.append(dict(evidence))
        return self.verdicts.pop(0)


def test_valid_promotion_reject_becomes_reportable_negative() -> None:
    gate = resolve_experiment_lifecycle_gate(
        runtime_evidence={
            "evidence_valid": True,
            "gate_statistic_defined": True,
            "gate_decision": "reject",
            "criterion_results": {
                "minimum_completed": {"value": 24.0, "passed": True},
                "primary_effect": {"value": 0.0, "passed": False},
            },
        },
        gate={
            "decision": "reject",
            "reason": "preregistered effect threshold not met",
            "confidence": 0.99,
            "risks": [],
            "required_changes": [],
        },
    )

    assert gate["decision"] == "complete_negative"
    assert gate["scientific_decision"] == "reject"
    assert gate["report_disposition"] == "reportable_negative"


def test_invalid_runtime_reject_remains_terminal_reject() -> None:
    gate = resolve_experiment_lifecycle_gate(
        runtime_evidence={
            "evidence_valid": False,
            "gate_statistic_defined": True,
            "gate_decision": "reject",
            "criterion_results": {
                "minimum_completed": {"value": 0.0, "passed": False},
            },
        },
        gate={"decision": "reject", "reason": "invalid evidence"},
    )

    assert gate["decision"] == "reject"


def test_report_evidence_uses_stable_nested_paths(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    pilot = root / "pilot"
    pilot.mkdir(parents=True)
    (pilot / "metrics.json").write_text(
        json.dumps(
            {
                "decision": "reject",
                "result_valid": True,
                "metrics": {
                    "endpoint_correct_diff": 0.0,
                    "total_model_calls": 504.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (pilot / "runtime_evidence.json").write_text(
        json.dumps(
            {
                "evidence_valid": True,
                "gate_decision": "reject",
                "gate_statistic_defined": True,
                "gpu_count": 1,
                "dataset_roles": {
                    "GSM8K screening": {
                        "role": "screening",
                        "split_id": "gsm8k-screening-v1",
                    }
                },
                "uncertainty": {
                    "available": False,
                    "decision_role": "descriptive",
                },
                "criterion_results": {
                    "primary_effect": {"value": 0.0, "passed": False}
                },
                "call_counts": {"total_calls": 504},
            }
        ),
        encoding="utf-8",
    )

    evidence = _collect_report_evidence(root)

    assert evidence["pilot"]["metrics"]["endpoint_correct_diff"] == 0.0
    assert evidence["pilot"]["runtime"]["call_counts"]["total_calls"] == 504
    assert evidence["pilot"]["runtime"]["gpu_count"] == 1
    assert evidence["pilot"]["runtime"]["dataset_roles"][
        "GSM8K screening"
    ]["role"] == "screening"
    assert evidence["pilot"]["runtime"]["uncertainty"][
        "decision_role"
    ] == "descriptive"


def test_report_retries_in_place_and_commits_corrected_package(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    idea.candidate["final_outcome"] = "informative_negative"
    store.save_idea(idea)
    current = store.current_dir(idea.idea_id)
    (current / "artifacts" / "pilot").mkdir(parents=True)
    (current / "plan.json").write_text(
        json.dumps({"workload": {"total_calls": 288}}),
        encoding="utf-8",
    )
    (current / "artifacts" / "pilot" / "metrics.json").write_text(
        json.dumps(
            {
                "decision": "reject",
                "result_valid": True,
                "metrics": {
                    "endpoint_correct_diff": 0.0,
                    "total_model_calls": 504.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (current / "artifacts" / "pilot" / "runtime_evidence.json").write_text(
        json.dumps(
            {
                "evidence_valid": True,
                "gate_decision": "reject",
                "gate_statistic_defined": True,
            }
        ),
        encoding="utf-8",
    )
    job = JobRecord(
        job_id=f"{idea.idea_id}-report",
        idea_id=idea.idea_id,
        kind=JobKind.REPORT,
    )
    attempt = AttemptRecord(
        attempt_id=f"{job.job_id}-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    first = {
        "title": "Draft",
        "claims": [
            {
                "claim": "Numeric results are unavailable.",
                "evidence_paths": ["/evidence/pilot/metrics"],
                "strength": "measured",
            }
        ],
        "limitations": [],
        "next_experiments": [],
        "paper_markdown": "# Draft",
    }
    repaired = {
        "title": "Corrected",
        "claims": [
            {
                "claim": "The measured endpoint contrast was zero.",
                "evidence_paths": [
                    "/evidence/pilot/metrics/endpoint_correct_diff"
                ],
                "strength": "measured",
            }
        ],
        "limitations": ["Pilot evidence only."],
        "next_experiments": [],
        "paper_markdown": "# Corrected\n\nThe endpoint contrast was 0.",
    }
    class _SequentialReportRole:
        def __init__(self, values):
            self.values = list(values)
            self.prompts: list[str] = []

        def call(self, prompt, **kwargs):
            del kwargs
            self.prompts.append(prompt)
            return SimpleNamespace(
                value=self.values.pop(0),
                total_tokens=10,
            )

    role = _SequentialReportRole([first, repaired])
    gate = _ReportGate(
        [
            GateVerdict(
                "retry",
                "include measured values",
                1.0,
                required_changes=("include measured values",),
                raw={
                    "decision": "retry",
                    "reason": "include measured values",
                    "confidence": 1.0,
                    "risks": [],
                    "required_changes": ["include measured values"],
                },
            ),
            GateVerdict(
                "complete",
                "evidence complete",
                1.0,
                raw={
                    "decision": "complete",
                    "reason": "evidence complete",
                    "confidence": 1.0,
                    "risks": [],
                    "required_changes": [],
                },
            ),
        ]
    )

    outcome = ReportJobExecutor(role, decision_gate=gate).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert outcome.decision == "complete"
    assert len(role.prompts) == 2
    report = json.loads(
        (store.current_dir(idea.idea_id) / "report.json").read_text()
    )
    assert report["title"] == "Corrected"
    assert gate.evidence[0]["plan"]["workload"]["total_calls"] == 288
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert durable.status is AttemptStatus.ACCEPTED
    assert len(durable.validation["report_revisions"]) == 2


def test_report_recovers_full_snapshot_after_interrupted_candidate_loss(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    current = store.current_dir(idea.idea_id)
    pilot = current / "artifacts" / "pilot"
    pilot.mkdir(parents=True)
    (current / "plan.json").write_text(
        json.dumps({"workload": {"total_calls": 1}}),
        encoding="utf-8",
    )
    (current / "main.py").write_text(
        "print('evidence')\n",
        encoding="utf-8",
    )
    (pilot / "metrics.json").write_text(
        json.dumps(
            {
                "decision": "reject",
                "result_valid": True,
                "metrics": {"endpoint_correct_diff": 0.0},
            }
        ),
        encoding="utf-8",
    )
    job = JobRecord(
        job_id=f"{idea.idea_id}-report",
        idea_id=idea.idea_id,
        kind=JobKind.REPORT,
    )
    attempt = AttemptRecord(
        attempt_id=f"{job.job_id}-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    original_snapshot = store.snapshot_current

    def interrupted_snapshot(record):
        candidate = original_snapshot(record)
        for child in list(candidate.iterdir()):
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
        return candidate

    store.snapshot_current = interrupted_snapshot  # type: ignore[method-assign]
    report = {
        "title": "Recovered report",
        "claims": [],
        "limitations": ["Pilot only."],
        "next_experiments": [],
        "paper_markdown": "# Recovered report",
    }

    outcome = ReportJobExecutor(_Role(report)).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    committed = store.current_dir(idea.idea_id)
    assert (committed / "paper.md").read_text() == "# Recovered report"
    assert (committed / "plan.json").exists()
    assert (committed / "main.py").exists()
    assert (committed / "artifacts" / "pilot" / "metrics.json").exists()


def test_report_uses_full_targeted_revision_budget(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    current = store.current_dir(idea.idea_id)
    (current / "artifacts" / "pilot").mkdir(parents=True)
    (current / "plan.json").write_text("{}", encoding="utf-8")
    (current / "artifacts" / "pilot" / "metrics.json").write_text(
        json.dumps(
            {
                "decision": "reject",
                "result_valid": True,
                "metrics": {"endpoint_correct_diff": 0.0},
            }
        ),
        encoding="utf-8",
    )
    job = JobRecord(
        job_id=f"{idea.idea_id}-report",
        idea_id=idea.idea_id,
        kind=JobKind.REPORT,
    )
    attempt = AttemptRecord(
        attempt_id=f"{job.job_id}-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    reports = [
        {
            "title": f"Draft {index}",
            "claims": [
                {
                    "claim": "The measured endpoint contrast was zero.",
                    "evidence_paths": [
                        "/evidence/pilot/metrics/endpoint_correct_diff"
                    ],
                    "strength": "measured",
                }
            ],
            "limitations": [],
            "next_experiments": [],
            "paper_markdown": f"# Draft {index}",
        }
        for index in range(4)
    ]

    class _SequentialReportRole:
        def __init__(self, values):
            self.values = list(values)

        def call(self, prompt, **kwargs):
            del prompt, kwargs
            return SimpleNamespace(
                value=self.values.pop(0),
                total_tokens=10,
            )

    gate = _ReportGate(
        [
            GateVerdict("retry", "repair one", 1.0),
            GateVerdict("retry", "repair two", 1.0),
            GateVerdict("retry", "repair three", 1.0),
            GateVerdict("complete", "evidence complete", 1.0),
        ]
    )
    outcome = ReportJobExecutor(
        _SequentialReportRole(reports),
        decision_gate=gate,
        max_revisions=3,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert outcome.decision == "complete"
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert len(durable.validation["report_revisions"]) == 4


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
    assert "fix arithmetic" not in role.prompts[0]
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


def test_legacy_design_retry_becomes_terminal_contract_reject(
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
    assert outcome.decision == "reject"
    assert "design_gate_contract_failure" in outcome.reason
    assert len(role.prompts) == 1
    assert len(gate.plans) == 1
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert [item["decision"] for item in durable.validation["design_revisions"]] == [
        "retry",
    ]


def test_semantic_design_blocker_is_terminal_reject(
    tmp_path: Path,
) -> None:
    class _BlockingGate:
        def review_design(self, idea, plan):
            del idea, plan
            return GateVerdict(
                "reject",
                "the contrast is not identifiable",
                1.0,
                raw={
                    "schema_version": 2,
                    "decision": "reject",
                    "reason": "the contrast is not identifiable",
                    "confidence": 1.0,
                    "blocker_codes": ["non_identifiable_contrast"],
                    "blockers": [
                        {
                            "code": "non_identifiable_contrast",
                            "evidence_paths": ["/plan/estimand"],
                            "explanation": "two mechanisms change together",
                        }
                    ],
                    "risks": [],
                    "required_changes": [],
                },
                blocker_codes=("non_identifiable_contrast",),
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
        decision_gate=_BlockingGate(),
        max_revisions=1,
    ).execute(
        idea=idea,
        job=job,
        attempt=attempt,
        store=store,
    )

    assert outcome.success
    assert outcome.decision == "reject"
    assert len(role.prompts) == 1
    durable = store.get_attempt(attempt.attempt_id)
    assert durable is not None
    assert durable.status is AttemptStatus.REJECTED
    assert durable.validation["design_revisions"][0]["blocker_codes"] == [
        "non_identifiable_contrast"
    ]


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
    prompt = DesignJobExecutor(
        _Role(_typed_draft()),
        available_models=("Qwen/Qwen2.5-1.5B-Instruct",),
        available_datasets=("openai/gsm8k",),
    )._prompt(
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
    assert "Qwen/Qwen2.5-1.5B-Instruct" in prompt
    assert "openai/gsm8k" in prompt


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
dataset = load_dataset("openai/gsm8k", split="test")
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
    build = json.loads(
        (store.attempt_dir(attempt) / "candidate" / "build.json").read_text(
            encoding="utf-8"
        )
    )
    assert build["controller_runtime"]["schema"] == (
        "autoresearch_v2.controller_runtime"
    )
    assert build["commands"]["smoke"][:2] == [
        "python",
        "_autoresearch_runtime.py",
    ]
    assert build["controller_runtime"]["core_commands"]["pilot"][-2:] == [
        "--output",
        "artifacts/pilot",
    ]
    assert (
        store.attempt_dir(attempt)
        / "candidate"
        / "_autoresearch_runtime.py"
    ).is_file()


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
dataset = load_dataset("openai/gsm8k", split="test")
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
