from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from researchclaw.autoresearch_v2.gates import GateVerdict
from researchclaw.autoresearch_v2.ideas import candidate_to_idea
from researchclaw.autoresearch_v2.jobs import DesignJobExecutor
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
                "split_role": "heldout",
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
        "confirmatory_followup": (
            "At Scale use more examples, seeds, and a new untouched split."
        ),
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
    assert "must NOT claim" in prompt
