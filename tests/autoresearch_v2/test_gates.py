from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from researchclaw.autoresearch_v2.gates import (
    DESIGN_BLOCKER_EVIDENCE_PREFIXES,
    LLMDecisionGate,
    _design_verdict,
    _report_preflight,
    _validate_design,
    design_preflight,
)
from researchclaw.autoresearch_v2.ideas import candidate_to_idea
from researchclaw.autoresearch_v2.protocols import compile_screening_protocol


def _idea():
    idea = candidate_to_idea(
        {
            "id": "closed-gate",
            "title": "Closed semantic design gate",
            "family": "calibration",
            "research_question": "Does calibrated feedback help?",
            "falsifiable_hypothesis": "It improves paired accuracy.",
            "closest_prior_work": ["Grounded paper"],
            "novelty_gap": "The exact contrast is not isolated.",
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
            "cheap_pilot": "32 paired examples",
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


def _draft():
    return {
        "protocol_template": "calibration_verifier",
        "pilot_objective": "screen feasibility",
        "pilot_claim_scope": "coarse signal only",
        "research_question": "Does calibrated feedback help?",
        "hypothesis": "It improves paired accuracy.",
        "primary_metric": "paired accuracy difference",
        "metric_direction": "maximize",
        "unit_of_analysis": "paired task",
        "dataset": "GSM8K",
        "screening_access_policy": {
            "input_access": True,
            "within_episode_feedback": False,
            "cross_example_adaptation": False,
            "hidden_labels_for_tuning": False,
            "threshold_tuning": False,
        },
        "models": [{"name": "Qwen2.5-1.5B-Instruct", "role": "subject"}],
        "baselines": ["no-self-improvement"],
        "ablations": ["remove calibration"],
        "arms": [
            {"name": "calibrated", "role": "treatment"},
            {"name": "no-self-improvement", "role": "control"},
        ],
        "pilot": {
            "max_gpus": 1,
            "development_examples": 16,
            "max_examples": 32,
            "max_seeds": 1,
            "timeout_sec": 3600,
        },
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
        "gate_statistic": {
            "name": "paired_accuracy_difference",
            "definition": "mean treatment minus control accuracy",
            "direction": "maximize",
            "threshold": {"value": 0.15, "scale": "proportion"},
            "undefined_policy": "reject",
        },
        "uncertainty": {
            "method": "paired_bootstrap",
            "cluster_unit": "task",
            "confidence_level": 0.9,
            "resamples": 2000,
        },
        "validity_criteria": [
            {
                "id": "completed",
                "metric": "completed_examples",
                "operator": ">=",
                "value": 28,
                "scale": "absolute",
                "description": "at least 28 complete pairs",
            }
        ],
        "promotion_criteria": [
            {
                "id": "primary",
                "metric": "paired_accuracy_difference",
                "operator": ">=",
                "value": 0.15,
                "scale": "proportion",
                "description": "coarse effect",
            }
        ],
        "estimand": "paired mean treatment-control accuracy difference",
        "sample_size_rationale": "screening resolution",
        "workload_budget": {"max_new_tokens": 256},
        "confirmatory_followup": {"claim": "Scale tests the stronger claim."},
    }


def _response(*, blockers=None, risks=None):
    return {
        "schema_version": 2,
        "reason": "Evidence-bound semantic audit.",
        "confidence": 0.9,
        "blockers": list(blockers or []),
        "risks": list(risks or []),
    }


def test_design_schema_without_blockers_promotes() -> None:
    value = _response(risks=["single-model external-validity limit"])
    assert _validate_design(value) == []
    verdict = _design_verdict(value, 7)
    assert verdict.decision == "promote"
    assert verdict.blocker_codes == ()
    assert verdict.raw["required_changes"] == []


@pytest.mark.parametrize(
    ("code", "path"),
    [
        (code, prefixes[0])
        for code, prefixes in DESIGN_BLOCKER_EVIDENCE_PREFIXES.items()
    ],
)
def test_each_closed_taxonomy_blocker_rejects(code: str, path: str) -> None:
    value = _response(
        blockers=[
            {
                "code": code,
                "evidence_paths": [path],
                "explanation": "Terminal semantic blocker.",
            }
        ]
    )
    assert _validate_design(value) == []
    verdict = _design_verdict(value, 11)
    assert verdict.decision == "reject"
    assert verdict.blocker_codes == (code,)


@pytest.mark.parametrize(
    "code",
    [
        "call_budget_mismatch",
        "bootstrap_unspecified",
        "split_leakage",
        "seed_schedule_missing",
        "parser_unspecified",
    ],
)
def test_mechanical_blocker_codes_are_forbidden(code: str) -> None:
    value = _response(
        blockers=[
            {
                "code": code,
                "evidence_paths": ["/plan/call_ledger"],
                "explanation": "Mechanical issue.",
            }
        ]
    )
    assert any(
        "code is not allowed" in error for error in _validate_design(value)
    )


def test_design_response_cannot_choose_retry_or_cite_mechanics() -> None:
    value = _response(
        blockers=[
            {
                "code": "non_identifiable_contrast",
                "evidence_paths": ["/plan/call_ledger"],
                "explanation": "Wrong evidence ownership.",
            }
        ]
    )
    value["decision"] = "retry"
    errors = _validate_design(value)
    assert "decision is Controller-derived and must be omitted" in errors
    assert any("disallowed path" in error for error in errors)


def test_report_preflight_rejects_missing_or_non_evidence_measured_paths() -> None:
    report = {
        "claims": [
            {
                "claim": "Measured effect.",
                "evidence_paths": [
                    "/idea/primary_metric",
                    "/evidence/pilot/metrics/missing",
                ],
                "strength": "measured",
            }
        ]
    }
    context = {
        "idea": {"primary_metric": "accuracy"},
        "evidence": {
            "pilot": {"metrics": {"endpoint_correct_diff": 0.0}}
        },
    }

    verdict = _report_preflight(report, context)

    assert verdict is not None
    assert verdict.decision == "retry"
    assert any(
        "measured claim must cite /evidence/" in change
        for change in verdict.required_changes
    )
    assert any(
        "missing evidence path" in change
        for change in verdict.required_changes
    )


def test_report_preflight_accepts_existing_json_pointers() -> None:
    report = {
        "claims": [
            {
                "claim": "Measured effect was zero.",
                "evidence_paths": [
                    "/evidence/pilot/metrics/endpoint_correct_diff"
                ],
                "strength": "measured",
            },
            {
                "claim": "The hypothesis predicted improvement.",
                "evidence_paths": ["/idea/falsifiable_hypothesis"],
                "strength": "hypothesis",
            },
        ]
    }
    context = {
        "idea": {"falsifiable_hypothesis": "improves"},
        "evidence": {
            "pilot": {"metrics": {"endpoint_correct_diff": 0.0}}
        },
    }

    assert _report_preflight(report, context) is None


class _Client:
    def __init__(self, value):
        self.value = value
        self.prompts: list[str] = []

    def chat(self, messages, **kwargs):
        del kwargs
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(
            content=json.dumps(self.value),
            model="decision",
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
        )


def test_design_gate_prompt_omits_compiler_mechanics() -> None:
    client = _Client(_response())
    gate = LLMDecisionGate(client=client)
    verdict = gate.review_design(
        _idea(),
        compile_screening_protocol(_idea(), _draft()),
    )
    assert verdict.decision == "promote"
    prompt = client.prompts[0]
    assert '"validate_plan": "passed"' in prompt
    assert '"call_ledger":' not in prompt
    assert '"decision_table":' not in prompt
    assert '"rng_seed": 1729' not in prompt
    assert "Do not return a decision" in prompt


def test_design_preflight_uses_deterministic_blocker_code() -> None:
    idea = _idea()
    idea.candidate.pop("novelty_evidence")
    verdict = design_preflight(idea)
    assert verdict is not None
    assert verdict.decision == "reject"
    assert verdict.blocker_codes == ("novelty_evidence_missing",)
    assert verdict.raw["blockers"][0]["code"] == "novelty_evidence_missing"
