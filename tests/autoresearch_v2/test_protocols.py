from __future__ import annotations

import pytest

from researchclaw.autoresearch_v2.ideas import candidate_to_idea
from researchclaw.autoresearch_v2.protocols import compile_screening_protocol
from researchclaw.autoresearch_v2.validation import (
    validate_plan,
    validate_runtime_against_contract,
)


def _idea():
    idea = candidate_to_idea(
        {
            "id": "typed-verifier",
            "title": "Verifier feedback bandwidth",
            "family": "verifier",
            "research_question": "How much verifier feedback is needed?",
            "falsifiable_hypothesis": (
                "Error-class feedback preserves most repair gain."
            ),
            "closest_prior_work": ["Self-repair"],
            "novelty_gap": "Feedback bandwidth is not isolated.",
            "datasets": ["HumanEval"],
            "models": ["Qwen2.5-Coder-1.5B-Instruct"],
            "compute": {"gpu_count": 1, "wall_clock_hours": 1},
            "primary_metric": "paired pass-rate difference",
            "baselines": ["no-self-improvement"],
            "ablations": ["remove verifier feedback"],
            "failure_safety_tests": ["heldout isolation"],
            "implementation_feasibility": "public stack",
            "licensing_feasibility": "verify licenses",
            "information_gain_if_true": "use less feedback",
            "information_gain_if_false": "rule out compression",
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
    return idea


def _draft():
    return {
        "protocol_template": "calibration_verifier",
        "pilot_objective": "screen protocol feasibility",
        "pilot_claim_scope": "coarse signal only",
        "research_question": "Does compressed feedback retain repair gain?",
        "hypothesis": "Compressed feedback retains at least half the gain.",
        "primary_metric": "paired pass-rate difference",
        "metric_direction": "maximize",
        "unit_of_analysis": "paired task",
        "dataset": "HumanEval",
        "models": [
            {"name": "Qwen2.5-Coder-1.5B-Instruct", "role": "subject"}
        ],
        "baselines": ["no-self-improvement control"],
        "ablations": ["remove verifier feedback"],
        "arms": [
            {"name": "compressed verifier", "role": "treatment"},
            {"name": "no-self-improvement", "role": "control"},
        ],
        "pilot": {
            "max_gpus": 1,
            "development_examples": 16,
            "max_examples": 32,
            "max_seeds": 1,
            "timeout_sec": 7200,
        },
        "call_ledger": {
            "components": [
                {
                    "name": "candidate_generation",
                    "scope": "per_example_seed",
                    "dataset_role": "screening",
                    "calls_per_unit": 1,
                },
                {
                    "name": "verifier_scoring",
                    "scope": "per_arm_example_seed",
                    "dataset_role": "screening",
                    "calls_per_unit": 2,
                },
            ]
        },
        "effect_threshold": {"value": 0.15, "scale": "proportion"},
        "promotion_rule": "promote on a coarse paired signal",
        "early_stop_rule": "retry invalid evidence; reject valid futility",
        "estimand": "paired mean treatment-control difference",
        "sample_size_rationale": "screening resolution only",
        "workload_budget": {"max_new_tokens": 256},
        "confirmatory_followup": {
            "claim": "Scale tests the stronger multi-seed claim."
        },
    }


def test_compiler_owns_arithmetic_splits_and_decision_regions() -> None:
    plan = compile_screening_protocol(_idea(), _draft())

    assert plan["sample_accounting"] == {
        "arms": 2,
        "development_examples": 16,
        "examples_per_arm": 32,
        "seeds": 1,
        "total_model_calls": 160,
    }
    assert plan["workload_budget"]["estimated_model_calls"] == 160
    assert plan["call_ledger"]["total_model_calls"] == 160
    assert [
        row["condition"]["region"] for row in plan["decision_table"]
    ] == [
        "invalid",
        "at_or_above_effect_threshold",
        "below_effect_threshold",
    ]
    assert len({item["split_id"] for item in plan["datasets"]}) == 3
    assert (
        plan["confirmatory_followup"]["split_id"]
        == plan["datasets"][2]["split_id"]
    )
    assert validate_plan(plan) == []


@pytest.mark.parametrize(
    ("metric_direction", "expected"),
    [
        (
            "maximize",
            {
                "invalid": "retry",
                "below_effect_threshold": "reject",
                "at_or_above_effect_threshold": "promote",
            },
        ),
        (
            "minimize",
            {
                "invalid": "retry",
                "below_effect_threshold": "promote",
                "at_or_above_effect_threshold": "reject",
            },
        ),
    ],
)
def test_compiler_decision_table_respects_metric_direction(
    metric_direction: str,
    expected: dict[str, str],
) -> None:
    draft = _draft()
    draft["metric_direction"] = metric_direction

    plan = compile_screening_protocol(_idea(), draft)

    actual = {
        row["condition"]["region"]: row["decision"]
        for row in plan["decision_table"]
    }
    assert actual == expected
    assert validate_plan(plan) == []


def test_compiler_rejects_unbounded_protocols_and_missing_control() -> None:
    draft = _draft()
    draft["protocol_template"] = "arbitrary_free_form"
    with pytest.raises(ValueError, match="unsupported protocol_template"):
        compile_screening_protocol(_idea(), draft)

    draft = _draft()
    draft["arms"] = [
        {"name": "treatment-a", "role": "treatment"},
        {"name": "treatment-b", "role": "comparison"},
    ]
    with pytest.raises(ValueError, match="reference control"):
        compile_screening_protocol(_idea(), draft)


def test_compiler_accounts_for_shared_initial_and_selected_repair_arms() -> None:
    draft = _draft()
    draft["arms"] = [
        {"name": "compressed repair", "role": "treatment"},
        {"name": "full-trace repair", "role": "comparison"},
        {"name": "no-self-improvement", "role": "control"},
    ]
    draft["call_ledger"] = {
        "components": [
            {
                "name": "candidate_generation",
                "scope": "per_example_seed",
                "dataset_role": "screening",
                "calls_per_unit": 1,
            },
            {
                "name": "adaptation",
                "scope": "per_arm_example_seed",
                "dataset_role": "screening",
                "arms": ["compressed repair", "full-trace repair"],
                "calls_per_unit": 3,
            },
        ]
    }

    plan = compile_screening_protocol(_idea(), draft)

    assert plan["call_ledger"]["total_model_calls"] == 224
    assert [
        component["total_calls"]
        for component in plan["call_ledger"]["components"]
    ] == [192, 32]
    assert validate_plan(plan) == []


def test_compiler_allows_one_component_across_distinct_ledger_scopes() -> None:
    draft = _draft()
    draft["call_ledger"] = {
        "components": [
            {
                "name": "adaptation",
                "scope": "per_arm_example_seed",
                "dataset_role": "development",
                "arms": ["compressed verifier"],
                "calls_per_unit": 1,
            },
            {
                "name": "adaptation",
                "scope": "per_arm_seed",
                "dataset_role": "none",
                "arms": ["compressed verifier"],
                "calls_per_unit": 2,
            },
        ]
    }

    plan = compile_screening_protocol(_idea(), draft)

    assert [item["total_calls"] for item in plan["call_ledger"]["components"]] == [
        16,
        2,
    ]
    assert plan["call_ledger"]["total_model_calls"] == 18
    assert validate_plan(plan) == []


def test_compiler_rejects_exact_duplicate_ledger_identity() -> None:
    draft = _draft()
    component = {
        "name": "adaptation",
        "scope": "per_arm_example_seed",
        "dataset_role": "development",
        "arms": ["compressed verifier"],
        "calls_per_unit": 1,
    }
    draft["call_ledger"] = {"components": [component, dict(component)]}

    with pytest.raises(
        ValueError,
        match="duplicate call_ledger component identity",
    ):
        compile_screening_protocol(_idea(), draft)


def test_runtime_call_ledger_is_enforced() -> None:
    plan = compile_screening_protocol(_idea(), _draft())
    runtime = {
        "gpu_count": 1,
        "examples_processed": 32,
        "examples_by_role": {
            "development": 0,
            "screening": 32,
        },
        "seeds": [0],
        "call_counts": {
            "candidate_generation": 32,
            "verifier_scoring": 129,
        },
    }
    errors = validate_runtime_against_contract(
        plan=plan,
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="pilot",
    )
    assert any(
        "exceed compiled call_ledger.total_model_calls=160" in error
        for error in errors
    )

    runtime["call_counts"]["verifier_scoring"] = 128
    assert validate_runtime_against_contract(
        plan=plan,
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="pilot",
    ) == []


def test_runtime_call_counts_aggregate_repeated_component_names() -> None:
    draft = _draft()
    draft["call_ledger"] = {
        "components": [
            {
                "name": "adaptation",
                "scope": "per_arm_example_seed",
                "dataset_role": "development",
                "arms": ["compressed verifier"],
                "calls_per_unit": 1,
            },
            {
                "name": "adaptation",
                "scope": "per_arm_seed",
                "dataset_role": "none",
                "arms": ["compressed verifier"],
                "calls_per_unit": 2,
            },
        ]
    }
    plan = compile_screening_protocol(_idea(), draft)
    runtime = {
        "gpu_count": 1,
        "examples_processed": 32,
        "examples_by_role": {
            "development": 16,
            "screening": 32,
        },
        "seeds": [0],
        "call_counts": {"adaptation": 18},
    }

    assert validate_runtime_against_contract(
        plan=plan,
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="pilot",
    ) == []

    runtime["call_counts"]["adaptation"] = 19
    errors = validate_runtime_against_contract(
        plan=plan,
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="pilot",
    )
    assert any(
        "call_counts[adaptation]=19 exceed compiled component budget=18"
        in error
        for error in errors
    )
