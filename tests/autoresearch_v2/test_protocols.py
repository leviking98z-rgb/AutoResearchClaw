from __future__ import annotations

import pytest

from researchclaw.autoresearch_v2.ideas import candidate_to_idea
from researchclaw.autoresearch_v2.protocols import (
    COMPILER_OWNED_DESIGN_FIELDS,
    compile_screening_protocol,
    design_gate_view,
)
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
        "screening_access_policy": {
            "input_access": True,
            "within_episode_feedback": False,
            "cross_example_adaptation": False,
            "hidden_labels_for_tuning": False,
            "threshold_tuning": False,
        },
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
        "gate_statistic": {
            "name": "paired_pass_rate_difference",
            "definition": (
                "mean paired treatment-minus-control pass-rate difference"
            ),
            "direction": "maximize",
            "threshold": {"value": 0.15, "scale": "proportion"},
            "undefined_policy": "reject",
        },
        "uncertainty": {
            "method": "paired_bootstrap",
            "cluster_unit": "task",
            "confidence_level": 0.90,
            "resamples": 2000,
        },
        "validity_criteria": [
            {
                "id": "completed_tasks",
                "metric": "completed_tasks",
                "operator": ">=",
                "value": 30,
                "scale": "absolute",
                "description": "at least 30 paired tasks completed",
            }
        ],
        "promotion_criteria": [
            {
                "id": "primary_effect",
                "metric": "paired_pass_rate_difference",
                "operator": ">=",
                "value": 0.15,
                "scale": "proportion",
                "description": "coarse paired improvement threshold",
            },
            {
                "id": "positive_ci",
                "metric": "paired_effect_ci_lower",
                "operator": ">",
                "value": 0.0,
                "scale": "absolute",
                "description": "uncertainty excludes no improvement",
            },
        ],
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
        "meets_all_promotion_criteria",
        "valid_otherwise",
    ]
    assert plan["compiler"]["version"] == 2
    assert plan["decision_contract"]["valid_otherwise"]["decision"] == "reject"
    assert (
        plan["datasets"][1]["access_policy"]
        == {
            **_draft()["screening_access_policy"],
            "available_before_scale": True,
        }
    )
    assert len({item["split_id"] for item in plan["datasets"]}) == 3
    assert (
        plan["confirmatory_followup"]["split_id"]
        == plan["datasets"][2]["split_id"]
    )
    assert validate_plan(plan) == []


def test_compiler_gate_direction_is_independent_of_raw_metric_direction() -> None:
    draft = _draft()
    draft["metric_direction"] = "minimize"
    draft["primary_metric"] = "selection regret"
    draft["gate_statistic"] = {
        "name": "relative_regret_reduction",
        "definition": (
            "(control mean regret - treatment mean regret) / "
            "control mean regret"
        ),
        "direction": "maximize",
        "threshold": {"value": 0.20, "scale": "proportion"},
        "undefined_policy": "reject",
    }
    draft["promotion_criteria"][0] = {
        "id": "primary_effect",
        "metric": "relative_regret_reduction",
        "operator": ">=",
        "value": 0.20,
        "scale": "proportion",
        "description": "relative regret reduction reaches 20%",
    }

    plan = compile_screening_protocol(_idea(), draft)

    actual = {
        row["condition"]["region"]: row["decision"]
        for row in plan["decision_table"]
    }
    assert actual == {
        "invalid": "retry",
        "meets_all_promotion_criteria": "promote",
        "valid_otherwise": "reject",
    }
    assert plan["metric_direction"] == "minimize"
    assert plan["gate_statistic"]["direction"] == "maximize"
    assert validate_plan(plan) == []


def test_compiler_rejects_primary_criterion_that_inverts_gate_direction() -> None:
    draft = _draft()
    draft["promotion_criteria"][0]["operator"] = "<="

    with pytest.raises(ValueError, match="operator conflicts"):
        compile_screening_protocol(_idea(), draft)


def test_compiler_encodes_within_episode_feedback_without_label_tuning() -> None:
    draft = _draft()
    draft["screening_access_policy"]["within_episode_feedback"] = True

    plan = compile_screening_protocol(_idea(), draft)
    screening = plan["datasets"][1]
    confirmatory = plan["datasets"][2]

    assert screening["used_for_adaptation"] is False
    assert screening["access_policy"]["within_episode_feedback"] is False
    assert screening["access_policy"]["cross_example_adaptation"] is False
    assert screening["access_policy"]["hidden_labels_for_tuning"] is False
    assert confirmatory["access_policy"]["input_access"] is True
    assert confirmatory["access_policy"]["available_before_scale"] is False
    assert validate_plan(plan) == []


def test_compiler_owns_single_subject_model_and_bootstrap_mechanics() -> None:
    draft = _draft()
    draft["models"] = [
        {"name": "Qwen-subject", "role": "subject"},
        {"name": "Qwen-verifier", "role": "verifier"},
    ]
    draft["pilot"]["development_examples"] = 31

    plan = compile_screening_protocol(_idea(), draft)

    assert plan["models"] == [{"name": "Qwen-subject", "role": "subject"}]
    assert plan["pilot"]["development_examples"] == 16
    assert plan["uncertainty"]["rng_seed"] == 1729
    assert plan["uncertainty"]["interval"] == "percentile"
    assert plan["uncertainty"]["decision_role"] == "descriptive"
    assert plan["uncertainty"]["undefined_resample_policy"] == "drop"
    assert plan["uncertainty"]["max_undefined_fraction"] == 0.05
    assert plan["uncertainty"]["excess_undefined_decision"] == "reject"
    assert validate_plan(plan) == []


def test_compiler_manifest_and_design_gate_view_have_disjoint_ownership() -> None:
    plan = compile_screening_protocol(_idea(), _draft())

    assert plan["compiler"]["mechanical_fields"] == list(
        COMPILER_OWNED_DESIGN_FIELDS
    )
    view = design_gate_view(plan)
    assert set(COMPILER_OWNED_DESIGN_FIELDS).isdisjoint(view)
    assert view["arms"] == plan["arms"]
    assert view["estimand"] == plan["estimand"]
    assert view["hypothesis"] == plan["hypothesis"]
    assert view["primary_metric"] == plan["primary_metric"]
    assert view["resources"]["subject_model"] == (
        "Qwen2.5-Coder-1.5B-Instruct"
    )
    assert view["resources"]["datasets"] == [
        "HumanEval screening partition"
    ]
    assert view["confirmatory_claim"] == (
        plan["confirmatory_followup"]["claim"]
    )
    assert "threshold" not in view["gate_statistic"]
    assert "call_ledger" not in view
    assert "uncertainty" not in view
    assert "required_runtime_evidence" not in view


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


def test_compiler_normalizes_common_ledger_scope_aliases() -> None:
    draft = _draft()
    draft["call_ledger"] = {
        "components": [
            {
                "name": "baseline_reference",
                "scope": "per_task",
                "dataset_role": "screening",
                "calls_per_unit": 1,
            }
        ]
    }

    plan = compile_screening_protocol(_idea(), draft)

    component = plan["call_ledger"]["components"][0]
    assert component["scope"] == "per_example_seed"
    assert component["total_calls"] == plan["pilot"]["max_examples"]
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
        "evidence_valid": True,
        "gate_statistic_defined": True,
        "criterion_results": {
            "completed_tasks": {"value": 32, "passed": True},
            "primary_effect": {"value": 0.20, "passed": True},
            "positive_ci": {"value": 0.01, "passed": True},
        },
        "gate_decision": "promote",
        "metrics": {
            "paired_pass_rate_difference": 0.20,
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
        "evidence_valid": True,
        "gate_statistic_defined": True,
        "criterion_results": {
            "completed_tasks": {"value": 32, "passed": True},
            "primary_effect": {"value": 0.20, "passed": True},
            "positive_ci": {"value": 0.01, "passed": True},
        },
        "gate_decision": "promote",
        "metrics": {
            "paired_pass_rate_difference": 0.20,
        },
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


def test_pilot_examples_processed_is_endpoint_count_not_all_roles() -> None:
    plan = compile_screening_protocol(_idea(), _draft())
    runtime = {
        "gpu_count": 1,
        "examples_processed": 48,
        "examples_by_role": {
            "development": 16,
            "screening": 32,
        },
        "seeds": [0],
        "call_counts": {
            "candidate_generation": 32,
            "verifier_scoring": 128,
        },
        "evidence_valid": True,
        "gate_statistic_defined": True,
        "criterion_results": {
            "completed_tasks": {"value": 32, "passed": True},
            "primary_effect": {"value": 0.20, "passed": True},
            "positive_ci": {"value": 0.01, "passed": True},
        },
        "gate_decision": "promote",
        "metrics": {
            "paired_pass_rate_difference": 0.20,
        },
    }

    errors = validate_runtime_against_contract(
        plan=plan,
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="pilot",
    )

    assert "examples_processed must equal examples_by_role[screening]" in errors
