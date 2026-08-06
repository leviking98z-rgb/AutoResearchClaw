from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.autoresearch_v2.validation import (
    validate_build_output,
    validate_experiment_artifacts,
    validate_plan,
    validate_research_implementation,
    validate_runtime_against_contract,
)


def _screening_plan() -> dict[str, object]:
    return {
        "research_question": "Does treatment improve held-out accuracy?",
        "hypothesis": "Treatment improves accuracy by at least two points.",
        "primary_metric": "accuracy_delta",
        "metric_direction": "maximize",
        "study_phase": "screening_pilot",
        "pilot_objective": "Screen for an effect large enough to confirm.",
        "pilot_claim_scope": "Feasibility evidence only; no final claim.",
        "unit_of_analysis": "example-seed evaluation",
        "datasets": [
            {
                "name": "development-set",
                "split_role": "dev",
                "split_id": "development-v1",
            },
            {
                "name": "screening-set",
                "split_role": "screening",
                "split_id": "screening-v1",
                "used_for_adaptation": False,
            },
            {
                "name": "heldout-set",
                "split_role": "heldout_confirmatory",
                "split_id": "confirmatory-v1",
                "used_for_adaptation": False,
            },
        ],
        "models": [{"name": "model-a", "role": "subject"}],
        "baselines": ["no-self-improvement"],
        "ablations": ["remove treatment"],
        "arms": [
            {"name": "control", "role": "baseline"},
            {"name": "treatment", "role": "candidate"},
        ],
        "pilot": {
            "max_gpus": 1,
            "max_examples": 50,
            "max_seeds": 2,
            "timeout_sec": 7200,
        },
        "promotion_rule": "valid delta >= 0.02",
        "early_stop_rule": "invalid evidence or delta < 0.02",
        "estimand": "mean paired accuracy delta across example-seed units",
        "sample_size_rationale": "100 units per arm resolve 0.01.",
        "sample_accounting": {
            "arms": 2,
            "examples_per_arm": 50,
            "seeds": 2,
            "calls_per_example": 1,
            "total_model_calls": 200,
        },
        "effect_threshold": {"value": 0.02, "scale": "proportion"},
        "workload_budget": {
            "conditions": 2,
            "models": 1,
            "examples": 50,
            "seeds": 2,
            "max_new_tokens": 512,
            "estimated_model_calls": 200,
        },
        "decision_table": [
            {
                "condition": {"evidence_valid": False},
                "decision": "retry",
            },
            {
                "condition": {
                    "evidence_valid": True,
                    "effect_threshold_met": False,
                },
                "decision": "reject",
            },
            {
                "condition": {
                    "evidence_valid": True,
                    "effect_threshold_met": True,
                },
                "decision": "promote",
            },
        ],
        "confirmatory_followup": {
            "required": True,
            "changes": [
                "more examples",
                "more independent seeds",
                "untouched heldout split",
            ],
            "claim": "the stronger claim that only Scale may support",
            "examples": 100,
            "independent_seeds": [11, 22, 33],
            "split_id": "confirmatory-v1",
            "untouched": True,
        },
        "adaptation": {"datasets": ["development-set"]},
        "required_runtime_evidence": [
            "model_loaded",
            "datasets_loaded",
            "examples_processed",
            "gpu_count",
            "gate_decision",
            "metrics",
        ],
    }


def _legacy_plan() -> dict[str, object]:
    value = _screening_plan()
    for field in (
        "study_phase",
        "pilot_objective",
        "pilot_claim_scope",
        "unit_of_analysis",
        "arms",
        "sample_accounting",
        "effect_threshold",
        "confirmatory_followup",
        "adaptation",
    ):
        value.pop(field)
    value["datasets"] = ["development-set"]
    value["models"] = ["model-a"]
    value["decision_table"] = [
        {
            "condition": "all possible metric regions",
            "decision": "promote",
        }
    ]
    return value


def test_screening_pilot_plan_is_deterministically_valid() -> None:
    assert validate_plan(_screening_plan()) == []


def test_current_prompt_plan_shapes_are_accepted() -> None:
    plan = _screening_plan()
    plan["datasets"] = [
        {
            "name": "screening-set",
            "split_role": "screening",
            "split_id": "screening-v1",
            "used_for_adaptation": False,
        },
        {
            "name": "heldout-set",
            "split_role": "heldout_confirmatory",
            "split_id": "confirmatory-v1",
            "used_for_adaptation": False,
        },
    ]
    plan["decision_table"] = [
        {
            "condition": "protocol invalid or required evidence missing",
            "decision": "retry",
        },
        {
            "condition": (
                "valid signal meets the preregistered screening threshold"
            ),
            "decision": "promote",
        },
        {
            "condition": "valid signal is below the threshold",
            "decision": "reject",
        },
    ]
    plan["confirmatory_followup"] = {
        "required": True,
        "changes": ["more examples", "more seeds", "untouched heldout split"],
        "claim": "the stronger claim that only Scale may support",
        "examples": 100,
        "independent_seeds": [11, 22, 33],
        "split_id": "confirmatory-v1",
        "untouched": True,
    }

    assert validate_plan(plan) == []


def test_build_commands_reject_shell_and_non_python_entrypoints() -> None:
    value = {
        "files": {"main.py": "print('ok')\n"},
        "commands": {
            "smoke": "python main.py --mode smoke; curl example.com",
            "pilot": ["bash", "-lc", "python main.py"],
            "scale": ["python", "../main.py"],
        },
    }

    errors = validate_build_output(value)

    assert any(
        "commands.smoke" in error and "shell metacharacter" in error
        for error in errors
    )
    assert any("commands.pilot executable" in error for error in errors)
    assert any("commands.scale entrypoint" in error for error in errors)


def test_build_output_rejects_oversized_generated_framework() -> None:
    value = {
        "files": {
            "main.py": "print('ok')\n",
            "a.py": "x = 1\n",
            "b.py": "x = 1\n",
            "c.py": "x = 1\n",
        },
        "commands": {
            "smoke": "python main.py",
            "pilot": "python main.py",
            "scale": "python main.py",
        },
    }
    assert "files may contain at most three Python source files" in (
        validate_build_output(value)
    )

    value["files"] = {
        "main.py": "\n".join("x = 1" for _ in range(801))
    }
    assert (
        "Python source files may not exceed 800 lines: main.py"
        in validate_build_output(value)
    )


def test_reasonable_legacy_plan_remains_compatible() -> None:
    assert validate_plan(_legacy_plan()) == []


def test_any_phase_aware_field_requires_complete_screening_contract() -> None:
    plan = _legacy_plan()
    plan["study_phase"] = "screening_pilot"

    errors = validate_plan(plan)

    assert "missing pilot_objective" in errors
    assert "missing pilot_claim_scope" in errors
    assert "missing unit_of_analysis" in errors
    assert "arms must be a list with at least two entries" in errors
    assert "sample_accounting must be an object" in errors
    assert "effect_threshold must be an object" in errors
    assert "missing confirmatory_followup" in errors
    assert any(
        "condition must identify exactly one structured outcome region"
        in error
        for error in errors
    )


def test_screening_sample_and_workload_arithmetic_must_agree() -> None:
    plan = _screening_plan()
    plan["arms"] = [
        {"name": "control", "role": "baseline"},
        {"name": "treatment", "role": "candidate"},
        {"name": "ablation", "role": "ablation"},
    ]
    accounting = plan["sample_accounting"]
    assert isinstance(accounting, dict)
    accounting["total_model_calls"] = 199
    workload = plan["workload_budget"]
    assert isinstance(workload, dict)
    workload["estimated_model_calls"] = 201
    workload["models"] = 2

    errors = validate_plan(plan)

    assert any("does not match len(arms)=3" in error for error in errors)
    assert any(
        "total_model_calls=199 does not equal" in error for error in errors
    )
    assert any(
        "estimated_model_calls=201 does not equal" in error
        for error in errors
    )
    assert any(
        "calls_per_example=1 does not match workload_budget.models=2"
        in error
        for error in errors
    )


def test_screening_sample_accounting_cannot_exceed_pilot_caps() -> None:
    plan = _screening_plan()
    accounting = plan["sample_accounting"]
    assert isinstance(accounting, dict)
    accounting["examples_per_arm"] = 51
    accounting["seeds"] = 3
    accounting["total_model_calls"] = 306
    workload = plan["workload_budget"]
    assert isinstance(workload, dict)
    workload["examples"] = 51
    workload["seeds"] = 3
    workload["estimated_model_calls"] = 306

    errors = validate_plan(plan)

    assert (
        "sample_accounting.examples_per_arm=51 exceeds "
        "pilot.max_examples=50"
    ) in errors
    assert (
        "sample_accounting.seeds=3 exceeds pilot.max_seeds=2"
    ) in errors


@pytest.mark.parametrize(
    ("scale", "threshold", "fragment"),
    [
        ("proportion", 0.005, "pilot sample resolution=0.01"),
        ("percentage_points", 0.5, "pilot sample resolution=1"),
        ("absolute", 0.005, "pilot sample resolution=0.01"),
    ],
)
def test_effect_threshold_must_be_resolvable_by_pilot_sample(
    scale: str,
    threshold: float,
    fragment: str,
) -> None:
    plan = _screening_plan()
    plan["effect_threshold"] = {"value": threshold, "scale": scale}

    errors = validate_plan(plan)

    assert any(fragment in error for error in errors)


@pytest.mark.parametrize(
    "effect_threshold",
    [
        {"value": 0, "scale": "proportion"},
        {"value": 1.1, "scale": "proportion"},
        {"value": 101, "scale": "percentage_points"},
        {"value": 2, "scale": "unknown"},
        {"value": "0.02", "scale": "proportion"},
    ],
)
def test_effect_threshold_shape_and_scale_are_strict(
    effect_threshold: dict[str, object],
) -> None:
    plan = _screening_plan()
    plan["effect_threshold"] = effect_threshold

    assert any(
        error.startswith("invalid effect_threshold")
        for error in validate_plan(plan)
    )


def test_decision_table_rejects_duplicate_and_missing_regions() -> None:
    plan = _screening_plan()
    table = plan["decision_table"]
    assert isinstance(table, list)
    table[2] = {
        "condition": {
            "evidence_valid": True,
            "effect_threshold_met": False,
        },
        "decision": "promote",
    }

    errors = validate_plan(plan)

    assert any("duplicate decision_table condition" in error for error in errors)
    assert any(
        "duplicate decision_table outcome region "
        "'below_effect_threshold'" in error
        for error in errors
    )
    assert any(
        "missing outcome regions: at_or_above_effect_threshold" in error
        for error in errors
    )


def test_decision_table_numeric_intervals_must_partition_outcomes() -> None:
    plan = _screening_plan()
    plan["decision_table"] = [
        {
            "condition": {"evidence_valid": False},
            "decision": "retry",
        },
        {
            "condition": {
                "operator": "<",
                "value": 0.02,
            },
            "decision": "reject",
        },
        {
            "condition": {
                "operator": ">=",
                "value": 0.02,
            },
            "decision": "promote",
        },
    ]
    assert validate_plan(plan) == []

    table = plan["decision_table"]
    assert isinstance(table, list)
    table[2] = {
        "condition": {"operator": ">", "value": 0.02},
        "decision": "promote",
    }
    errors = validate_plan(plan)
    assert any("uncovered boundary" in error for error in errors)

    table[1] = {
        "condition": {"operator": "<=", "value": 0.02},
        "decision": "reject",
    }
    table[2] = {
        "condition": {"operator": ">=", "value": 0.02},
        "decision": "promote",
    }
    errors = validate_plan(plan)
    assert any("overlap at their boundary" in error for error in errors)


def test_decision_table_must_respect_metric_direction() -> None:
    plan = _screening_plan()
    plan["metric_direction"] = "minimize"

    errors = validate_plan(plan)

    assert any(
        "region 'below_effect_threshold' must use 'promote'" in error
        for error in errors
    )
    assert any(
        "region 'at_or_above_effect_threshold' must use 'reject'" in error
        for error in errors
    )

    table = plan["decision_table"]
    assert isinstance(table, list)
    table[1]["decision"] = "promote"
    table[2]["decision"] = "reject"
    assert validate_plan(plan) == []


def test_typed_decision_contract_uses_gate_not_raw_metric_direction() -> None:
    from researchclaw.autoresearch_v2.ideas import candidate_to_idea
    from researchclaw.autoresearch_v2.protocols import (
        compile_screening_protocol,
    )

    idea = candidate_to_idea(
        {
            "id": "gate-direction",
            "title": "Regret reduction gate",
            "family": "calibration",
            "research_question": "Does the gate reduce regret?",
            "falsifiable_hypothesis": "Relative regret falls by 20%.",
            "closest_prior_work": ["Prior"],
            "novelty_gap": "Gap",
            "datasets": ["MBPP"],
            "models": ["Qwen"],
            "compute": {"gpu_count": 1, "wall_clock_hours": 1},
            "primary_metric": "regret",
            "baselines": ["no-self-improvement"],
            "ablations": ["remove gate"],
            "failure_safety_tests": ["heldout isolation"],
            "implementation_feasibility": "public",
            "licensing_feasibility": "public",
            "information_gain_if_true": "useful",
            "information_gain_if_false": "useful",
            "cheap_pilot": "32 tasks",
            "scores": {
                "novelty": 8,
                "scientific_importance": 8,
                "falsifiability": 8,
                "compute_tractability": 8,
                "reproducibility": 8,
                "meaningful_result_likelihood": 8,
                "risk": 2,
            },
        }
    )
    draft = {
        "protocol_template": "calibration_verifier",
        "pilot_objective": "screen the gate",
        "pilot_claim_scope": "coarse signal only",
        "research_question": "Does the gate reduce regret?",
        "hypothesis": "Relative regret reduction is at least 20%.",
        "primary_metric": "mean regret",
        "metric_direction": "minimize",
        "unit_of_analysis": "paired task",
        "dataset": "MBPP",
        "screening_access_policy": {
            "input_access": True,
            "within_episode_feedback": False,
            "cross_example_adaptation": False,
            "hidden_labels_for_tuning": False,
            "threshold_tuning": False,
        },
        "models": [{"name": "Qwen", "role": "subject"}],
        "baselines": ["no-self-improvement"],
        "ablations": ["remove gate"],
        "arms": [
            {"name": "calibrated gate", "role": "treatment"},
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
                    "name": "final_evaluation",
                    "scope": "per_arm_example_seed",
                    "dataset_role": "screening",
                    "calls_per_unit": 1,
                }
            ]
        },
        "gate_statistic": {
            "name": "relative_regret_reduction",
            "definition": "(control regret - treatment regret) / control regret",
            "direction": "maximize",
            "threshold": {"value": 0.20, "scale": "proportion"},
            "undefined_policy": "reject",
        },
        "uncertainty": {
            "method": "paired_cluster_bootstrap",
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
                "description": "operational completeness",
            }
        ],
        "promotion_criteria": [
            {
                "id": "relative_reduction",
                "metric": "relative_regret_reduction",
                "operator": ">=",
                "value": 0.20,
                "scale": "proportion",
                "description": "primary effect",
            }
        ],
        "estimand": "paired relative regret reduction",
        "sample_size_rationale": "screening resolution",
        "workload_budget": {"max_new_tokens": 64},
        "confirmatory_followup": {"claim": "Scale confirms the effect."},
    }

    plan = compile_screening_protocol(idea, draft)

    assert plan["metric_direction"] == "minimize"
    assert plan["gate_statistic"]["direction"] == "maximize"
    assert validate_plan(plan) == []
    plan["promotion_criteria"][0]["operator"] = "<="
    errors = validate_plan(plan)
    assert any("operator conflicts" in error for error in errors)


def test_typed_runtime_valid_undefined_gate_must_reject_not_retry() -> None:
    from researchclaw.autoresearch_v2.ideas import candidate_to_idea
    from researchclaw.autoresearch_v2.protocols import (
        compile_screening_protocol,
    )

    # Reuse the fully typed fixture from the compiler tests without coupling
    # test modules through an import.
    candidate = {
        "id": "runtime-contract",
        "title": "Runtime contract",
        "family": "verifier",
        "research_question": "Does it help?",
        "falsifiable_hypothesis": "It helps.",
        "closest_prior_work": ["Prior"],
        "novelty_gap": "Gap",
        "datasets": ["HumanEval"],
        "models": ["Qwen"],
        "compute": {"gpu_count": 1, "wall_clock_hours": 1},
        "primary_metric": "paired difference",
        "baselines": ["no-self-improvement"],
        "ablations": ["remove mechanism"],
        "failure_safety_tests": ["heldout isolation"],
        "implementation_feasibility": "public",
        "licensing_feasibility": "public",
        "information_gain_if_true": "useful",
        "information_gain_if_false": "useful",
        "cheap_pilot": "32 tasks",
        "scores": {
            "novelty": 8,
            "scientific_importance": 8,
            "falsifiability": 8,
            "compute_tractability": 8,
            "reproducibility": 8,
            "meaningful_result_likelihood": 8,
            "risk": 2,
        },
    }
    idea = candidate_to_idea(candidate)
    draft = {
        "protocol_template": "calibration_verifier",
        "pilot_objective": "screen",
        "pilot_claim_scope": "coarse",
        "research_question": "Does it help?",
        "hypothesis": "It helps.",
        "primary_metric": "paired difference",
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
        "models": [{"name": "Qwen", "role": "subject"}],
        "baselines": ["no-self-improvement"],
        "ablations": ["remove mechanism"],
        "arms": [
            {"name": "treatment", "role": "treatment"},
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
                    "name": "final_evaluation",
                    "scope": "per_arm_example_seed",
                    "dataset_role": "screening",
                    "calls_per_unit": 1,
                }
            ]
        },
        "gate_statistic": {
            "name": "paired_difference",
            "definition": "treatment minus control",
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
                "description": "operational completeness",
            }
        ],
        "promotion_criteria": [
            {
                "id": "primary_effect",
                "metric": "paired_difference",
                "operator": ">=",
                "value": 0.15,
                "scale": "proportion",
                "description": "primary effect",
            }
        ],
        "estimand": "paired mean difference",
        "sample_size_rationale": "screening resolution",
        "workload_budget": {"max_new_tokens": 64},
        "confirmatory_followup": {"claim": "Scale confirms the effect."},
    }
    plan = compile_screening_protocol(idea, draft)
    runtime = {
        "gpu_count": 1,
        "examples_processed": 32,
        "examples_by_role": {"development": 0, "screening": 32},
        "seeds": [0],
        "call_counts": {"final_evaluation": 64},
        "evidence_valid": True,
        "gate_statistic_defined": False,
        "criterion_results": {
            "completed_tasks": {"value": 32, "passed": True},
            "primary_effect": {"value": 0.0, "passed": False},
        },
        "gate_decision": "reject",
        "metrics": {"paired_difference": 0.0},
    }

    assert validate_runtime_against_contract(
        plan=plan,
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="pilot",
    ) == []
    runtime["gate_decision"] = "retry"
    errors = validate_runtime_against_contract(
        plan=plan,
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="pilot",
    )
    assert any("expected 'reject'" in error for error in errors)


def test_heldout_dataset_must_not_participate_in_adaptation() -> None:
    plan = _screening_plan()
    plan["adaptation"] = {
        "datasets": ["development-set", "heldout-set"],
    }

    errors = validate_plan(plan)

    assert any(
        "heldout data must not participate in adaptation" in error
        and "heldout-set" in error
        for error in errors
    )


def test_heldout_dataset_local_adaptation_flag_is_rejected() -> None:
    plan = _screening_plan()
    datasets = plan["datasets"]
    assert isinstance(datasets, list)
    heldout = datasets[2]
    assert isinstance(heldout, dict)
    heldout["used_for_adaptation"] = True

    errors = validate_plan(plan)

    assert any(
        "heldout dataset 'heldout-set' participates in adaptation" in error
        for error in errors
    )


@pytest.mark.parametrize(
    "adaptation_key",
    [
        "calibration_datasets",
        "memory_write_datasets",
        "prompt_selection_datasets",
    ],
)
def test_heldout_is_isolated_from_all_adaptive_paths(
    adaptation_key: str,
) -> None:
    plan = _screening_plan()
    plan[adaptation_key] = ["heldout-set"]

    assert any(
        "heldout data must not participate in adaptation" in error
        for error in validate_plan(plan)
    )


def test_confirmatory_followup_object_is_structurally_checked() -> None:
    plan = _screening_plan()
    plan["confirmatory_followup"] = {
        "required": False,
        "changes": [],
        "claim": "",
        "examples": 50,
        "independent_seeds": [11, 11],
        "split_id": "screening-v1",
        "untouched": False,
    }

    errors = validate_plan(plan)

    assert "confirmatory_followup.required must be true" in errors
    assert any(
        error.startswith("confirmatory_followup.changes") for error in errors
    )
    assert any(
        error.startswith("confirmatory_followup.claim") for error in errors
    )
    assert any(
        "examples must exceed pilot examples" in error for error in errors
    )
    assert (
        "confirmatory_followup.independent_seeds must be unique" in errors
    )
    assert "confirmatory_followup.untouched must be true" in errors
    assert any(
        "split_id must differ from the pilot screening split" in error
        for error in errors
    )


def test_production_screening_followup_fails_closed() -> None:
    plan = _screening_plan()
    plan["confirmatory_followup"] = (
        "Use more examples and seeds on untouched data."
    )

    assert (
        "confirmatory_followup must be an object for screening_pilot"
        in validate_plan(plan)
    )

    plan.pop("confirmatory_followup")
    assert "missing confirmatory_followup" in validate_plan(plan)


def test_runtime_evidence_accepts_dataset_roles_and_split_identifiers(
    tmp_path: Path,
) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "result_valid": True,
                "metrics": {"accuracy": 0.8},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime_evidence.json").write_text(
        json.dumps(
            {
                "model_loaded": "Qwen",
                "datasets_loaded": ["heldout-set"],
                "examples_processed": 100,
                "seeds": [11, 22, 33],
                "gpu_count": 1,
                "gate_decision": "promote",
                "metrics": {"accuracy": 0.8},
                "dataset_roles": {
                    "heldout-set": {
                        "role": "confirmatory",
                        "split_id": "confirmatory-v1",
                        "untouched": True,
                    }
                },
                "split_identifiers": {
                    "confirmatory": "confirmatory-v1",
                },
            }
        ),
        encoding="utf-8",
    )

    assert validate_experiment_artifacts(tmp_path)["ok"]


def test_runtime_evidence_accepts_role_map_with_top_level_split_ids(
    tmp_path: Path,
) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "result_valid": True,
                "metrics": {"accuracy": 0.8},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime_evidence.json").write_text(
        json.dumps(
            {
                "model_loaded": "Qwen",
                "datasets_loaded": ["heldout-set"],
                "examples_processed": 100,
                "seeds": [11, 22, 33],
                "gpu_count": 1,
                "gate_decision": "promote",
                "metrics": {"accuracy": 0.8},
                "dataset_roles": {
                    "heldout-set": {
                        "role": "confirmatory",
                        "untouched": True,
                    }
                },
                "confirmatory_split_id": "confirmatory-v1",
            }
        ),
        encoding="utf-8",
    )

    assert validate_experiment_artifacts(tmp_path)["ok"]


def test_artifact_metrics_must_be_numeric_and_consistent(
    tmp_path: Path,
) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "result_valid": True,
                "metrics": {"accuracy": "0.9"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime_evidence.json").write_text(
        json.dumps(
            {
                "model_loaded": "Qwen",
                "datasets_loaded": ["GSM8K"],
                "examples_processed": 10,
                "seeds": [0],
                "gpu_count": 1,
                "gate_decision": "promote",
                "metrics": {"accuracy": 0.8},
            }
        ),
        encoding="utf-8",
    )
    value = validate_experiment_artifacts(tmp_path)
    assert not value["ok"]
    assert any("numeric" in error for error in value["errors"])
    assert any("disagree" in error for error in value["errors"])


def test_research_implementation_requires_real_loaders(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text(
        "from pathlib import Path\n"
        "Path('metrics.json').write_text('{}')\n"
        "Path('runtime_evidence.json').write_text('{}')\n",
        encoding="utf-8",
    )
    value = validate_research_implementation(
        tmp_path,
        plan={"models": ["Qwen"], "datasets": ["GSM8K"]},
    )
    assert not value["ok"]
    assert "no real model loader call found" in value["errors"]
    assert "no real dataset/benchmark loader call found" in value["errors"]


def test_research_implementation_rejects_known_runtime_schema_mismatches(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text(
        """
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM

AutoModelForCausalLM.from_pretrained("Qwen")
load_dataset("gsm8k")
criterion_results = {
    "minimum_completed_examples": {"value": 1, "pass": True},
    "primary_effect": {"value": None, "pass": False},
}
call_counts = {"mutation_calls": 1}
examples_by_role = {"development_diagnostic": 1, "screening": 1}
runtime_evidence = {
    "model_loaded": True,
    "datasets_loaded": ["gsm8k"],
    "examples_processed": 2,
    "examples_by_role": examples_by_role,
    "gpu_count": 1,
    "seeds": [0],
    "gate_decision": "reject",
    "metrics": {"gain": 0.0},
    "call_counts": call_counts,
    "criterion_results": criterion_results,
}
metrics = {"result_valid": True, "gain": 0.0}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)
with open("runtime_evidence.json", "w") as f:
    json.dump(runtime_evidence, f)
""",
        encoding="utf-8",
    )
    plan = {
        "models": ["Qwen"],
        "datasets": ["GSM8K"],
        "required_runtime_evidence": [
            "model_loaded",
            "datasets_loaded",
            "examples_processed",
            "examples_by_role",
            "gpu_count",
            "seeds",
            "gate_decision",
            "metrics",
            "call_counts",
            "criterion_results",
        ],
        "validity_criteria": [{"id": "minimum_completed_examples"}],
        "promotion_criteria": [{"id": "primary_effect"}],
        "call_ledger": {
            "components": [
                {"name": "candidate_generation"},
                {"name": "final_evaluation"},
            ]
        },
    }
    value = validate_research_implementation(tmp_path, plan=plan)
    assert not value["ok"]
    assert any("result_valid and metrics" in error for error in value["errors"])
    assert any("model_loaded" in error for error in value["errors"])
    assert any("passed, not pass" in error for error in value["errors"])
    assert any("finite numbers" in error for error in value["errors"])


def _production_runtime_plan() -> dict[str, object]:
    return _screening_plan()


def _pilot_runtime() -> dict[str, object]:
    return {
        "examples_processed": 50,
        "seeds": [0, 1],
        "dataset_roles": {
            "screening-set": {
                "role": "screening",
                "split_id": "screening-v1",
                "untouched": False,
            }
        },
        "split_identifiers": {"screening": "screening-v1"},
    }


def _scale_runtime() -> dict[str, object]:
    return {
        "gpu_count": 1,
        "examples_processed": 100,
        "seeds": [11, 22, 33],
        "dataset_roles": {
            "heldout-set": {
                "role": "confirmatory",
                "split_id": "confirmatory-v1",
                "untouched": True,
            }
        },
        "split_identifiers": {"confirmatory": "confirmatory-v1"},
    }


@pytest.mark.parametrize(
    ("examples", "seeds", "expected_fragment"),
    [
        (100, [11, 22], "increase independent seed coverage"),
        (50, [11, 22, 33], "increase examples beyond pilot"),
    ],
)
def test_scale_must_expand_examples_and_independent_seed_coverage(
    examples: int,
    seeds: list[int],
    expected_fragment: str,
) -> None:
    runtime = _scale_runtime()
    runtime["examples_processed"] = examples
    runtime["seeds"] = seeds

    errors = validate_runtime_against_contract(
        plan=_production_runtime_plan(),
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="scale",
        pilot_runtime=_pilot_runtime(),
    )

    assert any(expected_fragment in error for error in errors)


def test_scale_accepts_strict_expansion_on_untouched_confirmatory_split() -> None:
    assert (
        validate_runtime_against_contract(
            plan=_production_runtime_plan(),
            runtime_evidence=_scale_runtime(),
            allocated_gpus=1,
            mode="scale",
            pilot_runtime=_pilot_runtime(),
        )
        == []
    )


def test_scale_rejects_reused_screening_split() -> None:
    runtime = _scale_runtime()
    dataset_roles = runtime["dataset_roles"]
    assert isinstance(dataset_roles, dict)
    heldout = dataset_roles["heldout-set"]
    assert isinstance(heldout, dict)
    heldout["split_id"] = "screening-v1"
    runtime["split_identifiers"] = {"confirmatory": "screening-v1"}

    errors = validate_runtime_against_contract(
        plan=_production_runtime_plan(),
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="scale",
        pilot_runtime=_pilot_runtime(),
    )

    assert any(
        "confirmatory split must differ from pilot screening split" in error
        for error in errors
    )


def test_scale_accepts_role_keyed_dataset_roles_with_split_identifiers() -> None:
    runtime = _scale_runtime()
    runtime["dataset_roles"] = {
        "confirmatory": {
            "role": "confirmatory",
            "untouched": True,
        },
    }
    runtime["split_identifiers"] = {
        "confirmatory": "confirmatory-v1",
    }

    assert (
        validate_runtime_against_contract(
            plan=_production_runtime_plan(),
            runtime_evidence=runtime,
            allocated_gpus=1,
            mode="scale",
            pilot_runtime=_pilot_runtime(),
        )
        == []
    )


def test_production_scale_fails_closed_without_runtime_split_contract() -> None:
    runtime = _scale_runtime()
    runtime.pop("dataset_roles")
    runtime.pop("split_identifiers")

    errors = validate_runtime_against_contract(
        plan=_production_runtime_plan(),
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="scale",
        pilot_runtime=_pilot_runtime(),
    )

    assert any(
        "must declare dataset_roles and split identifiers" in error
        for error in errors
    )


def test_scale_runtime_split_must_match_plan_and_be_untouched() -> None:
    runtime = _scale_runtime()
    dataset_roles = runtime["dataset_roles"]
    assert isinstance(dataset_roles, dict)
    heldout = dataset_roles["heldout-set"]
    assert isinstance(heldout, dict)
    heldout["split_id"] = "confirmatory-v2"
    heldout["untouched"] = False
    runtime["split_identifiers"] = {"confirmatory": "confirmatory-v2"}

    errors = validate_runtime_against_contract(
        plan=_production_runtime_plan(),
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="scale",
        pilot_runtime=_pilot_runtime(),
    )

    assert "scale runtime_evidence must mark confirmatory data untouched" in errors
    assert any(
        "does not match plan.confirmatory_followup.split_id" in error
        for error in errors
    )


def test_scale_runtime_split_declarations_must_agree() -> None:
    runtime = _scale_runtime()
    runtime["split_identifiers"] = {"confirmatory": "confirmatory-v2"}

    errors = validate_runtime_against_contract(
        plan=_production_runtime_plan(),
        runtime_evidence=runtime,
        allocated_gpus=1,
        mode="scale",
        pilot_runtime=_pilot_runtime(),
    )

    assert "scale runtime confirmatory split declarations disagree" in errors


def test_legacy_scale_keeps_optional_dataset_metadata_compatibility() -> None:
    assert (
        validate_runtime_against_contract(
            plan={},
            runtime_evidence={
                "gpu_count": 1,
                "examples_processed": 20,
                "seeds": [0, 1],
            },
            allocated_gpus=1,
            mode="scale",
            pilot_runtime={
                "examples_processed": 10,
                "seeds": [0],
            },
        )
        == []
    )
