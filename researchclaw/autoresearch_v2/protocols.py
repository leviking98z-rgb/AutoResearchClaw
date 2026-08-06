"""Typed screening-protocol compiler.

The model chooses scientific variables.  This module owns the mechanical
contract: dataset roles, immutable split identifiers, exact call arithmetic,
pilot/scale budgets, and exhaustive decision regions.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from .models import IdeaRecord

SUPPORTED_PROTOCOLS = frozenset(
    {
        "calibration_verifier",
        "stopping_policy",
        "memory_policy",
        "rollback_gate",
        "population_search",
    }
)

_PROTOCOL_ALIASES = {
    "calibration": "calibration_verifier",
    "verifier": "calibration_verifier",
    "verifier_reliability": "calibration_verifier",
    "stopping": "stopping_policy",
    "early_stopping": "stopping_policy",
    "memory": "memory_policy",
    "rollback": "rollback_gate",
    "population": "population_search",
    "population_diversity": "population_search",
}

_LEDGER_COMPONENTS = (
    "adaptation",
    "candidate_generation",
    "verifier_scoring",
    "calibration",
    "memory_writing",
    "shadow_continuation",
    "baseline_reference",
    "final_evaluation",
)
_LEDGER_SCOPES = frozenset(
    {
        "per_arm_example_seed",
        "per_example_seed",
        "per_arm_seed",
        "per_seed",
        "fixed",
    }
)
_LEDGER_SCOPE_ALIASES = {
    "per_task": "per_example_seed",
    "per_example": "per_example_seed",
    "per_arm_example": "per_arm_example_seed",
    "per_arm": "per_arm_seed",
    "once_per_seed": "per_seed",
    "once": "fixed",
    "global": "fixed",
}
_LEDGER_DATASET_ROLES = frozenset(
    {"development", "screening", "none"}
)
_CRITERION_OPERATORS = frozenset({"<", "<=", ">", ">=", "=="})
_THRESHOLD_SCALES = frozenset(
    {"proportion", "percentage_points", "absolute"}
)

_PROTOCOL_MARKERS = {
    "calibration_verifier": (
        "calibrat",
        "confidence",
        "verifier",
        "uncertainty",
        "feedback",
    ),
    "stopping_policy": (
        "stopping",
        "stop",
        "futility",
        "continue",
        "marginal",
    ),
    "memory_policy": (
        "memory",
        "retrieval",
        "reflection",
        "lesson",
        "archive",
    ),
    "rollback_gate": (
        "rollback",
        "revert",
        "checkpoint",
        "regression gate",
        "transaction",
    ),
    "population_search": (
        "population",
        "diversity",
        "mutation",
        "crossover",
        "lineage",
        "evolution",
    ),
}


def infer_protocol_template(idea: IdeaRecord) -> str:
    """Choose one supported protocol family from the admitted Idea."""

    text = (
        f"{idea.family} {idea.title} {idea.research_question} "
        f"{idea.falsifiable_hypothesis}"
    ).casefold()
    scores = {
        protocol: sum(text.count(marker) for marker in markers)
        for protocol, markers in _PROTOCOL_MARKERS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ""


def compile_screening_protocol(
    idea: IdeaRecord,
    draft: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile one model draft into an executable screening contract.

    The compiler intentionally does not invent the scientific estimand,
    threshold, intervention, or outcome.  It only normalizes the selected
    supported template and derives mechanical fields from an explicit ledger.
    """

    plan = copy.deepcopy(dict(draft))
    protocol = _canonical_protocol(
        plan.get("protocol_template")
        or plan.get("experiment_template")
        or infer_protocol_template(idea)
    )
    if not protocol:
        return plan
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported protocol_template: {protocol}")

    plan["protocol_template"] = protocol
    plan["study_phase"] = "screening_pilot"
    requested_screening_access_policy = _compile_screening_access_policy(
        plan.get("screening_access_policy")
    )
    # Pilot endpoints are always evaluated on a frozen screening partition.
    # Protocol families may adapt on the separate development partition, but
    # screening observations can never feed state, prompts, thresholds, or
    # selection. This removes a recurring and outcome-changing leakage choice
    # from model-authored prose.
    screening_access_policy = {
        **requested_screening_access_policy,
        "within_episode_feedback": False,
        "cross_example_adaptation": False,
        "hidden_labels_for_tuning": False,
        "threshold_tuning": False,
    }
    plan["datasets"] = _compile_datasets(
        idea,
        plan.get("dataset"),
        plan.get("datasets"),
        screening_access_policy=screening_access_policy,
    )
    plan["screening_access_policy"] = screening_access_policy
    plan["models"] = _compile_models(idea, plan.get("models"))
    plan["arms"] = _compile_arms(plan.get("arms"))
    plan["baselines"] = _compile_string_list(
        plan.get("baselines"),
        fallback=["no-self-improvement control"],
    )
    plan["ablations"] = _compile_string_list(
        plan.get("ablations"),
        fallback=["remove the proposed mechanism"],
    )

    pilot = _compile_pilot(plan.get("pilot"))
    plan["pilot"] = pilot
    ledger = _compile_call_ledger(
        plan.get("call_ledger"),
        plan.get("sample_accounting"),
        arms=plan["arms"],
        pilot=pilot,
    )
    plan["call_ledger"] = ledger
    total_calls = int(ledger["total_model_calls"])
    if total_calls > 512:
        raise ValueError(
            "screening pilot call_ledger.total_model_calls must be at most 512"
        )
    plan["sample_accounting"] = {
        "arms": len(plan["arms"]),
        "development_examples": pilot["development_examples"],
        "examples_per_arm": pilot["max_examples"],
        "seeds": pilot["max_seeds"],
        "total_model_calls": total_calls,
    }
    workload = (
        plan.get("workload_budget")
        if isinstance(plan.get("workload_budget"), Mapping)
        else {}
    )
    max_new_tokens = _positive_int(
        workload.get("max_new_tokens"),
        default=512,
        field="workload_budget.max_new_tokens",
    )
    plan["workload_budget"] = {
        "conditions": len(plan["arms"]),
        "development_examples": pilot["development_examples"],
        "examples": pilot["max_examples"],
        "seeds": pilot["max_seeds"],
        "max_new_tokens": max_new_tokens,
        "estimated_model_calls": total_calls,
    }
    gate_statistic = _compile_gate_statistic(
        plan.get("gate_statistic"),
        plan.get("effect_threshold"),
    )
    plan["gate_statistic"] = gate_statistic
    # Compatibility mirror for existing consumers. The executable direction
    # lives in gate_statistic and the primary promotion criterion.
    plan["effect_threshold"] = copy.deepcopy(gate_statistic["threshold"])
    plan["uncertainty"] = _compile_uncertainty(plan.get("uncertainty"))
    plan["validity_criteria"] = _compile_criteria(
        plan.get("validity_criteria"),
        field="validity_criteria",
        minimum=1,
        maximum=6,
    )
    plan["promotion_criteria"] = _compile_criteria(
        plan.get("promotion_criteria"),
        field="promotion_criteria",
        minimum=1,
        maximum=6,
    )
    _require_primary_gate_criterion(
        gate_statistic,
        plan["promotion_criteria"],
    )
    plan["decision_contract"] = _compile_decision_contract(
        validity_criteria=plan["validity_criteria"],
        promotion_criteria=plan["promotion_criteria"],
    )
    plan["decision_table"] = _compile_decision_table(
        promotion_criteria=plan["promotion_criteria"],
    )
    # Free-form prose is not executable. Keep it synchronized with the typed
    # criteria instead of asking a reviewer to reconcile two rule systems.
    plan["promotion_rule"] = (
        "After every validity criterion passes, promote if and only if every "
        "preregistered promotion criterion passes; valid evidence that fails "
        "any promotion criterion is rejected."
    )
    plan["early_stop_rule"] = (
        "Retry only when a preregistered validity criterion fails or required "
        "runtime evidence is missing or corrupt. Unfavorable, undefined, "
        "low-variance, or inconclusive scientific results are valid rejects."
    )
    plan["confirmatory_followup"] = _compile_confirmatory_followup(
        plan.get("confirmatory_followup"),
        pilot=pilot,
        confirmatory_split_id=plan["datasets"][2]["split_id"],
    )
    plan["required_runtime_evidence"] = [
        "model_loaded",
        "datasets_loaded",
        "examples_processed",
        "examples_by_role",
        "gpu_count",
        "gate_decision",
        "metrics",
        "dataset_roles",
        "split_identifiers",
        "call_counts",
        "evidence_valid",
        "gate_statistic_defined",
        "criterion_results",
    ]
    plan["compiler"] = {
        "name": "autoresearch_v2_protocol_compiler",
        "version": 2,
        "mechanical_fields": [
            "datasets",
            "screening_access_policy",
            "models",
            "pilot",
            "call_ledger",
            "sample_accounting",
            "workload_budget",
            "effect_threshold",
            "uncertainty",
            "decision_contract",
            "decision_table",
            "promotion_rule",
            "early_stop_rule",
            "confirmatory_followup",
        ],
    }
    return plan


def _compile_decision_table(
    *,
    promotion_criteria: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compile invalid/promote/valid-reject as an exhaustive partition."""

    return [
        {
            "condition": {"region": "invalid"},
            "decision": "retry",
        },
        {
            "condition": {
                "region": "meets_all_promotion_criteria",
                "criteria": [
                    str(item["id"]) for item in promotion_criteria
                ],
            },
            "decision": "promote",
        },
        {
            "condition": {"region": "valid_otherwise"},
            "decision": "reject",
        },
    ]


def validate_protocol_draft(value: Mapping[str, Any]) -> list[str]:
    """Validate model-owned fields before deterministic compilation."""

    if not isinstance(value, Mapping):
        return ["protocol draft must be an object"]
    errors: list[str] = []
    protocol = _canonical_protocol(value.get("protocol_template"))
    if protocol not in SUPPORTED_PROTOCOLS:
        errors.append(
            "protocol_template must be one of: "
            + ", ".join(sorted(SUPPORTED_PROTOCOLS))
        )
    for field in (
        "pilot_objective",
        "pilot_claim_scope",
        "research_question",
        "hypothesis",
        "primary_metric",
        "unit_of_analysis",
        "estimand",
        "sample_size_rationale",
    ):
        if not isinstance(value.get(field), str) or not str(
            value.get(field, "")
        ).strip():
            errors.append(f"missing {field}")
    if value.get("metric_direction") not in {"maximize", "minimize"}:
        errors.append("metric_direction must be maximize or minimize")
    try:
        _compile_screening_access_policy(
            value.get("screening_access_policy")
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    dataset = value.get("dataset")
    datasets = value.get("datasets")
    if not (
        isinstance(dataset, str)
        and dataset.strip()
        or isinstance(datasets, list)
        and datasets
    ):
        errors.append("missing dataset")
    for field in ("models", "baselines", "ablations"):
        if not isinstance(value.get(field), list) or not value[field]:
            errors.append(f"missing {field}")
    try:
        arms = _compile_arms(value.get("arms"))
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        arms = []
    try:
        pilot = _compile_pilot(value.get("pilot"))
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        pilot = {
            "max_gpus": 1,
            "development_examples": 16,
            "max_examples": 32,
            "max_seeds": 1,
            "timeout_sec": 7200,
        }
    if arms:
        try:
            ledger = _compile_call_ledger(
                value.get("call_ledger"),
                value.get("sample_accounting"),
                arms=arms,
                pilot=pilot,
            )
            if int(ledger["total_model_calls"]) > 512:
                errors.append(
                    "screening pilot call_ledger.total_model_calls must be "
                    "at most 512"
                )
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    try:
        gate_statistic = _compile_gate_statistic(
            value.get("gate_statistic"),
            value.get("effect_threshold"),
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        gate_statistic = {}
    try:
        _compile_uncertainty(value.get("uncertainty"))
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    try:
        _compile_criteria(
            value.get("validity_criteria"),
            field="validity_criteria",
            minimum=1,
            maximum=6,
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    try:
        promotion_criteria = _compile_criteria(
            value.get("promotion_criteria"),
            field="promotion_criteria",
            minimum=1,
            maximum=6,
        )
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        promotion_criteria = []
    if gate_statistic and promotion_criteria:
        try:
            _require_primary_gate_criterion(
                gate_statistic,
                promotion_criteria,
            )
        except ValueError as exc:
            errors.append(str(exc))
    return list(dict.fromkeys(errors))


def _compile_screening_access_policy(value: Any) -> dict[str, bool]:
    if not isinstance(value, Mapping):
        raise TypeError("screening_access_policy must be an object")
    required = (
        "input_access",
        "within_episode_feedback",
        "cross_example_adaptation",
        "hidden_labels_for_tuning",
        "threshold_tuning",
    )
    policy: dict[str, bool] = {}
    for field in required:
        raw = value.get(field)
        if not isinstance(raw, bool):
            raise TypeError(
                f"screening_access_policy.{field} must be boolean"
            )
        policy[field] = raw
    if not policy["input_access"]:
        raise ValueError(
            "screening_access_policy.input_access must be true"
        )
    if policy["hidden_labels_for_tuning"]:
        raise ValueError(
            "screening_access_policy.hidden_labels_for_tuning must be false"
        )
    if policy["threshold_tuning"]:
        raise ValueError(
            "screening_access_policy.threshold_tuning must be false"
        )
    return policy


def _compile_gate_statistic(
    value: Any,
    legacy_threshold: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("gate_statistic must be an object")
    name = _slug(str(value.get("name", "") or ""))
    if not name:
        raise ValueError(
            "gate_statistic.name must be a non-empty machine identifier"
        )
    definition = str(value.get("definition", "") or "").strip()
    if not definition:
        raise ValueError("gate_statistic.definition must be non-empty")
    direction = value.get("direction")
    if direction not in {"maximize", "minimize"}:
        raise ValueError(
            "gate_statistic.direction must be maximize or minimize"
        )
    threshold_value: Any = value.get("threshold")
    if not isinstance(threshold_value, Mapping):
        # Keep repairs local when an in-flight response still carries the old
        # effect_threshold object.
        threshold_value = legacy_threshold
    if not isinstance(threshold_value, Mapping):
        raise TypeError("gate_statistic.threshold must be an object")
    raw = threshold_value.get("value")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not float("-inf") < float(raw) < float("inf")
        or float(raw) <= 0
    ):
        raise ValueError(
            "gate_statistic.threshold.value must be finite and positive"
        )
    scale = threshold_value.get("scale")
    if scale not in _THRESHOLD_SCALES:
        raise ValueError(
            "gate_statistic.threshold.scale must be proportion, "
            "percentage_points, or absolute"
        )
    if scale == "proportion" and float(raw) > 1:
        raise ValueError(
            "gate_statistic.threshold.value proportion must be <= 1"
        )
    if scale == "percentage_points" and float(raw) > 100:
        raise ValueError(
            "gate_statistic.threshold.value percentage_points must be <= 100"
        )
    undefined_policy = _slug(
        str(value.get("undefined_policy", "reject") or "")
    )
    if undefined_policy != "reject":
        raise ValueError(
            "gate_statistic.undefined_policy must be reject"
        )
    return {
        "name": name,
        "definition": definition,
        "direction": direction,
        "threshold": {
            "value": float(raw),
            "scale": scale,
        },
        "undefined_policy": "reject",
    }


def _compile_uncertainty(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("uncertainty must be an object")
    method = _slug(str(value.get("method", "") or ""))
    if not method:
        raise ValueError("uncertainty.method must be non-empty")
    cluster_unit = str(value.get("cluster_unit", "") or "").strip()
    if not cluster_unit:
        raise ValueError("uncertainty.cluster_unit must be non-empty")
    confidence_level = value.get("confidence_level")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.5 < float(confidence_level) < 1.0
    ):
        raise ValueError(
            "uncertainty.confidence_level must be between 0.5 and 1"
        )
    resamples = value.get("resamples", 0)
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 0
    ):
        raise ValueError(
            "uncertainty.resamples must be a non-negative integer"
        )
    if "bootstrap" in method and resamples < 200:
        raise ValueError(
            "bootstrap uncertainty requires at least 200 resamples"
        )
    result = {
        "method": method,
        "cluster_unit": cluster_unit,
        "confidence_level": float(confidence_level),
        "resamples": resamples,
    }
    if "bootstrap" in method:
        # Bootstrap mechanics are Controller-owned so Design review does not
        # repeatedly ask the Worker to invent equivalent prose variants.
        result.update(
            {
                "rng_seed": 1729,
                "interval": "percentile",
                "lower_quantile": (1.0 - float(confidence_level)) / 2.0,
                "upper_quantile": 1.0
                - (1.0 - float(confidence_level)) / 2.0,
                "undefined_resample_policy": "drop",
                "max_undefined_fraction": 0.05,
                "excess_undefined_decision": "reject",
            }
        )
    return result


def _compile_criteria(
    value: Any,
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    if not minimum <= len(value) <= maximum:
        raise ValueError(
            f"{field} must contain between {minimum} and {maximum} entries"
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field}[{index}] must be an object")
        criterion_id = _slug(str(item.get("id", "") or ""))
        if not criterion_id:
            raise ValueError(f"{field}[{index}].id must be non-empty")
        if criterion_id in seen:
            raise ValueError(f"duplicate {field} id: {criterion_id}")
        seen.add(criterion_id)
        metric = _slug(str(item.get("metric", "") or ""))
        if not metric:
            raise ValueError(f"{field}[{index}].metric must be non-empty")
        operator = str(item.get("operator", "") or "").strip()
        if operator not in _CRITERION_OPERATORS:
            raise ValueError(
                f"{field}[{index}].operator must be one of "
                + ", ".join(sorted(_CRITERION_OPERATORS))
            )
        raw = item.get("value")
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not float("-inf") < float(raw) < float("inf")
        ):
            raise ValueError(f"{field}[{index}].value must be finite")
        scale = item.get("scale", "absolute")
        if scale not in _THRESHOLD_SCALES:
            raise ValueError(
                f"{field}[{index}].scale must be proportion, "
                "percentage_points, or absolute"
            )
        description = str(item.get("description", "") or "").strip()
        if not description:
            raise ValueError(
                f"{field}[{index}].description must be non-empty"
            )
        result.append(
            {
                "id": criterion_id,
                "metric": metric,
                "operator": operator,
                "value": float(raw),
                "scale": scale,
                "description": description,
            }
        )
    return result


def _require_primary_gate_criterion(
    gate_statistic: Mapping[str, Any],
    criteria: list[dict[str, Any]],
) -> None:
    name = str(gate_statistic["name"])
    threshold = gate_statistic["threshold"]
    matches = [
        item for item in criteria if str(item.get("metric")) == name
    ]
    if len(matches) != 1:
        raise ValueError(
            "promotion_criteria must contain exactly one primary criterion "
            f"for gate_statistic.name={name}"
        )
    primary = matches[0]
    expected_operators = (
        {">", ">="}
        if gate_statistic["direction"] == "maximize"
        else {"<", "<="}
    )
    if primary["operator"] not in expected_operators:
        raise ValueError(
            "primary promotion criterion operator conflicts with "
            f"gate_statistic.direction={gate_statistic['direction']}"
        )
    if (
        not _numbers_equal(primary["value"], threshold["value"])
        or primary["scale"] != threshold["scale"]
    ):
        raise ValueError(
            "primary promotion criterion must use the exact "
            "gate_statistic.threshold value and scale"
        )


def _compile_decision_contract(
    *,
    validity_criteria: list[dict[str, Any]],
    promotion_criteria: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "invalid": {
            "decision": "retry",
            "criteria": [str(item["id"]) for item in validity_criteria],
            "when": (
                "any validity criterion fails or required evidence is missing"
            ),
        },
        "meets_all_promotion_criteria": {
            "decision": "promote",
            "criteria": [str(item["id"]) for item in promotion_criteria],
            "when": "all validity and all promotion criteria pass",
        },
        "valid_otherwise": {
            "decision": "reject",
            "when": "valid evidence fails any promotion criterion",
        },
    }


def _numbers_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _canonical_protocol(value: Any) -> str:
    normalized = _slug(str(value or ""))
    return _PROTOCOL_ALIASES.get(normalized, normalized)


def _compile_datasets(
    idea: IdeaRecord,
    dataset: Any,
    existing: Any,
    *,
    screening_access_policy: Mapping[str, bool],
) -> list[dict[str, Any]]:
    names: list[str] = []
    if isinstance(dataset, str) and dataset.strip():
        names.append(dataset.strip())
    if isinstance(existing, list):
        for item in existing:
            name = (
                str(item.get("name", "") or "")
                if isinstance(item, Mapping)
                else str(item or "")
            ).strip()
            if name:
                names.append(name)
    candidate_datasets = idea.candidate.get("datasets", [])
    if isinstance(candidate_datasets, list):
        names.extend(
            str(item or "").strip()
            for item in candidate_datasets
            if str(item or "").strip()
        )
    base = names[0] if names else "public benchmark"
    slug = _slug(base) or "benchmark"
    return [
        {
            "name": f"{base} development partition",
            "split_role": "development",
            "split_id": f"{slug}-development-v1",
            "used_for_adaptation": True,
            "access_policy": {
                "input_access": True,
                "within_episode_feedback": True,
                "cross_example_adaptation": True,
                "hidden_labels_for_tuning": True,
                "threshold_tuning": True,
                "available_before_scale": True,
            },
        },
        {
            "name": f"{base} screening partition",
            "split_role": "screening",
            "split_id": f"{slug}-screening-v1",
            "used_for_adaptation": bool(
                screening_access_policy["within_episode_feedback"]
                or screening_access_policy["cross_example_adaptation"]
            ),
            "access_policy": {
                **dict(screening_access_policy),
                "available_before_scale": True,
            },
        },
        {
            "name": f"{base} confirmatory partition",
            "split_role": "heldout_confirmatory",
            "split_id": f"{slug}-confirmatory-v1",
            "used_for_adaptation": False,
            "untouched": True,
            "access_policy": {
                "input_access": True,
                "within_episode_feedback": False,
                "cross_example_adaptation": False,
                "hidden_labels_for_tuning": False,
                "threshold_tuning": False,
                "available_before_scale": False,
            },
        },
    ]


def _compile_models(
    idea: IdeaRecord,
    value: Any,
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                name = str(item.get("name", "") or "").strip()
                role = str(item.get("role", "subject") or "subject").strip()
            else:
                name = str(item or "").strip()
                role = "subject"
            if name:
                result.append({"name": name, "role": role})
    if result:
        subjects = [
            item for item in result if _slug(item["role"]) == "subject"
        ]
        return [subjects[0] if subjects else result[0]]
    candidates = idea.candidate.get("models", [])
    if isinstance(candidates, list):
        for item in candidates:
            name = str(item or "").strip()
            if name:
                result.append({"name": name, "role": "subject"})
    if result:
        return [result[0]]
    return [{"name": "open-weight model", "role": "subject"}]


def _compile_arms(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError("protocol arms must be a list")
    arms: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            name = str(item.get("name", "") or "").strip()
            role = str(item.get("role", "") or "").strip()
        else:
            name = str(item or "").strip()
            role = ""
        if not name:
            raise ValueError(f"protocol arms[{index}] is missing a name")
        if not role:
            role = "treatment" if index == 0 else "control"
        arms.append({"name": name, "role": role})
    if not 2 <= len(arms) <= 3:
        raise ValueError("screening protocol requires exactly 2 or 3 arms")
    control_text = " ".join(
        f"{arm['name']} {arm['role']}" for arm in arms
    ).casefold()
    if not any(
        marker in control_text
        for marker in (
            "no-self",
            "no self",
            "single-pass",
            "single pass",
            "frozen",
            "reference",
            "baseline",
            "control",
        )
    ):
        raise ValueError(
            "screening protocol arms require an independent "
            "no-self-improvement/reference control"
        )
    return arms


def _compile_pilot(value: Any) -> dict[str, int]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    examples = _positive_int(
        raw.get("max_examples", raw.get("examples")),
        default=32,
        field="pilot.max_examples",
    )
    if not 16 <= examples <= 32:
        raise ValueError("pilot.max_examples must be between 16 and 32")
    # Keep adaptation/tuning on a small disjoint development partition. The
    # exact size is mechanical and intentionally not a model degree of freedom.
    development_examples = min(16, examples)
    seeds = _positive_int(
        raw.get("max_seeds", raw.get("seeds")),
        default=1,
        field="pilot.max_seeds",
    )
    if seeds != 1:
        raise ValueError("screening pilot must use exactly one seed")
    max_gpus = _positive_int(
        raw.get("max_gpus"),
        default=1,
        field="pilot.max_gpus",
    )
    if max_gpus != 1:
        raise ValueError("screening pilot must use exactly one GPU")
    timeout_sec = _positive_int(
        raw.get("timeout_sec"),
        default=7200,
        field="pilot.timeout_sec",
    )
    return {
        "max_gpus": max_gpus,
        "development_examples": development_examples,
        "max_examples": examples,
        "max_seeds": seeds,
        "timeout_sec": timeout_sec,
    }


def _compile_call_ledger(
    value: Any,
    accounting: Any,
    *,
    arms: list[dict[str, str]],
    pilot: Mapping[str, int],
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    raw_components: Any = value
    if isinstance(value, Mapping):
        raw_components = value.get("components", value)
    if isinstance(raw_components, Mapping):
        raw_components = [
            {"name": name, "calls_per_unit": calls}
            for name, calls in raw_components.items()
            if name not in {"total_model_calls", "formula"}
        ]
    arm_names = [str(arm["name"]) for arm in arms]
    normalized_arm_names = {_slug(name): name for name in arm_names}
    seen_keys: set[tuple[str, str, str, tuple[str, ...]]] = set()
    if isinstance(raw_components, list):
        for item in raw_components:
            if not isinstance(item, Mapping):
                raise TypeError(
                    "call_ledger.components entries must be objects"
                )
            name = _slug(str(item.get("name", "") or ""))
            if name not in _LEDGER_COMPONENTS:
                raise ValueError(f"unsupported call_ledger component: {name}")
            scope = _slug(
                str(item.get("scope", "per_arm_example_seed") or "")
            )
            scope = _LEDGER_SCOPE_ALIASES.get(scope, scope)
            if scope not in _LEDGER_SCOPES:
                raise ValueError(
                    f"unsupported call_ledger scope for {name}: {scope}"
                )
            default_role = (
                "screening"
                if "example" in scope
                else "none"
            )
            dataset_role = _slug(
                str(item.get("dataset_role", default_role) or "")
            )
            if dataset_role == "dev":
                dataset_role = "development"
            if dataset_role not in _LEDGER_DATASET_ROLES:
                raise ValueError(
                    "call_ledger dataset_role must be development, "
                    "screening, or none"
                )
            if "example" in scope and dataset_role == "none":
                raise ValueError(
                    f"call_ledger component {name} requires a dataset_role"
                )
            if "example" not in scope and dataset_role != "none":
                raise ValueError(
                    f"call_ledger component {name} with scope {scope} "
                    "must use dataset_role=none"
                )
            calls = _nonnegative_int(
                item.get(
                    "calls_per_unit",
                    item.get(
                        "calls_per_arm_example_seed",
                        item.get("calls", item.get("count")),
                    ),
                ),
                field=f"call_ledger.{name}",
            )
            if calls:
                selected_arms = _selected_arms(
                    item.get("arms"),
                    scope=scope,
                    arm_names=arm_names,
                    normalized_arm_names=normalized_arm_names,
                )
                component_key = (
                    name,
                    scope,
                    dataset_role,
                    tuple(selected_arms),
                )
                if component_key in seen_keys:
                    raise ValueError(
                        "duplicate call_ledger component identity: "
                        f"{name}/{scope}/{dataset_role}/"
                        f"{','.join(selected_arms) or '-'}"
                    )
                seen_keys.add(component_key)
                multiplicity = _ledger_multiplicity(
                    scope=scope,
                    dataset_role=dataset_role,
                    selected_arm_count=len(selected_arms),
                    pilot=pilot,
                )
                components.append(
                    {
                        "name": name,
                        "scope": scope,
                        "dataset_role": dataset_role,
                        "arms": selected_arms,
                        "calls_per_unit": calls,
                        "multiplicity": multiplicity,
                        "total_calls": calls * multiplicity,
                    }
                )
    if not components and isinstance(accounting, Mapping):
        calls = _positive_int(
            accounting.get("calls_per_example"),
            default=1,
            field="sample_accounting.calls_per_example",
        )
        components = [
            {
                "name": "final_evaluation",
                "scope": "per_arm_example_seed",
                "dataset_role": "screening",
                "arms": arm_names,
                "calls_per_unit": calls,
                "multiplicity": (
                    len(arm_names)
                    * pilot["max_examples"]
                    * pilot["max_seeds"]
                ),
                "total_calls": (
                    calls
                    * len(arm_names)
                    * pilot["max_examples"]
                    * pilot["max_seeds"]
                ),
            }
        ]
    if not components:
        raise ValueError(
            "call_ledger must declare at least one model-call component"
        )
    order = {name: index for index, name in enumerate(_LEDGER_COMPONENTS)}
    components.sort(key=lambda item: order[str(item["name"])])
    total = sum(int(component["total_calls"]) for component in components)
    return {
        "components": components,
        "formula": "sum(component.calls_per_unit * component.multiplicity)",
        "total_model_calls": total,
    }


def _selected_arms(
    value: Any,
    *,
    scope: str,
    arm_names: list[str],
    normalized_arm_names: Mapping[str, str],
) -> list[str]:
    if "arm" not in scope:
        if value not in (None, [], ()):
            raise ValueError(
                f"call_ledger scope {scope} must not declare arms"
            )
        return []
    if value is None:
        return list(arm_names)
    if not isinstance(value, list) or not value:
        raise ValueError("call_ledger arms must be a non-empty list")
    selected: list[str] = []
    for raw in value:
        normalized = _slug(str(raw or ""))
        if normalized not in normalized_arm_names:
            raise ValueError(f"call_ledger references unknown arm: {raw}")
        canonical = normalized_arm_names[normalized]
        if canonical in selected:
            raise ValueError(f"duplicate call_ledger arm: {canonical}")
        selected.append(canonical)
    return selected


def _ledger_multiplicity(
    *,
    scope: str,
    dataset_role: str,
    selected_arm_count: int,
    pilot: Mapping[str, int],
) -> int:
    examples = (
        pilot["development_examples"]
        if dataset_role == "development"
        else pilot["max_examples"]
        if dataset_role == "screening"
        else 1
    )
    seeds = pilot["max_seeds"]
    if scope == "per_arm_example_seed":
        return selected_arm_count * examples * seeds
    if scope == "per_example_seed":
        return examples * seeds
    if scope == "per_arm_seed":
        return selected_arm_count * seeds
    if scope == "per_seed":
        return seeds
    return 1


def _compile_confirmatory_followup(
    value: Any,
    *,
    pilot: Mapping[str, int],
    confirmatory_split_id: str,
) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, Mapping) else {}
    examples = _positive_int(
        raw.get(
            "examples",
            raw.get("max_examples", pilot["max_examples"] * 3),
        ),
        default=pilot["max_examples"] * 3,
        field="confirmatory_followup.examples",
    )
    if examples <= pilot["max_examples"]:
        examples = pilot["max_examples"] * 3
    seeds = raw.get("independent_seeds")
    if not isinstance(seeds, list) or len(seeds) <= pilot["max_seeds"]:
        seeds = [11, 22, 33]
    claim = str(raw.get("claim", "") or "").strip() or (
        "Only Scale may support the stronger paper-level claim."
    )
    changes = raw.get("changes")
    if not isinstance(changes, list) or not changes:
        changes = [
            "increase examples beyond the screening pilot",
            "increase independent seed coverage",
            "use the preregistered untouched confirmatory split",
        ]
    return {
        "required": True,
        "changes": [str(item) for item in changes if str(item).strip()],
        "claim": claim,
        "examples": examples,
        "independent_seeds": list(seeds),
        "split_id": confirmatory_split_id,
        "untouched": True,
    }


def _compile_string_list(value: Any, *, fallback: list[str]) -> list[str]:
    if isinstance(value, list):
        result = [str(item).strip() for item in value if str(item).strip()]
        if result:
            return result
    return list(fallback)


def _positive_int(value: Any, *, default: int, field: str) -> int:
    if value in (None, ""):
        return int(default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
