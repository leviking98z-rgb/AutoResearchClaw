"""Deterministic candidate and runtime validation for v2 jobs."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import py_compile
import re
import shlex
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any


def validate_python_tree(root: Path) -> dict[str, Any]:
    from researchclaw.experiment.validator import validate_code

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    files = sorted(root.rglob("*.py"))
    if not files:
        errors.append(
            {
                "file": "",
                "category": "structure",
                "message": "candidate contains no Python files",
            }
        )
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(
                {
                    "file": str(path.relative_to(root)),
                    "category": "syntax",
                    "message": str(exc),
                }
            )
            continue
        result = validate_code(path.read_text(encoding="utf-8"))
        for issue in result.issues:
            row = {
                "file": str(path.relative_to(root)),
                "category": issue.category,
                "message": issue.message,
                "line": issue.line,
            }
            (errors if issue.severity == "error" else warnings).append(row)
    return {
        "ok": not errors,
        "files_checked": [str(path.relative_to(root)) for path in files],
        "errors": errors,
        "warnings": warnings,
    }


def validate_research_implementation(
    root: Path,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Verify the generated project contains a real model/benchmark path."""

    errors: list[str] = []
    evidence = {
        "model_loader_calls": [],
        "dataset_loader_calls": [],
        "artifact_writes": [],
        "source_sha256": {},
    }
    python_files = sorted(root.rglob("*.py"))
    for path in python_files:
        relative = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8")
        evidence["source_sha256"][relative] = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if (
                name.endswith(".from_pretrained")
                or name in {
                    "transformers.pipeline",
                    "pipeline",
                    "vllm.LLM",
                    "LLM",
                }
            ):
                evidence["model_loader_calls"].append(
                    {"file": relative, "line": node.lineno, "call": name}
                )
            if name in {
                "datasets.load_dataset",
                "load_dataset",
                "evaluate.load",
                "load",
            }:
                evidence["dataset_loader_calls"].append(
                    {"file": relative, "line": node.lineno, "call": name}
                )
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.endswith(
                        ("metrics.json", "runtime_evidence.json")
                    )
                ):
                    evidence["artifact_writes"].append(
                        {
                            "file": relative,
                            "line": node.lineno,
                            "path": arg.value,
                        }
                    )
    if not evidence["model_loader_calls"]:
        errors.append("no real model loader call found")
    if not evidence["dataset_loader_calls"]:
        errors.append("no real dataset/benchmark loader call found")
    artifact_names = {
        Path(str(row["path"])).name
        for row in evidence["artifact_writes"]
    }
    if "metrics.json" not in artifact_names:
        errors.append("no metrics.json artifact write found")
    if "runtime_evidence.json" not in artifact_names:
        errors.append("no runtime_evidence.json artifact write found")
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").casefold()
        for path in python_files
    )
    for marker in (
        "fake_model",
        "mock_model",
        "synthetic_accuracy",
        "random.random()",
    ):
        if marker in combined:
            errors.append(f"forbidden synthetic implementation marker: {marker}")
    declared_models = [
        str(item.get("name", "") if isinstance(item, dict) else item)
        for item in plan.get("models", [])
    ]
    declared_datasets = [
        str(item.get("name", "") if isinstance(item, dict) else item)
        for item in plan.get("datasets", [])
    ]
    evidence["declared_models"] = declared_models
    evidence["declared_datasets"] = declared_datasets
    return {"ok": not errors, "errors": errors, "evidence": evidence}


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


_SCREENING_PHASE = "screening_pilot"
_SCREENING_FIELDS = frozenset(
    {
        "pilot_objective",
        "pilot_claim_scope",
        "unit_of_analysis",
        "arms",
        "sample_accounting",
        "effect_threshold",
        "confirmatory_followup",
    }
)
_SCREENING_DATASET_ROLES = frozenset(
    {
        "screening",
        "screening_pilot",
        "screening_evaluation",
    }
)
_CONFIRMATORY_DATASET_ROLES = frozenset(
    {
        "confirmatory",
        "heldout",
        "heldout_confirmatory",
        "confirmatory_heldout",
    }
)
_OUTCOME_REGIONS = frozenset(
    {
        "invalid",
        "below_effect_threshold",
        "at_or_above_effect_threshold",
    }
)
_CONTRACT_OUTCOME_REGIONS = frozenset(
    {
        "invalid",
        "meets_all_promotion_criteria",
        "valid_otherwise",
    }
)
_REGION_ALIASES = {
    "invalid": "invalid",
    "invalid_evidence": "invalid",
    "invalid_or_inconclusive": "invalid",
    "invalid_or_missing": "invalid",
    "missing_or_invalid": "invalid",
    "inconclusive": "invalid",
    "unusable": "invalid",
    "below_threshold": "below_effect_threshold",
    "below_effect_threshold": "below_effect_threshold",
    "effect_below_threshold": "below_effect_threshold",
    "subthreshold": "below_effect_threshold",
    "at_or_above_threshold": "at_or_above_effect_threshold",
    "at_or_above_effect_threshold": "at_or_above_effect_threshold",
    "effect_at_or_above_threshold": "at_or_above_effect_threshold",
    "meets_threshold": "at_or_above_effect_threshold",
    "threshold_met": "at_or_above_effect_threshold",
    "above_threshold": "at_or_above_effect_threshold",
    "meets_all_promotion_criteria": "meets_all_promotion_criteria",
    "all_promotion_criteria_met": "meets_all_promotion_criteria",
    "all_criteria_met": "meets_all_promotion_criteria",
    "valid_otherwise": "valid_otherwise",
    "valid_but_not_promoted": "valid_otherwise",
    "valid_reject": "valid_otherwise",
}


def validate_plan(value: dict[str, Any]) -> list[str]:
    if not isinstance(value, Mapping):
        return ["plan must be an object"]

    errors: list[str] = []
    for field in (
        "research_question",
        "hypothesis",
        "primary_metric",
        "pilot",
        "promotion_rule",
        "early_stop_rule",
        "estimand",
        "sample_size_rationale",
        "workload_budget",
    ):
        if not value.get(field):
            errors.append(f"missing {field}")
    for field in ("datasets", "models", "baselines", "ablations"):
        if not isinstance(value.get(field), list) or not value[field]:
            errors.append(f"missing {field}")

    phase_aware = "study_phase" in value or any(
        field in value for field in _SCREENING_FIELDS
    )
    if phase_aware:
        _validate_protocol_compiler_contract(value, errors)

    pilot = value.get("pilot")
    pilot_numbers: dict[str, int] = {}
    if isinstance(pilot, dict):
        for field in ("max_gpus", "max_examples", "max_seeds", "timeout_sec"):
            number = _positive_number(
                pilot,
                field,
                f"pilot.{field}",
                errors,
                integer=phase_aware,
            )
            if isinstance(number, int):
                pilot_numbers[field] = number
    elif pilot:
        errors.append("pilot must be an object")

    workload = value.get("workload_budget")
    workload_numbers: dict[str, int] = {}
    compiled_ledger = isinstance(value.get("call_ledger"), Mapping)
    if isinstance(workload, dict):
        workload_fields = [
            "conditions",
            "examples",
            "seeds",
            "max_new_tokens",
            "estimated_model_calls",
        ]
        if not compiled_ledger:
            workload_fields.insert(1, "models")
        if "development_examples" in workload:
            workload_fields.append("development_examples")
        for field in workload_fields:
            number = _positive_number(
                workload,
                field,
                f"workload_budget.{field}",
                errors,
                integer=True,
                strict_type=phase_aware,
            )
            if isinstance(number, int):
                workload_numbers[field] = number
    elif workload:
        errors.append("workload_budget must be an object")

    if phase_aware:
        _validate_screening_plan(
            value,
            errors,
            pilot_numbers=pilot_numbers,
            workload_numbers=workload_numbers,
            compiled_ledger=compiled_ledger,
        )

    _validate_decision_table(
        value.get("decision_table"),
        errors,
        structured=phase_aware,
        metric_direction=value.get("metric_direction"),
        decision_contract=value.get("decision_contract"),
    )
    return _deduplicate(errors)


def _validate_protocol_compiler_contract(
    value: Mapping[str, Any],
    errors: list[str],
) -> None:
    protocol = value.get("protocol_template")
    if protocol is not None:
        from .protocols import SUPPORTED_PROTOCOLS

        if protocol not in SUPPORTED_PROTOCOLS:
            errors.append(
                "protocol_template must be one of: "
                + ", ".join(sorted(SUPPORTED_PROTOCOLS))
            )
    compiler = value.get("compiler")
    compiler_version = 0
    if isinstance(compiler, Mapping):
        try:
            compiler_version = int(compiler.get("version", 0) or 0)
        except (TypeError, ValueError):
            errors.append("compiler.version must be an integer")
    typed_decisions = compiler_version >= 2 or any(
        field in value
        for field in (
            "gate_statistic",
            "decision_contract",
            "screening_access_policy",
            "validity_criteria",
            "promotion_criteria",
        )
    )
    if typed_decisions:
        _validate_typed_decision_contract(value, errors)
        _validate_dataset_access_policies(value, errors)
    ledger = value.get("call_ledger")
    if ledger is None:
        return
    if not isinstance(ledger, Mapping):
        errors.append("call_ledger must be an object")
        return
    components = ledger.get("components")
    if not isinstance(components, list) or not components:
        errors.append("call_ledger.components must be a non-empty list")
        return
    compiled_total = 0
    identities: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            errors.append(
                f"invalid call_ledger.components[{index}]: must be an object"
            )
            continue
        name = component.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(
                f"missing call_ledger.components[{index}].name"
            )
        else:
            raw_arms = component.get("arms")
            arms = (
                tuple(str(arm) for arm in raw_arms)
                if isinstance(raw_arms, list)
                else ()
            )
            identity = (
                name,
                str(component.get("scope", "") or ""),
                str(component.get("dataset_role", "") or ""),
                arms,
            )
            if identity in identities:
                errors.append(
                    f"duplicate call_ledger component identity {identity!r}"
                )
            else:
                identities.add(identity)
        calls = component.get("calls_per_unit")
        if (
            isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls <= 0
        ):
            errors.append(
                "invalid call_ledger.components"
                f"[{index}].calls_per_unit: "
                "must be a positive integer"
            )
        multiplicity = component.get("multiplicity")
        if (
            isinstance(multiplicity, bool)
            or not isinstance(multiplicity, int)
            or multiplicity <= 0
        ):
            errors.append(
                f"invalid call_ledger.components[{index}].multiplicity"
            )
        total_calls = component.get("total_calls")
        if (
            isinstance(calls, int)
            and not isinstance(calls, bool)
            and isinstance(multiplicity, int)
            and not isinstance(multiplicity, bool)
            and total_calls != calls * multiplicity
        ):
            errors.append(
                f"call_ledger.components[{index}].total_calls="
                f"{total_calls} does not equal calls_per_unit * "
                f"multiplicity={calls * multiplicity}"
            )
        if isinstance(total_calls, int) and not isinstance(total_calls, bool):
            compiled_total += total_calls
    accounting = value.get("sample_accounting")
    if not isinstance(accounting, Mapping) or compiled_total <= 0:
        return
    declared_total = ledger.get("total_model_calls")
    if declared_total != compiled_total:
        errors.append(
            "call_ledger.total_model_calls="
            f"{declared_total} does not equal component total={compiled_total}"
        )
    if accounting.get("total_model_calls") != compiled_total:
        errors.append(
            "sample_accounting.total_model_calls="
            f"{accounting.get('total_model_calls')} does not equal "
            f"call_ledger total={compiled_total}"
        )


def _validate_typed_decision_contract(
    value: Mapping[str, Any],
    errors: list[str],
) -> None:
    gate = value.get("gate_statistic")
    gate_name = ""
    gate_direction = ""
    gate_threshold: Mapping[str, Any] = {}
    if not isinstance(gate, Mapping):
        errors.append("gate_statistic must be an object")
    else:
        gate_name = _slug(str(gate.get("name", "") or ""))
        if not gate_name:
            errors.append("gate_statistic.name must be non-empty")
        if not str(gate.get("definition", "") or "").strip():
            errors.append("gate_statistic.definition must be non-empty")
        gate_direction = str(gate.get("direction", "") or "")
        if gate_direction not in {"maximize", "minimize"}:
            errors.append(
                "gate_statistic.direction must be maximize or minimize"
            )
        if gate.get("undefined_policy") != "reject":
            errors.append("gate_statistic.undefined_policy must be reject")
        threshold = gate.get("threshold")
        if not isinstance(threshold, Mapping):
            errors.append("gate_statistic.threshold must be an object")
        else:
            gate_threshold = threshold
            _validate_threshold_shape(
                threshold,
                "gate_statistic.threshold",
                errors,
            )

    uncertainty = value.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        errors.append("uncertainty must be an object")
    else:
        method = _slug(str(uncertainty.get("method", "") or ""))
        if not method:
            errors.append("uncertainty.method must be non-empty")
        if not str(uncertainty.get("cluster_unit", "") or "").strip():
            errors.append("uncertainty.cluster_unit must be non-empty")
        confidence = uncertainty.get("confidence_level")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.5 < float(confidence) < 1
        ):
            errors.append(
                "uncertainty.confidence_level must be between 0.5 and 1"
            )
        resamples = uncertainty.get("resamples")
        if (
            isinstance(resamples, bool)
            or not isinstance(resamples, int)
            or resamples < 0
        ):
            errors.append(
                "uncertainty.resamples must be a non-negative integer"
            )
        elif "bootstrap" in method and resamples < 200:
            errors.append(
                "bootstrap uncertainty requires at least 200 resamples"
            )

    validity = _validate_typed_criteria(
        value.get("validity_criteria"),
        "validity_criteria",
        errors,
    )
    promotion = _validate_typed_criteria(
        value.get("promotion_criteria"),
        "promotion_criteria",
        errors,
    )
    if gate_name and gate_threshold and promotion:
        matching = [
            criterion
            for criterion in promotion
            if criterion.get("metric") == gate_name
        ]
        if len(matching) != 1:
            errors.append(
                "promotion_criteria must contain exactly one primary "
                f"criterion for gate_statistic.name={gate_name}"
            )
        else:
            primary = matching[0]
            allowed = (
                {">", ">="}
                if gate_direction == "maximize"
                else {"<", "<="}
                if gate_direction == "minimize"
                else set()
            )
            if primary.get("operator") not in allowed:
                errors.append(
                    "primary promotion criterion operator conflicts with "
                    f"gate_statistic.direction={gate_direction}"
                )
            if (
                not _numeric_equal(
                    primary.get("value"),
                    gate_threshold.get("value"),
                )
                or primary.get("scale") != gate_threshold.get("scale")
            ):
                errors.append(
                    "primary promotion criterion must use the exact "
                    "gate_statistic.threshold value and scale"
                )

    contract = value.get("decision_contract")
    expected = {
        "invalid": "retry",
        "meets_all_promotion_criteria": "promote",
        "valid_otherwise": "reject",
    }
    if not isinstance(contract, Mapping):
        errors.append("decision_contract must be an object")
    else:
        for region, decision in expected.items():
            row = contract.get(region)
            if not isinstance(row, Mapping):
                errors.append(f"decision_contract.{region} must be an object")
                continue
            if row.get("decision") != decision:
                errors.append(
                    f"decision_contract.{region}.decision must be {decision}"
                )
        invalid = contract.get("invalid")
        if isinstance(invalid, Mapping):
            declared = invalid.get("criteria")
            expected_ids = [str(item.get("id")) for item in validity]
            if declared != expected_ids:
                errors.append(
                    "decision_contract.invalid.criteria must exactly match "
                    "validity_criteria ids"
                )
        promoted = contract.get("meets_all_promotion_criteria")
        if isinstance(promoted, Mapping):
            declared = promoted.get("criteria")
            expected_ids = [str(item.get("id")) for item in promotion]
            if declared != expected_ids:
                errors.append(
                    "decision_contract.meets_all_promotion_criteria.criteria "
                    "must exactly match promotion_criteria ids"
                )


def _validate_typed_criteria(
    value: Any,
    field: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        errors.append(f"{field} must be a non-empty list")
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, criterion in enumerate(value):
        if not isinstance(criterion, Mapping):
            errors.append(f"{field}[{index}] must be an object")
            continue
        criterion_id = _slug(str(criterion.get("id", "") or ""))
        metric = _slug(str(criterion.get("metric", "") or ""))
        operator = str(criterion.get("operator", "") or "").strip()
        scale = criterion.get("scale")
        raw = criterion.get("value")
        if not criterion_id:
            errors.append(f"{field}[{index}].id must be non-empty")
        elif criterion_id in seen:
            errors.append(f"duplicate {field} id: {criterion_id}")
        else:
            seen.add(criterion_id)
        if not metric:
            errors.append(f"{field}[{index}].metric must be non-empty")
        if operator not in {"<", "<=", ">", ">=", "=="}:
            errors.append(
                f"{field}[{index}].operator must be a supported comparator"
            )
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
        ):
            errors.append(f"{field}[{index}].value must be finite")
        if scale not in {"proportion", "percentage_points", "absolute"}:
            errors.append(
                f"{field}[{index}].scale must be proportion, "
                "percentage_points, or absolute"
            )
        if not str(criterion.get("description", "") or "").strip():
            errors.append(f"{field}[{index}].description must be non-empty")
        result.append(
            {
                "id": criterion_id,
                "metric": metric,
                "operator": operator,
                "value": raw,
                "scale": scale,
            }
        )
    return result


def _validate_threshold_shape(
    threshold: Mapping[str, Any],
    path: str,
    errors: list[str],
) -> None:
    raw = threshold.get("value")
    if (
        isinstance(raw, bool)
        or not isinstance(raw, (int, float))
        or not math.isfinite(float(raw))
        or float(raw) <= 0
    ):
        errors.append(f"{path}.value must be finite and positive")
    scale = threshold.get("scale")
    if scale not in {"proportion", "percentage_points", "absolute"}:
        errors.append(
            f"{path}.scale must be proportion, percentage_points, or absolute"
        )
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        if scale == "proportion" and float(raw) > 1:
            errors.append(f"{path}.value proportion must be <= 1")
        if scale == "percentage_points" and float(raw) > 100:
            errors.append(
                f"{path}.value percentage_points must be <= 100"
            )


def _numeric_equal(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    except (TypeError, ValueError):
        return False


def _positive_number(
    value: Mapping[str, Any],
    field: str,
    path: str,
    errors: list[str],
    *,
    integer: bool,
    strict_type: bool = True,
) -> int | float | None:
    if field not in value:
        errors.append(f"invalid {path}: missing positive value")
        return None
    raw = value[field]
    if isinstance(raw, bool):
        errors.append(f"invalid {path}: must be positive")
        return None
    if integer and strict_type:
        if not isinstance(raw, int) or raw <= 0:
            errors.append(f"invalid {path}: must be a positive integer")
            return None
        return raw
    try:
        number = int(raw) if integer else float(raw)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"invalid {path}: must be positive")
        return None
    if number <= 0 or not math.isfinite(float(number)):
        errors.append(f"invalid {path}: must be positive")
        return None
    return number


def _validate_screening_plan(
    value: Mapping[str, Any],
    errors: list[str],
    *,
    pilot_numbers: Mapping[str, int],
    workload_numbers: Mapping[str, int],
    compiled_ledger: bool,
) -> None:
    phase = value.get("study_phase")
    if phase is None:
        errors.append("missing study_phase")
    elif phase != _SCREENING_PHASE:
        errors.append("study_phase must be 'screening_pilot'")

    for field in (
        "pilot_objective",
        "pilot_claim_scope",
        "unit_of_analysis",
    ):
        if field not in value:
            errors.append(f"missing {field}")
        elif not isinstance(value[field], str) or not value[field].strip():
            errors.append(f"invalid {field}: must be a non-empty string")
    arms = value.get("arms")
    arm_count: int | None = None
    if not isinstance(arms, list):
        errors.append("arms must be a list with at least two entries")
    else:
        arm_count = len(arms)
        if arm_count < 2:
            errors.append("arms must be a list with at least two entries")
        names: dict[str, int] = {}
        for index, arm in enumerate(arms):
            if not isinstance(arm, Mapping):
                errors.append(f"invalid arms[{index}]: must be an object")
                continue
            name = arm.get("name")
            role = arm.get("role")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"missing arms[{index}].name")
            else:
                normalized = _slug(name)
                if normalized in names:
                    errors.append(
                        f"duplicate arms name {name!r} at indexes "
                        f"{names[normalized]} and {index}"
                    )
                else:
                    names[normalized] = index
            if not isinstance(role, str) or not role.strip():
                errors.append(f"missing arms[{index}].role")

    accounting = value.get("sample_accounting")
    accounting_numbers: dict[str, int] = {}
    if not isinstance(accounting, Mapping):
        errors.append("sample_accounting must be an object")
    else:
        accounting_fields = [
            "arms",
            "examples_per_arm",
            "seeds",
            "total_model_calls",
        ]
        if not compiled_ledger:
            accounting_fields.insert(3, "calls_per_example")
        if "development_examples" in accounting:
            accounting_fields.append("development_examples")
        for field in accounting_fields:
            number = _positive_number(
                accounting,
                field,
                f"sample_accounting.{field}",
                errors,
                integer=True,
            )
            if isinstance(number, int):
                accounting_numbers[field] = number

    declared_arms = accounting_numbers.get("arms")
    if (
        arm_count is not None
        and declared_arms is not None
        and declared_arms != arm_count
    ):
        errors.append(
            "sample_accounting.arms="
            f"{declared_arms} does not match len(arms)={arm_count}"
        )

    if not compiled_ledger:
        factors = (
            accounting_numbers.get("arms"),
            accounting_numbers.get("examples_per_arm"),
            accounting_numbers.get("seeds"),
            accounting_numbers.get("calls_per_example"),
        )
        if all(number is not None for number in factors):
            expected_calls = math.prod(int(number) for number in factors)
            total_calls = accounting_numbers.get("total_model_calls")
            if total_calls is not None and total_calls != expected_calls:
                errors.append(
                    "sample_accounting.total_model_calls="
                    f"{total_calls} does not equal arms * examples_per_arm * "
                    f"seeds * calls_per_example={expected_calls}"
                )

    _validate_workload_arithmetic(
        accounting_numbers,
        workload_numbers,
        pilot_numbers,
        errors,
        compiled_ledger=compiled_ledger,
    )
    _validate_effect_threshold(value, accounting_numbers, errors)
    _validate_dataset_isolation(value, errors)
    _validate_confirmatory_followup(
        value.get("confirmatory_followup"),
        errors,
        pilot_numbers=pilot_numbers,
        workload_numbers=workload_numbers,
        screening_split_ids=_plan_split_ids(value, role="screening"),
        confirmatory_split_ids=_plan_split_ids(value, role="confirmatory"),
        production=True,
    )


def _validate_confirmatory_followup(
    followup: Any,
    errors: list[str],
    *,
    pilot_numbers: Mapping[str, int],
    workload_numbers: Mapping[str, int],
    screening_split_ids: set[str],
    confirmatory_split_ids: set[str],
    production: bool,
) -> None:
    if followup is None:
        errors.append("missing confirmatory_followup")
        return
    if isinstance(followup, str):
        if not followup.strip():
            errors.append(
                "invalid confirmatory_followup: must be non-empty"
            )
        elif production:
            errors.append(
                "confirmatory_followup must be an object for "
                "screening_pilot"
            )
        return
    if not isinstance(followup, Mapping):
        errors.append(
            "invalid confirmatory_followup: must be an object"
        )
        return
    if followup.get("required") is not True:
        errors.append("confirmatory_followup.required must be true")
    changes = followup.get("changes")
    if (
        not isinstance(changes, list)
        or not changes
        or any(
            not isinstance(change, str) or not change.strip()
            for change in changes
        )
    ):
        errors.append(
            "confirmatory_followup.changes must be a non-empty string list"
        )
    if (
        not isinstance(followup.get("claim"), str)
        or not followup["claim"].strip()
    ):
        errors.append(
            "confirmatory_followup.claim must be a non-empty string"
        )

    pilot_examples = pilot_numbers.get(
        "max_examples",
        workload_numbers.get("examples"),
    )
    pilot_seed_count = pilot_numbers.get(
        "max_seeds",
        workload_numbers.get("seeds"),
    )
    confirmatory_examples = _positive_integer_alias(
        followup,
        ("examples", "max_examples", "examples_per_arm"),
        "confirmatory_followup.examples",
        errors,
    )
    confirmatory_seed_count = _confirmatory_seed_count(followup, errors)
    confirmatory_split_id = _split_identifier(followup)
    if confirmatory_split_id is None:
        errors.append(
            "confirmatory_followup.split_id must identify an untouched "
            "confirmatory split"
        )
    untouched = followup.get(
        "untouched",
        followup.get("split_untouched"),
    )
    if untouched is not True:
        errors.append("confirmatory_followup.untouched must be true")

    if (
        pilot_examples is not None
        and confirmatory_examples is not None
        and confirmatory_examples <= pilot_examples
    ):
        errors.append(
            "confirmatory_followup.examples must exceed pilot examples "
            f"({confirmatory_examples} <= {pilot_examples})"
        )
    if (
        pilot_seed_count is not None
        and confirmatory_seed_count is not None
        and confirmatory_seed_count <= pilot_seed_count
    ):
        errors.append(
            "confirmatory_followup independent seed count must exceed pilot "
            f"seeds ({confirmatory_seed_count} <= {pilot_seed_count})"
        )
    if (
        confirmatory_split_id is not None
        and confirmatory_split_id in screening_split_ids
    ):
        errors.append(
            "confirmatory_followup.split_id must differ from the pilot "
            "screening split"
        )
    if (
        confirmatory_split_id is not None
        and confirmatory_split_ids
        and confirmatory_split_id not in confirmatory_split_ids
    ):
        errors.append(
            "confirmatory_followup.split_id must match a confirmatory "
            "dataset split"
        )
def _positive_integer_alias(
    value: Mapping[str, Any],
    aliases: tuple[str, ...],
    path: str,
    errors: list[str],
) -> int | None:
    found = [(field, value[field]) for field in aliases if field in value]
    if not found:
        errors.append(f"invalid {path}: missing positive integer")
        return None
    parsed: list[tuple[str, int]] = []
    for field, raw in found:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            errors.append(
                f"invalid confirmatory_followup.{field}: "
                "must be a positive integer"
            )
            continue
        parsed.append((field, raw))
    if len({number for _, number in parsed}) > 1:
        errors.append(
            f"{path} aliases disagree: "
            + ", ".join(f"{field}={number}" for field, number in parsed)
        )
        return None
    return parsed[0][1] if parsed else None


def _confirmatory_seed_count(
    followup: Mapping[str, Any],
    errors: list[str],
) -> int | None:
    seed_values = followup.get(
        "independent_seeds",
        followup.get("seeds"),
    )
    declared_count = _optional_positive_integer_alias(
        followup,
        ("independent_seed_count", "seed_count", "max_seeds"),
        "confirmatory_followup.independent_seed_count",
        errors,
    )
    list_count: int | None = None
    if isinstance(seed_values, list):
        if not seed_values:
            errors.append(
                "confirmatory_followup.independent_seeds must be a "
                "non-empty list"
            )
        elif any(isinstance(seed, (list, dict, set)) for seed in seed_values):
            errors.append(
                "confirmatory_followup.independent_seeds must contain "
                "scalar identifiers"
            )
        elif len({_identity(seed) for seed in seed_values}) != len(seed_values):
            errors.append(
                "confirmatory_followup.independent_seeds must be unique"
            )
        else:
            list_count = len(seed_values)
    elif seed_values is not None:
        if (
            isinstance(seed_values, bool)
            or not isinstance(seed_values, int)
            or seed_values <= 0
        ):
            errors.append(
                "confirmatory_followup.independent_seeds must be a "
                "positive integer or non-empty list"
            )
        else:
            list_count = seed_values
    elif declared_count is None:
        errors.append(
            "confirmatory_followup.independent_seeds must declare a "
            "positive count or non-empty list"
        )

    if (
        declared_count is not None
        and list_count is not None
        and declared_count != list_count
    ):
        errors.append(
            "confirmatory_followup independent seed declarations disagree"
        )
        return None
    return declared_count if declared_count is not None else list_count


def _optional_positive_integer_alias(
    value: Mapping[str, Any],
    aliases: tuple[str, ...],
    path: str,
    errors: list[str],
) -> int | None:
    if not any(field in value for field in aliases):
        return None
    return _positive_integer_alias(value, aliases, path, errors)


def _validate_workload_arithmetic(
    accounting: Mapping[str, int],
    workload: Mapping[str, int],
    pilot: Mapping[str, int],
    errors: list[str],
    *,
    compiled_ledger: bool,
) -> None:
    if not compiled_ledger:
        workload_factors = (
            workload.get("conditions"),
            workload.get("models"),
            workload.get("examples"),
            workload.get("seeds"),
        )
        if all(number is not None for number in workload_factors):
            expected = math.prod(int(number) for number in workload_factors)
            estimated = workload.get("estimated_model_calls")
            if estimated is not None and estimated != expected:
                errors.append(
                    "workload_budget.estimated_model_calls="
                    f"{estimated} does not equal conditions * models * "
                    f"examples * seeds={expected}"
                )

    field_pairs = [
        ("arms", "conditions"),
        ("examples_per_arm", "examples"),
        ("seeds", "seeds"),
        ("total_model_calls", "estimated_model_calls"),
    ]
    if not compiled_ledger:
        field_pairs.insert(3, ("calls_per_example", "models"))
    if (
        "development_examples" in accounting
        or "development_examples" in workload
    ):
        field_pairs.append(
            ("development_examples", "development_examples")
        )
    for accounting_field, workload_field in field_pairs:
        left = accounting.get(accounting_field)
        right = workload.get(workload_field)
        if left is not None and right is not None and left != right:
            errors.append(
                f"sample_accounting.{accounting_field}={left} does not match "
                f"workload_budget.{workload_field}={right}"
            )

    for accounting_field, pilot_field in (
        ("examples_per_arm", "max_examples"),
        ("seeds", "max_seeds"),
    ):
        planned = accounting.get(accounting_field)
        maximum = pilot.get(pilot_field)
        if (
            planned is not None
            and maximum is not None
            and planned > maximum
        ):
            errors.append(
                f"sample_accounting.{accounting_field}={planned} exceeds "
                f"pilot.{pilot_field}={maximum}"
            )


def _validate_effect_threshold(
    value: Mapping[str, Any],
    accounting: Mapping[str, int],
    errors: list[str],
) -> None:
    threshold = value.get("effect_threshold")
    if not isinstance(threshold, Mapping):
        errors.append("effect_threshold must be an object")
        return

    raw_value = threshold.get("value")
    if (
        isinstance(raw_value, bool)
        or not isinstance(raw_value, (int, float))
        or not math.isfinite(float(raw_value))
        or float(raw_value) <= 0
    ):
        errors.append("invalid effect_threshold.value: must be finite and > 0")
        effect_value = None
    else:
        effect_value = float(raw_value)

    scale = threshold.get("scale")
    if scale not in {"proportion", "percentage_points", "absolute"}:
        errors.append(
            "invalid effect_threshold.scale: expected "
            "'proportion', 'percentage_points', or 'absolute'"
        )
        scale = None
    elif effect_value is not None:
        if scale == "proportion" and effect_value > 1:
            errors.append(
                "invalid effect_threshold.value: proportion must be <= 1"
            )
        if scale == "percentage_points" and effect_value > 100:
            errors.append(
                "invalid effect_threshold.value: percentage_points must be "
                "<= 100"
            )

    examples = accounting.get("examples_per_arm")
    seeds = accounting.get("seeds")
    if effect_value is None or scale is None or examples is None or seeds is None:
        return
    analysis_units = examples * seeds
    resolution = (100.0 if scale == "percentage_points" else 1.0) / (
        analysis_units
    )
    if effect_value < resolution and not math.isclose(
        effect_value,
        resolution,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        errors.append(
            f"effect_threshold.value={effect_value:g} {scale} is below "
            f"pilot sample resolution={resolution:g} for "
            f"{analysis_units} analysis units per arm"
        )


def _validate_dataset_isolation(
    value: Mapping[str, Any],
    errors: list[str],
) -> None:
    datasets = value.get("datasets")
    if not isinstance(datasets, list):
        return

    heldout_names: list[str] = []
    seen_names: dict[str, int] = {}
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping):
            errors.append(
                f"invalid datasets[{index}]: screening_pilot datasets must "
                "be objects"
            )
            continue
        name = dataset.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"missing datasets[{index}].name")
            normalized_name = ""
        else:
            normalized_name = _compact(name)
            if normalized_name in seen_names:
                errors.append(
                    f"duplicate dataset name {name!r} at indexes "
                    f"{seen_names[normalized_name]} and {index}"
                )
            else:
                seen_names[normalized_name] = index
        split_role = dataset.get("split_role")
        if not isinstance(split_role, str):
            errors.append(
                f"invalid datasets[{index}].split_role: expected "
                "development, screening, or heldout_confirmatory"
            )
            continue
        normalized_role = _slug(split_role)
        if normalized_role in {"dev", "development", "screening"}:
            is_heldout = False
        elif normalized_role in {"heldout", "heldout_confirmatory"}:
            is_heldout = True
        else:
            errors.append(
                f"invalid datasets[{index}].split_role: expected "
                "development, screening, or heldout_confirmatory"
            )
            continue
        if is_heldout and normalized_name:
            heldout_names.append(normalized_name)
            for key, child in dataset.items():
                if _is_adaptation_key(str(key)) and _declares_use(child):
                    errors.append(
                        f"heldout dataset {name!r} participates in adaptation "
                        f"via datasets[{index}].{key}"
                    )
                    break
            for key in ("purpose", "usage", "used_for", "role"):
                if key in dataset and _mentions_adaptation(dataset[key]):
                    errors.append(
                        f"heldout dataset {name!r} participates in adaptation "
                        f"via datasets[{index}].{key}"
                    )
                    break

    if not heldout_names:
        return
    heldout_markers = {
        *heldout_names,
        "heldout",
        "heldoutdata",
        "heldoutdataset",
        "holdout",
        "holdoutdata",
        "testsplit",
    }
    for path, reference in _adaptation_references(value):
        if isinstance(reference, str):
            normalized = _compact(reference)
            if normalized in heldout_markers:
                errors.append(
                    "heldout data must not participate in adaptation: "
                    f"{path} references {reference!r}"
                )
        elif reference is True and any(
            marker in _compact(path)
            for marker in ("heldout", "holdout")
        ):
            errors.append(
                "heldout data must not participate in adaptation: "
                f"{path} is true"
            )


def _validate_dataset_access_policies(
    value: Mapping[str, Any],
    errors: list[str],
) -> None:
    top_level = value.get("screening_access_policy")
    expected_fields = (
        "input_access",
        "within_episode_feedback",
        "cross_example_adaptation",
        "hidden_labels_for_tuning",
        "threshold_tuning",
    )
    if not isinstance(top_level, Mapping):
        errors.append("screening_access_policy must be an object")
        top_level = {}
    for field in expected_fields:
        if not isinstance(top_level.get(field), bool):
            errors.append(
                f"screening_access_policy.{field} must be boolean"
            )
    if top_level.get("input_access") is not True:
        errors.append("screening_access_policy.input_access must be true")
    for field in ("hidden_labels_for_tuning", "threshold_tuning"):
        if top_level.get(field) is not False:
            errors.append(
                f"screening_access_policy.{field} must be false"
            )

    datasets = value.get("datasets")
    if not isinstance(datasets, list):
        return
    roles_seen: set[str] = set()
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, Mapping):
            continue
        role = _dataset_role(dataset.get("split_role"))
        if role is None:
            continue
        roles_seen.add(role)
        policy = dataset.get("access_policy")
        if not isinstance(policy, Mapping):
            errors.append(f"datasets[{index}].access_policy must be an object")
            continue
        for field in (*expected_fields, "available_before_scale"):
            if not isinstance(policy.get(field), bool):
                errors.append(
                    f"datasets[{index}].access_policy.{field} must be boolean"
                )
        if role == "screening":
            expected = {
                **dict(top_level),
                "available_before_scale": True,
            }
            for field, expected_value in expected.items():
                if policy.get(field) is not expected_value:
                    errors.append(
                        f"datasets[{index}].access_policy.{field} must match "
                        "screening_access_policy"
                    )
            adaptive = bool(
                top_level.get("within_episode_feedback")
                or top_level.get("cross_example_adaptation")
            )
            if dataset.get("used_for_adaptation") is not adaptive:
                errors.append(
                    f"datasets[{index}].used_for_adaptation must be {adaptive} "
                    "for the declared screening access policy"
                )
        if role == "confirmatory":
            expected_confirmatory = {
                "input_access": True,
                "within_episode_feedback": False,
                "cross_example_adaptation": False,
                "hidden_labels_for_tuning": False,
                "threshold_tuning": False,
                "available_before_scale": False,
            }
            for field, expected_value in expected_confirmatory.items():
                if policy.get(field) is not expected_value:
                    errors.append(
                        f"datasets[{index}].access_policy.{field} must be "
                        f"{expected_value} for confirmatory data"
                    )
            if dataset.get("untouched") is not True:
                errors.append(
                    f"datasets[{index}].untouched must be true for "
                    "confirmatory data"
                )
            if dataset.get("used_for_adaptation") is not False:
                errors.append(
                    f"datasets[{index}].used_for_adaptation must be false "
                    "for confirmatory data"
                )
    for required_role in ("development", "screening", "confirmatory"):
        if required_role not in roles_seen:
            errors.append(
                f"compiled protocol is missing {required_role} dataset role"
            )


def _plan_split_ids(
    plan: Mapping[str, Any],
    *,
    role: str,
) -> set[str]:
    datasets = plan.get("datasets")
    if not isinstance(datasets, list):
        return set()
    split_ids: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        split_role = _dataset_role(dataset.get("split_role"))
        if split_role != role:
            continue
        split_id = _split_identifier(dataset)
        if split_id is not None:
            split_ids.add(split_id)
    return split_ids


def _dataset_role(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = _slug(value)
    if normalized in _SCREENING_DATASET_ROLES:
        return "screening"
    if normalized in _CONFIRMATORY_DATASET_ROLES:
        return "confirmatory"
    if normalized in {"dev", "development"}:
        return "development"
    return None


def _split_identifier(value: Mapping[str, Any]) -> str | None:
    for field in (
        "split_id",
        "split_identifier",
        "dataset_split_id",
        "split",
    ):
        if field not in value:
            continue
        raw = value[field]
        if isinstance(raw, str) and raw.strip():
            return _compact(raw)
        if (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
        ):
            return _compact(str(raw))
    return None


def _runtime_dataset_roles(
    runtime: Mapping[str, Any],
) -> dict[str, tuple[str | None, str | None, bool | None]]:
    raw_roles = runtime.get("dataset_roles")
    if isinstance(raw_roles, Mapping):
        entries = []
        for dataset, declaration in raw_roles.items():
            if _dataset_role(dataset) is not None and (
                isinstance(declaration, str)
                or (
                    isinstance(declaration, (int, float))
                    and not isinstance(declaration, bool)
                )
            ):
                entries.append(
                    (
                        str(dataset),
                        {
                            "role": str(dataset),
                            "split_id": declaration,
                        },
                    )
                )
            else:
                entries.append((str(dataset), declaration))
    elif isinstance(raw_roles, list):
        entries = []
        for index, declaration in enumerate(raw_roles):
            if not isinstance(declaration, Mapping):
                entries.append((f"#{index}", declaration))
                continue
            name = declaration.get(
                "dataset",
                declaration.get("name", f"#{index}"),
            )
            entries.append((str(name), declaration))
    else:
        return {}

    roles: dict[str, tuple[str | None, str | None, bool | None]] = {}
    for dataset, declaration in entries:
        role: str | None
        split_id: str | None
        untouched: bool | None = None
        if isinstance(declaration, str):
            role = _dataset_role(declaration)
            split_id = None
        elif isinstance(declaration, Mapping):
            role = _dataset_role(
                declaration.get("role", declaration.get("split_role"))
            )
            split_id = _split_identifier(declaration)
            raw_untouched = declaration.get(
                "untouched",
                declaration.get("split_untouched"),
            )
            if isinstance(raw_untouched, bool):
                untouched = raw_untouched
        elif (
            _dataset_role(dataset) is not None
            and isinstance(declaration, (int, float))
            and not isinstance(declaration, bool)
        ):
            role = _dataset_role(dataset)
            split_id = _split_identifier({"split_id": declaration})
        else:
            role = None
            split_id = None
        roles[_compact(dataset)] = (role, split_id, untouched)
    return roles


def _runtime_split_ids(
    runtime: Mapping[str, Any],
) -> dict[str, str]:
    split_ids: dict[str, str] = {}
    for field in ("split_identifiers", "split_ids"):
        raw_ids = runtime.get(field)
        if not isinstance(raw_ids, Mapping):
            continue
        for raw_role, raw_split_id in raw_ids.items():
            role = _role_from_key(raw_role)
            split_id = _split_identifier({"split_id": raw_split_id})
            if role is not None and split_id is not None:
                split_ids[role] = split_id
    for raw_role, raw_split_id in runtime.items():
        role = _role_from_key(raw_role)
        if role is None or not _slug(str(raw_role)).endswith(
            ("_split", "_split_id", "_split_identifier")
        ):
            continue
        split_id = _split_identifier({"split_id": raw_split_id})
        if split_id is not None:
            split_ids[role] = split_id
    return split_ids


def _role_from_key(value: Any) -> str | None:
    normalized = _slug(str(value))
    for suffix in ("_split_identifier", "_split_id", "_split"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return _dataset_role(normalized)


def _runtime_role_split_ids(
    runtime: Mapping[str, Any],
    role: str,
) -> set[str]:
    split_ids = {
        split_id
        for declared_role, split_id, _ in _runtime_dataset_roles(
            runtime
        ).values()
        if declared_role == role and split_id is not None
    }
    top_level = _runtime_split_ids(runtime).get(role)
    if top_level is not None:
        split_ids.add(top_level)
    return split_ids


def _runtime_role_declarations(
    runtime: Mapping[str, Any],
    role: str,
) -> list[tuple[str | None, bool | None]]:
    declarations = [
        (split_id, untouched)
        for declared_role, split_id, untouched in _runtime_dataset_roles(
            runtime
        ).values()
        if declared_role == role
    ]
    if declarations:
        return declarations
    split_id = _runtime_split_ids(runtime).get(role)
    return [(split_id, None)] if split_id is not None else []


def _runtime_role_is_untouched(
    runtime: Mapping[str, Any],
    role: str,
) -> bool:
    return any(
        declared_role == role and untouched is True
        for declared_role, _, untouched in _runtime_dataset_roles(
            runtime
        ).values()
    )


def _runtime_split_declarations_agree(
    runtime: Mapping[str, Any],
    role: str,
) -> bool:
    split_ids = {
        split_id
        for split_id, _ in _runtime_role_declarations(runtime, role)
        if split_id is not None
    }
    top_level = _runtime_split_ids(runtime).get(role)
    if top_level is not None:
        split_ids.add(top_level)
    return len(split_ids) <= 1


def _runtime_declares_dataset_contract(
    runtime: Mapping[str, Any],
) -> bool:
    return "dataset_roles" in runtime and (
        "split_identifiers" in runtime
        or "split_ids" in runtime
        or any(
            _role_from_key(field) is not None
            and _slug(str(field)).endswith(
                ("_split", "_split_id", "_split_identifier")
            )
            for field in runtime
        )
    )


def _identity(value: Any) -> str:
    return f"{type(value).__name__}:{value!r}"


def _adaptation_references(
    value: Any,
    *,
    path: str = "plan",
    active: bool = False,
) -> list[tuple[str, Any]]:
    references: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            child_active = active or _is_adaptation_key(str(key))
            references.extend(
                _adaptation_references(
                    child,
                    path=child_path,
                    active=child_active,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            references.extend(
                _adaptation_references(
                    child,
                    path=f"{path}[{index}]",
                    active=active,
                )
            )
    elif active:
        references.append((path, value))
    return references


def _is_adaptation_key(value: str) -> bool:
    normalized = _slug(value)
    return any(
        marker in normalized
        for marker in (
            "adapt",
            "calibrat",
            "fine_tun",
            "finetun",
            "memory",
            "prompt",
            "train_dataset",
            "training_dataset",
            "tuning_dataset",
            "selection_dataset",
        )
    )


def _declares_use(value: Any) -> bool:
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and _slug(value) not in {
            "false",
            "none",
            "no",
            "never",
        }
    if isinstance(value, (list, tuple, set, Mapping)):
        return bool(value)
    return bool(value)


def _mentions_adaptation(value: Any) -> bool:
    if isinstance(value, str):
        normalized = _slug(value)
        return normalized in {
            "adapt",
            "adaptation",
            "calibration",
            "training",
            "tuning",
            "fine_tuning",
            "finetuning",
            "memory",
            "memory_writing",
            "model_selection",
            "prompting",
        }
    if isinstance(value, list):
        return any(_mentions_adaptation(item) for item in value)
    return False


def _validate_decision_table(
    table: Any,
    errors: list[str],
    *,
    structured: bool,
    metric_direction: Any = None,
    decision_contract: Any = None,
) -> None:
    if not isinstance(table, list) or not table:
        errors.append("decision_table must cover every outcome region")
        return

    conditions: list[tuple[int, Mapping[str, Any]]] = []
    decisions: list[tuple[int, Mapping[str, Any], str]] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(table):
        if not isinstance(row, Mapping):
            errors.append(f"invalid decision_table[{index}]")
            continue
        condition = row.get("condition")
        parsed_condition: Any = None
        if condition is None and structured:
            direct = {
                key: child
                for key, child in row.items()
                if key not in {"decision", "reason", "criteria"}
            }
            condition = direct or None
        if condition is None or (
            isinstance(condition, str) and not condition.strip()
        ):
            errors.append(f"missing decision_table[{index}].condition")
        else:
            parsed_condition = condition
            if structured and isinstance(condition, str):
                parsed_condition = _legacy_condition_region(condition)
            signature = _condition_signature(condition)
            if signature in seen:
                errors.append(
                    f"duplicate decision_table condition at indexes "
                    f"{seen[signature]} and {index}"
                )
            else:
                seen[signature] = index
            if structured:
                if parsed_condition is None:
                    errors.append(
                        f"decision_table[{index}].condition must identify "
                        "exactly one structured outcome region for "
                        "screening_pilot"
                    )
                else:
                    conditions.append((index, parsed_condition))
        decision = row.get("decision")
        if decision not in {"promote", "retry", "reject"}:
            errors.append(f"invalid decision_table[{index}].decision")
        elif structured and isinstance(parsed_condition, Mapping):
            decisions.append((index, parsed_condition, decision))

    if structured:
        if isinstance(decision_contract, Mapping):
            _validate_contract_outcomes(conditions, errors)
            _validate_contract_decisions(decisions, errors)
        else:
            _validate_structured_outcomes(conditions, errors)
            _validate_directional_decisions(
                decisions,
                metric_direction=metric_direction,
                errors=errors,
            )


def _validate_contract_outcomes(
    conditions: list[tuple[int, Mapping[str, Any]]],
    errors: list[str],
) -> None:
    regions: dict[str, int] = {}
    for index, condition in conditions:
        region, region_error = _outcome_region(condition)
        if region_error:
            errors.append(
                f"invalid decision_table[{index}].condition: {region_error}"
            )
            continue
        if region not in _CONTRACT_OUTCOME_REGIONS:
            errors.append(
                f"invalid decision_table[{index}].condition: expected one of "
                + ", ".join(sorted(_CONTRACT_OUTCOME_REGIONS))
            )
            continue
        if region in regions:
            errors.append(
                f"duplicate decision_table outcome region {region!r} at "
                f"indexes {regions[region]} and {index}"
            )
        else:
            regions[region] = index
    missing = sorted(_CONTRACT_OUTCOME_REGIONS - set(regions))
    if missing:
        errors.append(
            "decision_table missing outcome regions: " + ", ".join(missing)
        )


def _validate_contract_decisions(
    decisions: list[tuple[int, Mapping[str, Any], str]],
    errors: list[str],
) -> None:
    expected = {
        "invalid": "retry",
        "meets_all_promotion_criteria": "promote",
        "valid_otherwise": "reject",
    }
    for index, condition, decision in decisions:
        region, region_error = _outcome_region(condition)
        if region_error or region not in expected:
            continue
        if decision != expected[region]:
            errors.append(
                f"decision_table[{index}].decision={decision!r} conflicts "
                f"with decision_contract region {region!r}: must use "
                f"{expected[region]!r}"
            )


def _validate_directional_decisions(
    decisions: list[tuple[int, Mapping[str, Any], str]],
    *,
    metric_direction: Any,
    errors: list[str],
) -> None:
    """Reject categorical decision tables that invert metric direction."""

    if metric_direction not in {"maximize", "minimize"}:
        errors.append("metric_direction must be maximize or minimize")
        return

    if metric_direction == "maximize":
        expected = {
            "invalid": "retry",
            "below_effect_threshold": "reject",
            "at_or_above_effect_threshold": "promote",
        }
    else:
        expected = {
            "invalid": "retry",
            "below_effect_threshold": "promote",
            "at_or_above_effect_threshold": "reject",
        }

    for index, condition, decision in decisions:
        region, region_error = _outcome_region(condition)
        if region_error or region is None:
            # Numeric intervals and malformed regions are handled by the
            # exhaustive-outcome validator.
            continue
        expected_decision = expected[region]
        if decision != expected_decision:
            errors.append(
                f"decision_table[{index}].decision={decision!r} conflicts "
                f"with metric_direction={metric_direction!r}: outcome "
                f"region {region!r} must use {expected_decision!r}"
            )


def _legacy_condition_region(
    condition: str,
) -> Mapping[str, str] | None:
    normalized = _slug(condition)
    invalid = any(
        marker in normalized
        for marker in (
            "invalid",
            "missing",
            "inconclusive",
            "unusable",
        )
    )
    below = any(
        marker in normalized
        for marker in (
            "below_threshold",
            "below_the_threshold",
            "does_not_meet",
            "fails_threshold",
            "subthreshold",
        )
    )
    above = any(
        marker in normalized
        for marker in (
            "above_threshold",
            "above_the_threshold",
            "at_or_above",
            "meets_threshold",
            "meets_the_preregistered_screening_threshold",
            "threshold_met",
        )
    )
    matched = [
        region
        for region, present in (
            ("invalid", invalid),
            ("below_effect_threshold", below),
            ("at_or_above_effect_threshold", above),
        )
        if present
    ]
    if len(matched) != 1:
        return None
    return {"region": matched[0]}


def _condition_signature(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return repr(value)


def _validate_structured_outcomes(
    conditions: list[tuple[int, Mapping[str, Any]]],
    errors: list[str],
) -> None:
    regions: dict[str, int] = {}
    intervals: list[tuple[float | None, float | None, bool, bool, int]] = []
    malformed = False

    for index, condition in conditions:
        region, region_error = _outcome_region(condition)
        if region_error:
            errors.append(
                f"invalid decision_table[{index}].condition: {region_error}"
            )
            malformed = True
            continue
        if region is not None:
            if region in regions:
                errors.append(
                    f"duplicate decision_table outcome region {region!r} at "
                    f"indexes {regions[region]} and {index}"
                )
            else:
                regions[region] = index
            continue
        interval, interval_error = _condition_interval(condition, index)
        if interval_error:
            errors.append(
                f"invalid decision_table[{index}].condition: "
                f"{interval_error}"
            )
            malformed = True
            continue
        if interval is None:
            errors.append(
                f"invalid decision_table[{index}].condition: expected an "
                "outcome region or numeric interval"
            )
            malformed = True
            continue
        intervals.append(interval)

    categorical_valid = set(regions) - {"invalid"}
    if intervals and categorical_valid:
        errors.append(
            "decision_table must not mix categorical effect regions with "
            "numeric intervals"
        )
        return

    if intervals:
        if "invalid" not in regions:
            errors.append("decision_table missing outcome region: invalid")
        _validate_interval_partition(intervals, errors)
        return

    missing = sorted(_OUTCOME_REGIONS - set(regions))
    if missing:
        errors.append(
            "decision_table missing outcome regions: " + ", ".join(missing)
        )
    if malformed and not conditions:
        errors.append(
            "decision_table coverage cannot be established from malformed "
            "conditions"
        )


def _outcome_region(
    condition: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    for key in ("region", "outcome_region"):
        if key in condition:
            raw = condition[key]
            if not isinstance(raw, str):
                return None, f"{key} must be a string"
            canonical = _REGION_ALIASES.get(_slug(raw))
            if canonical is None:
                return None, f"unknown outcome region {raw!r}"
            return canonical, None

    valid: bool | None = None
    for key in ("evidence_valid", "valid"):
        if key in condition:
            if not isinstance(condition[key], bool):
                return None, f"{key} must be boolean"
            valid = bool(condition[key])
            break

    status = None
    for key in ("evidence_status", "validity", "status"):
        if key in condition:
            if not isinstance(condition[key], str):
                return None, f"{key} must be a string"
            status = _slug(condition[key])
            break
    if status in {
        "invalid",
        "missing",
        "inconclusive",
        "unusable",
        "invalid_or_inconclusive",
    }:
        valid = False
    elif status in {"valid", "usable", "complete"}:
        valid = True
    elif status is not None:
        return None, f"unknown evidence status {status!r}"

    if valid is False:
        return "invalid", None

    effect_region = None
    for key in ("effect_region", "threshold_region"):
        if key in condition:
            if not isinstance(condition[key], str):
                return None, f"{key} must be a string"
            effect_region = _REGION_ALIASES.get(_slug(condition[key]))
            if effect_region is None or effect_region == "invalid":
                return None, f"unknown effect region {condition[key]!r}"
            break
    if effect_region is not None:
        return effect_region, None

    for key in ("effect_threshold_met", "threshold_met"):
        if key in condition:
            if not isinstance(condition[key], bool):
                return None, f"{key} must be boolean"
            return (
                "at_or_above_effect_threshold"
                if condition[key]
                else "below_effect_threshold"
            ), None

    if valid is True:
        return None, "valid evidence condition must define its effect region"
    return None, None


def _condition_interval(
    condition: Mapping[str, Any],
    index: int,
) -> tuple[
    tuple[float | None, float | None, bool, bool, int] | None,
    str | None,
]:
    operator = condition.get("operator", condition.get("op"))
    if operator is not None:
        if not isinstance(operator, str):
            return None, "operator must be a string"
        raw_bound = condition.get("value", condition.get("threshold"))
        bound, error = _finite_bound(raw_bound, "value")
        if error:
            return None, error
        normalized = operator.strip().casefold()
        if normalized in {"<", "lt", "less_than"}:
            return (None, bound, False, False, index), None
        if normalized in {"<=", "le", "lte", "at_most"}:
            return (None, bound, False, True, index), None
        if normalized in {">", "gt", "greater_than"}:
            return (bound, None, False, False, index), None
        if normalized in {">=", "ge", "gte", "at_least"}:
            return (bound, None, True, False, index), None
        if normalized in {"==", "eq", "equal"}:
            return (bound, bound, True, True, index), None
        return None, f"unsupported operator {operator!r}"

    lower_keys = ("lower", "min", "minimum", "lower_bound")
    upper_keys = ("upper", "max", "maximum", "upper_bound")
    has_lower = any(key in condition for key in lower_keys)
    has_upper = any(key in condition for key in upper_keys)
    if not has_lower and not has_upper:
        return None, None
    raw_lower = next(
        (condition[key] for key in lower_keys if key in condition),
        None,
    )
    raw_upper = next(
        (condition[key] for key in upper_keys if key in condition),
        None,
    )
    lower, error = _finite_bound(raw_lower, "lower", unbounded=True)
    if error:
        return None, error
    upper, error = _finite_bound(raw_upper, "upper", unbounded=True)
    if error:
        return None, error

    lower_inclusive = condition.get("lower_inclusive", lower is not None)
    upper_inclusive = condition.get("upper_inclusive", False)
    if not isinstance(lower_inclusive, bool):
        return None, "lower_inclusive must be boolean"
    if not isinstance(upper_inclusive, bool):
        return None, "upper_inclusive must be boolean"
    if lower is None:
        lower_inclusive = False
    if upper is None:
        upper_inclusive = False
    if lower is not None and upper is not None:
        if lower > upper:
            return None, "lower bound exceeds upper bound"
        if lower == upper and not (lower_inclusive and upper_inclusive):
            return None, "interval is empty"
    return (
        lower,
        upper,
        lower_inclusive,
        upper_inclusive,
        index,
    ), None


def _finite_bound(
    value: Any,
    name: str,
    *,
    unbounded: bool = False,
) -> tuple[float | None, str | None]:
    if value is None and unbounded:
        return None, None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        return None, f"{name} must be a finite number"
    return float(value), None


def _validate_interval_partition(
    intervals: list[tuple[float | None, float | None, bool, bool, int]],
    errors: list[str],
) -> None:
    ordered = sorted(
        intervals,
        key=lambda item: (
            float("-inf") if item[0] is None else item[0],
            item[4],
        ),
    )
    first = ordered[0]
    if first[0] is not None:
        errors.append(
            "decision_table numeric intervals have a gap before "
            f"index {first[4]}"
        )

    for prior, current in pairwise(ordered):
        prior_upper = prior[1]
        current_lower = current[0]
        if prior_upper is None:
            errors.append(
                "decision_table numeric intervals overlap at indexes "
                f"{prior[4]} and {current[4]}"
            )
            continue
        if current_lower is None:
            errors.append(
                "decision_table numeric intervals overlap at indexes "
                f"{prior[4]} and {current[4]}"
            )
            continue
        equal = math.isclose(
            prior_upper,
            current_lower,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        if prior_upper < current_lower and not equal:
            errors.append(
                "decision_table numeric intervals have a gap between indexes "
                f"{prior[4]} and {current[4]}"
            )
        elif prior_upper > current_lower and not equal:
            errors.append(
                "decision_table numeric intervals overlap at indexes "
                f"{prior[4]} and {current[4]}"
            )
        elif prior[3] and current[2]:
            errors.append(
                "decision_table numeric intervals overlap at their boundary "
                f"between indexes {prior[4]} and {current[4]}"
            )
        elif not prior[3] and not current[2]:
            errors.append(
                "decision_table numeric intervals have an uncovered boundary "
                f"between indexes {prior[4]} and {current[4]}"
            )

    last = ordered[-1]
    if last[1] is not None:
        errors.append(
            "decision_table numeric intervals have a gap after "
            f"index {last[4]}"
        )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _deduplicate(errors: list[str]) -> list[str]:
    return list(dict.fromkeys(errors))


def validate_build_output(value: dict[str, Any]) -> list[str]:
    reserved = sorted(
        {
            "remote_smoke",
            "execution_attestation",
            "execution_contract",
            "attestation_sha256",
            "contract_sha256",
        }.intersection(value)
    )
    if reserved:
        return [
            "Controller-owned build fields are forbidden: "
            + ", ".join(reserved)
        ]
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        return ["files must be a non-empty object"]
    errors: list[str] = []
    for filename, content in files.items():
        if (
            not isinstance(filename, str)
            or filename.startswith("/")
            or ".." in Path(filename).parts
        ):
            errors.append(f"invalid filename {filename!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"empty file {filename!r}")
    commands = value.get("commands")
    if not isinstance(commands, dict):
        errors.append("commands must be an object")
    else:
        for field in ("smoke", "pilot", "scale"):
            command = commands.get(field)
            if isinstance(command, str):
                try:
                    argv = shlex.split(command)
                except ValueError as exc:
                    errors.append(f"invalid commands.{field}: {exc}")
                    continue
            elif isinstance(command, list) and all(
                isinstance(item, str) and item
                for item in command
            ):
                argv = list(command)
            else:
                argv = []
            if not argv:
                errors.append(f"missing commands.{field}")
                continue
            errors.extend(
                validate_execution_argv(
                    argv,
                    path=f"commands.{field}",
                )
            )
    return errors


def validate_execution_argv(
    argv: list[str] | tuple[str, ...],
    *,
    path: str = "command",
) -> list[str]:
    """Allow only a direct Python entrypoint with inert string arguments."""

    errors: list[str] = []
    if not argv:
        return [f"{path} must be a non-empty argv list"]
    executable = Path(argv[0]).name.casefold()
    if executable not in {
        "python",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
    }:
        errors.append(
            f"{path} executable must be an approved Python interpreter"
        )
    if len(argv) < 2:
        errors.append(f"{path} must include a Python entrypoint")
        return errors
    entrypoint = Path(argv[1])
    if (
        entrypoint.is_absolute()
        or ".." in entrypoint.parts
        or entrypoint.suffix != ".py"
    ):
        errors.append(
            f"{path} entrypoint must be a relative .py file without '..'"
        )
    for index, argument in enumerate(argv):
        if "\x00" in argument or "\n" in argument or "\r" in argument:
            errors.append(f"{path}[{index}] contains a control character")
        if any(
            token in argument
            for token in (";", "&&", "||", "|", ">", "<", "`", "$(")
        ):
            errors.append(f"{path}[{index}] contains a shell metacharacter")
    return errors


def validate_metrics_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"metrics unreadable: {exc}"]}
    errors: list[str] = []
    if not isinstance(value, dict):
        errors.append("metrics root must be an object")
        value = {}
    if value.get("result_valid") is not True:
        errors.append("result_valid must be true")
    if not isinstance(value.get("metrics"), dict) or not value.get("metrics"):
        errors.append("finite metrics object is required")
    else:
        for name, metric in value["metrics"].items():
            if isinstance(metric, bool) or not isinstance(
                metric,
                (int, float),
            ):
                errors.append(f"metric {name} must be numeric")
            elif not math.isfinite(float(metric)):
                errors.append(f"metric {name} is not finite")
    return {"ok": not errors, "errors": errors, "metrics": value}


def validate_runtime_evidence_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "errors": [f"runtime evidence unreadable: {exc}"],
            "evidence": {},
        }
    if not isinstance(value, dict):
        return {
            "ok": False,
            "errors": ["runtime evidence root must be an object"],
            "evidence": {},
        }
    errors: list[str] = []
    required = (
        "model_loaded",
        "datasets_loaded",
        "examples_processed",
        "seeds",
        "gpu_count",
        "gate_decision",
        "metrics",
    )
    for field in required:
        if field not in value:
            errors.append(f"missing runtime evidence {field}")
    if not str(value.get("model_loaded", "") or "").strip():
        errors.append("model_loaded must identify a real model")
    if not isinstance(value.get("datasets_loaded"), list) or not value.get(
        "datasets_loaded"
    ):
        errors.append("datasets_loaded must be a non-empty list")
    for field in ("examples_processed", "gpu_count"):
        try:
            number = int(value.get(field, -1))
        except (TypeError, ValueError):
            errors.append(f"invalid runtime evidence {field}")
        else:
            if number < 0:
                errors.append(f"invalid runtime evidence {field}")
    seeds = value.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("seeds must be a non-empty list")
    elif any(isinstance(seed, (list, dict, set)) for seed in seeds):
        errors.append("seeds must contain scalar identifiers")
    elif len({_identity(seed) for seed in seeds}) != len(seeds):
        errors.append("seeds must contain unique independent identifiers")

    if "dataset_roles" in value:
        dataset_roles = value["dataset_roles"]
        if (
            not isinstance(dataset_roles, (Mapping, list))
            or not dataset_roles
        ):
            errors.append(
                "dataset_roles must be a non-empty object or list"
            )
        else:
            parsed_roles = _runtime_dataset_roles(value)
            if not parsed_roles:
                errors.append("dataset_roles contains no valid declarations")
            for dataset, (role, split_id, untouched) in parsed_roles.items():
                if role is None:
                    errors.append(
                        f"invalid dataset_roles declaration for {dataset!r}"
                    )
                if role == "confirmatory" and untouched is not True:
                    errors.append(
                        f"dataset_roles[{dataset!r}].untouched must be true "
                        "for confirmatory data"
                    )
    for split_field in ("split_identifiers", "split_ids"):
        if split_field not in value:
            continue
        split_ids = value[split_field]
        if not isinstance(split_ids, Mapping) or not split_ids:
            errors.append(f"{split_field} must be a non-empty object")
        elif not _runtime_split_ids({split_field: split_ids}):
            errors.append(
                f"{split_field} contains no valid role/split declarations"
            )
    if "dataset_roles" in value:
        top_level_split_ids = _runtime_split_ids(value)
        for dataset, (role, split_id, _) in _runtime_dataset_roles(
            value
        ).items():
            if (
                role is not None
                and split_id is None
                and role not in top_level_split_ids
            ):
                errors.append(
                    f"dataset_roles[{dataset!r}] must declare split_id"
                )
    if "dataset_roles" in value and not (
        "split_identifiers" in value
        or "split_ids" in value
        or any(
            _role_from_key(field) is not None
            and _slug(str(field)).endswith(
                ("_split", "_split_id", "_split_identifier")
            )
            for field in value
        )
    ):
        errors.append(
            "dataset_roles requires split identifiers for declared roles"
        )
    if "call_counts" in value:
        call_counts = value["call_counts"]
        if not isinstance(call_counts, Mapping) or not call_counts:
            errors.append("call_counts must be a non-empty object")
        else:
            for name, count in call_counts.items():
                if (
                    not isinstance(name, str)
                    or not name.strip()
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                ):
                    errors.append(
                        "call_counts must map non-empty component names "
                        "to non-negative integers"
                    )
                    break
    if "examples_by_role" in value:
        examples_by_role = value["examples_by_role"]
        if not isinstance(examples_by_role, Mapping) or not examples_by_role:
            errors.append("examples_by_role must be a non-empty object")
        else:
            for role, count in examples_by_role.items():
                normalized_role = _dataset_role(role)
                if (
                    normalized_role not in {
                        "development",
                        "screening",
                        "confirmatory",
                    }
                    or isinstance(count, bool)
                    or not isinstance(count, int)
                    or count < 0
                ):
                    errors.append(
                        "examples_by_role must map valid dataset roles "
                        "to non-negative integers"
                    )
                    break
    if "evidence_valid" in value and not isinstance(
        value["evidence_valid"],
        bool,
    ):
        errors.append("evidence_valid must be boolean")
    if "gate_statistic_defined" in value and not isinstance(
        value["gate_statistic_defined"],
        bool,
    ):
        errors.append("gate_statistic_defined must be boolean")
    if "criterion_results" in value:
        criterion_results = value["criterion_results"]
        if not isinstance(criterion_results, Mapping) or not criterion_results:
            errors.append("criterion_results must be a non-empty object")
        else:
            for criterion_id, result in criterion_results.items():
                if (
                    not isinstance(criterion_id, str)
                    or not criterion_id.strip()
                    or not isinstance(result, Mapping)
                    or not isinstance(result.get("passed"), bool)
                    or isinstance(result.get("value"), bool)
                    or not isinstance(result.get("value"), (int, float))
                    or not math.isfinite(float(result.get("value")))
                ):
                    errors.append(
                        "criterion_results must map criterion ids to "
                        "{value: finite number, passed: boolean}"
                    )
                    break
    if not isinstance(value.get("metrics"), dict) or not value.get("metrics"):
        errors.append("runtime evidence metrics must be a non-empty object")
    else:
        for name, metric in value["metrics"].items():
            if isinstance(metric, bool) or not isinstance(
                metric,
                (int, float),
            ):
                errors.append(
                    f"runtime evidence metric {name} must be numeric"
                )
            elif not math.isfinite(float(metric)):
                errors.append(
                    f"runtime evidence metric {name} is not finite"
                )
    decision = str(value.get("gate_decision", "") or "").casefold()
    if decision not in {
        "promote",
        "continue",
        "reject",
        "stop",
        "complete_negative",
    }:
        errors.append("invalid gate_decision")
    return {
        "ok": not errors,
        "errors": errors,
        "evidence": value,
    }


def validate_experiment_artifacts(output_dir: Path) -> dict[str, Any]:
    metrics = validate_metrics_file(output_dir / "metrics.json")
    runtime = validate_runtime_evidence_file(
        output_dir / "runtime_evidence.json"
    )
    errors = [
        *metrics.get("errors", []),
        *runtime.get("errors", []),
    ]
    metrics_payload = metrics.get("metrics", {})
    measured = (
        metrics_payload.get("metrics", {})
        if isinstance(metrics_payload, dict)
        else {}
    )
    reported = runtime.get("evidence", {}).get("metrics", {})
    if (
        isinstance(measured, dict)
        and isinstance(reported, dict)
        and measured != reported
    ):
        errors.append(
            "metrics.json and runtime_evidence.json metrics disagree"
        )
    return {
        "ok": not errors,
        "errors": errors,
        "metrics": metrics.get("metrics", {}),
        "runtime_evidence": runtime.get("evidence", {}),
        "files": [
            str(path.relative_to(output_dir))
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        ]
        if output_dir.is_dir()
        else [],
    }


def validate_runtime_against_contract(
    *,
    plan: dict[str, Any],
    runtime_evidence: dict[str, Any],
    allocated_gpus: int,
    mode: str,
    pilot_runtime: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        reported_gpus = int(runtime_evidence.get("gpu_count", -1))
    except (TypeError, ValueError):
        reported_gpus = -1
    if reported_gpus != int(allocated_gpus):
        errors.append(
            f"reported gpu_count={reported_gpus} does not match "
            f"allocation={allocated_gpus}"
        )
    pilot = plan.get("pilot", {})
    if mode == "pilot" and isinstance(pilot, dict):
        for evidence_field, plan_field in (
            ("examples_processed", "max_examples"),
        ):
            try:
                actual = int(runtime_evidence.get(evidence_field, -1))
                maximum = int(pilot.get(plan_field, -1))
            except (TypeError, ValueError):
                continue
            if maximum > 0 and actual > maximum:
                errors.append(
                    f"{evidence_field}={actual} exceeds pilot {plan_field}={maximum}"
                )
        seeds = runtime_evidence.get("seeds", [])
        try:
            max_seeds = int(pilot.get("max_seeds", -1))
        except (TypeError, ValueError):
            max_seeds = -1
        if (
            isinstance(seeds, list)
            and max_seeds > 0
            and len(seeds) > max_seeds
        ):
            errors.append(
                f"seed count={len(seeds)} exceeds pilot max_seeds={max_seeds}"
            )
        _validate_runtime_call_ledger(
            plan=plan,
            runtime_evidence=runtime_evidence,
            errors=errors,
        )
        if isinstance(plan.get("call_ledger"), Mapping):
            examples_by_role = runtime_evidence.get("examples_by_role")
            if not isinstance(examples_by_role, Mapping):
                errors.append(
                    "runtime_evidence.examples_by_role is required by "
                    "compiled protocol"
                )
            else:
                for role, maximum_field in (
                    ("development", "development_examples"),
                    ("screening", "max_examples"),
                ):
                    try:
                        actual = int(examples_by_role.get(role, -1))
                        maximum = int(pilot.get(maximum_field, -1))
                    except (TypeError, ValueError):
                        continue
                    if maximum >= 0 and actual > maximum:
                        errors.append(
                            f"examples_by_role[{role}]={actual} exceeds "
                            f"pilot {maximum_field}={maximum}"
                        )
    if mode in {"pilot", "scale"}:
        _validate_runtime_decision_contract(
            plan=plan,
            runtime_evidence=runtime_evidence,
            errors=errors,
        )
    if mode == "scale" and pilot_runtime:
        try:
            scale_examples = int(
                runtime_evidence.get("examples_processed", -1)
            )
            pilot_examples = int(
                pilot_runtime.get("examples_processed", -1)
            )
        except (TypeError, ValueError):
            scale_examples = pilot_examples = -1
        scale_seeds = runtime_evidence.get("seeds", [])
        prior_seeds = pilot_runtime.get("seeds", [])
        scale_seed_ids = (
            {_identity(seed) for seed in scale_seeds}
            if isinstance(scale_seeds, list)
            else set()
        )
        pilot_seed_ids = (
            {_identity(seed) for seed in prior_seeds}
            if isinstance(prior_seeds, list)
            else set()
        )
        if scale_examples <= pilot_examples:
            errors.append(
                "scale run must increase examples beyond pilot "
                f"({scale_examples} <= {pilot_examples})"
            )
        if len(scale_seed_ids) <= len(pilot_seed_ids):
            errors.append(
                "scale run must increase independent seed coverage beyond "
                f"pilot ({len(scale_seed_ids)} <= {len(pilot_seed_ids)})"
            )

        production = _is_production_screening_plan(plan)
        if production:
            _validate_scale_dataset_contract(
                plan=plan,
                runtime_evidence=runtime_evidence,
                pilot_runtime=pilot_runtime,
                errors=errors,
            )
    return errors


def _validate_runtime_decision_contract(
    *,
    plan: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    errors: list[str],
) -> None:
    contract = plan.get("decision_contract")
    if not isinstance(contract, Mapping):
        return
    evidence_valid = runtime_evidence.get("evidence_valid")
    if not isinstance(evidence_valid, bool):
        errors.append(
            "runtime_evidence.evidence_valid is required by decision_contract"
        )
        return
    gate_defined = runtime_evidence.get("gate_statistic_defined")
    if not isinstance(gate_defined, bool):
        errors.append(
            "runtime_evidence.gate_statistic_defined is required by "
            "decision_contract"
        )
    results = runtime_evidence.get("criterion_results")
    if not isinstance(results, Mapping):
        errors.append(
            "runtime_evidence.criterion_results is required by "
            "decision_contract"
        )
        return
    expected: dict[str, Mapping[str, Any]] = {}
    for field in ("validity_criteria", "promotion_criteria"):
        criteria = plan.get(field)
        if not isinstance(criteria, list):
            continue
        for criterion in criteria:
            if isinstance(criterion, Mapping):
                criterion_id = str(criterion.get("id", "") or "")
                if criterion_id:
                    expected[criterion_id] = criterion
    missing = sorted(set(expected) - {str(key) for key in results})
    if missing:
        errors.append(
            "runtime_evidence.criterion_results missing criteria: "
            + ", ".join(missing)
        )
    parsed: dict[str, bool] = {}
    for criterion_id, criterion in expected.items():
        result = results.get(criterion_id)
        if not isinstance(result, Mapping):
            continue
        passed = result.get("passed")
        measured = result.get("value")
        if not isinstance(passed, bool):
            errors.append(
                f"criterion_results[{criterion_id}].passed must be boolean"
            )
            continue
        if (
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or not math.isfinite(float(measured))
        ):
            errors.append(
                f"criterion_results[{criterion_id}].value must be finite"
            )
            continue
        recomputed = _criterion_passes(criterion, float(measured))
        if recomputed is not passed:
            errors.append(
                f"criterion_results[{criterion_id}].passed disagrees with "
                "the preregistered operator/value"
            )
        parsed[criterion_id] = passed
    validity_ids = [
        str(item.get("id"))
        for item in plan.get("validity_criteria", [])
        if isinstance(item, Mapping)
    ]
    promotion_ids = [
        str(item.get("id"))
        for item in plan.get("promotion_criteria", [])
        if isinstance(item, Mapping)
    ]
    validity_pass = all(parsed.get(item) is True for item in validity_ids)
    promotion_pass = all(
        parsed.get(item) is True for item in promotion_ids
    )
    if evidence_valid is not validity_pass:
        errors.append(
            "runtime_evidence.evidence_valid disagrees with validity_criteria"
        )
    decision = str(runtime_evidence.get("gate_decision", "") or "").casefold()
    if not evidence_valid:
        expected_decision = "retry"
    elif gate_defined is False:
        expected_decision = "reject"
    elif promotion_pass:
        expected_decision = "promote"
    else:
        expected_decision = "reject"
    normalized = {
        "continue": "promote",
        "stop": "reject",
    }.get(decision, decision)
    if normalized != expected_decision:
        errors.append(
            f"runtime gate_decision={decision!r} conflicts with compiled "
            f"decision_contract; expected {expected_decision!r}"
        )
    gate = plan.get("gate_statistic")
    if isinstance(gate, Mapping):
        gate_name = str(gate.get("name", "") or "")
        metrics = runtime_evidence.get("metrics")
        if (
            gate_defined is True
            and gate_name
            and (
                not isinstance(metrics, Mapping)
                or gate_name not in metrics
            )
        ):
            errors.append(
                "runtime metrics missing defined gate statistic "
                f"{gate_name!r}"
            )


def _criterion_passes(
    criterion: Mapping[str, Any],
    measured: float,
) -> bool:
    threshold = float(criterion.get("value"))
    operator = criterion.get("operator")
    if operator == "<":
        return measured < threshold
    if operator == "<=":
        return measured <= threshold
    if operator == ">":
        return measured > threshold
    if operator == ">=":
        return measured >= threshold
    return math.isclose(
        measured,
        threshold,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def _validate_runtime_call_ledger(
    *,
    plan: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    errors: list[str],
) -> None:
    ledger = plan.get("call_ledger")
    if not isinstance(ledger, Mapping):
        return
    expected_components = ledger.get("components")
    if not isinstance(expected_components, list):
        return
    actual = runtime_evidence.get("call_counts")
    if not isinstance(actual, Mapping):
        errors.append(
            "runtime_evidence.call_counts is required by compiled protocol"
        )
        return
    expected: dict[str, int] = {}
    for component in expected_components:
        if not isinstance(component, Mapping):
            continue
        name = str(component.get("name", "") or "")
        if not name:
            continue
        try:
            expected[name] = expected.get(name, 0) + int(
                component.get("total_calls", -1)
            )
        except (TypeError, ValueError):
            continue
    actual_names = {
        str(name)
        for name, count in actual.items()
        if isinstance(name, str)
        and not isinstance(count, bool)
        and isinstance(count, int)
        and count >= 0
    }
    missing = sorted(set(expected) - actual_names)
    if missing:
        errors.append(
            "runtime_evidence.call_counts missing components: "
            + ", ".join(missing)
        )
    try:
        actual_total = sum(int(actual[name]) for name in expected)
        maximum = int(ledger.get("total_model_calls", -1))
    except (KeyError, TypeError, ValueError):
        return
    if maximum >= 0 and actual_total > maximum:
        errors.append(
            f"runtime model calls={actual_total} exceed compiled "
            f"call_ledger.total_model_calls={maximum}"
        )
    for name, maximum_component_calls in expected.items():
        try:
            actual_component_calls = int(actual[name])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            maximum_component_calls >= 0
            and actual_component_calls > maximum_component_calls
        ):
            errors.append(
                f"runtime call_counts[{name}]={actual_component_calls} "
                f"exceed compiled component budget="
                f"{maximum_component_calls}"
            )


def _is_production_screening_plan(plan: Mapping[str, Any]) -> bool:
    if plan.get("study_phase") == _SCREENING_PHASE:
        return True
    followup = plan.get("confirmatory_followup")
    if not isinstance(followup, Mapping):
        return False
    return (
        followup.get("required") is True
        or _split_identifier(followup) is not None
        or followup.get("untouched") is True
        or followup.get("split_untouched") is True
    )


def _validate_scale_dataset_contract(
    *,
    plan: Mapping[str, Any],
    runtime_evidence: Mapping[str, Any],
    pilot_runtime: Mapping[str, Any],
    errors: list[str],
) -> None:
    followup = plan.get("confirmatory_followup")
    if not isinstance(followup, Mapping):
        errors.append(
            "production scale requires structured "
            "plan.confirmatory_followup"
        )
        return

    planned_confirmatory_split = _split_identifier(followup)
    if planned_confirmatory_split is None:
        errors.append(
            "production scale plan is missing confirmatory split_id"
        )
    if followup.get(
        "untouched",
        followup.get("split_untouched"),
    ) is not True:
        errors.append(
            "production scale plan must mark confirmatory split untouched"
        )

    pilot_screening_splits = _runtime_role_split_ids(
        pilot_runtime,
        "screening",
    )
    if not pilot_screening_splits:
        pilot_screening_splits = _plan_split_ids(plan, role="screening")
    if not pilot_screening_splits:
        errors.append(
            "pilot runtime must identify the screening split for "
            "production scale"
        )

    if not _runtime_declares_dataset_contract(runtime_evidence):
        errors.append(
            "production scale runtime_evidence must declare dataset_roles "
            "and split identifiers"
        )

    scale_confirmatory_splits = _runtime_role_split_ids(
        runtime_evidence,
        "confirmatory",
    )
    confirmatory_declarations = _runtime_role_declarations(
        runtime_evidence,
        "confirmatory",
    )
    if not scale_confirmatory_splits:
        errors.append(
            "scale runtime_evidence must identify a confirmatory split"
        )
    if not _runtime_role_is_untouched(
        runtime_evidence,
        "confirmatory",
    ):
        errors.append(
            "scale runtime_evidence must mark confirmatory data untouched"
        )
    if confirmatory_declarations and any(
        untouched is not True
        for _, untouched in confirmatory_declarations
    ):
        errors.append(
            "every confirmatory dataset role must be explicitly untouched"
        )
    if not _runtime_split_declarations_agree(
        runtime_evidence,
        "confirmatory",
    ):
        errors.append(
            "scale runtime confirmatory split declarations disagree"
        )

    reused = pilot_screening_splits & scale_confirmatory_splits
    if reused:
        errors.append(
            "scale confirmatory split must differ from pilot screening split: "
            + ", ".join(sorted(reused))
        )
    if (
        planned_confirmatory_split is not None
        and scale_confirmatory_splits
        and planned_confirmatory_split not in scale_confirmatory_splits
    ):
        errors.append(
            "scale runtime confirmatory split does not match "
            "plan.confirmatory_followup.split_id"
        )
    if (
        planned_confirmatory_split is not None
        and any(
            split_id is not None
            and split_id != planned_confirmatory_split
            for split_id, _ in confirmatory_declarations
        )
    ):
        errors.append(
            "every confirmatory dataset role must use the preregistered "
            "confirmatory split"
        )
    plan_confirmatory_splits = _plan_split_ids(plan, role="confirmatory")
    if (
        planned_confirmatory_split is not None
        and plan_confirmatory_splits
        and planned_confirmatory_split not in plan_confirmatory_splits
    ):
        errors.append(
            "plan.confirmatory_followup.split_id does not match the "
            "confirmatory dataset split"
        )
