"""Deterministic candidate and runtime validation for v2 jobs."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import py_compile
import re
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
_OUTCOME_REGIONS = frozenset(
    {
        "invalid",
        "below_effect_threshold",
        "at_or_above_effect_threshold",
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
    if isinstance(workload, dict):
        for field in (
            "conditions",
            "models",
            "examples",
            "seeds",
            "max_new_tokens",
            "estimated_model_calls",
        ):
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
        )

    _validate_decision_table(
        value.get("decision_table"),
        errors,
        structured=phase_aware,
    )
    return _deduplicate(errors)


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
    _validate_confirmatory_followup(value.get("confirmatory_followup"), errors)

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
        for field in (
            "arms",
            "examples_per_arm",
            "seeds",
            "calls_per_example",
            "total_model_calls",
        ):
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
    )
    _validate_effect_threshold(value, accounting_numbers, errors)
    _validate_dataset_isolation(value, errors)


def _validate_confirmatory_followup(
    followup: Any,
    errors: list[str],
) -> None:
    if followup is None:
        errors.append("missing confirmatory_followup")
        return
    if isinstance(followup, str):
        if not followup.strip():
            errors.append(
                "invalid confirmatory_followup: must be non-empty"
            )
        return
    if not isinstance(followup, Mapping):
        errors.append(
            "invalid confirmatory_followup: must be a string or object"
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


def _validate_workload_arithmetic(
    accounting: Mapping[str, int],
    workload: Mapping[str, int],
    pilot: Mapping[str, int],
    errors: list[str],
) -> None:
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
                f"{estimated} does not equal conditions * models * examples "
                f"* seeds={expected}"
            )

    for accounting_field, workload_field in (
        ("arms", "conditions"),
        ("examples_per_arm", "examples"),
        ("seeds", "seeds"),
        ("calls_per_example", "models"),
        ("total_model_calls", "estimated_model_calls"),
    ):
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
) -> None:
    if not isinstance(table, list) or not table:
        errors.append("decision_table must cover every outcome region")
        return

    conditions: list[tuple[int, Mapping[str, Any]]] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(table):
        if not isinstance(row, Mapping):
            errors.append(f"invalid decision_table[{index}]")
            continue
        condition = row.get("condition")
        if condition is None and structured:
            direct = {
                key: child
                for key, child in row.items()
                if key not in {"decision", "reason"}
            }
            condition = direct or None
        if condition is None or (
            isinstance(condition, str) and not condition.strip()
        ):
            errors.append(f"missing decision_table[{index}].condition")
        else:
            parsed_condition: Any = condition
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
        if row.get("decision") not in {"promote", "retry", "reject"}:
            errors.append(f"invalid decision_table[{index}].decision")

    if structured:
        _validate_structured_outcomes(conditions, errors)


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
            if not str(commands.get(field, "") or "").strip():
                errors.append(f"missing commands.{field}")
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
        if (
            scale_examples <= pilot_examples
            and len(scale_seeds if isinstance(scale_seeds, list) else [])
            <= len(prior_seeds if isinstance(prior_seeds, list) else [])
        ):
            errors.append(
                "scale run did not increase examples or seed coverage"
            )
    return errors
