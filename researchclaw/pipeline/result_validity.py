"""Fail-closed validity checks for production experiment results.

The pipeline accepts metrics from several execution backends and historical
schemas.  A clean process exit is therefore not sufficient evidence that an
experiment really ran: generated code can catch an exception, print finite
placeholder metrics, and still exit with return code zero.

This module centralizes the checks used by execution, analysis, repair, and
diagnosis.  Metric values such as zero accuracy or zero effect size are not
failures by themselves.  Explicit execution-failure evidence is what makes a
result invalid.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

OVERALL_CONDITION = "__overall__"

_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SEED_TOKEN_RE = re.compile(r"^(?:seed[_-]?)?(\d+)$", re.IGNORECASE)
_CONDITION_RE = re.compile(r"\bcondition\s*=\s*([^\s,;]+)", re.IGNORECASE)
_SEED_RE = re.compile(r"\bseed\s*=\s*([^\s,;]+)", re.IGNORECASE)
_METRIC_VALUE_RE = re.compile(
    rf"\b([A-Za-z_][\w.]*)\s*:\s*({_FLOAT_RE})(?:\s|$)"
)
_SUCCESS_RATE_RATIO_RE = re.compile(
    r"\bsuccess_rate\s*[:=]\s*(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
_SUCCESS_RATE_VALUE_RE = re.compile(
    rf"\b[\w.-]*success_rate\s*[:=]\s*({_FLOAT_RE})",
    re.IGNORECASE,
)
_STEPS_COMPLETED_RE = re.compile(
    rf"\b(?:pipeline_)?steps_completed\s*[:=]\s*({_FLOAT_RE})",
    re.IGNORECASE,
)
_SUCCESSFUL_SEEDS_RE = re.compile(
    rf"\b(?:successful|succeeded|completed)[_-]?seeds?"
    rf"(?:[_-]?(?:count|total))?\s*[:=]\s*({_FLOAT_RE})",
    re.IGNORECASE,
)

_CONDITION_CONTAINERS = frozenset(
    {"conditions", "per_condition", "condition_summaries", "condition_metrics"}
)
_FAILED_STATUS_VALUES = frozenset(
    {"failed", "failure", "error", "crashed", "crash", "invalid"}
)
_PARTIAL_STATUS_VALUES = frozenset({"partial", "timed_out", "timeout"})

_SUCCESS_RATE_NAMES = frozenset(
    {
        "success_rate",
        "seed_success_rate",
        "seeds_success_rate",
        "run_success_rate",
    }
)
_STEPS_COMPLETED_NAMES = frozenset(
    {
        "steps_completed",
        "pipeline_steps_completed",
        "completed_steps",
        "num_steps_completed",
        "n_steps_completed",
    }
)
_SUCCESSFUL_SEED_COUNT_NAMES = frozenset(
    {
        "successful_seed",
        "successful_seeds",
        "successful_seed_count",
        "successful_seeds_count",
        "num_successful_seeds",
        "n_successful_seeds",
        "succeeded_seed",
        "succeeded_seeds",
        "completed_seed",
        "completed_seeds",
    }
)
_METADATA_METRIC_NAMES = frozenset(
    {
        "returncode",
        "elapsed_sec",
        "time_budget_sec",
        "timed_out",
        "nan_count",
        "seed_count",
        "n_seeds",
        "num_seeds",
        "seeds_run",
        "total_seeds",
        "pipeline_steps_failed",
        "figures_produced",
        "scripts_generated",
        "models_generated",
        "data_files_produced",
    }
)

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE)
_EXPLICIT_FAILURE_RES = (
    re.compile(r"(?:^|\n)\s*FAIL:", re.IGNORECASE),
    re.compile(r"\bNaN/divergence\b", re.IGNORECASE),
)
_DATASET_FAILURE_RES = (
    re.compile(r"\bDatasetNotFoundError\b", re.IGNORECASE),
    re.compile(
        r"\bdataset\b.{0,120}\b(?:missing|not found|unavailable|"
        r"does(?:n't| not) exist|cannot be accessed|failed to load)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:missing|not found|unavailable)\b.{0,80}\bdataset\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bFileNotFoundError\b.{0,160}\b(?:dataset|datasets|data/)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
_SYNTHETIC_FALLBACK_RES = (
    re.compile(r"\busing synthetic data\b", re.IGNORECASE),
    re.compile(r"\bsynthetic(?:/random)?\s+data\s+fallback\b", re.IGNORECASE),
    re.compile(
        r"\b(?:dataset|data)\b.{0,120}\b(?:failed|missing|unavailable)\b"
        r".{0,120}\b(?:synthetic|random)\b.{0,40}\b(?:fallback|instead)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\bfall(?:ing)?\s+back\s+to\s+(?:synthetic|random)\s+data\b",
        re.IGNORECASE,
    ),
)


@dataclass
class ProductionResultValidity:
    """Result of checking one production experiment execution."""

    valid: bool
    successful: bool
    valid_metrics: dict[str, float] = field(default_factory=dict)
    valid_conditions: set[str] = field(default_factory=set)
    invalid_conditions: dict[str, list[str]] = field(default_factory=dict)
    successful_seed_count: int = 0
    seed_evidence_present: bool = False
    reasons: list[str] = field(default_factory=list)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalized_name(name: Any) -> str:
    text = str(name).strip().replace("-", "_").replace(".", "_")
    return re.sub(r"_+", "_", text.casefold())


def _leaf_name(key: Any) -> str:
    return _normalized_name(str(key).split("/")[-1])


def _is_success_rate(name: str) -> bool:
    return name in _SUCCESS_RATE_NAMES or name.endswith("_success_rate")


def _is_steps_completed(name: str) -> bool:
    return name in _STEPS_COMPLETED_NAMES or name.endswith("_steps_completed")


def _is_successful_seed_count(name: str) -> bool:
    return name in _SUCCESSFUL_SEED_COUNT_NAMES


def _is_control_metric(name: str) -> bool:
    return (
        _is_success_rate(name)
        or _is_steps_completed(name)
        or _is_successful_seed_count(name)
        or name in _METADATA_METRIC_NAMES
        or name.endswith("_agent_success")
    )


def _metric_condition(key: str) -> str | None:
    parts = [part for part in str(key).split("/") if part]
    if len(parts) < 2:
        return None
    if _normalized_name(parts[0]) in {"metrics", "results", "structured_results"}:
        return None
    return parts[0]


def _metric_seed(key: str) -> str | None:
    parts = [part for part in str(key).split("/") if part]
    for part in parts[1:-1]:
        match = _SEED_TOKEN_RE.fullmatch(part)
        if match:
            return match.group(1)
    return None


def _collect_flat_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    finite: dict[str, float] = {}
    if not isinstance(metrics, Mapping):
        return finite
    for key, value in metrics.items():
        number = _finite_float(value)
        if number is not None:
            finite[str(key)] = number
    return finite


def _collect_structured_metrics(
    value: Any,
    *,
    condition: str | None = None,
) -> dict[str, float]:
    """Extract canonical metric fields from common structured-result schemas."""

    collected: dict[str, float] = {}
    if not isinstance(value, Mapping):
        return collected

    explicit_condition = value.get("condition")
    if isinstance(explicit_condition, str) and explicit_condition.strip():
        condition = explicit_condition.strip()

    metrics = value.get("metrics")
    if isinstance(metrics, Mapping):
        for key, metric_value in metrics.items():
            number = _finite_float(metric_value)
            if number is None:
                continue
            metric_key = f"{condition}/{key}" if condition else str(key)
            collected[metric_key] = number

    primary = _finite_float(value.get("primary_metric"))
    if primary is not None:
        metric_key = f"{condition}/primary_metric" if condition else "primary_metric"
        collected[metric_key] = primary

    for container_name in _CONDITION_CONTAINERS:
        container = value.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for condition_name, condition_data in container.items():
            condition_text = str(condition_name)
            if not isinstance(condition_data, Mapping):
                continue
            nested = _collect_structured_metrics(
                condition_data,
                condition=condition_text,
            )
            collected.update(nested)
            for key, metric_value in condition_data.items():
                if key in {"metrics", "seeds", "results"}:
                    continue
                number = _finite_float(metric_value)
                if number is None:
                    continue
                collected.setdefault(f"{condition_text}/{key}", number)

    rows = value.get("results")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row_condition = row.get("condition", condition)
            row_condition_text = (
                str(row_condition)
                if row_condition is not None and str(row_condition).strip()
                else None
            )
            seed = row.get("seed")
            for key, metric_value in row.items():
                if key in {"condition", "seed", "status", "success", "error"}:
                    continue
                number = _finite_float(metric_value)
                if number is None:
                    continue
                prefix = row_condition_text or ""
                if seed is not None and prefix:
                    metric_key = f"{prefix}/{seed}/{key}"
                elif prefix:
                    metric_key = f"{prefix}/{key}"
                else:
                    metric_key = str(key)
                collected.setdefault(metric_key, number)

    nested_structured = value.get("structured_results")
    if isinstance(nested_structured, Mapping):
        collected.update(
            _collect_structured_metrics(
                nested_structured,
                condition=condition,
            )
        )
    return collected


def _walk_numeric_fields(
    value: Any,
    *,
    path: tuple[str, ...] = (),
    condition: str | None = None,
) -> list[tuple[tuple[str, ...], str | None, float]]:
    fields: list[tuple[tuple[str, ...], str | None, float]] = []
    if isinstance(value, Mapping):
        explicit_condition = value.get("condition")
        if isinstance(explicit_condition, str) and explicit_condition.strip():
            condition = explicit_condition.strip()
        for key, child in value.items():
            key_text = str(key)
            normalized = _normalized_name(key_text)
            if normalized in _CONDITION_CONTAINERS and isinstance(child, Mapping):
                for condition_name, condition_data in child.items():
                    fields.extend(
                        _walk_numeric_fields(
                            condition_data,
                            path=path + (key_text, str(condition_name)),
                            condition=str(condition_name),
                        )
                    )
                continue
            fields.extend(
                _walk_numeric_fields(
                    child,
                    path=path + (key_text,),
                    condition=condition,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.extend(
                _walk_numeric_fields(
                    child,
                    path=path + (str(index),),
                    condition=condition,
                )
            )
    else:
        number = _finite_float(value)
        if number is not None and path:
            fields.append((path, condition, number))
    return fields


def _structured_failure_flags(
    value: Any,
    *,
    condition: str | None = None,
) -> list[tuple[str | None, str]]:
    failures: list[tuple[str | None, str]] = []
    if isinstance(value, Mapping):
        explicit_condition = value.get("condition")
        if isinstance(explicit_condition, str) and explicit_condition.strip():
            condition = explicit_condition.strip()

        status = str(value.get("status", "") or "").strip().casefold()
        if status in _FAILED_STATUS_VALUES:
            failures.append((condition, f"structured status is {status}"))
        if value.get("success") is False:
            failures.append((condition, "structured success flag is false"))

        for key, child in value.items():
            normalized = _normalized_name(key)
            if normalized in _CONDITION_CONTAINERS and isinstance(child, Mapping):
                for condition_name, condition_data in child.items():
                    failures.extend(
                        _structured_failure_flags(
                            condition_data,
                            condition=str(condition_name),
                        )
                    )
            elif isinstance(child, (Mapping, list)):
                failures.extend(
                    _structured_failure_flags(child, condition=condition)
                )
    elif isinstance(value, list):
        for child in value:
            failures.extend(_structured_failure_flags(child, condition=condition))
    return failures


def _append_unique(items: list[str], message: str) -> None:
    if message not in items:
        items.append(message)


def _condition_for_output_line(line: str) -> str | None:
    match = _CONDITION_RE.search(line)
    return match.group(1) if match else None


def _output_failure_events(output: str) -> list[tuple[str | None, str]]:
    """Return fatal output signals, attributed to a condition when possible."""

    events: list[tuple[str | None, str]] = []
    current_condition: str | None = None

    def add(condition: str | None, reason: str) -> None:
        event = (condition, reason)
        if event not in events:
            events.append(event)

    for line in output.splitlines():
        line_condition = _condition_for_output_line(line)
        if line_condition:
            current_condition = line_condition
        condition = line_condition or current_condition
        if _TRACEBACK_RE.search(line):
            add(condition, "traceback detected in experiment output")
        if any(pattern.search(line) for pattern in _EXPLICIT_FAILURE_RES):
            add(
                condition,
                "explicit failure signal detected in experiment output",
            )
        if any(pattern.search(line) for pattern in _DATASET_FAILURE_RES):
            add(
                condition,
                "dataset missing or unavailable in experiment output",
            )
        if any(pattern.search(line) for pattern in _SYNTHETIC_FALLBACK_RES):
            add(
                condition,
                "synthetic or random data fallback detected in experiment output",
            )

    whole_output_checks = (
        (
            _TRACEBACK_RE.search(output) is not None,
            "traceback detected in experiment output",
        ),
        (
            any(pattern.search(output) for pattern in _EXPLICIT_FAILURE_RES),
            "explicit failure signal detected in experiment output",
        ),
        (
            any(pattern.search(output) for pattern in _DATASET_FAILURE_RES),
            "dataset missing or unavailable in experiment output",
        ),
        (
            any(pattern.search(output) for pattern in _SYNTHETIC_FALLBACK_RES),
            "synthetic or random data fallback detected in experiment output",
        ),
    )
    for matched, reason in whole_output_checks:
        if matched and not any(event_reason == reason for _, event_reason in events):
            add(None, reason)
    return events


def assess_production_result(
    *,
    returncode: int | None,
    timed_out: bool,
    metrics: Mapping[str, Any] | None,
    stdout: str = "",
    stderr: str = "",
    structured_results: Any = None,
    status: str | None = None,
    declared_valid: bool | None = None,
) -> ProductionResultValidity:
    """Assess whether a production run contains genuine experiment results.

    ``valid`` means the run has usable, non-placeholder result data.  A timed
    out run can be valid partial evidence, but ``successful`` is true only for
    a complete zero-return-code execution.
    """

    reasons: list[str] = []
    condition_reasons: dict[str, list[str]] = defaultdict(list)
    status_text = str(status or "").strip().casefold()

    if declared_valid is False:
        _append_unique(reasons, "result was explicitly marked invalid")
    if returncode is not None and int(returncode) != 0 and not timed_out:
        _append_unique(reasons, f"non-zero return code: {int(returncode)}")
    if status_text in _FAILED_STATUS_VALUES:
        _append_unique(reasons, f"run status is {status_text}")

    output = f"{stdout or ''}\n{stderr or ''}"
    for condition, failure_reason in _output_failure_events(output):
        target = condition_reasons[condition] if condition else reasons
        _append_unique(target, failure_reason)

    finite_metrics = _collect_flat_metrics(metrics)
    for key, value in _collect_structured_metrics(structured_results).items():
        finite_metrics.setdefault(key, value)

    condition_scientific_metrics: dict[str, set[str]] = defaultdict(set)
    top_level_scientific_metrics: set[str] = set()
    named_conditions: set[str] = set()
    seed_ids: set[tuple[str, str]] = set()
    seed_evidence_present = False

    for key in finite_metrics:
        condition = _metric_condition(key)
        leaf = _leaf_name(key)
        if condition:
            named_conditions.add(condition)
        if not _is_control_metric(leaf):
            if condition:
                condition_scientific_metrics[condition].add(key)
            else:
                top_level_scientific_metrics.add(key)
        seed = _metric_seed(key)
        if seed is not None:
            seed_evidence_present = True
            if not _is_control_metric(leaf):
                seed_ids.add((condition or OVERALL_CONDITION, seed))

    def record_zero_guard(
        *,
        name: str,
        value: float,
        condition: str | None,
    ) -> None:
        target = condition_reasons[condition] if condition else reasons
        if _is_success_rate(name) and value <= 0:
            _append_unique(target, "success_rate is 0")
        elif _is_steps_completed(name) and value <= 0:
            _append_unique(target, "steps_completed is 0")
        elif _is_successful_seed_count(name) and value <= 0:
            _append_unique(target, "successful seed count is 0")

    for key, value in finite_metrics.items():
        record_zero_guard(
            name=_leaf_name(key),
            value=value,
            condition=_metric_condition(key),
        )

    if structured_results is not None:
        for path, condition, value in _walk_numeric_fields(structured_results):
            record_zero_guard(
                name=_normalized_name(path[-1]),
                value=value,
                condition=condition,
            )
        for condition, failure in _structured_failure_flags(structured_results):
            target = condition_reasons[condition] if condition else reasons
            _append_unique(target, failure)

    output_seed_mentions: set[tuple[str, str]] = set()
    output_successful_seeds: set[tuple[str, str]] = set()
    for line in output.splitlines():
        condition = _condition_for_output_line(line)
        seed_match = _SEED_RE.search(line)
        if seed_match:
            seed_evidence_present = True
            seed_key = (condition or OVERALL_CONDITION, seed_match.group(1))
            output_seed_mentions.add(seed_key)
            lowered = line.casefold()
            failed_line = any(
                marker in lowered
                for marker in (
                    "failed",
                    "error",
                    "traceback",
                    "dataset missing",
                    "dataset not found",
                    "skipped",
                )
            )
            if not failed_line:
                for metric_match in _METRIC_VALUE_RE.finditer(line):
                    metric_name = _normalized_name(metric_match.group(1))
                    metric_value = _finite_float(metric_match.group(2))
                    if metric_value is not None and not _is_control_metric(metric_name):
                        output_successful_seeds.add(seed_key)
                        break

        ratio_match = _SUCCESS_RATE_RATIO_RE.search(line)
        if ratio_match:
            numerator, denominator = ratio_match.groups()
            rate = int(numerator) / int(denominator) if int(denominator) else 0.0
            record_zero_guard(
                name="success_rate",
                value=rate,
                condition=condition,
            )
        else:
            value_match = _SUCCESS_RATE_VALUE_RE.search(line)
            if value_match:
                value = _finite_float(value_match.group(1))
                if value is not None:
                    record_zero_guard(
                        name="success_rate",
                        value=value,
                        condition=condition,
                    )

        steps_match = _STEPS_COMPLETED_RE.search(line)
        if steps_match:
            value = _finite_float(steps_match.group(1))
            if value is not None:
                record_zero_guard(
                    name="steps_completed",
                    value=value,
                    condition=condition,
                )

        seeds_match = _SUCCESSFUL_SEEDS_RE.search(line)
        if seeds_match:
            value = _finite_float(seeds_match.group(1))
            if value is not None:
                record_zero_guard(
                    name="successful_seeds",
                    value=value,
                    condition=condition,
                )

    seed_ids.update(output_successful_seeds)
    if output_seed_mentions and not output_successful_seeds and not seed_ids:
        affected = {
            condition
            for condition, _seed in output_seed_mentions
            if condition != OVERALL_CONDITION
        }
        if affected:
            for condition in affected:
                _append_unique(
                    condition_reasons[condition],
                    "no successful seed result",
                )
        else:
            _append_unique(reasons, "no successful seed result")

    if not finite_metrics:
        _append_unique(reasons, "no finite metrics were produced")
    elif not condition_scientific_metrics and not top_level_scientific_metrics:
        _append_unique(reasons, "no finite scientific metrics were produced")

    all_condition_names = named_conditions | set(condition_reasons)
    for condition in all_condition_names:
        if not condition_scientific_metrics.get(condition):
            _append_unique(
                condition_reasons[condition],
                "no finite scientific metrics were produced",
            )

    global_failure = bool(reasons)
    condition_level_valid = {
        condition
        for condition in named_conditions
        if condition_scientific_metrics.get(condition)
        and not condition_reasons.get(condition)
    }
    valid_conditions = set(condition_level_valid)
    if not named_conditions and top_level_scientific_metrics and not reasons:
        valid_conditions.add(OVERALL_CONDITION)

    if named_conditions and not condition_level_valid and not global_failure:
        _append_unique(reasons, "no condition produced a valid result")
    if not named_conditions and not top_level_scientific_metrics and not reasons:
        _append_unique(reasons, "no valid result was produced")

    top_level_result_valid = bool(
        top_level_scientific_metrics and not all_condition_names
    )
    has_scientific_result = bool(
        condition_level_valid or top_level_result_valid
    )
    valid = not reasons and has_scientific_result
    valid_metrics: dict[str, float] = {}
    if valid:
        for key, value in finite_metrics.items():
            condition = _metric_condition(key)
            if condition is None or condition in valid_conditions:
                valid_metrics[key] = value

    successful = (
        valid
        and (returncode is None or int(returncode) == 0)
        and not timed_out
        and status_text not in _PARTIAL_STATUS_VALUES
        and status_text not in _FAILED_STATUS_VALUES
    )
    if timed_out:
        _append_unique(reasons, "execution timed out; only partial results are available")

    return ProductionResultValidity(
        valid=valid,
        successful=successful,
        valid_metrics=valid_metrics,
        valid_conditions=valid_conditions,
        invalid_conditions={
            condition: messages
            for condition, messages in condition_reasons.items()
            if messages
        },
        successful_seed_count=len(seed_ids),
        seed_evidence_present=seed_evidence_present,
        reasons=reasons,
    )


def assess_run_payload(payload: Mapping[str, Any]) -> ProductionResultValidity:
    """Assess a persisted Stage 12/13 run payload."""

    status = str(payload.get("status", "") or "")
    returncode = payload.get("returncode")
    if returncode is None:
        returncode = (
            1
            if status.casefold() in _FAILED_STATUS_VALUES
            else None
        )
    declared_valid = payload.get("result_valid")
    return assess_production_result(
        returncode=(int(returncode) if returncode is not None else None),
        timed_out=bool(payload.get("timed_out", False)),
        metrics=(
            payload.get("metrics")
            if isinstance(payload.get("metrics"), Mapping)
            else payload.get("key_metrics")
        ),
        stdout=str(payload.get("stdout", "") or ""),
        stderr=str(payload.get("stderr", "") or ""),
        structured_results=payload.get("structured_results"),
        status=status,
        declared_valid=(
            bool(declared_valid) if isinstance(declared_valid, bool) else None
        ),
    )


def assess_experiment_summary(
    summary: Mapping[str, Any],
) -> ProductionResultValidity:
    """Assess a Stage 14/repair experiment summary."""

    best_run = (
        summary.get("best_run")
        if isinstance(summary.get("best_run"), Mapping)
        else {}
    )
    combined_metrics: dict[str, Any] = {}
    if isinstance(best_run, Mapping) and isinstance(best_run.get("metrics"), Mapping):
        combined_metrics.update(best_run["metrics"])

    condition_summaries = summary.get(
        "condition_summaries",
        summary.get("condition_metrics", {}),
    )
    if isinstance(condition_summaries, Mapping):
        for condition_name, condition_data in condition_summaries.items():
            if not isinstance(condition_data, Mapping):
                continue
            metrics = condition_data.get("metrics")
            if isinstance(metrics, Mapping):
                for key, value in metrics.items():
                    combined_metrics.setdefault(
                        f"{condition_name}/{key}",
                        value,
                    )
            for control_name in (
                "success_rate",
                "steps_completed",
                "pipeline_steps_completed",
                "successful_seeds",
                "successful_seed_count",
            ):
                if control_name in condition_data:
                    combined_metrics.setdefault(
                        f"{condition_name}/{control_name}",
                        condition_data[control_name],
                    )

    status = str(best_run.get("status", "") or "") if best_run else ""
    returncode = best_run.get("returncode") if best_run else None
    if returncode is None:
        returncode = 1 if status.casefold() in _FAILED_STATUS_VALUES else 0

    declared_valid: bool | None = None
    for candidate in (
        summary.get("result_valid"),
        best_run.get("result_valid") if best_run else None,
    ):
        if isinstance(candidate, bool):
            declared_valid = candidate
            break

    structured = summary.get("structured_results")
    if structured is None:
        structured = {"condition_summaries": condition_summaries}

    return assess_production_result(
        returncode=int(returncode),
        timed_out=bool(best_run.get("timed_out", False)) if best_run else False,
        metrics=combined_metrics,
        stdout=str(best_run.get("stdout", "") or "") if best_run else "",
        stderr=str(best_run.get("stderr", "") or "") if best_run else "",
        structured_results=structured,
        status=status,
        declared_valid=declared_valid,
    )
