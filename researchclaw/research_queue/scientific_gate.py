"""Lightweight deterministic scientific-contract validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from .models import (
    IdeaRecord,
    MetricDirection,
    MetricGuardrail,
    MetricRelation,
    ResearchSpec,
)

NUMERICAL_EFFECT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class ScientificGateResult:
    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "checks": dict(self.checks),
        }


def research_spec_from_idea(
    idea: IdeaRecord,
    *,
    benchmark_id: str = "",
    treatment_api: str = "",
) -> ResearchSpec:
    if idea.research_spec is not None:
        return idea.research_spec
    is_cifar = benchmark_id == "cifar10_calibration"
    return ResearchSpec(
        question=idea.question,
        hypothesis=idea.hypothesis,
        treatment=idea.treatment,
        control=idea.control,
        primary_metric=idea.primary_metric,
        metric_direction=_infer_metric_direction(idea.primary_metric),
        guardrails=(
            "Treatment and control must be evaluated on identical held-out data.",
        ),
        validity_conditions=(
            "The primary metric and all declared guardrails must be reported.",
        ),
        compute_matching=(
            "Treatment and control use the same calibration data and evaluation split.",
        ),
        stopping_rules=(
            "Stop when the scientific validity conditions fail.",
            "Stop when treatment does not improve the primary metric.",
        ),
        benchmark_id=benchmark_id,
        treatment_api=treatment_api,
        guardrail_metrics=(
            MetricGuardrail(
                metric="accuracy",
                direction=MetricDirection.MAXIMIZE,
            ),
        ),
        calibration_split="clean" if is_cifar else "",
        evaluation_split="corrupted" if is_cifar else "",
        pairing_strategy="disjoint_example_blocks" if is_cifar else "",
        require_per_example_argmax=is_cifar,
        required_compute_accounting=(
            (
                "calibration_examples",
                "evaluation_examples",
                "model_forward_examples",
            )
            if is_cifar
            else ()
        ),
    )


def validate_research_spec(
    spec: ResearchSpec,
    *,
    benchmark_id: str = "",
) -> ScientificGateResult:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    required = {
        "question": spec.question,
        "hypothesis": spec.hypothesis,
        "treatment": spec.treatment,
        "control": spec.control,
        "primary_metric": spec.primary_metric,
    }
    for name, value in required.items():
        checks[f"has_{name}"] = bool(value.strip())
        if not value.strip():
            errors.append(f"missing {name}")

    distinct = spec.treatment.strip().casefold() != spec.control.strip().casefold()
    checks["treatment_control_distinct"] = distinct
    if not distinct:
        errors.append("treatment and control are identical")

    checks["has_guardrails"] = bool(spec.guardrails)
    checks["has_validity_conditions"] = bool(spec.validity_conditions)
    checks["has_compute_matching"] = bool(spec.compute_matching)
    checks["has_stopping_rules"] = bool(spec.stopping_rules)
    for name, values in (
        ("guardrails", spec.guardrails),
        ("validity_conditions", spec.validity_conditions),
        ("compute_matching", spec.compute_matching),
        ("stopping_rules", spec.stopping_rules),
    ):
        if not values:
            errors.append(f"missing {name}")

    checks["metric_direction_valid"] = spec.metric_direction in {
        MetricDirection.MAXIMIZE,
        MetricDirection.MINIMIZE,
    }
    checks["minimum_pairs_valid"] = spec.minimum_pairs >= 1
    if spec.minimum_pairs < 1:
        errors.append("minimum_pairs must be at least 1")
    checks["confidence_level_valid"] = 0.5 <= spec.confidence_level < 1.0
    if not checks["confidence_level_valid"]:
        errors.append("confidence_level must be in [0.5, 1.0)")
    checks["guardrail_metrics_valid"] = all(
        item.metric.strip() and item.tolerance >= 0.0 for item in spec.guardrail_metrics
    )
    if not checks["guardrail_metrics_valid"]:
        errors.append("guardrail_metrics contain an invalid metric or tolerance")
    expected_benchmark = benchmark_id.strip()
    checks["benchmark_matches"] = (
        not expected_benchmark or spec.benchmark_id == expected_benchmark
    )
    if expected_benchmark and spec.benchmark_id != expected_benchmark:
        errors.append(
            f"benchmark_id must be {expected_benchmark!r}, got {spec.benchmark_id!r}"
        )
    if (expected_benchmark or spec.benchmark_id) == "cifar10_calibration":
        checks["has_executable_evidence_contract"] = bool(
            spec.guardrail_metrics
            and spec.minimum_pairs >= 2
            and spec.primary_requires_effect_ci
        )
        if not checks["has_executable_evidence_contract"]:
            errors.append(
                "cifar10_calibration requires guardrail_metrics, "
                "minimum_pairs >= 2, and primary_requires_effect_ci=true"
            )
        required_compute = {
            "calibration_examples",
            "evaluation_examples",
            "model_forward_examples",
        }
        declared_compute = {
            _normalized_name(item)
            for item in spec.required_compute_accounting
        }
        checks["has_frozen_protocol_contract"] = bool(
            _normalized_name(spec.calibration_split) == "clean"
            and _normalized_name(spec.evaluation_split) == "corrupted"
            and _normalized_name(spec.pairing_strategy)
            == "disjoint_example_blocks"
            and spec.require_per_example_argmax
            and required_compute <= declared_compute
        )
        if not checks["has_frozen_protocol_contract"]:
            errors.append(
                "cifar10_calibration requires clean calibration, corrupted "
                "evaluation, disjoint example-block pairs, per-example "
                "argmax evidence, and compute "
                "accounting for calibration_examples, evaluation_examples, "
                "and model_forward_examples"
            )

    combined = " ".join(
        (
            spec.primary_metric,
            *spec.guardrails,
            *spec.validity_conditions,
            *spec.compute_matching,
        )
    ).casefold()
    if "coverage" in combined and not _contains_threshold(combined):
        warnings.append(
            "coverage is mentioned without an explicit numeric target or tolerance"
        )
    if "risk" in spec.primary_metric.casefold() and "coverage" not in combined:
        warnings.append("risk comparison does not explicitly require matched coverage")
    if "width" in spec.primary_metric.casefold() and "coverage" not in combined:
        warnings.append("interval-width comparison does not explicitly guard coverage")

    return ScientificGateResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
    )


def validate_benchmark_result(
    spec: ResearchSpec,
    result: Mapping[str, Any],
    *,
    baseline_prefix: str = "baseline",
    treatment_prefix: str = "treatment",
) -> ScientificGateResult:
    metrics_raw = result.get("metrics", {})
    metrics = dict(metrics_raw) if isinstance(metrics_raw, Mapping) else {}
    per_seed_raw = result.get("per_seed", ())
    per_seed = list(per_seed_raw) if isinstance(per_seed_raw, (list, tuple)) else []
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {
        "execution_status_ok": str(result.get("status", "")).casefold()
        in {"ok", "success", "succeeded", "passed"}
    }
    if not checks["execution_status_ok"]:
        errors.append(str(result.get("error", "") or "benchmark execution failed"))

    primary = _metric_key(spec.primary_metric)
    treatment_key = f"{treatment_prefix}_{primary}"
    baseline_key = f"{baseline_prefix}_{primary}"
    checks["primary_metric_reported"] = (
        treatment_key in metrics and baseline_key in metrics
    )
    if not checks["primary_metric_reported"]:
        errors.append(
            f"benchmark did not report {treatment_key!r} and {baseline_key!r}"
        )

    guardrail_keys = {
        _metric_key(item)
        for item in spec.guardrails
        if _metric_key(item) in {"accuracy", "nll", "ece"}
    }
    missing_guardrails = [
        key
        for key in sorted(guardrail_keys)
        if f"{treatment_prefix}_{key}" not in metrics
        or f"{baseline_prefix}_{key}" not in metrics
    ]
    checks["guardrails_reported"] = not missing_guardrails
    if missing_guardrails:
        errors.append(
            "benchmark omitted guardrail metrics: " + ", ".join(missing_guardrails)
        )

    checks["minimum_pairs_met"] = len(per_seed) >= spec.minimum_pairs
    if not checks["minimum_pairs_met"]:
        errors.append(
            "insufficient independent pairs: "
            f"required {spec.minimum_pairs}, observed {len(per_seed)}"
        )

    evidence_raw = result.get("evidence", {})
    evidence = dict(evidence_raw) if isinstance(evidence_raw, Mapping) else {}
    protocol_raw = evidence.get("protocol", {})
    protocol = (
        dict(protocol_raw) if isinstance(protocol_raw, Mapping) else {}
    )
    if spec.calibration_split:
        observed = str(protocol.get("calibration_split", "") or "")
        checks["calibration_split_attested"] = (
            _normalized_name(observed)
            == _normalized_name(spec.calibration_split)
        )
        if not checks["calibration_split_attested"]:
            errors.append(
                "benchmark protocol did not attest required calibration split "
                f"{spec.calibration_split!r}; observed {observed!r}"
            )
    if spec.evaluation_split:
        observed = str(protocol.get("evaluation_split", "") or "")
        checks["evaluation_split_attested"] = (
            _normalized_name(observed)
            == _normalized_name(spec.evaluation_split)
        )
        if not checks["evaluation_split_attested"]:
            errors.append(
                "benchmark protocol did not attest required evaluation split "
                f"{spec.evaluation_split!r}; observed {observed!r}"
            )
    if spec.pairing_strategy:
        observed = str(protocol.get("pairing_strategy", "") or "")
        checks["pairing_strategy_attested"] = (
            _normalized_name(observed)
            == _normalized_name(spec.pairing_strategy)
        )
        if not checks["pairing_strategy_attested"]:
            errors.append(
                "benchmark protocol did not attest required pairing strategy "
                f"{spec.pairing_strategy!r}; observed {observed!r}"
            )

    if spec.require_per_example_argmax:
        argmax_raw = evidence.get("argmax", {})
        argmax = dict(argmax_raw) if isinstance(argmax_raw, Mapping) else {}
        per_pair_argmax = bool(per_seed) and all(
            isinstance(row, Mapping)
            and isinstance(row.get("evidence"), Mapping)
            and row["evidence"].get("argmax_preserved") is True
            and _zero_count(row["evidence"].get("argmax_changed_count"))
            and bool(row["evidence"].get("baseline_predictions_sha256"))
            and bool(row["evidence"].get("treatment_predictions_sha256"))
            for row in per_seed
        )
        checks["per_example_argmax_attested"] = (
            argmax.get("argmax_preserved") is True
            and _zero_count(argmax.get("argmax_changed_count"))
            and argmax.get("per_example_prediction_hashes") is True
            and per_pair_argmax
        )
        if not checks["per_example_argmax_attested"]:
            errors.append(
                "benchmark did not prove per-example argmax preservation"
            )

    if spec.required_compute_accounting:
        compute_raw = evidence.get("compute_accounting", {})
        compute = dict(compute_raw) if isinstance(compute_raw, Mapping) else {}
        observed_dimensions = {
            _normalized_name(str(item))
            for item in compute.get("matched_dimensions", ())
        }
        required_dimensions = {
            _normalized_name(item) for item in spec.required_compute_accounting
        }
        checks["compute_matching_attested"] = (
            compute.get("all_declared_dimensions_matched") is True
            and required_dimensions <= observed_dimensions
        )
        if not checks["compute_matching_attested"]:
            missing = sorted(required_dimensions - observed_dimensions)
            errors.append(
                "benchmark did not attest required compute matching"
                + (f": missing {', '.join(missing)}" if missing else "")
            )

    if spec.primary_requires_effect_ci:
        interval = _effect_interval(
            result,
            metric=primary,
            direction=spec.metric_direction,
            baseline_prefix=baseline_prefix,
            treatment_prefix=treatment_prefix,
            confidence_level=spec.confidence_level,
        )
        checks["primary_effect_ci_reported"] = interval is not None
        if interval is None:
            errors.append(
                f"benchmark did not report a paired {spec.confidence_level:.1%} "
                f"effect confidence interval for {primary}"
            )
        else:
            lower, _ = interval
            checks["primary_effect_ci_supports_hypothesis"] = (
                lower > max(spec.minimum_effect, NUMERICAL_EFFECT_EPSILON)
            )

    guardrail_specs = _guardrail_specs(spec)
    checks["guardrail_contracts_present"] = bool(guardrail_specs)
    for guardrail in guardrail_specs:
        metric = _metric_key(guardrail.metric)
        baseline_key = f"{baseline_prefix}_{metric}"
        treatment_key = f"{treatment_prefix}_{metric}"
        reported_key = f"{metric}_guardrail_reported"
        checks[reported_key] = baseline_key in metrics and treatment_key in metrics
        if not checks[reported_key]:
            errors.append(
                f"benchmark did not report guardrail {baseline_key!r} "
                f"and {treatment_key!r}"
            )
            continue
        baseline = _finite_float(metrics[baseline_key])
        treatment = _finite_float(metrics[treatment_key])
        if baseline is None or treatment is None:
            checks[f"{metric}_guardrail_finite"] = False
            errors.append(f"{metric} guardrail contains non-finite values")
            continue
        checks[f"{metric}_guardrail_finite"] = True
        effect = _beneficial_effect(
            baseline,
            treatment,
            direction=guardrail.direction,
        )
        passed = (
            abs(treatment - baseline) <= guardrail.tolerance
            if guardrail.relation is MetricRelation.EQUAL
            else effect >= -guardrail.tolerance
        )
        checks[f"{metric}_guardrail_passed"] = passed
        if not passed:
            if guardrail.relation is MetricRelation.EQUAL:
                errors.append(
                    f"treatment changed {metric} by "
                    f"{treatment - baseline:.6f}, exceeding equality "
                    f"tolerance {guardrail.tolerance:.6f}"
                )
            else:
                errors.append(
                    f"treatment degraded {metric} by {-effect:.6f}, exceeding "
                    f"tolerance {guardrail.tolerance:.6f}"
                )

        if guardrail.per_pair:
            pair_check = _pair_guardrail_passed(
                per_seed,
                metric=metric,
                direction=guardrail.direction,
                relation=guardrail.relation,
                tolerance=guardrail.tolerance,
            )
            checks[f"{metric}_per_pair_guardrail_passed"] = pair_check is True
            if pair_check is None:
                errors.append(
                    f"benchmark did not report per-pair {metric} guardrail values"
                )
            elif not pair_check:
                errors.append(f"{metric} guardrail failed for at least one pair")

        if guardrail.require_effect_ci:
            interval = _effect_interval(
                result,
                metric=metric,
                direction=guardrail.direction,
                baseline_prefix=baseline_prefix,
                treatment_prefix=treatment_prefix,
                confidence_level=spec.confidence_level,
            )
            checks[f"{metric}_effect_ci_reported"] = interval is not None
            if interval is None:
                errors.append(
                    f"benchmark did not report a paired {spec.confidence_level:.1%} "
                    f"effect confidence interval for guardrail {metric}"
                )
            else:
                lower, _ = interval
                ci_passed = lower >= -guardrail.tolerance
                checks[f"{metric}_effect_ci_passed"] = ci_passed
                if not ci_passed:
                    errors.append(
                        f"{metric} effect CI lower endpoint {lower:.6f} "
                        f"was below {-guardrail.tolerance:.6f}"
                    )

    return ScientificGateResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
    )


def hypothesis_supported(
    spec: ResearchSpec,
    result_or_metrics: Mapping[str, Any],
    *,
    minimum_effect: float | None = None,
) -> bool | None:
    nested_metrics = result_or_metrics.get("metrics")
    metrics = (
        nested_metrics if isinstance(nested_metrics, Mapping) else result_or_metrics
    )
    key = _metric_key(spec.primary_metric)
    treatment_key = f"treatment_{key}"
    baseline_key = f"baseline_{key}"
    if treatment_key not in metrics or baseline_key not in metrics:
        return None
    treatment = float(metrics[treatment_key])
    baseline = float(metrics[baseline_key])
    threshold = (
        spec.minimum_effect
        if minimum_effect is None
        else max(spec.minimum_effect, minimum_effect)
    )
    threshold = max(threshold, NUMERICAL_EFFECT_EPSILON)
    if spec.primary_requires_effect_ci:
        if not isinstance(nested_metrics, Mapping):
            return None
        interval = _effect_interval(
            result_or_metrics,
            metric=key,
            direction=spec.metric_direction,
            baseline_prefix="baseline",
            treatment_prefix="treatment",
            confidence_level=spec.confidence_level,
        )
        if interval is None:
            return None
        return interval[0] > threshold
    if spec.metric_direction is MetricDirection.MINIMIZE:
        return baseline - treatment > threshold
    return treatment - baseline > threshold


def _guardrail_specs(spec: ResearchSpec) -> tuple[MetricGuardrail, ...]:
    if spec.guardrail_metrics:
        return spec.guardrail_metrics
    # Backward compatibility for old ResearchSpecs. This deliberately handles
    # only the tiny metric vocabulary owned by the benchmark adapter; arbitrary
    # natural-language scientific rules are not interpreted here.
    text = " ".join(spec.guardrails).casefold()
    values: list[MetricGuardrail] = []
    if any(term in text for term in ("accuracy", "argmax", "prediction")):
        values.append(
            MetricGuardrail(
                metric="accuracy",
                direction=MetricDirection.MAXIMIZE,
                relation=(
                    MetricRelation.EQUAL
                    if any(term in text for term in ("exact", "equal", "unchanged"))
                    else MetricRelation.NO_WORSE
                ),
            )
        )
    if "nll" in text or "negative log" in text:
        values.append(
            MetricGuardrail(
                metric="nll",
                direction=MetricDirection.MINIMIZE,
            )
        )
    return tuple(values)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _zero_count(value: Any) -> bool:
    number = _finite_float(value)
    return number == 0.0


def _beneficial_effect(
    baseline: float,
    treatment: float,
    *,
    direction: MetricDirection,
) -> float:
    if direction is MetricDirection.MINIMIZE:
        return baseline - treatment
    return treatment - baseline


def _pair_guardrail_passed(
    rows: list[Any],
    *,
    metric: str,
    direction: MetricDirection,
    relation: MetricRelation,
    tolerance: float,
) -> bool | None:
    if not rows:
        return None
    found = False
    for raw in rows:
        if not isinstance(raw, Mapping):
            return None
        baseline_raw = raw.get("baseline")
        treatment_raw = raw.get("treatment")
        if not isinstance(baseline_raw, Mapping) or not isinstance(
            treatment_raw, Mapping
        ):
            return None
        baseline = _finite_float(baseline_raw.get(metric))
        treatment = _finite_float(treatment_raw.get(metric))
        if baseline is None or treatment is None:
            return None
        found = True
        if relation is MetricRelation.EQUAL:
            if abs(treatment - baseline) > tolerance:
                return False
        elif (
            _beneficial_effect(
                baseline,
                treatment,
                direction=direction,
            )
            < -tolerance
        ):
            return False
    return True if found else None


def _effect_interval(
    result: Mapping[str, Any],
    *,
    metric: str,
    direction: MetricDirection,
    baseline_prefix: str,
    treatment_prefix: str,
    confidence_level: float,
) -> tuple[float, float] | None:
    uncertainty_raw = result.get("uncertainty", {})
    uncertainty = dict(uncertainty_raw) if isinstance(uncertainty_raw, Mapping) else {}
    for key in (
        f"effect_{metric}_ci",
        f"effect_{metric}_ci_{round(confidence_level * 100):d}",
    ):
        interval = uncertainty.get(key)
        parsed = _parse_interval(interval)
        if parsed is not None:
            return parsed

    per_seed_raw = result.get("per_seed", ())
    if not isinstance(per_seed_raw, (list, tuple)):
        return None
    effects: list[float] = []
    effect_key = f"effect_{metric}"
    for raw in per_seed_raw:
        if not isinstance(raw, Mapping):
            return None
        explicit = _finite_float(raw.get(effect_key))
        if explicit is not None:
            effects.append(explicit)
            continue
        baseline_raw = raw.get(baseline_prefix)
        treatment_raw = raw.get(treatment_prefix)
        if not isinstance(baseline_raw, Mapping) or not isinstance(
            treatment_raw, Mapping
        ):
            return None
        baseline = _finite_float(baseline_raw.get(metric))
        treatment = _finite_float(treatment_raw.get(metric))
        if baseline is None or treatment is None:
            return None
        effects.append(
            _beneficial_effect(
                baseline,
                treatment,
                direction=direction,
            )
        )
    if len(effects) < 2:
        return None
    return _bootstrap_mean_interval(
        effects,
        confidence_level=confidence_level,
    )


def _parse_interval(value: Any) -> tuple[float, float] | None:
    if isinstance(value, Mapping):
        lower = _finite_float(value.get("lower"))
        upper = _finite_float(value.get("upper"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower = _finite_float(value[0])
        upper = _finite_float(value[1])
    else:
        return None
    if lower is None or upper is None or lower > upper:
        return None
    return lower, upper


def _bootstrap_mean_interval(
    values: list[float],
    *,
    confidence_level: float,
    samples: int = 10000,
) -> tuple[float, float]:
    """Deterministic paired bootstrap CI without a scipy dependency."""

    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
    )
    return float(lower), float(upper)


def _metric_key(value: str) -> str:
    text = value.strip().casefold()
    aliases = {
        "expected calibration error": "ece",
        "ece": "ece",
        "negative log likelihood": "nll",
        "negative log-likelihood": "nll",
        "nll": "nll",
        "accuracy": "accuracy",
    }
    for phrase, key in aliases.items():
        if phrase in text:
            return key
    token = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return token or "primary"


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _infer_metric_direction(primary_metric: str) -> MetricDirection:
    lowered = primary_metric.casefold()
    if any(
        term in lowered
        for term in (
            "error",
            "loss",
            "nll",
            "ece",
            "gap",
            "risk",
            "width",
            "latency",
        )
    ):
        return MetricDirection.MINIMIZE
    return MetricDirection.MAXIMIZE


def _contains_threshold(value: str) -> bool:
    return bool(re.search(r"\d+(?:\.\d+)?\s*%|0\.\d+|±|\+/-|tolerance|target", value))
