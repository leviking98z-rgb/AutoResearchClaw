"""Versioned benchmark capabilities and pre-execution compatibility checks.

The Queue core does not know domain-specific datasets or metrics.  It only
requires each concrete benchmark to expose a small, frozen capability profile
that can be checked against a :class:`ResearchSpec` before treatment generation
or GPU allocation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import ResearchSpec

TREATMENT_API = (
    "build_treatment(); treatment.fit(calibration_logits, calibration_labels); "
    "treatment.transform(evaluation_logits, state)"
)
CIFAR10_ECE_MINIMUM_EFFECT = 1e-3


@dataclass(frozen=True, slots=True)
class BenchmarkProfile:
    """Machine-readable capabilities of one versioned benchmark."""

    benchmark_id: str
    version: int
    available_pairs: int
    metrics: tuple[str, ...]
    calibration_split: str
    evaluation_split: str
    pairing_strategy: str
    evidence_capabilities: tuple[str, ...]
    compute_accounting: tuple[str, ...]
    treatment_api: str
    minimum_effects: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "metrics",
            "evidence_capabilities",
            "compute_accounting",
        ):
            value[name] = list(value[name])
        value["minimum_effects"] = dict(sorted(self.minimum_effects.items()))
        return value

    def minimum_effect_for(self, metric: str) -> float:
        """Return the frozen practical-significance floor for ``metric``."""

        return max(
            0.0,
            float(self.minimum_effects.get(_metric_key(metric), 0.0) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkCompatibility:
    """Deterministic result of matching a ResearchSpec to a profile."""

    passed: bool
    errors: tuple[str, ...]
    checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": list(self.errors),
            "checks": dict(self.checks),
        }


def load_benchmark_profile(
    benchmark_id: str,
    config_path: str | Path,
) -> BenchmarkProfile:
    """Build the profile for a configured adapter.

    This intentionally remains a tiny explicit registry.  Adding Kafka, a
    plugin runtime, or dynamic imports here would recreate the old v2 control
    plane without improving research quality.
    """

    normalized = benchmark_id.strip()
    if normalized != "cifar10_calibration":
        raise ValueError(f"unsupported benchmark profile: {normalized!r}")

    from researchclaw.benchmark_adapter.cifar10_calibration import BenchmarkConfig

    config = BenchmarkConfig.from_file(config_path)
    return BenchmarkProfile(
        benchmark_id=normalized,
        version=2,
        available_pairs=len(config.seeds),
        metrics=("accuracy", "ece", "nll"),
        calibration_split="clean",
        evaluation_split=(
            "clean"
            if config.corruption.strip().casefold() in {"", "none", "clean"}
            else "corrupted"
        ),
        pairing_strategy="disjoint_example_blocks",
        evidence_capabilities=(
            "disjoint_pairs",
            "paired_effect_ci",
            "per_pair_metrics",
            "per_example_argmax",
            "protocol_attestation",
        ),
        compute_accounting=(
            "calibration_examples",
            "evaluation_examples",
            "model_forward_examples",
        ),
        treatment_api=TREATMENT_API,
        minimum_effects={"ece": CIFAR10_ECE_MINIMUM_EFFECT},
    )


def validate_benchmark_compatibility(
    spec: ResearchSpec,
    profile: BenchmarkProfile,
) -> BenchmarkCompatibility:
    """Reject impossible contracts before code generation or GPU allocation."""

    errors: list[str] = []
    checks: dict[str, bool] = {}

    checks["benchmark_id_matches"] = spec.benchmark_id == profile.benchmark_id
    if not checks["benchmark_id_matches"]:
        errors.append(
            f"ResearchSpec benchmark_id {spec.benchmark_id!r} does not match "
            f"profile {profile.benchmark_id!r}"
        )

    checks["minimum_pairs_available"] = (
        spec.minimum_pairs <= profile.available_pairs
    )
    if not checks["minimum_pairs_available"]:
        errors.append(
            f"ResearchSpec requires {spec.minimum_pairs} independent pairs, "
            f"but benchmark profile provides {profile.available_pairs}"
        )

    required_metrics = {
        _metric_key(spec.primary_metric),
        *(_metric_key(item.metric) for item in spec.guardrail_metrics),
    }
    available_metrics = {_metric_key(item) for item in profile.metrics}
    missing_metrics = sorted(required_metrics - available_metrics)
    checks["metrics_available"] = not missing_metrics
    if missing_metrics:
        errors.append(
            "benchmark profile does not provide metrics: "
            + ", ".join(missing_metrics)
        )

    capabilities = set(profile.evidence_capabilities)
    needs_effect_ci = spec.primary_requires_effect_ci or any(
        item.require_effect_ci for item in spec.guardrail_metrics
    )
    checks["effect_ci_supported"] = (
        not needs_effect_ci or "paired_effect_ci" in capabilities
    )
    if not checks["effect_ci_supported"]:
        errors.append("ResearchSpec requires paired effect CIs but profile cannot emit them")

    needs_per_pair = any(item.per_pair for item in spec.guardrail_metrics)
    checks["per_pair_metrics_supported"] = (
        not needs_per_pair or "per_pair_metrics" in capabilities
    )
    if not checks["per_pair_metrics_supported"]:
        errors.append(
            "ResearchSpec requires per-pair guardrails but profile only emits aggregates"
        )

    checks["per_example_argmax_supported"] = (
        not spec.require_per_example_argmax
        or "per_example_argmax" in capabilities
    )
    if not checks["per_example_argmax_supported"]:
        errors.append(
            "ResearchSpec requires per-example argmax evidence but profile "
            "cannot emit it"
        )

    checks["calibration_split_matches"] = (
        not spec.calibration_split
        or _normalized_name(spec.calibration_split)
        == _normalized_name(profile.calibration_split)
    )
    if not checks["calibration_split_matches"]:
        errors.append(
            f"ResearchSpec requires calibration_split={spec.calibration_split!r}, "
            f"but profile provides {profile.calibration_split!r}"
        )

    checks["evaluation_split_matches"] = (
        not spec.evaluation_split
        or _normalized_name(spec.evaluation_split)
        == _normalized_name(profile.evaluation_split)
    )
    if not checks["evaluation_split_matches"]:
        errors.append(
            f"ResearchSpec requires evaluation_split={spec.evaluation_split!r}, "
            f"but profile provides {profile.evaluation_split!r}"
        )

    checks["pairing_strategy_matches"] = (
        not spec.pairing_strategy
        or _normalized_name(spec.pairing_strategy)
        == _normalized_name(profile.pairing_strategy)
    )
    if not checks["pairing_strategy_matches"]:
        errors.append(
            f"ResearchSpec requires pairing_strategy={spec.pairing_strategy!r}, "
            f"but profile provides {profile.pairing_strategy!r}"
        )

    required_compute = {
        _normalized_name(item) for item in spec.required_compute_accounting
    }
    available_compute = {
        _normalized_name(item) for item in profile.compute_accounting
    }
    missing_compute = sorted(required_compute - available_compute)
    checks["compute_accounting_supported"] = not missing_compute
    if missing_compute:
        errors.append(
            "benchmark profile cannot attest required compute dimensions: "
            + ", ".join(missing_compute)
        )

    checks["treatment_api_matches"] = (
        not spec.treatment_api or spec.treatment_api == profile.treatment_api
    )
    if not checks["treatment_api_matches"]:
        errors.append("ResearchSpec treatment_api does not match benchmark profile")

    minimum_effect = profile.minimum_effect_for(spec.primary_metric)
    checks["minimum_effect_meets_profile"] = (
        spec.minimum_effect >= minimum_effect
    )
    if not checks["minimum_effect_meets_profile"]:
        errors.append(
            f"ResearchSpec minimum_effect {spec.minimum_effect:.12g} is below "
            f"the frozen {profile.benchmark_id} floor "
            f"{minimum_effect:.12g} for {_metric_key(spec.primary_metric)!r}"
        )

    return BenchmarkCompatibility(
        passed=not errors,
        errors=tuple(errors),
        checks=checks,
    )


def build_benchmark_plan(
    *,
    profile: BenchmarkProfile,
    config_path: str | Path,
) -> dict[str, Any]:
    """Return the immutable scientific plan persisted before implementation."""

    from researchclaw.benchmark_adapter.cifar10_calibration import BenchmarkConfig

    source = Path(config_path).expanduser().resolve()
    config = BenchmarkConfig.from_file(source)
    return {
        "schema_version": 1,
        "benchmark_profile": profile.to_dict(),
        "protocol": {
            "dataset_url": config.dataset_url,
            "dataset_sha256": config.dataset_sha256,
            "weights_url": config.weights_url,
            "weights_sha256": config.weights_sha256,
            "model_source_repo": config.model_source_repo,
            "model_source_commit": config.model_source_commit,
            "model_name": config.model_name,
            "seeds": list(config.seeds),
            "examples_per_pair": config.examples,
            "calibration_examples_per_pair": config.calibration_examples,
            "calibration_split": profile.calibration_split,
            "evaluation_split": profile.evaluation_split,
            "pairing_strategy": profile.pairing_strategy,
            "corruption": config.corruption,
            "corruption_severity": config.corruption_severity,
            "ece_bins": config.ece_bins,
        },
        "config_path": str(source),
        "config_sha256": _sha256(source),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _metric_key(value: str) -> str:
    text = value.strip().casefold()
    aliases = {
        "expected calibration error": "ece",
        "negative log likelihood": "nll",
        "negative log-likelihood": "nll",
    }
    for phrase, key in aliases.items():
        if phrase in text:
            return key
    return _normalized_name(text)
