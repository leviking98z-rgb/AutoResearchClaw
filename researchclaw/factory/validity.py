"""Fail-closed experiment-summary validity adapter.

When the production validity module is installed, Factory mode delegates to
it. The compact fallback keeps the Factory package importable and safe on a
clean upstream checkout: explicit failure evidence or missing scientific
metrics can never be treated as valid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ExperimentValidity:
    valid: bool
    successful: bool
    successful_seed_count: int = 0
    seed_evidence_present: bool = False
    reasons: list[str] = field(default_factory=list)


def _finite_metrics(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, Mapping):
        for item in value.values():
            result.extend(_finite_metrics(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(_finite_metrics(item))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def assess_experiment_summary(summary: Mapping[str, Any]) -> ExperimentValidity:
    try:
        from researchclaw.pipeline.result_validity import (
            assess_experiment_summary as production_assessment,
        )
    except ImportError:
        production_assessment = None
    if production_assessment is not None:
        result = production_assessment(summary)
        return ExperimentValidity(
            valid=bool(result.valid),
            successful=bool(result.successful),
            successful_seed_count=int(result.successful_seed_count),
            seed_evidence_present=bool(result.seed_evidence_present),
            reasons=list(result.reasons),
        )

    best = summary.get("best_run")
    best_run = dict(best) if isinstance(best, Mapping) else {}
    reasons: list[str] = []
    returncode = best_run.get("returncode", 0)
    status = str(best_run.get("status", "") or "").casefold()
    stdout = str(best_run.get("stdout", "") or "")
    stderr = str(best_run.get("stderr", "") or "")
    declared = summary.get("result_valid", best_run.get("result_valid"))
    if declared is not True:
        reasons.append("experiment did not explicitly declare valid evidence")
    try:
        if int(returncode) != 0:
            reasons.append(f"experiment returncode={returncode}")
    except (TypeError, ValueError):
        reasons.append("experiment returncode is malformed")
    if status in {"failed", "failure", "error", "crashed", "invalid"}:
        reasons.append(f"experiment status={status}")
    combined_output = f"{stdout}\n{stderr}".casefold()
    for marker in (
        "traceback (most recent call last)",
        "datasetnotfounderror",
        "fail:",
        "using synthetic data",
    ):
        if marker in combined_output:
            reasons.append(f"failure marker: {marker}")
    scientific = _finite_metrics(best_run.get("metrics"))
    scientific.extend(_finite_metrics(summary.get("condition_summaries")))
    if not scientific:
        reasons.append("no finite scientific metrics were produced")
    seed_raw = summary.get("successful_seed_count", 0)
    try:
        seed_count = max(0, int(seed_raw))
    except (TypeError, ValueError):
        seed_count = 0
    valid = not reasons
    return ExperimentValidity(
        valid=valid,
        successful=valid,
        successful_seed_count=seed_count,
        seed_evidence_present="successful_seed_count" in summary,
        reasons=reasons,
    )
