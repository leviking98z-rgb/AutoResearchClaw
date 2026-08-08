"""Portfolio-level benchmark metrics for Research Queue runs."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import BudgetLevel
from .store import ResearchQueueStore


def build_portfolio_report(
    store: ResearchQueueStore,
    *,
    system_id: str = "",
    window_seconds: float = 7200.0,
) -> dict[str, Any]:
    """Compute VCO, latency, cost, funnel, and safety metrics from artifacts."""

    ideas = store.list_ideas()
    runs = store.list_runs()
    events = store.list_events(limit=1_000_000)
    timestamps = [
        parsed
        for parsed in (_parse_time(event.get("timestamp")) for event in events)
        if parsed is not None
    ]
    start = min(timestamps) if timestamps else None
    end = max(timestamps) if timestamps else start

    concluded_at: dict[str, datetime] = {}
    benchmark_completed_at: dict[str, datetime] = {}
    for event in events:
        timestamp = _parse_time(event.get("timestamp"))
        if timestamp is None:
            continue
        idea_id = str(event.get("idea_id", "") or "")
        if event.get("event") == "idea_concluded" and idea_id:
            concluded_at[idea_id] = timestamp
        if event.get("event") == "benchmark_completed" and idea_id:
            benchmark_completed_at[idea_id] = timestamp

    outcomes: list[dict[str, Any]] = []
    for idea in ideas:
        final_review = _final_review(store.idea_dir(idea.idea_id))
        execution_passed = final_review.get("execution_passed") is True
        scientific_valid = final_review.get("scientific_valid") is True
        supported_raw = final_review.get("hypothesis_supported")
        conclusive = type(supported_raw) is bool
        valid_conclusive = execution_passed and scientific_valid and conclusive
        completed = benchmark_completed_at.get(
            idea.idea_id,
            concluded_at.get(idea.idea_id),
        )
        outcomes.append(
            {
                "idea_id": idea.idea_id,
                "title": idea.title,
                "queue_conclusion": (
                    idea.conclusion.value if idea.conclusion is not None else None
                ),
                "execution_passed": execution_passed,
                "scientific_valid": scientific_valid,
                "hypothesis_supported": (
                    supported_raw if conclusive else None
                ),
                "promotion_decision": final_review.get("promotion_decision"),
                "valid_conclusive_outcome": valid_conclusive,
                "tokens": max(0, idea.total_tokens),
                "gpu_seconds": max(0.0, idea.gpu_seconds),
                "completed_at": completed.isoformat() if completed else None,
                "reason": str(
                    final_review.get("reason", idea.last_reason) or ""
                ),
            }
        )

    vco_rows = [row for row in outcomes if row["valid_conclusive_outcome"]]
    window_end = (
        start + timedelta(seconds=max(0.0, window_seconds))
        if start is not None
        else None
    )
    window_vco_rows = [
        row
        for row in vco_rows
        if window_end is not None
        and (completed := _parse_time(row.get("completed_at"))) is not None
        and completed <= window_end
    ]
    vco_at_window = len(window_vco_rows)
    first_vco = min(
        (
            completed
            for completed in (
                _parse_time(row.get("completed_at")) for row in vco_rows
            )
            if completed is not None
        ),
        default=None,
    )
    ttfv = (
        (first_vco - start).total_seconds()
        if start is not None and first_vco is not None
        else None
    )

    idea_accounted_tokens = sum(max(0, idea.total_tokens) for idea in ideas)
    audit_tokens = _audit_total_tokens(store.root / "llm-audit")
    total_tokens = max(idea_accounted_tokens, audit_tokens)
    total_gpu_seconds = sum(max(0.0, idea.gpu_seconds) for idea in ideas)
    accepted = [
        row for row in outcomes if row["promotion_decision"] == "accept"
    ]
    invalid_accepts = [
        row
        for row in accepted
        if not (
            row["execution_passed"]
            and row["scientific_valid"]
            and row["hypothesis_supported"] is True
        )
    ]

    benchmark_run_ids = {
        str(event.get("run_id"))
        for event in events
        if event.get("event") == "benchmark_started" and event.get("run_id")
    }
    run_ideas_by_budget = {
        level.value: {
            run.idea_id
            for run in runs
            if run.budget is level and run.run_id not in benchmark_run_ids
        }
        for level in BudgetLevel
    }
    funnel = {
        "generated": len(ideas),
        "admitted": len(
            {
                str(event.get("idea_id"))
                for event in events
                if event.get("event") == "idea_admitted"
                and event.get("idea_id")
            }
        ),
        "reached_B0": len(run_ideas_by_budget[BudgetLevel.B0.value]),
        "reached_B1": len(run_ideas_by_budget[BudgetLevel.B1.value]),
        "reached_B2": len(run_ideas_by_budget[BudgetLevel.B2.value]),
        "selected_for_benchmark": len(
            {
                str(event.get("idea_id"))
                for event in events
                if event.get("event") == "idea_selected_for_promotion"
                and event.get("idea_id")
            }
        ),
        "benchmark_completed": len(
            {
                str(event.get("idea_id"))
                for event in events
                if event.get("event") == "benchmark_completed"
                and event.get("idea_id")
            }
        ),
        "scientifically_valid": sum(
            1 for row in outcomes if row["scientific_valid"]
        ),
        "valid_conclusive": len(vco_rows),
    }

    cumulative: list[dict[str, Any]] = []
    for index, row in enumerate(
        sorted(
            vco_rows,
            key=lambda item: item.get("completed_at") or "",
        ),
        start=1,
    ):
        completed = _parse_time(row.get("completed_at"))
        cumulative.append(
            {
                "idea_id": row["idea_id"],
                "completed_at": row["completed_at"],
                "elapsed_seconds": (
                    (completed - start).total_seconds()
                    if completed is not None and start is not None
                    else None
                ),
                "cumulative_vco": index,
            }
        )

    duration_seconds = (
        max(0.0, (end - start).total_seconds())
        if start is not None and end is not None
        else 0.0
    )
    summary = {
        "schema_version": 1,
        "system_id": system_id,
        "started_at": start.isoformat() if start else None,
        "ended_at": end.isoformat() if end else None,
        "duration_seconds": duration_seconds,
        "window_seconds": window_seconds,
        "vco": len(vco_rows),
        "vco_at_window": vco_at_window,
        "valid_positive": sum(
            1
            for row in window_vco_rows
            if row["hypothesis_supported"] is True
        ),
        "valid_negative": sum(
            1
            for row in window_vco_rows
            if row["hypothesis_supported"] is False
        ),
        "ttfv_seconds": ttfv,
        "tokens_per_vco": _ratio(total_tokens, vco_at_window),
        "gpu_seconds_per_vco": _ratio(total_gpu_seconds, vco_at_window),
        "invalid_accepts": len(invalid_accepts),
        "accepts": len(accepted),
        "false_accept_rate": _ratio(len(invalid_accepts), len(accepted)),
    }
    usage = {
        "total_tokens": total_tokens,
        "idea_accounted_tokens": idea_accounted_tokens,
        "audit_tokens": audit_tokens,
        "total_gpu_seconds": total_gpu_seconds,
        "ideas": len(ideas),
        "runs": len(runs),
        "tokens_by_conclusion": _sum_by(
            outcomes,
            key="queue_conclusion",
            value="tokens",
        ),
        "gpu_seconds_by_conclusion": _sum_by(
            outcomes,
            key="queue_conclusion",
            value="gpu_seconds",
        ),
    }
    return {
        "summary": summary,
        "funnel": funnel,
        "usage": usage,
        "timeline": cumulative,
        "outcomes": outcomes,
        "conclusion_counts": dict(
            Counter(row["queue_conclusion"] or "none" for row in outcomes)
        ),
    }


def write_portfolio_report(
    store: ResearchQueueStore,
    output_dir: str | Path,
    *,
    system_id: str = "",
    window_seconds: float = 7200.0,
) -> dict[str, Any]:
    report = build_portfolio_report(
        store,
        system_id=system_id,
        window_seconds=window_seconds,
    )
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "portfolio-report.json": report,
        "summary.json": report["summary"],
        "funnel.json": report["funnel"],
        "usage.json": report["usage"],
        "timeline.json": report["timeline"],
    }
    for name, value in payloads.items():
        (root / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return report


def compare_portfolio_reports(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two reports without pretending zero-VCO ratios are finite."""

    base = dict(baseline.get("summary", baseline))
    current = dict(candidate.get("summary", candidate))
    directions = {
        "vco_at_window": "higher",
        "ttfv_seconds": "lower",
        "tokens_per_vco": "lower",
        "gpu_seconds_per_vco": "lower",
        "false_accept_rate": "lower",
    }
    metrics: dict[str, Any] = {}
    for name, direction in directions.items():
        baseline_value = base.get(name)
        candidate_value = current.get(name)
        metrics[name] = {
            "direction": direction,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "absolute_delta": _delta(candidate_value, baseline_value),
            "relative_change": _relative_change(
                candidate_value,
                baseline_value,
            ),
            "improved": _improved(
                candidate_value,
                baseline_value,
                direction=direction,
            ),
        }
    return {
        "schema_version": 1,
        "baseline_system_id": base.get("system_id", ""),
        "candidate_system_id": current.get("system_id", ""),
        "metrics": metrics,
    }


def _final_review(idea_dir: Path) -> dict[str, Any]:
    for path in (
        idea_dir / "final_review.json",
        idea_dir / "benchmark" / "final_review.json",
    ):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _audit_total_tokens(root: Path) -> int:
    total = 0
    for path in root.glob("*/calls.jsonl*"):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                total += max(0, int(value.get("total_tokens", 0) or 0))
    return total


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _ratio(numerator: float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator > 0 else None


def _sum_by(
    rows: list[dict[str, Any]],
    *,
    key: str,
    value: str,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        name = str(row.get(key) or "none")
        totals[name] = totals.get(name, 0.0) + float(row.get(value, 0.0) or 0.0)
    return totals


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _delta(candidate: Any, baseline: Any) -> float | None:
    current = _finite(candidate)
    previous = _finite(baseline)
    return current - previous if current is not None and previous is not None else None


def _relative_change(candidate: Any, baseline: Any) -> float | None:
    delta = _delta(candidate, baseline)
    previous = _finite(baseline)
    if delta is None or previous in {None, 0.0}:
        return None
    return delta / abs(previous)


def _improved(candidate: Any, baseline: Any, *, direction: str) -> bool | None:
    current = _finite(candidate)
    previous = _finite(baseline)
    if current is None or previous is None:
        return None
    return current > previous if direction == "higher" else current < previous
