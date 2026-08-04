"""Derived Factory metrics for retrospective optimization and operations."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import median
from typing import Any

from .io import atomic_write_json, iter_jsonl
from .models import IdeaStatus, WorkItemStatus, utc_now
from .store import FactoryStore


def _timestamp(value: object) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 6)


def build_factory_observability(
    store: FactoryStore,
    *,
    max_events: int | None = 200_000,
) -> dict[str, Any]:
    """Aggregate append-only state into a compact, fail-soft review artifact."""

    retained_limit = (
        max_events + 1
        if max_events is not None and max_events > 0
        else None
    )
    events = iter_jsonl(store.events_path, max_lines=retained_limit)
    window_truncated = bool(
        max_events is not None
        and max_events > 0
        and len(events) > max_events
    )
    if window_truncated:
        events = events[-max_events:]
    event_counts = Counter(str(row.get("type", "unknown")) for row in events)
    failure_reasons: Counter[str] = Counter()
    gate_reasons: Counter[str] = Counter()
    queued_at: dict[tuple[str, int], float] = {}
    started_at: dict[tuple[str, int], float] = {}
    attempts_seen: set[tuple[str, int]] = set()
    retried_attempts: set[tuple[str, int]] = set()
    queue_waits: list[float] = []
    run_times: list[float] = []
    tick_times: list[float] = []
    first_time = None
    last_time = None

    for row in events:
        observed = _timestamp(row.get("timestamp"))
        if observed is not None:
            first_time = observed if first_time is None else min(first_time, observed)
            last_time = observed if last_time is None else max(last_time, observed)
        event_type = str(row.get("type", ""))
        item_id = str(row.get("item_id", "") or "")
        try:
            attempt = max(0, int(row.get("attempt", 0) or 0))
        except (TypeError, ValueError):
            attempt = 0
        reason = str(
            row.get("reason_code")
            or row.get("failure_reason")
            or row.get("error")
            or ""
        ).strip()
        if "fail" in event_type or "reject" in event_type:
            failure_reasons[reason or event_type] += 1
        if event_type == "gate_decided":
            gate_reasons[str(row.get("reason_code", "unknown"))] += 1
        if event_type == "factory_tick":
            try:
                tick_times.append(
                    max(0.0, float(row.get("elapsed_sec", 0.0) or 0.0))
                )
            except (TypeError, ValueError):
                pass
        if not item_id or observed is None:
            continue
        attempt_key = (item_id, attempt)
        attempts_seen.add(attempt_key)
        if "queued" in event_type or "waiting" in event_type:
            queued_at.setdefault(attempt_key, observed)
        elif event_type in {"work_item_started", "gpu_work_item_started"}:
            started_at[attempt_key] = observed
            queued = queued_at.pop(attempt_key, None)
            if queued is not None:
                queue_waits.append(max(0.0, observed - queued))
        elif event_type in {
            "work_item_succeeded",
            "work_item_failed",
            "gpu_work_item_finished",
            "gpu_work_item_cancelled",
            "work_item_cancelled",
            "work_item_retry_wait",
        }:
            started = started_at.pop(attempt_key, None)
            if started is not None:
                run_times.append(max(0.0, observed - started))
            queued_at.pop(attempt_key, None)
            if event_type == "work_item_retry_wait":
                retried_attempts.add(attempt_key)

    ideas = store.list_ideas()
    items = store.list_work_items()
    ledgers = [store.load_budget(idea.idea_id) for idea in ideas]
    duration_hours = (
        max(0.0, (last_time - first_time) / 3600.0)
        if first_time is not None and last_time is not None
        else 0.0
    )
    research_outcomes = sum(
        idea.status
        in {
            IdeaStatus.COMPLETED,
            IdeaStatus.COMPLETED_NEGATIVE,
        }
        for idea in ideas
    )
    promoted_ideas = sum(
        1
        for idea in ideas
        if idea.status.value
        in {
            "building",
            "smoke",
            "pilot",
            "validating",
            "paper",
            "completed",
            "completed_negative",
        }
    )
    rejected_ideas = sum(
        idea.status.value in {"parked", "rejected", "failed"}
        for idea in ideas
    )
    succeeded_items = sum(
        item.status is WorkItemStatus.SUCCEEDED for item in items
    )
    retried_items = len(retried_attempts)
    # Historical journals may predate attempt-terminal events. Retain a
    # conservative snapshot fallback without double-counting observed retries.
    if not retried_items:
        retried_items = sum(max(0, item.attempt - 1) for item in items)
    queue_stats = {
        "samples": len(queue_waits),
        "p50_sec": round(median(queue_waits), 6) if queue_waits else 0.0,
        "p95_sec": _percentile(queue_waits, 0.95),
        "max_sec": round(max(queue_waits), 6) if queue_waits else 0.0,
    }
    runtime_stats = {
        "samples": len(run_times),
        "p50_sec": round(median(run_times), 6) if run_times else 0.0,
        "p95_sec": _percentile(run_times, 0.95),
        "max_sec": round(max(run_times), 6) if run_times else 0.0,
    }
    tick_stats = {
        "samples": len(tick_times),
        "p50_sec": round(median(tick_times), 6) if tick_times else 0.0,
        "p95_sec": _percentile(tick_times, 0.95),
        "max_sec": round(max(tick_times), 6) if tick_times else 0.0,
    }
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "events": {
            "retained": len(events),
            "window_truncated": window_truncated,
            "by_type": dict(event_counts.most_common()),
            "first_timestamp": (
                datetime.fromtimestamp(first_time).astimezone().isoformat()
                if first_time is not None
                else ""
            ),
            "last_timestamp": (
                datetime.fromtimestamp(last_time).astimezone().isoformat()
                if last_time is not None
                else ""
            ),
        },
        "throughput": {
            "observation_hours": round(duration_hours, 6),
            "ideas_terminal": research_outcomes,
            "work_items_succeeded": succeeded_items,
            # The durable state below is lifetime data, while the timestamp
            # window may be a bounded journal tail. Do not publish a lifetime
            # numerator divided by a truncated observation window.
            "ideas_terminal_per_hour": (
                round(research_outcomes / duration_hours, 6)
                if duration_hours > 0 and not window_truncated
                else 0.0
            ),
            "work_items_succeeded_per_hour": (
                round(succeeded_items / duration_hours, 6)
                if duration_hours > 0 and not window_truncated
                else 0.0
            ),
            "rate_basis": (
                "unavailable_truncated_event_window"
                if window_truncated
                else "lifetime_state_over_full_event_window"
            ),
        },
        "latency": {
            "factory_tick": tick_stats,
            "queue_wait": queue_stats,
            "work_item_runtime": runtime_stats,
        },
        "reliability": {
            "retries": retried_items,
            "failure_reasons": dict(failure_reasons.most_common(50)),
            "gate_reasons": dict(gate_reasons.most_common(50)),
        },
        "outcomes": {
            "ideas_total": len(ideas),
            "promoted_beyond_screen": promoted_ideas,
            "terminal": research_outcomes,
            "rejected_or_parked": rejected_ideas,
            "screen_conversion_rate": (
                round(promoted_ideas / len(ideas), 6)
                if ideas
                else 0.0
            ),
            "terminal_yield_rate": (
                round(research_outcomes / len(ideas), 6)
                if ideas
                else 0.0
            ),
        },
        "budgets": {
            "gpu_hours_total": round(
                sum(ledger.gpu_hours() for ledger in ledgers),
                6,
            ),
            "llm_calls_total": sum(ledger.llm_calls for ledger in ledgers),
            "engineering_repairs_total": sum(
                ledger.engineering_repairs for ledger in ledgers
            ),
        },
    }
    atomic_write_json(store.root / "observability_summary.json", summary)
    return summary
