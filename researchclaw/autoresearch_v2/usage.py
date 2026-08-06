"""Operational usage aggregation for the AutoResearch v2 dashboard."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import BudgetConfig, UsageMonitoringConfig
from .models import IdeaRecord, JobRecord
from .store import V2Store

_TIERS = ("decision", "worker", "utility")
_MAX_TREND_POINTS = 2_000


@dataclass(slots=True)
class _AuditFileCache:
    """Incremental state for one append-only or periodically rotated log."""

    identity: tuple[int, int] | None = None
    offset: int = 0
    mtime_ns: int = -1
    pending: bytes = b""
    records: list[dict[str, Any]] = field(default_factory=list)


class UsageMonitor:
    """Aggregate durable LLM audit logs and scientific resource counters."""

    def __init__(
        self,
        *,
        store: V2Store,
        budgets: BudgetConfig,
        config: UsageMonitoringConfig,
        gpu_total: int = 0,
    ) -> None:
        self.store = store
        self.budgets = budgets
        self.config = config
        self.gpu_total = max(0, int(gpu_total))
        self._audit_cache: dict[Path, _AuditFileCache] = {}
        self._cache_lock = threading.Lock()

    def collect(
        self,
        *,
        hours: int | None = None,
        bucket_minutes: int | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(UTC)
        history_hours = max(
            1,
            min(
                int(hours or self.config.history_hours),
                24 * 366,
            ),
        )
        bucket_size = max(
            1,
            min(
                int(bucket_minutes or self.config.bucket_minutes),
                24 * 60,
            ),
        )
        bucket_size = max(
            bucket_size,
            math.ceil(
                history_hours * 60 / _MAX_TREND_POINTS
            ),
        )
        calls = list(self._read_calls())
        ideas = self.store.list_ideas()
        jobs = self.store.list_jobs()
        llm = self._llm_summary(
            calls,
            now=now,
            history_hours=history_hours,
            bucket_minutes=bucket_size,
        )
        gpu = self._gpu_summary(
            ideas,
            jobs,
            now=now,
            history_hours=history_hours,
            bucket_minutes=bucket_size,
        )
        costs = self._cost_summary(llm=llm, gpu=gpu)
        budgets = self._budget_summary(
            ideas=ideas,
            calls=calls,
            llm=llm,
            gpu=gpu,
            costs=costs,
            now=now,
        )
        alerts = self._alerts(
            ideas=ideas,
            jobs=jobs,
            llm=llm,
            gpu=gpu,
            budgets=budgets,
            now=now,
        )
        return {
            "generated_at": now.isoformat(timespec="milliseconds"),
            "window": {
                "history_hours": history_hours,
                "bucket_minutes": bucket_size,
                "timezone": "UTC",
            },
            "llm": llm,
            "gpu": gpu,
            "costs": costs,
            "budgets": budgets,
            "alerts": alerts,
            "ideas": self._idea_usage(ideas),
        }

    def _read_calls(self) -> list[dict[str, Any]]:
        root = self.store.root / "llm-audit"
        paths = [
            (root / tier / "calls.jsonl", tier)
            for tier in _TIERS
        ]
        # Store maintenance keeps one archive. Include it so cumulative and
        # monthly totals do not suddenly drop when calls.jsonl is rotated.
        paths.extend(
            (root / tier / "calls.jsonl.1", tier)
            for tier in _TIERS
        )
        with self._cache_lock:
            active_paths = {path for path, _tier in paths}
            for cached in set(self._audit_cache) - active_paths:
                self._audit_cache.pop(cached, None)
            for path, tier in paths:
                self._refresh_audit_file(path, tier=tier)
            records: list[dict[str, Any]] = []
            for path, _tier in paths:
                cached = self._audit_cache.get(path)
                if cached is not None:
                    records.extend(cached.records)
        records.sort(
            key=lambda row: (
                row.get("_timestamp") or datetime.min.replace(tzinfo=UTC),
                str(row.get("tier", "")),
            )
        )
        return records

    def _refresh_audit_file(self, path: Path, *, tier: str) -> None:
        cache = self._audit_cache.setdefault(path, _AuditFileCache())
        try:
            stat = path.stat()
        except OSError:
            cache.identity = None
            cache.offset = 0
            cache.mtime_ns = -1
            cache.pending = b""
            cache.records.clear()
            return
        identity = (int(stat.st_dev), int(stat.st_ino))
        unchanged = (
            cache.identity == identity
            and cache.offset == stat.st_size
            and cache.mtime_ns == stat.st_mtime_ns
        )
        if unchanged:
            return
        if (
            cache.identity != identity
            or stat.st_size < cache.offset
            or (
                stat.st_size == cache.offset
                and stat.st_mtime_ns != cache.mtime_ns
            )
        ):
            cache.identity = identity
            cache.offset = 0
            cache.pending = b""
            cache.records.clear()
        try:
            with path.open("rb") as stream:
                stream.seek(cache.offset)
                chunk = stream.read()
        except OSError:
            return
        cache.offset += len(chunk)
        cache.mtime_ns = stat.st_mtime_ns
        complete, separator, pending = (cache.pending + chunk).rpartition(
            b"\n"
        )
        if not separator:
            cache.pending += chunk
            return
        cache.pending = pending
        for raw_line in complete.splitlines():
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(value, Mapping):
                continue
            record = dict(value)
            record.setdefault("tier", tier)
            record["_timestamp"] = _parse_time(record.get("timestamp"))
            cache.records.append(record)

    def _llm_summary(
        self,
        calls: list[dict[str, Any]],
        *,
        now: datetime,
        history_hours: int,
        bucket_minutes: int,
    ) -> dict[str, Any]:
        totals = _empty_usage_row()
        by_tier: dict[str, dict[str, Any]] = {}
        by_model: dict[str, dict[str, Any]] = {}
        by_role: dict[str, dict[str, Any]] = {}
        for call in calls:
            _add_call(totals, call)
            for table, key in (
                (by_tier, str(call.get("tier", "unknown") or "unknown")),
                (by_model, str(call.get("model", "unknown") or "unknown")),
                (by_role, str(call.get("role", "unknown") or "unknown")),
            ):
                row = table.setdefault(key, _empty_usage_row())
                _add_call(row, call)
        window_start = now - timedelta(hours=history_hours)
        recent = [
            call
            for call in calls
            if (
                isinstance(call.get("_timestamp"), datetime)
                and call["_timestamp"] >= window_start
            )
        ]
        last_hour = [
            call
            for call in calls
            if (
                isinstance(call.get("_timestamp"), datetime)
                and call["_timestamp"] >= now - timedelta(hours=1)
            )
        ]
        last_day = [
            call
            for call in calls
            if (
                isinstance(call.get("_timestamp"), datetime)
                and call["_timestamp"] >= now - timedelta(hours=24)
            )
        ]
        burn_1h = sum(
            max(0, int(call.get("total_tokens", 0) or 0))
            for call in last_hour
        )
        burn_24h = sum(
            max(0, int(call.get("total_tokens", 0) or 0))
            for call in last_day
        )
        largest_call = max(
            (
                {
                    "timestamp": (
                        call["_timestamp"].isoformat(
                            timespec="milliseconds"
                        )
                        if isinstance(call.get("_timestamp"), datetime)
                        else None
                    ),
                    "tier": str(
                        call.get("tier", "unknown") or "unknown"
                    ),
                    "role": str(
                        call.get("role", "unknown") or "unknown"
                    ),
                    "model": str(
                        call.get("model", "unknown") or "unknown"
                    ),
                    "total_tokens": max(
                        0,
                        int(call.get("total_tokens", 0) or 0),
                    ),
                }
                for call in recent
            ),
            key=lambda row: row["total_tokens"],
            default=None,
        )
        return {
            "totals": totals,
            "by_tier": _table_rows(by_tier, key_name="tier"),
            "by_model": _table_rows(by_model, key_name="model"),
            "by_role": _table_rows(by_role, key_name="role"),
            "trend": _bucket_calls(
                recent,
                start=window_start,
                end=now,
                bucket_minutes=bucket_minutes,
            ),
            "burn_rate": {
                "tokens_last_hour": burn_1h,
                "tokens_per_hour_24h": burn_24h / 24.0,
                "tokens_last_24h": burn_24h,
                "projected_30d_tokens_at_24h_rate": round(
                    (burn_24h / 24.0) * 24 * 30
                ),
            },
            "largest_call": largest_call,
        }

    def _gpu_summary(
        self,
        ideas: list[IdeaRecord],
        jobs: list[JobRecord],
        *,
        now: datetime,
        history_hours: int,
        bucket_minutes: int,
    ) -> dict[str, Any]:
        running = [
            job
            for job in jobs
            if job.status.value == "running" and job.requires_gpu
        ]
        allocated = sum(
            max(0, int(job.result.get("allocated_gpus", 0) or 0))
            for job in running
        )
        pending = sum(
            1
            for job in jobs
            if job.requires_gpu
            and job.status.value in {"ready", "retry_wait"}
        )
        total_gpu_hours = (
            sum(max(0.0, idea.gpu_seconds_spent) for idea in ideas)
            / 3600.0
        )
        window_start = now - timedelta(hours=history_hours)
        ticks = self.store.list_events_between(
            event_type="controller_tick",
            started_at=window_start.isoformat(timespec="milliseconds"),
            ended_at=now.isoformat(timespec="milliseconds"),
        )
        trend = _bucket_gpu_ticks(
            ticks,
            start=window_start,
            end=now,
            bucket_minutes=bucket_minutes,
        )
        idle_since = _gpu_idle_since(ticks, now=now, allocated=allocated)
        state_minutes = _gpu_state_minutes(
            ticks,
            start=window_start,
            end=now,
        )
        pending_created_at = [
            stamp
            for job in jobs
            if job.requires_gpu
            and job.status.value in {"ready", "retry_wait"}
            and (stamp := _parse_time(job.created_at)) is not None
        ]
        oldest_pending_age = (
            max(
                0.0,
                (
                    now - min(pending_created_at)
                ).total_seconds()
                / 60.0,
            )
            if pending_created_at
            else 0.0
        )
        return {
            "total_gpu_hours": total_gpu_hours,
            "capacity_gpus": self.gpu_total,
            "allocated_gpus": allocated,
            "current_utilization": (
                allocated / self.gpu_total
                if self.gpu_total
                else 0.0
            ),
            "pending_jobs": pending,
            "oldest_pending_job_age_minutes": oldest_pending_age,
            "running_jobs": len(running),
            "idle_since": (
                idle_since.isoformat(timespec="milliseconds")
                if idle_since is not None
                else None
            ),
            "idle_minutes": (
                max(0.0, (now - idle_since).total_seconds() / 60.0)
                if idle_since is not None
                else 0.0
            ),
            "state_minutes": state_minutes,
            "trend": trend,
        }

    def _cost_summary(
        self,
        *,
        llm: Mapping[str, Any],
        gpu: Mapping[str, Any],
    ) -> dict[str, Any]:
        by_model: list[dict[str, Any]] = []
        llm_total = 0.0
        priced_calls = 0
        total_calls = 0
        for row in llm["by_model"]:
            total_calls += int(row["calls"])
            price = _model_price(
                str(row["model"]),
                self.config.model_prices,
            )
            priced = price is not None
            cost = 0.0
            if price is not None:
                priced_calls += int(row["calls"])
                cost = (
                    row["prompt_tokens"]
                    * price["input_per_million_usd"]
                    + row["completion_tokens"]
                    * price["output_per_million_usd"]
                ) / 1_000_000.0
                llm_total += cost
            by_model.append(
                {
                    **row,
                    "estimated_cost_usd": cost if priced else None,
                    "priced": priced,
                }
            )
        gpu_total = (
            float(gpu["total_gpu_hours"])
            * self.config.gpu_hour_cost_usd
        )
        return {
            "estimated": bool(
                self.config.model_prices
                or self.config.gpu_hour_cost_usd
            ),
            "currency": "USD",
            "llm_estimated_usd": llm_total,
            "gpu_estimated_usd": gpu_total,
            "total_estimated_usd": llm_total + gpu_total,
            "priced_calls": priced_calls,
            "total_calls": total_calls,
            "coverage": (
                priced_calls / total_calls if total_calls else 0.0
            ),
            "by_model": by_model,
            "gpu_hour_cost_usd": self.config.gpu_hour_cost_usd,
            "disclaimer": (
                "Estimated from configured rates; CLI subscription or "
                "provider billing can differ."
            ),
        }

    def _budget_summary(
        self,
        *,
        ideas: list[IdeaRecord],
        calls: list[dict[str, Any]],
        llm: Mapping[str, Any],
        gpu: Mapping[str, Any],
        costs: Mapping[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        token_cap = self.budgets.max_llm_tokens_per_idea
        gpu_cap = max(
            self.budgets.pilot_gpu_hours,
            self.budgets.scale_gpu_hours,
        )
        idea_rows: list[dict[str, Any]] = []
        for idea in sorted(
            ideas,
            key=lambda row: (
                row.llm_tokens_spent / token_cap if token_cap else 0.0
            ),
            reverse=True,
        ):
            token_ratio = (
                idea.llm_tokens_spent / token_cap if token_cap else 0.0
            )
            gpu_hours = idea.gpu_seconds_spent / 3600.0
            gpu_ratio = gpu_hours / gpu_cap if gpu_cap else 0.0
            if max(token_ratio, gpu_ratio) < self.config.warning_threshold:
                continue
            idea_rows.append(
                {
                    "idea_id": idea.idea_id,
                    "title": idea.title,
                    "status": idea.status.value,
                    "token_used": idea.llm_tokens_spent,
                    "token_limit": token_cap,
                    "token_ratio": token_ratio,
                    "gpu_hours_used": gpu_hours,
                    "gpu_hours_limit": gpu_cap,
                    "gpu_ratio": gpu_ratio,
                    "severity": _severity(
                        max(token_ratio, gpu_ratio),
                        warning=self.config.warning_threshold,
                        critical=self.config.critical_threshold,
                    ),
                }
            )
        month_start = now.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        calls_month = [
            call
            for call in calls
            if (
                isinstance(call.get("_timestamp"), datetime)
                and call["_timestamp"] >= month_start
            )
        ]
        monthly_tokens = sum(
            max(0, int(call.get("total_tokens", 0) or 0))
            for call in calls_month
        )
        elapsed_days = max(
            1.0 / 24.0,
            (now - month_start).total_seconds() / 86400.0,
        )
        days_in_month = _days_in_month(now)
        return {
            "idea_token_limit": token_cap,
            "idea_gpu_hours_limit": gpu_cap,
            "warning_threshold": self.config.warning_threshold,
            "critical_threshold": self.config.critical_threshold,
            "ideas_near_limit": idea_rows,
            "monthly": {
                "month": month_start.strftime("%Y-%m"),
                "tokens_used": monthly_tokens,
                "tokens_budget": self.config.monthly_token_budget,
                "tokens_ratio": _ratio(
                    monthly_tokens,
                    self.config.monthly_token_budget,
                ),
                "projected_tokens": round(
                    monthly_tokens / elapsed_days * days_in_month
                ),
                "gpu_hours_used": gpu["total_gpu_hours"],
                "gpu_hours_budget": (
                    self.config.monthly_gpu_hours_budget
                ),
                "gpu_hours_ratio": _ratio(
                    gpu["total_gpu_hours"],
                    self.config.monthly_gpu_hours_budget,
                ),
                "estimated_cost_usd": costs["total_estimated_usd"],
                "cost_budget_usd": self.config.monthly_cost_budget_usd,
                "cost_ratio": _ratio(
                    costs["total_estimated_usd"],
                    self.config.monthly_cost_budget_usd,
                ),
            },
            "audit_token_total": llm["totals"]["total_tokens"],
            "idea_accounted_token_total": sum(
                idea.llm_tokens_spent for idea in ideas
            ),
        }

    def _alerts(
        self,
        *,
        ideas: list[IdeaRecord],
        jobs: list[JobRecord],
        llm: Mapping[str, Any],
        gpu: Mapping[str, Any],
        budgets: Mapping[str, Any],
        now: datetime,
    ) -> list[dict[str, Any]]:
        del ideas, now
        alerts: list[dict[str, Any]] = []
        for row in budgets["ideas_near_limit"]:
            ratio = max(row["token_ratio"], row["gpu_ratio"])
            alerts.append(
                _alert(
                    severity=row["severity"],
                    code="idea_budget_pressure",
                    title=f"Idea budget {ratio:.0%}",
                    message=(
                        f"{row['title']} uses "
                        f"{row['token_used']:,}/{row['token_limit']:,} "
                        "tokens and "
                        f"{row['gpu_hours_used']:.2f}/"
                        f"{row['gpu_hours_limit']:.2f} GPU-h."
                    ),
                    idea_id=row["idea_id"],
                )
            )
        monthly = budgets["monthly"]
        for key, label in (
            ("tokens_ratio", "Monthly token budget"),
            ("gpu_hours_ratio", "Monthly GPU budget"),
            ("cost_ratio", "Monthly cost budget"),
        ):
            ratio = monthly[key]
            if ratio is None or ratio < self.config.warning_threshold:
                continue
            alerts.append(
                _alert(
                    severity=_severity(
                        ratio,
                        warning=self.config.warning_threshold,
                        critical=self.config.critical_threshold,
                    ),
                    code=f"monthly_{key}",
                    title=label,
                    message=f"{label} is at {ratio:.0%}.",
                )
            )
        burn = llm["burn_rate"]["tokens_last_hour"]
        if (
            self.config.token_burn_warning_per_hour
            and burn >= self.config.token_burn_warning_per_hour
        ):
            alerts.append(
                _alert(
                    severity="warning",
                    code="token_burn_rate",
                    title="High token burn rate",
                    message=f"{burn:,} tokens were used in the last hour.",
                )
            )
        largest_call = llm.get("largest_call")
        if (
            isinstance(largest_call, Mapping)
            and self.config.single_call_token_warning
            and int(largest_call.get("total_tokens", 0) or 0)
            >= self.config.single_call_token_warning
        ):
            alerts.append(
                _alert(
                    severity="warning",
                    code="single_call_token_spike",
                    title="Large single LLM call",
                    message=(
                        f"{largest_call.get('model', 'unknown')} used "
                        f"{int(largest_call['total_tokens']):,} tokens in "
                        f"one {largest_call.get('tier', 'unknown')} call."
                    ),
                )
            )
        if (
            gpu["pending_jobs"] > 0
            and gpu["allocated_gpus"] == 0
            and self.config.gpu_idle_warning_minutes
            and gpu["idle_minutes"]
            >= self.config.gpu_idle_warning_minutes
        ):
            alerts.append(
                _alert(
                    severity="critical",
                    code="gpu_idle_with_backlog",
                    title="GPU backlog is not running",
                    message=(
                        f"{gpu['pending_jobs']} GPU jobs are pending while "
                        f"the pool has been idle for "
                        f"{gpu['idle_minutes']:.0f} minutes."
                    ),
                )
            )
        failed_jobs = sum(
            1 for job in jobs if job.status.value == "failed"
        )
        if failed_jobs:
            alerts.append(
                _alert(
                    severity="info",
                    code="failed_jobs",
                    title="Failed jobs retained",
                    message=(
                        f"{failed_jobs} failed jobs remain available for "
                        "audit and failure-rate analysis."
                    ),
                )
            )
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(
            key=lambda row: (
                severity_order.get(row["severity"], 3),
                row["code"],
                row.get("idea_id", ""),
            )
        )
        return alerts

    @staticmethod
    def _idea_usage(ideas: list[IdeaRecord]) -> list[dict[str, Any]]:
        rows = [
            {
                "idea_id": idea.idea_id,
                "title": idea.title,
                "status": idea.status.value,
                "family": idea.family,
                "llm_tokens": idea.llm_tokens_spent,
                "llm_calls": idea.llm_calls,
                "gpu_hours": idea.gpu_seconds_spent / 3600.0,
            }
            for idea in ideas
        ]
        rows.sort(
            key=lambda row: (
                -row["llm_tokens"],
                -row["gpu_hours"],
                row["idea_id"],
            )
        )
        return rows


def _empty_usage_row() -> dict[str, Any]:
    return {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _add_call(row: dict[str, Any], call: Mapping[str, Any]) -> None:
    row["calls"] += 1
    if call.get("outcome") == "success":
        row["successful_calls"] += 1
    else:
        row["failed_calls"] += 1
    row["prompt_tokens"] += max(
        0,
        int(call.get("prompt_tokens", 0) or 0),
    )
    row["completion_tokens"] += max(
        0,
        int(call.get("completion_tokens", 0) or 0),
    )
    row["total_tokens"] += max(
        0,
        int(call.get("total_tokens", 0) or 0),
    )


def _table_rows(
    table: Mapping[str, Mapping[str, Any]],
    *,
    key_name: str,
) -> list[dict[str, Any]]:
    rows = [
        {key_name: key, **dict(value)}
        for key, value in table.items()
    ]
    rows.sort(
        key=lambda row: (
            -int(row["total_tokens"]),
            str(row[key_name]),
        )
    )
    return rows


def _bucket_calls(
    calls: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
    bucket_minutes: int,
) -> list[dict[str, Any]]:
    buckets = _bucket_grid(
        start=start,
        end=end,
        bucket_minutes=bucket_minutes,
    )
    lookup = {row["_start"]: row for row in buckets}
    for call in calls:
        stamp = call.get("_timestamp")
        if not isinstance(stamp, datetime):
            continue
        key = _floor_time(stamp, bucket_minutes)
        row = lookup.get(key)
        if row is None:
            continue
        row["calls"] += 1
        row["prompt_tokens"] += max(
            0,
            int(call.get("prompt_tokens", 0) or 0),
        )
        row["completion_tokens"] += max(
            0,
            int(call.get("completion_tokens", 0) or 0),
        )
        row["total_tokens"] += max(
            0,
            int(call.get("total_tokens", 0) or 0),
        )
        tier = str(call.get("tier", "unknown") or "unknown")
        row["by_tier"][tier] = row["by_tier"].get(tier, 0) + max(
            0,
            int(call.get("total_tokens", 0) or 0),
        )
    return [_public_bucket(row) for row in buckets]


def _bucket_gpu_ticks(
    ticks: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
    bucket_minutes: int,
) -> list[dict[str, Any]]:
    buckets = _bucket_grid(
        start=start,
        end=end,
        bucket_minutes=bucket_minutes,
    )
    lookup = {row["_start"]: row for row in buckets}
    for tick in ticks:
        stamp = _parse_time(tick.get("timestamp"))
        if stamp is None:
            continue
        row = lookup.get(_floor_time(stamp, bucket_minutes))
        if row is None:
            continue
        gpu = tick.get("gpu")
        gpu = gpu if isinstance(gpu, Mapping) else {}
        row["samples"] += 1
        row["utilization_sum"] += max(
            0.0,
            float(gpu.get("utilization", 0.0) or 0.0),
        )
        row["allocated_max"] = max(
            row["allocated_max"],
            max(0, int(gpu.get("allocated_gpus", 0) or 0)),
        )
        row["pending_max"] = max(
            row["pending_max"],
            max(0, int(gpu.get("pending_jobs", 0) or 0)),
        )
    _add_gpu_state_durations(
        buckets,
        ticks,
        start=start,
        end=end,
        bucket_minutes=bucket_minutes,
    )
    result: list[dict[str, Any]] = []
    for row in buckets:
        samples = row["samples"]
        result.append(
            {
                "timestamp": row["_start"].isoformat(
                    timespec="milliseconds"
                ),
                "utilization": (
                    row["utilization_sum"] / samples if samples else 0.0
                ),
                "allocated_gpus": row["allocated_max"],
                "pending_jobs": row["pending_max"],
                "samples": samples,
                "state_minutes": {
                    state: row["state_seconds"][state] / 60.0
                    for state in _GPU_STATES
                },
            }
        )
    return result


def _bucket_grid(
    *,
    start: datetime,
    end: datetime,
    bucket_minutes: int,
) -> list[dict[str, Any]]:
    first = _floor_time(start, bucket_minutes)
    final = _floor_time(end, bucket_minutes)
    step = timedelta(minutes=bucket_minutes)
    result: list[dict[str, Any]] = []
    cursor = first
    while cursor <= final:
        result.append(
            {
                "_start": cursor,
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "by_tier": {},
                "samples": 0,
                "utilization_sum": 0.0,
                "allocated_max": 0,
                "pending_max": 0,
                "state_seconds": {
                    state: 0.0 for state in _GPU_STATES
                },
            }
        )
        cursor += step
    return result


_GPU_STATES = ("running", "backlog_idle", "idle", "unobserved")


def _gpu_state_minutes(
    ticks: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> dict[str, float]:
    totals = {state: 0.0 for state in _GPU_STATES}
    for segment_start, segment_end, state in _gpu_state_segments(
        ticks,
        start=start,
        end=end,
    ):
        totals[state] += max(
            0.0,
            (segment_end - segment_start).total_seconds(),
        )
    return {
        state: seconds / 60.0
        for state, seconds in totals.items()
    }


def _add_gpu_state_durations(
    buckets: list[dict[str, Any]],
    ticks: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
    bucket_minutes: int,
) -> None:
    lookup = {row["_start"]: row for row in buckets}
    step = timedelta(minutes=bucket_minutes)
    for segment_start, segment_end, state in _gpu_state_segments(
        ticks,
        start=start,
        end=end,
    ):
        cursor = segment_start
        while cursor < segment_end:
            bucket_start = _floor_time(cursor, bucket_minutes)
            boundary = min(segment_end, bucket_start + step)
            row = lookup.get(bucket_start)
            if row is not None:
                row["state_seconds"][state] += max(
                    0.0,
                    (boundary - cursor).total_seconds(),
                )
            cursor = boundary


def _gpu_state_segments(
    ticks: list[dict[str, Any]],
    *,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, datetime, str]]:
    observations = [
        (stamp, _gpu_state(tick))
        for tick in ticks
        if (
            (stamp := _parse_time(tick.get("timestamp"))) is not None
            and start <= stamp <= end
        )
    ]
    observations.sort(key=lambda row: row[0])
    segments: list[tuple[datetime, datetime, str]] = []
    cursor = start
    state = "unobserved"
    for stamp, observed_state in observations:
        bounded = min(max(stamp, start), end)
        if bounded > cursor:
            segments.append((cursor, bounded, state))
        cursor = bounded
        state = observed_state
    if cursor < end:
        segments.append((cursor, end, state))
    return segments


def _gpu_state(tick: Mapping[str, Any]) -> str:
    gpu = tick.get("gpu")
    gpu = gpu if isinstance(gpu, Mapping) else {}
    allocated = max(0, int(gpu.get("allocated_gpus", 0) or 0))
    pending = max(0, int(gpu.get("pending_jobs", 0) or 0))
    if allocated > 0:
        return "running"
    if pending > 0:
        return "backlog_idle"
    return "idle"


def _public_bucket(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": row["_start"].isoformat(timespec="milliseconds"),
        "calls": row["calls"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
        "by_tier": row["by_tier"],
    }


def _gpu_idle_since(
    ticks: list[dict[str, Any]],
    *,
    now: datetime,
    allocated: int,
) -> datetime | None:
    if allocated > 0:
        return None
    for tick in reversed(ticks):
        gpu = tick.get("gpu")
        gpu = gpu if isinstance(gpu, Mapping) else {}
        if int(gpu.get("allocated_gpus", 0) or 0) > 0:
            stamp = _parse_time(tick.get("timestamp"))
            return stamp or now
    if ticks:
        return _parse_time(ticks[0].get("timestamp"))
    return now


def _model_price(
    model: str,
    prices: Mapping[str, Mapping[str, float]],
) -> dict[str, float] | None:
    if model in prices:
        return dict(prices[model])
    suffix = model.split("/", 1)[-1]
    for key, value in prices.items():
        if key.split("/", 1)[-1] == suffix:
            return dict(value)
    return None


def _parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _floor_time(value: datetime, bucket_minutes: int) -> datetime:
    epoch_minute = math.floor(value.timestamp() / 60)
    floored = (epoch_minute // bucket_minutes) * bucket_minutes
    return datetime.fromtimestamp(floored * 60, tz=UTC)


def _ratio(used: float, limit: float) -> float | None:
    return used / limit if limit > 0 else None


def _severity(
    ratio: float,
    *,
    warning: float,
    critical: float,
) -> str:
    if ratio >= 1.0:
        return "critical"
    if ratio >= critical:
        return "critical"
    if ratio >= warning:
        return "warning"
    return "info"


def _alert(
    *,
    severity: str,
    code: str,
    title: str,
    message: str,
    idea_id: str = "",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "code": code,
        "title": title,
        "message": message,
        "idea_id": idea_id,
    }


def _days_in_month(value: datetime) -> int:
    if value.month == 12:
        following = value.replace(
            year=value.year + 1,
            month=1,
            day=1,
        )
    else:
        following = value.replace(month=value.month + 1, day=1)
    current = value.replace(day=1)
    return (following - current).days
