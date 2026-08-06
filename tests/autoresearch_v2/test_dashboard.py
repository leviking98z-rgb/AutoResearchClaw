from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from researchclaw.autoresearch_v2.config import (
    BudgetConfig,
    UsageMonitoringConfig,
    V2Config,
)
from researchclaw.autoresearch_v2.dashboard import (
    V2Dashboard,
    configured_gpu_total,
)
from researchclaw.autoresearch_v2.models import (
    IdeaRecord,
    IdeaStatus,
    JobKind,
    JobRecord,
)
from researchclaw.autoresearch_v2.store import V2Store
from researchclaw.autoresearch_v2.usage import UsageMonitor


def test_dashboard_exposes_lanes_usage_and_controls(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    store.save_idea(
        IdeaRecord(
            idea_id="idea-dashboard",
            title="Dashboard idea",
            research_question="Does it work?",
            falsifiable_hypothesis="It improves accuracy.",
            primary_metric="accuracy",
            candidate={},
            status=IdeaStatus.PILOTING,
            gpu_seconds_spent=7200,
            llm_tokens_spent=1234,
        )
    )
    store.event("controller_tick", ideas_total=1)
    dashboard = V2Dashboard(
        store,
        gpu_total=8,
        target_utilization=0.9,
    )
    store.writer_lock_path.write_text(
        json.dumps({"pid": 1}),
        encoding="utf-8",
    )
    value = dashboard.collect()
    assert value["controller"]["gpu_hours_total"] == 2
    assert value["lanes"]["pilot"][0]["idea_id"] == "idea-dashboard"
    assert value["controls"]["can_pause"]

    store.set_control("pause", "test")
    paused = dashboard.collect()
    assert paused["controller"]["status"] == "paused"
    assert paused["controls"]["can_resume"]


def test_dashboard_reports_stopped_without_live_controller(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    store.writer_lock_path.write_text(
        json.dumps({"pid": 999_999_999}),
        encoding="utf-8",
    )
    value = V2Dashboard(store).collect()
    assert value["controller"]["status"] == "stopped"
    assert not value["controls"]["can_pause"]
    assert not value["controls"]["can_stop"]


def test_dashboard_caches_usage_for_polling_clients(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    monitor = Mock()
    monitor.collect.return_value = {"marker": "usage"}
    dashboard = V2Dashboard(
        store,
        usage_monitor=monitor,
        usage_cache_ttl_sec=60,
    )

    assert dashboard.usage(hours=24)["marker"] == "usage"
    assert dashboard.usage(hours=24)["marker"] == "usage"

    monitor.collect.assert_called_once_with(
        hours=24,
        bucket_minutes=None,
    )


def test_config_contains_shared_workspace_and_infohub_defaults() -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "population": {
                    "active_idea_target": 1,
                    "max_active_ideas": 1,
                }
            }
        }
    )
    assert config.gpu.shared_workspace_root.startswith("/root/shared/")
    assert config.literature.enabled
    assert "8077" in config.literature.url


def test_dashboard_uses_desired_resource_manager_gpu_capacity(
    tmp_path: Path,
) -> None:
    shared_root = tmp_path / "shared-runs"
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "state_dir": str(shared_root / "canary"),
                "population": {
                    "active_idea_target": 1,
                    "max_active_ideas": 1,
                },
                "gpu": {
                    "enabled": True,
                    "mode": "resource_manager",
                    "pool_config": "",
                    "shared_workspace_root": str(shared_root),
                    "resource_manager": {
                        "owner": "019fc877-7045-7a40-935d-d2bef7883945",
                        "min_gpus": 8,
                        "desired_gpus": 32,
                        "max_gpus": 64,
                    },
                },
            }
        }
    )

    assert configured_gpu_total(config) == 32


def test_health_uses_controller_tick_not_unrelated_event(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    store.event("controller_tick")
    store.event("dashboard_control_used")
    health = V2Dashboard(store).health(stale_after_sec=120)
    assert health["status"] == "ok"


def test_health_detects_stale_controller_lock(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    store.event("controller_tick")
    store.writer_lock_path.write_text(
        json.dumps({"pid": 999_999_999}),
        encoding="utf-8",
    )
    health = V2Dashboard(store).health(stale_after_sec=120)
    assert health["status"] == "degraded"
    assert health["writer_lock_state"] == "stale"
    assert "controller_lock_stale" in health["reasons"]


def test_usage_monitor_aggregates_models_trends_budgets_and_alerts(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = IdeaRecord(
        idea_id="idea-expensive",
        title="Expensive idea",
        research_question="Does it work?",
        falsifiable_hypothesis="It works.",
        primary_metric="accuracy",
        candidate={},
        status=IdeaStatus.PILOTING,
        llm_tokens_spent=850,
        gpu_seconds_spent=1800,
        llm_calls=2,
    )
    store.save_idea(idea)
    store.save_job(
        JobRecord(
            job_id="idea-expensive-pilot",
            idea_id=idea.idea_id,
            kind=JobKind.PILOT,
            requires_gpu=True,
        )
    )
    audit = tmp_path / "llm-audit" / "worker" / "calls.jsonl"
    audit.parent.mkdir(parents=True)
    now = datetime.now(UTC)
    audit.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": (
                            now - timedelta(minutes=30)
                        ).isoformat(),
                        "role": "coding_engineer",
                        "tier": "worker",
                        "outcome": "success",
                        "model": "codebuddy/claude-sonnet-5",
                        "prompt_tokens": 60_000,
                        "completion_tokens": 40_000,
                        "total_tokens": 100_000,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": now.isoformat(),
                        "role": "coding_engineer",
                        "tier": "worker",
                        "outcome": "error",
                        "error": "offline",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for minutes in (40, 20, 0):
        store.event(
            "controller_tick",
            gpu={
                "allocated_gpus": 0,
                "pending_jobs": 1,
                "utilization": 0.0,
            },
            observed_at=(now - timedelta(minutes=minutes)).isoformat(),
        )
    monitor = UsageMonitor(
        store=store,
        budgets=BudgetConfig(
            max_llm_tokens_per_idea=1000,
            pilot_gpu_hours=1,
            scale_gpu_hours=1,
        ),
        config=UsageMonitoringConfig(
            history_hours=24,
            bucket_minutes=60,
            warning_threshold=0.5,
            critical_threshold=0.8,
            token_burn_warning_per_hour=500,
            single_call_token_warning=100_000,
            gpu_idle_warning_minutes=0,
            monthly_token_budget=150_000,
            model_prices={
                "claude-sonnet-5": {
                    "input_per_million_usd": 3,
                    "output_per_million_usd": 15,
                }
            },
        ),
        gpu_total=8,
    )

    usage = monitor.collect()

    assert usage["llm"]["totals"]["calls"] == 2
    assert usage["llm"]["totals"]["failed_calls"] == 1
    assert usage["llm"]["totals"]["total_tokens"] == 100_000
    assert usage["llm"]["by_tier"][0]["tier"] == "worker"
    assert usage["costs"]["coverage"] == 0.5
    assert usage["costs"]["total_estimated_usd"] > 0
    assert usage["budgets"]["ideas_near_limit"][0]["severity"] == "critical"
    codes = {alert["code"] for alert in usage["alerts"]}
    assert "idea_budget_pressure" in codes
    assert "token_burn_rate" in codes
    assert "single_call_token_spike" in codes
    assert usage["llm"]["trend"]
    assert usage["gpu"]["trend"]
    assert usage["gpu"]["state_minutes"]["backlog_idle"] > 0
    assert usage["gpu"]["oldest_pending_job_age_minutes"] >= 0


def test_usage_monitor_reads_appended_audit_data_incrementally(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    audit = tmp_path / "llm-audit" / "decision" / "calls.jsonl"
    audit.parent.mkdir(parents=True)
    first = {
        "timestamp": datetime.now(UTC).isoformat(),
        "outcome": "success",
        "model": "model-a",
        "total_tokens": 10,
    }
    second = {
        "timestamp": datetime.now(UTC).isoformat(),
        "outcome": "success",
        "model": "model-a",
        "total_tokens": 20,
    }
    audit.write_text(json.dumps(first) + "\n", encoding="utf-8")
    monitor = UsageMonitor(
        store=store,
        budgets=BudgetConfig(),
        config=UsageMonitoringConfig(),
    )

    assert monitor.collect()["llm"]["totals"]["total_tokens"] == 10
    with audit.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(second) + "\n")
    assert monitor.collect()["llm"]["totals"]["total_tokens"] == 30


def test_usage_monitor_keeps_rotated_audit_history(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    audit = tmp_path / "llm-audit" / "worker" / "calls.jsonl"
    audit.parent.mkdir(parents=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "outcome": "success",
        "model": "model-a",
        "total_tokens": 10,
    }
    audit.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monitor = UsageMonitor(
        store=store,
        budgets=BudgetConfig(),
        config=UsageMonitoringConfig(),
    )
    assert monitor.collect()["llm"]["totals"]["calls"] == 1

    audit.replace(audit.with_suffix(".jsonl.1"))
    record["total_tokens"] = 20
    audit.write_text(json.dumps(record) + "\n", encoding="utf-8")

    usage = monitor.collect()
    assert usage["llm"]["totals"]["calls"] == 2
    assert usage["llm"]["totals"]["total_tokens"] == 30


def test_usage_gpu_window_is_not_limited_to_latest_5000_events(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    old = datetime.now(UTC) - timedelta(days=6)
    with store.connect() as conn:
        conn.execute(
            """
            INSERT INTO events(
                timestamp,event_type,idea_id,job_id,attempt_id,payload_json
            ) VALUES(?,?,?,?,?,?)
            """,
            (
                old.isoformat(),
                "controller_tick",
                None,
                None,
                None,
                json.dumps(
                    {
                        "gpu": {
                            "allocated_gpus": 1,
                            "pending_jobs": 0,
                            "utilization": 0.5,
                        }
                    }
                ),
            ),
        )
        conn.executemany(
            """
            INSERT INTO events(
                timestamp,event_type,idea_id,job_id,attempt_id,payload_json
            ) VALUES(?,?,?,?,?,?)
            """,
            [
                (
                    datetime.now(UTC).isoformat(),
                    "noise",
                    None,
                    None,
                    None,
                    "{}",
                )
                for _ in range(5001)
            ],
        )
    monitor = UsageMonitor(
        store=store,
        budgets=BudgetConfig(),
        config=UsageMonitoringConfig(history_hours=168),
        gpu_total=2,
    )

    usage = monitor.collect()

    assert any(
        row["samples"] == 1 and row["allocated_gpus"] == 1
        for row in usage["gpu"]["trend"]
    )


def test_usage_monitor_caps_trend_cardinality(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    monitor = UsageMonitor(
        store=store,
        budgets=BudgetConfig(),
        config=UsageMonitoringConfig(bucket_minutes=1),
    )

    usage = monitor.collect(hours=24 * 366, bucket_minutes=1)

    assert usage["window"]["bucket_minutes"] >= 264
    assert len(usage["llm"]["trend"]) <= 2001
