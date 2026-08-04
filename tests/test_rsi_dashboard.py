"""Tests for the campaign-aware RSI web dashboard."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from researchclaw.rsi.dashboard import CampaignDashboard, create_dashboard_app


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "cycle-0002"
    run.mkdir(parents=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    _write_json(
        campaign / "campaign.json",
        {
            "campaign_id": "test-rsi",
            "created_at": now,
            "run_policy": {"continuous": True},
        },
    )
    _write_json(
        campaign / "state.json",
        {
            "campaign_id": "test-rsi",
            "status": "running",
            "phase": "pipeline",
            "active_cycle": 2,
            "active_run_dir": str(run),
            "completed_cycles": 1,
            "successful_cycles": 0,
            "failed_cycles": 1,
            "continuous": True,
            "selected_topic": "Calibration-aware RSI gates",
            "updated_at": now,
        },
    )
    _write_json(
        campaign / "supervisor_heartbeat.json",
        {
            "timestamp": now,
            "cycle": 2,
            "phase": "pipeline",
            "status": "running",
        },
    )
    _write_json(
        campaign / "monitor_snapshot.json",
        {
            "generated_at": now,
            "overall": "degraded",
            "checks": {
                "supervisor": {
                    "status": "ok",
                    "detail": "heartbeat fresh",
                    "observed_at": now,
                },
                "progress": {
                    "status": "ok",
                    "detail": "durable progress",
                    "observed_at": now,
                },
                "pool": {
                    "status": "ok",
                    "detail": "Ray ready",
                    "observed_at": now,
                    "data": {
                        "pool": {
                            "pool_id": "test-pool",
                            "ray_started": True,
                            "claimed": True,
                            "resources": {
                                "total_gpu": 32,
                                "available_gpu": 31,
                                "alive_nodes": 4,
                            },
                        }
                    },
                },
            },
        },
    )
    _write_json(
        run / "checkpoint.json",
        {
            "last_completed_stage": 15,
            "last_completed_name": "RESEARCH_DECISION",
            "run_id": "rc-test",
            "timestamp": now,
        },
    )
    _write_json(
        run / "selected_topic.json",
        {
            "id": "calibration-gates",
            "title": "Calibration-aware RSI gates",
            "research_question": "Can early calibration drift predict collapse?",
            "falsifiable_hypothesis": "Drift predicts later regression.",
            "primary_metric": "held-out accuracy",
        },
    )
    (run / "pipeline.log").write_text(
        "[rc-test] Stage 15/23 RESEARCH_DECISION — done (5.0s)\n"
        "[rc-test] Decision: REFINE → rollback to ITERATIVE_REFINE (attempt 2/2)\n"
        "[rc-test] Stage 13/23 ITERATIVE_REFINE — running...",
        encoding="utf-8",
    )
    (campaign / "events.jsonl").write_text(
        json.dumps({"timestamp": now, "type": "cycle_started"}) + "\n",
        encoding="utf-8",
    )
    return campaign, run


def test_collect_distinguishes_current_stage_from_checkpoint(
    tmp_path: Path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    payload = CampaignDashboard(
        campaign,
        pool_root=tmp_path / "pools",
        control_enabled=False,
    ).collect()

    assert payload["progress"]["current_stage"]["number"] == 13
    assert payload["progress"]["current_stage"]["name"] == "ITERATIVE_REFINE"
    assert payload["progress"]["checkpoint"]["number"] == 15
    assert payload["progress"]["current_stage"]["rollback"]["decision"] == "REFINE"
    stage_by_number = {
        item["number"]: item for item in payload["progress"]["stages"]
    }
    assert stage_by_number[13]["status"] == "running"
    assert stage_by_number[14]["status"] == "revisit"
    assert stage_by_number[15]["status"] == "revisit"


def test_iso_heartbeat_and_degraded_monitor_do_not_mark_core_dead(
    tmp_path: Path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    payload = CampaignDashboard(
        campaign,
        pool_root=tmp_path / "pools",
        control_enabled=False,
    ).collect()

    assert payload["campaign"]["heartbeat_fresh"] is True
    assert payload["monitor"]["overall"] == "degraded"
    assert payload["monitor"]["core_alive"] is True
    assert payload["monitor"]["infrastructure"]["gpu_total"] == 32
    assert payload["monitor"]["infrastructure"]["alive_nodes"] == 4


def test_current_task_reports_download_progress(tmp_path: Path) -> None:
    campaign, run = _campaign(tmp_path)
    metadata_dir = run / "stage-13" / "sandbox"
    pool_root = tmp_path / "pools"
    pool_id = "test-pool"
    task_id = "rc-pool-abc123"
    task_dir = pool_root / pool_id / "tasks" / task_id
    task_dir.mkdir(parents=True)
    config = tmp_path / "pool.yaml"
    config.write_text(
        f"clusterbridge_pool:\n  pool_id: {pool_id}\n  log_root: {pool_root}\n",
        encoding="utf-8",
    )
    _write_json(
        metadata_dir / ".clusterbridge_pool_task.json",
        {
            "task_id": task_id,
            "state": "starting",
            "pool_config": str(config),
        },
    )
    (task_dir / "stdout.log").write_text(
        "Running condition: CAAG\n--- Seed 0 (1/3) ---\n",
        encoding="utf-8",
    )
    (task_dir / "stderr.log").write_text(
        "  25%|██▌       | 42.5M/170M [00:10<00:30, 4.0MB/s]\n",
        encoding="utf-8",
    )

    payload = CampaignDashboard(
        campaign,
        pool_root=pool_root,
        control_enabled=False,
    ).collect()

    task = payload["experiment"]
    assert task["task_id"] == task_id
    assert task["progress"]["condition"] == "CAAG"
    assert task["progress"]["seed"] == "0 (1/3)"
    assert task["progress"]["download"]["percent"] == 25.0


def test_pool_journal_terminal_event_overrides_task_started(
    tmp_path: Path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    pool = tmp_path / "pools" / "test-pool"
    task_id = "rc-pool-abc123"
    task_dir = pool / "tasks" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "stdout.log").write_text("done\n", encoding="utf-8")
    (task_dir / "stderr.log").write_text("", encoding="utf-8")
    (pool / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "task_started",
                        "task_id": task_id,
                        "time": "2026-08-04T03:00:00+00:00",
                    }
                ),
                json.dumps(
                    {
                        "event": "task_finished",
                        "task_id": task_id,
                        "returncode": 0,
                        "time": "2026-08-04T03:10:00+00:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = CampaignDashboard(
        campaign,
        pool_root=tmp_path / "pools",
        control_enabled=False,
    ).collect()

    assert payload["experiment"]["task_id"] == task_id
    assert payload["experiment"]["state"] == "finished"
    assert payload["experiment"]["returncode"] == 0


def test_api_serves_frontend_events_logs_and_safe_artifacts(
    tmp_path: Path,
) -> None:
    campaign, run = _campaign(tmp_path)
    artifact = run / "stage-14" / "analysis.md"
    artifact.parent.mkdir()
    artifact.write_text("# Analysis", encoding="utf-8")
    app = create_dashboard_app(
        campaign,
        pool_root=tmp_path / "pools",
        control_enabled=False,
    )

    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/api/dashboard").status_code == 200
        assert client.get("/api/events?limit=10").json()["events"][0]["type"] == "cycle_started"
        assert "Stage 13/23" in client.get(
            "/api/logs?source=pipeline&tail=20"
        ).json()["text"]
        artifacts = client.get("/api/artifacts").json()["artifacts"]
        assert any(item["path"] == "stage-14/analysis.md" for item in artifacts)
        assert client.get("/api/artifacts/stage-14/analysis.md").status_code == 200
        assert client.get("/api/artifacts/../../campaign.json").status_code in {
            400,
            404,
        }
        assert client.post(
            "/api/control/pause", json={"reason": "test"}
        ).status_code == 403


def test_pause_is_cooperative_and_does_not_create_stop_marker(
    tmp_path: Path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    dashboard = CampaignDashboard(
        campaign,
        pool_root=tmp_path / "pools",
        control_enabled=True,
    )

    result = dashboard.pause("test maintenance")

    assert result["accepted"] is True
    assert (campaign / "control" / "pause").is_file()
    assert (campaign / "pause.request.json").is_file()
    assert not (campaign / "control" / "stop").exists()
