from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from researchclaw.factory.dashboard import create_dashboard_app
from researchclaw.factory.io import tail_jsonl
from researchclaw.factory.models import Idea, IdeaStatus
from researchclaw.factory.store import FactoryStore


def test_factory_dashboard_exposes_kanban_and_controls(tmp_path: Path) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = Idea(
        idea_id="idea-dashboard",
        title="Dashboard idea",
        research_question="question",
        falsifiable_hypothesis="hypothesis",
        primary_metric="accuracy",
        status=IdeaStatus.PILOT,
    )
    store.save_idea(idea)
    run_dir = tmp_path / "ideas" / idea.idea_id / "runs" / "pipeline"
    run_dir.mkdir(parents=True)
    (run_dir / "pipeline.log").write_text(
        "first line\nsecond line\n",
        encoding="utf-8",
    )
    operational_path = (
        tmp_path
        / "ideas"
        / idea.idea_id
        / "operational_events.jsonl"
    )
    operational_path.write_text(
        '{"event":"worker_launched","timestamp":"now"}\n',
        encoding="utf-8",
    )
    app = create_dashboard_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["tick_age_sec"] is not None
        payload = client.get("/api/dashboard").json()
        assert payload["lanes"]["pilot"][0]["idea_id"] == idea.idea_id
        assert "observability" in payload
        assert "factory_tick" in payload["observability"]["latency"]
        assert "terminal_yield_rate" in payload["observability"]["outcomes"]
        timeline = client.get(
            f"/api/ideas/{idea.idea_id}/events?limit=10"
        ).json()
        assert timeline["idea_id"] == idea.idea_id
        assert timeline["events"][-1]["type"] == "idea_saved"
        log = client.get(
            f"/api/ideas/{idea.idea_id}/logs?source=pipeline&limit=1"
        )
        assert log.status_code == 200
        assert log.text == "second line"
        operational = client.get(
            f"/api/ideas/{idea.idea_id}/logs"
            "?source=operational_events&limit=1"
        )
        assert operational.status_code == 200
        assert "worker_launched" in operational.text
        assert client.get("/api/ideas/%2E%2E/events").status_code in {
            400,
            404,
        }
        assert client.post(
            "/api/control/pause", json={"reason": "test"}
        ).status_code == 200
        assert client.get("/api/dashboard").json()["controls"]["can_resume"]


def test_factory_health_reports_stale_running_tick(tmp_path: Path) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    state = store.load_state()
    state.update(
        {
            "status": "running",
            "tick": 7,
            "updated_at": "2020-01-01T00:00:00+00:00",
        }
    )
    # Preserve the intentionally stale timestamp instead of save_state(),
    # which refreshes updated_at.
    store.state_path.write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    app = create_dashboard_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/health?stale_after_sec=1").json()

    assert health["status"] == "degraded"
    assert health["factory_status"] == "running"
    assert health["tick"] == 7
    assert health["reasons"] == ["factory_tick_stale"]


def test_jsonl_tail_is_bounded_and_skips_partial_rows(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index in range(100):
            handle.write(f'{{"index": {index}}}\n')
        handle.write('{"partial":')

    rows = tail_jsonl(path, limit=3, max_bytes=256)

    assert [row["index"] for row in rows] == [97, 98, 99]


def test_observability_yield_counts_only_research_outcomes(
    tmp_path: Path,
) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    statuses = (
        IdeaStatus.COMPLETED,
        IdeaStatus.COMPLETED_NEGATIVE,
        IdeaStatus.REJECTED,
        IdeaStatus.FAILED,
        IdeaStatus.PARKED,
    )
    for index, status in enumerate(statuses):
        store.save_idea(
            Idea(
                idea_id=f"idea-{index}",
                title=f"Idea {index}",
                research_question="question",
                falsifiable_hypothesis="hypothesis",
                primary_metric="accuracy",
                status=status,
            )
        )

    payload = create_dashboard_app(tmp_path)
    with TestClient(payload) as client:
        observability = client.get("/api/dashboard").json()["observability"]

    assert observability["throughput"]["ideas_terminal"] == 2
    assert observability["outcomes"]["terminal"] == 2
    assert observability["outcomes"]["terminal_yield_rate"] == 0.4


def test_observability_latencies_are_scoped_to_each_attempt(
    tmp_path: Path,
) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    base = {
        "factory_id": store.factory_id,
        "idea_id": "idea-1",
        "item_id": "item-1",
    }
    rows = (
        ("2026-01-01T00:00:00+00:00", "work_item_queued", 1),
        ("2026-01-01T00:00:10+00:00", "work_item_started", 1),
        ("2026-01-01T00:00:20+00:00", "work_item_retry_wait", 1),
        ("2026-01-01T00:01:00+00:00", "work_item_queued", 2),
        ("2026-01-01T00:01:30+00:00", "work_item_started", 2),
        ("2026-01-01T00:02:00+00:00", "work_item_succeeded", 2),
    )
    with store.events_path.open("w", encoding="utf-8") as stream:
        for timestamp, event_type, attempt in rows:
            stream.write(
                json.dumps(
                    {
                        **base,
                        "timestamp": timestamp,
                        "type": event_type,
                        "attempt": attempt,
                    }
                )
                + "\n"
            )

    from researchclaw.factory.observability import build_factory_observability

    summary = build_factory_observability(store)

    assert summary["latency"]["queue_wait"] == {
        "samples": 2,
        "p50_sec": 20.0,
        "p95_sec": 30.0,
        "max_sec": 30.0,
    }
    assert summary["latency"]["work_item_runtime"] == {
        "samples": 2,
        "p50_sec": 20.0,
        "p95_sec": 30.0,
        "max_sec": 30.0,
    }
    assert summary["reliability"]["retries"] == 1


def test_observability_does_not_mix_lifetime_yield_with_truncated_window(
    tmp_path: Path,
) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    store.save_idea(
        Idea(
            idea_id="idea-complete",
            title="Completed Idea",
            research_question="question",
            falsifiable_hypothesis="hypothesis",
            primary_metric="accuracy",
            status=IdeaStatus.COMPLETED,
        )
    )
    with store.events_path.open("w", encoding="utf-8") as stream:
        for second in range(3):
            stream.write(
                json.dumps(
                    {
                        "factory_id": store.factory_id,
                        "timestamp": (
                            f"2026-01-01T00:00:0{second}+00:00"
                        ),
                        "type": "factory_tick",
                        "elapsed_sec": 0.1,
                    }
                )
                + "\n"
            )

    from researchclaw.factory.observability import build_factory_observability

    summary = build_factory_observability(store, max_events=2)

    assert summary["events"]["window_truncated"] is True
    assert summary["throughput"]["ideas_terminal"] == 1
    assert summary["throughput"]["ideas_terminal_per_hour"] == 0.0
    assert (
        summary["throughput"]["rate_basis"]
        == "unavailable_truncated_event_window"
    )
