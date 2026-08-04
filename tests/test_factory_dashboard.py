from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from researchclaw.factory.dashboard import create_dashboard_app
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
    app = create_dashboard_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        payload = client.get("/api/dashboard").json()
        assert payload["lanes"]["pilot"][0]["idea_id"] == idea.idea_id
        assert client.post(
            "/api/control/pause", json={"reason": "test"}
        ).status_code == 200
        assert client.get("/api/dashboard").json()["controls"]["can_resume"]
