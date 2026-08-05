from __future__ import annotations

from pathlib import Path

from researchclaw.autoresearch_v2.config import V2Config
from researchclaw.autoresearch_v2.dashboard import V2Dashboard
from researchclaw.autoresearch_v2.models import IdeaRecord, IdeaStatus
from researchclaw.autoresearch_v2.store import V2Store


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
    value = dashboard.collect()
    assert value["controller"]["gpu_hours_total"] == 2
    assert value["lanes"]["pilot"][0]["idea_id"] == "idea-dashboard"
    assert value["controls"]["can_pause"]

    store.set_control("pause", "test")
    paused = dashboard.collect()
    assert paused["controller"]["status"] == "paused"
    assert paused["controls"]["can_resume"]


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
