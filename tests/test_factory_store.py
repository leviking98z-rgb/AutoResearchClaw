from __future__ import annotations

import json
from pathlib import Path

from researchclaw.factory.models import (
    Idea,
    IdeaStatus,
    ResourceRequest,
    WorkItem,
    WorkItemStatus,
    WorkKind,
)
from researchclaw.factory.store import FactoryStore


def _idea(identifier: str, title: str) -> Idea:
    return Idea(
        idea_id=identifier,
        title=title,
        research_question="question",
        falsifiable_hypothesis="hypothesis",
        primary_metric="metric",
        priority=0.5,
    )


def test_store_persists_reservoir_and_ideas_without_duplicates(
    tmp_path: Path,
) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    first = _idea("idea-a", "Same   Title")
    duplicate_title = _idea("idea-b", "same title")

    assert store.add_candidates([first, duplicate_title]) == [first]
    assert [idea.idea_id for idea in store.load_reservoir()] == ["idea-a"]

    admitted = store.remove_candidate("idea-a")
    assert admitted is not None
    admitted.status = IdeaStatus.SCREENING
    store.save_idea(admitted)

    restarted = FactoryStore(tmp_path)
    restarted.initialize()
    assert restarted.get_idea("idea-a") == admitted
    assert restarted.load_reservoir() == []
    events = [
        json.loads(line)
        for line in (
            tmp_path / "ideas" / "idea-a" / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "idea_saved"
    assert events[-1]["status"] == "screening"


def test_store_writes_itemized_per_idea_work_log(tmp_path: Path) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = _idea("idea-items", "Itemized idea")
    idea.status = IdeaStatus.PILOT
    store.save_idea(idea)
    item = WorkItem(
        item_id="idea-items-pilot",
        idea_id=idea.idea_id,
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=2,
            max_gpus=2,
            cpus=8,
            timeout_sec=600,
        ),
        status=WorkItemStatus.RUNNING,
        attempt=1,
        result={"task_id": "task-1"},
    )

    store.save_work_item(item, event_type="gpu_work_item_started")

    events = [
        json.loads(line)
        for line in (
            tmp_path / "ideas" / idea.idea_id / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    work_event = events[-1]
    assert work_event["type"] == "gpu_work_item_started"
    assert work_event["item_id"] == item.item_id
    assert work_event["resources"]["preferred_gpus"] == 2
    assert work_event["result"]["task_id"] == "task-1"


def test_control_files_survive_restart(tmp_path: Path) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    store.set_control("pause", "test")
    assert FactoryStore(tmp_path).control_requested("pause") is True
    store.clear_control("pause")
    assert store.control_requested("pause") is False
