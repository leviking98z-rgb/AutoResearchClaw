from __future__ import annotations

import json
from pathlib import Path

from researchclaw.factory.io import append_jsonl
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
    global_events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(
        event["type"] == "candidate_deduplicated"
        and event["idea_id"] == "idea-b"
        and event["reason_code"] == "duplicate_title"
        for event in global_events
    )
    events = [
        json.loads(line)
        for line in (
            tmp_path / "ideas" / "idea-a" / "events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "idea_saved"
    assert events[-1]["status"] == "screening"


def test_add_candidates_deduplicates_id_of_persisted_active_idea(
    tmp_path: Path,
) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    original = _idea("idea-active", "Original active study")
    original.status = IdeaStatus.SCREENING
    store.save_idea(original)
    duplicate_id = _idea("idea-active", "Different generated title")

    assert store.add_candidates([duplicate_id]) == []
    assert store.load_reservoir() == []
    assert store.get_idea(original.idea_id) == original

    events = [
        json.loads(line)
        for line in store.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["type"] == "candidate_deduplicated"
        and event["idea_id"] == original.idea_id
        and event["title"] == duplicate_id.title
        and event["reason_code"] == "duplicate_id"
        for event in events
    )


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


def test_queued_work_item_is_reconsidered(tmp_path: Path) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = _idea("idea-queued", "Queued idea")
    idea.status = IdeaStatus.SCREENING
    store.save_idea(idea)
    item = WorkItem(
        item_id="idea-queued-screen",
        idea_id=idea.idea_id,
        kind=WorkKind.PIPELINE,
        profile="screen",
        status=WorkItemStatus.QUEUED,
    )
    store.save_work_item(item)

    ready = store.ready_work_items()

    assert [current.item_id for current in ready] == [item.item_id]
    restored = store.get_work_item(item.item_id)
    assert restored is not None
    assert restored.status is WorkItemStatus.READY


def test_process_writer_lock_rejects_second_store(tmp_path: Path) -> None:
    import pytest

    first = FactoryStore(tmp_path)
    second = FactoryStore(tmp_path)
    first.initialize()
    second.initialize()
    first.acquire_writer_lock()
    try:
        with pytest.raises(RuntimeError, match="another Factory writer"):
            second.acquire_writer_lock()
    finally:
        first.release_writer_lock()
    second.acquire_writer_lock()
    second.release_writer_lock()


def test_append_jsonl_redacts_nested_credentials(tmp_path: Path) -> None:
    path = tmp_path / "operational.jsonl"

    append_jsonl(
        path,
        {
            "event": "provider_failed",
            "api_key": "secret-key",
            "request": {
                "Authorization": "Bearer secret",
                "model": "example-model",
            },
        },
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["api_key"] == "[REDACTED]"
    assert record["request"]["Authorization"] == "[REDACTED]"
    assert record["request"]["model"] == "example-model"
