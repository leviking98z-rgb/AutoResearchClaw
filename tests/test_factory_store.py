from __future__ import annotations

from pathlib import Path

from researchclaw.factory.models import Idea, IdeaStatus
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


def test_control_files_survive_restart(tmp_path: Path) -> None:
    store = FactoryStore(tmp_path)
    store.initialize()
    store.set_control("pause", "test")
    assert FactoryStore(tmp_path).control_requested("pause") is True
    store.clear_control("pause")
    assert store.control_requested("pause") is False
