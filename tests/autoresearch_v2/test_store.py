from __future__ import annotations

from pathlib import Path

from researchclaw.autoresearch_v2.models import (
    AttemptStatus,
    IdeaRecord,
    JobKind,
    JobRecord,
)
from researchclaw.autoresearch_v2.store import V2Store


def _idea() -> IdeaRecord:
    return IdeaRecord(
        idea_id="idea-test",
        title="Test idea",
        research_question="Does the mechanism work?",
        falsifiable_hypothesis="The mechanism improves accuracy.",
        primary_metric="accuracy",
        candidate={"cheap_pilot": "one GPU"},
    )


def test_sqlite_roundtrip_and_wal(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="idea-test-build",
        idea_id=idea.idea_id,
        kind=JobKind.BUILD,
    )
    store.save_job(job)

    assert store.get_idea(idea.idea_id) == idea
    assert store.get_job(job.job_id) == job
    assert list(tmp_path.glob("autoresearch.db-wal")) or store.db_path.exists()


def test_failed_candidate_never_mutates_current(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="idea-test-build",
        idea_id=idea.idea_id,
        kind=JobKind.BUILD,
    )
    store.save_job(job)

    first = store.create_attempt(job)
    candidate = store.prepare_candidate(first)
    (candidate / "main.py").write_text("print('accepted')\n", encoding="utf-8")
    store.commit_candidate(first)
    assert (store.current_dir(idea.idea_id) / "main.py").read_text() == (
        "print('accepted')\n"
    )

    job.attempt = 1
    second = store.create_attempt(job)
    rejected = store.snapshot_current(second)
    (rejected / "main.py").write_text("def broken(:\n", encoding="utf-8")
    second.status = AttemptStatus.REJECTED
    store.save_attempt(second)

    assert (store.current_dir(idea.idea_id) / "main.py").read_text() == (
        "print('accepted')\n"
    )
    assert (store.attempt_dir(second) / "candidate" / "main.py").read_text() == (
        "def broken(:\n"
    )


def test_commit_replaces_current_atomically_without_stale_files(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="idea-test-build",
        idea_id=idea.idea_id,
        kind=JobKind.BUILD,
    )
    store.save_job(job)

    first = store.create_attempt(job)
    candidate = store.prepare_candidate(first)
    (candidate / "main.py").write_text("print(1)\n", encoding="utf-8")
    (candidate / "stale.py").write_text("stale = True\n", encoding="utf-8")
    store.commit_candidate(first)

    job.attempt = 1
    second = store.create_attempt(job)
    candidate = store.prepare_candidate(second)
    (candidate / "main.py").write_text("print(2)\n", encoding="utf-8")
    store.commit_candidate(second)

    current = store.current_dir(idea.idea_id)
    assert (current / "main.py").read_text() == "print(2)\n"
    assert not (current / "stale.py").exists()
