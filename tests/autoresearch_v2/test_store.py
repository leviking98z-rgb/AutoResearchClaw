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


def test_create_attempt_is_not_durable_until_saved(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="idea-test-design",
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
    )
    store.save_job(job)

    attempt = store.create_attempt(job)
    assert store.get_attempt(attempt.attempt_id) is None
    assert not store.attempt_dir(attempt).exists()

    store.save_attempt(attempt)
    assert store.get_attempt(attempt.attempt_id) == attempt


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


def test_initialize_restores_interrupted_current_swap(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea_dir = store.idea_dir("idea-crash")
    idea_dir.mkdir(parents=True)
    backup = idea_dir / ".current.previous"
    backup.mkdir()
    (backup / "main.py").write_text("print('old')\n", encoding="utf-8")
    staged = idea_dir / ".current.attempt.tmp"
    staged.mkdir()
    (staged / "main.py").write_text("partial\n", encoding="utf-8")

    V2Store(tmp_path).initialize()
    assert (idea_dir / "current" / "main.py").read_text() == (
        "print('old')\n"
    )
    assert not backup.exists()
    assert not staged.exists()


def test_maintenance_rotates_logs_and_prunes_old_failed_candidates(
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
    for number in range(3):
        job.attempt = number
        attempt = store.create_attempt(job)
        candidate = store.prepare_candidate(attempt)
        (candidate / "main.py").write_text("broken\n", encoding="utf-8")
        attempt.status = AttemptStatus.REJECTED
        store.save_attempt(attempt)
    store.events_path.write_text("x" * 20, encoding="utf-8")

    result = store.maintain(
        event_jsonl_max_bytes=10,
        llm_audit_max_bytes=10,
        keep_failed_attempts_per_job=1,
    )
    assert result["rotated_logs"] == 1
    assert result["pruned_candidates"] == 2
    assert store.events_path.with_suffix(".jsonl.1").is_file()
