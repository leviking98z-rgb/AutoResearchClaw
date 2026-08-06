from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from researchclaw.autoresearch_v2.models import (
    AttemptStatus,
    IdeaRecord,
    JobKind,
    JobRecord,
    JobStatus,
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


def test_database_can_live_outside_shared_artifact_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared"
    database = tmp_path / "local" / "autoresearch.db"
    store = V2Store(root, db_path=database)
    store.initialize()
    store.save_idea(_idea())

    assert database.is_file()
    assert not (root / "autoresearch.db").exists()
    assert (root / "ideas" / "idea-test" / "idea.json").is_file()


def test_database_backup_and_restore_roundtrip(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    database = tmp_path / "local" / "autoresearch.db"
    backup = root / "autoresearch.db.backup"
    store = V2Store(
        root,
        db_path=database,
        db_backup_path=backup,
    )
    store.initialize()
    store.save_idea(_idea())

    assert store.backup_database() == backup
    assert backup.is_file()
    database.unlink()

    restored = V2Store(
        root,
        db_path=database,
        db_backup_path=backup,
    )
    restored.initialize(recover_filesystem=False)

    restored_idea = restored.get_idea("idea-test")
    assert restored_idea is not None
    assert restored_idea.idea_id == "idea-test"
    assert restored_idea.title == "Test idea"


def test_database_backup_loop_runs_without_controller_tick(
    tmp_path: Path,
) -> None:
    root = tmp_path / "shared"
    database = tmp_path / "local" / "autoresearch.db"
    backup = root / "autoresearch.db.backup"
    store = V2Store(
        root,
        db_path=database,
        db_backup_path=backup,
        backup_interval_sec=0.01,
    )
    store.initialize()
    store.save_idea(_idea())
    store.start_database_backup_loop()
    try:
        deadline = time.monotonic() + 2.0
        while not backup.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert backup.is_file()
        assert store.database_backup_status()["running"] is True
    finally:
        store.stop_database_backup_loop()


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


def test_concurrent_commits_do_not_destroy_staged_snapshot(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    jobs = [
        JobRecord(
            job_id=f"idea-test-build-{index}",
            idea_id=idea.idea_id,
            kind=JobKind.BUILD,
        )
        for index in range(2)
    ]
    for job in jobs:
        store.save_job(job)
    attempts = [store.create_attempt(job) for job in jobs]
    for index, attempt in enumerate(attempts):
        candidate = store.prepare_candidate(attempt)
        (candidate / "main.py").write_text(
            f"print({index})\n",
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        committed = list(executor.map(store.commit_candidate, attempts))

    assert all(path == store.current_dir(idea.idea_id) for path in committed)
    assert (
        store.current_dir(idea.idea_id) / "main.py"
    ).read_text(encoding="utf-8") in {"print(0)\n", "print(1)\n"}
    assert not list(store.idea_dir(idea.idea_id).glob(".current.*.tmp"))
    assert not list(store.idea_dir(idea.idea_id).glob(".current.previous*"))


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


def test_read_only_initialize_does_not_remove_live_staging(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea_dir = store.idea_dir("idea-live")
    idea_dir.mkdir(parents=True)
    staged = idea_dir / ".current.attempt.tmp"
    staged.mkdir()
    (staged / "main.py").write_text("in progress\n", encoding="utf-8")

    V2Store(tmp_path).initialize(recover_filesystem=False)

    assert (staged / "main.py").read_text(encoding="utf-8") == "in progress\n"


def test_explicit_recovery_requires_writer_lock(tmp_path: Path) -> None:
    store = V2Store(tmp_path)
    store.initialize(recover_filesystem=False)

    try:
        store.recover_filesystem_commits()
    except RuntimeError as exc:
        assert "writer lock" in str(exc)
    else:
        raise AssertionError("recovery unexpectedly ran without writer lock")

    store.acquire_writer_lock()
    store.recover_filesystem_commits()
    store.release_writer_lock()


def test_writer_recovery_quarantines_candidate_without_attempt_row(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="idea-test-pilot",
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
    )
    store.save_job(job)
    attempt = store.create_attempt(job)
    candidate = store.prepare_candidate(attempt)
    (candidate / "partial.json").write_text("{}", encoding="utf-8")
    assert store.get_attempt(attempt.attempt_id) is None

    store.acquire_writer_lock()
    try:
        store.recover_filesystem_commits()
    finally:
        store.release_writer_lock()

    assert not candidate.exists()
    quarantined = (
        store.attempt_dir(attempt) / "candidate.interrupted-orphan"
    )
    assert (quarantined / "partial.json").read_text(encoding="utf-8") == "{}"
    assert store.snapshot_current(attempt).is_dir()
    events = store.list_events(limit=10)
    assert any(
        event["event_type"]
        == "orphan_attempt_candidate_quarantined"
        and event["attempt_id"] == attempt.attempt_id
        for event in events
    )


def test_writer_recovery_keeps_durable_attempt_candidate(
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
    attempt = store.create_attempt(job)
    candidate = store.prepare_candidate(attempt)
    (candidate / "partial.json").write_text("{}", encoding="utf-8")
    store.save_attempt(attempt)

    store.acquire_writer_lock()
    try:
        store.recover_filesystem_commits()
    finally:
        store.release_writer_lock()

    assert (candidate / "partial.json").read_text(encoding="utf-8") == "{}"
    assert not list(
        store.attempt_dir(attempt).glob("candidate.interrupted-orphan*")
    )


def test_retry_workspace_recovery_quarantines_reused_attempt_candidate(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="idea-test-pilot",
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.RETRY_WAIT,
        attempt=3,
        attempt_id="",
    )
    store.save_job(job)
    attempt = store.create_attempt(job)
    attempt.status = AttemptStatus.RUNNING
    store.save_attempt(attempt)
    candidate = store.snapshot_current(attempt)
    (candidate / "partial.json").write_text("{}", encoding="utf-8")

    store.acquire_writer_lock()
    try:
        store.recover_retry_candidate_workspaces()
    finally:
        store.release_writer_lock()

    assert not candidate.exists()
    quarantined = (
        store.attempt_dir(attempt) / "candidate.interrupted-retry"
    )
    assert (quarantined / "partial.json").read_text(encoding="utf-8") == "{}"
    assert store.snapshot_current(attempt).is_dir()
    events = store.list_events(limit=10)
    assert any(
        event["event_type"]
        == "retry_attempt_candidate_quarantined"
        and event["attempt_id"] == attempt.attempt_id
        for event in events
    )


def test_retry_workspace_recovery_preserves_accepted_candidate(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    job = JobRecord(
        job_id="idea-test-pilot",
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.RETRY_WAIT,
    )
    store.save_job(job)
    attempt = store.create_attempt(job)
    candidate = store.prepare_candidate(attempt)
    (candidate / "result.json").write_text("{}", encoding="utf-8")
    attempt.status = AttemptStatus.ACCEPTED
    store.save_attempt(attempt)

    store.acquire_writer_lock()
    try:
        store.recover_retry_candidate_workspaces()
    finally:
        store.release_writer_lock()

    assert (candidate / "result.json").is_file()
    assert not list(
        store.attempt_dir(attempt).glob("candidate.interrupted-retry*")
    )


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
