"""SQLite source of truth plus immutable attempt directories."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import (
    AttemptRecord,
    AttemptStatus,
    IdeaRecord,
    IdeaStatus,
    JobRecord,
    JobStatus,
    utc_now,
)


class V2Store:
    """Single-database durable store with content-addressed attempt commits."""

    def __init__(
        self,
        root: str | Path,
        *,
        db_path: str | Path | None = None,
        db_backup_path: str | Path | None = None,
        backup_interval_sec: float = 60.0,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.db_path = (
            Path(db_path).expanduser().resolve()
            if db_path is not None and str(db_path).strip()
            else self.root / "autoresearch.db"
        )
        self.db_backup_path = (
            Path(db_backup_path).expanduser().resolve()
            if db_backup_path is not None and str(db_backup_path).strip()
            else None
        )
        if self.db_backup_path == self.db_path:
            raise ValueError("database backup path must differ from database")
        self.backup_interval_sec = max(
            0.001,
            float(backup_interval_sec),
        )
        self.ideas_root = self.root / "ideas"
        self.events_path = self.root / "events.jsonl"
        self.writer_lock_path = self.root / "controller.lock"
        self.control_dir = self.root / "control"
        self._writer_lock_stream: Any | None = None
        self._event_lock = threading.Lock()
        self._backup_lock = threading.Lock()
        self._backup_thread: threading.Thread | None = None
        self._backup_stop = threading.Event()
        self._last_backup_error = ""
        self._last_backup_at = ""
        # Job executors commit accepted snapshots from a shared thread pool.
        # Serialize filesystem swaps in-process so dashboard/status
        # initialization cannot treat an active ``.current.*.tmp`` directory
        # as crash debris while a worker is committing it.
        self._commit_lock = threading.Lock()

    def initialize(self, *, recover_filesystem: bool = True) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ideas_root.mkdir(parents=True, exist_ok=True)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._restore_database_backup_if_needed()
        if recover_filesystem:
            self._recover_filesystem_commits()
        # The controller may already be doing a large WAL transaction while a
        # read-only process (dashboard/status) starts.  Setting journal_mode is
        # a database-wide write and can therefore fail with "database is
        # locked" even though the schema already exists.  Only bootstrap WAL
        # before the database exists; normal initializers create/check schema
        # without trying to rewrite the journal mode.
        bootstrap = not self.db_path.exists()
        with self.connect() as conn:
            if bootstrap:
                conn.execute("PRAGMA journal_mode=WAL;")
            conn.executescript(
                """
                PRAGMA synchronous=NORMAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS ideas (
                    idea_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ideas_status_priority
                    ON ideas(status, priority DESC, updated_at);
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idea_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    FOREIGN KEY(idea_id) REFERENCES ideas(idea_id)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_kind
                    ON jobs(status, kind, updated_at);
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    idea_id TEXT NOT NULL,
                    number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    UNIQUE(job_id, number),
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id),
                    FOREIGN KEY(idea_id) REFERENCES ideas(idea_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    idea_id TEXT,
                    job_id TEXT,
                    attempt_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_type_timestamp
                    ON events(event_type, timestamp);
                """
            )

    def _restore_database_backup_if_needed(self) -> bool:
        """Restore a missing hot database from its durable backup."""

        backup = self.db_backup_path
        if self.db_path.exists() or backup is None or not backup.is_file():
            return False
        temporary = self.db_path.with_name(
            f".{self.db_path.name}.{os.getpid()}.restore.tmp"
        )
        temporary.unlink(missing_ok=True)
        try:
            shutil.copy2(backup, temporary)
            self._validate_database_file(temporary)
            os.replace(temporary, self.db_path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return True

    @staticmethod
    def _validate_database_file(path: Path) -> None:
        uri = f"{path.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        try:
            row = conn.execute("PRAGMA quick_check").fetchone()
            if row is None or str(row[0]).casefold() != "ok":
                raise RuntimeError(
                    f"SQLite backup failed quick_check: {row!r}"
                )
        finally:
            conn.close()

    def backup_database(self) -> Path | None:
        """Snapshot the hot SQLite database to an atomic durable file."""

        destination = self.db_backup_path
        if destination is None:
            return None
        if destination == self.db_path:
            raise ValueError("database backup path must differ from database")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.backup.tmp"
        )
        with self._backup_lock:
            try:
                temporary.unlink(missing_ok=True)
                source = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                )
                target = sqlite3.connect(
                    temporary,
                    timeout=30.0,
                )
                try:
                    source.backup(target)
                finally:
                    target.close()
                    source.close()
                self._validate_database_file(temporary)
                os.replace(temporary, destination)
                self._last_backup_at = utc_now()
                self._last_backup_error = ""
            except BaseException as exc:
                temporary.unlink(missing_ok=True)
                self._last_backup_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                raise
        return destination

    def start_database_backup_loop(self) -> None:
        if self.db_backup_path is None:
            return
        thread = self._backup_thread
        if thread is not None and thread.is_alive():
            return
        self._backup_stop.clear()
        self._backup_thread = threading.Thread(
            target=self._database_backup_loop,
            name="autoresearch-v2-db-backup",
            daemon=True,
        )
        self._backup_thread.start()

    def _database_backup_loop(self) -> None:
        while not self._backup_stop.is_set():
            try:
                self.backup_database()
            except (
                OSError,
                RuntimeError,
                sqlite3.Error,
            ):
                # Backups are an availability layer. The local SQLite database
                # remains authoritative while the process is alive, and the
                # next interval retries without blocking controller ticks.
                if self._backup_stop.wait(self.backup_interval_sec):
                    break
                continue
            if self._backup_stop.wait(self.backup_interval_sec):
                break

    def stop_database_backup_loop(
        self,
        *,
        final_backup: bool = False,
    ) -> None:
        self._backup_stop.set()
        thread = self._backup_thread
        self._backup_thread = None
        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=min(30.0, self.backup_interval_sec + 1.0))
        if final_backup and self.db_backup_path is not None:
            self.backup_database()

    def database_backup_status(self) -> dict[str, Any]:
        thread = self._backup_thread
        return {
            "enabled": self.db_backup_path is not None,
            "database_path": str(self.db_path),
            "backup_path": (
                str(self.db_backup_path)
                if self.db_backup_path is not None
                else ""
            ),
            "interval_sec": self.backup_interval_sec,
            "running": bool(thread is not None and thread.is_alive()),
            "last_backup_at": self._last_backup_at,
            "last_error": self._last_backup_error,
        }

    def _recover_filesystem_commits(self) -> None:
        """Repair interrupted current-directory swaps before DB recovery."""

        if not self.ideas_root.is_dir():
            return
        # Avoid scanning every historical idea on CephFS. Only interrupted
        # swaps can leave these hidden paths, and a bounded find returns those
        # paths in one filesystem traversal.
        import subprocess

        try:
            completed = subprocess.run(
                [
                    "find",
                    str(self.ideas_root),
                    "-mindepth",
                    "2",
                    "-maxdepth",
                    "2",
                    "(",
                    "-name",
                    ".current.previous*",
                    "-o",
                    "-name",
                    ".current.*.tmp",
                    ")",
                    "-print0",
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            # Recovery is best-effort. Durable job recovery recreates missing
            # candidates, and later starts can retry hidden-path cleanup.
            return
        if completed.returncode != 0:
            return
        paths = [
            Path(value.decode("utf-8", errors="surrogateescape"))
            for value in completed.stdout.split(b"\0")
            if value
        ]
        parents = {path.parent for path in paths}
        for idea_dir in parents:
            current = idea_dir / "current"
            backups = sorted(
                path
                for path in paths
                if path.parent == idea_dir
                and path.name.startswith(".current.previous")
            )
            staged = sorted(
                path
                for path in paths
                if path.parent == idea_dir and path.name.endswith(".tmp")
            )
            if not current.exists():
                recoverable = next(
                    (path for path in reversed(backups) if path.is_dir()),
                    None,
                )
                if recoverable is not None:
                    os.replace(recoverable, current)
            for backup in backups:
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)
            for path in staged:
                shutil.rmtree(path, ignore_errors=True)

    def recover_filesystem_commits(self) -> None:
        """Run attempt-workspace recovery while holding the writer lock."""

        if self._writer_lock_stream is None:
            raise RuntimeError(
                "filesystem recovery requires the controller writer lock"
            )
        self._recover_filesystem_commits()
        self._recover_orphan_attempt_candidates()

    def recover_retry_candidate_workspaces(self) -> None:
        """Quarantine candidates left by refunded/retryable attempts.

        A durable attempt row can survive a controller interruption before its
        parent Job is updated. Startup recovery may then refund that attempt
        number and clear ``job.attempt_id``. The next dispatch intentionally
        reuses the number, so its old candidate must be preserved under an
        audit name rather than blocking ``snapshot_current`` forever.
        """

        if self._writer_lock_stream is None:
            raise RuntimeError(
                "retry workspace recovery requires the controller writer lock"
            )
        for job in self.list_jobs(
            statuses={JobStatus.READY, JobStatus.RETRY_WAIT}
        ):
            next_number = job.attempt + 1
            attempt_id = f"{job.job_id}-attempt-{next_number:02d}"
            attempt = self.get_attempt(attempt_id)
            candidate = (
                self.idea_dir(job.idea_id)
                / "attempts"
                / job.job_id
                / f"attempt-{next_number:02d}"
                / "candidate"
            )
            if not candidate.is_dir():
                continue
            if (
                attempt is not None
                and attempt.status is AttemptStatus.ACCEPTED
            ):
                # An accepted attempt is reconciled by the Controller before a
                # new attempt is scheduled; never move accepted evidence.
                continue
            quarantine = candidate.with_name(
                "candidate.interrupted-retry"
            )
            if quarantine.exists():
                quarantine = candidate.with_name(
                    "candidate.interrupted-retry."
                    + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                )
            try:
                os.replace(candidate, quarantine)
            except FileNotFoundError:
                continue
            self.event(
                "retry_attempt_candidate_quarantined",
                idea_id=job.idea_id,
                job_id=job.job_id,
                attempt_id=attempt_id,
                candidate=str(candidate),
                quarantine_path=str(quarantine),
                attempt_status=(
                    attempt.status.value if attempt is not None else ""
                ),
            )

    def _recover_orphan_attempt_candidates(self) -> None:
        """Quarantine candidate workspaces that have no durable attempt row.

        An executor creates ``candidate/`` before its attempt is saved. If the
        controller is killed in that small window, the next dispatch reuses
        the same scientific attempt number and would otherwise fail forever
        with ``FileExistsError``. The writer lock makes it safe to move only
        workspaces whose attempt IDs are absent from SQLite. Keeping the
        orphan under an audit name preserves partial output for diagnosis
        without letting it block recovery.
        """

        if not self.ideas_root.is_dir():
            return
        import subprocess

        try:
            completed = subprocess.run(
                [
                    "find",
                    str(self.ideas_root),
                    "-mindepth",
                    "5",
                    "-maxdepth",
                    "5",
                    "-type",
                    "d",
                    "-name",
                    "candidate",
                    "-print0",
                ],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        if completed.returncode != 0:
            return
        candidate_paths = [
            Path(value.decode("utf-8", errors="surrogateescape"))
            for value in completed.stdout.split(b"\0")
            if value
        ]
        if not candidate_paths:
            return
        with self.connect() as conn:
            durable_attempt_ids = {
                str(row[0])
                for row in conn.execute("SELECT attempt_id FROM attempts")
            }
        for candidate in candidate_paths:
            attempt_dir = candidate.parent
            job_dir = attempt_dir.parent
            match = re.fullmatch(r"attempt-(\d+)", attempt_dir.name)
            if match is None:
                continue
            attempt_id = (
                f"{job_dir.name}-attempt-{int(match.group(1)):02d}"
            )
            if attempt_id in durable_attempt_ids:
                continue
            orphan = attempt_dir / "candidate.interrupted-orphan"
            if orphan.exists():
                orphan = attempt_dir / (
                    "candidate.interrupted-orphan."
                    + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                )
            try:
                os.replace(candidate, orphan)
            except FileNotFoundError:
                continue
            self.event(
                "orphan_attempt_candidate_quarantined",
                attempt_id=attempt_id,
                job_id=job_dir.name,
                candidate=str(candidate),
                quarantine_path=str(orphan),
            )

    def acquire_writer_lock(self) -> None:
        if self._writer_lock_stream is not None:
            return
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        stream = self.writer_lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise RuntimeError(
                f"another AutoResearch v2 controller holds "
                f"{self.writer_lock_path}"
            ) from None
        stream.seek(0)
        stream.truncate()
        stream.write(
            json.dumps(
                {"pid": os.getpid(), "acquired_at": utc_now()},
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        self._writer_lock_stream = stream

    def release_writer_lock(self) -> None:
        stream = self._writer_lock_stream
        if stream is None:
            return
        import fcntl

        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
            self._writer_lock_stream = None

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def event(
        self,
        event_type: str,
        *,
        idea_id: str = "",
        job_id: str = "",
        attempt_id: str = "",
        **payload: Any,
    ) -> None:
        timestamp = utc_now()
        record = {
            "timestamp": timestamp,
            "event_type": event_type,
            "idea_id": idea_id or None,
            "job_id": job_id or None,
            "attempt_id": attempt_id or None,
            **payload,
        }
        with self._event_lock:
            with self.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO events(
                        timestamp,event_type,idea_id,job_id,attempt_id,payload_json
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        timestamp,
                        event_type,
                        idea_id or None,
                        job_id or None,
                        attempt_id or None,
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
            self._append_event_record(record)

    def save_transition(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Atomically persist one lifecycle transition in SQLite."""

        timestamp = utc_now()
        idea.updated_at = timestamp
        job.updated_at = timestamp
        attempt.updated_at = timestamp
        idea_data = idea.to_dict()
        job_data = job.to_dict()
        attempt_data = attempt.to_dict()
        record = {
            "timestamp": timestamp,
            "event_type": event_type,
            "idea_id": idea.idea_id,
            "job_id": job.job_id,
            "attempt_id": attempt.attempt_id,
            **dict(payload),
        }
        with self._event_lock:
            with self.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO ideas(
                        idea_id,status,priority,updated_at,data_json
                    ) VALUES(?,?,?,?,?)
                    ON CONFLICT(idea_id) DO UPDATE SET
                        status=excluded.status,
                        priority=excluded.priority,
                        updated_at=excluded.updated_at,
                        data_json=excluded.data_json
                    """,
                    (
                        idea.idea_id,
                        idea.status.value,
                        idea.priority,
                        timestamp,
                        json.dumps(
                            idea_data,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO jobs(
                        job_id,idea_id,kind,status,updated_at,data_json
                    ) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        status=excluded.status,
                        updated_at=excluded.updated_at,
                        data_json=excluded.data_json
                    """,
                    (
                        job.job_id,
                        job.idea_id,
                        job.kind.value,
                        job.status.value,
                        timestamp,
                        json.dumps(
                            job_data,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO attempts(
                        attempt_id,job_id,idea_id,number,status,
                        updated_at,data_json
                    ) VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(attempt_id) DO UPDATE SET
                        status=excluded.status,
                        updated_at=excluded.updated_at,
                        data_json=excluded.data_json
                    """,
                    (
                        attempt.attempt_id,
                        attempt.job_id,
                        attempt.idea_id,
                        attempt.number,
                        attempt.status.value,
                        timestamp,
                        json.dumps(
                            attempt_data,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO events(
                        timestamp,event_type,idea_id,job_id,attempt_id,
                        payload_json
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        timestamp,
                        event_type,
                        idea.idea_id,
                        job.job_id,
                        attempt.attempt_id,
                        json.dumps(
                            dict(payload),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
            self._atomic_json(
                self.idea_dir(idea.idea_id) / "idea.json",
                idea_data,
            )
            self._atomic_json(
                self.attempt_dir(attempt) / "attempt.json",
                attempt_data,
            )
            self._append_event_record(record)

    def _append_event_record(self, record: Mapping[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    def maintain(
        self,
        *,
        event_jsonl_max_bytes: int,
        llm_audit_max_bytes: int,
        keep_failed_attempts_per_job: int,
    ) -> dict[str, int]:
        rotated = 0
        pruned = 0
        if self._rotate_file(
            self.events_path,
            max_bytes=event_jsonl_max_bytes,
        ):
            rotated += 1
        audit_root = self.root / "llm-audit"
        if audit_root.is_dir():
            for path in audit_root.rglob("*.jsonl"):
                if self._rotate_file(
                    path,
                    max_bytes=llm_audit_max_bytes,
                ):
                    rotated += 1
        keep = max(1, int(keep_failed_attempts_per_job))
        by_job: dict[str, list[AttemptRecord]] = {}
        for attempt in self.list_attempts():
            if attempt.status in {
                AttemptStatus.REJECTED,
                AttemptStatus.FAILED,
                AttemptStatus.CANCELLED,
            }:
                by_job.setdefault(attempt.job_id, []).append(attempt)
        for attempts in by_job.values():
            attempts.sort(
                key=lambda item: (item.number, item.attempt_id),
                reverse=True,
            )
            for attempt in attempts[keep:]:
                candidate = self.attempt_dir(attempt) / "candidate"
                if candidate.is_dir():
                    shutil.rmtree(candidate, ignore_errors=True)
                    pruned += 1
        with self.connect() as conn:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        return {"rotated_logs": rotated, "pruned_candidates": pruned}

    @staticmethod
    def _rotate_file(path: Path, *, max_bytes: int) -> bool:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size <= max(1, int(max_bytes)):
            return False
        archive = path.with_suffix(path.suffix + ".1")
        archive.unlink(missing_ok=True)
        os.replace(path, archive)
        path.touch(mode=0o600)
        return True

    def save_idea(self, idea: IdeaRecord) -> None:
        idea.updated_at = utc_now()
        data = idea.to_dict()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ideas(idea_id,status,priority,updated_at,data_json)
                VALUES(?,?,?,?,?)
                ON CONFLICT(idea_id) DO UPDATE SET
                    status=excluded.status,
                    priority=excluded.priority,
                    updated_at=excluded.updated_at,
                    data_json=excluded.data_json
                """,
                (
                    idea.idea_id,
                    idea.status.value,
                    idea.priority,
                    idea.updated_at,
                    json.dumps(data, ensure_ascii=False, sort_keys=True),
                ),
            )
        idea_dir = self.idea_dir(idea.idea_id)
        idea_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_json(idea_dir / "idea.json", data)

    def get_idea(self, idea_id: str) -> IdeaRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM ideas WHERE idea_id=?", (idea_id,)
            ).fetchone()
        return IdeaRecord.from_dict(json.loads(row[0])) if row else None

    def list_ideas(
        self,
        *,
        statuses: set[IdeaStatus] | None = None,
    ) -> list[IdeaRecord]:
        params: list[Any] = []
        sql = "SELECT data_json FROM ideas"
        if statuses:
            marks = ",".join("?" for _ in statuses)
            sql += f" WHERE status IN ({marks})"
            params.extend(status.value for status in statuses)
        sql += " ORDER BY priority DESC, updated_at, idea_id"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [IdeaRecord.from_dict(json.loads(row[0])) for row in rows]

    def save_job(self, job: JobRecord) -> None:
        job.updated_at = utc_now()
        data = job.to_dict()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(job_id,idea_id,kind,status,updated_at,data_json)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    data_json=excluded.data_json
                """,
                (
                    job.job_id,
                    job.idea_id,
                    job.kind.value,
                    job.status.value,
                    job.updated_at,
                    json.dumps(data, ensure_ascii=False, sort_keys=True),
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return JobRecord.from_dict(json.loads(row[0])) if row else None

    def list_jobs(
        self,
        *,
        statuses: set[JobStatus] | None = None,
        idea_id: str = "",
    ) -> list[JobRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if statuses:
            marks = ",".join("?" for _ in statuses)
            clauses.append(f"status IN ({marks})")
            params.extend(status.value for status in statuses)
        if idea_id:
            clauses.append("idea_id=?")
            params.append(idea_id)
        sql = "SELECT data_json FROM jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at, job_id"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [JobRecord.from_dict(json.loads(row[0])) for row in rows]

    def create_attempt(self, job: JobRecord) -> AttemptRecord:
        number = job.attempt + 1
        attempt = AttemptRecord(
            attempt_id=f"{job.job_id}-attempt-{number:02d}",
            idea_id=job.idea_id,
            job_id=job.job_id,
            number=number,
        )
        return attempt

    def save_attempt(self, attempt: AttemptRecord) -> None:
        attempt.updated_at = utc_now()
        data = attempt.to_dict()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO attempts(
                    attempt_id,job_id,idea_id,number,status,updated_at,data_json
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(attempt_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    data_json=excluded.data_json
                """,
                (
                    attempt.attempt_id,
                    attempt.job_id,
                    attempt.idea_id,
                    attempt.number,
                    attempt.status.value,
                    attempt.updated_at,
                    json.dumps(data, ensure_ascii=False, sort_keys=True),
                ),
            )
        self._atomic_json(self.attempt_dir(attempt) / "attempt.json", data)

    def get_attempt(self, attempt_id: str) -> AttemptRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM attempts WHERE attempt_id=?",
                (attempt_id,),
            ).fetchone()
        return AttemptRecord.from_dict(json.loads(row[0])) if row else None

    def list_attempts(
        self,
        *,
        idea_id: str = "",
        job_id: str = "",
    ) -> list[AttemptRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if idea_id:
            clauses.append("idea_id=?")
            params.append(idea_id)
        if job_id:
            clauses.append("job_id=?")
            params.append(job_id)
        sql = "SELECT data_json FROM attempts"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at, attempt_id"
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [AttemptRecord.from_dict(json.loads(row[0])) for row in rows]

    def list_events(
        self,
        *,
        limit: int = 200,
        idea_id: str = "",
        after_seq: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["seq>?"]
        params: list[Any] = [max(0, int(after_seq))]
        if idea_id:
            clauses.append("idea_id=?")
            params.append(idea_id)
        params.append(max(1, min(int(limit), 5000)))
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT seq,timestamp,event_type,idea_id,job_id,attempt_id,
                       payload_json
                FROM events
                WHERE """
                + " AND ".join(clauses)
                + " ORDER BY seq DESC LIMIT ?",
                params,
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in reversed(rows):
            result.append(
                {
                    "seq": int(row["seq"]),
                    "timestamp": str(row["timestamp"]),
                    "event_type": str(row["event_type"]),
                    "idea_id": row["idea_id"],
                    "job_id": row["job_id"],
                    "attempt_id": row["attempt_id"],
                    **json.loads(row["payload_json"]),
                }
            )
        return result

    def list_events_between(
        self,
        *,
        event_type: str,
        started_at: str,
        ended_at: str,
    ) -> list[dict[str, Any]]:
        """Read one event stream directly from SQLite for a time window."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT seq,timestamp,event_type,idea_id,job_id,attempt_id,
                       payload_json
                FROM events
                WHERE event_type=? AND timestamp>=? AND timestamp<=?
                ORDER BY timestamp, seq
                """,
                (event_type, started_at, ended_at),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "timestamp": str(row["timestamp"]),
                "event_type": str(row["event_type"]),
                "idea_id": row["idea_id"],
                "job_id": row["job_id"],
                "attempt_id": row["attempt_id"],
                **json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def set_control(self, name: str, reason: str = "") -> Path:
        if name not in {"pause", "stop"}:
            raise ValueError("unsupported control")
        path = self.control_dir / name
        self._atomic_json(
            path,
            {"requested_at": utc_now(), "reason": str(reason)[:500]},
        )
        return path

    def clear_control(self, name: str) -> None:
        if name not in {"pause", "stop"}:
            raise ValueError("unsupported control")
        (self.control_dir / name).unlink(missing_ok=True)

    def control_requested(self, name: str) -> bool:
        return (self.control_dir / name).is_file()

    def writer_status(self) -> dict[str, Any]:
        """Return whether the durable controller-lock record names a live PID."""

        try:
            value = json.loads(
                self.writer_lock_path.read_text(encoding="utf-8")
            )
            pid = int(value.get("pid", 0) or 0)
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return {"state": "missing", "pid": 0}
        if pid <= 0:
            return {"state": "stale", "pid": 0}
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return {"state": "stale", "pid": pid}
        except PermissionError:
            pass
        return {"state": "live", "pid": pid}

    def idea_dir(self, idea_id: str) -> Path:
        return self.ideas_root / idea_id

    def attempt_dir(self, attempt: AttemptRecord) -> Path:
        return (
            self.idea_dir(attempt.idea_id)
            / "attempts"
            / attempt.job_id
            / f"attempt-{attempt.number:02d}"
        )

    def current_dir(self, idea_id: str) -> Path:
        return self.idea_dir(idea_id) / "current"

    def prepare_candidate(self, attempt: AttemptRecord) -> Path:
        path = self.attempt_dir(attempt) / "candidate"
        if path.exists():
            raise FileExistsError(f"attempt candidate already exists: {path}")
        path.mkdir(parents=True, exist_ok=False)
        return path

    def commit_candidate(self, attempt: AttemptRecord) -> Path:
        candidate = self.attempt_dir(attempt) / "candidate"
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        with self._commit_lock:
            current = self.current_dir(attempt.idea_id)
            staged = current.with_name(f".current.{attempt.attempt_id}.tmp")
            if staged.exists():
                shutil.rmtree(staged)
            shutil.copytree(candidate, staged)
            # Give every commit a private backup path.  Startup recovery
            # accepts both the legacy name and these attempt-scoped backups.
            backup = current.with_name(
                f".current.previous.{attempt.attempt_id}"
            )
            if backup.exists():
                shutil.rmtree(backup)
            if current.exists():
                os.replace(current, backup)
            try:
                os.replace(staged, current)
            except Exception:
                if not current.exists() and backup.is_dir():
                    os.replace(backup, current)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        attempt.status = AttemptStatus.ACCEPTED
        self.save_attempt(attempt)
        self.event(
            "attempt_committed",
            idea_id=attempt.idea_id,
            job_id=attempt.job_id,
            attempt_id=attempt.attempt_id,
            current=str(current),
        )
        return current

    def snapshot_current(self, attempt: AttemptRecord) -> Path:
        """Copy accepted current state into a new immutable candidate."""

        candidate = self.prepare_candidate(attempt)
        current = self.current_dir(attempt.idea_id)
        if current.is_dir():
            for child in current.iterdir():
                target = candidate / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
        return candidate

    def merge_current_into_candidate(self, attempt: AttemptRecord) -> Path:
        """Backfill files missing from an existing immutable candidate.

        A controller restart can refund a running attempt after its candidate
        directory has been removed. A detached worker from the interrupted
        process may still finish later and recreate only the files it writes.
        Before accepting that candidate, merge the latest committed snapshot
        into it without overwriting outputs produced by the attempt.
        """

        candidate = self.attempt_dir(attempt) / "candidate"
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        current = self.current_dir(attempt.idea_id)
        if not current.is_dir():
            return candidate
        for child in current.iterdir():
            target = candidate / child.name
            if target.exists() or target.is_symlink():
                continue
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        return candidate

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temp = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temp, path)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
