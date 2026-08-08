"""Minimal SQLite state store and immutable artifact helpers."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import IdeaRecord, IdeaStatus, RunRecord, RunStatus, utc_now


class ResearchQueueStore:
    """One-process prototype store.

    SQLite is the source of truth for queue state. Revision and run directories
    contain the human-readable experiment artifacts.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.db_path = self.root / "research_queue.db"
        self.ideas_root = self.root / "ideas"
        self.events_path = self.root / "events.jsonl"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.ideas_root.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
                PRAGMA synchronous=NORMAL;
                CREATE TABLE IF NOT EXISTS ideas (
                    idea_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rq_ideas_status
                    ON ideas(status, priority DESC, updated_at);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    idea_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    budget TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rq_runs_idea
                    ON runs(idea_id, revision, budget, updated_at);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    idea_id TEXT,
                    run_id TEXT,
                    payload_json TEXT NOT NULL
                );
                """
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert_idea(self, idea: IdeaRecord) -> None:
        idea.updated_at = utc_now()
        payload = json.dumps(
            idea.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ideas(
                    idea_id, status, priority, updated_at, data_json
                ) VALUES (?, ?, ?, ?, ?)
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
                    payload,
                ),
            )
        idea_dir = self.idea_dir(idea.idea_id)
        idea_dir.mkdir(parents=True, exist_ok=True)
        self.write_json_atomic(idea_dir / "idea.json", idea.to_dict())

    def get_idea(self, idea_id: str) -> IdeaRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM ideas WHERE idea_id=?",
                (idea_id,),
            ).fetchone()
        if row is None:
            return None
        return IdeaRecord.from_mapping(json.loads(row["data_json"]))

    def list_ideas(
        self,
        *,
        statuses: set[IdeaStatus] | None = None,
    ) -> list[IdeaRecord]:
        params: list[Any] = []
        query = "SELECT data_json FROM ideas"
        if statuses:
            values = sorted(status.value for status in statuses)
            query += " WHERE status IN (" + ",".join("?" for _ in values) + ")"
            params.extend(values)
        query += " ORDER BY priority DESC, updated_at, idea_id"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [IdeaRecord.from_mapping(json.loads(row["data_json"])) for row in rows]

    def count_ideas(self, status: IdeaStatus | None = None) -> int:
        with self.connect() as conn:
            if status is None:
                row = conn.execute("SELECT COUNT(*) FROM ideas").fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM ideas WHERE status=?",
                    (status.value,),
                ).fetchone()
        return int(row[0] if row is not None else 0)

    def upsert_run(self, run: RunRecord) -> None:
        run.updated_at = utc_now()
        payload = json.dumps(
            run.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, idea_id, status, budget, revision,
                    updated_at, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    data_json=excluded.data_json
                """,
                (
                    run.run_id,
                    run.idea_id,
                    run.status.value,
                    run.budget.value,
                    run.revision,
                    run.updated_at,
                    payload,
                ),
            )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT data_json FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return RunRecord.from_mapping(json.loads(row["data_json"]))

    def list_runs(
        self,
        *,
        idea_id: str | None = None,
        statuses: set[RunStatus] | None = None,
    ) -> list[RunRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if idea_id is not None:
            clauses.append("idea_id=?")
            params.append(idea_id)
        if statuses:
            values = sorted(status.value for status in statuses)
            clauses.append("status IN (" + ",".join("?" for _ in values) + ")")
            params.extend(values)
        query = "SELECT data_json FROM runs"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at, run_id"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [RunRecord.from_mapping(json.loads(row["data_json"])) for row in rows]

    def event(
        self,
        event_type: str,
        *,
        idea_id: str = "",
        run_id: str = "",
        **payload: Any,
    ) -> None:
        timestamp = utc_now()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO events(
                    timestamp, event_type, idea_id, run_id, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    event_type,
                    idea_id or None,
                    run_id or None,
                    encoded,
                ),
            )
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "event": event_type,
                        "idea_id": idea_id,
                        "run_id": run_id,
                        **payload,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, timestamp, event_type, idea_id, run_id,
                       payload_json
                FROM events ORDER BY seq DESC LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "timestamp": row["timestamp"],
                "event": row["event_type"],
                "idea_id": row["idea_id"] or "",
                "run_id": row["run_id"] or "",
                **json.loads(row["payload_json"]),
            }
            for row in reversed(rows)
        ]

    def idea_dir(self, idea_id: str) -> Path:
        return self.ideas_root / idea_id

    def revision_dir(self, idea_id: str, revision: int) -> Path:
        return self.idea_dir(idea_id) / "revisions" / f"revision-{revision:03d}"

    def run_dir(self, idea_id: str, run_id: str) -> Path:
        return self.idea_dir(idea_id) / "runs" / run_id

    def write_revision(
        self,
        idea_id: str,
        revision: int,
        value: Mapping[str, Any],
    ) -> Path:
        directory = self.revision_dir(idea_id, revision)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "revision.json"
        self.write_json_atomic(path, value)
        return path

    def read_revision(
        self,
        idea_id: str,
        revision: int,
    ) -> dict[str, Any]:
        path = self.revision_dir(idea_id, revision) / "revision.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError(f"revision is not a JSON object: {path}")
        return dict(value)

    @staticmethod
    def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    dict(value),
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def snapshot(self) -> dict[str, Any]:
        idea_counts = {status.value: self.count_ideas(status) for status in IdeaStatus}
        runs = self.list_runs()
        run_counts = {
            status.value: sum(1 for run in runs if run.status is status)
            for status in RunStatus
        }
        return {
            "root": str(self.root),
            "database": str(self.db_path),
            "ideas": idea_counts,
            "runs": run_counts,
            "latest_events": self.list_events(limit=20),
        }
