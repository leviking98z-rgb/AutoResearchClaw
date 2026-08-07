"""Cross-process GPU capacity accounting for one physical execution pool."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ACTIVE_TASK_STATES = {
    "reserved",
    "submitted",
    "running",
    "probe_degraded",
    "orphaned",
}
_TERMINAL_TASK_STATES = {
    "cancelled",
    "failed",
    "finished",
    "lost",
    "timed_out",
}


@dataclass(frozen=True, slots=True)
class GlobalLease:
    pool_id: str
    task_id: str
    owner_id: str
    idea_id: str
    job_id: str
    allocated_gpus: int
    state: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class LeaseReservation:
    admitted: bool
    allocated_gpus: int = 0
    reason: str = ""


def stable_pool_identity(
    *,
    nodes: Sequence[Mapping[str, Any]],
    expected_total_gpus: int,
) -> str:
    """Return a path-independent identity for one physical GPU topology."""

    normalized = sorted(
        (
            {
                "address": str(node.get("address", "")),
                "ray_ip": str(node.get("ray_ip", node.get("address", ""))),
                "gpu_ids": sorted(int(value) for value in node.get("gpu_ids", ())),
            }
            for node in nodes
        ),
        key=lambda item: (item["address"], item["ray_ip"], item["gpu_ids"]),
    )
    payload = json.dumps(
        {
            "nodes": normalized,
            "expected_total_gpus": int(expected_total_gpus),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"clusterbridge-{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def shared_registry_path(state_dir: str | Path, pool_id: str) -> Path:
    """Choose one shared registry path for all controllers of a pool."""

    resolved = Path(state_dir).expanduser().resolve()
    cluster_root = next(
        (path for path in (resolved, *resolved.parents) if path.name == ".clusters"),
        None,
    )
    root = (
        cluster_root / ".leases" / "autoresearch-v2"
        if cluster_root is not None
        else resolved.parent / ".autoresearch-v2-gpu-leases"
    )
    return root / f"{pool_id}.sqlite3"


class SharedGPULeaseRegistry:
    """SQLite-backed atomic reservations shared by multiple controllers."""

    def __init__(
        self,
        path: str | Path,
        *,
        pool_id: str,
        total_gpus: int,
        reserved_gpus: int = 0,
        max_share_per_idea: float = 0.5,
        owner_ttl_sec: float = 120.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.pool_id = str(pool_id)
        self.total_gpus = max(0, int(total_gpus))
        self.reserved_gpus = max(0, int(reserved_gpus))
        self.max_share_per_idea = float(max_share_per_idea)
        self.owner_ttl_sec = max(1.0, float(owner_ttl_sec))
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def usable_gpus(self) -> int:
        return max(0, self.total_gpus - self.reserved_gpus)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS owners (
                    pool_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    PRIMARY KEY(pool_id, owner_id)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    pool_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    idea_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    allocated_gpus INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(pool_id, task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_global_gpu_leases_owner
                    ON leases(pool_id, owner_id);
                CREATE INDEX IF NOT EXISTS idx_global_gpu_leases_idea
                    ON leases(pool_id, idea_id);
                CREATE TABLE IF NOT EXISTS pools (
                    pool_id TEXT PRIMARY KEY,
                    total_gpus INTEGER NOT NULL,
                    reserved_gpus INTEGER NOT NULL
                );
                """
            )
            row = connection.execute(
                """
                SELECT total_gpus, reserved_gpus
                FROM pools
                WHERE pool_id=?
                """,
                (self.pool_id,),
            ).fetchone()
            expected = (self.total_gpus, self.reserved_gpus)
            if row is None:
                connection.execute(
                    """
                    INSERT INTO pools(pool_id, total_gpus, reserved_gpus)
                    VALUES (?, ?, ?)
                    """,
                    (self.pool_id, *expected),
                )
            elif (
                int(row["total_gpus"]),
                int(row["reserved_gpus"]),
            ) != expected:
                raise ValueError(
                    "shared GPU lease registry capacity mismatch for "
                    f"{self.pool_id}: database has "
                    f"{int(row['total_gpus'])} total/"
                    f"{int(row['reserved_gpus'])} reserved, requested "
                    f"{self.total_gpus} total/{self.reserved_gpus} reserved"
                )

    def heartbeat(
        self,
        owner_id: str,
        *,
        prune_expired_orphans: bool = True,
    ) -> None:
        now = self.clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if prune_expired_orphans:
                self._prune_expired_orphans(connection, now)
            self._touch_owner(connection, owner_id, now)
            connection.execute(
                """
                UPDATE leases
                SET expires_at=?, updated_at=?
                WHERE pool_id=? AND owner_id=? AND state != 'orphaned'
                """,
                (
                    now + self.owner_ttl_sec,
                    now,
                    self.pool_id,
                    owner_id,
                ),
            )
            connection.commit()

    def reserve(
        self,
        *,
        owner_id: str,
        task_id: str,
        idea_id: str,
        job_id: str,
        min_gpus: int,
        preferred_gpus: int,
        max_gpus: int,
        prune_expired_orphans: bool = True,
    ) -> LeaseReservation:
        now = self.clock()
        minimum = max(0, int(min_gpus))
        preferred = max(minimum, int(preferred_gpus or minimum))
        maximum = max(minimum, int(max_gpus or preferred or minimum))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # Expired orphan reservations no longer protect a running owner
            # and must not strand capacity when no matching pool task exists.
            # Active orphaned tasks are extended by ``reap_stale`` before
            # callers reserve through GPUBroker.
            if prune_expired_orphans:
                self._prune_expired_orphans(connection, now)
            self._touch_owner(connection, owner_id, now)
            existing = connection.execute(
                """
                SELECT owner_id, allocated_gpus
                FROM leases
                WHERE pool_id=? AND task_id=?
                """,
                (self.pool_id, task_id),
            ).fetchone()
            if existing is not None:
                if str(existing["owner_id"]) == owner_id:
                    connection.commit()
                    return LeaseReservation(
                        True,
                        int(existing["allocated_gpus"]),
                        "existing_global_lease",
                    )
                connection.rollback()
                return LeaseReservation(
                    False,
                    0,
                    "task_reserved_by_other_owner",
                )

            rows = connection.execute(
                """
                SELECT idea_id, allocated_gpus
                FROM leases
                WHERE pool_id=? AND expires_at > ?
                """,
                (self.pool_id, now),
            ).fetchall()
            used = sum(int(row["allocated_gpus"]) for row in rows)
            used_by_idea = sum(
                int(row["allocated_gpus"])
                for row in rows
                if str(row["idea_id"]) == idea_id
            )
            available = max(0, self.usable_gpus - used)
            idea_cap = max(
                minimum,
                int(self.usable_gpus * self.max_share_per_idea),
            )
            cap = min(
                available,
                max(0, idea_cap - used_by_idea),
                maximum,
            )
            if cap < minimum:
                connection.rollback()
                return LeaseReservation(
                    False,
                    0,
                    "global_insufficient_capacity",
                )
            allocated = max(minimum, min(preferred, cap))
            connection.execute(
                """
                INSERT INTO leases(
                    pool_id, task_id, owner_id, idea_id, job_id,
                    allocated_gpus, state, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.pool_id,
                    task_id,
                    owner_id,
                    idea_id,
                    job_id,
                    allocated,
                    "reserved",
                    now + self.owner_ttl_sec,
                    now,
                ),
            )
            connection.commit()
            return LeaseReservation(True, allocated, "admitted")

    def adopt(
        self,
        *,
        owner_id: str,
        task_id: str,
        idea_id: str,
        job_id: str,
        allocated_gpus: int,
    ) -> None:
        """Transfer or recreate accounting for an already-running pool task."""

        now = self.clock()
        allocated = max(1, int(allocated_gpus))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._touch_owner(connection, owner_id, now)
            connection.execute(
                """
                INSERT INTO leases(
                    pool_id, task_id, owner_id, idea_id, job_id,
                    allocated_gpus, state, expires_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pool_id, task_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    idea_id=excluded.idea_id,
                    job_id=excluded.job_id,
                    allocated_gpus=excluded.allocated_gpus,
                    state=excluded.state,
                    expires_at=excluded.expires_at,
                    updated_at=excluded.updated_at
                """,
                (
                    self.pool_id,
                    task_id,
                    owner_id,
                    idea_id,
                    job_id,
                    allocated,
                    "running",
                    now + self.owner_ttl_sec,
                    now,
                ),
            )
            connection.commit()

    def mark_state(self, owner_id: str, task_id: str, state: str) -> None:
        now = self.clock()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE leases
                SET state=?, expires_at=?, updated_at=?
                WHERE pool_id=? AND task_id=? AND owner_id=?
                """,
                (
                    state,
                    now + self.owner_ttl_sec,
                    now,
                    self.pool_id,
                    task_id,
                    owner_id,
                ),
            )

    def release(self, owner_id: str, task_id: str, *, force: bool = False) -> None:
        with self._connect() as connection:
            if force:
                connection.execute(
                    "DELETE FROM leases WHERE pool_id=? AND task_id=?",
                    (self.pool_id, task_id),
                )
            else:
                connection.execute(
                    """
                    DELETE FROM leases
                    WHERE pool_id=? AND task_id=? AND owner_id=?
                    """,
                    (self.pool_id, task_id, owner_id),
                )

    def detach(self, owner_id: str, task_id: str) -> None:
        """Keep capacity reserved while an active task has no live owner."""

        now = self.clock()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE leases
                SET state='orphaned', expires_at=?, updated_at=?
                WHERE pool_id=? AND task_id=? AND owner_id=?
                """,
                (
                    now + self.owner_ttl_sec,
                    now,
                    self.pool_id,
                    task_id,
                    owner_id,
                ),
            )

    def close_owner(self, owner_id: str) -> None:
        now = self.clock()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO owners(pool_id, owner_id, heartbeat_at, status)
                VALUES (?, ?, ?, 'closed')
                ON CONFLICT(pool_id, owner_id) DO UPDATE SET
                    heartbeat_at=excluded.heartbeat_at,
                    status='closed'
                """,
                (self.pool_id, owner_id, now),
            )
            connection.execute(
                """
                UPDATE leases
                SET state='orphaned', expires_at=?, updated_at=?
                WHERE pool_id=? AND owner_id=?
                """,
                (
                    now + self.owner_ttl_sec,
                    now,
                    self.pool_id,
                    owner_id,
                ),
            )
            connection.commit()

    def list_leases(self) -> list[GlobalLease]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT pool_id, task_id, owner_id, idea_id, job_id,
                       allocated_gpus, state, expires_at
                FROM leases
                WHERE pool_id=?
                ORDER BY task_id
                """,
                (self.pool_id,),
            ).fetchall()
        return [
            GlobalLease(
                pool_id=str(row["pool_id"]),
                task_id=str(row["task_id"]),
                owner_id=str(row["owner_id"]),
                idea_id=str(row["idea_id"]),
                job_id=str(row["job_id"]),
                allocated_gpus=int(row["allocated_gpus"]),
                state=str(row["state"]),
                expires_at=float(row["expires_at"]),
            )
            for row in rows
        ]

    def reap_stale(
        self,
        probe_task: Callable[[str], Any],
        *,
        release_unverified_expired: bool = True,
    ) -> list[str]:
        """Reclaim stale-owner leases after terminal evidence.

        ``release_unverified_expired`` preserves the legacy registry API for
        explicit maintenance callers. Live Brokers disable it: a TTL proves
        that the Controller owner disappeared, not that the detached remote
        GPU process stopped, so deleting an unverified lease could permit a
        duplicate submission.
        """

        now = self.clock()
        cutoff = now - self.owner_ttl_sec
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT leases.task_id, leases.expires_at
                FROM leases
                LEFT JOIN owners
                  ON owners.pool_id=leases.pool_id
                 AND owners.owner_id=leases.owner_id
                WHERE leases.pool_id=?
                  AND (
                    (
                      leases.state='orphaned'
                      AND leases.expires_at <= ?
                    )
                    OR owners.owner_id IS NULL
                    OR owners.status != 'active'
                    OR owners.heartbeat_at < ?
                  )
                """,
                (self.pool_id, now, cutoff),
            ).fetchall()
        reclaimed: list[str] = []
        for row in rows:
            task_id = str(row["task_id"])
            try:
                state = _state(probe_task(task_id))
            except Exception:  # noqa: BLE001
                if (
                    release_unverified_expired
                    and float(row["expires_at"]) <= now
                ):
                    self.release("", task_id, force=True)
                    reclaimed.append(task_id)
                continue
            if state in _TERMINAL_TASK_STATES:
                self.release("", task_id, force=True)
                reclaimed.append(task_id)
            elif state in _ACTIVE_TASK_STATES:
                self._extend_orphan(task_id, state, now)
            elif (
                release_unverified_expired
                and float(row["expires_at"]) <= now
            ):
                self.release("", task_id, force=True)
                reclaimed.append(task_id)
        return reclaimed

    def _extend_orphan(self, task_id: str, state: str, now: float) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE leases
                SET state=?, expires_at=?, updated_at=?
                WHERE pool_id=? AND task_id=?
                """,
                (
                    state,
                    now + self.owner_ttl_sec,
                    now,
                    self.pool_id,
                    task_id,
                ),
            )

    def _touch_owner(
        self,
        connection: sqlite3.Connection,
        owner_id: str,
        now: float,
    ) -> None:
        connection.execute(
            """
            INSERT INTO owners(pool_id, owner_id, heartbeat_at, status)
            VALUES (?, ?, ?, 'active')
            ON CONFLICT(pool_id, owner_id) DO UPDATE SET
                heartbeat_at=excluded.heartbeat_at,
                status='active'
            """,
            (self.pool_id, owner_id, now),
        )

    def _prune_expired_orphans(
        self,
        connection: sqlite3.Connection,
        now: float,
    ) -> None:
        connection.execute(
            """
            DELETE FROM leases
            WHERE pool_id=?
              AND state='orphaned'
              AND expires_at <= ?
            """,
            (self.pool_id, now),
        )


def _state(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("state", "unknown"))
    return str(getattr(value, "state", "unknown"))
