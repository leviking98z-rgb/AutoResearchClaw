"""Atomic snapshots and append-only events for one Research Factory."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .budgets import BudgetLedger
from .io import atomic_write_json
from .models import (
    Idea,
    IdeaStatus,
    ResourceLease,
    WorkItem,
    WorkItemStatus,
    utc_now,
)


class FactoryStore:
    """Single-writer filesystem repository.

    Mutations are serialized by a re-entrant process-local lock and every
    durable state transition is accompanied by an append-only event.  The
    orchestrator is still required to be the sole process-level writer.
    """

    def __init__(self, root: str | Path, *, factory_id: str = "research-factory"):
        self.root = Path(root).expanduser().resolve()
        self.factory_id = factory_id
        self.state_path = self.root / "state.json"
        self.manifest_path = self.root / "factory.json"
        self.events_path = self.root / "events.jsonl"
        self.reservoir_path = self.root / "reservoir" / "candidates.json"
        self.scheduler_dir = self.root / "scheduler"
        self.leases_path = self.scheduler_dir / "leases.json"
        self.ideas_dir = self.root / "ideas"
        self.shared_cache_dir = self.root / "shared-cache"
        self.control_dir = self.root / "control"
        self._lock = threading.RLock()
        self._event_lock = threading.Lock()

    def initialize(self) -> None:
        for directory in (
            self.root,
            self.reservoir_path.parent,
            self.scheduler_dir,
            self.ideas_dir,
            self.shared_cache_dir,
            self.control_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            atomic_write_json(
                self.manifest_path,
                {
                    "schema_version": 1,
                    "factory_id": self.factory_id,
                    "created_at": utc_now(),
                    "automatic_submission_enabled": False,
                },
            )
        if not self.state_path.exists():
            self.save_state(
                {
                    "schema_version": 1,
                    "factory_id": self.factory_id,
                    "status": "initialized",
                    "paused": False,
                    "pid": None,
                    "tick": 0,
                }
            )
        if not self.reservoir_path.exists():
            atomic_write_json(self.reservoir_path, [])
        if not self.leases_path.exists():
            atomic_write_json(self.leases_path, [])

    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    def load_state(self) -> dict[str, Any]:
        value = self._read_json(self.state_path, {})
        if not isinstance(value, dict):
            raise TypeError(f"malformed Factory state: {self.state_path}")
        return value

    def save_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(state)
        value["updated_at"] = utc_now()
        atomic_write_json(self.state_path, value)
        return value

    def event(self, event_type: str, **payload: Any) -> dict[str, Any]:
        value = {
            "timestamp": utc_now(),
            "type": event_type,
            "factory_id": self.factory_id,
            **payload,
        }
        line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with self._event_lock, self.events_path.open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return value

    def idea_dir(self, idea_id: str) -> Path:
        return self.ideas_dir / idea_id

    def idea_event(
        self,
        idea_id: str,
        event_type: str,
        **payload: Any,
    ) -> dict[str, Any]:
        """Append an Idea-local timeline in addition to the global journal."""

        value = {
            "timestamp": utc_now(),
            "type": event_type,
            "factory_id": self.factory_id,
            "idea_id": idea_id,
            **payload,
        }
        path = self.idea_dir(idea_id) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        with self._event_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        return value

    def _idea_path(self, idea_id: str) -> Path:
        return self.idea_dir(idea_id) / "idea.json"

    def save_idea(self, idea: Idea, *, event_type: str = "idea_saved") -> Idea:
        with self._lock:
            idea.updated_at = utc_now()
            root = self.idea_dir(idea.idea_id)
            for directory in (
                root,
                root / "workspace",
                root / "runs",
                root / "evidence",
            ):
                directory.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self._idea_path(idea.idea_id), idea.to_dict())
            self.event(
                event_type,
                idea_id=idea.idea_id,
                status=idea.status.value,
            )
            self.idea_event(
                idea.idea_id,
                event_type,
                status=idea.status.value,
                budget_tier=idea.budget_tier.value,
                priority=idea.priority,
                current_item_id=idea.current_item_id,
                exit_reason=idea.exit_reason,
            )
        return idea

    def get_idea(self, idea_id: str) -> Idea | None:
        value = self._read_json(self._idea_path(idea_id), None)
        return Idea.from_dict(value) if isinstance(value, Mapping) else None

    def list_ideas(
        self,
        *,
        statuses: Iterable[IdeaStatus] | None = None,
    ) -> list[Idea]:
        allowed = set(statuses) if statuses is not None else None
        ideas: list[Idea] = []
        if not self.ideas_dir.exists():
            return ideas
        for path in sorted(self.ideas_dir.glob("*/idea.json")):
            value = self._read_json(path, None)
            if not isinstance(value, Mapping):
                continue
            idea = Idea.from_dict(value)
            if allowed is None or idea.status in allowed:
                ideas.append(idea)
        return ideas

    def load_reservoir(self) -> list[Idea]:
        value = self._read_json(self.reservoir_path, [])
        if not isinstance(value, list):
            raise TypeError("malformed candidate reservoir")
        return [
            Idea.from_dict(item)
            for item in value
            if isinstance(item, Mapping)
        ]

    def save_reservoir(self, ideas: Iterable[Idea]) -> None:
        atomic_write_json(
            self.reservoir_path,
            [idea.to_dict() for idea in ideas],
        )

    def add_candidates(self, ideas: Iterable[Idea]) -> list[Idea]:
        with self._lock:
            current = self.load_reservoir()
            known_ids = {idea.idea_id for idea in current}
            known_titles = {idea.normalized_title for idea in current}
            known_titles.update(
                idea.normalized_title for idea in self.list_ideas()
            )
            added: list[Idea] = []
            for idea in ideas:
                if idea.idea_id in known_ids or idea.normalized_title in known_titles:
                    continue
                if idea.status is not IdeaStatus.CANDIDATE:
                    raise ValueError("reservoir only accepts CANDIDATE Ideas")
                current.append(idea)
                known_ids.add(idea.idea_id)
                known_titles.add(idea.normalized_title)
                added.append(idea)
            current.sort(key=lambda item: (-item.priority, item.idea_id))
            self.save_reservoir(current)
            for idea in added:
                self.event(
                    "candidate_added",
                    idea_id=idea.idea_id,
                    priority=idea.priority,
                )
            return added

    def remove_candidate(self, idea_id: str) -> Idea | None:
        with self._lock:
            current = self.load_reservoir()
            selected = next(
                (idea for idea in current if idea.idea_id == idea_id), None
            )
            if selected is None:
                return None
            self.save_reservoir(
                idea for idea in current if idea.idea_id != idea_id
            )
            self.event("candidate_removed", idea_id=idea_id)
            return selected

    def _work_items_path(self, idea_id: str) -> Path:
        return self.idea_dir(idea_id) / "work_items.json"

    def list_work_items(self, idea_id: str | None = None) -> list[WorkItem]:
        idea_ids = (
            [idea_id]
            if idea_id
            else [
                path.parent.name
                for path in self.ideas_dir.glob("*/work_items.json")
            ]
        )
        result: list[WorkItem] = []
        for current_id in sorted(idea_ids):
            value = self._read_json(self._work_items_path(current_id), [])
            if not isinstance(value, list):
                continue
            result.extend(
                WorkItem.from_dict(item)
                for item in value
                if isinstance(item, Mapping)
            )
        return result

    def save_work_item(
        self,
        item: WorkItem,
        *,
        event_type: str = "work_item_saved",
    ) -> WorkItem:
        with self._lock:
            items = {
                current.item_id: current
                for current in self.list_work_items(item.idea_id)
            }
            item.updated_at = utc_now()
            items[item.item_id] = item
            atomic_write_json(
                self._work_items_path(item.idea_id),
                [
                    current.to_dict()
                    for current in sorted(
                        items.values(), key=lambda value: value.item_id
                    )
                ],
            )
            self.event(
                event_type,
                idea_id=item.idea_id,
                item_id=item.item_id,
                status=item.status.value,
                attempt=item.attempt,
            )
            self.idea_event(
                item.idea_id,
                event_type,
                item_id=item.item_id,
                kind=item.kind.value,
                profile=item.profile,
                status=item.status.value,
                attempt=item.attempt,
                resources=asdict(item.resources),
                result=item.result,
            )
        return item

    def get_work_item(self, item_id: str) -> WorkItem | None:
        return next(
            (
                item
                for item in self.list_work_items()
                if item.item_id == item_id
            ),
            None,
        )

    def ready_work_items(self) -> list[WorkItem]:
        items = self.list_work_items()
        status_by_id = {item.item_id: item.status for item in items}
        ready: list[WorkItem] = []
        for item in items:
            if item.status not in {
                WorkItemStatus.PENDING,
                WorkItemStatus.READY,
                WorkItemStatus.RETRY_WAIT,
            }:
                continue
            if all(
                status_by_id.get(dep) is WorkItemStatus.SUCCEEDED
                for dep in item.dependencies
            ):
                if item.status is not WorkItemStatus.READY:
                    item.status = WorkItemStatus.READY
                    self.save_work_item(
                        item, event_type="work_item_became_ready"
                    )
                ready.append(item)
        return ready

    def budget_path(self, idea_id: str) -> Path:
        return self.idea_dir(idea_id) / "budget.json"

    def load_budget(self, idea_id: str) -> BudgetLedger:
        value = self._read_json(self.budget_path(idea_id), None)
        if isinstance(value, Mapping):
            return BudgetLedger.from_dict(value)
        return BudgetLedger(idea_id=idea_id)

    def save_budget(self, ledger: BudgetLedger) -> None:
        atomic_write_json(self.budget_path(ledger.idea_id), ledger.to_dict())

    def list_leases(self) -> list[ResourceLease]:
        value = self._read_json(self.leases_path, [])
        if not isinstance(value, list):
            raise TypeError("malformed lease snapshot")
        return [
            ResourceLease.from_dict(item)
            for item in value
            if isinstance(item, Mapping)
        ]

    def save_leases(self, leases: Iterable[ResourceLease]) -> None:
        atomic_write_json(
            self.leases_path,
            [lease.to_dict() for lease in leases],
        )

    def control_requested(self, name: str) -> bool:
        if name not in {"pause", "stop"}:
            raise ValueError(f"unknown control: {name}")
        return (self.control_dir / name).exists()

    def set_control(self, name: str, reason: str = "") -> Path:
        if name not in {"pause", "stop"}:
            raise ValueError(f"unknown control: {name}")
        path = self.control_dir / name
        atomic_write_json(
            path,
            {"requested_at": utc_now(), "reason": reason, "pid": os.getpid()},
        )
        return path

    def clear_control(self, name: str) -> None:
        (self.control_dir / name).unlink(missing_ok=True)

    def snapshot(self) -> dict[str, Any]:
        ideas = self.list_ideas()
        reservoir = self.load_reservoir()
        items = self.list_work_items()
        return {
            **self.load_state(),
            "reservoir_size": len(reservoir),
            "ideas_total": len(ideas),
            "ideas_by_status": {
                status.value: sum(idea.status is status for idea in ideas)
                for status in IdeaStatus
            },
            "work_items_by_status": {
                status.value: sum(item.status is status for item in items)
                for status in WorkItemStatus
            },
            "leases": [lease.to_dict() for lease in self.list_leases()],
        }
