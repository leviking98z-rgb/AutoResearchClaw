"""Durable filesystem primitives for RSI campaigns.

Campaign state is small and read frequently by both the supervisor and the
control CLIs.  State writes therefore use ``fsync`` + ``os.replace`` while
events are kept in a separate append-only JSONL journal.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON from *path*, returning *default* when the file is absent."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically replace *path* with a durable JSON representation of *value*."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace *path* with durable UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync so a rename survives a machine restart."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def cleanup_atomic_temp_files(root: Path) -> list[Path]:
    """Remove stale hidden ``*.tmp`` files left by interrupted atomic writes."""

    removed: list[Path] = []
    if not root.exists():
        return removed
    # Campaign artifacts can contain large generated trees on shared CephFS.
    # Atomic state/config writes only leave temp files in the campaign's own
    # metadata directories, so avoid recursively walking every run artifact at
    # supervisor startup.
    directories = (
        root,
        root / "control",
        root / "diagnostics",
        root / "shared",
        root / "shared" / "prompts",
        root / "shared" / "topic_selections",
    )
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob(".*.tmp"):
            if not path.is_file():
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            removed.append(path)
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        for run_dir in runs_dir.glob("cycle-*"):
            if not run_dir.is_dir():
                continue
            for path in run_dir.glob(".*.tmp"):
                if not path.is_file():
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed.append(path)
    return removed


class EventLog:
    """Append-only JSONL event journal."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(
        self,
        event_type: str,
        *,
        campaign_id: str,
        cycle: int | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "timestamp": utc_now(),
            "type": event_type,
            "campaign_id": campaign_id,
        }
        if cycle is not None:
            event["cycle"] = cycle
        event.update(payload)
        line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return event

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events


class CampaignStore:
    """Filesystem layout and persistence adapter for one campaign."""

    def __init__(self, campaign_dir: Path) -> None:
        self.root = campaign_dir.expanduser().resolve()
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "events.jsonl"
        self.heartbeat_path = self.root / "heartbeat.json"
        self.supervisor_heartbeat_path = self.root / "supervisor_heartbeat.json"
        self.manifest_path = self.root / "campaign.json"
        self.policy_path = self.root / "run_policy.json"
        self.control_dir = self.root / "control"
        self.runs_dir = self.root / "runs"
        self.shared_dir = self.root / "shared"
        self.shared_skills_dir = self.shared_dir / "skills"
        self.shared_prompts_dir = self.shared_dir / "prompts"
        self.shared_prompt_path = self.shared_prompts_dir / "campaign_guidance.md"
        self.shared_brief_path = self.shared_dir / "brief.md"
        # The campaign brief is the immutable scientific/safety contract.
        # Topic refinements live separately so diagnosis cannot silently
        # weaken that contract or the no-publication boundary.
        self.shared_topic_patch_path = self.shared_dir / "topic_patch.json"
        # Failed-cycle engineering guidance is transient and scoped to a
        # concrete failure signature.  It must never be appended permanently
        # to campaign guidance because a stale workaround can corrupt later
        # topics or silently weaken scientific gates.
        self.shared_repair_patch_path = self.shared_dir / "repair_patch.json"
        self.shared_topic_candidates_path = (
            self.shared_dir / "topic_candidates.json"
        )
        self.shared_selected_topic_path = self.shared_dir / "selected_topic.json"
        self.shared_topic_selection_path = self.shared_dir / "topic_selection.md"
        self.topic_selections_dir = self.shared_dir / "topic_selections"
        self.diagnostics_dir = self.root / "diagnostics"
        self.log = EventLog(self.events_path)

    def initialize(self) -> None:
        for directory in (
            self.root,
            self.control_dir,
            self.runs_dir,
            self.shared_skills_dir,
            self.shared_prompts_dir,
            self.topic_selections_dir,
            self.diagnostics_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> dict[str, Any]:
        state = read_json(self.state_path)
        if not isinstance(state, dict):
            raise FileNotFoundError(f"campaign state not found: {self.state_path}")
        return state

    def save_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        updated = dict(state)
        updated["updated_at"] = utc_now()
        atomic_write_json(self.state_path, updated)
        return updated

    def write_heartbeat(self, payload: Mapping[str, Any]) -> None:
        heartbeat = dict(payload)
        heartbeat.setdefault("timestamp", utc_now())
        atomic_write_json(self.heartbeat_path, heartbeat)
        # Compatibility name consumed by the independent RSI monitor.
        atomic_write_json(self.supervisor_heartbeat_path, heartbeat)

    def control_path(self, name: str) -> Path:
        if name not in {"pause", "stop"}:
            raise ValueError(f"unknown campaign control: {name}")
        return self.control_dir / name

    def set_control(self, name: str, reason: str = "") -> Path:
        path = self.control_path(name)
        payload = {
            "requested_at": utc_now(),
            "reason": reason,
            "pid": os.getpid(),
        }
        atomic_write_json(path, payload)
        if name == "pause":
            # Compatibility with the independently-developed monitor/pause CLI.
            atomic_write_json(
                self.root / "pause.request.json",
                {
                    "action": "pause",
                    "requested_at": payload["requested_at"],
                    "requested_by": f"pid:{payload['pid']}",
                    "reason": reason,
                    "semantics": "cooperative_pause_not_stop",
                },
            )
        return path

    def clear_control(self, name: str) -> None:
        self.control_path(name).unlink(missing_ok=True)
        if name == "pause":
            (self.root / "pause.request.json").unlink(missing_ok=True)

    def control_requested(self, name: str) -> bool:
        if self.control_path(name).exists():
            return True
        return name == "pause" and (self.root / "pause.request.json").exists()
