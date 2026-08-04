"""Campaign-aware web dashboard for persistent RSI research runs.

The generic ResearchClaw dashboard tracks pipelines started inside the web
process.  Production RSI campaigns are instead supervised by systemd and keep
their durable state on shared storage.  This module provides a deliberately
small FastAPI application that reads that durable state without owning or
restarting the research pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from researchclaw.pipeline.stages import PHASE_MAP, Stage

from .storage import CampaignStore, utc_now
from .supervisor import _process_start_ticks

DEFAULT_CAMPAIGN_DIR = Path(
    os.environ.get(
        "RSI_CAMPAIGN_DIR",
        "/root/shared/.clusters/.workdir/autoresearch-rsi/"
        "rsi-autonomous-llm-self-improvement",
    )
).expanduser()
DEFAULT_POOL_ROOT = Path(
    os.environ.get(
        "RSI_POOL_ROOT",
        "/root/shared/.clusters/.tmp/autoresearch-rsi/pools",
    )
).expanduser()

_STAGE_LINE = re.compile(
    r"\[(?P<run_id>[^\]]+)\]\s+Stage\s+"
    r"(?P<number>\d{1,2})/23\s+(?P<name>[A-Z0-9_]+)\s+"
    r"—\s+(?P<status>running|done|failed|paused|retrying)",
    re.IGNORECASE,
)
_DECISION_ROLLBACK = re.compile(
    r"Decision:\s+(?P<decision>[A-Z_]+)\s+→\s+rollback to\s+"
    r"(?P<target>[A-Z0-9_]+)",
    re.IGNORECASE,
)
_DOWNLOAD_PROGRESS = re.compile(
    r"(?P<downloaded>\d+(?:\.\d+)?)(?P<download_unit>[kMGT]?)/"
    r"(?P<total>\d+(?:\.\d+)?)(?P<total_unit>[kMGT]?)"
)
_TASK_ID = re.compile(r"^rc-pool-[a-z0-9]+$")
_ARTIFACT_EXTENSIONS = {
    ".bib",
    ".csv",
    ".html",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".png",
    ".svg",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
_ARTIFACT_NAMES = {
    "analysis.md",
    "artifact_manifest.json",
    "decision.md",
    "experiment_summary.json",
    "experiment_summary_best.json",
    "paper.pdf",
    "paper_final.md",
    "paper_final_verified.md",
    "paper.md",
    "references.bib",
    "references_verified.bib",
    "reproducibility.md",
    "results_table.tex",
    "selected_topic.json",
    "topic_candidates.json",
    "topic_selection.md",
    "verification_report.json",
}
_TEXT_LOG_LIMIT_BYTES = 2_000_000
_SCAN_TIMEOUT_SEC = 2.5
_IO_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="rsi-dashboard-io",
)


class ControlRequest(BaseModel):
    """A cooperative operator action from the local dashboard."""

    reason: str = Field(default="", max_length=500)


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError):
        return default


def _read_text(path: Path, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return default


def _run_with_timeout(
    function: Any,
    *args: Any,
    timeout: float = _SCAN_TIMEOUT_SEC,
    default: Any = None,
) -> Any:
    """Run potentially slow CephFS metadata work with a bounded wait."""

    future = _IO_POOL.submit(function, *args)
    try:
        return future.result(timeout=max(0.05, timeout))
    except (concurrent.futures.TimeoutError, OSError):
        future.cancel()
        return default


def _tail_lines(path: Path, limit: int) -> list[str]:
    """Read the last *limit* text lines without loading an unbounded log."""

    limit = max(1, min(int(limit), 2000))
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > _TEXT_LOG_LIMIT_BYTES:
                handle.seek(-_TEXT_LOG_LIMIT_BYTES, os.SEEK_END)
                handle.readline()
            payload = handle.read()
    except (FileNotFoundError, OSError):
        return []
    text = payload.decode("utf-8", errors="replace")
    return list(deque(text.splitlines(), maxlen=limit))


def _iso_age_seconds(value: object, *, now: datetime | None = None) -> float | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    current = now or datetime.now(UTC)
    return max(0.0, (current - parsed).total_seconds())


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except (OverflowError, OSError, TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_process_identity_alive(pid: object, start_ticks: object) -> bool:
    try:
        parsed_pid = int(pid)
        parsed_ticks = int(start_ticks)
    except (TypeError, ValueError):
        return False
    return parsed_pid > 0 and _process_start_ticks(parsed_pid) == parsed_ticks


def _is_pid_alive(pid: object) -> bool:
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    return parsed > 0 and Path(f"/proc/{parsed}").exists()


def _safe_child(root: Path, candidate: Path) -> Path:
    """Resolve *candidate* and ensure it is contained by *root*."""

    resolved_root = root.expanduser().resolve()
    resolved = candidate.expanduser().resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"path escapes allowed root: {candidate}")
    return resolved


def _localize_campaign_path(campaign_dir: Path, raw: object) -> Path | None:
    """Map a persisted path alias back into the selected campaign directory."""

    text = str(raw or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        try:
            campaign_text = str(campaign_dir).rstrip("/")
            candidate_text = str(candidate)
            if candidate_text == campaign_text or candidate_text.startswith(
                campaign_text + "/"
            ):
                return candidate
        except (OSError, ValueError):
            pass
    marker = f"/{campaign_dir.name}/"
    if marker in text:
        suffix = text.split(marker, 1)[1]
        try:
            return _safe_child(campaign_dir, campaign_dir / suffix)
        except ValueError:
            return None
    if candidate.name == campaign_dir.name:
        return campaign_dir.resolve()
    return None


def _latest_cycle_dir(campaign_dir: Path, state: Mapping[str, Any]) -> Path | None:
    for key in ("active_run_dir", "last_run_dir", "best_run_dir"):
        localized = _localize_campaign_path(campaign_dir, state.get(key))
        if localized is not None:
            return localized
    runs_dir = campaign_dir / "runs"
    try:
        candidates = [path for path in runs_dir.glob("cycle-*") if path.is_dir()]
    except OSError:
        return None
    return max(candidates, key=lambda path: path.name, default=None)


def _parse_stage_progress(
    run_dir: Path | None,
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    current_number: int | None = None
    current_name = ""
    current_status = ""
    current_run_id = str(checkpoint.get("run_id") or "")
    rollback: dict[str, str] | None = None
    last_line = ""
    if run_dir is not None:
        lines = _tail_lines(run_dir / "pipeline.log", 600)
        for line in lines:
            rollback_match = _DECISION_ROLLBACK.search(line)
            if rollback_match:
                rollback = {
                    "decision": rollback_match.group("decision").upper(),
                    "target": rollback_match.group("target").upper(),
                }
            match = _STAGE_LINE.search(line)
            if not match:
                continue
            current_number = int(match.group("number"))
            current_name = match.group("name").upper()
            current_status = match.group("status").lower()
            current_run_id = match.group("run_id")
            last_line = line
    if current_number is None:
        try:
            current_number = int(checkpoint.get("last_completed_stage") or 0) or None
        except (TypeError, ValueError):
            current_number = None
        current_name = str(checkpoint.get("last_completed_name") or "")
        current_status = "checkpoint" if current_number else ""
    return {
        "number": current_number,
        "name": current_name,
        "status": current_status,
        "run_id": current_run_id,
        "rollback": rollback,
        "source_line": last_line,
    }


def _stage_statuses(
    run_dir: Path | None,
    *,
    current_stage: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        checkpoint_number = int(checkpoint.get("last_completed_stage") or 0)
    except (TypeError, ValueError):
        checkpoint_number = 0
    current_number = current_stage.get("number")
    current_status = str(current_stage.get("status") or "")
    phase_by_number: dict[int, str] = {}
    for phase, stages in PHASE_MAP.items():
        for stage in stages:
            phase_by_number[int(stage)] = phase

    result: list[dict[str, Any]] = []
    rollback_active = (
        isinstance(current_number, int)
        and current_status == "running"
        and checkpoint_number > current_number
    )
    for stage in Stage:
        number = int(stage)
        if number == current_number and current_status in {
            "running",
            "failed",
            "paused",
            "retrying",
        }:
            status = current_status
        elif rollback_active and current_number < number <= checkpoint_number:
            status = "revisit"
        elif number <= checkpoint_number:
            status = "done"
        else:
            status = "pending"
        result.append(
            {
                "number": number,
                "name": stage.name,
                "phase": phase_by_number.get(number, ""),
                "status": status,
                "checkpoint": number == checkpoint_number,
            }
        )
    return result


def _task_metadata_files_unbounded(run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    try:
        stage_entries = [
            entry
            for entry in os.scandir(run_dir)
            if entry.name.startswith("stage-")
            and entry.is_dir(follow_symlinks=False)
        ]
    except OSError:
        return []
    for stage_entry in stage_entries:
        stack = [(Path(stage_entry.path), 0)]
        while stack:
            directory, depth = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                if (
                    entry.name == ".clusterbridge_pool_task.json"
                    and entry.is_file(follow_symlinks=False)
                ):
                    paths.append(Path(entry.path))
                elif depth < 3 and entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), depth + 1))
    return paths


def _task_metadata_files(run_dir: Path | None) -> list[Path]:
    if run_dir is None:
        return []
    # Generated experiment workspaces can contain thousands of files on
    # CephFS. Task metadata is shallow, but even metadata calls can occasionally
    # stall. Bound the recursive scan so the API remains responsive.
    paths = _run_with_timeout(
        _task_metadata_files_unbounded,
        run_dir,
        timeout=_SCAN_TIMEOUT_SEC,
        default=[],
    )
    if not paths:
        cached = run_dir / ".rsi_dashboard_task.json"
        value = _read_json(cached, {})
        if isinstance(value, dict):
            raw_path = str(value.get("metadata_path") or "")
            if raw_path:
                candidate = Path(raw_path)
                if candidate.is_file():
                    paths = [candidate]
    if not paths:
        return []
    unique = {str(path): path for path in paths}
    sorted_paths = sorted(
        unique.values(),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    try:
        from .storage import atomic_write_json

        atomic_write_json(
            run_dir / ".rsi_dashboard_task.json",
            {
                "metadata_path": str(sorted_paths[0]),
                "updated_at": utc_now(),
            },
        )
    except (IndexError, OSError):
        pass
    return sorted_paths


def _pool_log_root_from_metadata(
    task_metadata: Mapping[str, Any],
    *,
    fallback: Path,
) -> Path:
    config_path = Path(str(task_metadata.get("pool_config") or "")).expanduser()
    config_text = _read_text(config_path)
    match = re.search(
        r"^\s*log_root:\s*[\"']?([^\"'\n]+)",
        config_text,
        re.MULTILINE,
    )
    if match:
        return Path(match.group(1).strip()).expanduser()
    return fallback


def _discover_task_from_pool(
    pool_log_root: Path,
) -> tuple[dict[str, Any], Path] | None:
    """Find the newest task via the pool journal, avoiding run-tree scans."""

    try:
        pool_dirs = [
            Path(entry.path)
            for entry in os.scandir(pool_log_root)
            if entry.is_dir(follow_symlinks=False)
        ]
    except OSError:
        return None
    latest: tuple[datetime, dict[str, Any], Path] | None = None
    state_map = {
        "task_started": "running",
        "task_finished": "finished",
        "task_timed_out": "timed_out",
        "task_failed": "failed",
        "task_cancelled": "cancelled",
    }
    for pool_dir in pool_dirs:
        task_states: dict[str, tuple[datetime, dict[str, Any]]] = {}
        for event in reversed(_event_tail(pool_dir / "events.jsonl", 300)):
            task_id = str(event.get("task_id") or "")
            event_name = str(event.get("event") or "")
            if not _TASK_ID.fullmatch(task_id) or not event_name.startswith(
                "task_"
            ):
                continue
            observed = _parse_timestamp(event.get("time")) or datetime.min.replace(
                tzinfo=UTC
            )
            metadata = {
                "task_id": task_id,
                "state": state_map.get(
                    event_name, event_name.removeprefix("task_")
                ),
                "returncode": event.get("returncode"),
                "event_time": event.get("time"),
                "pool_id": pool_dir.name,
            }
            current = task_states.get(task_id)
            if current is None or observed > current[0]:
                task_states[task_id] = (observed, metadata)
        for observed, metadata in task_states.values():
            if latest is None or observed > latest[0]:
                latest = (observed, metadata, pool_dir)
    if latest is None:
        return None
    return latest[1], latest[2]


def _task_from_hint(
    task_id: object,
    *,
    pool_log_root: Path,
    state: str = "running",
) -> tuple[dict[str, Any], Path] | None:
    parsed = str(task_id or "")
    if not _TASK_ID.fullmatch(parsed):
        return None
    try:
        pool_dirs = [
            Path(entry.path)
            for entry in os.scandir(pool_log_root)
            if entry.is_dir(follow_symlinks=False)
        ]
    except OSError:
        return None
    for pool_dir in pool_dirs:
        task_dir = pool_dir / "tasks" / parsed
        try:
            os.stat(task_dir)
        except OSError:
            continue
        else:
            return (
                {
                    "task_id": parsed,
                    "state": state,
                    "pool_id": pool_dir.name,
                },
                pool_dir,
            )
    return None


def _human_bytes(number: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = max(0.0, number)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _progress_value(number: str, unit: str) -> float:
    factors = {"": 1.0, "k": 1_000.0, "M": 1_000_000.0, "G": 1e9, "T": 1e12}
    return float(number) * factors.get(unit, 1.0)


def _task_progress(stdout_lines: list[str], stderr_lines: list[str]) -> dict[str, Any]:
    combined = stdout_lines + stderr_lines
    condition = ""
    seed = ""
    activity = ""
    for line in combined:
        stripped = line.strip()
        condition_match = re.search(r"Running condition:\s*(.+)", stripped)
        if condition_match:
            condition = condition_match.group(1).strip(" =")
        seed_match = re.search(r"---\s+Seed\s+(.+?)\s+---", stripped)
        if seed_match:
            seed = seed_match.group(1).strip()
        if stripped and not stripped.startswith(("/", "import ")):
            activity = stripped[-500:]

    progress_match: re.Match[str] | None = None
    normalized_stderr = "\n".join(stderr_lines).replace("\r", "\n")
    for match in _DOWNLOAD_PROGRESS.finditer(normalized_stderr):
        progress_match = match
    download: dict[str, Any] | None = None
    if progress_match:
        downloaded = _progress_value(
            progress_match.group("downloaded"),
            progress_match.group("download_unit"),
        )
        total = _progress_value(
            progress_match.group("total"),
            progress_match.group("total_unit"),
        )
        percent = min(100.0, downloaded / total * 100) if total > 0 else 0.0
        download = {
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "downloaded": _human_bytes(downloaded),
            "total": _human_bytes(total),
            "percent": round(percent, 2),
        }
    return {
        "condition": condition,
        "seed": seed,
        "activity": activity,
        "download": download,
    }


def _current_task(
    run_dir: Path | None,
    *,
    fallback_pool_root: Path,
    task_hint: object = None,
) -> dict[str, Any] | None:
    discovered = _task_from_hint(
        task_hint,
        pool_log_root=fallback_pool_root,
    ) or _discover_task_from_pool(fallback_pool_root)
    metadata_files = [] if discovered is not None else _task_metadata_files(run_dir)
    metadata_path: Path | None = None
    task_pool_dir: Path | None = None
    if discovered is not None:
        metadata, task_pool_dir = discovered
    elif metadata_files:
        metadata_path = metadata_files[0]
        metadata = _read_json(metadata_path, {})
        if not isinstance(metadata, dict):
            metadata = {}
    else:
        return None
    task_id = str(metadata.get("task_id") or "")
    if not _TASK_ID.fullmatch(task_id):
        return {
            "task_id": task_id,
            "state": str(metadata.get("state") or "unknown"),
            "returncode": metadata.get("returncode"),
            "metadata_path": str(metadata_path) if metadata_path else None,
            "progress": {},
        }
    if task_pool_dir is None:
        pool_log_root = _pool_log_root_from_metadata(
            metadata, fallback=fallback_pool_root
        )
        pool_id = str(metadata.get("pool_id") or "")
        config_text = _read_text(Path(str(metadata.get("pool_config") or "")))
        pool_match = re.search(
            r"^\s*pool_id:\s*[\"']?([^\"'\n]+)",
            config_text,
            re.MULTILINE,
        )
        if pool_match:
            pool_id = pool_match.group(1).strip()
        if not pool_id:
            pool_id = pool_log_root.name
        candidate = pool_log_root / pool_id
        task_pool_dir = candidate if candidate.is_dir() else pool_log_root
    task_dir = task_pool_dir / "tasks" / task_id
    stdout_lines = _tail_lines(task_dir / "stdout.log", 120)
    stderr_lines = _tail_lines(task_dir / "stderr.log", 160)
    pid_text = _read_text(task_dir / "pid").strip()
    pid = int(pid_text) if pid_text.isdigit() else None
    state = str(metadata.get("state") or "unknown")
    if state in {"starting", "running", "submitted", "unknown"} and _is_pid_alive(pid):
        state = "running"
    return {
        "task_id": task_id,
        "state": state,
        "returncode": metadata.get("returncode"),
        "pid": pid,
        "pid_alive": _is_pid_alive(pid),
        "metadata_path": str(metadata_path) if metadata_path else None,
        "task_dir": str(task_dir),
        "updated_at": datetime.fromtimestamp(
            max(
                (
                    path.stat().st_mtime
                    for path in (
                        metadata_path,
                        task_dir / "stdout.log",
                        task_dir / "stderr.log",
                    )
                    if path is not None and path.exists()
                ),
                default=time.time(),
            ),
            tz=UTC,
        ).isoformat(timespec="seconds"),
        "progress": _task_progress(stdout_lines, stderr_lines),
        "stdout_tail": stdout_lines[-24:],
        "stderr_tail": stderr_lines[-24:],
    }


def _compact_experiment(task: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if task is None:
        return None
    return {
        "task_id": task.get("task_id"),
        "state": task.get("state"),
        "returncode": task.get("returncode"),
        "pid": task.get("pid"),
        "pid_alive": task.get("pid_alive"),
        "updated_at": task.get("updated_at"),
        "progress": task.get("progress") or {},
    }


def _monitor_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    checks = snapshot.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    rendered: list[dict[str, Any]] = []
    for name, raw in checks.items():
        check = raw if isinstance(raw, dict) else {}
        data = check.get("data") if isinstance(check.get("data"), dict) else {}
        rendered.append(
            {
                "name": str(name),
                "status": str(check.get("status") or "unknown"),
                "detail": str(check.get("detail") or ""),
                "observed_at": check.get("observed_at"),
                "data": data,
            }
        )
    priority = {
        "supervisor": 0,
        "progress": 1,
        "pool": 2,
        "lease": 3,
        "bridge": 4,
        "cluster": 5,
        "gpu": 6,
        "campaign": 7,
        "checkpoint": 8,
    }
    rendered.sort(key=lambda item: priority.get(item["name"], 99))

    pool = checks.get("pool") if isinstance(checks.get("pool"), dict) else {}
    pool_data = pool.get("data") if isinstance(pool.get("data"), dict) else {}
    pool_info = pool_data.get("pool") if isinstance(pool_data.get("pool"), dict) else {}
    resources = (
        pool_info.get("resources")
        if isinstance(pool_info.get("resources"), dict)
        else {}
    )
    supervisor = (
        checks.get("supervisor")
        if isinstance(checks.get("supervisor"), dict)
        else {}
    )
    progress = (
        checks.get("progress")
        if isinstance(checks.get("progress"), dict)
        else {}
    )
    core_alive = (
        str(supervisor.get("status")) == "ok"
        and str(progress.get("status")) in {"ok", "degraded"}
    )
    return {
        "overall": str(snapshot.get("overall") or "unknown"),
        "core_alive": core_alive,
        "generated_at": snapshot.get("generated_at"),
        "paused": bool(snapshot.get("paused")),
        "checks": rendered,
        "infrastructure": {
            "pool_status": pool.get("status"),
            "pool_id": pool_info.get("pool_id"),
            "ray_started": pool_info.get("ray_started"),
            "claimed": pool_info.get("claimed"),
            "gpu_total": resources.get("total_gpu"),
            "gpu_available": resources.get("available_gpu"),
            "cpu_total": resources.get("total_cpu"),
            "alive_nodes": resources.get("alive_nodes"),
            "nodes": resources.get("nodes") or [],
        },
    }


def _artifact_candidates_unbounded(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    ignored_parts = {
        ".git",
        "__pycache__",
        "_clusterbridge_pool_project_1",
        "agent_sandbox",
        "refine_sandbox_v1",
        "refine_sandbox_v1_fix",
        "runs",
        "sandbox",
    }
    try:
        for current_root, dirnames, filenames in os.walk(run_dir):
            relative_root = Path(current_root).relative_to(run_dir)
            depth = len(relative_root.parts)
            dirnames[:] = [
                name
                for name in dirnames
                if name not in ignored_parts
                and not name.startswith(".")
                and depth < 3
            ]
            for filename in filenames:
                path = Path(current_root) / filename
                if filename.startswith("."):
                    continue
                if (
                    filename in _ARTIFACT_NAMES
                    or path.suffix.lower() in {".pdf", ".png", ".svg"}
                    or (
                        path.suffix.lower() in _ARTIFACT_EXTENSIONS
                        and any(
                            token in filename.lower()
                            for token in (
                                "paper",
                                "analysis",
                                "result",
                                "figure",
                                "topic",
                                "evidence",
                                "manifest",
                                "decision",
                                "reference",
                                "reproduc",
                            )
                        )
                    )
                ):
                    candidates.append(path)
    except OSError:
        return candidates
    return candidates


def _artifact_entries(run_dir: Path | None, limit: int = 120) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    candidates = _run_with_timeout(
        _artifact_candidates_unbounded,
        run_dir,
        timeout=4.0,
        default=[],
    )
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    result: list[dict[str, Any]] = []
    for path in candidates[: max(1, min(limit, 500))]:
        relative = path.relative_to(run_dir).as_posix()
        stat = path.stat()
        result.append(
            {
                "name": path.name,
                "path": relative,
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC
                ).isoformat(timespec="seconds"),
                "url": f"/api/artifacts/{relative}",
                "kind": path.suffix.lower().lstrip(".") or "file",
            }
        )
    return result


def _event_tail(path: Path, limit: int) -> list[dict[str, Any]]:
    events: deque[dict[str, Any]] = deque(maxlen=max(1, min(limit, 1000)))
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    events.append(value)
    except (FileNotFoundError, OSError):
        pass
    return list(reversed(events))


def _request_is_local(request: Request) -> bool:
    if request.client is None:
        return True
    return request.client.host in {"127.0.0.1", "::1", "localhost", "testclient"}


class CampaignDashboard:
    """Aggregate one durable RSI campaign into frontend-friendly JSON."""

    def __init__(
        self,
        campaign_dir: Path,
        *,
        repo_root: Path | None = None,
        pool_root: Path = DEFAULT_POOL_ROOT,
        control_enabled: bool = True,
    ) -> None:
        self.campaign_dir = campaign_dir.expanduser().resolve()
        self.repo_root = (
            repo_root.expanduser().resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.pool_root = pool_root.expanduser()
        self.control_enabled = control_enabled
        self.store = CampaignStore(self.campaign_dir)
        self._cache_payload: dict[str, Any] | None = None
        self._cache_monotonic = 0.0
        self._cache_ttl_sec = 3.0
        self._task_hint: str | None = None
        self._task_hint_refreshed_at = 0.0

    def _refresh_task_hint(self) -> None:
        discovered = _discover_task_from_pool(self.pool_root)
        if discovered is not None:
            task_id = str(discovered[0].get("task_id") or "")
            if _TASK_ID.fullmatch(task_id):
                self._task_hint = task_id
        self._task_hint_refreshed_at = time.monotonic()

    def collect(self) -> dict[str, Any]:
        now_monotonic = time.monotonic()
        if (
            self._cache_payload is not None
            and now_monotonic - self._cache_monotonic < self._cache_ttl_sec
        ):
            return self._cache_payload
        state = _read_json(self.campaign_dir / "state.json", {})
        manifest = _read_json(self.campaign_dir / "campaign.json", {})
        heartbeat = _read_json(
            self.campaign_dir / "supervisor_heartbeat.json",
            _read_json(self.campaign_dir / "heartbeat.json", {}),
        )
        monitor_snapshot = _read_json(
            self.campaign_dir / "monitor_snapshot.json", {}
        )
        if not isinstance(state, dict):
            state = {}
        if not isinstance(manifest, dict):
            manifest = {}
        if not isinstance(heartbeat, dict):
            heartbeat = {}
        if not isinstance(monitor_snapshot, dict):
            monitor_snapshot = {}

        run_dir = _latest_cycle_dir(self.campaign_dir, state)
        checkpoint = (
            _read_json(run_dir / "checkpoint.json", {})
            if run_dir is not None
            else {}
        )
        selected_topic = (
            _read_json(run_dir / "selected_topic.json", {})
            if run_dir is not None
            else {}
        )
        if not isinstance(checkpoint, dict):
            checkpoint = {}
        if not isinstance(selected_topic, dict):
            selected_topic = {}
        current_stage = _parse_stage_progress(run_dir, checkpoint)
        monitor = _monitor_summary(monitor_snapshot)
        task_hint = self._task_hint
        for check in monitor.get("checks", []):
            if check.get("name") != "progress":
                continue
            progress_data = check.get("data") or {}
            progress_path = str(progress_data.get("latest_progress_path") or "")
            match = re.search(r"tasks/(rc-pool-[a-z0-9]+)", progress_path)
            if match:
                task_hint = match.group(1)
                break
        if task_hint is None:
            progress_path = str(
                ((monitor_snapshot.get("checks") or {}).get("progress") or {})
                .get("data", {})
                .get("latest_progress_path", "")
            )
            match = re.search(r"tasks/(rc-pool-[a-z0-9]+)", progress_path)
            if match:
                task_hint = match.group(1)
        if (
            task_hint is None
            or now_monotonic - self._task_hint_refreshed_at > 60.0
        ):
            if self._task_hint_refreshed_at == 0.0:
                self._refresh_task_hint()
            else:
                _IO_POOL.submit(self._refresh_task_hint)
            task_hint = task_hint or self._task_hint

        supervisor_alive = _is_process_identity_alive(
            state.get("pid"),
            state.get("supervisor_start_ticks"),
        )
        heartbeat_age = _iso_age_seconds(heartbeat.get("timestamp"))
        heartbeat_fresh = heartbeat_age is not None and heartbeat_age <= 300
        running = (
            str(state.get("status") or "") == "running"
            and supervisor_alive
            and heartbeat_fresh
        )
        if running:
            effective_status = "running"
        elif self.store.control_requested("pause") or str(
            state.get("status") or ""
        ).startswith("paused"):
            effective_status = "pausing" if supervisor_alive else "paused"
        else:
            effective_status = str(state.get("status") or "unknown")

        created_at = manifest.get("created_at") or state.get("created_at")
        runtime_sec = _iso_age_seconds(created_at)
        active_cycle = state.get("active_cycle") or heartbeat.get("cycle")
        if active_cycle is None and run_dir is not None:
            match = re.search(r"(\d+)$", run_dir.name)
            active_cycle = int(match.group(1)) if match else None

        topic_title = (
            selected_topic.get("title")
            or state.get("selected_topic")
            or manifest.get("topic")
            or ""
        )
        payload = {
            "generated_at": utc_now(),
            "campaign": {
                "id": manifest.get("campaign_id")
                or state.get("campaign_id")
                or self.campaign_dir.name,
                "directory": str(self.campaign_dir),
                "status": effective_status,
                "raw_status": state.get("status"),
                "phase": state.get("phase") or heartbeat.get("phase"),
                "continuous": bool(
                    state.get("continuous")
                    or (manifest.get("run_policy") or {}).get("continuous")
                ),
                "cycle": active_cycle,
                "completed_cycles": state.get("completed_cycles", 0),
                "successful_cycles": state.get("successful_cycles", 0),
                "failed_cycles": state.get("failed_cycles", 0),
                "best_cycle": state.get("best_cycle"),
                "best_score": state.get("best_score"),
                "last_score": state.get("last_composite_score"),
                "runtime_sec": runtime_sec,
                "created_at": created_at,
                "updated_at": state.get("updated_at"),
                "supervisor_pid": state.get("pid"),
                "supervisor_alive": supervisor_alive,
                "child_pid": state.get("active_child_pid")
                or heartbeat.get("child_pid"),
                "heartbeat_at": heartbeat.get("timestamp"),
                "heartbeat_age_sec": heartbeat_age,
                "heartbeat_fresh": heartbeat_fresh,
                "pause_requested": self.store.control_requested("pause"),
                "automatic_submission_enabled": bool(
                    state.get("automatic_submission_enabled", False)
                ),
            },
            "run": {
                "directory": str(run_dir) if run_dir is not None else None,
                "name": run_dir.name if run_dir is not None else None,
                "run_id": current_stage.get("run_id")
                or checkpoint.get("run_id"),
            },
            "topic": {
                "title": topic_title,
                "id": selected_topic.get("id")
                or state.get("selected_topic_id"),
                "research_question": selected_topic.get("research_question"),
                "hypothesis": selected_topic.get("falsifiable_hypothesis"),
                "primary_metric": selected_topic.get("primary_metric"),
                "weighted_score": selected_topic.get("weighted_score"),
                "selection_rationale": selected_topic.get(
                    "selection_rationale"
                ),
                "candidate_count": state.get("topic_candidate_count"),
                "compute": selected_topic.get("compute") or {},
                "models": selected_topic.get("models") or [],
                "datasets": selected_topic.get("datasets") or [],
            },
            "progress": {
                "current_stage": current_stage,
                "checkpoint": {
                    "number": checkpoint.get("last_completed_stage"),
                    "name": checkpoint.get("last_completed_name"),
                    "timestamp": checkpoint.get("timestamp"),
                    "run_id": checkpoint.get("run_id"),
                },
                "stages": _stage_statuses(
                    run_dir,
                    current_stage=current_stage,
                    checkpoint=checkpoint,
                ),
            },
            "experiment": _compact_experiment(
                _current_task(
                    run_dir,
                    fallback_pool_root=self.pool_root,
                    task_hint=(
                        task_hint
                        if task_hint and task_hint != self._task_hint
                        else None
                    ),
                )
            ),
            "monitor": monitor,
            "controls": {
                "enabled": self.control_enabled,
                "can_pause": self.control_enabled and running,
                "can_resume": self.control_enabled
                and effective_status in {
                    "paused",
                    "crashed",
                    "stopped",
                    "paused_failure_threshold",
                    "paused_no_improvement",
                }
                and not supervisor_alive,
                "dangerous_stop_exposed": False,
            },
            "last_comparison": state.get("last_comparison") or {},
            "last_error": state.get("last_error"),
        }
        self._cache_payload = payload
        self._cache_monotonic = time.monotonic()
        return payload

    def logs(self, source: str, tail: int) -> dict[str, Any]:
        state = _read_json(self.campaign_dir / "state.json", {})
        if not isinstance(state, dict):
            state = {}
        run_dir = _latest_cycle_dir(self.campaign_dir, state)
        if source == "pipeline":
            path = run_dir / "pipeline.log" if run_dir is not None else None
        elif source == "pipeline_events":
            path = (
                run_dir / "pipeline_events.jsonl"
                if run_dir is not None
                else None
            )
        elif source == "llm_audit":
            audit_paths = (
                list((run_dir / "audit").glob("llm-*.jsonl"))
                if run_dir is not None
                else []
            )
            path = (
                max(
                    audit_paths,
                    key=lambda candidate: candidate.stat().st_mtime,
                )
                if audit_paths
                else None
            )
        elif source.startswith("llm_audit:"):
            role = source.partition(":")[2]
            if not re.fullmatch(r"[a-z0-9_.-]+", role):
                raise ValueError("invalid LLM audit role")
            path = (
                run_dir / "audit" / f"llm-{role}.jsonl"
                if run_dir is not None
                else None
            )
        elif source == "observability":
            path = (
                run_dir / "observability_summary.json"
                if run_dir is not None
                else None
            )
        elif source == "supervisor":
            path = self.campaign_dir / "supervisor.log"
        elif source in {"experiment", "experiment_stdout", "experiment_stderr"}:
            task = _current_task(run_dir, fallback_pool_root=self.pool_root)
            if not task or not task.get("task_dir"):
                path = None
            else:
                filename = (
                    "stderr.log"
                    if source == "experiment_stderr"
                    else "stdout.log"
                )
                path = Path(str(task["task_dir"])) / filename
        else:
            raise ValueError(f"unsupported log source: {source}")
        lines = _tail_lines(path, tail) if path is not None else []
        return {
            "source": source,
            "path": str(path) if path is not None else None,
            "lines": lines,
            "text": "\n".join(lines),
        }

    def pause(self, reason: str) -> dict[str, Any]:
        state = self.store.load_state()
        if self.store.control_requested("pause"):
            return {
                "accepted": True,
                "already_requested": True,
                "status": state.get("status"),
            }
        path = self.store.set_control(
            "pause", reason or "dashboard requested cooperative pause"
        )
        self.store.log.append(
            "pause_requested",
            campaign_id=str(
                state.get("campaign_id") or self.campaign_dir.name
            ),
            reason=reason or "dashboard requested cooperative pause",
            source="rsi_dashboard",
        )
        try:
            pid = int(state.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        notified = False
        if pid > 0 and hasattr(signal, "SIGUSR1"):
            try:
                os.kill(pid, signal.SIGUSR1)
                notified = True
            except (ProcessLookupError, PermissionError):
                pass
        return {
            "accepted": True,
            "already_requested": False,
            "control_path": str(path),
            "supervisor_notified": notified,
            "semantics": "cooperative_pause_not_stop",
        }

    def resume(self, reason: str) -> dict[str, Any]:
        state = self.store.load_state()
        if _is_process_identity_alive(
            state.get("pid"), state.get("supervisor_start_ticks")
        ):
            raise RuntimeError("supervisor is already running")
        campaign_id = str(
            state.get("campaign_id") or self.campaign_dir.name
        )
        unit = f"autoresearch-rsi-supervisor@{campaign_id}.service"
        command = ["systemctl", "--no-block", "start", unit]
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.store.log.append(
            "resume_requested_from_dashboard",
            campaign_id=campaign_id,
            reason=reason or "dashboard requested resume",
            unit=unit,
            returncode=completed.returncode,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"systemctl returned {completed.returncode}"
            )
        return {
            "accepted": True,
            "unit": unit,
            "command": command,
            "detail": completed.stdout.strip(),
        }

    def artifact_path(self, relative_path: str) -> Path:
        state = _read_json(self.campaign_dir / "state.json", {})
        if not isinstance(state, dict):
            state = {}
        run_dir = _latest_cycle_dir(self.campaign_dir, state)
        if run_dir is None:
            raise FileNotFoundError("no campaign run directory")
        if Path(relative_path).is_absolute():
            raise ValueError("artifact path must be relative")
        path = _safe_child(run_dir, run_dir / relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        return path


def create_dashboard_app(
    campaign_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    pool_root: str | Path = DEFAULT_POOL_ROOT,
    control_enabled: bool = True,
) -> FastAPI:
    dashboard = CampaignDashboard(
        Path(campaign_dir),
        repo_root=Path(repo_root) if repo_root is not None else None,
        pool_root=Path(pool_root),
        control_enabled=control_enabled,
    )
    static_dir = Path(__file__).with_name("static")
    app = FastAPI(
        title="ResearchClaw RSI Dashboard",
        version="1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.dashboard = dashboard
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def local_control_guard(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/api/control/") and not _request_is_local(
            request
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "control endpoints are local-only"},
            )
        return await call_next(request)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        payload = dashboard.collect()
        return {
            "status": "ok",
            "service": "researchclaw-rsi-dashboard",
            "campaign_id": payload["campaign"]["id"],
            "campaign_status": payload["campaign"]["status"],
            "generated_at": payload["generated_at"],
        }

    @app.get("/api/dashboard")
    def dashboard_state() -> dict[str, Any]:
        return dashboard.collect()

    @app.get("/api/events")
    def events(
        limit: int = Query(default=80, ge=1, le=1000),
    ) -> dict[str, Any]:
        return {
            "events": _event_tail(dashboard.store.events_path, limit),
            "limit": limit,
        }

    @app.get("/api/logs")
    def logs(
        source: str = Query(default="pipeline"),
        tail: int = Query(default=160, ge=1, le=2000),
    ) -> dict[str, Any]:
        try:
            return dashboard.logs(source, tail)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/artifacts")
    def artifacts(
        limit: int = Query(default=120, ge=1, le=500),
    ) -> dict[str, Any]:
        state = _read_json(dashboard.campaign_dir / "state.json", {})
        run_dir = _latest_cycle_dir(
            dashboard.campaign_dir,
            state if isinstance(state, dict) else {},
        )
        return {
            "run_dir": str(run_dir) if run_dir is not None else None,
            "artifacts": _artifact_entries(run_dir, limit=limit),
        }

    @app.get("/api/artifacts/{relative_path:path}")
    def artifact(relative_path: str) -> FileResponse:
        try:
            path = dashboard.artifact_path(relative_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        disposition = "inline" if path.suffix.lower() in {
            ".html",
            ".md",
            ".pdf",
            ".png",
            ".svg",
            ".txt",
        } else None
        return FileResponse(
            path,
            filename=None if disposition else path.name,
            content_disposition_type=disposition or "attachment",
        )

    @app.post("/api/control/pause")
    def pause(request: Request, payload: ControlRequest) -> dict[str, Any]:
        if not dashboard.control_enabled:
            raise HTTPException(status_code=403, detail="controls are disabled")
        if not _request_is_local(request):
            raise HTTPException(status_code=403, detail="local access required")
        try:
            return dashboard.pause(payload.reason.strip())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/control/resume")
    def resume(request: Request, payload: ControlRequest) -> dict[str, Any]:
        if not dashboard.control_enabled:
            raise HTTPException(status_code=403, detail="controls are disabled")
        if not _request_is_local(request):
            raise HTTPException(status_code=403, detail="local access required")
        try:
            return dashboard.resume(payload.reason.strip())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (RuntimeError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m researchclaw.rsi.dashboard",
        description="Serve a campaign-aware ResearchClaw RSI dashboard.",
    )
    parser.add_argument(
        "--campaign-dir",
        default=str(DEFAULT_CAMPAIGN_DIR),
        help="durable RSI campaign directory",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--no-controls",
        action="store_true",
        help="disable cooperative pause/resume endpoints",
    )
    parser.add_argument(
        "--pool-root",
        default=str(DEFAULT_POOL_ROOT),
        help="ClusterBridge pool log root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    campaign_dir = Path(args.campaign_dir).expanduser().resolve()
    if not (campaign_dir / "campaign.json").is_file():
        print(
            f"rsi-dashboard: campaign not found: {campaign_dir}",
            file=sys.stderr,
        )
        return 2
    import uvicorn

    app = create_dashboard_app(
        campaign_dir,
        pool_root=args.pool_root,
        control_enabled=not args.no_controls,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
