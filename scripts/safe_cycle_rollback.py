#!/usr/bin/env python3
"""Fail-closed rollback helper for an existing RSI cycle directory.

The helper is deliberately independent from the pipeline and supervisor core.
It never signals a process, cancels a remote task, changes campaign state, or
releases resources.  Its only mutating operation is an atomic, same-filesystem
archive of invalid run artifacts followed by an atomic checkpoint rewind.

Default mode is a read-only plan.  ``--apply`` is accepted only when the
campaign is already durably paused, no recorded supervisor/pipeline process is
alive, the supervisor lock is free, and every recorded ClusterBridge pool task
is terminal.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


_RUN_NAME_RE = re.compile(r"cycle-(\d{4})$")
_STAGE_DIR_RE = re.compile(r"stage-(\d{2})(?:_.*)?$")
_TERMINAL_TASK_STATES = {"finished", "timed_out", "failed", "cancelled"}

_ROOT_ARTIFACTS = (
    "checkpoint.json",
    "heartbeat.json",
    "events.jsonl",
    "experiment_diagnosis.json",
    "repair_prompt.txt",
    "experiment_repair_result.json",
    "experiment_summary_best.json",
    "analysis_best.md",
    "results.json",
    # ``execute_pipeline`` rewrites this after every bounded phase.  Treat the
    # previous phase's summary as transactional scratch, not as a final Cycle2
    # result that should survive into the next phase.
    "pipeline_summary.json",
    "decision_history.json",
    "quality_warning.txt",
    "iteration_context.json",
    "iteration_summary.json",
    "REPAIR_PROMPT.md",
    "requirements_verdict.json",
    "degradation_signal.json",
    "rsi_evidence.json",
)

_REQUIRED_PRIOR_OUTPUTS: dict[int, tuple[str, ...]] = {
    1: ("goal.md", "hardware_profile.json"),
    2: ("problem_tree.md",),
    3: ("search_plan.yaml", "sources.json", "queries.json"),
    4: ("candidates.jsonl",),
    5: ("shortlist.jsonl",),
    6: ("cards/",),
    7: ("synthesis.md",),
    8: ("hypotheses.md",),
}


class SafetyError(RuntimeError):
    """Raised when rollback would not be safe."""


@dataclass(frozen=True)
class ProcessRecord:
    role: str
    pid: int
    expected_start_ticks: int | None
    observed_start_ticks: int | None
    alive: bool
    identity_matches: bool


@dataclass(frozen=True)
class TaskRecord:
    metadata_path: str
    task_id: str
    state: str
    terminal: bool


@dataclass(frozen=True)
class RollbackPlan:
    campaign_dir: str
    run_dir: str
    cycle: int
    from_stage: int
    checkpoint_stage: int
    checkpoint_name: str
    checkpoint_run_id: str
    run_id: str
    backup_dir: str
    move_names: tuple[str, ...]
    pause_requested: bool
    campaign_status: str
    campaign_phase: str
    campaign_identity_preserved: bool
    supervisor_lock_free: bool
    processes: tuple[ProcessRecord, ...]
    tasks: tuple[TaskRecord, ...]
    prior_artifacts_complete: bool
    safe_to_apply: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _timestamp_slug() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return default


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _process_start_ticks(pid: int) -> int | None:
    if pid <= 0:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _process_record(
    role: str,
    pid_value: Any,
    expected_ticks_value: Any,
) -> ProcessRecord:
    pid = _optional_int(pid_value) or 0
    expected_ticks = _optional_int(expected_ticks_value)
    observed_ticks = _process_start_ticks(pid)
    alive = observed_ticks is not None
    identity_matches = (
        alive
        and expected_ticks is not None
        and observed_ticks == expected_ticks
    )
    return ProcessRecord(
        role=role,
        pid=pid,
        expected_start_ticks=expected_ticks,
        observed_start_ticks=observed_ticks,
        alive=alive,
        identity_matches=identity_matches,
    )


def _pause_requested(campaign_dir: Path) -> bool:
    return any(
        path.exists()
        for path in (
            campaign_dir / "control" / "pause",
            campaign_dir / "pause.request.json",
        )
    )


def _lock_is_free(lock_path: Path) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


@contextmanager
def _helper_lock(run_dir: Path) -> Iterable[None]:
    """Prevent two helper actions from mutating/running the same cycle."""

    lock_path = run_dir.expanduser().resolve() / ".safe_cycle_rollback.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SafetyError(
                f"another safe-cycle-rollback action holds {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "acquired_at": _utc_now()}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _campaign_guard_lock(campaign_dir: Path) -> Iterable[None]:
    """Exclude a supervisor start for the whole rollback or bounded rerun."""

    lock_path = campaign_dir.expanduser().resolve() / "supervisor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SafetyError(
                f"cannot guard recovery because supervisor.lock is held: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_run_identity(campaign_dir: Path, run_dir: Path) -> int:
    match = _RUN_NAME_RE.fullmatch(run_dir.name)
    if match is None:
        raise SafetyError(f"run directory must be named cycle-NNNN: {run_dir}")
    try:
        expected_parent = (campaign_dir / "runs").resolve()
        actual_parent = run_dir.resolve().parent
    except OSError as exc:
        raise SafetyError(f"cannot resolve campaign/run paths: {exc}") from exc
    if actual_parent != expected_parent:
        raise SafetyError(
            f"run is not directly below {expected_parent}: {run_dir.resolve()}"
        )
    return int(match.group(1))


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _checkpoint_name(stage: int) -> str:
    try:
        from researchclaw.pipeline.stages import Stage

        return Stage(stage).name
    except Exception as exc:
        raise SafetyError(f"invalid checkpoint stage {stage}: {exc}") from exc


def _validate_prior_artifacts(run_dir: Path, checkpoint_stage: int) -> list[str]:
    missing: list[str] = []
    for stage in range(1, min(checkpoint_stage, 8) + 1):
        stage_dir = run_dir / f"stage-{stage:02d}"
        if not stage_dir.is_dir():
            missing.append(str(stage_dir))
            continue
        for relative in _REQUIRED_PRIOR_OUTPUTS.get(stage, ()):
            target = stage_dir / relative.rstrip("/")
            if relative.endswith("/"):
                if not target.is_dir():
                    missing.append(str(target) + "/")
            elif not target.is_file() or target.stat().st_size == 0:
                missing.append(str(target))
    return missing


def _prior_stage_run_ids(
    run_dir: Path,
    checkpoint_stage: int,
) -> tuple[dict[int, str], list[str]]:
    run_ids: dict[int, str] = {}
    missing: list[str] = []
    for stage in range(1, checkpoint_stage + 1):
        decision_path = run_dir / f"stage-{stage:02d}" / "decision.json"
        decision = _read_json(decision_path, None)
        if not isinstance(decision, Mapping):
            missing.append(str(decision_path))
            continue
        run_id = str(decision.get("run_id", "") or "").strip()
        if not run_id:
            missing.append(f"{decision_path}:run_id")
            continue
        run_ids[stage] = run_id
    return run_ids, missing


def _task_records(run_dir: Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for path in sorted(run_dir.rglob(".clusterbridge_pool_task.json")):
        value = _read_json(path, {})
        if not isinstance(value, Mapping):
            records.append(
                TaskRecord(
                    metadata_path=str(path),
                    task_id="",
                    state="invalid_metadata",
                    terminal=False,
                )
            )
            continue
        task_id = str(value.get("task_id", "") or "").strip()
        state = str(value.get("state", "") or "").strip().lower()
        records.append(
            TaskRecord(
                metadata_path=str(path),
                task_id=task_id,
                state=state or "unknown",
                terminal=bool(task_id) and state in _TERMINAL_TASK_STATES,
            )
        )
    return records


def _move_names(run_dir: Path, from_stage: int) -> tuple[str, ...]:
    names: set[str] = set()
    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        match = _STAGE_DIR_RE.fullmatch(child.name)
        if match is not None and int(match.group(1)) >= from_stage:
            names.add(child.name)
    for name in _ROOT_ARTIFACTS:
        if (run_dir / name).exists():
            names.add(name)
    for name in ("audit", "experiment_memory", "hitl"):
        path = run_dir / name
        if path.is_dir() and any(path.iterdir()):
            names.add(name)
    return tuple(sorted(names))


def _new_backup_dir(run_dir: Path, from_stage: int) -> Path:
    root = run_dir / "rollback_backups"
    stem = f"from-stage-{from_stage:02d}-{_timestamp_slug()}"
    candidate = root / stem
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stem}-{suffix:02d}"
        suffix += 1
    return candidate


def _archive_phase_summary(run_dir: Path, from_stage: int, to_stage: int) -> Path | None:
    """Move the previous bounded phase summary outside the canonical root."""

    summary_path = run_dir / "pipeline_summary.json"
    if not summary_path.exists():
        return None
    archive_dir = run_dir / "rollback_backups" / "phase_summaries"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"before-stage-{from_stage:02d}-to-{to_stage:02d}-"
        f"{_timestamp_slug()}.json"
    )
    destination = archive_dir / stem
    suffix = 1
    while destination.exists():
        destination = archive_dir / f"{Path(stem).stem}-{suffix:02d}.json"
        suffix += 1
    os.replace(summary_path, destination)
    return destination


def _rewrite_final_pipeline_summary(
    run_dir: Path,
    *,
    run_id: str,
    from_stage: int,
    to_stage: int,
) -> None:
    """Normalize the final bounded summary so supervisor sees a full Cycle2 run."""

    summary_path = run_dir / "pipeline_summary.json"
    summary = _read_json(summary_path, None)
    if not isinstance(summary, Mapping):
        raise SafetyError(f"final pipeline summary is missing or invalid: {summary_path}")
    expected_stages = to_stage - from_stage + 1
    try:
        final_stage = int(summary.get("final_stage", -1))
        stages_executed = int(summary.get("stages_executed", -1))
        stages_failed = int(summary.get("stages_failed", -1))
        stages_paused = int(summary.get("stages_paused", -1))
        stages_blocked = int(summary.get("stages_blocked", -1))
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"invalid final pipeline summary counters: {exc}") from exc
    if (
        str(summary.get("run_id", "") or "") != run_id
        or final_stage != to_stage
        or stages_executed != expected_stages
        or stages_failed != 0
        or stages_paused != 0
        or stages_blocked != 0
        or str(summary.get("final_status", "") or "").lower() != "done"
    ):
        raise SafetyError(
            "refusing to normalize incomplete final pipeline summary: "
            f"run_id={summary.get('run_id')!r}, final_stage={final_stage}, "
            f"stages_executed={stages_executed}, failed={stages_failed}, "
            f"paused={stages_paused}, blocked={stages_blocked}, "
            f"final_status={summary.get('final_status')!r}"
        )
    normalized = dict(summary)
    # Preserve the runner's bounded counts and scope in recovery-specific
    # fields.  The canonical counters are cumulative because stage-01..08 were
    # retained and their decision artifacts remain part of this same run.
    normalized.update(
        {
            "from_stage": 1,
            "stages_executed": to_stage,
            "stages_done": to_stage,
            "recovered_from_stage": from_stage,
            "bounded_final_phase_stages": expected_stages,
            "bounded_final_phase_done": int(summary.get("stages_done", 0) or 0),
            "summary_scope": "cycle_recovery_full_pipeline",
            "normalized_at": _utc_now(),
        }
    )
    _atomic_write_json(summary_path, normalized)


def build_plan(
    *,
    campaign_dir: Path,
    run_dir: Path,
    from_stage: int = 9,
    backup_dir: Path | None = None,
    _supervisor_lock_held: bool = False,
) -> RollbackPlan:
    campaign_dir = campaign_dir.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    cycle = _validate_run_identity(campaign_dir, run_dir)
    if from_stage < 2 or from_stage > 23:
        raise SafetyError("--from-stage must be between 2 and 23")
    checkpoint_stage = from_stage - 1
    checkpoint_name = _checkpoint_name(checkpoint_stage)

    state = _read_json(campaign_dir / "state.json", {})
    if not isinstance(state, Mapping):
        raise SafetyError(f"invalid campaign state: {campaign_dir / 'state.json'}")
    checkpoint = _read_json(run_dir / "checkpoint.json", {})
    if not isinstance(checkpoint, Mapping):
        raise SafetyError(f"invalid checkpoint: {run_dir / 'checkpoint.json'}")
    checkpoint_run_id = str(checkpoint.get("run_id", "") or "").strip()

    pause = _pause_requested(campaign_dir)
    status = str(state.get("status", "") or "").strip().lower()
    phase = str(state.get("phase", "") or "").strip().lower()
    state_active_cycle = _optional_int(state.get("active_cycle"))
    state_next_cycle = _optional_int(state.get("next_cycle"))
    state_completed_cycles = _optional_int(state.get("completed_cycles"))
    active_run_raw = str(state.get("active_run_dir", "") or "").strip()
    active_run_matches = bool(active_run_raw) and (
        Path(active_run_raw).expanduser().resolve() == run_dir
    )
    unfinished_cycle = (
        state_next_cycle == cycle
        and (
            state_completed_cycles is None
            or state_completed_cycles < cycle
        )
    )
    active_identity_matches = state_active_cycle == cycle and active_run_matches
    paused_identity_matches = (
        state_active_cycle is None
        and not active_run_raw
    )
    # ``_finish_for_control("pause")`` intentionally clears active_cycle and
    # active_run_dir while leaving next_cycle unchanged.  That is still the
    # same unfinished Cycle2 identity, not a finalized/advanced campaign.
    campaign_identity_preserved = unfinished_cycle and (
        active_identity_matches or paused_identity_matches
    )
    lock_free = (
        True
        if _supervisor_lock_held
        else _lock_is_free(campaign_dir / "supervisor.lock")
    )
    processes = (
        _process_record(
            "supervisor",
            state.get("pid"),
            state.get("supervisor_start_ticks"),
        ),
        _process_record(
            "pipeline_child",
            state.get("active_child_pid"),
            state.get("active_child_start_ticks"),
        ),
    )
    tasks = tuple(_task_records(run_dir))
    missing_prior = _validate_prior_artifacts(run_dir, checkpoint_stage)
    prior_run_ids, missing_prior_decisions = _prior_stage_run_ids(
        run_dir,
        checkpoint_stage,
    )
    unique_prior_run_ids = set(prior_run_ids.values())
    run_id = (
        next(iter(unique_prior_run_ids))
        if len(unique_prior_run_ids) == 1
        else checkpoint_run_id
    )
    move_names = _move_names(run_dir, from_stage)
    resolved_backup = (
        backup_dir.expanduser().resolve()
        if backup_dir is not None
        else _new_backup_dir(run_dir, from_stage)
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if not pause:
        blockers.append(
            "campaign has no durable pause marker; monitor/supervisor restart is not suppressed"
        )
    if not campaign_identity_preserved:
        blockers.append(
            "campaign state no longer identifies this run as the unfinished active "
            f"cycle (active_cycle={state_active_cycle}, next_cycle={state_next_cycle}, "
            f"completed_cycles={state_completed_cycles}, active_run_matches="
            f"{active_run_matches})"
        )
    if status not in {"paused", "paused_single_cycle", "paused_failure",
                      "paused_failure_threshold", "paused_no_improvement",
                      "interrupted"}:
        blockers.append(f"campaign status is not quiescent: {status or '<missing>'}")
    if phase not in {"", "idle", "exited"}:
        blockers.append(f"campaign phase is not idle: {phase}")
    if not lock_free:
        blockers.append("supervisor.lock is held")
    for process in processes:
        if process.alive:
            identity = (
                "recorded identity matches"
                if process.identity_matches
                else "PID exists but identity is stale/ambiguous"
            )
            blockers.append(
                f"{process.role} pid {process.pid} is alive ({identity})"
            )
    for task in tasks:
        if not task.terminal:
            blockers.append(
                f"non-terminal/unknown pool task {task.task_id or '<missing-id>'}: "
                f"{task.state} ({task.metadata_path})"
            )
    if missing_prior:
        blockers.append(
            "required pre-rollback artifacts are missing: "
            + ", ".join(missing_prior[:12])
            + (" ..." if len(missing_prior) > 12 else "")
        )
    if missing_prior_decisions:
        blockers.append(
            "required prior stage decisions/run_ids are missing: "
            + ", ".join(missing_prior_decisions[:12])
            + (" ..." if len(missing_prior_decisions) > 12 else "")
        )
    if len(unique_prior_run_ids) > 1:
        detail = ", ".join(
            f"stage-{stage:02d}={value}"
            for stage, value in sorted(prior_run_ids.items())
        )
        blockers.append(f"prior stage decisions have inconsistent run_ids: {detail}")
    if not checkpoint_run_id:
        blockers.append("checkpoint has no run_id")
    if not run_id:
        blockers.append("cannot determine preserved pipeline run_id")
    if (
        checkpoint_run_id
        and run_id
        and checkpoint_run_id != run_id
        and len(unique_prior_run_ids) == 1
    ):
        warnings.append(
            "checkpoint run_id differs from retained Stage 1-"
            f"{checkpoint_stage} lineage; rollback will restore {run_id!r} "
            f"instead of resumed CLI id {checkpoint_run_id!r}"
        )
    if not move_names:
        blockers.append(f"no artifacts at or after Stage {from_stage} were found")
    if not _path_is_within(resolved_backup, run_dir / "rollback_backups"):
        blockers.append(
            "backup directory must be below RUN/rollback_backups to stay outside stage-* scans"
        )
    if resolved_backup.exists():
        manifest = _read_json(resolved_backup / "rollback_manifest.json", {})
        resumable_interrupted = (
            isinstance(manifest, Mapping)
            and manifest.get("status") == "interrupted"
            and int(manifest.get("from_stage", -1)) == from_stage
            and str(manifest.get("run_id", "") or "") == run_id
            and str(manifest.get("run_dir", "") or "") == str(run_dir)
        )
        if resumable_interrupted:
            warnings.append(
                f"resuming interrupted rollback transaction: {resolved_backup}"
            )
        else:
            blockers.append(f"backup destination already exists: {resolved_backup}")
    if any(name.startswith("stage-") for name in resolved_backup.parts[-1:]):
        blockers.append("backup leaf must not begin with stage-")
    if int(checkpoint.get("last_completed_stage", 0) or 0) < checkpoint_stage:
        warnings.append(
            "checkpoint is already before the requested rollback boundary; "
            "artifact archiving is still possible but should be reviewed"
        )

    return RollbackPlan(
        campaign_dir=str(campaign_dir),
        run_dir=str(run_dir),
        cycle=cycle,
        from_stage=from_stage,
        checkpoint_stage=checkpoint_stage,
        checkpoint_name=checkpoint_name,
        checkpoint_run_id=checkpoint_run_id,
        run_id=run_id,
        backup_dir=str(resolved_backup),
        move_names=move_names,
        pause_requested=pause,
        campaign_status=status,
        campaign_phase=phase,
        campaign_identity_preserved=campaign_identity_preserved,
        supervisor_lock_free=lock_free,
        processes=processes,
        tasks=tasks,
        prior_artifacts_complete=not missing_prior,
        safe_to_apply=not blockers,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def _write_transaction_manifest(
    backup_dir: Path,
    *,
    plan: RollbackPlan,
    status: str,
    moves: Iterable[Mapping[str, Any]],
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "updated_at": _utc_now(),
        "campaign_dir": plan.campaign_dir,
        "run_dir": plan.run_dir,
        "cycle": plan.cycle,
        "from_stage": plan.from_stage,
        "checkpoint_stage": plan.checkpoint_stage,
        "checkpoint_name": plan.checkpoint_name,
        "checkpoint_run_id": plan.checkpoint_run_id,
        "run_id": plan.run_id,
        "moves": list(moves),
        "safety_plan": _plan_dict(plan),
    }
    if error:
        payload["error"] = error
    _atomic_write_json(backup_dir / "rollback_manifest.json", payload)


def apply_plan(plan: RollbackPlan) -> Path:
    if not plan.safe_to_apply:
        raise SafetyError("rollback plan is unsafe: " + "; ".join(plan.blockers))

    campaign_dir = Path(plan.campaign_dir)
    run_dir = Path(plan.run_dir)
    backup_dir = Path(plan.backup_dir)

    # Acquire the same lock used by the supervisor before re-checking. This
    # closes the plan/apply race and prevents a manual/systemd resume from
    # starting a duplicate pipeline while artifacts are being moved.
    with _campaign_guard_lock(campaign_dir):
        fresh = build_plan(
            campaign_dir=campaign_dir,
            run_dir=run_dir,
            from_stage=plan.from_stage,
            backup_dir=backup_dir,
            _supervisor_lock_held=True,
        )
        if not fresh.safe_to_apply:
            raise SafetyError(
                "safety state changed before apply: " + "; ".join(fresh.blockers)
            )

        # Keep the live checkpoint readable until the replacement is written.
        # All other artifacts are renamed; checkpoint.json is copied into the
        # backup and then atomically overwritten in place.
        ordered_names = tuple(
            name for name in fresh.move_names if name != "checkpoint.json"
        )
        manifest_path = backup_dir / "rollback_manifest.json"
        if backup_dir.exists():
            manifest = _read_json(manifest_path, {})
            if not isinstance(manifest, Mapping) or manifest.get("status") != "interrupted":
                raise SafetyError(
                    f"backup destination exists without an interrupted manifest: "
                    f"{backup_dir}"
                )
            if (
                int(manifest.get("from_stage", -1)) != fresh.from_stage
                or str(manifest.get("run_id", "") or "") != fresh.run_id
                or str(manifest.get("run_dir", "") or "") != fresh.run_dir
            ):
                raise SafetyError(
                    f"interrupted manifest does not match current rollback: "
                    f"{manifest_path}"
                )
            moved = [
                dict(item)
                for item in manifest.get("moves", [])
                if isinstance(item, Mapping)
            ]
        else:
            backup_dir.mkdir(parents=True, exist_ok=False)
            moved = []
            _write_transaction_manifest(
                backup_dir,
                plan=fresh,
                status="in_progress",
                moves=moved,
            )
        try:
            for name in ordered_names:
                source = run_dir / name
                destination = backup_dir / name
                if destination.exists() and not source.exists():
                    continue
                if not source.exists():
                    raise SafetyError(f"artifact disappeared during rollback: {source}")
                if destination.exists():
                    raise SafetyError(f"backup destination collision: {destination}")
                os.replace(source, destination)
                moved.append(
                    {
                        "source": str(source),
                        "destination": str(destination),
                        "moved_at": _utc_now(),
                    }
                )
                _write_transaction_manifest(
                    backup_dir,
                    plan=fresh,
                    status="in_progress",
                    moves=moved,
                )

            checkpoint_source = run_dir / "checkpoint.json"
            checkpoint_destination = backup_dir / "checkpoint.json"
            if not checkpoint_destination.exists():
                if not checkpoint_source.is_file():
                    raise SafetyError(
                        f"checkpoint disappeared during rollback: {checkpoint_source}"
                    )
                shutil.copy2(checkpoint_source, checkpoint_destination)
                moved.append(
                    {
                        "source": str(checkpoint_source),
                        "destination": str(checkpoint_destination),
                        "operation": "copy",
                        "moved_at": _utc_now(),
                    }
                )
                _write_transaction_manifest(
                    backup_dir,
                    plan=fresh,
                    status="in_progress",
                    moves=moved,
                )

            checkpoint = {
                "last_completed_stage": fresh.checkpoint_stage,
                "last_completed_name": fresh.checkpoint_name,
                "run_id": fresh.run_id,
                "timestamp": _utc_now(),
                "rollback_from_stage": fresh.from_stage,
                "rollback_backup": str(backup_dir.relative_to(run_dir)),
            }
            _atomic_write_json(run_dir / "checkpoint.json", checkpoint)
            _write_transaction_manifest(
                backup_dir,
                plan=fresh,
                status="complete",
                moves=moved,
            )
            return backup_dir
        except BaseException as exc:
            try:
                _write_transaction_manifest(
                    backup_dir,
                    plan=fresh,
                    status="interrupted",
                    moves=moved,
                    error=f"{type(exc).__name__}: {exc}",
                )
            except Exception:
                pass
            raise


def run_pipeline_phase(
    *,
    campaign_dir: Path,
    run_dir: Path,
    from_stage: int,
    to_stage: int,
    skip_preflight: bool = False,
    allow_followup: bool = False,
) -> dict[str, Any]:
    """Run one bounded pipeline phase while preserving checkpoint ``run_id``.

    This bypasses the public CLI's unconditional ``_generate_run_id`` call but
    otherwise uses the same RCConfig, AdapterBundle, and execute_pipeline API.
    It refuses to run unless the same quiescence checks used by ``--apply``
    pass and the checkpoint is exactly at ``from_stage - 1``. The supervisor
    lock is held for the whole bounded phase to exclude duplicate launches.
    """

    with _campaign_guard_lock(campaign_dir):
        return _run_pipeline_phase_guarded(
            campaign_dir=campaign_dir,
            run_dir=run_dir,
            from_stage=from_stage,
            to_stage=to_stage,
            skip_preflight=skip_preflight,
            allow_followup=allow_followup,
        )


def _run_pipeline_phase_guarded(
    *,
    campaign_dir: Path,
    run_dir: Path,
    from_stage: int,
    to_stage: int,
    skip_preflight: bool,
    allow_followup: bool,
) -> dict[str, Any]:
    if to_stage < from_stage:
        raise SafetyError("--to-stage must be greater than or equal to --from-stage")
    plan = build_plan(
        campaign_dir=campaign_dir,
        run_dir=run_dir,
        from_stage=from_stage,
        _supervisor_lock_held=True,
    )
    # A successful rollback deliberately leaves no artifacts at or after the
    # next phase boundary.  That condition is a blocker for ``--apply`` (there
    # would be nothing to archive), but it is the expected precondition for a
    # bounded rerun.
    allowed_plan_blockers: set[str] = {
        blocker
        for blocker in plan.blockers
        if blocker == f"no artifacts at or after Stage {from_stage} were found"
    }
    if allow_followup:
        allowed_plan_blockers.update(
            blocker
            for blocker in plan.blockers
            if blocker.startswith("required pre-rollback artifacts are missing:")
        )
    effective_blockers = [
        blocker for blocker in plan.blockers if blocker not in allowed_plan_blockers
    ]
    if effective_blockers:
        raise SafetyError(
            "pipeline phase is unsafe to start: " + "; ".join(effective_blockers)
        )

    checkpoint = _read_json(Path(plan.run_dir) / "checkpoint.json", {})
    if not isinstance(checkpoint, Mapping):
        raise SafetyError("checkpoint is missing or invalid")
    completed = _optional_int(checkpoint.get("last_completed_stage"))
    if completed != from_stage - 1:
        raise SafetyError(
            f"checkpoint stage is {completed}, expected {from_stage - 1}; "
            "do not skip or repeat a phase accidentally"
        )
    checkpoint_run_id = str(checkpoint.get("run_id", "") or "").strip()
    if checkpoint_run_id != plan.run_id:
        raise SafetyError(
            f"checkpoint run_id changed during safety checks: "
            f"{checkpoint_run_id!r} != {plan.run_id!r}"
        )
    run_path = Path(plan.run_dir)
    for stage_num in range(1, from_stage):
        decision_path = run_path / f"stage-{stage_num:02d}" / "decision.json"
        decision = _read_json(decision_path, None)
        if not isinstance(decision, Mapping):
            raise SafetyError(
                f"missing prior stage decision required for run_id continuity: "
                f"{decision_path}"
            )
        decision_run_id = str(decision.get("run_id", "") or "").strip()
        if decision_run_id != plan.run_id:
            raise SafetyError(
                f"prior stage run_id mismatch at Stage {stage_num}: "
                f"{decision_run_id!r} != {plan.run_id!r}; rollback/rerun must "
                "restart from the earliest mismatched stage"
            )

    try:
        from researchclaw.adapters import AdapterBundle
        from researchclaw.config import RCConfig
        from researchclaw.llm import create_llm_client
        from researchclaw.pipeline.runner import execute_pipeline
        from researchclaw.pipeline.stages import Stage, StageStatus
    except Exception as exc:
        raise SafetyError(f"cannot import ResearchClaw runtime: {exc}") from exc

    try:
        from_stage_enum = Stage(from_stage)
        to_stage_enum = Stage(to_stage)
    except ValueError as exc:
        raise SafetyError(f"invalid stage boundary: {exc}") from exc

    config_path = run_path / "config.yaml"
    try:
        config = RCConfig.load(config_path, check_paths=False)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise SafetyError(f"cannot load {config_path}: {exc}") from exc
    api_key_env = str(config.llm.api_key_env or "").strip()
    if api_key_env and not os.environ.get(api_key_env):
        os.environ[api_key_env] = "local-bridge"
    if not skip_preflight:
        client = create_llm_client(config)
        ok, message = client.preflight()
        if not ok:
            raise SafetyError(f"LLM preflight failed: {message}")

    previous_summary = _archive_phase_summary(run_path, from_stage, to_stage)
    results = execute_pipeline(
        run_dir=run_path,
        run_id=plan.run_id,
        config=config,
        adapters=AdapterBundle(),
        from_stage=from_stage_enum,
        to_stage=to_stage_enum,
        auto_approve_gates=True,
        stop_on_gate=False,
        skip_noncritical=False,
        kb_root=Path(config.knowledge_base.root)
        if config.knowledge_base.root
        else None,
    )
    failed = sum(result.status == StageStatus.FAILED for result in results)
    paused = sum(result.status == StageStatus.PAUSED for result in results)
    blocked = sum(
        result.status == StageStatus.BLOCKED_APPROVAL for result in results
    )
    final_stage = int(results[-1].stage) if results else None
    expected_stages = to_stage - from_stage + 1
    success = (
        bool(results)
        and final_stage == to_stage
        and len(results) == expected_stages
        and failed == 0
        and paused == 0
        and blocked == 0
    )
    summary_normalized = False
    if success and to_stage == 23:
        _rewrite_final_pipeline_summary(
            run_path,
            run_id=plan.run_id,
            from_stage=from_stage,
            to_stage=to_stage,
        )
        summary_normalized = True
    return {
        "run_dir": str(run_path),
        "run_id": plan.run_id,
        "from_stage": from_stage,
        "to_stage": to_stage,
        "previous_summary_archived": (
            str(previous_summary) if previous_summary is not None else None
        ),
        "final_stage": final_stage,
        "expected_stages": expected_stages,
        "stages_executed": len(results),
        "failed": failed,
        "paused": paused,
        "blocked": blocked,
        "summary_normalized": summary_normalized,
        "success": success,
    }


def _plan_dict(plan: RollbackPlan) -> dict[str, Any]:
    value = asdict(plan)
    value["processes"] = [asdict(item) for item in plan.processes]
    value["tasks"] = [asdict(item) for item in plan.tasks]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or apply a fail-closed rollback of an existing RSI cycle. "
            "Default is read-only; --apply never stops processes or remote tasks."
        )
    )
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--from-stage", type=int, default=9)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="optional destination below RUN/rollback_backups",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="archive artifacts and rewind checkpoint after all safety checks pass",
    )
    parser.add_argument(
        "--run-phase",
        action="store_true",
        help=(
            "run a bounded pipeline phase with execute_pipeline while preserving "
            "the checkpoint run_id"
        ),
    )
    parser.add_argument(
        "--to-stage",
        type=int,
        help="inclusive phase boundary required by --run-phase",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip LLM preflight for --run-phase",
    )
    parser.add_argument(
        "--allow-followup",
        action="store_true",
        help=(
            "allow a later bounded phase whose Stage 1-8 rollback prerequisites "
            "are no longer relevant; required when --from-stage is greater than 9"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        selected_actions = sum(
            bool(value)
            for value in (
                args.apply,
                args.run_phase,
            )
        )
        if selected_actions > 1:
            raise SafetyError(
                "--apply and --run-phase are mutually exclusive"
            )
        if args.run_phase:
            if args.to_stage is None:
                raise SafetyError("--run-phase requires --to-stage")
            if args.from_stage > 9 and not args.allow_followup:
                raise SafetyError(
                    "later bounded phases require --allow-followup; without it "
                    "the helper assumes an initial rollback boundary"
                )
            with _helper_lock(args.run):
                result = run_pipeline_phase(
                    campaign_dir=args.campaign,
                    run_dir=args.run,
                    from_stage=args.from_stage,
                    to_stage=args.to_stage,
                    skip_preflight=args.skip_preflight,
                    allow_followup=args.allow_followup,
                )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["success"] else 1
        if args.apply:
            with _helper_lock(args.run):
                plan = build_plan(
                    campaign_dir=args.campaign,
                    run_dir=args.run,
                    from_stage=args.from_stage,
                    backup_dir=args.backup_dir,
                )
                print(json.dumps(_plan_dict(plan), ensure_ascii=False, indent=2))
                backup_dir = apply_plan(plan)
        else:
            plan = build_plan(
                campaign_dir=args.campaign,
                run_dir=args.run,
                from_stage=args.from_stage,
                backup_dir=args.backup_dir,
            )
            print(json.dumps(_plan_dict(plan), ensure_ascii=False, indent=2))
            return 0 if plan.safe_to_apply else 2
        print(f"Rollback complete: {backup_dir}")
        return 0
    except SafetyError as exc:
        print(f"safe-cycle-rollback: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(
            f"safe-cycle-rollback: unexpected {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
