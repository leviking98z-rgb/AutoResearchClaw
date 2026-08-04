"""Per-Idea pipeline actor and isolated worker-process lifecycle."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import FactoryConfig
from .io import atomic_write_json
from .models import (
    Idea,
    IdeaStatus,
    ResourceRequest,
    WorkItem,
    WorkItemStatus,
    WorkKind,
    utc_now,
)

PROFILE_STAGES: dict[str, tuple[str, str]] = {
    "screen": ("TOPIC_INIT", "HYPOTHESIS_GEN"),
    "build": ("EXPERIMENT_DESIGN", "RESOURCE_PLANNING"),
    "pilot": ("EXPERIMENT_RUN", "RESEARCH_DECISION"),
    "validation": ("EXPERIMENT_RUN", "RESEARCH_DECISION"),
    "repair": ("ITERATIVE_REFINE", "RESEARCH_DECISION"),
    "paper": ("PAPER_OUTLINE", "CITATION_VERIFY"),
}


def profile_for_status(status: IdeaStatus) -> str:
    return {
        IdeaStatus.SCREENING: "screen",
        IdeaStatus.BUILDING: "build",
        IdeaStatus.SMOKE: "pilot",
        IdeaStatus.PILOT: "pilot",
        IdeaStatus.VALIDATING: "validation",
        IdeaStatus.REPAIR: "repair",
        IdeaStatus.PAPER: "paper",
    }[status]


def work_item_for_idea(idea: Idea) -> WorkItem:
    profile = profile_for_status(idea.status)
    attempt_limit = 3 if profile == "repair" else 2
    compute = idea.candidate.get("compute", {})
    requested_gpus = 0
    if profile in {"pilot", "validation", "repair"}:
        try:
            requested_gpus = max(1, int(compute.get("gpu_count", 1)))
        except (AttributeError, TypeError, ValueError):
            requested_gpus = 1
    max_gpus = requested_gpus
    timeout_sec = 3600.0
    try:
        timeout_sec = max(
            60.0, float(compute.get("wall_clock_hours", 1)) * 3600.0
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return WorkItem(
        item_id=f"{idea.idea_id}-{profile}",
        idea_id=idea.idea_id,
        kind=(
            WorkKind.GPU_EXPERIMENT
            if requested_gpus
            else (WorkKind.PAPER if profile == "paper" else WorkKind.PIPELINE)
        ),
        profile=profile,
        resources=ResourceRequest(
            min_gpus=1 if requested_gpus else 0,
            preferred_gpus=requested_gpus,
            max_gpus=max_gpus,
            cpus=8 if requested_gpus else 2,
            timeout_sec=timeout_sec,
            placement="single_node" if requested_gpus else "any",
            preemptible=profile in {"pilot", "validation"},
            checkpointable=profile in {"validation"},
        ),
        status=WorkItemStatus.READY,
        attempt_limit=attempt_limit,
        metadata={"priority": idea.priority},
    )


@dataclass(frozen=True, slots=True)
class WorkerProbe:
    state: str
    returncode: int | None = None
    pid: int | None = None
    started_at: str = ""
    finished_at: str = ""


class IdeaWorker(Protocol):
    def start(
        self,
        *,
        idea: Idea,
        item: WorkItem,
        idea_dir: Path,
    ) -> WorkerProbe: ...

    def probe(self, *, item: WorkItem, idea_dir: Path) -> WorkerProbe: ...

    def cancel(self, *, item: WorkItem, idea_dir: Path) -> WorkerProbe: ...


def _pid_matches(pid: int, start_ticks: int | None) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    if start_ticks is None or not sys.platform.startswith("linux"):
        return True
    try:
        observed = int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        return False
    return observed == start_ticks


def _start_ticks(pid: int) -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().split()[21])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        return None


class PipelineIdeaWorker:
    """Launch one bounded profile through the existing ResearchClaw CLI."""

    def __init__(self, config: FactoryConfig) -> None:
        self.config = config

    @staticmethod
    def _worker_dir(idea_dir: Path, item: WorkItem) -> Path:
        return idea_dir / "workers" / item.item_id / f"attempt-{item.attempt + 1:02d}"

    def _command(
        self,
        *,
        idea: Idea,
        item: WorkItem,
        idea_dir: Path,
    ) -> list[str]:
        if not self.config.worker.pipeline_config:
            raise ValueError("factory.worker.pipeline_config is required")
        from_stage, to_stage = PROFILE_STAGES[item.profile]
        run_dir = idea_dir / "runs" / "pipeline"
        if from_stage != "TOPIC_INIT" and not (
            run_dir / "checkpoint.json"
        ).exists():
            raise FileNotFoundError(
                f"{item.profile} requires prior pipeline checkpoint: {run_dir}"
            )
        argv = [
            self.config.worker.python or sys.executable,
            "-m",
            "researchclaw",
            "run",
            "--config",
            self.config.worker.pipeline_config,
            "--topic",
            idea.title,
            "--output",
            str(run_dir),
            "--from-stage",
            from_stage,
            "--to-stage",
            to_stage,
        ]
        if self.config.worker.skip_preflight:
            argv.append("--skip-preflight")
        if self.config.worker.auto_approve:
            argv.append("--auto-approve")
        return argv

    def start(
        self,
        *,
        idea: Idea,
        item: WorkItem,
        idea_dir: Path,
    ) -> WorkerProbe:
        worker_dir = self._worker_dir(idea_dir, item)
        worker_dir.mkdir(parents=True, exist_ok=True)
        state_path = worker_dir / "state.json"
        if state_path.exists():
            existing = self.probe(item=item, idea_dir=idea_dir)
            if existing.state in {"running", "finished"}:
                return existing

        stdout = (worker_dir / "stdout.log").open("ab", buffering=0)
        stderr = (worker_dir / "stderr.log").open("ab", buffering=0)
        argv = self._command(idea=idea, item=item, idea_dir=idea_dir)
        child_env = os.environ.copy()
        child_env.update(
            {
                "RESEARCHCLAW_IDEA_ID": idea.idea_id,
                "RESEARCHCLAW_WORK_ITEM_ID": item.item_id,
                "RESEARCHCLAW_GPU_REQUEST": str(
                    int(item.metadata.get("allocated_gpus", 0) or 0)
                ),
            }
        )
        process = subprocess.Popen(
            argv,
            cwd=Path(__file__).resolve().parents[2],
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env=child_env,
        )
        stdout.close()
        stderr.close()
        state = {
            "state": "running",
            "pid": process.pid,
            "start_ticks": _start_ticks(process.pid),
            "started_at": utc_now(),
            "argv": argv,
        }
        atomic_write_json(state_path, state)
        return WorkerProbe(
            state="running",
            pid=process.pid,
            started_at=state["started_at"],
        )

    def probe(self, *, item: WorkItem, idea_dir: Path) -> WorkerProbe:
        worker_dir = self._worker_dir(idea_dir, item)
        state_path = worker_dir / "state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return WorkerProbe("missing")
        if state.get("state") == "finished":
            return WorkerProbe(
                "finished",
                returncode=int(state.get("returncode", -1)),
                pid=int(state.get("pid", 0)) or None,
                started_at=str(state.get("started_at", "")),
                finished_at=str(state.get("finished_at", "")),
            )
        pid = int(state.get("pid", 0) or 0)
        ticks = state.get("start_ticks")
        start_ticks = int(ticks) if isinstance(ticks, int) else None
        if _pid_matches(pid, start_ticks):
            return WorkerProbe(
                "running",
                pid=pid,
                started_at=str(state.get("started_at", "")),
            )

        # A reaped child no longer exposes its return code to a restarted
        # Factory.  The CLI writes pipeline_summary.json; use it as durable
        # completion evidence and otherwise fail closed.
        summary_path = idea_dir / "runs" / "pipeline" / "pipeline_summary.json"
        returncode = -1
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            summary = {}
        final = str(summary.get("final_status", "")).casefold()
        if final in {"done", "completed", "success", "succeeded"}:
            returncode = 0
        state.update(
            {
                "state": "finished",
                "returncode": returncode,
                "finished_at": utc_now(),
            }
        )
        atomic_write_json(state_path, state)
        return WorkerProbe(
            "finished",
            returncode=returncode,
            pid=pid or None,
            started_at=str(state.get("started_at", "")),
            finished_at=str(state["finished_at"]),
        )

    def cancel(self, *, item: WorkItem, idea_dir: Path) -> WorkerProbe:
        probe = self.probe(item=item, idea_dir=idea_dir)
        if probe.state != "running" or not probe.pid:
            return probe
        try:
            os.killpg(probe.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + self.config.worker.graceful_shutdown_sec
        while time.monotonic() < deadline:
            if not _pid_matches(probe.pid, None):
                break
            time.sleep(0.1)
        if _pid_matches(probe.pid, None):
            try:
                os.killpg(probe.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        worker_dir = self._worker_dir(idea_dir, item)
        state_path = worker_dir / "state.json"
        state = {
            "state": "finished",
            "pid": probe.pid,
            "returncode": 130,
            "started_at": probe.started_at,
            "finished_at": utc_now(),
        }
        atomic_write_json(state_path, state)
        return WorkerProbe(
            "finished",
            returncode=130,
            pid=probe.pid,
            started_at=probe.started_at,
            finished_at=state["finished_at"],
        )


class SimulatedIdeaWorker:
    """Fast deterministic worker used for orchestration and recovery tests."""

    def __init__(self, *, delay_sec: float = 0.0, fail_profiles: set[str] | None = None):
        self.delay_sec = delay_sec
        self.fail_profiles = set(fail_profiles or ())
        self.started: dict[str, float] = {}
        self.cancelled: set[str] = set()

    def start(
        self,
        *,
        idea: Idea,
        item: WorkItem,
        idea_dir: Path,
    ) -> WorkerProbe:
        del idea, idea_dir
        self.started.setdefault(item.item_id, time.monotonic())
        return WorkerProbe("running", pid=None, started_at=utc_now())

    def probe(self, *, item: WorkItem, idea_dir: Path) -> WorkerProbe:
        if item.item_id in self.cancelled:
            return WorkerProbe("finished", returncode=130)
        started = self.started.get(item.item_id)
        if started is None:
            return WorkerProbe("missing")
        if time.monotonic() - started < self.delay_sec:
            return WorkerProbe("running")
        if item.profile not in self.fail_profiles:
            evidence_dir = idea_dir / "evidence"
            evidence_dir.mkdir(parents=True, exist_ok=True)
            reason_by_profile = {
                "screen": "SCREEN_COMPLETE",
                "build": "BUILD_COMPLETE",
                "paper": "PAPER_COMPLETE",
            }
            reason = reason_by_profile.get(item.profile)
            if reason:
                atomic_write_json(
                    evidence_dir / f"simulated-{reason}.json",
                    {"profile": item.profile, "simulated": True},
                )
            if item.profile in {"pilot", "validation", "repair"}:
                summary_path = (
                    idea_dir
                    / "runs"
                    / "pipeline"
                    / "stage-14"
                    / "experiment_summary.json"
                )
                atomic_write_json(
                    summary_path,
                    {
                        "result_valid": True,
                        "success_probability": 0.99,
                        "primary_effect_size": 0.10,
                        "successful_seed_count": 3,
                        "best_run": {
                            "status": "completed",
                            "returncode": 0,
                            "result_valid": True,
                            "metrics": {
                                "baseline/seed-0/accuracy": 0.5,
                                "method/seed-0/accuracy": 0.6,
                                "method/successful_seed_count": 3,
                            },
                        },
                        "condition_summaries": {
                            "baseline": {
                                "metrics": {"accuracy": 0.5},
                                "successful_seed_count": 3,
                            },
                            "method": {
                                "metrics": {"accuracy": 0.6},
                                "successful_seed_count": 3,
                            },
                        },
                    },
                )
        return WorkerProbe(
            "finished",
            returncode=1 if item.profile in self.fail_profiles else 0,
            finished_at=utc_now(),
        )

    def cancel(self, *, item: WorkItem, idea_dir: Path) -> WorkerProbe:
        del idea_dir
        self.cancelled.add(item.item_id)
        return WorkerProbe("finished", returncode=130, finished_at=utc_now())
