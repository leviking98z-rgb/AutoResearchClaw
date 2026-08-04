"""Per-Idea pipeline actor and isolated worker-process lifecycle."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from researchclaw.observability import OperationalEventLogger

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


_TERMINAL_PROC_STATES = frozenset({"Z", "X", "x"})


def _proc_state_and_start_ticks(pid: int) -> tuple[str, int] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        _, separator, trailing_fields = stat.rpartition(")")
        if not separator:
            return None
        fields = trailing_fields.split()
        return fields[0], int(fields[19])
    except (FileNotFoundError, OSError, IndexError, ValueError):
        return None


def _pid_matches(pid: int, start_ticks: int | None) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    if not sys.platform.startswith("linux"):
        return True
    observed = _proc_state_and_start_ticks(pid)
    if observed is None:
        return False
    state, observed_start_ticks = observed
    if state in _TERMINAL_PROC_STATES:
        return False
    return start_ticks is None or observed_start_ticks == start_ticks


def _start_ticks(pid: int) -> int | None:
    if not sys.platform.startswith("linux"):
        return None
    observed = _proc_state_and_start_ticks(pid)
    return observed[1] if observed is not None else None


class PipelineIdeaWorker:
    """Launch one bounded profile through the existing ResearchClaw CLI."""

    def __init__(self, config: FactoryConfig) -> None:
        self.config = config

    @staticmethod
    def _worker_dir(idea_dir: Path, item: WorkItem) -> Path:
        attempt = max(1, int(item.attempt))
        return idea_dir / "workers" / item.item_id / f"attempt-{attempt:02d}"

    @staticmethod
    def _selected_topic(idea: Idea) -> dict[str, object]:
        """Build the authoritative Stage-1/Stage-9 contract for one Idea."""

        candidate = dict(idea.candidate)
        candidate.update(
            {
                "id": str(candidate.get("id") or idea.idea_id),
                "idea_id": idea.idea_id,
                "title": idea.title,
                "research_question": idea.research_question,
                "falsifiable_hypothesis": idea.falsifiable_hypothesis,
                "primary_metric": idea.primary_metric,
                "family": idea.family,
                "source": idea.source,
                "parent_ids": list(idea.parent_ids),
            }
        )
        return candidate

    def _prepare_idea_config(
        self,
        *,
        idea: Idea,
        item: WorkItem,
        idea_dir: Path,
    ) -> tuple[Path, Path, Path]:
        """Materialize a config that cannot inherit another Idea's topic."""

        if not self.config.worker.pipeline_config:
            raise ValueError("factory.worker.pipeline_config is required")
        base_config = Path(
            self.config.worker.pipeline_config
        ).expanduser().resolve()
        raw = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"pipeline config root must be a mapping: {base_config}")
        data = {str(key): value for key, value in raw.items()}

        contract_dir = idea_dir / "contract"
        contract_dir.mkdir(parents=True, exist_ok=True)
        attempt = max(1, int(item.attempt))
        attempt_dir = (
            contract_dir / item.item_id / f"attempt-{attempt:02d}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        selected_topic_path = attempt_dir / "selected_topic.json"
        atomic_write_json(selected_topic_path, self._selected_topic(idea))
        # Stable convenience pointer for operators and downstream tooling. The
        # worker itself always receives the immutable attempt-scoped path.
        atomic_write_json(
            contract_dir / "selected_topic.json",
            self._selected_topic(idea),
        )

        research_raw = data.get("research")
        research = (
            {str(key): value for key, value in research_raw.items()}
            if isinstance(research_raw, Mapping)
            else {}
        )
        research.update(
            {
                "topic": idea.title,
                "selected_topic_file": str(selected_topic_path),
                "autonomous_topic_selection": False,
            }
        )
        data["research"] = research

        experiment_raw = data.get("experiment")
        experiment = (
            {str(key): value for key, value in experiment_raw.items()}
            if isinstance(experiment_raw, Mapping)
            else {}
        )
        try:
            requested_timeout = max(
                60,
                int(float(item.resources.timeout_sec)),
            )
        except (TypeError, ValueError):
            requested_timeout = 3600
        existing_timeout = experiment.get("time_budget_sec")
        try:
            configured_timeout = max(1, int(existing_timeout))
        except (TypeError, ValueError):
            configured_timeout = requested_timeout
        experiment["time_budget_sec"] = min(
            configured_timeout,
            requested_timeout,
        )
        data["experiment"] = experiment

        config_path = attempt_dir / "pipeline.yaml"
        config_path.write_text(
            yaml.safe_dump(
                data,
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (contract_dir / "pipeline.yaml").write_text(
            config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return config_path, selected_topic_path, attempt_dir

    def _command(
        self,
        *,
        idea: Idea,
        item: WorkItem,
        idea_dir: Path,
    ) -> list[str]:
        from_stage, to_stage = PROFILE_STAGES[item.profile]
        run_dir = idea_dir / "runs" / "pipeline"
        if from_stage != "TOPIC_INIT" and not (
            run_dir / "checkpoint.json"
        ).exists():
            raise FileNotFoundError(
                f"{item.profile} requires prior pipeline checkpoint: {run_dir}"
            )
        config_path, selected_topic_path, contract_attempt_dir = (
            self._prepare_idea_config(
                idea=idea,
                item=item,
                idea_dir=idea_dir,
            )
        )
        item.metadata.update(
            {
                "pipeline_config": str(config_path),
                "selected_topic_file": str(selected_topic_path),
                "contract_attempt_dir": str(contract_attempt_dir),
            }
        )
        argv = [
            self.config.worker.python or sys.executable,
            "-m",
            "researchclaw",
            "run",
            "--config",
            str(config_path),
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

        argv = self._command(idea=idea, item=item, idea_dir=idea_dir)
        operational_log_path = idea_dir / "operational_events.jsonl"
        operational_log = OperationalEventLogger(
            operational_log_path,
            component="factory.worker",
            context={
                "factory_id": self.config.factory_id,
                "idea_id": idea.idea_id,
                "work_item_id": item.item_id,
                "attempt": item.attempt,
                "profile": item.profile,
            },
        )
        operational_log.emit(
            "worker_launch_requested",
            argv=[str(part) for part in argv],
            worker_dir=str(worker_dir),
            allocated_gpus=int(item.metadata.get("allocated_gpus", 0) or 0),
        )
        stdout = (worker_dir / "stdout.log").open("ab", buffering=0)
        stderr = (worker_dir / "stderr.log").open("ab", buffering=0)
        child_env = os.environ.copy()
        child_env.update(
            {
                "RESEARCHCLAW_FACTORY_ID": self.config.factory_id,
                "RESEARCHCLAW_IDEA_ID": idea.idea_id,
                "RESEARCHCLAW_WORK_ITEM_ID": item.item_id,
                "RESEARCHCLAW_WORK_ITEM_ATTEMPT": str(item.attempt),
                "RESEARCHCLAW_GPU_REQUEST": str(
                    int(item.metadata.get("allocated_gpus", 0) or 0)
                ),
                "RESEARCHCLAW_OPERATIONAL_LOG": str(operational_log_path),
            }
        )
        try:
            process = subprocess.Popen(
                argv,
                cwd=Path(__file__).resolve().parents[2],
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=child_env,
            )
        finally:
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
        operational_log.emit(
            "worker_launched",
            outcome="running",
            pid=process.pid,
            start_ticks=state["start_ticks"],
        )
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
        OperationalEventLogger(
            idea_dir / "operational_events.jsonl",
            component="factory.worker",
            context={
                "idea_id": item.idea_id,
                "work_item_id": item.item_id,
                "attempt": item.attempt,
                "profile": item.profile,
            },
        ).emit(
            "worker_finished",
            level="INFO" if returncode == 0 else "ERROR",
            outcome="succeeded" if returncode == 0 else "failed",
            reason_code="" if returncode == 0 else f"WORKER_EXIT_{returncode}",
            pid=pid or None,
            returncode=returncode,
            started_at=str(state.get("started_at", "")),
            finished_at=str(state["finished_at"]),
        )
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
        OperationalEventLogger(
            idea_dir / "operational_events.jsonl",
            component="factory.worker",
            context={
                "idea_id": item.idea_id,
                "work_item_id": item.item_id,
                "attempt": item.attempt,
                "profile": item.profile,
            },
        ).emit(
            "worker_cancelled",
            level="WARNING",
            outcome="cancelled",
            reason_code="OPERATOR_OR_FACTORY_STOP",
            pid=probe.pid,
            returncode=130,
        )
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
