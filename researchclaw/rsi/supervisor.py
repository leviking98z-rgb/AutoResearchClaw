"""Persistent campaign-level RSI supervisor."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from researchclaw.runtime_dependencies import (
    dependencies_satisfied,
    ensure_runtime_dependencies,
)

from .configuration import load_yaml_mapping, prepare_cycle_config
from .diagnosis import (
    apply_diagnosis,
    apply_failure_repair,
    build_llm_client,
    diagnose_cycle,
    run_campaign_aevolve,
)
from .evidence import collect_evidence, evidence_succeeded
from .storage import (
    CampaignStore,
    atomic_write_json,
    atomic_write_text,
    cleanup_atomic_temp_files,
    read_json,
    utc_now,
)
from .topic_selection import (
    normalize_topic_action,
    persist_topic_selection,
    select_topics,
    selection_from_artifacts,
    selection_with_candidate,
)

DEFAULT_CAMPAIGN_ROOT = Path(
    os.environ.get(
        "RESEARCHCLAW_RSI_ROOT",
        "/root/shared/.clusters/.workdir/autoresearch-rsi",
    )
).expanduser()

RUN_POLICY_SCHEMA_VERSION = 1


def _slug(text: str, max_length: int = 36) -> str:
    import re

    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:max_length].strip("-") or "research"


def new_campaign_id(topic: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_slug(topic)}-{uuid.uuid4().hex[:6]}"


def resolve_campaign(
    campaign: str | Path,
    *,
    campaign_root: Path = DEFAULT_CAMPAIGN_ROOT,
) -> Path:
    """Resolve either an explicit campaign directory or an ID below *root*."""

    candidate = Path(campaign).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.resolve()
    return (campaign_root.expanduser() / candidate).resolve()


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _process_identity_matches(pid: int, start_ticks: int | None) -> bool:
    if pid <= 0:
        return False
    observed = _process_start_ticks(pid)
    if observed is None:
        return False
    return start_ticks is not None and observed == start_ticks


@dataclass(frozen=True)
class SupervisorOptions:
    campaign_dir: Path
    repo_root: Path
    base_config: Path
    topic: str
    max_cycles: int = 20
    continuous: bool = False
    single_cycle: bool = False
    dry_run: bool = False
    max_consecutive_failures: int = 3
    max_no_improvement_cycles: int = 5
    backoff_initial_sec: float = 30.0
    backoff_max_sec: float = 900.0
    heartbeat_interval_sec: float = 15.0
    control_poll_sec: float = 1.0
    model: str = "codebuddy/claude-sonnet-5"
    bridge_url: str = "http://127.0.0.1:8787/v1"
    api_key_env: str = "BRIDGE_LOCAL_API_KEY"
    llm_timeout_sec: int = 1800
    skip_preflight: bool = False
    no_aevolve: bool = False
    resume_existing: bool = False
    pipeline_extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


def run_policy_from_options(options: SupervisorOptions) -> dict[str, Any]:
    """Return the durable supervisor policy used by submit/resume/monitor."""

    return {
        "schema_version": RUN_POLICY_SCHEMA_VERSION,
        "continuous": bool(options.continuous),
        "max_cycles": int(options.max_cycles),
        "single_cycle": bool(options.single_cycle),
        "dry_run": bool(options.dry_run),
        "max_consecutive_failures": int(options.max_consecutive_failures),
        "max_no_improvement_cycles": int(options.max_no_improvement_cycles),
        "backoff_initial_sec": float(options.backoff_initial_sec),
        "backoff_max_sec": float(options.backoff_max_sec),
        "heartbeat_interval_sec": float(options.heartbeat_interval_sec),
        "control_poll_sec": float(options.control_poll_sec),
        "model": str(options.model),
        "bridge_url": str(options.bridge_url),
        "api_key_env": str(options.api_key_env),
        "llm_timeout_sec": int(options.llm_timeout_sec),
        "skip_preflight": bool(options.skip_preflight),
        "no_aevolve": bool(options.no_aevolve),
        "pipeline_extra_args": [str(value) for value in options.pipeline_extra_args],
        "automatic_submission_enabled": False,
    }


class _Heartbeat:
    def __init__(
        self,
        supervisor: CampaignSupervisor,
        *,
        phase: str,
        cycle: int,
        run_dir: Path | None = None,
        child_pid: Callable[[], int | None] | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.phase = phase
        self.cycle = cycle
        self.run_dir = run_dir
        self.child_pid = child_pid or (lambda: None)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self.tick()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stop_event.wait(
            max(0.1, self.supervisor.options.heartbeat_interval_sec)
        ):
            self.tick()

    def tick(self) -> None:
        self.supervisor.store.write_heartbeat(
            {
                "campaign_id": self.supervisor.campaign_id,
                "supervisor_pid": os.getpid(),
                "child_pid": self.child_pid(),
                "cycle": self.cycle,
                "phase": self.phase,
                "run_dir": str(self.run_dir) if self.run_dir else None,
                "status": self.supervisor.state.get("status"),
            }
        )

    def __exit__(self, *_: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.tick()


class CampaignSupervisor:
    """Run, diagnose and evolve ResearchClaw cycles until paused or stopped."""

    def __init__(
        self,
        options: SupervisorOptions,
        *,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        llm_factory: Callable[[Path], Any] = build_llm_client,
        diagnosis_fn: Callable[..., dict[str, Any]] = diagnose_cycle,
        aevolve_fn: Callable[..., list[str]] = run_campaign_aevolve,
    ) -> None:
        self.options = options
        self.store = CampaignStore(options.campaign_dir)
        self._popen_factory = popen_factory
        self._sleep = sleep
        self._llm_factory = llm_factory
        self._diagnosis_fn = diagnosis_fn
        self._aevolve_fn = aevolve_fn
        self._child: subprocess.Popen[str] | None = None
        self._signal: int | None = None
        self._cancel_event = threading.Event()
        self._campaign_lock: Any | None = None
        self.state: dict[str, Any] = {}
        self.campaign_id = options.campaign_dir.name

    def initialize(self) -> None:
        if not self.store.state_path.exists():
            self.store.initialize()
        if self.store.state_path.exists():
            self.state = self.store.load_state()
            self.campaign_id = str(
                self.state.get("campaign_id", self.options.campaign_dir.name)
            )
            if not self.store.policy_path.exists():
                atomic_write_json(
                    self.store.policy_path,
                    run_policy_from_options(self.options),
                )
            return

        self.campaign_id = self.options.campaign_dir.name
        atomic_write_text(
            self.store.shared_brief_path,
            self.options.topic.strip() + "\n",
        )
        atomic_write_text(
            self.store.shared_prompt_path,
            "# Campaign RSI Guidance\n\n"
            "Treat the campaign brief as authoritative. Preserve reproducibility, "
            "ground every numerical claim in experiment artifacts, and never "
            "submit or publish automatically.\n",
        )
        manifest = {
            "schema_version": 1,
            "campaign_id": self.campaign_id,
            "created_at": utc_now(),
            "repo_root": str(self.options.repo_root),
            "base_config": str(self.options.base_config),
            "topic": self.options.topic,
            "model": self.options.model,
            "bridge_url": self.options.bridge_url,
            "run_policy": run_policy_from_options(self.options),
            "automatic_submission_enabled": False,
        }
        atomic_write_json(self.store.manifest_path, manifest)
        atomic_write_json(
            self.store.policy_path,
            run_policy_from_options(self.options),
        )
        self.state = self.store.save_state(
            {
                "schema_version": 1,
                "campaign_id": self.campaign_id,
                "created_at": manifest["created_at"],
                "status": "created",
                "phase": "idle",
                "topic": self.options.topic,
                "continuous": self.options.continuous,
                "max_cycles": self.options.max_cycles,
                "next_cycle": 1,
                "active_cycle": None,
                "active_run_dir": None,
                "completed_cycles": 0,
                "successful_cycles": 0,
                "failed_cycles": 0,
                "consecutive_failures": 0,
                "failure_signature": None,
                "failure_signature_details": None,
                "consecutive_same_failure": 0,
                "failure_recovery_action": None,
                "consecutive_no_improvement": 0,
                "best_score": None,
                "best_cycle": None,
                "best_run_dir": None,
                "best_evidence_path": None,
                "best_by_topic": {},
                "last_error": None,
                "last_diagnosis": None,
                "pending_topic_action": {
                    "topic_action": "keep",
                    "topic_patch": "",
                    "pivot_reason": "",
                    "preferred_candidate_id": "",
                },
                "last_mutations": [],
                "pid": None,
                "supervisor_start_ticks": None,
                "active_child_pid": None,
                "active_child_start_ticks": None,
                "automatic_submission_enabled": False,
            }
        )
        self.store.log.append(
            "campaign_created",
            campaign_id=self.campaign_id,
            topic=self.options.topic,
            base_config=str(self.options.base_config),
            dry_run=self.options.dry_run,
        )

    def run(self) -> int:
        self.store.root.mkdir(parents=True, exist_ok=True)
        self._acquire_campaign_lock()
        try:
            return self._run_locked()
        finally:
            self._release_campaign_lock()

    def _run_locked(self) -> int:
        self.initialize()
        # The local Bridge accepts a placeholder bearer token, but the
        # OpenAI-compatible client still requires its configured environment
        # variable to exist.  Background submission already sets this; mirror
        # that behavior for foreground runs and direct API use.
        if not os.environ.get(self.options.api_key_env):
            os.environ[self.options.api_key_env] = "local-bridge"
        if self.options.resume_existing:
            self.store.clear_control("pause")
            self.store.clear_control("stop")
        current_start_ticks = _process_start_ticks(os.getpid())
        self._reconcile_interrupted_execution(
            current_pid=os.getpid(),
            current_start_ticks=current_start_ticks,
        )
        if self.store.control_requested("stop"):
            self._transition(
                status="stopped",
                phase="idle",
                pid=None,
                supervisor_start_ticks=None,
            )
            return 0
        if self.store.control_requested("pause"):
            self._transition(
                status="paused",
                phase="idle",
                pid=None,
                supervisor_start_ticks=None,
            )
            return 0

        self._install_signal_handlers()
        self._transition(
            status="running",
            phase="idle",
            pid=os.getpid(),
            supervisor_start_ticks=current_start_ticks,
        )
        self.store.log.append(
            "supervisor_started",
            campaign_id=self.campaign_id,
            pid=os.getpid(),
            next_cycle=int(self.state.get("next_cycle", 1)),
        )
        removed_temps = cleanup_atomic_temp_files(self.store.root)
        if removed_temps:
            self.store.log.append(
                "stale_atomic_temps_removed",
                campaign_id=self.campaign_id,
                paths=[
                    str(path.relative_to(self.store.root))
                    for path in removed_temps
                ],
            )
        try:
            while True:
                action = self._control_action()
                if action:
                    return self._finish_for_control(action)

                cycle = int(self.state.get("next_cycle", 1))
                if (
                    not self.options.continuous
                    and self.options.max_cycles
                    and cycle > self.options.max_cycles
                ):
                    self._transition(
                        status="completed",
                        phase="idle",
                        pid=None,
                        supervisor_start_ticks=None,
                    )
                    self.store.log.append(
                        "campaign_completed",
                        campaign_id=self.campaign_id,
                        reason="max_cycles",
                        completed_cycles=self.state.get("completed_cycles", 0),
                    )
                    return 0

                outcome = self._run_cycle(cycle)
                if outcome == "stopped":
                    return self._finish_for_control("stop")
                if outcome == "paused":
                    return self._finish_for_control("pause")

                if self.options.single_cycle:
                    status = (
                        "paused_failure"
                        if not bool(outcome)
                        else "paused_single_cycle"
                    )
                    self._transition(
                        status=status,
                        phase="idle",
                        pid=None,
                        supervisor_start_ticks=None,
                    )
                    self.store.log.append(
                        "single_cycle_complete",
                        campaign_id=self.campaign_id,
                        cycle=cycle,
                        success=bool(outcome),
                    )
                    return 0 if outcome else 1

                same_failure_count = int(
                    self.state.get("consecutive_same_failure", 0)
                )
                same_failure_threshold = max(
                    1, self.options.max_consecutive_failures
                )
                generic_failure_threshold = (
                    int(self.state.get("consecutive_failures", 0))
                    >= same_failure_threshold
                )
                if (
                    same_failure_count >= same_failure_threshold
                    or generic_failure_threshold
                ):
                    threshold_kind = (
                        "same_failure_signature"
                        if same_failure_count >= same_failure_threshold
                        else "consecutive_failures"
                    )
                    self._transition(
                        status="paused_failure_threshold",
                        phase="idle",
                        pid=None,
                        supervisor_start_ticks=None,
                    )
                    self.store.set_control(
                        "pause",
                        (
                            "automatic pause after "
                            f"{same_failure_count} repeated failures with the "
                            "same signature"
                            if threshold_kind == "same_failure_signature"
                            else (
                                "automatic pause after "
                                f"{self.state['consecutive_failures']} "
                                "consecutive failures"
                            )
                        ),
                    )
                    self.store.log.append(
                        "failure_threshold_reached",
                        campaign_id=self.campaign_id,
                        cycle=cycle,
                        consecutive_failures=self.state["consecutive_failures"],
                        consecutive_same_failure=same_failure_count,
                        failure_signature=self.state.get("failure_signature"),
                        threshold_kind=threshold_kind,
                    )
                    return 1

                if (
                    not self.options.continuous
                    and int(self.state.get("consecutive_no_improvement", 0))
                    >= max(1, self.options.max_no_improvement_cycles)
                ):
                    self._transition(
                        status="paused_no_improvement",
                        phase="idle",
                        pid=None,
                        supervisor_start_ticks=None,
                    )
                    self.store.set_control(
                        "pause",
                        (
                            "automatic pause after "
                            f"{self.state['consecutive_no_improvement']} "
                            "cycles without accepted improvement"
                        ),
                    )
                    self.store.log.append(
                        "no_improvement_threshold_reached",
                        campaign_id=self.campaign_id,
                        cycle=cycle,
                        consecutive_no_improvement=self.state[
                            "consecutive_no_improvement"
                        ],
                    )
                    return 1

                if not outcome:
                    delay = self._backoff_delay(
                        int(self.state.get("consecutive_failures", 1))
                    )
                    control = self._interruptible_backoff(delay, cycle)
                    if control:
                        return self._finish_for_control(control)
        except BaseException as exc:
            self._transition(
                status="crashed",
                phase="idle",
                pid=None,
                supervisor_start_ticks=None,
                last_error=f"{type(exc).__name__}: {exc}",
            )
            self.store.log.append(
                "supervisor_crashed",
                campaign_id=self.campaign_id,
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=30),
            )
            raise
        finally:
            self.store.write_heartbeat(
                {
                    "campaign_id": self.campaign_id,
                    "supervisor_pid": os.getpid(),
                    "child_pid": None,
                    "cycle": self.state.get("active_cycle"),
                    "phase": "exited",
                    "status": self.state.get("status"),
                }
            )

    def _acquire_campaign_lock(self) -> None:
        lock_path = self.store.root / "supervisor.lock"
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            handle.close()
            raise RuntimeError(
                "campaign supervisor lock is already held"
                + (f": {owner}" if owner else "")
            ) from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "pid": os.getpid(),
                "campaign_id": self.campaign_id,
                "instance_id": uuid.uuid4().hex,
                "acquired_at": utc_now(),
                "process_start_ticks": _process_start_ticks(os.getpid()),
            },
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._campaign_lock = handle

    def _release_campaign_lock(self) -> None:
        handle = self._campaign_lock
        self._campaign_lock = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _run_cycle(self, cycle: int) -> bool | str:
        run_dir = self.store.runs_dir / f"cycle-{cycle:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        resumed_from_cycle = self._seed_failed_cycle_resume(run_dir, cycle)
        self._seed_run_memory(run_dir)
        campaign_brief = self._current_brief()
        effective_topic = campaign_brief
        selected_topic_path: Path | None = None
        selected_topic_id = ""
        topic_action = self._pending_topic_action()
        autonomous_selection = self._autonomous_topic_selection_enabled()
        if autonomous_selection and not self.options.dry_run:
            self._transition(
                status="running",
                phase="topic_selection",
                active_cycle=cycle,
                active_run_dir=str(run_dir),
                active_child_pid=None,
                active_child_start_ticks=None,
                last_error=None,
            )
            try:
                with _Heartbeat(
                    self,
                    phase="topic_selection",
                    cycle=cycle,
                    run_dir=run_dir,
                ):
                    try:
                        selection = selection_from_artifacts(
                            candidates_path=run_dir / "topic_candidates.json",
                            selected_path=run_dir / "selected_topic.json",
                        )
                        selection_source = "resumed_cycle_artifact"
                    except (
                        FileNotFoundError,
                        json.JSONDecodeError,
                        TypeError,
                        ValueError,
                        OSError,
                    ):
                        try:
                            incumbent = selection_from_artifacts(
                                candidates_path=(
                                    self.store.shared_topic_candidates_path
                                ),
                                selected_path=(
                                    self.store.shared_selected_topic_path
                                ),
                            )
                        except (
                            FileNotFoundError,
                            json.JSONDecodeError,
                            TypeError,
                            ValueError,
                            OSError,
                        ):
                            incumbent = None
                        if incumbent is not None and topic_action[
                            "topic_action"
                        ] in {"keep", "refine"}:
                            selection = incumbent
                            selection_source = (
                                "incumbent_refined"
                                if topic_action["topic_action"] == "refine"
                                else "incumbent"
                            )
                        elif (
                            incumbent is not None
                            and topic_action["topic_action"] == "pivot"
                            and topic_action["preferred_candidate_id"]
                        ):
                            selection = selection_with_candidate(
                                incumbent,
                                candidate_id=topic_action[
                                    "preferred_candidate_id"
                                ],
                                rationale=topic_action["pivot_reason"],
                            )
                            selection_source = "diagnosis_preferred_candidate"
                        else:
                            selector_config = prepare_cycle_config(
                                base_config=self.options.base_config,
                                output_path=run_dir / "topic_selector_config.yaml",
                                store=self.store,
                                topic=campaign_brief,
                                campaign_brief=campaign_brief,
                                autonomous_topic_selection=True,
                                model=self.options.model,
                                bridge_url=self.options.bridge_url,
                                api_key_env=self.options.api_key_env,
                                timeout_sec=self.options.llm_timeout_sec,
                                cycle=cycle,
                            )
                            try:
                                selector_llm = self._llm_factory(
                                    selector_config,
                                    role="topic_selector",
                                )
                            except TypeError:
                                # Backward-compatible test/custom factories may
                                # still accept only the config path.
                                selector_llm = self._llm_factory(selector_config)
                            previous_selection = read_json(
                                self.store.shared_selected_topic_path,
                                None,
                            )
                            selection = select_topics(
                                llm=selector_llm,
                                brief=campaign_brief,
                                cycle=cycle,
                                previous_selection=(
                                    previous_selection
                                    if isinstance(previous_selection, dict)
                                    else None
                                ),
                            )
                            selection_source = (
                                "llm_pivot"
                                if topic_action["topic_action"] == "pivot"
                                else "llm_initial"
                            )
                    persist_topic_selection(
                        shared_dir=self.store.shared_dir,
                        run_dir=run_dir,
                        selection=selection,
                        cycle=cycle,
                    )
                    effective_topic = selection.effective_topic
                    selected_topic_id = str(selection.selected["id"])
                    if topic_action["topic_action"] == "pivot":
                        # A refinement belongs to its incumbent topic. Never
                        # inject it into a newly selected research question.
                        self.store.shared_topic_patch_path.unlink(
                            missing_ok=True
                        )
                    selected_topic_path = run_dir / "selected_topic.json"
                    self.store.log.append(
                        "topic_selected",
                        campaign_id=self.campaign_id,
                        cycle=cycle,
                        selected_candidate_id=selection.selected["id"],
                        selected_topic=effective_topic,
                        candidate_count=len(selection.document["candidates"]),
                        selected_score=selection.selected["weighted_score"],
                        source=selection_source,
                        topic_action=topic_action["topic_action"],
                        pivot_reason=topic_action["pivot_reason"],
                        artifacts=[
                            str(run_dir / "topic_candidates.json"),
                            str(run_dir / "selected_topic.json"),
                            str(run_dir / "topic_selection.md"),
                        ],
                    )
                    self._transition(
                        selected_topic=effective_topic,
                        selected_topic_id=selection.selected["id"],
                        selected_topic_path=str(selected_topic_path),
                        topic_candidate_count=len(
                            selection.document["candidates"]
                        ),
                    )
            except InterruptedError:
                action = self._control_action() or "pause"
                return "stopped" if action == "stop" else "paused"
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                (
                    selection_failure_signature,
                    selection_failure_details,
                ) = self._failure_signature_from_parts(
                    topic_id=str(self.state.get("selected_topic_id") or ""),
                    stage="topic_selection",
                    status="failed",
                    error=error,
                    returncode=1,
                    final_status="failed",
                )
                previous_signature = str(
                    self.state.get("failure_signature") or ""
                )
                same_failure_count = (
                    int(self.state.get("consecutive_same_failure", 0)) + 1
                    if selection_failure_signature == previous_signature
                    else 1
                )
                recovery_action = (
                    "auto_repair"
                    if same_failure_count == 1
                    else "regenerate"
                    if same_failure_count == 2
                    else "quarantine"
                )
                self._transition(
                    status="running",
                    phase="idle",
                    active_cycle=None,
                    active_run_dir=None,
                    active_child_pid=None,
                    active_child_start_ticks=None,
                    next_cycle=cycle,
                    consecutive_failures=(
                        int(self.state.get("consecutive_failures", 0)) + 1
                    ),
                    failure_signature=selection_failure_signature,
                    failure_signature_details=selection_failure_details,
                    consecutive_same_failure=same_failure_count,
                    failure_recovery_action=recovery_action,
                    last_error=error,
                )
                self.store.log.append(
                    "topic_selection_failed",
                    campaign_id=self.campaign_id,
                    cycle=cycle,
                    error=error,
                )
                return False
        config_path = prepare_cycle_config(
            base_config=self.options.base_config,
            output_path=run_dir / "config.yaml",
            store=self.store,
            topic=effective_topic,
            campaign_brief=campaign_brief,
            selected_topic_path=selected_topic_path,
            autonomous_topic_selection=autonomous_selection,
            model=self.options.model,
            bridge_url=self.options.bridge_url,
            api_key_env=self.options.api_key_env,
            timeout_sec=self.options.llm_timeout_sec,
            cycle=cycle,
        )
        command = self._pipeline_command(run_dir, config_path)
        self._transition(
            status="running",
            phase="pipeline",
            active_cycle=cycle,
            active_run_dir=str(run_dir),
            active_child_pid=None,
            active_child_start_ticks=None,
            last_error=None,
        )
        self.store.log.append(
            "cycle_started",
            campaign_id=self.campaign_id,
            cycle=cycle,
            run_dir=str(run_dir),
            command=command,
            resumed_from_cycle=resumed_from_cycle,
            resume_stage=self._checkpoint_next_stage_name(run_dir),
        )

        returncode = self._run_pipeline(cycle, run_dir, command)
        action = self._control_action()
        if action:
            self.store.log.append(
                "cycle_interrupted",
                campaign_id=self.campaign_id,
                cycle=cycle,
                control=action,
                pipeline_returncode=returncode,
            )
            return "stopped" if action == "stop" else "paused"

        best_evidence = self._load_best_evidence(selected_topic_id)
        evidence = collect_evidence(
            run_dir,
            pipeline_returncode=returncode,
            command=command,
            best_evidence=best_evidence,
            topic_id=selected_topic_id,
        )
        pipeline_success = evidence_succeeded(evidence)
        comparison = evidence.get("comparison")
        accepted = (
            pipeline_success
            and isinstance(comparison, dict)
            and bool(comparison.get("accepted", False))
        )
        success = pipeline_success
        quality_score = evidence.get("quality_score")
        composite_score = evidence.get("composite_score")
        failure_signature, failure_signature_details = self._failure_signature(
            evidence,
            returncode=returncode,
            topic_id=selected_topic_id,
        )
        previous_failure_signature = str(
            self.state.get("failure_signature") or ""
        )
        previous_same_failure = int(
            self.state.get("consecutive_same_failure", 0)
        )
        if success:
            same_failure_count = 0
            failure_signature = ""
            failure_signature_details = None
            recovery_action = None
        else:
            same_failure_count = (
                previous_same_failure + 1
                if failure_signature
                and failure_signature == previous_failure_signature
                else 1
            )
            if same_failure_count <= 1:
                recovery_action = "auto_repair"
            elif same_failure_count == 2:
                recovery_action = "regenerate"
            else:
                recovery_action = "quarantine"

        diagnosis: dict[str, Any] = {}
        mutations: list[str] = []
        diagnostic_error: str | None = None
        if self.options.dry_run:
            diagnosis = {
                "summary": "Dry-run cycle; no LLM diagnosis was requested.",
                "strengths": [],
                "weaknesses": [],
                "next_cycle_priorities": [],
                "brief_patch": "",
                "prompt_patch": "",
                "topic_action": "keep",
                "topic_patch": "",
                "pivot_reason": "",
                "preferred_candidate_id": "",
                "stop_recommended": False,
                "stop_reason": "",
            }
            atomic_write_json(
                self.store.diagnostics_dir / f"cycle-{cycle:04d}.json",
                diagnosis,
            )
        else:
            self._transition(phase="diagnosis")
            try:
                with _Heartbeat(
                    self,
                    phase="diagnosis",
                    cycle=cycle,
                    run_dir=run_dir,
                ):
                    llm = self._llm_factory(config_path)
                    diagnosis = self._diagnosis_fn(
                        llm=llm,
                        evidence=evidence,
                        brief=campaign_brief,
                        cancel_event=self._cancel_event,
                    )
                    if not self.options.no_aevolve:
                        staged_skills = (
                            run_dir / "evolution" / "candidate_skills"
                        )
                        mutations = self._aevolve_fn(
                            llm=llm,
                            run_dir=run_dir,
                            skills_dir=staged_skills,
                            evidence=evidence,
                            cancel_event=self._cancel_event,
                        )
                    if accepted:
                        apply_diagnosis(
                            store=self.store,
                            cycle=cycle,
                            diagnosis=diagnosis,
                        )
                        self._promote_cycle_evolution(run_dir, cycle)
                    elif not success:
                        repair_result = apply_failure_repair(
                            store=self.store,
                            cycle=cycle,
                            diagnosis=diagnosis,
                            failure_signature=failure_signature,
                            recovery_action=str(recovery_action),
                        )
                        self.store.log.append(
                            "failure_repair_evaluated",
                            campaign_id=self.campaign_id,
                            cycle=cycle,
                            **repair_result,
                        )
                        mutations = []
                    else:
                        self.store.shared_repair_patch_path.unlink(
                            missing_ok=True
                        )
                        atomic_write_json(
                            self.store.diagnostics_dir
                            / f"cycle-{cycle:04d}.json",
                            diagnosis,
                        )
                        mutations = []
            except InterruptedError:
                action = self._control_action() or "pause"
                self._cancel_remote_pool_tasks(action)
                return "stopped" if action == "stop" else "paused"
            except Exception as exc:  # noqa: BLE001
                diagnostic_error = f"{type(exc).__name__}: {exc}"
                self.store.shared_repair_patch_path.unlink(missing_ok=True)
                self.store.log.append(
                    "diagnosis_failed",
                    campaign_id=self.campaign_id,
                    cycle=cycle,
                    error=diagnostic_error,
                )

        failures = int(self.state.get("consecutive_failures", 0))
        if success:
            failures = 0
            self.store.shared_repair_patch_path.unlink(missing_ok=True)
        else:
            failures += 1
        no_improvement = int(
            self.state.get("consecutive_no_improvement", 0)
        )
        if accepted:
            no_improvement = 0
        else:
            no_improvement += 1

        completed = int(self.state.get("completed_cycles", 0)) + 1
        successful = int(self.state.get("successful_cycles", 0)) + int(success)
        failed = int(self.state.get("failed_cycles", 0)) + int(not success)
        pending_topic_action = normalize_topic_action(diagnosis)
        if (
            pending_topic_action["topic_action"] == "refine"
            and not accepted
        ):
            # Bounded refinements are incumbent mutations and are promoted
            # only when the cycle itself is accepted. Evidence-invalidating
            # pivots remain eligible after a negative/failed pilot.
            pending_topic_action = normalize_topic_action(None)
        if pending_topic_action["topic_action"] == "pivot":
            self.store.shared_repair_patch_path.unlink(missing_ok=True)

        updates: dict[str, Any] = {
            "status": "running",
            "phase": "idle",
            "active_cycle": None,
            "active_run_dir": None,
            "active_child_pid": None,
            "active_child_start_ticks": None,
            "next_cycle": cycle + 1,
            "completed_cycles": completed,
            "successful_cycles": successful,
            "failed_cycles": failed,
            "consecutive_failures": failures,
            "failure_signature": failure_signature or None,
            "failure_signature_details": failure_signature_details,
            "consecutive_same_failure": same_failure_count,
            "failure_recovery_action": recovery_action,
            "consecutive_no_improvement": no_improvement,
            "last_run_dir": str(run_dir),
            "last_pipeline_returncode": returncode,
            "last_composite_score": composite_score,
            "last_comparison": comparison,
            "last_error": diagnostic_error
            or (
                None
                if success
                else self._failure_summary(evidence, returncode)
            ),
            "last_diagnosis": diagnosis,
            "pending_topic_action": pending_topic_action,
            "last_mutations": mutations,
        }
        if accepted:
            evidence_path = run_dir / "rsi_evidence.json"
            best_by_topic = self._best_by_topic()
            topic_key = selected_topic_id or "__campaign__"
            best_by_topic[topic_key] = {
                "score": composite_score,
                "cycle": cycle,
                "run_dir": str(run_dir),
                "evidence_path": str(evidence_path),
            }
            updates["best_by_topic"] = best_by_topic
            # Preserve the legacy summary as the most recently accepted
            # incumbent while topic-local comparisons use ``best_by_topic``.
            updates.update(
                {
                    "best_score": composite_score,
                    "best_cycle": cycle,
                    "best_run_dir": str(run_dir),
                    "best_evidence_path": str(evidence_path),
                }
            )
        self._transition(**updates)
        self.store.log.append(
            "cycle_completed",
            campaign_id=self.campaign_id,
            cycle=cycle,
            success=success,
            pipeline_returncode=returncode,
            quality_score=quality_score,
            composite_score=composite_score,
            accepted_as_best=accepted,
            consecutive_failures=failures,
            consecutive_same_failure=same_failure_count,
            failure_signature=failure_signature or None,
            failure_recovery_action=recovery_action,
            consecutive_no_improvement=no_improvement,
            mutations=mutations,
            diagnosis_error=diagnostic_error,
        )

        if (
            not self.options.continuous
            and bool(diagnosis.get("stop_recommended", False))
        ):
            self.store.set_control(
                "pause",
                str(diagnosis.get("stop_reason", "LLM recommended review")),
            )
            return "paused"
        return success

    @staticmethod
    def _checkpoint_next_stage_name(run_dir: Path) -> str | None:
        checkpoint = read_json(run_dir / "checkpoint.json", None)
        if not isinstance(checkpoint, dict):
            return None
        try:
            last_stage = int(checkpoint.get("last_completed_stage"))
        except (TypeError, ValueError):
            return None
        if last_stage >= 23:
            return None
        from researchclaw.pipeline.stages import Stage

        try:
            return Stage(last_stage + 1).name
        except ValueError:
            return None

    def _seed_failed_cycle_resume(self, run_dir: Path, cycle: int) -> int | None:
        """Seed a new cycle from the last failed/paused run.

        The outer RSI supervisor deliberately keeps cycle directories immutable
        after diagnosis.  Previously this meant every failed cycle restarted at
        Stage 1.  We now copy only the successfully checkpointed prefix into the
        new cycle, write the checkpoint last, and let the existing ``--resume``
        path continue from the failed stage.

        A topic pivot/refinement intentionally disables this carry-forward:
        changed scientific intent must invalidate downstream artifacts instead
        of accidentally inheriting the previous hypothesis.
        """

        if cycle <= 1 or (run_dir / "checkpoint.json").is_file():
            return None
        if str(self.state.get("failure_recovery_action") or "") in {
            "regenerate",
            "quarantine",
        }:
            self.store.log.append(
                "cycle_resume_skipped_for_regeneration",
                campaign_id=self.campaign_id,
                cycle=cycle,
                previous_failure_signature=self.state.get("failure_signature"),
            )
            return None
        pending = self.state.get("pending_topic_action")
        if isinstance(pending, Mapping):
            topic_action = normalize_topic_action(pending).get(
                "topic_action", "keep"
            )
        else:
            topic_action = "keep"
        if topic_action != "keep":
            return None
        previous_raw = self.state.get("last_run_dir")
        if not previous_raw:
            return None
        previous = Path(str(previous_raw))
        if not previous.is_dir() or previous.resolve() == run_dir.resolve():
            return None
        summary = read_json(previous / "pipeline_summary.json", None)
        if not isinstance(summary, dict):
            return None
        if str(summary.get("final_status", "")).lower() == "done":
            return None
        checkpoint = read_json(previous / "checkpoint.json", None)
        if not isinstance(checkpoint, dict):
            return None
        try:
            last_completed = int(checkpoint.get("last_completed_stage"))
        except (TypeError, ValueError):
            return None
        if not 1 <= last_completed < 23:
            return None

        copied: list[str] = []
        for stage_number in range(1, last_completed + 1):
            source = previous / f"stage-{stage_number:02d}"
            destination = run_dir / source.name
            if not source.is_dir() or destination.exists():
                continue
            shutil.copytree(source, destination)
            copied.append(source.name)
        for filename in (
            "analysis_best.md",
            "topic_candidates.json",
            "selected_topic.json",
            "topic_selection.md",
        ):
            source = previous / filename
            destination = run_dir / filename
            if source.is_file() and not destination.exists():
                shutil.copy2(source, destination)
                copied.append(filename)

        # Publish the checkpoint last.  A crash during copying therefore
        # causes a safe Stage-1 restart instead of resuming from partial data.
        try:
            source_cycle = int(previous.name.removeprefix("cycle-"))
        except ValueError:
            source_cycle = cycle - 1
        checkpoint_copy = dict(checkpoint)
        checkpoint_copy["resumed_from_cycle"] = source_cycle
        checkpoint_copy["resumed_from_run_dir"] = str(previous)
        atomic_write_json(run_dir / "checkpoint.json", checkpoint_copy)
        manifest = {
            "source_cycle": source_cycle,
            "source_run_dir": str(previous),
            "target_cycle": cycle,
            "last_completed_stage": last_completed,
            "resume_stage": self._checkpoint_next_stage_name(run_dir),
            "copied": copied,
            "timestamp": utc_now(),
        }
        atomic_write_json(run_dir / "resume_manifest.json", manifest)
        self.store.log.append(
            "cycle_resume_seeded",
            campaign_id=self.campaign_id,
            cycle=cycle,
            **manifest,
        )
        return source_cycle

    def _pipeline_command(self, run_dir: Path, config_path: Path) -> list[str]:
        if self.options.dry_run:
            code = (
                "import json,pathlib,sys;"
                "p=pathlib.Path(sys.argv[1]);p.mkdir(parents=True,exist_ok=True);"
                "(p/'pipeline_summary.json').write_text(json.dumps({"
                "'run_id':'rsi-dry-run','stages_executed':0,'stages_done':0,"
                "'stages_paused':0,'stages_blocked':0,'stages_failed':0,"
                "'degraded':False,'from_stage':1,'final_stage':0,"
                "'final_status':'done','content_metrics':{}},indent=2))"
            )
            return [sys.executable, "-c", code, str(run_dir)]

        command = [
            sys.executable,
            "-m",
            "researchclaw",
            "run",
            "--config",
            str(config_path),
            "--output",
            str(run_dir),
            "--auto-approve",
            "--no-graceful-degradation",
        ]
        if (run_dir / "checkpoint.json").is_file():
            command.append("--resume")
        if self.options.skip_preflight:
            command.append("--skip-preflight")
        command.extend(self.options.pipeline_extra_args)
        return command

    def _run_pipeline(
        self,
        cycle: int,
        run_dir: Path,
        command: Sequence[str],
    ) -> int:
        log_path = run_dir / "pipeline.log"
        env = os.environ.copy()
        env.update(self.options.env)
        if not env.get(self.options.api_key_env):
            env[self.options.api_key_env] = "local-bridge"
        with log_path.open("a", encoding="utf-8") as log_handle:
            dependency_results = ensure_runtime_dependencies(
                python_executable=command[0],
                auto_install=True,
                env=env,
            )
            for result in dependency_results:
                log_handle.write(
                    f"[{utc_now()}] runtime dependency {result.module}: "
                    f"{result.status}"
                    + (f" ({result.detail})" if result.detail else "")
                    + "\n"
                )
            log_handle.flush()
            if not dependencies_satisfied(dependency_results):
                self.store.log.append(
                    "runtime_dependency_failure",
                    campaign_id=self.campaign_id,
                    cycle=cycle,
                    python=str(command[0]),
                    dependencies=[
                        result.to_dict() for result in dependency_results
                    ],
                )
                log_handle.write(
                    f"[{utc_now()}] required runtime dependency installation "
                    "failed; pipeline was not started\n"
                )
                log_handle.flush()
                return 70
            log_handle.write(
                f"[{utc_now()}] command: {json.dumps(list(command), ensure_ascii=False)}\n"
            )
            log_handle.flush()
            self._child = self._popen_factory(
                list(command),
                cwd=self.options.repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self._transition(
                active_child_pid=self._child.pid,
                active_child_start_ticks=_process_start_ticks(self._child.pid),
            )
            with _Heartbeat(
                self,
                phase="pipeline",
                cycle=cycle,
                run_dir=run_dir,
                child_pid=lambda: self._child.pid if self._child else None,
            ) as heartbeat:
                while True:
                    returncode = self._child.poll()
                    if returncode is not None:
                        return int(returncode)
                    action = self._control_action()
                    if action:
                        self._terminate_child(action)
                        return int(self._child.wait())
                    heartbeat.tick()
                    self._sleep(max(0.05, self.options.control_poll_sec))
            # The return paths above intentionally bypass this point.  The
            # child identity remains durable until cycle finalization so a
            # SIGKILL between poll() and state update can still be reconciled.

    def _terminate_child(self, reason: str) -> None:
        child = self._child
        if child is None or child.poll() is not None:
            return
        self.store.log.append(
            "pipeline_termination_requested",
            campaign_id=self.campaign_id,
            cycle=self.state.get("active_cycle"),
            reason=reason,
            child_pid=child.pid,
        )
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            child.terminate()
        deadline = time.monotonic() + 10.0
        while child.poll() is None and time.monotonic() < deadline:
            self._sleep(0.1)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                child.kill()
        self._cancel_remote_pool_tasks(reason)

    def _cancel_remote_pool_tasks(
        self,
        reason: str,
        *,
        run_dir: Path | None = None,
    ) -> None:
        run_dir_raw = run_dir or self.state.get("active_run_dir")
        if not run_dir_raw:
            return
        run_dir = Path(str(run_dir_raw))
        task_ids: set[str] = set()
        metadata_paths = list(run_dir.rglob(".clusterbridge_pool_task.json"))
        pool_config = self.options.repo_root / "config.cluster32.yaml"
        try:
            from researchclaw.cluster import ClusterBridgePoolConfig

            task_root = (
                ClusterBridgePoolConfig.from_file(pool_config).state_dir
                / "experiments"
            )
        except Exception:  # noqa: BLE001
            task_root = None
        if task_root is not None and task_root.is_dir():
            known_ids = {
                path.parent.name
                for path in metadata_paths
                if path.name == ".clusterbridge_pool_task.json"
            }
            for path in task_root.glob("*/.clusterbridge_pool_task.json"):
                if path.parent.name in known_ids:
                    metadata_paths.append(path)
        for path in metadata_paths:
            value = read_json(path, {})
            if not isinstance(value, dict):
                continue
            task_id = str(value.get("task_id", "")).strip()
            state = str(value.get("state", "")).strip()
            if task_id and state not in {"finished", "timed_out", "failed"}:
                task_ids.add(task_id)
        if not task_ids:
            return
        pool_cli = self.options.repo_root / "bin" / "cluster-pool"
        for task_id in sorted(task_ids):
            result = subprocess.run(
                [
                    str(pool_cli),
                    "--config",
                    str(pool_config),
                    "cancel-task",
                    task_id,
                ],
                cwd=self.options.repo_root,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            self.store.log.append(
                "remote_pool_task_cancelled",
                campaign_id=self.campaign_id,
                cycle=self.state.get("active_cycle"),
                task_id=task_id,
                reason=reason,
                returncode=result.returncode,
                detail=(result.stderr or result.stdout)[-4000:],
            )

    def _reconcile_interrupted_execution(
        self,
        *,
        current_pid: int | None = None,
        current_start_ticks: int | None = None,
    ) -> None:
        """Clean tasks left by a supervisor that died before finalization."""

        if str(self.state.get("status", "")) != "running":
            return
        prior_pid = int(self.state.get("pid") or 0)
        prior_start = self.state.get("supervisor_start_ticks")
        prior_start_ticks = self._optional_nonnegative_int(prior_start)
        if (
            current_pid is not None
            and prior_pid == current_pid
            and prior_start_ticks is not None
            and prior_start_ticks == current_start_ticks
        ):
            # The background launcher may publish this process identity early
            # so its startup handshake is not blocked by slow shared-CephFS
            # metadata scans. It is the current supervisor, not an orphan.
            return
        if prior_start_ticks is not None and _process_identity_matches(
            prior_pid,
            prior_start_ticks,
        ):
            # The campaign lock normally prevents reaching this branch, but
            # fail closed if state says the previous supervisor is still alive.
            raise RuntimeError(
                f"previous campaign supervisor still appears alive: {prior_pid}"
            )

        run_dir_raw = self.state.get("active_run_dir")
        run_dir = Path(str(run_dir_raw)) if run_dir_raw else None
        child_pid = int(self.state.get("active_child_pid") or 0)
        child_start = self.state.get("active_child_start_ticks")
        child_start_ticks = self._optional_nonnegative_int(child_start)
        local_outcome = "not_recorded"
        if child_start_ticks is not None and _process_identity_matches(
            child_pid,
            child_start_ticks,
        ):
            local_outcome = self._terminate_orphan_process_group(child_pid)
        elif child_pid:
            local_outcome = "already_exited_or_reused"

        remote_error: str | None = None
        if run_dir is not None:
            try:
                self._cancel_remote_pool_tasks(
                    "supervisor_restart_reconciliation",
                    run_dir=run_dir,
                )
            except Exception as exc:  # noqa: BLE001
                remote_error = f"{type(exc).__name__}: {exc}"

        self.store.log.append(
            "interrupted_execution_reconciled",
            campaign_id=self.campaign_id,
            cycle=self.state.get("active_cycle"),
            previous_supervisor_pid=prior_pid or None,
            child_pid=child_pid or None,
            local_outcome=local_outcome,
            remote_error=remote_error,
            run_dir=str(run_dir) if run_dir is not None else None,
        )
        self._transition(
            status="interrupted",
            phase="idle",
            pid=None,
            supervisor_start_ticks=None,
            active_child_pid=None,
            active_child_start_ticks=None,
            last_error=(
                "recovered execution left by a terminated supervisor"
                + (f"; remote cleanup error: {remote_error}" if remote_error else "")
            ),
        )

    @staticmethod
    def _optional_nonnegative_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None

    def _terminate_orphan_process_group(self, pid: int) -> str:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return "already_exited"
        except (PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                return "already_exited"
        deadline = time.monotonic() + 10.0
        while _process_start_ticks(pid) is not None and time.monotonic() < deadline:
            self._sleep(0.1)
        if _process_start_ticks(pid) is None:
            return "terminated"
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return "terminated"
        except (PermissionError, OSError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                return "terminated"
        return "killed"

    def _promote_cycle_evolution(self, run_dir: Path, cycle: int) -> None:
        candidate_skills = run_dir / "evolution" / "candidate_skills"
        if candidate_skills.is_dir():
            for source in sorted(candidate_skills.iterdir()):
                if not source.is_dir():
                    continue
                destination = self.store.shared_skills_dir / source.name
                temporary = self.store.shared_skills_dir / (
                    f".{source.name}.cycle-{cycle:04d}.tmp"
                )
                if temporary.exists():
                    shutil.rmtree(temporary)
                shutil.copytree(source, temporary)
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(temporary, destination)

        evolution_dir = run_dir / "evolution"
        lessons = evolution_dir / "lessons.jsonl"
        if lessons.is_file():
            shared_lessons = self.store.shared_dir / "lessons.jsonl"
            seed = read_json(run_dir / ".rsi_seed.json", {})
            prior_size = (
                int(seed.get("lessons_bytes", 0))
                if isinstance(seed, dict)
                else 0
            )
            with lessons.open("rb") as source:
                source.seek(min(prior_size, lessons.stat().st_size))
                delta = source.read()
            if delta:
                with shared_lessons.open("ab") as target:
                    target.write(delta)

        prompt_patch = evolution_dir / "prompt_patches.md"
        if prompt_patch.is_file():
            text = prompt_patch.read_text(encoding="utf-8").strip()
            if text:
                with self.store.shared_prompt_path.open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write(f"\n\n## Cycle {cycle} A-Evolve patches\n{text}\n")

        knowledge = evolution_dir / "knowledge_entries.jsonl"
        if knowledge.is_file():
            shared_knowledge = self.store.shared_dir / "knowledge_entries.jsonl"
            seed = read_json(run_dir / ".rsi_seed.json", {})
            prior_size = (
                int(seed.get("knowledge_bytes", 0))
                if isinstance(seed, dict)
                else 0
            )
            with knowledge.open("rb") as source:
                source.seek(min(prior_size, knowledge.stat().st_size))
                delta = source.read()
            if delta:
                with shared_knowledge.open("ab") as target:
                    target.write(delta)

    def _seed_run_memory(self, run_dir: Path) -> None:
        """Copy campaign lessons into the run-local store used by the pipeline."""

        shared_lessons = self.store.shared_dir / "lessons.jsonl"
        evolution_dir = run_dir / "evolution"
        evolution_dir.mkdir(parents=True, exist_ok=True)
        seeded_bytes = 0
        if shared_lessons.is_file():
            destination = evolution_dir / "lessons.jsonl"
            shutil.copy2(shared_lessons, destination)
            seeded_bytes = destination.stat().st_size
        shared_knowledge = self.store.shared_dir / "knowledge_entries.jsonl"
        seeded_knowledge_bytes = 0
        if shared_knowledge.is_file():
            destination = evolution_dir / "knowledge_entries.jsonl"
            shutil.copy2(shared_knowledge, destination)
            seeded_knowledge_bytes = destination.stat().st_size
        atomic_write_json(
            run_dir / ".rsi_seed.json",
            {
                "lessons_bytes": seeded_bytes,
                "shared_lessons": str(shared_lessons),
                "knowledge_bytes": seeded_knowledge_bytes,
                "shared_knowledge": str(shared_knowledge),
                "seeded_at": utc_now(),
            },
        )

    def _load_best_evidence(
        self,
        topic_id: str = "",
    ) -> dict[str, Any] | None:
        if topic_id:
            record = self._best_by_topic().get(topic_id)
            if isinstance(record, Mapping):
                raw_path = record.get("evidence_path")
                if raw_path:
                    value = read_json(Path(str(raw_path)), None)
                    if isinstance(value, dict) and self._evidence_matches_topic(
                        value, topic_id
                    ):
                        return value
                run_dir = record.get("run_dir")
                if run_dir:
                    value = read_json(
                        Path(str(run_dir)) / "rsi_evidence.json",
                        None,
                    )
                    if isinstance(value, dict) and self._evidence_matches_topic(
                        value, topic_id
                    ):
                        return value
        raw_path = self.state.get("best_evidence_path")
        if raw_path:
            value = read_json(Path(str(raw_path)), None)
            if isinstance(value, dict) and self._evidence_matches_topic(
                value, topic_id
            ):
                return value
        best_run = self.state.get("best_run_dir")
        if best_run:
            value = read_json(Path(str(best_run)) / "rsi_evidence.json", None)
            if isinstance(value, dict) and self._evidence_matches_topic(
                value, topic_id
            ):
                return value
        return None

    def _best_by_topic(self) -> dict[str, Any]:
        value = self.state.get("best_by_topic")
        if not isinstance(value, Mapping):
            return {}
        return {str(key): item for key, item in value.items()}

    @staticmethod
    def _evidence_matches_topic(
        evidence: Mapping[str, Any],
        topic_id: str,
    ) -> bool:
        if not topic_id:
            return True
        incumbent_id = str(evidence.get("topic_id", "") or "").strip()
        # Legacy evidence without a topic ID is comparable only before the
        # campaign has an explicit autonomous selection.
        return bool(incumbent_id) and incumbent_id == topic_id

    def _interruptible_backoff(self, delay: float, cycle: int) -> str | None:
        if delay <= 0:
            return None
        self._transition(phase="backoff", backoff_until=time.time() + delay)
        self.store.log.append(
            "backoff_started",
            campaign_id=self.campaign_id,
            cycle=cycle,
            delay_sec=delay,
        )
        remaining = delay
        while remaining > 0:
            action = self._control_action()
            if action:
                return action
            chunk = min(max(0.05, self.options.control_poll_sec), remaining)
            self.store.write_heartbeat(
                {
                    "campaign_id": self.campaign_id,
                    "supervisor_pid": os.getpid(),
                    "child_pid": None,
                    "cycle": cycle,
                    "phase": "backoff",
                    "remaining_sec": round(remaining, 3),
                    "status": self.state.get("status"),
                }
            )
            self._sleep(chunk)
            remaining -= chunk
        self._transition(phase="idle", backoff_until=None)
        return None

    def _backoff_delay(self, failures: int) -> float:
        exponent = max(0, failures - 1)
        return min(
            max(0.0, self.options.backoff_initial_sec) * (2**exponent),
            max(0.0, self.options.backoff_max_sec),
        )

    def _control_action(self) -> str | None:
        if self._signal in {signal.SIGTERM, signal.SIGINT}:
            return "stop"
        if hasattr(signal, "SIGUSR1") and self._signal == signal.SIGUSR1:
            return "pause"
        if self.store.control_requested("stop"):
            return "stop"
        if self.store.control_requested("pause"):
            return "pause"
        return None

    def _finish_for_control(self, action: str) -> int:
        status = "stopped" if action == "stop" else "paused"
        self._transition(
            status=status,
            phase="idle",
            pid=None,
            supervisor_start_ticks=None,
            active_cycle=None,
            active_run_dir=None,
            active_child_pid=None,
            active_child_start_ticks=None,
        )
        self.store.log.append(
            f"campaign_{status}",
            campaign_id=self.campaign_id,
            reason=action,
        )
        return 0

    def _transition(self, **updates: Any) -> None:
        self.state.update(updates)
        self.state = self.store.save_state(self.state)

    def _current_brief(self) -> str:
        try:
            text = self.store.shared_brief_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            text = self.options.topic.strip()
        return text or self.options.topic.strip()

    def _pending_topic_action(self) -> dict[str, str]:
        """Read the accepted action that governs the next topic decision."""

        return normalize_topic_action(self.state.get("pending_topic_action"))

    def _autonomous_topic_selection_enabled(self) -> bool:
        """Return whether the base campaign config delegates topic choice."""

        try:
            data = load_yaml_mapping(self.options.base_config)
        except (FileNotFoundError, OSError, TypeError, ValueError):
            return False
        research = data.get("research")
        return bool(
            isinstance(research, dict)
            and research.get("autonomous_topic_selection", False)
        )

    @staticmethod
    def _failure_summary(evidence: dict[str, Any], returncode: int) -> str:
        failures = evidence.get("failures")
        if isinstance(failures, list) and failures:
            first = failures[0]
            if isinstance(first, dict):
                return (
                    f"pipeline exit {returncode}; "
                    f"{first.get('stage_name', 'stage')}: "
                    f"{first.get('error') or first.get('status')}"
                )
        return f"pipeline exit {returncode} or incomplete summary"

    @staticmethod
    def _failure_signature(
        evidence: Mapping[str, Any],
        *,
        returncode: int,
        topic_id: str,
    ) -> tuple[str, dict[str, Any] | None]:
        """Build a stable idea+stage+error signature for circuit breaking."""

        stage = "pipeline"
        stage_number: int | None = None
        error = ""
        status = ""
        failures = evidence.get("failures")
        if isinstance(failures, Sequence) and not isinstance(
            failures, (str, bytes, bytearray)
        ):
            for item in failures:
                if not isinstance(item, Mapping):
                    continue
                stage = str(item.get("stage_name") or item.get("stage") or stage)
                try:
                    stage_number = int(item.get("stage"))
                except (TypeError, ValueError):
                    stage_number = None
                error = str(item.get("error") or "")
                status = str(item.get("status") or "")
                break
        summary = evidence.get("pipeline_summary")
        final_status = (
            str(summary.get("final_status") or "")
            if isinstance(summary, Mapping)
            else ""
        )
        return CampaignSupervisor._failure_signature_from_parts(
            topic_id=topic_id,
            stage=stage,
            stage_number=stage_number,
            status=status,
            error=error,
            returncode=returncode,
            final_status=final_status,
        )

    @staticmethod
    def _failure_signature_from_parts(
        *,
        topic_id: str,
        stage: str,
        status: str,
        error: str,
        returncode: int,
        final_status: str,
        stage_number: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        normalized_error = " ".join(str(error).casefold().split())
        normalized_error = re.sub(r"/[^\s:]+", "<path>", normalized_error)
        normalized_error = re.sub(
            r"\b(?:pid|request|attempt|retry|run|job|task)[-_ ]?(?:id)?"
            r"[:=# ]+\d+\b",
            lambda match: re.sub(r"\d+", "<n>", match.group(0)),
            normalized_error,
        )
        normalized_error = re.sub(
            r"\b[0-9a-f]{12,}\b",
            "<id>",
            normalized_error,
        )[:1000]
        details = {
            "topic_id": str(topic_id or "__campaign__"),
            "stage": str(stage or "pipeline"),
            "stage_number": stage_number,
            "status": str(status or ""),
            "error": normalized_error,
            "returncode": int(returncode),
            "final_status": str(final_status or ""),
        }
        canonical = json.dumps(details, sort_keys=True, ensure_ascii=False)
        signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
        return signature, details

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def handler(signum: int, _frame: object) -> None:
            self._signal = signum
            self._cancel_event.set()

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, handler)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, handler)


def campaign_status(campaign_dir: Path) -> dict[str, Any]:
    store = CampaignStore(campaign_dir)
    state = store.load_state()
    heartbeat = read_json(store.heartbeat_path, {})
    if isinstance(heartbeat, dict):
        state["heartbeat"] = heartbeat
    state["campaign_dir"] = str(store.root)
    state["pause_requested"] = store.control_requested("pause")
    state["stop_requested"] = store.control_requested("stop")
    return state
