"""Steady-state multi-Idea controller for AutoResearch v2."""

from __future__ import annotations

import concurrent.futures
import json
import random
import shlex
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .config import V2Config
from .gpu import GPUBroker
from .ideas import IdeaAdmission, IdeaGenerator
from .jobs import JobExecutor, JobOutcome, SimulatedJobExecutor, _experiment_gate
from .models import (
    ACTIVE_IDEA_STATUSES,
    AttemptRecord,
    AttemptStatus,
    IdeaRecord,
    IdeaStatus,
    JobKind,
    JobRecord,
    JobStatus,
    utc_now,
)
from .store import V2Store
from .validation import (
    validate_experiment_artifacts,
    validate_runtime_against_contract,
)

_KIND_FOR_STATUS = {
    IdeaStatus.NEW: JobKind.DESIGN,
    IdeaStatus.DESIGNING: JobKind.DESIGN,
    IdeaStatus.BUILDING: JobKind.BUILD,
    IdeaStatus.PILOTING: JobKind.PILOT,
    IdeaStatus.SCALING: JobKind.SCALE,
    IdeaStatus.REPORTING: JobKind.REPORT,
}

_STATUS_FOR_KIND = {
    JobKind.DESIGN: IdeaStatus.DESIGNING,
    JobKind.BUILD: IdeaStatus.BUILDING,
    JobKind.PILOT: IdeaStatus.PILOTING,
    JobKind.SCALE: IdeaStatus.SCALING,
    JobKind.REPORT: IdeaStatus.REPORTING,
}

_LLM_KINDS = {JobKind.DESIGN, JobKind.BUILD, JobKind.REPORT}


@dataclass(slots=True)
class _Running:
    future: concurrent.futures.Future[JobOutcome]
    job_id: str
    attempt_id: str


class V2Controller:
    """One controller, SQLite source of truth, isolated immutable attempts."""

    def __init__(
        self,
        *,
        config: V2Config,
        store: V2Store,
        generator: IdeaGenerator,
        executors: Mapping[JobKind, JobExecutor] | None = None,
        gpu_broker: GPUBroker | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.store = store
        self.generator = generator
        self.executors = dict(executors or {})
        self._simulation_mode = (
            generator.__class__.__name__ == "StaticIdeaGenerator"
        )
        self.default_executor: JobExecutor = (
            SimulatedJobExecutor()
            if self._simulation_mode
            else _MissingExecutor()
        )
        self.gpu_broker = gpu_broker
        self.admission = IdeaAdmission(
            duplicate_threshold=config.admission.duplicate_threshold,
            max_same_family=config.population.max_same_family,
            minimum_score=config.admission.minimum_score,
            semantic_duplicate_threshold=(
                config.admission.semantic_duplicate_threshold
            ),
            require_novelty_evidence=(
                config.admission.require_novelty_evidence
                and not self._simulation_mode
            ),
        )
        self.sleep = sleep
        self._pool = _ControllerThreadPool(
            max_workers=max(
                config.concurrency.max_cpu_jobs,
                config.concurrency.max_llm_jobs,
            ),
            thread_name_prefix="autoresearch-v2",
            on_shutdown=store.release_writer_lock,
        )
        self._running: dict[str, _Running] = {}
        self._stop = False
        self._initialized = False
        self._tick_count = 0

    def initialize(self) -> None:
        if self._initialized:
            return
        self.store.initialize()
        self.store.acquire_writer_lock()
        self._recover_interrupted_jobs()
        self._initialized = True
        self.store.event(
            "controller_initialized",
            max_cpu_jobs=self.config.concurrency.max_cpu_jobs,
            max_llm_jobs=self.config.concurrency.max_llm_jobs,
            max_gpu_jobs=self.config.concurrency.max_gpu_jobs,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)
        if self.gpu_broker is not None:
            self.gpu_broker.close()
        self.store.release_writer_lock()
        self._initialized = False

    def run(
        self,
        *,
        max_ticks: int | None = None,
        until_idle: bool = False,
    ) -> int:
        self.initialize()
        ticks = 0
        try:
            while not self._stop:
                try:
                    self.tick()
                except Exception as exc:  # noqa: BLE001
                    self.store.event(
                        "controller_tick_failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                ticks += 1
                if max_ticks is not None and ticks >= max_ticks:
                    break
                if until_idle and self.is_idle():
                    break
                self.sleep(self.config.concurrency.poll_interval_sec)
        finally:
            # Bounded CLI/test runs stop scheduling at max_ticks, but work
            # already admitted in the final tick must still be collected into
            # durable state before the process exits.
            self._drain_local_futures()
            self.close()
        return ticks

    def _drain_local_futures(self) -> None:
        # A bounded CLI run is a bounded scheduler probe, not an implicit
        # request to finish every downstream stage. Drain only work that was
        # already submitted by the final tick. The next invocation resumes the
        # durable state and schedules subsequent jobs.
        while self._running:
            concurrent.futures.wait(
                [entry.future for entry in self._running.values()],
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            self._collect_finished()

    def request_stop(self) -> None:
        self._stop = True

    def tick(self) -> dict[str, Any]:
        self._tick_count += 1
        if self.store.control_requested("stop"):
            self._stop = True
        self._collect_finished()
        self._collect_gpu_finished()
        self._enforce_liveness_budgets()
        if not self.store.control_requested("pause") and not self._stop:
            self._maintain_reservoir()
            self._admit_reservoir()
            self._ensure_jobs()
            self._dispatch()
        if (
            self._tick_count
            % self.config.retention.maintenance_interval_ticks
            == 0
        ):
            maintenance = self.store.maintain(
                event_jsonl_max_bytes=(
                    self.config.retention.event_jsonl_max_mb
                    * 1024
                    * 1024
                ),
                llm_audit_max_bytes=(
                    self.config.retention.llm_audit_max_mb
                    * 1024
                    * 1024
                ),
                keep_failed_attempts_per_job=(
                    self.config.retention.keep_failed_attempts_per_job
                ),
            )
            self.store.event("maintenance_completed", **maintenance)
        snapshot = self.snapshot()
        self.store.event("controller_tick", **snapshot)
        return snapshot

    def is_idle(self) -> bool:
        active = self.store.list_ideas(statuses=set(ACTIVE_IDEA_STATUSES))
        ready = self.store.list_jobs(
            statuses={
                JobStatus.READY,
                JobStatus.RUNNING,
                JobStatus.RETRY_WAIT,
            }
        )
        gpu_running = bool(
            self.gpu_broker and self.gpu_broker.leases
        )
        return not active and not ready and not self._running and not gpu_running

    def _maintain_reservoir(self) -> None:
        ideas = self.store.list_ideas()
        reservoir = [
            idea for idea in ideas if idea.status is IdeaStatus.RESERVOIR
        ]
        if len(reservoir) >= self.config.population.reservoir_low_watermark:
            return
        needed = max(
            self.config.population.generation_batch_size,
            self.config.population.reservoir_target - len(reservoir),
        )
        try:
            generated = self.generator.generate(
                count=needed,
                existing=ideas,
            )
        except Exception as exc:  # noqa: BLE001
            self.store.event(
                "idea_generation_failed",
                requested=needed,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        added = 0
        rejected = 0
        known = list(ideas)
        for idea in generated:
            decision = self.admission.decide(idea, existing=known)
            if not decision.admitted:
                idea.status = IdeaStatus.REJECTED
                idea.exit_reason = decision.reason
                self.store.save_idea(idea)
                self.store.event(
                    "idea_rejected",
                    idea_id=idea.idea_id,
                    reason=decision.reason,
                    duplicate_of=decision.duplicate_of,
                )
                rejected += 1
                known.append(idea)
                continue
            idea.status = IdeaStatus.RESERVOIR
            self.store.save_idea(idea)
            self.store.event(
                "idea_added_to_reservoir",
                idea_id=idea.idea_id,
                score=idea.score,
                family=idea.family,
            )
            known.append(idea)
            added += 1
        self.store.event(
            "idea_generation_batch",
            requested=needed,
            generated=len(generated),
            reservoir_added=added,
            rejected=rejected,
        )

    def _admit_reservoir(self) -> None:
        ideas = self.store.list_ideas()
        active = [idea for idea in ideas if idea.status in ACTIVE_IDEA_STATUSES]
        capacity = min(
            self.config.population.active_idea_target - len(active),
            self.config.population.max_active_ideas - len(active),
        )
        if capacity <= 0:
            return
        active_families: dict[str, int] = {}
        for idea in active:
            active_families[idea.family] = (
                active_families.get(idea.family, 0) + 1
            )
        reservoir = sorted(
            (
                idea
                for idea in ideas
                if idea.status is IdeaStatus.RESERVOIR
            ),
            key=lambda item: (-item.priority, item.created_at, item.idea_id),
        )
        admitted = 0
        for idea in reservoir:
            if admitted >= capacity:
                break
            if (
                active_families.get(idea.family, 0)
                >= self.config.population.max_same_family
            ):
                continue
            idea.status = IdeaStatus.DESIGNING
            idea.last_progress_at = utc_now()
            self.store.save_idea(idea)
            active_families[idea.family] = (
                active_families.get(idea.family, 0) + 1
            )
            admitted += 1
            self.store.event(
                "idea_admitted",
                idea_id=idea.idea_id,
                score=idea.score,
                family=idea.family,
            )

    def _ensure_jobs(self) -> None:
        existing = {job.job_id: job for job in self.store.list_jobs()}
        for idea in self.store.list_ideas(statuses=set(ACTIVE_IDEA_STATUSES)):
            if idea.current_job_id:
                current = existing.get(idea.current_job_id)
                if current and current.status in {
                    JobStatus.READY,
                    JobStatus.RUNNING,
                    JobStatus.RETRY_WAIT,
                }:
                    continue
            kind = _KIND_FOR_STATUS.get(idea.status)
            if kind is None:
                continue
            base = f"{idea.idea_id}-{kind.value}"
            index = 1
            job_id = base
            while job_id in existing:
                index += 1
                job_id = f"{base}-{index:02d}"
            compute = idea.candidate.get("compute", {})
            requested_gpus = 0
            if kind in {JobKind.PILOT, JobKind.SCALE}:
                try:
                    requested_gpus = max(
                        1, int(compute.get("gpu_count", 1))
                    )
                except (AttributeError, TypeError, ValueError):
                    requested_gpus = 1
            max_gpus = (
                self.config.gpu.pilot_max_gpus
                if kind is JobKind.PILOT
                else self.config.gpu.scale_max_gpus
            )
            job = JobRecord(
                job_id=job_id,
                idea_id=idea.idea_id,
                kind=kind,
                attempt_limit=(
                    self.config.budgets.max_build_attempts
                    if kind is JobKind.BUILD
                    else self.config.budgets.max_job_attempts
                ),
                requires_gpu=bool(requested_gpus),
                min_gpus=1 if requested_gpus else 0,
                preferred_gpus=(
                    min(requested_gpus, max_gpus)
                    if requested_gpus
                    else 0
                ),
                max_gpus=max_gpus if requested_gpus else 0,
                timeout_sec=self._job_timeout(idea, kind),
            )
            idea.current_job_id = job.job_id
            idea.status = _STATUS_FOR_KIND[kind]
            self.store.save_job(job)
            self.store.save_idea(idea)
            existing[job.job_id] = job
            self.store.event(
                "job_created",
                idea_id=idea.idea_id,
                job_id=job.job_id,
                kind=kind.value,
            )

    def _job_timeout(self, idea: IdeaRecord, kind: JobKind) -> float:
        if kind not in {JobKind.PILOT, JobKind.SCALE}:
            return 3600.0
        compute = idea.candidate.get("compute", {})
        try:
            requested = float(compute.get("wall_clock_hours", 1) or 1)
        except (AttributeError, TypeError, ValueError):
            requested = 1.0
        cap = (
            self.config.budgets.pilot_gpu_hours
            if kind is JobKind.PILOT
            else self.config.budgets.scale_gpu_hours
        )
        return max(60.0, min(requested, cap) * 3600.0)

    def _dispatch(self) -> None:
        running_llm = sum(
            1
            for entry in self._running.values()
            if (
                (job := self.store.get_job(entry.job_id)) is not None
                and job.kind in _LLM_KINDS
            )
        )
        running_cpu = len(self._running)
        running_gpu = len(self.gpu_broker.leases) if self.gpu_broker else 0
        ready = self.store.list_jobs(
            statuses={JobStatus.READY, JobStatus.RETRY_WAIT}
        )
        now = datetime.now(UTC)
        ready = [
            job
            for job in ready
            if (
                job.status is JobStatus.READY
                or not job.retry_not_before
                or (
                    (retry_at := _parse_time(job.retry_not_before)) is None
                    or retry_at <= now
                )
            )
        ]
        priorities = {
            idea.idea_id: idea.priority for idea in self.store.list_ideas()
        }
        gpu_ready = [
            job
            for job in ready
            if job.requires_gpu and self.config.gpu.enabled
        ]
        disabled_gpu = [
            job
            for job in ready
            if (
                job.requires_gpu
                and not self.config.gpu.enabled
                and not self._simulation_mode
            )
        ]
        cpu_ready = [
            job
            for job in ready
            if (
                not job.requires_gpu
                or (
                    self._simulation_mode
                    and not self.config.gpu.enabled
                )
            )
        ]
        for job in disabled_gpu:
            idea = self.store.get_idea(job.idea_id)
            if idea is not None:
                self._reject_budget(
                    idea,
                    job,
                    "gpu_required_but_disabled",
                )
        if self.gpu_broker is not None:
            gpu_ready = self.gpu_broker.scheduler.order(
                gpu_ready,
                priorities,
            )
        cpu_ready.sort(
            key=lambda job: (
                -priorities.get(job.idea_id, 0.0),
                job.created_at,
            )
        )
        # Dispatch GPU jobs first. They do not consume local CPU worker slots.
        for job in gpu_ready:
            if not self.config.gpu.enabled:
                continue
            if running_gpu >= self.config.concurrency.max_gpu_jobs:
                break
            idea = self.store.get_idea(job.idea_id)
            if idea is None:
                continue
            budget_reason = self._budget_block_reason(idea, job)
            if budget_reason:
                self._reject_budget(idea, job, budget_reason)
                continue
            if self.gpu_broker is None:
                continue
            attempt = self.store.create_attempt(job)
            attempt.status = AttemptStatus.RUNNING
            attempt.started_at = utc_now()
            candidate = self.store.snapshot_current(attempt)
            try:
                command, output_dir = self._gpu_command(
                    idea=idea,
                    job=job,
                    attempt=attempt,
                    candidate=candidate,
                )
            except Exception as exc:  # noqa: BLE001
                attempt.status = AttemptStatus.FAILED
                attempt.error = f"{type(exc).__name__}: {exc}"
                attempt.finished_at = utc_now()
                self.store.save_attempt(attempt)
                outcome = JobOutcome(
                    False,
                    "retry",
                    "gpu_command_preparation_failed",
                    {"error": attempt.error},
                )
                job.attempt = attempt.number
                self._apply_outcome(idea, job, attempt, outcome)
                continue
            job.command = command
            job.expected_output_dir = str(output_dir)
            job.attempt_id = attempt.attempt_id
            job.submitted_task_id = (
                f"{job.job_id}-attempt-{attempt.number:02d}"
            )
            self.store.save_attempt(attempt)
            try:
                decision = self.gpu_broker.submit(
                    job,
                    priorities=priorities,
                )
            except Exception as exc:  # noqa: BLE001
                attempt.status = AttemptStatus.FAILED
                attempt.error = f"{type(exc).__name__}: {exc}"
                attempt.finished_at = utc_now()
                self.store.save_attempt(attempt)
                outcome = JobOutcome(
                    False,
                    "retry",
                    "gpu_submission_failed",
                    {"error": attempt.error},
                )
                job.attempt = attempt.number
                self._apply_outcome(idea, job, attempt, outcome)
                continue
            if not decision.admitted:
                # Capacity rejection is not an attempt. Remove the provisional
                # durable row and directory so the same attempt number remains
                # available on the next scheduling tick.
                with self.store.connect() as conn:
                    conn.execute(
                        "DELETE FROM attempts WHERE attempt_id=?",
                        (attempt.attempt_id,),
                    )
                shutil.rmtree(
                    self.store.attempt_dir(attempt),
                    ignore_errors=True,
                )
                continue
            job.attempt = attempt.number
            job.status = JobStatus.RUNNING
            job.submitted_task_id = decision.task_id
            job.result = {
                "allocated_gpus": decision.allocated_gpus,
                "submitted_at": utc_now(),
                "task_id": decision.task_id,
                "output_dir": str(output_dir),
            }
            self.store.save_job(job)
            running_gpu += 1
            self.store.event(
                "gpu_job_submitted",
                idea_id=job.idea_id,
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
                allocated_gpus=decision.allocated_gpus,
                task_id=decision.task_id,
                output_dir=str(output_dir),
            )

        for job in cpu_ready:
            if running_cpu >= self.config.concurrency.max_cpu_jobs:
                break
            if job.kind in _LLM_KINDS:
                if running_llm >= self.config.concurrency.max_llm_jobs:
                    continue
                running_llm += 1
            idea = self.store.get_idea(job.idea_id)
            if idea is None:
                continue
            budget_reason = self._budget_block_reason(idea, job)
            if budget_reason:
                self._reject_budget(idea, job, budget_reason)
                continue
            attempt = self.store.create_attempt(job)
            attempt.status = AttemptStatus.RUNNING
            attempt.started_at = utc_now()
            self.store.save_attempt(attempt)
            job.status = JobStatus.RUNNING
            job.attempt = attempt.number
            job.attempt_id = attempt.attempt_id
            self.store.save_job(job)
            executor = self.executors.get(job.kind, self.default_executor)
            future = self._pool.submit(
                executor.execute,
                idea=idea,
                job=job,
                attempt=attempt,
                store=self.store,
            )
            self._running[job.job_id] = _Running(
                future=future,
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
            )
            running_cpu += 1
            self.store.event(
                "job_started",
                idea_id=idea.idea_id,
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
            )

    def _gpu_command(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        candidate: Path,
    ) -> tuple[str, Path]:
        build = json.loads(
            (candidate / "build.json").read_text(encoding="utf-8")
        )
        mode = "pilot" if job.kind is JobKind.PILOT else "scale"
        raw_command = str(build["commands"][mode]).strip()
        if not raw_command:
            raise ValueError(f"missing build command for {mode}")
        shared_root = Path(
            self.config.gpu.shared_workspace_root
        ).expanduser().resolve()
        output_dir = candidate / "artifacts" / mode
        if (
            not candidate.is_relative_to(shared_root)
            and self.gpu_broker is not None
            and self.gpu_broker.pool.__class__.__module__.startswith(
                "researchclaw.experiment.clusterbridge_pool"
            )
        ):
            raise ValueError(
                "GPU candidate must live on the configured shared Ceph "
                f"workspace: {candidate} is outside {shared_root}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        # Generated commands use paths relative to the project snapshot. Set
        # cwd explicitly, and pass the absolute artifact contract path.
        env = (
            f"AUTORESEARCH_V2_OUTPUT_DIR={shlex.quote(str(output_dir))} "
            f"AUTORESEARCH_V2_ATTEMPT_ID={shlex.quote(attempt.attempt_id)} "
            f"AUTORESEARCH_V2_IDEA_ID={shlex.quote(idea.idea_id)} "
        )
        command = (
            "set -euo pipefail; "
            f"cd {shlex.quote(str(candidate))}; "
            f"mkdir -p {shlex.quote(str(output_dir))}; "
            f"{env}bash -lc {shlex.quote(raw_command)}"
        )
        return command, output_dir

    def _collect_gpu_finished(self) -> None:
        if self.gpu_broker is None:
            return
        try:
            completed = self.gpu_broker.reconcile()
        except Exception as exc:  # noqa: BLE001
            self.store.event(
                "gpu_reconcile_failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        for job_id, result in completed:
            job = self.store.get_job(job_id)
            if job is None:
                continue
            idea = self.store.get_idea(job.idea_id)
            if idea is None:
                continue
            attempt = (
                self.store.get_attempt(job.attempt_id)
                if job.attempt_id
                else None
            )
            if attempt is None:
                attempt = self.store.create_attempt(job)
            attempt_dir = self.store.attempt_dir(attempt)
            (attempt_dir / "stdout.log").write_text(
                str(result.get("stdout", "") or ""),
                encoding="utf-8",
            )
            (attempt_dir / "stderr.log").write_text(
                str(result.get("stderr", "") or ""),
                encoding="utf-8",
            )
            returncode = int(result.get("returncode", -1))
            elapsed_sec = float(result.get("elapsed_sec", 0.0) or 0.0)
            allocated = int(
                result.get(
                    "allocated_gpus",
                    job.result.get("allocated_gpus", 0),
                )
                or 0
            )
            output_dir = (
                Path(job.expected_output_dir)
                if job.expected_output_dir
                else self.store.attempt_dir(attempt)
                / "candidate"
                / "artifacts"
                / (
                    "pilot"
                    if job.kind is JobKind.PILOT
                    else "scale"
                )
            )
            validation = validate_experiment_artifacts(output_dir)
            plan = self._read_current_json(idea.idea_id, "plan.json")
            pilot_runtime = (
                self._read_current_json(
                    idea.idea_id,
                    "artifacts/pilot/runtime_evidence.json",
                )
                if job.kind is JobKind.SCALE
                else None
            )
            contract_errors = validate_runtime_against_contract(
                plan=plan,
                runtime_evidence=validation.get(
                    "runtime_evidence",
                    {},
                ),
                allocated_gpus=allocated,
                mode=job.kind.value,
                pilot_runtime=pilot_runtime,
            )
            if contract_errors:
                validation["errors"].extend(contract_errors)
                validation["ok"] = False
            attempt.output_manifest = {
                **dict(result),
                "output_dir": str(output_dir),
                "files": validation.get("files", []),
            }
            attempt.validation = validation
            if returncode != 0 or not validation["ok"]:
                attempt.status = AttemptStatus.REJECTED
                attempt.error = (
                    f"returncode={returncode}; "
                    + "; ".join(validation.get("errors", []))
                )
                attempt.finished_at = utc_now()
                self.store.save_attempt(attempt)
                outcome = JobOutcome(
                    False,
                    "retry",
                    "gpu_experiment_invalid",
                    {
                        "returncode": returncode,
                        "validation": validation,
                        "pool_result": dict(result),
                    },
                    elapsed_sec=elapsed_sec,
                )
            else:
                gate = _experiment_gate(validation["metrics"])
                decision_gate = getattr(
                    self.executors.get(job.kind),
                    "decision_gate",
                    None,
                )
                gate_tokens = 0
                if decision_gate is not None:
                    verdict = decision_gate.review_experiment(
                        idea,
                        kind=job.kind,
                        plan=plan,
                        metrics=validation["metrics"],
                        runtime_evidence=validation["runtime_evidence"],
                    )
                    gate = {
                        "decision": verdict.decision,
                        "reason": verdict.reason,
                        "confidence": verdict.confidence,
                        "risks": list(verdict.risks),
                        "required_changes": list(
                            verdict.required_changes
                        ),
                    }
                    gate_tokens = verdict.tokens
                decision_path = (
                    output_dir / "decision_review.json"
                )
                decision_path.write_text(
                    json.dumps(
                        gate,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                attempt.validation["decision_gate"] = gate
                if gate["decision"] == "retry":
                    attempt.status = AttemptStatus.REJECTED
                    attempt.error = str(gate["reason"])
                    attempt.finished_at = utc_now()
                    self.store.save_attempt(attempt)
                    outcome = JobOutcome(
                        False,
                        "retry",
                        str(gate["reason"]),
                        {
                            "metrics": validation["metrics"],
                            "gate": gate,
                        },
                        tokens=gate_tokens,
                        elapsed_sec=elapsed_sec,
                    )
                else:
                    attempt.status = AttemptStatus.VALIDATING
                    attempt.finished_at = utc_now()
                    self.store.save_attempt(attempt)
                    self.store.commit_candidate(attempt)
                    outcome = JobOutcome(
                        True,
                        str(gate["decision"]),
                        str(gate["reason"]),
                        {
                            "metrics": validation["metrics"],
                            "runtime_evidence": validation[
                                "runtime_evidence"
                            ],
                            "gate": gate,
                            "pool_result": dict(result),
                        },
                        tokens=gate_tokens,
                        elapsed_sec=elapsed_sec,
                    )
            # Charge exact GPU seconds once. _apply_outcome receives zero GPU
            # elapsed to avoid double counting.
            idea.gpu_seconds_spent += allocated * max(0.0, elapsed_sec)
            outcome = JobOutcome(
                outcome.success,
                outcome.decision,
                outcome.reason,
                outcome.result,
                tokens=outcome.tokens,
                elapsed_sec=0.0,
            )
            self._apply_outcome(idea, job, attempt, outcome)

    def _collect_finished(self) -> None:
        for job_id, running in list(self._running.items()):
            if not running.future.done():
                continue
            job = self.store.get_job(job_id)
            attempt = self.store.get_attempt(running.attempt_id)
            if job is None or attempt is None:
                del self._running[job_id]
                continue
            idea = self.store.get_idea(job.idea_id)
            if idea is None:
                del self._running[job_id]
                continue
            try:
                outcome = running.future.result()
            except Exception as exc:  # noqa: BLE001
                outcome = JobOutcome(
                    False,
                    "retry",
                    f"{type(exc).__name__}: {exc}",
                    {},
                )
                attempt.status = AttemptStatus.FAILED
                attempt.error = outcome.reason
                attempt.finished_at = utc_now()
                self.store.save_attempt(attempt)
            self._apply_outcome(idea, job, attempt, outcome)
            del self._running[job_id]

    def _apply_outcome(
        self,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        outcome: JobOutcome,
    ) -> None:
        idea.llm_tokens_spent += max(0, int(outcome.tokens))
        if job.kind in _LLM_KINDS or outcome.tokens:
            idea.llm_calls += 1
        if job.requires_gpu and outcome.elapsed_sec:
            allocated = int(
                job.result.get(
                    "allocated_gpus",
                    max(job.min_gpus, job.preferred_gpus),
                )
                or 0
            )
            idea.gpu_seconds_spent += allocated * max(
                0.0, float(outcome.elapsed_sec)
            )
        job.result = {
            "decision": outcome.decision,
            "reason": outcome.reason,
            **outcome.result,
        }
        attempt.finished_at = attempt.finished_at or utc_now()
        if not outcome.success:
            if job.attempt < job.attempt_limit:
                job.status = JobStatus.RETRY_WAIT
                delay_sec = (
                    0.0
                    if self._simulation_mode
                    else min(
                        300.0,
                        2.0 ** max(0, job.attempt - 1)
                        + random.uniform(0.0, 1.0),
                    )
                )
                job.retry_not_before = (
                    datetime.now(UTC)
                    + timedelta(seconds=delay_sec)
                ).isoformat(timespec="milliseconds")
                idea.status = _STATUS_FOR_KIND[job.kind]
            else:
                job.status = JobStatus.FAILED
                idea.status = IdeaStatus.QUARANTINED
                idea.exit_reason = outcome.reason
                idea.current_job_id = ""
        else:
            job.status = JobStatus.SUCCEEDED
            job.retry_not_before = ""
            idea.current_job_id = ""
            if outcome.decision == "promote":
                idea.status = {
                    JobKind.DESIGN: IdeaStatus.BUILDING,
                    JobKind.BUILD: IdeaStatus.PILOTING,
                    JobKind.PILOT: IdeaStatus.SCALING,
                    JobKind.SCALE: IdeaStatus.REPORTING,
                    JobKind.REPORT: IdeaStatus.COMPLETED,
                }[job.kind]
            elif outcome.decision == "complete":
                idea.status = IdeaStatus.COMPLETED
                idea.exit_reason = outcome.reason
            elif outcome.decision == "complete_negative":
                # Valid negative evidence still receives a report.
                if job.kind in {JobKind.PILOT, JobKind.SCALE}:
                    idea.status = IdeaStatus.REPORTING
                    idea.candidate["final_outcome"] = "informative_negative"
                else:
                    idea.status = IdeaStatus.COMPLETED_NEGATIVE
                idea.exit_reason = outcome.reason
            elif outcome.decision == "reject":
                idea.status = IdeaStatus.REJECTED
                idea.exit_reason = outcome.reason
            else:
                idea.status = IdeaStatus.QUARANTINED
                idea.exit_reason = f"unknown decision: {outcome.decision}"
        idea.last_progress_at = utc_now()
        self.store.save_transition(
            idea=idea,
            job=job,
            attempt=attempt,
            event_type="job_finished",
            payload={
                "success": outcome.success,
                "decision": outcome.decision,
                "reason": outcome.reason,
                "next_status": idea.status.value,
            },
        )

    def _recover_interrupted_jobs(self) -> None:
        for job in self.store.list_jobs(statuses={JobStatus.RUNNING}):
            idea = self.store.get_idea(job.idea_id)
            if (
                job.requires_gpu
                and self.gpu_broker is not None
                and job.submitted_task_id
            ):
                self.gpu_broker.adopt(
                    job,
                    task_id=job.submitted_task_id,
                    allocated_gpus=int(
                        job.result.get("allocated_gpus", job.min_gpus) or 1
                    ),
                )
                self.store.event(
                    "gpu_job_adopted",
                    idea_id=job.idea_id,
                    job_id=job.job_id,
                    attempt_id=job.attempt_id,
                    task_id=job.submitted_task_id,
                )
                continue
            accepted = (
                self.store.get_attempt(job.attempt_id)
                if job.attempt_id
                else None
            )
            if (
                accepted is not None
                and accepted.status is AttemptStatus.ACCEPTED
                and idea is not None
            ):
                job.status = JobStatus.SUCCEEDED
                job.retry_not_before = ""
                idea.current_job_id = ""
                idea.status = {
                    JobKind.DESIGN: IdeaStatus.BUILDING,
                    JobKind.BUILD: IdeaStatus.PILOTING,
                    JobKind.PILOT: IdeaStatus.SCALING,
                    JobKind.SCALE: IdeaStatus.REPORTING,
                    JobKind.REPORT: IdeaStatus.COMPLETED,
                }[job.kind]
                idea.last_progress_at = utc_now()
                self.store.save_job(job)
                self.store.save_idea(idea)
                self.store.event(
                    "accepted_job_reconciled",
                    idea_id=job.idea_id,
                    job_id=job.job_id,
                    attempt_id=accepted.attempt_id,
                    status=idea.status.value,
                )
                continue
            job.status = (
                JobStatus.RETRY_WAIT
                if job.attempt < job.attempt_limit
                else JobStatus.FAILED
            )
            if job.status is JobStatus.RETRY_WAIT:
                job.retry_not_before = datetime.now(UTC).isoformat(
                    timespec="milliseconds"
                )
            self.store.save_job(job)
            if job.attempt_id:
                attempt = self.store.get_attempt(job.attempt_id)
                if (
                    attempt is not None
                    and attempt.status
                    in {
                        AttemptStatus.CREATED,
                        AttemptStatus.RUNNING,
                        AttemptStatus.VALIDATING,
                    }
                ):
                    attempt.status = AttemptStatus.FAILED
                    attempt.error = "controller_interrupted"
                    attempt.finished_at = utc_now()
                    self.store.save_attempt(attempt)
            if idea is not None:
                idea.status = (
                    _STATUS_FOR_KIND[job.kind]
                    if job.status is JobStatus.RETRY_WAIT
                    else IdeaStatus.QUARANTINED
                )
                if job.status is JobStatus.FAILED:
                    idea.current_job_id = ""
                    idea.exit_reason = "controller_interrupted_attempt_limit"
                self.store.save_idea(idea)
            self.store.event(
                "interrupted_job_recovered",
                idea_id=job.idea_id,
                job_id=job.job_id,
                status=job.status.value,
            )

    def _budget_block_reason(
        self,
        idea: IdeaRecord,
        job: JobRecord,
    ) -> str:
        if (
            idea.llm_tokens_spent
            >= self.config.budgets.max_llm_tokens_per_idea
            and job.kind in _LLM_KINDS
        ):
                return "llm_token_budget_exhausted"
        if job.requires_gpu:
            budget_hours = (
                self.config.budgets.pilot_gpu_hours
                if job.kind is JobKind.PILOT
                else self.config.budgets.scale_gpu_hours
            )
            # GPU accounting is GPU-seconds, so the configured budget is also
            # interpreted as GPU-hours rather than wall-clock hours.
            budget_seconds = budget_hours * 3600.0
            if idea.gpu_seconds_spent >= budget_seconds:
                return f"{job.kind.value}_gpu_budget_exhausted"
            remaining = budget_seconds - idea.gpu_seconds_spent
            projected = max(1, job.min_gpus) * job.timeout_sec
            if projected > remaining + 1e-9:
                # Shrink timeout to the exact remaining GPU-seconds when a
                # useful bounded run still fits.
                job.timeout_sec = max(
                    60.0,
                    remaining / max(1, job.preferred_gpus),
                )
                if (
                    max(1, job.preferred_gpus) * job.timeout_sec
                    > remaining + 1e-9
                ):
                    return f"{job.kind.value}_gpu_budget_insufficient"
                self.store.save_job(job)
        created = _parse_time(idea.created_at)
        if created is not None:
            age_hours = (
                datetime.now(UTC) - created
            ).total_seconds() / 3600.0
            if age_hours >= (
                self.config.budgets.max_wall_clock_hours_per_idea
            ):
                return "wall_clock_budget_exhausted"
        return ""

    def _reject_budget(
        self,
        idea: IdeaRecord,
        job: JobRecord,
        reason: str,
    ) -> None:
        job.status = JobStatus.CANCELLED
        job.result = {"decision": "reject", "reason": reason}
        idea.status = IdeaStatus.REJECTED
        idea.exit_reason = reason
        idea.current_job_id = ""
        self.store.save_job(job)
        self.store.save_idea(idea)
        self.store.event(
            "budget_gate_rejected",
            idea_id=idea.idea_id,
            job_id=job.job_id,
            reason=reason,
        )

    def _enforce_liveness_budgets(self) -> None:
        now = datetime.now(UTC)
        for idea in self.store.list_ideas(statuses=set(ACTIVE_IDEA_STATUSES)):
            last = _parse_time(idea.last_progress_at)
            if last is None:
                continue
            stagnant_hours = (now - last).total_seconds() / 3600.0
            if stagnant_hours < self.config.budgets.max_no_progress_hours:
                continue
            job = (
                self.store.get_job(idea.current_job_id)
                if idea.current_job_id
                else None
            )
            if job is not None and job.status is JobStatus.RUNNING:
                if job.requires_gpu and self.gpu_broker is not None:
                    self.gpu_broker.cancel(job.job_id)
                job.status = JobStatus.CANCELLED
                job.result = {
                    "decision": "reject",
                    "reason": "no_progress_timeout",
                }
                self.store.save_job(job)
            idea.status = IdeaStatus.QUARANTINED
            idea.exit_reason = "no_progress_timeout"
            idea.current_job_id = ""
            self.store.save_idea(idea)
            self.store.event(
                "idea_quarantined_no_progress",
                idea_id=idea.idea_id,
                stagnant_hours=stagnant_hours,
            )

    def snapshot(self) -> dict[str, Any]:
        ideas = self.store.list_ideas()
        jobs = self.store.list_jobs()
        ideas_by_status: dict[str, int] = {}
        jobs_by_status: dict[str, int] = {}
        for idea in ideas:
            ideas_by_status[idea.status.value] = (
                ideas_by_status.get(idea.status.value, 0) + 1
            )
        for job in jobs:
            jobs_by_status[job.status.value] = (
                jobs_by_status.get(job.status.value, 0) + 1
            )
        pending_gpu = sum(
            1
            for job in jobs
            if job.requires_gpu
            and job.status in {JobStatus.READY, JobStatus.RETRY_WAIT}
        )
        gpu = (
            self.gpu_broker.snapshot(pending_jobs=pending_gpu)
            if self.gpu_broker is not None
            else {
                "total_gpus": 0,
                "allocated_gpus": 0,
                "available_gpus": 0,
                "utilization": 0.0,
                "target_utilization": self.config.gpu.target_utilization,
                "pending_jobs": pending_gpu,
                "leases": [],
            }
        )
        return {
            "timestamp": utc_now(),
            "status": (
                "stopping"
                if self._stop
                else "paused"
                if self.store.control_requested("pause")
                else "running"
            ),
            "ideas_total": len(ideas),
            "ideas_by_status": ideas_by_status,
            "jobs_total": len(jobs),
            "jobs_by_status": jobs_by_status,
            "running_futures": len(self._running),
            "running_gpu_jobs": len(gpu.get("leases", [])),
            "llm_tokens_total": sum(
                idea.llm_tokens_spent for idea in ideas
            ),
            "gpu_hours_total": sum(
                idea.gpu_seconds_spent for idea in ideas
            )
            / 3600.0,
            "gpu": gpu,
        }

    def _read_current_json(
        self,
        idea_id: str,
        filename: str,
    ) -> dict[str, Any]:
        root = self.store.current_dir(idea_id).resolve()
        path = (root / filename).resolve()
        if not path.is_relative_to(root):
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return dict(value) if isinstance(value, Mapping) else {}


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class _MissingExecutor:
    def execute(self, **kwargs: Any) -> JobOutcome:
        job = kwargs.get("job")
        kind = getattr(job, "kind", "unknown")
        raise RuntimeError(f"no production executor configured for {kind}")


class _ControllerThreadPool(concurrent.futures.ThreadPoolExecutor):
    """Compatibility shim: direct pool shutdown also releases writer lock."""

    def __init__(self, *args: Any, on_shutdown: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._on_shutdown = on_shutdown

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        try:
            super().shutdown(
                wait=wait,
                cancel_futures=cancel_futures,
            )
        finally:
            self._on_shutdown()
