"""Steady-state multi-Idea controller for AutoResearch v2."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import random
import shlex
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .attestation import (
    canonical_json_sha256,
    create_execution_attestation,
    create_execution_contract,
    verify_execution_attestation,
    verify_execution_contract,
    write_execution_attestation,
    write_execution_contract,
)
from .config import V2Config
from .gpu import GPUBroker
from .ideas import IdeaAdmission, IdeaGenerator
from .jobs import (
    JobExecutor,
    JobOutcome,
    SimulatedJobExecutor,
    _experiment_gate,
    resolve_experiment_lifecycle_gate,
)
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
from .research_memory import ResearchMemory
from .store import V2Store
from .validation import (
    validate_execution_argv,
    validate_experiment_artifacts,
    validate_research_implementation,
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


@dataclass(slots=True)
class _IdeaGeneration:
    future: concurrent.futures.Future[list[IdeaRecord]]
    requested: int


@dataclass(slots=True)
class _ResearchMemorySync:
    future: concurrent.futures.Future[None]


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
        configured_gpu_capacity: int = 0,
        research_memory: ResearchMemory | None = None,
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
        self.research_memory = research_memory
        self.configured_gpu_capacity = max(
            0,
            int(configured_gpu_capacity),
        )
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
        # Idea generation performs several InfoHub requests and two LLM calls.
        # It must never block the scheduling/heartbeat thread or occupy a Job
        # executor worker.  The single worker also guarantees one portfolio
        # board is generated at a time.
        self._idea_pool = _ControllerThreadPool(
            max_workers=1,
            thread_name_prefix="autoresearch-v2-ideas",
            on_shutdown=lambda: None,
        )
        self._research_memory_pool = _ControllerThreadPool(
            max_workers=1,
            thread_name_prefix="autoresearch-v2-memory",
            on_shutdown=lambda: None,
        )
        self._running: dict[str, _Running] = {}
        self._idea_generation: _IdeaGeneration | None = None
        self._research_memory_sync: _ResearchMemorySync | None = None
        self._idea_generation_failures = 0
        self._idea_generation_retry_not_before = 0.0
        self._research_memory_fingerprints: dict[str, str] = {}
        self._stop = False
        self._stop_reason = ""
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
        self._research_memory_pool.shutdown(
            wait=True,
            cancel_futures=False,
        )
        self._idea_pool.shutdown(wait=True, cancel_futures=False)
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
            # durable state before the process exits. A service/terminal stop
            # must not wait for long, non-cancellable CLI-backed LLM calls:
            # leave those jobs durable as RUNNING and let startup recovery
            # refund/requeue them.
            if self._stop:
                self._idea_pool.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
                self._research_memory_pool.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
                self._pool.shutdown(wait=False, cancel_futures=True)
                # ThreadPoolExecutor registers a private atexit hook that joins
                # every worker even after shutdown(wait=False). Service stops
                # intentionally recover durable RUNNING jobs on the next
                # process, so these non-cancellable CLI-backed workers must be
                # removed from that interpreter-exit join registry.
                self._pool.detach_workers_for_process_exit()
                self._idea_pool.detach_workers_for_process_exit()
                self._research_memory_pool.detach_workers_for_process_exit()
                # Do not synchronously probe or stop a remote pool from a
                # POSIX-signal path. Any submitted GPU tasks remain durable and
                # are adopted by startup recovery; the process exiting stops
                # its daemon heartbeat/keepalive threads.
                self.store.release_writer_lock()
                self._initialized = False
            else:
                self._drain_local_futures()
                self.close()
        return ticks

    def _drain_local_futures(self) -> None:
        # A bounded CLI run is a bounded scheduler probe, not an implicit
        # request to finish every downstream stage. Drain only work that was
        # already submitted by the final tick. The next invocation resumes the
        # durable state and schedules subsequent jobs.
        while self._running or self._idea_generation is not None:
            futures = [
                entry.future for entry in self._running.values()
            ]
            if self._idea_generation is not None:
                futures.append(self._idea_generation.future)
            concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            self._collect_finished()
            self._collect_idea_generation()

    def request_stop(self, reason: str = "requested") -> None:
        if self._stop:
            return
        self._stop = True
        self._stop_reason = str(reason or "requested")
        if self._initialized:
            self.store.event(
                "controller_stop_requested",
                reason=self._stop_reason,
                running_futures=len(self._running),
                idea_generation_running=(
                    self._idea_generation is not None
                ),
                running_gpu_jobs=(
                    len(self.gpu_broker.leases)
                    if self.gpu_broker is not None
                    else 0
                ),
            )

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    def tick(self) -> dict[str, Any]:
        self._tick_count += 1
        if self.store.control_requested("stop"):
            self.request_stop(reason="control_stop")
        self._collect_finished()
        self._collect_idea_generation()
        self._collect_research_memory_sync()
        self._collect_gpu_finished()
        self._enforce_liveness_budgets()
        if not self.store.control_requested("pause") and not self._stop:
            # Static generators are test/simulation fixtures and complete
            # synchronously.  Production generation runs after dispatch so
            # ready scientific work always has first claim on LLM capacity.
            if self._simulation_mode:
                self._maintain_reservoir()
            self._admit_reservoir()
            self._ensure_jobs()
            self._dispatch()
            if not self._simulation_mode:
                self._maintain_reservoir()
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
        if (
            self.research_memory is not None
            and self._tick_count
            % self.config.research_memory.reconcile_interval_ticks
            == 0
        ):
            self._start_research_memory_sync()
        snapshot = self.snapshot()
        self.store.event("controller_tick", **snapshot)
        return snapshot

    def _start_research_memory_sync(self) -> None:
        if self._research_memory_sync is not None:
            return
        self._research_memory_sync = _ResearchMemorySync(
            future=self._research_memory_pool.submit(
                self._reconcile_research_memory
            )
        )

    def _collect_research_memory_sync(self) -> None:
        running = self._research_memory_sync
        if running is None or not running.future.done():
            return
        self._research_memory_sync = None
        try:
            running.future.result()
        except Exception as exc:  # noqa: BLE001
            self.store.event(
                "research_memory_reconcile_failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _reconcile_research_memory(self) -> None:
        """Best-effort InfoHub sync; never block the scientific state machine."""

        for idea in self.store.list_ideas():
            fingerprint = json.dumps(
                {
                    "status": idea.status.value,
                    "updated_at": idea.updated_at,
                    "current_job_id": idea.current_job_id,
                    "exit_reason": idea.exit_reason,
                    "final_outcome": idea.candidate.get(
                        "final_outcome",
                        "",
                    ),
                    "llm_tokens": idea.llm_tokens_spent,
                    "llm_calls": idea.llm_calls,
                    "gpu_seconds": round(idea.gpu_seconds_spent, 6),
                },
                sort_keys=True,
            )
            if self._research_memory_fingerprints.get(
                idea.idea_id
            ) == fingerprint:
                continue
            result = self.research_memory.reconcile(idea)
            if result.ok:
                self._research_memory_fingerprints[
                    idea.idea_id
                ] = fingerprint
                self.store.event(
                    "research_memory_synced",
                    idea_id=idea.idea_id,
                    external_id=result.external_id,
                    status=idea.status.value,
                )
            else:
                self.store.event(
                    "research_memory_sync_failed",
                    idea_id=idea.idea_id,
                    external_id=result.external_id,
                    error=result.error,
                )

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
        return (
            not active
            and not ready
            and not self._running
            and self._idea_generation is None
            and not gpu_running
        )

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
        if self._simulation_mode:
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
            self._apply_generated_ideas(
                generated,
                requested=needed,
            )
            return
        if self._idea_generation is not None:
            return
        if time.monotonic() < self._idea_generation_retry_not_before:
            return
        if self._running_llm_count() >= self.config.concurrency.max_llm_jobs:
            return
        existing = tuple(ideas)
        future = self._idea_pool.submit(
            self.generator.generate,
            count=needed,
            existing=existing,
        )
        self._idea_generation = _IdeaGeneration(
            future=future,
            requested=needed,
        )
        self.store.event(
            "idea_generation_started",
            requested=needed,
            reservoir_size=len(reservoir),
        )

    def _collect_idea_generation(self) -> None:
        running = self._idea_generation
        if running is None or not running.future.done():
            return
        self._idea_generation = None
        try:
            generated = running.future.result()
        except Exception as exc:  # noqa: BLE001
            self._idea_generation_failures += 1
            delay_sec = min(
                300.0,
                30.0 * (2 ** (self._idea_generation_failures - 1)),
            )
            self._idea_generation_retry_not_before = (
                time.monotonic() + delay_sec
            )
            self.store.event(
                "idea_generation_failed",
                requested=running.requested,
                error=f"{type(exc).__name__}: {exc}",
                retry_in_sec=delay_sec,
            )
            return
        result = self._apply_generated_ideas(
            generated,
            requested=running.requested,
        )
        if result["reservoir_added"] > 0:
            self._idea_generation_failures = 0
            self._idea_generation_retry_not_before = 0.0
            return
        self._idea_generation_failures += 1
        delay_sec = min(
            300.0,
            30.0 * (2 ** (self._idea_generation_failures - 1)),
        )
        self._idea_generation_retry_not_before = time.monotonic() + delay_sec

    def _apply_generated_ideas(
        self,
        generated: list[IdeaRecord],
        *,
        requested: int,
    ) -> dict[str, int]:
        ideas = self.store.list_ideas()
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
            requested=requested,
            generated=len(generated),
            reservoir_added=added,
            rejected=rejected,
            invalid_generated=len(
                getattr(self.generator, "last_rejections", ())
            ),
        )
        return {
            "generated": len(generated),
            "reservoir_added": added,
            "rejected": rejected,
        }

    def _running_llm_count(self) -> int:
        local = sum(
            1
            for entry in self._running.values()
            if (
                (job := self.store.get_job(entry.job_id)) is not None
                and job.kind in _LLM_KINDS
            )
        ) + int(self._idea_generation is not None)
        if self._simulation_mode:
            return local
        # A service restart deliberately detaches non-cancellable CLI-backed
        # LLM calls and refunds their durable Jobs. Those old bridge requests
        # can still be consuming the shared local model gateway for several
        # minutes, so count live gateway children as well as this process's
        # in-memory futures before dispatching replacements.
        return max(local, _live_model_gateway_calls())

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
            remote_smoke = bool(
                kind is JobKind.PILOT
                and self.config.gpu.enabled
                and self.config.execution.smoke_environment
                in {"auto", "gpu_pool"}
                and not self._has_verified_remote_smoke(idea.idea_id)
            )
            dispatch_kind = JobKind.BUILD if remote_smoke else kind
            base = f"{idea.idea_id}-{kind.value}"
            index = 1
            job_id = base
            while job_id in existing:
                index += 1
                job_id = f"{base}-{index:02d}"
            compute = idea.candidate.get("compute", {})
            requested_gpus = 0
            if remote_smoke:
                requested_gpus = 1
            elif kind in {JobKind.PILOT, JobKind.SCALE}:
                try:
                    requested_gpus = max(
                        1, int(compute.get("gpu_count", 1))
                    )
                except (AttributeError, TypeError, ValueError):
                    requested_gpus = 1
            max_gpus = (
                1
                if remote_smoke
                else
                self.config.gpu.pilot_max_gpus
                if kind is JobKind.PILOT
                else self.config.gpu.scale_max_gpus
            )
            job = JobRecord(
                job_id=job_id,
                idea_id=idea.idea_id,
                kind=dispatch_kind,
                attempt_limit=(
                    self.config.budgets.max_build_attempts
                    if dispatch_kind is JobKind.BUILD
                    else 1
                    if dispatch_kind is JobKind.DESIGN
                    else 1
                    if dispatch_kind is JobKind.REPORT
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
                timeout_sec=(
                    self.config.execution.smoke_timeout_sec
                    if remote_smoke
                    else self._job_timeout(idea, kind)
                ),
                result=self._initial_job_result(
                    idea=idea,
                    kind=kind,
                    remote_smoke=remote_smoke,
                ),
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
                kind=dispatch_kind.value,
                remote_smoke=remote_smoke,
            )

    @staticmethod
    def _initial_job_result(
        *,
        idea: IdeaRecord,
        kind: JobKind,
        remote_smoke: bool,
    ) -> dict[str, Any]:
        if remote_smoke:
            return {
                "remote_smoke": True,
                "next_kind": JobKind.PILOT.value,
            }
        if kind is JobKind.BUILD:
            feedback = idea.candidate.pop(
                "_autoresearch_v2_build_repair_feedback",
                None,
            )
            if isinstance(feedback, Mapping):
                return {"remote_smoke_repair": dict(feedback)}
        return {}

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

    @staticmethod
    def _is_remote_smoke_job(job: JobRecord) -> bool:
        return bool(job.result.get("remote_smoke"))

    def _has_verified_remote_smoke(self, idea_id: str) -> bool:
        for job in self.store.list_jobs():
            if (
                job.idea_id != idea_id
                or not self._is_remote_smoke_job(job)
                or job.status is not JobStatus.SUCCEEDED
                or not job.attempt_id
            ):
                continue
            attempt = self.store.get_attempt(job.attempt_id)
            if (
                attempt is None
                or attempt.status is not AttemptStatus.ACCEPTED
                or attempt.validation.get("ok") is not True
            ):
                continue
            remote = attempt.validation.get("remote_smoke", {})
            if not isinstance(remote, Mapping):
                continue
            if remote.get("verified") is not True:
                continue
            if not str(remote.get("attestation_sha256", "")).strip():
                continue
            return True
        return False

    def _dispatch(self) -> None:
        running_llm = self._running_llm_count()
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
            if self._block_unchanged_gpu_implementation(
                idea=idea,
                job=job,
            ):
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
                if self._is_remote_smoke_job(job):
                    command, output_dir = self._gpu_smoke_command(
                        idea=idea,
                        job=job,
                        attempt=attempt,
                        candidate=candidate,
                    )
                else:
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
                if (
                    self._is_remote_smoke_job(job)
                    and "Build-to-Runtime contract" in attempt.error
                ):
                    self._queue_build_repair(
                        idea=idea,
                        job=job,
                        attempt=attempt,
                        reason="remote_smoke_preflight_invalid",
                        diagnostics={"error": attempt.error},
                    )
                    continue
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
            prior_result = dict(job.result)
            job.result = {
                **prior_result,
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
        mode = "pilot" if job.kind is JobKind.PILOT else "scale"
        return self._trusted_gpu_command(
            idea=idea,
            job=job,
            attempt=attempt,
            candidate=candidate,
            mode=mode,
            max_gpus=job.max_gpus,
            preferred_gpus=job.preferred_gpus,
        )

    def _gpu_smoke_command(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        candidate: Path,
    ) -> tuple[str, Path]:
        return self._trusted_gpu_command(
            idea=idea,
            job=job,
            attempt=attempt,
            candidate=candidate,
            mode="smoke",
            max_gpus=1,
            preferred_gpus=1,
        )

    def _trusted_gpu_command(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        candidate: Path,
        mode: str,
        max_gpus: int,
        preferred_gpus: int,
    ) -> tuple[str, Path]:
        build_path = candidate / "build.json"
        build = json.loads(build_path.read_text(encoding="utf-8"))
        if not self._simulation_mode:
            plan = json.loads(
                (candidate / "plan.json").read_text(encoding="utf-8")
            )
            implementation = validate_research_implementation(
                candidate,
                plan=plan,
                controller_runtime=bool(
                    build.get("controller_runtime")
                ),
            )
            if not implementation.get("ok"):
                raise ValueError(
                    "generated implementation violates the "
                    "Build-to-Runtime contract: "
                    + "; ".join(implementation.get("errors", []))
                )
        raw_argv = build["commands"][mode]
        if isinstance(raw_argv, str):
            argv = shlex.split(raw_argv)
        elif isinstance(raw_argv, list) and all(
            isinstance(item, str) for item in raw_argv
        ):
            argv = list(raw_argv)
        else:
            argv = []
        command_errors = validate_execution_argv(
            argv,
            path=f"commands.{mode}",
        )
        if command_errors:
            raise ValueError("; ".join(command_errors))
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
        execution_argv = [
            self.config.execution.python_executable,
            *argv[1:],
        ]
        bound_build_path = build_path
        if mode == "smoke":
            provenance = candidate / "trusted" / "remote_smoke"
            provenance.mkdir(parents=True, exist_ok=True)
            bound_build_path = provenance / "executed_build.json"
            shutil.copy2(build_path, bound_build_path)
        contract = create_execution_contract(
            idea_id=idea.idea_id,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            mode=mode,
            argv=execution_argv,
            cwd=candidate,
            entrypoint=argv[1],
            output_dir=output_dir,
            resource_limits={
                "min_gpus": (
                    1
                    if mode == "smoke"
                    else job.min_gpus
                ),
                "max_gpus": max_gpus,
                "preferred_gpus": preferred_gpus,
                "timeout_sec": job.timeout_sec,
            },
            plan_path=candidate / "plan.json",
            build_path=bound_build_path,
            allowed_env_keys=self.config.execution.allowed_env_keys,
        )
        contract_path = self.store.attempt_dir(attempt) / (
            "execution_contract.json"
        )
        write_execution_contract(contract_path, contract)
        contract_hash = canonical_json_sha256(contract)
        job.result = {
            **job.result,
            "execution_contract_path": str(contract_path),
            "execution_contract_sha256": contract_hash,
            "execution_mode": mode,
        }
        self.store.save_job(job)
        runner = (
            "import json, os, subprocess, sys\n"
            "contract=json.load(open(sys.argv[1], encoding='utf-8'))\n"
            "candidate=sys.argv[2]\n"
            "allowed=set(contract['allowed_env_keys'])\n"
            "env={k:v for k,v in os.environ.items() if k in allowed}\n"
            "for key in ('PATH','HOME','LANG','LC_ALL','LD_LIBRARY_PATH',"
            "'PYTHONPATH','HF_HOME','TORCH_HOME','CUDA_HOME',"
            "'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER',"
            "'NVIDIA_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'):\n"
            "    if key in os.environ: env[key]=os.environ[key]\n"
            "expected=int(env.get('AUTORESEARCH_V2_GPU_COUNT','0'))\n"
            "visible=env.get('CUDA_VISIBLE_DEVICES','')\n"
            "if expected > 0:\n"
            "    devices=[x for x in visible.split(',') if x.strip()]\n"
            "    if len(devices) != expected:\n"
            "        raise SystemExit('trusted GPU visibility mismatch: "
            "expected=%d visible=%r' % (expected, visible))\n"
            "env['PYTHONUNBUFFERED']='1'\n"
            "env['TOKENIZERS_PARALLELISM']='false'\n"
            "raise SystemExit(subprocess.run("
            "contract['argv'], cwd=candidate, env=env, shell=False).returncode)"
        )
        command = (
            "set -euo pipefail; "
            f"mkdir -p {shlex.quote(str(output_dir))}; "
            f"{shlex.quote(self.config.execution.python_executable)} "
            f"-c {shlex.quote(runner)} "
            f"{shlex.quote(str(contract_path))} "
            f"{shlex.quote(str(candidate))}"
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
            stdout_path = attempt_dir / "stdout.log"
            stderr_path = attempt_dir / "stderr.log"
            stdout_path.write_text(
                str(result.get("stdout", "") or ""),
                encoding="utf-8",
            )
            stderr_path.write_text(
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
                    "smoke"
                    if self._is_remote_smoke_job(job)
                    else "pilot"
                    if job.kind is JobKind.PILOT
                    else "scale"
                )
            )
            if self._is_remote_smoke_job(job):
                self._complete_remote_smoke(
                    idea=idea,
                    job=job,
                    attempt=attempt,
                    result=result,
                    output_dir=output_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    returncode=returncode,
                    allocated_gpus=allocated,
                    elapsed_sec=elapsed_sec,
                )
                continue
            plan = self._read_current_json(idea.idea_id, "plan.json")
            source_compiler = self._candidate_runtime_compiler(
                output_dir.parent.parent
            )
            active_compiler = self._runtime_compiler_identity()
            if (
                source_compiler
                and source_compiler != active_compiler
            ):
                wrapper_path = (
                    output_dir.parent.parent
                    / "_autoresearch_runtime.py"
                )
                from .runtime_wrapper import wrapper_source

                wrapper_path.write_text(
                    wrapper_source(),
                    encoding="utf-8",
                )
                self.store.event(
                    "runtime_wrapper_refreshed",
                    idea_id=idea.idea_id,
                    job_id=job.job_id,
                    attempt_id=attempt.attempt_id,
                    previous=source_compiler,
                    current=active_compiler,
                )
            wrapper = (
                {"compiled": False, "simulation": True}
                if self._simulation_mode
                else self._normalize_controller_runtime_artifacts(
                    output_dir=output_dir,
                    plan=plan,
                    mode=job.kind.value,
                    allocated_gpus=allocated,
                    returncode=returncode,
                )
            )
            validation = validate_experiment_artifacts(output_dir)
            if wrapper.get("error"):
                validation["errors"] = [
                    *validation.get("errors", []),
                    str(wrapper["error"]),
                ]
                validation["ok"] = False
            if self._simulation_mode:
                validation["execution_attestation"] = {
                    "simulation": True
                }
            else:
                attestation = self._attest_gpu_execution(
                    idea=idea,
                    job=job,
                    attempt=attempt,
                    output_dir=output_dir,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    returncode=returncode,
                    allocated_gpus=allocated,
                    elapsed_sec=elapsed_sec,
                )
                if attestation.get("errors"):
                    validation["errors"].extend(attestation["errors"])
                    validation["ok"] = False
                else:
                    validation["execution_attestation"] = attestation
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
                raw_artifacts_present = (
                    (output_dir / "_raw" / "metrics.json").is_file()
                    and (
                        output_dir
                        / "_raw"
                        / "runtime_evidence.json"
                    ).is_file()
                )
                deterministic_contract_failure = bool(
                    raw_artifacts_present
                    and (wrapper.get("error") or contract_errors)
                )
                if deterministic_contract_failure:
                    diagnostics = {
                        "failure_class": "runtime_contract_invalid",
                        "failure_code": (
                            f"{job.kind.value}_runtime_contract_invalid"
                        ),
                        "source_stage": job.kind.value,
                        "returncode": returncode,
                        "errors": list(validation.get("errors", [])),
                        **self._implementation_failure_fingerprint(
                            idea_id=idea.idea_id,
                            errors=list(validation.get("errors", [])),
                        ),
                    }
                    idea.candidate[
                        "_autoresearch_v2_last_implementation_failure"
                    ] = diagnostics
                    self._queue_build_repair(
                        idea=idea,
                        job=job,
                        attempt=attempt,
                        reason=(
                            f"{job.kind.value}_runtime_contract_invalid"
                        ),
                        diagnostics=diagnostics,
                    )
                    continue
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
                gate = resolve_experiment_lifecycle_gate(
                    runtime_evidence=validation["runtime_evidence"],
                    gate=gate,
                )
                decision_path = attempt_dir / "decision_review.json"
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

    def _complete_remote_smoke(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        result: Mapping[str, Any],
        output_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
        returncode: int,
        allocated_gpus: int,
        elapsed_sec: float,
    ) -> None:
        plan = self._read_current_json(idea.idea_id, "plan.json")
        wrapper = self._normalize_controller_runtime_artifacts(
            output_dir=output_dir,
            plan=plan,
            mode="smoke",
            allocated_gpus=allocated_gpus,
            returncode=returncode,
        )
        smoke_artifacts = validate_experiment_artifacts(output_dir)
        if wrapper.get("error"):
            smoke_artifacts["errors"] = [
                *smoke_artifacts.get("errors", []),
                str(wrapper["error"]),
            ]
            smoke_artifacts["ok"] = False
        trusted_evidence = result.get("trusted_gpu_evidence")
        if not isinstance(trusted_evidence, Mapping):
            trusted_evidence = {}
        trusted_evidence_path = (
            self.store.attempt_dir(attempt)
            / "trusted"
            / "remote_smoke"
            / "trusted_gpu_evidence.json"
        )
        trusted_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        trusted_evidence_path.write_text(
            json.dumps(
                dict(trusted_evidence),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        attestation = self._attest_gpu_execution(
            idea=idea,
            job=job,
            attempt=attempt,
            output_dir=output_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            returncode=returncode,
            allocated_gpus=allocated_gpus,
            elapsed_sec=elapsed_sec,
        )
        evidence = smoke_artifacts.get("runtime_evidence", {})
        smoke_errors = list(smoke_artifacts.get("errors", []))
        if not isinstance(evidence, Mapping):
            evidence = {}
        if int(evidence.get("examples_processed", 0) or 0) <= 0:
            smoke_errors.append(
                "remote smoke must process at least one real benchmark example"
            )
        if int(evidence.get("gpu_count", 0) or 0) != allocated_gpus:
            smoke_errors.append(
                "remote smoke runtime gpu_count must match allocated_gpus"
            )
        smoke_errors.extend(
            validate_runtime_against_contract(
                plan=plan,
                runtime_evidence=dict(evidence),
                allocated_gpus=allocated_gpus,
                mode="smoke",
            )
        )
        try:
            trusted_allocated = int(
                trusted_evidence.get("allocated_gpus", -1)
            )
        except (TypeError, ValueError):
            trusted_allocated = -1
        visible = str(
            trusted_evidence.get("cuda_visible_devices", "") or ""
        )
        visible_devices = [
            item.strip() for item in visible.split(",") if item.strip()
        ]
        gpu_uuids = trusted_evidence.get("gpu_uuids")
        if trusted_allocated != allocated_gpus:
            smoke_errors.append(
                "trusted GPU evidence allocation must match allocated_gpus"
            )
        if len(visible_devices) != allocated_gpus:
            smoke_errors.append(
                "trusted GPU evidence CUDA visibility must match allocation"
            )
        if (
            not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != allocated_gpus
            or any(not str(item).strip() for item in gpu_uuids)
        ):
            smoke_errors.append(
                "trusted GPU evidence must identify every allocated GPU UUID"
            )
        if not str(
            trusted_evidence.get("ray_task_id", "") or ""
        ).strip() or not str(
            trusted_evidence.get("ray_node_id", "") or ""
        ).strip():
            smoke_errors.append(
                "trusted GPU evidence must record Ray task and node identity"
            )
        try:
            trusted_returncode = int(
                trusted_evidence.get("returncode", -1)
            )
        except (TypeError, ValueError):
            trusted_returncode = -1
        if trusted_returncode != returncode:
            smoke_errors.append(
                "trusted GPU evidence returncode disagrees with pool result"
            )
        if float(
            trusted_evidence.get("peak_gpu_memory_mb", 0.0) or 0.0
        ) <= 0:
            smoke_errors.append(
                "trusted GPU evidence must record non-zero GPU memory usage"
            )
        if float(
            trusted_evidence.get(
                "peak_gpu_utilization_percent",
                0.0,
            )
            or 0.0
        ) <= 0:
            smoke_errors.append(
                "trusted GPU evidence must record non-zero GPU utilization"
            )
        smoke_errors.extend(attestation.get("errors", []))
        validation: dict[str, Any] = {
            "ok": returncode == 0 and not smoke_errors,
            "remote_smoke": {
                "executed": True,
                "environment": "gpu_pool",
                "returncode": returncode,
                "allocated_gpus": allocated_gpus,
                "runtime_evidence": dict(evidence),
                "trusted_gpu_evidence": dict(trusted_evidence),
                "trusted_gpu_evidence_path": str(trusted_evidence_path),
                "execution_attestation": attestation,
            },
            "metrics": smoke_artifacts.get("metrics", {}),
            "runtime_evidence": dict(evidence),
            "errors": smoke_errors,
        }
        attempt.output_manifest = {
            **dict(result),
            "output_dir": str(output_dir),
        }
        attempt.validation = validation
        if validation["ok"]:
            remote_smoke = {
                "ok": True,
                "verified": True,
                "attempt_id": attempt.attempt_id,
                "contract_path": attestation.get("contract_path", ""),
                "attestation_path": attestation.get("path", ""),
                "attestation_sha256": attestation.get("sha256", ""),
                "runtime_evidence": dict(evidence),
                "trusted_gpu_evidence": dict(trusted_evidence),
                "trusted_gpu_evidence_path": str(trusted_evidence_path),
            }
            attempt.validation["remote_smoke"].update(remote_smoke)
            attempt.status = AttemptStatus.VALIDATING
            attempt.finished_at = utc_now()
            self.store.save_attempt(attempt)
            self.store.commit_candidate(attempt)
            outcome = JobOutcome(
                True,
                "promote",
                "remote_smoke_accepted",
                {
                    "remote_smoke": remote_smoke,
                    "pool_result": dict(result),
                },
                elapsed_sec=elapsed_sec,
            )
        else:
            attempt.status = AttemptStatus.REJECTED
            attempt.error = (
                f"returncode={returncode}; "
                + "; ".join(validation["errors"])
            )
            attempt.finished_at = utc_now()
            self.store.save_attempt(attempt)
            infrastructure_code = self._remote_smoke_infrastructure_code(
                result=result,
                stderr=stderr_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ),
                returncode=returncode,
            )
            if infrastructure_code:
                outcome = JobOutcome(
                    False,
                    "retry",
                    "remote_smoke_infrastructure_retry",
                    {
                        "failure_class": "infrastructure_transient",
                        "failure_code": infrastructure_code,
                        "consume_attempt": False,
                        "returncode": returncode,
                        "validation": validation,
                        "pool_result": dict(result),
                    },
                    elapsed_sec=elapsed_sec,
                )
            else:
                idea.gpu_seconds_spent += allocated_gpus * max(
                    0.0,
                    elapsed_sec,
                )
                diagnostics = {
                    "failure_class": "runtime_contract_invalid",
                    "failure_code": "remote_smoke_validation_failed",
                    "returncode": returncode,
                    "errors": validation["errors"],
                    **self._implementation_failure_fingerprint(
                        idea_id=idea.idea_id,
                        errors=validation["errors"],
                    ),
                }
                prior = idea.candidate.get(
                    "_autoresearch_v2_last_implementation_failure"
                )
                if (
                    isinstance(prior, Mapping)
                    and prior.get("fingerprint")
                    == diagnostics.get("fingerprint")
                ):
                    diagnostics["no_progress"] = True
                    self.store.event(
                        "remote_smoke_no_progress",
                        idea_id=idea.idea_id,
                        job_id=job.job_id,
                        attempt_id=attempt.attempt_id,
                        fingerprint=diagnostics.get("fingerprint", ""),
                    )
                idea.candidate[
                    "_autoresearch_v2_last_implementation_failure"
                ] = diagnostics
                self._queue_build_repair(
                    idea=idea,
                    job=job,
                    attempt=attempt,
                    reason="remote_smoke_invalid",
                    diagnostics=diagnostics,
                )
                return
        idea.gpu_seconds_spent += allocated_gpus * max(0.0, elapsed_sec)
        self._apply_outcome(
            idea,
            job,
            attempt,
            JobOutcome(
                outcome.success,
                outcome.decision,
                outcome.reason,
                outcome.result,
                tokens=outcome.tokens,
                elapsed_sec=0.0,
            ),
        )

    @staticmethod
    def _normalize_controller_runtime_artifacts(
        *,
        output_dir: Path,
        plan: Mapping[str, Any],
        mode: str,
        allocated_gpus: int,
        returncode: int,
    ) -> dict[str, Any]:
        """Recovery path for candidates built before the wrapper existed."""

        try:
            from .runtime_wrapper import normalize_runtime_artifacts

            value = normalize_runtime_artifacts(
                output_dir=output_dir,
                plan=plan,
                mode=mode,
                allocated_gpus=allocated_gpus,
                core_returncode=returncode,
                cwd=output_dir.parent.parent,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": f"runtime_artifact_compile_failed: {exc}"}
        return {"compiled": True, **value}

    @staticmethod
    def _runtime_compiler_identity() -> str:
        from .runtime_wrapper import WRAPPER_SCHEMA, WRAPPER_VERSION

        return f"{WRAPPER_SCHEMA}:v{WRAPPER_VERSION}"

    @staticmethod
    def _candidate_runtime_compiler(candidate: Path) -> str:
        build_path = candidate / "build.json"
        try:
            build = json.loads(build_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return ""
        runtime = build.get("controller_runtime")
        if not isinstance(runtime, Mapping):
            return ""
        schema = str(runtime.get("schema", "") or "")
        try:
            version = int(runtime.get("version", -1))
        except (TypeError, ValueError):
            return ""
        return f"{schema}:v{version}" if schema and version >= 0 else ""

    def _implementation_failure_fingerprint(
        self,
        *,
        idea_id: str,
        errors: list[Any],
    ) -> dict[str, str]:
        current = self.store.current_dir(idea_id)
        implementation = validate_research_implementation(
            current,
            plan=self._read_current_json(idea_id, "plan.json"),
            controller_runtime=True,
        )
        source_hashes = implementation.get("evidence", {}).get(
            "source_sha256",
            {},
        )
        try:
            build = self._read_current_json(idea_id, "build.json")
        except (OSError, ValueError, json.JSONDecodeError):
            build = {}
        runtime_compiler = self._runtime_compiler_identity()
        implementation_identity = {
            "source_sha256": source_hashes,
            "build_sha256": canonical_json_sha256(build),
            "runtime_compiler": runtime_compiler,
        }
        normalized_errors = sorted(
            " ".join(str(error).casefold().split())
            for error in errors
        )
        payload = {
            "implementation": implementation_identity,
            "errors": normalized_errors,
        }
        return {
            "fingerprint": canonical_json_sha256(payload),
            "source_sha256": canonical_json_sha256(
                implementation_identity
            ),
            "runtime_compiler": runtime_compiler,
        }

    def _block_unchanged_gpu_implementation(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
    ) -> bool:
        """Fail closed before GPU allocation when Build made no source change."""

        prior = idea.candidate.get(
            "_autoresearch_v2_last_implementation_failure"
        )
        if not isinstance(prior, Mapping):
            return False
        if not (
            self._is_remote_smoke_job(job)
            or job.kind in {JobKind.PILOT, JobKind.SCALE}
        ):
            return False
        prior_source = str(prior.get("source_sha256", "") or "")
        prior_errors = prior.get("errors")
        prior_compiler = str(prior.get("runtime_compiler", "") or "")
        if (
            not prior_source
            or not isinstance(prior_errors, list)
            or not prior_compiler
        ):
            return False
        current = self._implementation_failure_fingerprint(
            idea_id=idea.idea_id,
            errors=prior_errors,
        )
        if (
            current.get("source_sha256") != prior_source
            or current.get("runtime_compiler") != prior_compiler
        ):
            return False
        diagnostics = {
            **dict(prior),
            **current,
            "no_progress": True,
            "blocked_before_gpu_submit": True,
        }
        idea.status = IdeaStatus.QUARANTINED
        reason = (
            "remote_smoke_no_progress"
            if self._is_remote_smoke_job(job)
            else "gpu_implementation_no_progress"
        )
        event_type = (
            "remote_smoke_blocked_no_progress"
            if self._is_remote_smoke_job(job)
            else "gpu_implementation_blocked_no_progress"
        )
        idea.exit_reason = reason
        idea.current_job_id = ""
        idea.last_progress_at = utc_now()
        job.status = JobStatus.FAILED
        job.retry_not_before = ""
        job.result = {
            **job.result,
            "decision": "quarantine",
            "reason": reason,
            "failure_class": "implementation_invalid",
            "diagnostics": diagnostics,
        }
        self.store.save_job(job)
        self.store.save_idea(idea)
        self.store.event(
            event_type,
            idea_id=idea.idea_id,
            job_id=job.job_id,
            fingerprint=current.get("fingerprint", ""),
            source_sha256=current.get("source_sha256", ""),
        )
        return True

    def _queue_build_repair(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        reason: str,
        diagnostics: Mapping[str, Any],
    ) -> None:
        """Return an invalid implementation to Build instead of rerunning it."""

        try:
            repair_count = int(
                idea.candidate.get(
                    "_autoresearch_v2_build_repair_count",
                    0,
                )
                or 0
            ) + 1
        except (TypeError, ValueError):
            repair_count = 1
        if repair_count > self.config.budgets.max_build_attempts:
            job.attempt = max(job.attempt, job.attempt_limit)
            self._apply_outcome(
                idea,
                job,
                attempt,
                JobOutcome(
                    False,
                    "retry",
                    "build_repair_budget_exhausted",
                    {
                        "failure_class": "implementation_invalid",
                        "failure_code": reason,
                        "diagnostics": dict(diagnostics),
                    },
                ),
            )
            return
        feedback = {
            "source_job_id": job.job_id,
            "source_attempt_id": attempt.attempt_id,
            "reason": reason,
            "repair_count": repair_count,
            "diagnostics": dict(diagnostics),
        }
        idea.candidate[
            "_autoresearch_v2_build_repair_count"
        ] = repair_count
        idea.candidate[
            "_autoresearch_v2_build_repair_feedback"
        ] = feedback
        idea.status = IdeaStatus.BUILDING
        idea.current_job_id = ""
        idea.exit_reason = ""
        idea.last_progress_at = utc_now()
        job.status = JobStatus.FAILED
        job.retry_not_before = ""
        job.result = {
            **job.result,
            "decision": "repair_build",
            "reason": reason,
            "failure_class": "implementation_invalid",
            "diagnostics": dict(diagnostics),
        }
        self.store.save_transition(
            idea=idea,
            job=job,
            attempt=attempt,
            event_type="gpu_implementation_returned_to_build",
            payload={
                "reason": reason,
                "repair_count": repair_count,
                "diagnostics": dict(diagnostics),
            },
        )

    @staticmethod
    def _remote_smoke_infrastructure_code(
        *,
        result: Mapping[str, Any],
        stderr: str,
        returncode: int,
    ) -> str:
        """Classify failures outside the generated experiment's control."""

        if bool(result.get("timed_out")) or returncode == 124:
            if any(
                marker in stderr
                for marker in (
                    "Network is unreachable",
                    "Connection timed out",
                    "Temporary failure in name resolution",
                )
            ):
                return "dependency_network_unreachable"
            return "gpu_task_timeout"
        if str(result.get("pool_state", "") or "") in {"lost", "unknown"}:
            return "gpu_task_lost"
        if result.get("error"):
            return "gpu_pool_collect_failed"
        return ""

    def _attest_gpu_execution(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        output_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
        returncode: int,
        allocated_gpus: int,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        contract_path = (
            self.store.attempt_dir(attempt) / "execution_contract.json"
        )
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            expected_hash = str(
                job.result.get("execution_contract_sha256", "") or ""
            )
            actual_hash = canonical_json_sha256(contract)
            if not expected_hash or actual_hash != expected_hash:
                raise ValueError(
                    "execution contract hash differs from durable pre-run hash"
                )
            candidate = self.store.attempt_dir(attempt) / "candidate"
            build_path = candidate / "build.json"
            if contract.get("mode") == "smoke":
                build_path = (
                    candidate
                    / "trusted"
                    / "remote_smoke"
                    / "executed_build.json"
                )
            verify_execution_contract(
                contract,
                idea_id=idea.idea_id,
                job_id=job.job_id,
                attempt_id=attempt.attempt_id,
                mode=str(contract["mode"]),
                argv=list(contract["argv"]),
                cwd=candidate,
                entrypoint=str(contract["entrypoint"]),
                output_dir=str(contract["output_dir"]),
                resource_limits=dict(contract["resource_limits"]),
                plan_path=candidate / "plan.json",
                build_path=build_path,
                allowed_env_keys=self.config.execution.allowed_env_keys,
            )
            key = self._attestation_key()
            ended = datetime.now(UTC)
            started = ended - timedelta(
                seconds=max(0.0, float(elapsed_sec))
            )
            attestation_value = create_execution_attestation(
                contract,
                signing_key=key,
                key_id=self.config.execution.attestation_key_id,
                started_at=started,
                ended_at=ended,
                returncode=returncode,
                allocated_gpus=allocated_gpus,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_dir=output_dir,
            )
            attestation_path = (
                self.store.attempt_dir(attempt)
                / "execution_attestation.json"
            )
            attestation_hash = write_execution_attestation(
                attestation_path,
                attestation_value,
            )
            verification_errors = verify_execution_attestation(
                contract_path,
                attestation_path,
                candidate,
                signing_key=key,
                key_id=self.config.execution.attestation_key_id,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                artifact_dir=output_dir,
                returncode=returncode,
                allocated_gpus=allocated_gpus,
            )
            if verification_errors:
                raise ValueError(
                    "execution attestation verification failed: "
                    + "; ".join(verification_errors)
                )
        except Exception as exc:  # noqa: BLE001
            return {"errors": [f"{type(exc).__name__}: {exc}"]}
        return {
            "path": str(attestation_path),
            "sha256": attestation_hash,
            "contract_path": str(contract_path),
            "contract_sha256": expected_hash,
            "verified": True,
            "key_id": self.config.execution.attestation_key_id,
            "idea_id": idea.idea_id,
            "job_id": job.job_id,
        }

    def _attestation_key(self) -> bytes:
        configured = self.config.execution.attestation_key_file.strip()
        key_path = (
            Path(configured).expanduser().resolve()
            if configured
            else self.config.root / ".controller-attestation.key"
        )

        def read_key() -> bytes:
            raw = key_path.read_bytes().strip()
            try:
                decoded = base64.urlsafe_b64decode(raw)
            except (ValueError, TypeError):
                decoded = raw
            if len(decoded) < 32:
                raise ValueError(
                    f"attestation key is too short: {key_path}"
                )
            return decoded

        if key_path.exists():
            return read_key()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(32)
        encoded = base64.urlsafe_b64encode(key)
        temp_path = key_path.with_name(
            f".{key_path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
        )
        try:
            descriptor = os.open(
                temp_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.write(descriptor, encoded + b"\n")
            os.fsync(descriptor)
        finally:
            if "descriptor" in locals():
                os.close(descriptor)
        try:
            os.link(temp_path, key_path)
        except FileExistsError:
            return read_key()
        finally:
            temp_path.unlink(missing_ok=True)
        return read_key()

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
        prior_result = dict(job.result)
        job.result = {
            **prior_result,
            "decision": outcome.decision,
            "reason": outcome.reason,
            **outcome.result,
        }
        attempt.finished_at = attempt.finished_at or utc_now()
        if not outcome.success:
            consume_attempt = outcome.result.get("consume_attempt", True)
            if consume_attempt is False:
                infrastructure_retries = int(
                    job.result.get("infrastructure_retries", 0) or 0
                ) + 1
                job.result["infrastructure_retries"] = infrastructure_retries
                if infrastructure_retries <= 5:
                    job.attempt_limit += 1
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
                if self._is_remote_smoke_job(job):
                    idea.status = IdeaStatus.PILOTING
                else:
                    idea.status = {
                        JobKind.DESIGN: IdeaStatus.BUILDING,
                        JobKind.BUILD: IdeaStatus.PILOTING,
                        JobKind.PILOT: IdeaStatus.SCALING,
                        JobKind.SCALE: IdeaStatus.REPORTING,
                        JobKind.REPORT: IdeaStatus.COMPLETED,
                    }[job.kind]
            elif outcome.decision == "complete":
                if (
                    job.kind is JobKind.REPORT
                    and idea.candidate.get("final_outcome")
                    == "informative_negative"
                ):
                    idea.status = IdeaStatus.COMPLETED_NEGATIVE
                else:
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
                durable_gate = accepted.validation.get(
                    "decision_gate",
                    {},
                )
                durable_decision = str(
                    durable_gate.get("decision", "")
                    if isinstance(durable_gate, Mapping)
                    else ""
                ).casefold()
                if durable_decision == "complete_negative":
                    idea.status = IdeaStatus.REPORTING
                    idea.candidate[
                        "final_outcome"
                    ] = "informative_negative"
                elif durable_decision == "reject":
                    idea.status = IdeaStatus.REJECTED
                elif self._is_remote_smoke_job(job):
                    idea.status = IdeaStatus.PILOTING
                else:
                    idea.status = {
                        JobKind.DESIGN: IdeaStatus.BUILDING,
                        JobKind.BUILD: IdeaStatus.PILOTING,
                        JobKind.PILOT: IdeaStatus.SCALING,
                        JobKind.SCALE: IdeaStatus.REPORTING,
                        JobKind.REPORT: (
                            IdeaStatus.COMPLETED_NEGATIVE
                            if idea.candidate.get("final_outcome")
                            == "informative_negative"
                            else IdeaStatus.COMPLETED
                        ),
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
            refunded = False
            interrupted_attempt_id = job.attempt_id
            interrupted_attempt: AttemptRecord | None = None
            if job.attempt_id:
                attempt = self.store.get_attempt(job.attempt_id)
                interrupted_attempt = attempt
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
                    # The process disappeared before a scientific outcome was
                    # observed. Keep the audit row, but refund this attempt so
                    # repeated service restarts cannot exhaust retry budget.
                    if job.attempt == attempt.number:
                        job.attempt = max(0, job.attempt - 1)
                        refunded = True
            if refunded and interrupted_attempt is not None:
                # A refunded interruption reuses the same scientific attempt
                # number. Preserve the failed attempt row for audit, but remove
                # only its incomplete candidate workspace so prepare_candidate
                # or snapshot_current can recreate it cleanly.
                shutil.rmtree(
                    self.store.attempt_dir(interrupted_attempt) / "candidate",
                    ignore_errors=True,
                )
            job.status = JobStatus.RETRY_WAIT
            job.retry_not_before = datetime.now(UTC).isoformat(
                timespec="milliseconds"
            )
            job.attempt_id = ""
            job.submitted_task_id = ""
            job.result = {
                **job.result,
                "infrastructure_interruption": "controller_interrupted",
                "interrupted_attempt_id": interrupted_attempt_id,
                "attempt_refunded": refunded,
            }
            self.store.save_job(job)
            if idea is not None:
                idea.status = _STATUS_FOR_KIND[job.kind]
                idea.exit_reason = ""
                self.store.save_idea(idea)
            self.store.event(
                "interrupted_job_recovered",
                idea_id=job.idea_id,
                job_id=job.job_id,
                status=job.status.value,
                interrupted_attempt_id=interrupted_attempt_id,
                attempt_refunded=refunded,
                scientific_attempt=job.attempt,
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
                if (
                    job.kind is JobKind.PILOT
                    or self._is_remote_smoke_job(job)
                )
                else self.config.budgets.scale_gpu_hours
            )
            # GPU accounting is GPU-seconds, so the configured budget is also
            # interpreted as GPU-hours rather than wall-clock hours.
            budget_seconds = budget_hours * 3600.0
            if idea.gpu_seconds_spent >= budget_seconds:
                stage = (
                    "smoke"
                    if self._is_remote_smoke_job(job)
                    else job.kind.value
                )
                return f"{stage}_gpu_budget_exhausted"
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
                    stage = (
                        "smoke"
                        if self._is_remote_smoke_job(job)
                        else job.kind.value
                    )
                    return f"{stage}_gpu_budget_insufficient"
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
                "total_gpus": self.configured_gpu_capacity,
                "allocated_gpus": 0,
                "available_gpus": 0,
                "utilization": 0.0,
                "target_utilization": self.config.gpu.target_utilization,
                "pending_jobs": pending_gpu,
                "leases": [],
                "state": (
                    "unavailable"
                    if self.config.gpu.enabled
                    else "disabled"
                ),
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
            "idea_generation_running": (
                self._idea_generation is not None
            ),
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


def _live_model_gateway_calls() -> int:
    """Count live plain-LLM CLI calls owned by the local bridge service."""

    bridge_pids: set[int] = set()
    process_rows: list[tuple[int, int, str]] = []
    proc = Path("/proc")
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(
                encoding="utf-8",
                errors="replace",
            )
            closing = stat.rfind(")")
            ppid = int(stat[closing + 2 :].split()[1])
            cmdline = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except (OSError, ValueError, IndexError):
            continue
        pid = int(entry.name)
        process_rows.append((pid, ppid, cmdline))
        if "claude-bridge" in cmdline and (
            "researchclaw" in cmdline or "uvicorn app:app" in cmdline
        ):
            bridge_pids.add(pid)
    if not bridge_pids:
        return 0
    count = 0
    for _, ppid, cmdline in process_rows:
        if ppid not in bridge_pids:
            continue
        if (
            "codebuddy -p" in cmdline
            and "plain language-model API backend" in cmdline
            and "--tools  --strict-mcp-config" in cmdline
        ):
            count += 1
    return count


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

    def detach_workers_for_process_exit(self) -> None:
        """Prevent Python's executor atexit hook from rejoining live workers."""

        try:
            from concurrent.futures import thread as thread_module
        except ImportError:
            return
        for worker in tuple(self._threads):
            thread_module._threads_queues.pop(worker, None)
