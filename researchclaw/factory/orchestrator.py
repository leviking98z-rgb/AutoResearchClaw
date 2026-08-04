"""Steady-state continuous multi-Idea Factory control loop."""

from __future__ import annotations

import os
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .actor import (
    IdeaWorker,
    PipelineIdeaWorker,
    SimulatedIdeaWorker,
    work_item_for_idea,
)
from .admission import AdmissionController
from .budgets import budget_allows, tier_limit_hours
from .config import FactoryConfig
from .gates import EvidenceGate
from .generator import CandidateGenerator
from .gpu_broker import GPUBroker
from .io import atomic_write_json
from .models import (
    ACTIVE_IDEA_STATUSES,
    GateAction,
    IdeaStatus,
    WorkItem,
    WorkItemStatus,
    WorkKind,
    utc_now,
)
from .observability import build_factory_observability
from .scheduler import FactoryScheduler
from .store import FactoryStore


class FactoryOrchestrator:
    """Single-writer coordinator that never terminates on paper completion."""

    def __init__(
        self,
        *,
        config: FactoryConfig,
        store: FactoryStore,
        generator: CandidateGenerator,
        worker: IdeaWorker | None = None,
        scheduler: FactoryScheduler | None = None,
        gate: EvidenceGate | None = None,
        gpu_broker: GPUBroker | None = None,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self.config = config
        self.store = store
        self.generator = generator
        self.worker = worker or (
            SimulatedIdeaWorker(delay_sec=config.worker.simulation_delay_sec)
            if config.worker.simulation
            else PipelineIdeaWorker(config)
        )
        self.scheduler = scheduler or FactoryScheduler(config)
        self.gate = gate or EvidenceGate(config)
        self.gpu_broker = gpu_broker
        self._sleep = sleep
        self._monotonic = monotonic
        self._stop = threading.Event()
        self._last_generation = float("-inf")
        self._generation_failures = 0

    def initialize(self) -> None:
        self.store.initialize()
        self.store.acquire_writer_lock()
        state = self.store.load_state()
        state.update(
            {
                "status": "running" if self.config.enabled else "disabled",
                "pid": os.getpid(),
                "started_at": state.get("started_at") or utc_now(),
            }
        )
        self.store.save_state(state)
        self.store.event("factory_initialized", enabled=self.config.enabled)
        self.store.event(
            "operational_logging_ready",
            factory_events=str(self.store.events_path),
            idea_event_pattern=str(
                self.store.ideas_dir / "<idea_id>" / "events.jsonl"
            ),
            idea_operational_pattern=str(
                self.store.ideas_dir
                / "<idea_id>"
                / "operational_events.jsonl"
            ),
        )

    def request_stop(self) -> None:
        self._stop.set()

    def run(
        self,
        *,
        once: bool = False,
        max_ticks: int | None = None,
    ) -> int:
        self.initialize()
        if not self.config.enabled:
            raise RuntimeError(
                "Factory mode is disabled; set factory.enabled: true explicitly"
            )
        previous_handlers: dict[int, Any] = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                previous_handlers[signum] = signal.signal(
                    signum, lambda *_: self.request_stop()
                )
            except (ValueError, OSError):
                pass
        ticks = 0
        try:
            while not self._stop.is_set():
                if self.store.control_requested("stop"):
                    self._stop.set()
                    break
                self.tick()
                ticks += 1
                if once or (max_ticks is not None and ticks >= max_ticks):
                    break
                self._sleep(self.config.scheduler.poll_interval_sec)
        finally:
            try:
                if self._stop.is_set():
                    self.cancel_active()
                state = self.store.load_state()
                state.update(
                    {
                        "status": (
                            "stopped" if self._stop.is_set() else "idle"
                        ),
                        "pid": None,
                        "stopped_at": utc_now(),
                    }
                )
                self.store.save_state(state)
                for signum, handler in previous_handlers.items():
                    try:
                        signal.signal(signum, handler)
                    except (ValueError, OSError):
                        pass
            finally:
                self.store.release_writer_lock()
        return ticks

    def tick(self) -> dict[str, Any]:
        tick_started = self._monotonic()
        state = self.store.load_state()
        state["tick"] = int(state.get("tick", 0)) + 1
        state["status"] = "paused" if self.store.control_requested("pause") else "running"
        state["pid"] = os.getpid()
        self.store.save_state(state)

        ideas = self.store.list_ideas()
        items = self.store.list_work_items()
        self._reconcile_workers(items=items)
        if self.gpu_broker is not None:
            try:
                completed = self.gpu_broker.reconcile()
            except Exception as exc:  # noqa: BLE001
                completed = []
                self.store.event(
                    "gpu_broker_reconcile_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            for item in completed:
                if item.status is WorkItemStatus.SUCCEEDED:
                    self._complete_item(item)
                else:
                    self._fail_item(item, reason="GPU_TASK_FAILED")

        if not self.store.control_requested("pause"):
            reservoir = self.store.load_reservoir()
            self._refill_reservoir(reservoir=reservoir)
            # Candidate generation may have changed the reservoir.
            reservoir = self.store.load_reservoir()
            self._admit_ideas(ideas=ideas, candidates=reservoir)
            # Admission and reconciliation may have changed both collections.
            ideas = self.store.list_ideas()
            items = self.store.list_work_items()
            self._ensure_work_items(ideas=ideas, items=items)
            items = self.store.list_work_items()
            self._dispatch_ready_items(ideas=ideas, items=items)

        snapshot = self.store.snapshot()
        self.store.event(
            "factory_tick",
            tick=state["tick"],
            elapsed_sec=round(
                max(0.0, self._monotonic() - tick_started),
                6,
            ),
            reservoir_size=snapshot["reservoir_size"],
            ideas_by_status=snapshot["ideas_by_status"],
            work_items_by_status=snapshot["work_items_by_status"],
        )
        build_factory_observability(self.store)
        return snapshot

    def _refill_reservoir(
        self,
        *,
        reservoir: list[Any] | None = None,
    ) -> None:
        reservoir = (
            self.store.load_reservoir()
            if reservoir is None
            else reservoir
        )
        now = self._monotonic()
        if len(reservoir) >= self.config.reservoir.low_watermark:
            return
        backoff = (
            self.config.reservoir.retry_backoff_sec
            * max(1, min(16, 2 ** max(0, self._generation_failures - 1)))
        )
        interval = (
            backoff
            if self._generation_failures
            else self.config.reservoir.generation_interval_sec
        )
        if now - self._last_generation < interval:
            return
        desired = max(
            self.config.reservoir.generation_batch_size,
            self.config.reservoir.target_size - len(reservoir),
        )
        context = {
            "factory": self.store.snapshot(),
            "archive": [idea.to_dict() for idea in self.store.list_ideas()],
        }
        self._last_generation = now
        try:
            generated = self.generator.generate(count=desired, context=context)
        except Exception as exc:  # noqa: BLE001
            self._generation_failures += 1
            self.store.event(
                "candidate_generation_failed",
                error=f"{type(exc).__name__}: {exc}",
                consecutive_failures=self._generation_failures,
            )
            return
        added = self.store.add_candidates(generated)
        self._generation_failures = 0
        self.store.event(
            "candidate_generation_completed",
            requested=desired,
            generated=len(generated),
            added=len(added),
        )

    def _admit_ideas(
        self,
        *,
        ideas: list[Any] | None = None,
        candidates: list[Any] | None = None,
    ) -> None:
        ideas = self.store.list_ideas() if ideas is None else list(ideas)
        slots = self.scheduler.active_slots_available(ideas)
        if slots <= 0:
            return
        controller = AdmissionController(self.config)
        candidates = (
            self.store.load_reservoir()
            if candidates is None
            else candidates
        )
        for candidate in sorted(
            candidates, key=lambda idea: (-idea.priority, idea.idea_id)
        ):
            if slots <= 0:
                break
            if not self.scheduler.status_slot_available(
                IdeaStatus.SCREENING, ideas
            ):
                break
            decision = controller.decide(candidate, existing=ideas)
            self.store.remove_candidate(candidate.idea_id)
            if not decision.admitted:
                controller.archive_rejection(candidate, decision)
                if decision.reason_code == "DUPLICATE_ID":
                    # The persisted Idea owns this ID. Saving the rejected
                    # candidate would overwrite the active or archived
                    # record with unrelated generated content.
                    self.store.event(
                        "candidate_rejected",
                        idea_id=candidate.idea_id,
                        status=candidate.status.value,
                    )
                    self.store.idea_event(
                        candidate.idea_id,
                        "candidate_rejected",
                        status=candidate.status.value,
                        budget_tier=candidate.budget_tier.value,
                        priority=candidate.priority,
                        current_item_id=candidate.current_item_id,
                        exit_reason=candidate.exit_reason,
                    )
                else:
                    self.store.save_idea(
                        candidate, event_type="candidate_rejected"
                    )
                self.store.event(
                    "admission_rejected",
                    idea_id=candidate.idea_id,
                    reason_code=decision.reason_code,
                    detail=decision.detail,
                )
                if decision.reason_code != "DUPLICATE_ID":
                    ideas.append(candidate)
                continue
            candidate.status = IdeaStatus.SCREENING
            self.store.save_idea(candidate, event_type="idea_admitted")
            ideas.append(candidate)
            slots -= 1

    def _ensure_work_items(
        self,
        *,
        ideas: list[Any] | None = None,
        items: list[WorkItem] | None = None,
    ) -> None:
        existing = {
            item.item_id: item
            for item in (
                self.store.list_work_items()
                if items is None
                else items
            )
        }
        active_ideas = (
            self.store.list_ideas(statuses=ACTIVE_IDEA_STATUSES)
            if ideas is None
            else [
                idea
                for idea in ideas
                if idea.status in ACTIVE_IDEA_STATUSES
            ]
        )
        for idea in active_ideas:
            if idea.current_item_id:
                current = existing.get(idea.current_item_id)
                if current is not None and not current.terminal:
                    continue
            item = work_item_for_idea(idea)
            if self.config.worker.simulation:
                item.kind = WorkKind.PIPELINE
            # Re-entering any profile after a terminal Work Item gets a
            # deterministic round suffix rather than mutating past evidence.
            base = item.item_id
            round_index = 1
            while item.item_id in existing:
                old = existing[item.item_id]
                if not old.terminal:
                    break
                round_index += 1
                item.item_id = f"{base}-round-{round_index:02d}"
            if item.item_id in existing and not existing[item.item_id].terminal:
                idea.current_item_id = item.item_id
                self.store.save_idea(idea)
                continue
            idea.current_item_id = item.item_id
            self.store.save_work_item(item, event_type="work_item_created")
            self.store.save_idea(idea)
            existing[item.item_id] = item

    def _dispatch_ready_items(
        self,
        *,
        ideas: list[Any] | None = None,
        items: list[WorkItem] | None = None,
    ) -> None:
        ideas = self.store.list_ideas() if ideas is None else list(ideas)
        idea_by_id = {idea.idea_id: idea for idea in ideas}
        ready = self.scheduler.order(self.store.ready_work_items(), ideas)
        all_items = (
            self.store.list_work_items()
            if items is None
            else items
        )
        running_local = sum(
            item.status is WorkItemStatus.RUNNING
            and (
                item.kind is not WorkKind.GPU_EXPERIMENT
                or item.metadata.get("gpu_execution")
                == "pipeline_driver_over_shared_pool"
            )
            for item in all_items
        )
        for item in ready:
            idea = idea_by_id.get(item.idea_id)
            if idea is None or idea.status not in ACTIVE_IDEA_STATUSES:
                continue
            ledger = self.store.load_budget(idea.idea_id)
            if item.kind is WorkKind.GPU_EXPERIMENT:
                requested_gpu_seconds, remaining_gpu_seconds = (
                    self._requested_gpu_seconds(
                        idea=idea,
                        item=item,
                        ledger=ledger,
                    )
                )
                if (
                    remaining_gpu_seconds is not None
                    and remaining_gpu_seconds <= 1e-12
                ):
                    self._reject_for_budget(
                        idea=idea,
                        item=item,
                        reason="GPU_BUDGET_EXHAUSTED",
                        requested_gpu_seconds=requested_gpu_seconds,
                        remaining_gpu_seconds=remaining_gpu_seconds,
                    )
                    continue
                if not budget_allows(
                    ledger,
                    self.config.budgets,
                    idea.budget_tier,
                    requested_gpu_seconds=requested_gpu_seconds,
                ):
                    self._reject_for_budget(
                        idea=idea,
                        item=item,
                        reason="GPU_BUDGET_EXHAUSTED",
                        requested_gpu_seconds=requested_gpu_seconds,
                        remaining_gpu_seconds=remaining_gpu_seconds,
                    )
                    continue
            elif not budget_allows(
                ledger,
                self.config.budgets,
                idea.budget_tier,
            ):
                self._reject_for_budget(
                    idea=idea,
                    item=item,
                    reason="LLM_BUDGET_EXHAUSTED",
                )
                continue
            needs_local_worker = (
                item.kind is not WorkKind.GPU_EXPERIMENT
                or self.config.worker.simulation
                or (
                    self.gpu_broker is not None
                    and not item.command
                )
            )
            if (
                needs_local_worker
                and running_local >= self.config.scheduler.llm_slots
            ):
                item.status = WorkItemStatus.QUEUED
                self.store.save_work_item(
                    item,
                    event_type="work_item_queued",
                )
                continue
            if item.kind is WorkKind.GPU_EXPERIMENT:
                delegated_to_local_worker = False
                if self.gpu_broker:
                    if not item.command:
                        # Stage 12 remains the per-Idea experiment driver in
                        # the first migration slice. It connects to the already
                        # prepared shared Ray address; it never owns the pool
                        # lifecycle. Phase 4 compiles individual seed/condition
                        # commands directly into GPU Work Items.
                        admitted, allocated, reason = (
                            self.gpu_broker.can_admit_driver(item)
                        )
                        if not admitted:
                            item.status = WorkItemStatus.QUEUED
                            self.store.save_work_item(
                                item,
                                event_type="gpu_driver_waiting_for_lease",
                            )
                            continue
                        item.metadata["gpu_execution"] = (
                            "pipeline_driver_over_shared_pool"
                        )
                        item.metadata["allocated_gpus"] = allocated
                        item.metadata["lease_reason"] = reason
                        self.store.save_work_item(
                            item, event_type="gpu_driver_delegated"
                        )
                        delegated_to_local_worker = True
                    else:
                        try:
                            submission = self.gpu_broker.submit(item)
                        except Exception as exc:  # noqa: BLE001
                            self.store.event(
                                "gpu_work_item_submit_failed",
                                idea_id=item.idea_id,
                                item_id=item.item_id,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            self.store.idea_event(
                                item.idea_id,
                                "gpu_work_item_submit_failed",
                                item_id=item.item_id,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            self._fail_item(
                                item,
                                reason=f"GPU_SUBMIT_{type(exc).__name__.upper()}",
                            )
                            continue
                        if not submission.admitted:
                            item.status = WorkItemStatus.QUEUED
                            self.store.save_work_item(
                                item, event_type="gpu_work_item_queued"
                            )
                        continue
                if (
                    not delegated_to_local_worker
                    and not self.config.worker.simulation
                ):
                    item.status = WorkItemStatus.QUEUED
                    self.store.save_work_item(
                        item, event_type="gpu_work_item_waiting_for_broker"
                    )
                    continue
            item.status = WorkItemStatus.RUNNING
            item.attempt += 1
            observed = self._read_mapping(
                self.store.idea_dir(idea.idea_id)
                / "runs"
                / "pipeline"
                / "observability_summary.json"
            )
            observed_llm = observed.get("llm")
            item.metadata["llm_calls_before"] = (
                int(observed_llm.get("calls", 0) or 0)
                if isinstance(observed_llm, dict)
                else 0
            )
            self.store.save_work_item(item, event_type="work_item_started")
            try:
                probe = self.worker.start(
                    idea=idea,
                    item=item,
                    idea_dir=self.store.idea_dir(idea.idea_id),
                )
            except Exception as exc:  # noqa: BLE001
                self.store.event(
                    "worker_start_failed",
                    idea_id=item.idea_id,
                    item_id=item.item_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.store.idea_event(
                    item.idea_id,
                    "worker_start_failed",
                    item_id=item.item_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._fail_item(
                    item,
                    reason=f"WORKER_START_{type(exc).__name__.upper()}",
                )
                continue
            item.result = {
                **item.result,
                "worker_pid": probe.pid,
                "started_at": probe.started_at or utc_now(),
            }
            self.store.save_work_item(
                item,
                event_type="work_item_worker_launched",
            )
            running_local += 1

    def _requested_gpu_seconds(
        self,
        *,
        idea: Any,
        item: WorkItem,
        ledger: Any,
    ) -> tuple[float, float | None]:
        requested_gpus = max(
            int(item.resources.min_gpus),
            int(item.resources.preferred_gpus),
        )
        timeout_sec = float(item.resources.timeout_sec)
        estimate = requested_gpus * timeout_sec
        limit_hours = tier_limit_hours(
            self.config.budgets,
            idea.budget_tier,
        )
        if limit_hours is None:
            return estimate, None
        remaining = max(
            0.0,
            limit_hours * 3600.0
            - float(
                ledger.gpu_seconds_by_tier.get(
                    idea.budget_tier.value,
                    0.0,
                )
            ),
        )
        # Reserve the full worst-case request.  Silently shrinking this value
        # to the remaining balance would let a task keep its original timeout
        # and overspend the supposedly hard tier budget.
        return estimate, remaining

    def _reconcile_workers(
        self,
        *,
        items: list[WorkItem] | None = None,
    ) -> None:
        for item in (
            self.store.list_work_items()
            if items is None
            else items
        ):
            if (
                item.status is not WorkItemStatus.RUNNING
                or (
                    item.kind is WorkKind.GPU_EXPERIMENT
                    and item.metadata.get("gpu_execution")
                    != "pipeline_driver_over_shared_pool"
                )
            ):
                continue
            idea = self.store.get_idea(item.idea_id)
            if idea is None:
                continue
            try:
                probe = self.worker.probe(
                    item=item,
                    idea_dir=self.store.idea_dir(item.idea_id),
                )
            except Exception as exc:  # noqa: BLE001
                self.store.event(
                    "worker_probe_failed",
                    idea_id=item.idea_id,
                    item_id=item.item_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.store.idea_event(
                    item.idea_id,
                    "worker_probe_failed",
                    item_id=item.item_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                continue
            if probe.state == "running":
                continue
            if probe.state == "finished" and probe.returncode == 0:
                item.status = WorkItemStatus.SUCCEEDED
                item.result = {
                    "returncode": 0,
                    "started_at": (
                        probe.started_at
                        or item.result.get("started_at", "")
                    ),
                    "finished_at": probe.finished_at,
                }
                self.store.save_work_item(
                    item, event_type="work_item_succeeded"
                )
                self._complete_item(item)
                continue
            self._fail_item(
                item,
                reason=(
                    f"WORKER_{probe.state.upper()}"
                    if probe.returncode is None
                    else f"WORKER_EXIT_{probe.returncode}"
                ),
            )

    def _complete_item(self, item: WorkItem) -> None:
        driver_elapsed_sec = 0.0
        if (
            self.gpu_broker is not None
            and item.metadata.get("gpu_execution")
            == "pipeline_driver_over_shared_pool"
        ):
            driver_elapsed_sec = self.gpu_broker.release_driver(item.item_id)
            if driver_elapsed_sec > 0:
                item.result["elapsed_sec"] = driver_elapsed_sec
                self.store.save_work_item(
                    item,
                    event_type="gpu_driver_usage_captured",
                )
        idea = self.store.get_idea(item.idea_id)
        if idea is None:
            return
        run_dir = self.store.idea_dir(idea.idea_id) / "runs" / "pipeline"
        ledger = self.store.load_budget(idea.idea_id)
        accounting = self._record_usage(
            idea=idea,
            item=item,
            run_dir=run_dir,
            ledger=ledger,
        )
        decision = self.gate.evaluate(idea, run_dir=run_dir, ledger=ledger)
        decision_path = (
            self.store.idea_dir(idea.idea_id)
            / "evidence"
            / f"gate-{item.item_id}.json"
        )
        atomic_write_json(decision_path, decision.to_dict())
        self.store.event(
            "gate_decided",
            idea_id=idea.idea_id,
            item_id=item.item_id,
            decision=decision.decision.value,
            reason_code=decision.reason_code,
        )
        self.store.idea_event(
            idea.idea_id,
            "gate_decided",
            item_id=item.item_id,
            decision=decision.decision.value,
            reason_code=decision.reason_code,
            current_tier=decision.current_tier.value,
            next_tier=(
                decision.next_tier.value
                if decision.next_tier is not None
                else None
            ),
            next_status=(
                decision.next_status.value
                if decision.next_status is not None
                else None
            ),
            evidence_refs=list(decision.evidence_refs),
            details=decision.details,
            accounting=accounting,
        )
        if decision.decision is GateAction.REPAIR:
            ledger.record_repair()
        if decision.next_tier is not None:
            idea.budget_tier = decision.next_tier
        if decision.next_status is not None:
            idea.status = decision.next_status
        if decision.decision in {
            GateAction.REJECT,
            GateAction.PARK,
            GateAction.COMPLETE,
            GateAction.COMPLETE_NEGATIVE,
        }:
            idea.exit_reason = decision.reason_code
        idea.current_item_id = ""
        self.store.save_budget(ledger)
        self.store.save_idea(idea, event_type="idea_gate_applied")

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, Any]:
        try:
            import json

            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, ValueError):
            return {}
        return dict(value) if isinstance(value, dict) else {}

    def _record_usage(
        self,
        *,
        idea: Any,
        item: WorkItem,
        run_dir: Path,
        ledger: Any,
    ) -> dict[str, Any]:
        """Account completed Work Item usage once, from durable artifacts."""

        if item.attempt < 1:
            return {}
        if (
            item.kind is WorkKind.GPU_EXPERIMENT
            and int(item.metadata.get("allocated_gpus", 0) or 0) <= 0
            and item.metadata.get("gpu_execution")
            != "pipeline_driver_over_shared_pool"
        ):
            # A deterministic submission may fail before any physical GPU task
            # is admitted. It consumes retry budget, but it must not create a
            # zero-usage accounting record that obscures the real execution.
            return {}
        accounting_id = f"{item.item_id}:attempt-{item.attempt:02d}"
        existing = ledger.usage_records.get(accounting_id)
        if isinstance(existing, dict):
            accounting = dict(existing)
            item.metadata["budget_accounted"] = True
            item.metadata["budget_accounting_id"] = accounting_id
            item.metadata["budget_accounting"] = accounting
            return accounting

        summary = self._read_mapping(run_dir / "observability_summary.json")
        llm = summary.get("llm")
        llm_data = llm if isinstance(llm, dict) else {}
        total_calls = int(llm_data.get("calls", 0) or 0)
        previous_calls = int(
            item.metadata.get("llm_calls_before", 0) or 0
        )
        llm_calls = max(0, total_calls - previous_calls)

        allocated_gpus = int(
            item.metadata.get("allocated_gpus", 0) or 0
        )
        elapsed_sec = 0.0
        try:
            elapsed_sec = max(
                0.0, float(item.result.get("elapsed_sec", 0.0) or 0.0)
            )
        except (TypeError, ValueError):
            elapsed_sec = 0.0
        if elapsed_sec <= 0:
            started = self._parse_timestamp(item.result.get("started_at"))
            finished = self._parse_timestamp(item.result.get("finished_at"))
            if started is not None and finished is not None:
                elapsed_sec = max(0.0, (finished - started).total_seconds())
        accounting, _ = ledger.record_attempt_usage(
            accounting_id,
            idea.budget_tier,
            gpu_count=allocated_gpus,
            elapsed_sec=elapsed_sec,
            llm_calls=llm_calls,
        )
        accounting.update(
            {
                "llm_calls_total_observed": total_calls,
            }
        )
        # Commit the ledger before the Work Item marker. A crash between these
        # writes is safe because the attempt-scoped accounting ID is durable
        # and idempotent in the ledger.
        self.store.save_budget(ledger)
        item.metadata["budget_accounted"] = True
        item.metadata["budget_accounting_id"] = accounting_id
        item.metadata["budget_accounting"] = accounting
        self.store.save_work_item(
            item,
            event_type="work_item_budget_accounted",
        )
        self.store.event(
            "budget_usage_recorded",
            idea_id=idea.idea_id,
            item_id=item.item_id,
            **accounting,
        )
        self.store.idea_event(
            idea.idea_id,
            "budget_usage_recorded",
            item_id=item.item_id,
            **accounting,
        )
        return accounting

    def _record_terminal_gpu_usage(
        self,
        item: WorkItem,
    ) -> dict[str, Any]:
        """Charge terminal failed/cancelled GPU attempts before retry or exit."""

        idea = self.store.get_idea(item.idea_id)
        if idea is None:
            return {}
        ledger = self.store.load_budget(idea.idea_id)
        accounting = self._record_usage(
            idea=idea,
            item=item,
            run_dir=(
                self.store.idea_dir(idea.idea_id)
                / "runs"
                / "pipeline"
            ),
            ledger=ledger,
        )
        self.store.save_budget(ledger)
        return accounting

    def _reject_for_budget(
        self,
        *,
        idea: Any,
        item: WorkItem,
        reason: str,
        requested_gpu_seconds: float = 0.0,
        remaining_gpu_seconds: float | None = None,
    ) -> None:
        ledger = self.store.load_budget(idea.idea_id)
        item.status = WorkItemStatus.CANCELLED
        item.result = {
            "failure_reason": reason,
            "requested_gpu_seconds": round(requested_gpu_seconds, 6),
            "remaining_gpu_seconds": (
                round(remaining_gpu_seconds, 6)
                if remaining_gpu_seconds is not None
                else None
            ),
        }
        self.store.save_work_item(
            item,
            event_type="work_item_budget_rejected",
        )
        idea.status = IdeaStatus.REJECTED
        idea.exit_reason = reason
        idea.current_item_id = ""
        self.store.save_idea(idea, event_type="idea_budget_rejected")
        self.store.event(
            "budget_rejected",
            idea_id=idea.idea_id,
            item_id=item.item_id,
            reason_code=reason,
            tier=idea.budget_tier.value,
            requested_gpu_seconds=round(requested_gpu_seconds, 6),
            remaining_gpu_seconds=(
                round(remaining_gpu_seconds, 6)
                if remaining_gpu_seconds is not None
                else None
            ),
            gpu_hours_used=ledger.gpu_hours(idea.budget_tier),
            llm_calls=ledger.llm_calls,
        )

    def _fail_item(self, item: WorkItem, *, reason: str) -> None:
        elapsed_sec = 0.0
        if (
            self.gpu_broker is not None
            and item.metadata.get("gpu_execution")
            == "pipeline_driver_over_shared_pool"
        ):
            elapsed_sec = self.gpu_broker.release_driver(item.item_id)
        result = dict(item.result)
        result["failure_reason"] = reason
        if elapsed_sec > 0:
            result.setdefault("elapsed_sec", elapsed_sec)
        item.result = result
        self._record_terminal_gpu_usage(item)
        if item.attempt < item.attempt_limit:
            item.status = WorkItemStatus.RETRY_WAIT
            self.store.save_work_item(item, event_type="work_item_retry_wait")
            return
        item.status = WorkItemStatus.FAILED
        self._save_failed_item(item, reason=reason)
        idea = self.store.get_idea(item.idea_id)
        if idea is None:
            return
        idea.status = IdeaStatus.FAILED
        idea.exit_reason = reason
        idea.current_item_id = ""
        self.store.save_idea(idea, event_type="idea_failed")

    def _save_failed_item(self, item: WorkItem, *, reason: str) -> None:
        """Persist a terminal failure with enriched global/local events."""

        self.store.save_work_item(
            item,
            event_type="work_item_failed",
            event_payload={
                "failure_reason": reason,
                "profile": item.profile,
                "kind": item.kind.value,
            },
        )

    def cancel_active(self) -> None:
        for item in self.store.list_work_items():
            if item.status not in {
                WorkItemStatus.RUNNING,
                WorkItemStatus.QUEUED,
                WorkItemStatus.READY,
            }:
                continue
            if (
                item.metadata.get("gpu_execution")
                == "pipeline_driver_over_shared_pool"
            ):
                if item.status is WorkItemStatus.RUNNING:
                    self.worker.cancel(
                        item=item,
                        idea_dir=self.store.idea_dir(item.idea_id),
                    )
                if self.gpu_broker is not None:
                    elapsed_sec = self.gpu_broker.release_driver(item.item_id)
                    if elapsed_sec > 0:
                        item.result["elapsed_sec"] = elapsed_sec
                item.status = WorkItemStatus.CANCELLED
                self.store.save_work_item(
                    item,
                    event_type="work_item_cancelled",
                )
                self._record_terminal_gpu_usage(item)
            elif item.kind is WorkKind.GPU_EXPERIMENT and self.gpu_broker:
                self.gpu_broker.cancel_item(item)
                self._record_terminal_gpu_usage(item)
            elif item.status is WorkItemStatus.RUNNING:
                self.worker.cancel(
                    item=item,
                    idea_dir=self.store.idea_dir(item.idea_id),
                )
                item.status = WorkItemStatus.CANCELLED
                self.store.save_work_item(
                    item, event_type="work_item_cancelled"
                )
            else:
                item.status = WorkItemStatus.CANCELLED
                self.store.save_work_item(
                    item, event_type="work_item_cancelled"
                )
