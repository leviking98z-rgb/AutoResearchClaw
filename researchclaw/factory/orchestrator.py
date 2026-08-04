"""Steady-state continuous multi-Idea Factory control loop."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

from .actor import (
    IdeaWorker,
    PipelineIdeaWorker,
    SimulatedIdeaWorker,
    work_item_for_idea,
)
from .admission import AdmissionController
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
            state = self.store.load_state()
            state.update(
                {
                    "status": "stopped" if self._stop.is_set() else "idle",
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
        return ticks

    def tick(self) -> dict[str, Any]:
        state = self.store.load_state()
        state["tick"] = int(state.get("tick", 0)) + 1
        state["status"] = "paused" if self.store.control_requested("pause") else "running"
        state["pid"] = os.getpid()
        self.store.save_state(state)

        self._reconcile_workers()
        if self.gpu_broker is not None:
            completed = self.gpu_broker.reconcile()
            for item in completed:
                if item.status is WorkItemStatus.SUCCEEDED:
                    self._complete_item(item)
                else:
                    self._fail_item(item, reason="GPU_TASK_FAILED")

        if not self.store.control_requested("pause"):
            self._refill_reservoir()
            self._admit_ideas()
            self._ensure_work_items()
            self._dispatch_ready_items()

        snapshot = self.store.snapshot()
        self.store.event(
            "factory_tick",
            tick=state["tick"],
            reservoir_size=snapshot["reservoir_size"],
            ideas_by_status=snapshot["ideas_by_status"],
            work_items_by_status=snapshot["work_items_by_status"],
        )
        return snapshot

    def _refill_reservoir(self) -> None:
        reservoir = self.store.load_reservoir()
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

    def _admit_ideas(self) -> None:
        ideas = self.store.list_ideas()
        slots = self.scheduler.active_slots_available(ideas)
        if slots <= 0:
            return
        controller = AdmissionController(self.config)
        candidates = self.store.load_reservoir()
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
                self.store.save_idea(
                    candidate, event_type="candidate_rejected"
                )
                self.store.event(
                    "admission_rejected",
                    idea_id=candidate.idea_id,
                    reason_code=decision.reason_code,
                    detail=decision.detail,
                )
                ideas.append(candidate)
                continue
            candidate.status = IdeaStatus.SCREENING
            self.store.save_idea(candidate, event_type="idea_admitted")
            ideas.append(candidate)
            slots -= 1

    def _ensure_work_items(self) -> None:
        existing = {
            item.item_id: item for item in self.store.list_work_items()
        }
        for idea in self.store.list_ideas(statuses=ACTIVE_IDEA_STATUSES):
            if idea.current_item_id:
                current = existing.get(idea.current_item_id)
                if current is not None and not current.terminal:
                    continue
            item = work_item_for_idea(idea)
            if self.config.worker.simulation:
                item.kind = WorkKind.PIPELINE
            # Re-entering the same profile after a bounded repair gets a
            # deterministic round suffix rather than mutating past evidence.
            base = item.item_id
            round_index = 1
            while item.item_id in existing:
                old = existing[item.item_id]
                if old.status is not WorkItemStatus.SUCCEEDED:
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

    def _dispatch_ready_items(self) -> None:
        ideas = self.store.list_ideas()
        idea_by_id = {idea.idea_id: idea for idea in ideas}
        ready = self.scheduler.order(self.store.ready_work_items(), ideas)
        running_local = sum(
            item.status is WorkItemStatus.RUNNING
            and item.kind is not WorkKind.GPU_EXPERIMENT
            for item in self.store.list_work_items()
        )
        for item in ready:
            idea = idea_by_id.get(item.idea_id)
            if idea is None or idea.status not in ACTIVE_IDEA_STATUSES:
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
                        item.kind = WorkKind.PIPELINE
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
                        submission = self.gpu_broker.submit(item)
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
            if running_local >= self.config.scheduler.llm_slots:
                item.status = WorkItemStatus.QUEUED
                self.store.save_work_item(item, event_type="work_item_queued")
                continue
            item.status = WorkItemStatus.RUNNING
            item.attempt += 1
            self.store.save_work_item(item, event_type="work_item_started")
            self.worker.start(
                idea=idea,
                item=item,
                idea_dir=self.store.idea_dir(idea.idea_id),
            )
            running_local += 1

    def _reconcile_workers(self) -> None:
        for item in self.store.list_work_items():
            if (
                item.status is not WorkItemStatus.RUNNING
                or item.kind is WorkKind.GPU_EXPERIMENT
            ):
                continue
            idea = self.store.get_idea(item.idea_id)
            if idea is None:
                continue
            probe = self.worker.probe(
                item=item,
                idea_dir=self.store.idea_dir(item.idea_id),
            )
            if probe.state == "running":
                continue
            if probe.state == "finished" and probe.returncode == 0:
                item.status = WorkItemStatus.SUCCEEDED
                item.result = {
                    "returncode": 0,
                    "started_at": probe.started_at,
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
        if (
            self.gpu_broker is not None
            and item.metadata.get("gpu_execution")
            == "pipeline_driver_over_shared_pool"
        ):
            self.gpu_broker.release_driver(item.item_id)
        idea = self.store.get_idea(item.idea_id)
        if idea is None:
            return
        run_dir = self.store.idea_dir(idea.idea_id) / "runs" / "pipeline"
        ledger = self.store.load_budget(idea.idea_id)
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

    def _fail_item(self, item: WorkItem, *, reason: str) -> None:
        if (
            self.gpu_broker is not None
            and item.metadata.get("gpu_execution")
            == "pipeline_driver_over_shared_pool"
        ):
            self.gpu_broker.release_driver(item.item_id)
        item.result = {"failure_reason": reason}
        if item.attempt < item.attempt_limit:
            item.status = WorkItemStatus.RETRY_WAIT
            self.store.save_work_item(item, event_type="work_item_retry_wait")
            return
        item.status = WorkItemStatus.FAILED
        self.store.save_work_item(item, event_type="work_item_failed")
        idea = self.store.get_idea(item.idea_id)
        if idea is None:
            return
        idea.status = IdeaStatus.FAILED
        idea.exit_reason = reason
        idea.current_item_id = ""
        self.store.save_idea(idea, event_type="idea_failed")

    def cancel_active(self) -> None:
        for item in self.store.list_work_items():
            if item.status not in {
                WorkItemStatus.RUNNING,
                WorkItemStatus.QUEUED,
                WorkItemStatus.READY,
            }:
                continue
            if item.kind is WorkKind.GPU_EXPERIMENT and self.gpu_broker:
                self.gpu_broker.cancel_item(item)
            elif item.status is WorkItemStatus.RUNNING:
                self.worker.cancel(
                    item=item,
                    idea_dir=self.store.idea_dir(item.idea_id),
                )
                item.status = WorkItemStatus.CANCELLED
                self.store.save_work_item(
                    item, event_type="work_item_cancelled"
                )
