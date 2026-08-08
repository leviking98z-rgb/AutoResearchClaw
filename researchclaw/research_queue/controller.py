"""Async continuous research loop for the prototype."""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .benchmark_profile import TREATMENT_API
from .config import ResearchQueueConfig
from .execution import RunBackend
from .models import (
    BudgetLevel,
    Conclusion,
    GenerationBatch,
    IdeaRecord,
    IdeaStatus,
    PreparedRevision,
    ReviewAction,
    ReviewDecision,
    RunRecord,
    RunStatus,
    new_id,
    utc_now,
)
from .promotion import BenchmarkPromotionBridge
from .research_memory import ResearchMemory
from .scientific_gate import validate_research_spec
from .store import ResearchQueueStore
from .workers import (
    IdeaProducer,
    PreparationWorker,
    ResearchSpecWorker,
    ReviewWorker,
    materialize_revision,
    title_similarity,
    validate_proposal,
    validate_python_sources,
)


class ResearchQueueController:
    """One state owner coordinating multiple asynchronous Idea loops."""

    def __init__(
        self,
        *,
        config: ResearchQueueConfig,
        store: ResearchQueueStore,
        producer: IdeaProducer,
        preparer: PreparationWorker,
        reviewer: ReviewWorker,
        run_backend: RunBackend,
        spec_worker: ResearchSpecWorker | None = None,
        promotion_bridge: BenchmarkPromotionBridge | None = None,
        research_memory: ResearchMemory | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.producer = producer
        self.preparer = preparer
        self.reviewer = reviewer
        self.run_backend = run_backend
        self.spec_worker = spec_worker
        self.promotion_bridge = promotion_bridge
        self.research_memory = research_memory
        self.store.initialize()
        self._llm_slots = asyncio.Semaphore(config.concurrency.max_llm_jobs)
        self._run_slots = asyncio.Semaphore(config.concurrency.max_run_jobs)
        self._idea_tasks: dict[str, asyncio.Task[None]] = {}
        self._generation_task: asyncio.Task[None] | None = None
        self._next_generation_at = 0.0
        (
            self._generation_batches_started,
            self._next_generation_at,
        ) = self._restore_generation_schedule()
        self._producer_exhausted = False
        self._stop = asyncio.Event()
        self._started = False

    async def _call_llm(self, function: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run one blocking provider call with a controller-level deadline.

        The backend has its own network timeout, but the queue also needs a
        finite bound so a malformed CLI/provider interaction cannot pin one
        Idea forever during an unattended canary.
        """

        async with self._llm_slots:
            future = asyncio.create_task(
                asyncio.to_thread(function, *args, **kwargs),
            )
            done, _ = await asyncio.wait(
                {future},
                timeout=self.config.concurrency.llm_call_timeout_sec,
            )
            if done:
                return future.result()
            # asyncio cannot forcibly stop work already executing in a thread.
            # Keep a reference and consume its eventual result/exception while
            # releasing the Idea loop; the provider's own finite timeout reaps
            # the underlying request.
            future.add_done_callback(self._consume_background_result)
            raise TimeoutError(
                f"LLM call exceeded {self.config.concurrency.llm_call_timeout_sec:.1f}s"
            )

    @staticmethod
    def _consume_background_result(task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    def _restore_generation_schedule(self) -> tuple[int, float]:
        """Resume paid-generation throttling from durable event history."""

        events = self.store.list_events(limit=100000)
        generation_events = [
            item for item in events if item.get("event") == "idea_generation_started"
        ]
        if not generation_events:
            # Backward compatibility for states created before the explicit
            # start event existed.
            generation_events = [
                item
                for item in events
                if item.get("event")
                in {"idea_generation_completed", "idea_generation_failed"}
            ]
        if not generation_events:
            return 0, 0.0
        latest = generation_events[-1]
        timestamp = str(latest.get("timestamp", "") or "")
        try:
            started_at = datetime.fromisoformat(timestamp)
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            ready_at = started_at + timedelta(
                seconds=self.config.limits.generation_interval_sec
            )
            delay = max(0.0, (ready_at - datetime.now(UTC)).total_seconds())
        except ValueError:
            delay = self.config.limits.generation_interval_sec
        return len(generation_events), time.monotonic() + delay

    async def run(
        self,
        *,
        max_seconds: float | None = None,
        until_idle: bool = False,
    ) -> dict[str, Any]:
        self.store.initialize()
        self._started = True
        self.store.event(
            "controller_started",
            system_id=self.config.system_id,
            simulation=self.config.execution.simulation,
            backend=self.config.execution.backend,
        )
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + max(0.0, max_seconds) if max_seconds is not None else None
        )
        try:
            while not self._stop.is_set():
                await self.tick()
                if deadline is not None and loop.time() >= deadline:
                    break
                if until_idle and self.is_idle():
                    break
                await asyncio.sleep(self.config.concurrency.poll_interval_sec)
        finally:
            await self._drain()
            await self.run_backend.close()
            self.store.event("controller_stopped")
            self._started = False
        return self.snapshot()

    async def tick(self) -> None:
        self._collect_finished_tasks()
        self._enforce_token_budget()
        self._admit_candidates()
        self._start_active_idea_tasks()
        self._ensure_candidate_supply()

    def request_stop(self) -> None:
        self._stop.set()

    def is_idle(self) -> bool:
        candidates = self.store.count_ideas(IdeaStatus.CANDIDATE)
        active = self.store.count_ideas(IdeaStatus.ACTIVE)
        generation_done = self._generation_task is None or self._generation_task.done()
        return (
            candidates == 0 and active == 0 and not self._idea_tasks and generation_done
        )

    def _collect_finished_tasks(self) -> None:
        if self._generation_task is not None and self._generation_task.done():
            try:
                self._generation_task.result()
            except Exception as exc:  # noqa: BLE001
                self.store.event(
                    "idea_generation_failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            self._generation_task = None
        for idea_id, task in list(self._idea_tasks.items()):
            if not task.done():
                continue
            self._idea_tasks.pop(idea_id, None)
            try:
                task.result()
            except Exception as exc:  # noqa: BLE001
                idea = self.store.get_idea(idea_id)
                if idea is not None:
                    idea.status = IdeaStatus.QUARANTINED
                    idea.last_reason = f"{type(exc).__name__}: {exc}"
                    idea.next_action = ""
                    self.store.upsert_idea(idea)
                self.store.event(
                    "idea_loop_failed",
                    idea_id=idea_id,
                    error=f"{type(exc).__name__}: {exc}",
                )

    def _ensure_candidate_supply(self) -> None:
        if self._generation_task is not None or self._producer_exhausted:
            return
        if self._token_budget_exhausted():
            return
        if (
            self.config.limits.generation_max_batches > 0
            and self._generation_batches_started
            >= self.config.limits.generation_max_batches
        ):
            return
        if time.monotonic() < self._next_generation_at:
            return
        limits = self.config.limits
        total = self.store.count_ideas()
        if limits.max_total_ideas > 0 and total >= limits.max_total_ideas:
            return
        candidates = self.store.count_ideas(IdeaStatus.CANDIDATE)
        if candidates >= limits.candidate_target:
            return
        requested = min(
            limits.generation_batch_size,
            limits.candidate_target - candidates,
        )
        if limits.max_total_ideas > 0:
            requested = min(requested, limits.max_total_ideas - total)
        if requested <= 0:
            return
        self._next_generation_at = time.monotonic() + limits.generation_interval_sec
        self._generation_batches_started += 1
        self.store.event(
            "idea_generation_started",
            batch_number=self._generation_batches_started,
            requested=requested,
        )
        self._generation_task = asyncio.create_task(
            self._generate(requested),
            name="research-queue-generate",
        )

    async def _generate(self, requested: int) -> None:
        existing = self.store.list_ideas()
        batch: GenerationBatch = await self._call_llm(
            self.producer.generate,
            requested,
            existing=existing,
        )
        if batch.exhausted:
            self._producer_exhausted = True
        accepted = 0
        for proposal in batch.ideas:
            errors = validate_proposal(proposal)
            duplicate = self._duplicate_title(
                proposal.title,
                existing + self.store.list_ideas(),
            )
            if duplicate:
                errors.append(f"duplicate of {duplicate}")
            if errors:
                self.store.event(
                    "candidate_rejected",
                    title=proposal.title,
                    reasons=errors,
                )
                continue
            idea = IdeaRecord.from_proposal(proposal)
            idea.total_tokens += int(batch.usage.get("total_tokens", 0) or 0) // max(
                1, len(batch.ideas)
            )
            self.store.upsert_idea(idea)
            self.store.event(
                "candidate_created",
                idea_id=idea.idea_id,
                title=idea.title,
            )
            accepted += 1
        self.store.event(
            "idea_generation_completed",
            requested=requested,
            generated=len(batch.ideas),
            accepted=accepted,
            exhausted=batch.exhausted,
            usage=batch.usage,
        )

    def _duplicate_title(
        self,
        title: str,
        existing: list[IdeaRecord],
    ) -> str:
        threshold = self.config.limits.duplicate_threshold
        for idea in existing:
            if title_similarity(title, idea.title) >= threshold:
                return idea.idea_id
        return ""

    def _admit_candidates(self) -> None:
        if self._token_budget_exhausted():
            return
        active_count = self.store.count_ideas(IdeaStatus.ACTIVE)
        available = self.config.limits.max_active_ideas - active_count
        if available <= 0:
            return
        candidates = self.store.list_ideas(statuses={IdeaStatus.CANDIDATE})
        for idea in candidates[:available]:
            idea.status = IdeaStatus.ACTIVE
            direct = bool(
                self.config.promotion.enabled
                and self.config.promotion.direct_all_admitted
                and self.promotion_bridge is not None
            )
            idea.next_action = "scientific_gate" if direct else "prepare"
            idea.last_reason = (
                "admitted directly to frozen benchmark"
                if direct
                else "admitted"
            )
            self.store.upsert_idea(idea)
            self.store.event(
                "idea_admitted",
                idea_id=idea.idea_id,
                priority=idea.priority,
            )
            if direct:
                self.store.event(
                    "idea_selected_for_promotion",
                    idea_id=idea.idea_id,
                    conclusion="not_piloted",
                    reason="direct_all_admitted",
                )

    def _start_active_idea_tasks(self) -> None:
        if self._token_budget_exhausted():
            return
        active = self.store.list_ideas(statuses={IdeaStatus.ACTIVE})
        for idea in active:
            if idea.idea_id in self._idea_tasks:
                continue
            self._idea_tasks[idea.idea_id] = asyncio.create_task(
                self._idea_loop(idea.idea_id),
                name=f"research-queue-{idea.idea_id}",
            )

    async def _idea_loop(self, idea_id: str) -> None:
        while not self._stop.is_set():
            idea = self.store.get_idea(idea_id)
            if idea is None or idea.status is not IdeaStatus.ACTIVE:
                return
            action = idea.next_action or "prepare"
            if (
                action in {"prepare", "run", "review"}
                and idea.step_count >= self.config.limits.max_steps_per_idea
            ):
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "maximum prototype steps reached",
                )
                return
            if action == "prepare":
                await self._prepare(idea)
            elif action == "run":
                await self._run(idea)
            elif action == "review":
                await self._review(idea)
            elif action == "scientific_gate":
                await self._scientific_gate(idea)
            elif action == "promote":
                await self._promote(idea)
            else:
                raise ValueError(f"unknown next_action: {action}")

    async def _prepare(self, idea: IdeaRecord) -> None:
        revision = idea.current_revision + 1
        if revision > self.config.limits.max_revisions_per_idea:
            self._conclude(
                idea,
                Conclusion.INCONCLUSIVE,
                "maximum revisions reached",
            )
            return
        previous: dict[str, Any] | None = None
        if idea.current_revision > 0:
            previous = self.store.read_revision(
                idea.idea_id,
                idea.current_revision,
            )
        self.store.event(
            "prepare_started",
            idea_id=idea.idea_id,
            revision=revision,
        )
        feedback = idea.last_reason
        repairs = 0
        total_usage: dict[str, Any] = {}
        while True:
            prepared = await self._call_llm(
                self.preparer.prepare,
                idea,
                revision=revision,
                budget=self.config.budget(idea.current_budget),
                previous_revision=previous,
                feedback=feedback,
            )
            usage_tokens = int(prepared.usage.get("total_tokens", 0) or 0)
            idea.total_tokens += usage_tokens
            total_usage["total_tokens"] = (
                int(total_usage.get("total_tokens", 0) or 0) + usage_tokens
            )
            for name in ("prompt_tokens", "completion_tokens"):
                total_usage[name] = int(total_usage.get(name, 0) or 0) + int(
                    prepared.usage.get(name, 0) or 0
                )
            total_usage["model"] = prepared.usage.get(
                "model",
                total_usage.get("model", ""),
            )
            # Persist metered usage before deterministic validation.
            self.store.upsert_idea(idea)
            errors = self._prepared_revision_errors(prepared, revision)
            if not errors:
                break
            self.store.event(
                "prepare_validation_failed",
                idea_id=idea.idea_id,
                revision=revision,
                repair_attempt=repairs,
                errors=errors,
                usage=prepared.usage,
            )
            if repairs >= self.config.limits.max_prepare_repairs:
                raise ValueError(
                    "prepared revision failed deterministic validation: "
                    + "; ".join(errors)
                )
            repairs += 1
            feedback = (
                "Your generated project failed deterministic validation. "
                "Return a complete corrected project, not a patch. "
                "Do not add undeclared dependencies. Errors:\n- " + "\n- ".join(errors)
            )
        prepared.usage = total_usage
        revision_dir = self.store.revision_dir(idea.idea_id, revision)
        materialize_revision(revision_dir, prepared)
        idea.current_revision = revision
        idea.next_action = "run"
        idea.step_count += 1
        idea.last_reason = f"revision {revision} prepared"
        self.store.upsert_idea(idea)
        self.store.event(
            "prepare_completed",
            idea_id=idea.idea_id,
            revision=revision,
            requested_gpus=prepared.requested_gpus,
            usage=prepared.usage,
            repair_attempts=repairs,
        )

    def _prepared_revision_errors(
        self,
        prepared: PreparedRevision,
        expected_revision: int,
    ) -> list[str]:
        errors: list[str] = []
        if prepared.revision != expected_revision:
            errors.append("preparer returned the wrong revision number")
        if prepared.requested_gpus > self.config.gpu.max_gpus_per_run:
            errors.append("revision exceeds max_gpus_per_run")
        source_files = prepared.plan.get("source_files", {})
        if not isinstance(source_files, Mapping):
            errors.append("revision source_files must be an object")
        else:
            errors.extend(
                validate_python_sources(
                    source_files,
                    allowed_imports=(self.config.execution.allowed_python_imports),
                )
            )
        return errors

    async def _run(self, idea: IdeaRecord) -> None:
        prepared = PreparedRevision.from_mapping(
            self.store.read_revision(
                idea.idea_id,
                idea.current_revision,
            )
        )
        budget = self.config.budget(idea.current_budget)
        if prepared.requested_gpus > 0:
            requested_gpus = min(
                self.config.gpu.max_gpus_per_run,
                max(prepared.requested_gpus, budget.gpus),
            )
        else:
            requested_gpus = 0
        run_id = new_id(f"run-{idea.current_budget.value.lower()}")
        output_dir = self.store.run_dir(idea.idea_id, run_id)
        run = RunRecord(
            run_id=run_id,
            idea_id=idea.idea_id,
            revision=idea.current_revision,
            budget=idea.current_budget,
            requested_gpus=requested_gpus,
            timeout_sec=budget.timeout_sec,
            command=prepared.command,
            output_dir=str(output_dir),
        )
        self.store.upsert_run(run)
        self.store.event(
            "run_waiting",
            idea_id=idea.idea_id,
            run_id=run.run_id,
            budget=run.budget.value,
            requested_gpus=run.requested_gpus,
        )
        async with self._run_slots:
            run.status = RunStatus.RUNNING
            run.started_at = utc_now()
            self.store.upsert_run(run)
            self.store.event(
                "run_started",
                idea_id=idea.idea_id,
                run_id=run.run_id,
                budget=run.budget.value,
                requested_gpus=run.requested_gpus,
            )
            result = await self.run_backend.run(
                run,
                revision_dir=self.store.revision_dir(
                    idea.idea_id,
                    idea.current_revision,
                ),
                output_dir=output_dir,
                env={
                    "RESEARCH_QUEUE_IDEA_ID": idea.idea_id,
                    "RESEARCH_QUEUE_RUN_ID": run.run_id,
                    "RESEARCH_QUEUE_REVISION": str(idea.current_revision),
                    "RESEARCH_QUEUE_BUDGET": idea.current_budget.value,
                    "RESEARCH_QUEUE_OUTPUT_DIR": str(output_dir),
                    "RESEARCH_QUEUE_BUDGET_JSON": json.dumps(
                        budget.to_dict(),
                        ensure_ascii=False,
                    ),
                },
            )
        expected_parameters = budget.parameters
        applied_parameters = result.usage.get("budget_parameters")
        if (
            not self.config.execution.simulation
            and result.ok
            and applied_parameters != expected_parameters
        ):
            result.ok = False
            result.error = (
                "experiment did not attest the exact applied budget "
                f"parameters: expected {expected_parameters!r}, "
                f"got {applied_parameters!r}"
            )
        run.result = result.to_dict()
        run.error = result.error
        run.finished_at = utc_now()
        run.status = RunStatus.SUCCEEDED if result.ok else RunStatus.FAILED
        self.store.upsert_run(run)
        idea.gpu_seconds += float(result.usage.get("gpu_seconds", 0.0) or 0.0)
        idea.step_count += 1
        if result.ok:
            idea.next_action = "review"
            idea.last_reason = "run completed"
            idea.infra_failures = 0
        else:
            idea.infra_failures += 1
            if idea.infra_failures > self.config.limits.max_infra_retries:
                idea.status = IdeaStatus.QUARANTINED
                idea.next_action = ""
                idea.last_reason = result.error or "run failed"
            else:
                idea.next_action = "prepare"
                idea.last_reason = result.error or "run failed; revise"
        self.store.upsert_idea(idea)
        self.store.event(
            "run_completed",
            idea_id=idea.idea_id,
            run_id=run.run_id,
            status=run.status.value,
            result=result.to_dict(),
        )

    async def _review(self, idea: IdeaRecord) -> None:
        history = self.store.list_runs(idea_id=idea.idea_id)
        if not history:
            raise RuntimeError("review requested without run history")
        latest = history[-1]
        self.store.event(
            "review_started",
            idea_id=idea.idea_id,
            run_id=latest.run_id,
        )
        limits = {
            "max_revisions_per_idea": (self.config.limits.max_revisions_per_idea),
            "max_runs_per_budget": (self.config.limits.max_runs_per_budget),
            "max_steps_per_idea": self.config.limits.max_steps_per_idea,
            "remaining_steps_after_review": max(
                0,
                self.config.limits.max_steps_per_idea - (idea.step_count + 1),
            ),
        }
        decision: ReviewDecision = await self._call_llm(
            self.reviewer.review,
            idea,
            run=latest,
            history=history,
            limits=limits,
        )
        idea.total_tokens += int(decision.usage.get("total_tokens", 0) or 0)
        idea.step_count += 1
        # Keep provider usage durable even if a future deterministic review
        # transition rejects the returned decision.
        self.store.upsert_idea(idea)
        self._apply_review(idea, decision, history)
        self.store.event(
            "review_completed",
            idea_id=idea.idea_id,
            run_id=latest.run_id,
            decision=decision.to_dict(),
        )

    def _apply_review(
        self,
        idea: IdeaRecord,
        decision: ReviewDecision,
        history: list[RunRecord],
    ) -> None:
        if decision.action is ReviewAction.RUN_MORE:
            if not self._has_step_budget(idea, required=1):
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "run_more requires another Run but no step budget remains",
                )
                return
            count = sum(
                1
                for run in history
                if run.budget is idea.current_budget
                and run.revision == idea.current_revision
                and run.status is RunStatus.SUCCEEDED
            )
            if count >= self.config.limits.max_runs_per_budget:
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "run_more exceeded max_runs_per_budget",
                )
                return
            idea.next_action = "run"
        elif decision.action is ReviewAction.ESCALATE:
            if not self._has_step_budget(idea, required=2):
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "escalation requires a Run and Review but insufficient "
                    "step budget remains",
                )
                return
            expected = idea.current_budget.next()
            if expected is None or decision.next_budget is not expected:
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "invalid budget escalation",
                )
                return
            idea.current_budget = expected
            idea.next_action = "run"
        elif decision.action is ReviewAction.REVISE:
            if not self._has_step_budget(idea, required=3):
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "revision requires Prepare, Run, and Review but "
                    "insufficient step budget remains",
                )
                return
            if idea.current_revision >= self.config.limits.max_revisions_per_idea:
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "revision limit reached",
                )
                return
            idea.current_budget = BudgetLevel.B0
            idea.next_action = "prepare"
        elif decision.action is ReviewAction.CONCLUDE:
            conclusion = decision.conclusion or Conclusion.INCONCLUSIVE
            if self._should_promote(idea, conclusion):
                idea.conclusion = conclusion
                idea.last_reason = decision.reason
                idea.next_action = "scientific_gate"
                self.store.upsert_idea(idea)
                self.store.event(
                    "idea_selected_for_promotion",
                    idea_id=idea.idea_id,
                    conclusion=conclusion.value,
                    reason=decision.reason,
                )
                return
            self._conclude(
                idea,
                conclusion,
                decision.reason,
            )
            return
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported decision {decision.action}")
        idea.last_reason = decision.reason
        self.store.upsert_idea(idea)

    def _should_promote(
        self,
        idea: IdeaRecord,
        conclusion: Conclusion,
    ) -> bool:
        if self.promotion_bridge is None or not self.config.promotion.enabled:
            return False
        if idea.priority < self.config.promotion.minimum_priority:
            return False
        if conclusion.value not in set(self.config.promotion.trigger_conclusions):
            return False
        selected = {
            str(event.get("idea_id", ""))
            for event in self.store.list_events(limit=100000)
            if event.get("event") == "idea_selected_for_promotion"
        }
        return len(selected) < self.config.promotion.max_promotions

    async def _scientific_gate(self, idea: IdeaRecord) -> None:
        if self.spec_worker is None:
            self._conclude(
                idea,
                Conclusion.INCONCLUSIVE,
                "scientific gate enabled without a ResearchSpec worker",
            )
            return
        benchmark_root = self.store.idea_dir(idea.idea_id) / "benchmark"
        benchmark_root.mkdir(parents=True, exist_ok=True)
        self.store.event(
            "scientific_gate_started",
            idea_id=idea.idea_id,
            benchmark_id=self.config.promotion.benchmark_id,
        )
        feedback = ""
        usage: dict[str, Any] = {}
        result = None
        compatibility = None
        benchmark_profile = (
            self.promotion_bridge.profile_dict()
            if self.promotion_bridge is not None
            else {}
        )
        for attempt in range(self.config.scientific_gate.max_repairs + 1):
            spec, current_usage = await self._call_llm(
                self.spec_worker.build,
                idea,
                benchmark_id=self.config.promotion.benchmark_id,
                treatment_api=TREATMENT_API,
                benchmark_profile=benchmark_profile,
                feedback=feedback,
            )
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[name] = int(usage.get(name, 0) or 0) + int(
                    current_usage.get(name, 0) or 0
                )
            if current_usage.get("model"):
                usage["model"] = current_usage["model"]
            idea.total_tokens += int(current_usage.get("total_tokens", 0) or 0)
            self.store.upsert_idea(idea)
            result = validate_research_spec(
                spec,
                benchmark_id=self.config.promotion.benchmark_id,
            )
            compatibility = (
                self.promotion_bridge.compatibility(spec)
                if self.promotion_bridge is not None
                else None
            )
            compatibility_errors = (
                list(compatibility.errors) if compatibility is not None else []
            )
            self.store.write_json_atomic(
                benchmark_root / f"scientific-gate-{attempt + 1:02d}.json",
                {
                    "research_spec": spec.to_dict(),
                    "research_spec_gate": result.to_dict(),
                    "benchmark_compatibility": (
                        compatibility.to_dict()
                        if compatibility is not None
                        else {
                            "passed": False,
                            "errors": ["benchmark promotion bridge is unavailable"],
                            "checks": {},
                        }
                    ),
                    "usage": current_usage,
                },
            )
            if result.passed and compatibility is not None and compatibility.passed:
                idea.research_spec = spec
                self.store.write_json_atomic(
                    benchmark_root / "research_spec.json",
                    spec.to_dict(),
                )
                self.store.write_json_atomic(
                    benchmark_root / "benchmark_plan.json",
                    self.promotion_bridge.benchmark_plan,
                )
                self.store.write_json_atomic(
                    benchmark_root / "benchmark_compatibility.json",
                    compatibility.to_dict(),
                )
                break
            all_errors = [*result.errors, *compatibility_errors]
            feedback = (
                "Your ResearchSpec failed deterministic validation. Correct "
                "only the listed issues and return a complete ResearchSpec. "
                "Errors: " + "; ".join(all_errors)
            )
        if (
            result is None
            or not result.passed
            or compatibility is None
            or not compatibility.passed
            or idea.research_spec is None
        ):
            gate_errors = list(result.errors) if result is not None else []
            if compatibility is not None:
                gate_errors.extend(compatibility.errors)
            reason = (
                "; ".join(gate_errors)
                if gate_errors
                else "ResearchSpec generation failed"
            )
            self.store.event(
                "scientific_gate_rejected",
                idea_id=idea.idea_id,
                reason=reason,
                usage=usage,
            )
            self._conclude(
                idea,
                Conclusion.INCONCLUSIVE,
                f"scientific contract rejected: {reason}",
            )
            return
        idea.next_action = "promote"
        idea.last_reason = "scientific contract accepted"
        idea.step_count += 1
        self.store.upsert_idea(idea)
        self.store.event(
            "scientific_gate_passed",
            idea_id=idea.idea_id,
            gate={
                "research_spec": result.to_dict(),
                "benchmark_compatibility": compatibility.to_dict(),
            },
            usage=usage,
        )

    async def _promote(self, idea: IdeaRecord) -> None:
        if self.promotion_bridge is None or idea.research_spec is None:
            self._conclude(
                idea,
                Conclusion.INCONCLUSIVE,
                "promotion requested without bridge or ResearchSpec",
            )
            return
        self.store.event(
            "promotion_started",
            idea_id=idea.idea_id,
            benchmark_id=self.config.promotion.benchmark_id,
        )
        outcome = await self.promotion_bridge.promote(
            idea,
            spec=idea.research_spec,
        )
        idea.total_tokens += int(
            outcome.usage.get("total_tokens", 0) or 0
        )
        idea.step_count += 1
        final_conclusion = (
            Conclusion.POSITIVE
            if outcome.hypothesis_supported is True
            and outcome.scientific_valid
            else (
                Conclusion.NEGATIVE
                if outcome.hypothesis_supported is False
                and outcome.scientific_valid
                else Conclusion.INCONCLUSIVE
            )
        )
        idea.conclusion = final_conclusion
        self.store.write_json_atomic(
            self.store.idea_dir(idea.idea_id) / "final_review.json",
            outcome.to_dict(),
        )
        self._conclude(
            idea,
            final_conclusion,
            outcome.reason,
        )

    def _has_step_budget(self, idea: IdeaRecord, *, required: int) -> bool:
        return (
            idea.step_count + max(0, int(required))
            <= self.config.limits.max_steps_per_idea
        )

    def _conclude(
        self,
        idea: IdeaRecord,
        conclusion: Conclusion,
        reason: str,
    ) -> None:
        idea.status = IdeaStatus.CONCLUDED
        idea.conclusion = conclusion
        idea.next_action = ""
        idea.last_reason = reason
        self.store.upsert_idea(idea)
        self.store.event(
            "idea_concluded",
            idea_id=idea.idea_id,
            conclusion=conclusion.value,
            reason=reason,
        )
        self._write_research_note(idea)

    def _audit_usage(self) -> dict[str, Any]:
        totals = {
            "calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        by_tier: dict[str, dict[str, Any]] = {}
        by_model: dict[str, dict[str, Any]] = {}
        by_role: dict[str, dict[str, Any]] = {}
        audit_root = self.config.root / "llm-audit"
        for path in sorted(audit_root.glob("*/calls.jsonl*")):
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            fallback_tier = path.parent.name or "unknown"
            for line in lines:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # A concurrently appended final line is retried on the
                    # next snapshot/tick after it becomes complete.
                    continue
                if not isinstance(value, Mapping):
                    continue
                tier = str(value.get("tier", fallback_tier) or fallback_tier)
                model = str(value.get("model", "unknown") or "unknown")
                role = str(value.get("role", "unknown") or "unknown")
                for table, key in (
                    (by_tier, tier),
                    (by_model, model),
                    (by_role, role),
                ):
                    row = table.setdefault(
                        key,
                        {
                            "calls": 0,
                            "successful_calls": 0,
                            "failed_calls": 0,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    )
                    self._add_audit_call(row, value)
                self._add_audit_call(totals, value)
        return {
            **totals,
            "by_tier": self._usage_rows(by_tier, "tier"),
            "by_model": self._usage_rows(by_model, "model"),
            "by_role": self._usage_rows(by_role, "role"),
        }

    @staticmethod
    def _add_audit_call(
        row: dict[str, Any],
        value: Mapping[str, Any],
    ) -> None:
        row["calls"] += 1
        if value.get("outcome") == "success":
            row["successful_calls"] += 1
        else:
            row["failed_calls"] += 1
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            row[field] += max(0, int(value.get(field, 0) or 0))

    @staticmethod
    def _usage_rows(
        table: Mapping[str, Mapping[str, Any]],
        key_name: str,
    ) -> list[dict[str, Any]]:
        return [{key_name: key, **dict(value)} for key, value in sorted(table.items())]

    def _usage_snapshot(self) -> dict[str, Any]:
        ideas = self.store.list_ideas()
        idea_accounted_tokens = sum(max(0, idea.total_tokens) for idea in ideas)
        audit = self._audit_usage()
        audit_tokens = int(audit["total_tokens"])
        # Both sources describe the same calls. Use the greater durable view
        # rather than summing them: Idea accounting gives per-Idea attribution,
        # while audit accounting also captures rejected generations and model
        # responses that fail before a result object can be persisted.
        total_tokens = max(idea_accounted_tokens, audit_tokens)
        return {
            "total_tokens": total_tokens,
            "idea_accounted_tokens": idea_accounted_tokens,
            "audit_tokens": audit_tokens,
            "max_total_tokens": self.config.limits.max_total_tokens,
            "token_budget_exhausted": (
                self.config.limits.max_total_tokens > 0
                and total_tokens >= self.config.limits.max_total_tokens
            ),
            "gpu_seconds": sum(max(0.0, idea.gpu_seconds) for idea in ideas),
            "llm": audit,
        }

    def _total_tokens(self) -> int:
        return int(self._usage_snapshot()["total_tokens"])

    def _token_budget_exhausted(self) -> bool:
        limit = self.config.limits.max_total_tokens
        return limit > 0 and self._total_tokens() >= limit

    def _enforce_token_budget(self) -> None:
        if not self._token_budget_exhausted():
            return
        if not self._stop.is_set():
            self.store.event(
                "token_budget_exhausted",
                total_tokens=self._total_tokens(),
                max_total_tokens=self.config.limits.max_total_tokens,
            )
        self._stop.set()

    def _write_research_note(self, idea: IdeaRecord) -> Path:
        runs = self.store.list_runs(idea_id=idea.idea_id)
        lines = [
            f"# {idea.title}",
            "",
            f"- **Idea ID:** `{idea.idea_id}`",
            f"- **Status:** `{idea.status.value}`",
            f"- **Conclusion:** `{idea.conclusion.value if idea.conclusion else ''}`",
            f"- **Revisions:** {idea.current_revision}",
            f"- **Total tokens:** {idea.total_tokens}",
            f"- **GPU seconds:** {idea.gpu_seconds:.3f}",
            "",
            "## Research question",
            "",
            idea.question,
            "",
            "## Hypothesis",
            "",
            idea.hypothesis,
            "",
            "## Treatment",
            "",
            idea.treatment,
            "",
            "## Control",
            "",
            idea.control,
            "",
            "## Primary metric",
            "",
            idea.primary_metric,
            "",
            "## Runs",
            "",
            "| Run | Revision | Budget | Status | GPUs | Result |",
            "|---|---:|---|---|---:|---|",
        ]
        for run in runs:
            metrics = run.result.get("metrics", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{run.run_id}`",
                        str(run.revision),
                        run.budget.value,
                        run.status.value,
                        str(run.requested_gpus),
                        f"`{json.dumps(metrics, ensure_ascii=False)}`",
                    ]
                )
                + " |"
            )
        lines += [
            "",
            "## Final reason",
            "",
            idea.last_reason,
            "",
            "## Provenance",
            "",
            "- Idea record: `idea.json`",
            "- Revisions: `revisions/`",
            "- Raw runs: `runs/`",
        ]
        path = self.store.idea_dir(idea.idea_id) / "research_note.md"
        if idea.research_spec is not None:
            lines += [
                "",
                "## ResearchSpec",
                "",
                "```json",
                json.dumps(
                    idea.research_spec.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
                "```",
            ]
        final_review_path = self.store.idea_dir(idea.idea_id) / "final_review.json"
        if final_review_path.is_file():
            lines += [
                "",
                "## Real benchmark final review",
                "",
                "```json",
                final_review_path.read_text(encoding="utf-8").strip(),
                "```",
            ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        if self.research_memory is not None:
            result = self.research_memory.reconcile(idea)
            self.store.event(
                (
                    "research_memory_synced"
                    if result.ok
                    else "research_memory_sync_failed"
                ),
                idea_id=idea.idea_id,
                external_id=result.external_id,
                error=result.error,
            )
        return path

    async def _drain(self) -> None:
        tasks = list(self._idea_tasks.values())
        if self._generation_task is not None:
            tasks.append(self._generation_task)
        if not tasks:
            return
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for idea_id in list(self._idea_tasks):
            idea = self.store.get_idea(idea_id)
            if idea is not None and idea.status is IdeaStatus.ACTIVE:
                idea.last_reason = "controller stopped during active work"
                self.store.upsert_idea(idea)
        self._idea_tasks.clear()
        self._generation_task = None

    def snapshot(self) -> dict[str, Any]:
        store = self.store.snapshot()
        usage = self._usage_snapshot()
        conclusions = Counter(
            idea.conclusion.value
            for idea in self.store.list_ideas(statuses={IdeaStatus.CONCLUDED})
            if idea.conclusion is not None
        )
        return {
            "system_id": self.config.system_id,
            "started": self._started,
            "state": store,
            "conclusions": dict(conclusions),
            "active_idea_tasks": sorted(self._idea_tasks),
            "generation_running": bool(
                self._generation_task is not None and not self._generation_task.done()
            ),
            "generation_batches_started": self._generation_batches_started,
            "generation_max_batches": (self.config.limits.generation_max_batches),
            "next_generation_in_sec": max(
                0.0,
                self._next_generation_at - time.monotonic(),
            ),
            "producer_exhausted": self._producer_exhausted,
            "path_reachability": self.config.path_reachability(),
            "usage": usage,
            "execution": self.run_backend.snapshot(),
        }
