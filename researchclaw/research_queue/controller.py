"""Async continuous research loop for the prototype."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
from .store import ResearchQueueStore
from .workers import (
    IdeaProducer,
    PreparationWorker,
    ReviewWorker,
    materialize_revision,
    title_similarity,
    validate_proposal,
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
    ) -> None:
        self.config = config
        self.store = store
        self.producer = producer
        self.preparer = preparer
        self.reviewer = reviewer
        self.run_backend = run_backend
        self._llm_slots = asyncio.Semaphore(config.concurrency.max_llm_jobs)
        self._run_slots = asyncio.Semaphore(config.concurrency.max_run_jobs)
        self._idea_tasks: dict[str, asyncio.Task[None]] = {}
        self._generation_task: asyncio.Task[None] | None = None
        self._producer_exhausted = False
        self._stop = asyncio.Event()
        self._started = False

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
        self._generation_task = asyncio.create_task(
            self._generate(requested),
            name="research-queue-generate",
        )

    async def _generate(self, requested: int) -> None:
        existing = self.store.list_ideas()
        async with self._llm_slots:
            batch: GenerationBatch = await asyncio.to_thread(
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
        active_count = self.store.count_ideas(IdeaStatus.ACTIVE)
        available = self.config.limits.max_active_ideas - active_count
        if available <= 0:
            return
        candidates = self.store.list_ideas(statuses={IdeaStatus.CANDIDATE})
        for idea in candidates[:available]:
            idea.status = IdeaStatus.ACTIVE
            idea.next_action = "prepare"
            idea.last_reason = "admitted"
            self.store.upsert_idea(idea)
            self.store.event(
                "idea_admitted",
                idea_id=idea.idea_id,
                priority=idea.priority,
            )

    def _start_active_idea_tasks(self) -> None:
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
            if idea.step_count >= self.config.limits.max_steps_per_idea:
                self._conclude(
                    idea,
                    Conclusion.INCONCLUSIVE,
                    "maximum prototype steps reached",
                )
                return
            action = idea.next_action or "prepare"
            if action == "prepare":
                await self._prepare(idea)
            elif action == "run":
                await self._run(idea)
            elif action == "review":
                await self._review(idea)
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
        async with self._llm_slots:
            prepared: PreparedRevision = await asyncio.to_thread(
                self.preparer.prepare,
                idea,
                revision=revision,
                budget=self.config.budget(idea.current_budget),
                previous_revision=previous,
                feedback=idea.last_reason,
            )
        if prepared.revision != revision:
            raise ValueError("preparer returned the wrong revision number")
        if prepared.requested_gpus > self.config.gpu.max_gpus_per_run:
            raise ValueError("revision exceeds max_gpus_per_run")
        revision_dir = self.store.revision_dir(idea.idea_id, revision)
        materialize_revision(revision_dir, prepared)
        idea.current_revision = revision
        idea.next_action = "run"
        idea.step_count += 1
        idea.total_tokens += int(prepared.usage.get("total_tokens", 0) or 0)
        idea.last_reason = f"revision {revision} prepared"
        self.store.upsert_idea(idea)
        self.store.event(
            "prepare_completed",
            idea_id=idea.idea_id,
            revision=revision,
            requested_gpus=prepared.requested_gpus,
            usage=prepared.usage,
        )

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
        }
        async with self._llm_slots:
            decision: ReviewDecision = await asyncio.to_thread(
                self.reviewer.review,
                idea,
                run=latest,
                history=history,
                limits=limits,
            )
        idea.total_tokens += int(decision.usage.get("total_tokens", 0) or 0)
        idea.step_count += 1
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
            self._conclude(
                idea,
                decision.conclusion or Conclusion.INCONCLUSIVE,
                decision.reason,
            )
            return
        else:  # pragma: no cover - enum exhaustiveness
            raise ValueError(f"unsupported decision {decision.action}")
        idea.last_reason = decision.reason
        self.store.upsert_idea(idea)

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
        self._write_research_note(idea)
        self.store.event(
            "idea_concluded",
            idea_id=idea.idea_id,
            conclusion=conclusion.value,
            reason=reason,
        )

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
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
            "producer_exhausted": self._producer_exhausted,
            "execution": self.run_backend.snapshot(),
        }
