from __future__ import annotations

import asyncio
import sys

from researchclaw.research_queue.config import ResearchQueueConfig
from researchclaw.research_queue.controller import ResearchQueueController
from researchclaw.research_queue.execution import GPUSlotPool, LocalRunBackend
from researchclaw.research_queue.models import (
    BudgetLevel,
    Conclusion,
    IdeaProposal,
    IdeaStatus,
    PreparedRevision,
    ReviewAction,
    ReviewDecision,
    RunResult,
)
from researchclaw.research_queue.store import ResearchQueueStore
from researchclaw.research_queue.workers import (
    SimulatedPreparationWorker,
    SimulatedReviewWorker,
    StaticIdeaProducer,
)


def _proposal(title: str, scenario: str) -> IdeaProposal:
    return IdeaProposal(
        title=title,
        question=f"Question for {title}?",
        hypothesis=f"Hypothesis for {title}.",
        treatment=f"Treatment for {title}",
        control=f"Control for {title}",
        primary_metric="effect",
        metadata={
            "scenario": scenario,
            "sleep_sec": 0.01,
            "requested_gpus": 1,
        },
    )


def test_simulated_controller_completes_async_closed_loop(tmp_path) -> None:
    config = ResearchQueueConfig.from_mapping(
        {
            "research_queue": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "limits": {
                    "candidate_target": 4,
                    "generation_batch_size": 4,
                    "max_active_ideas": 4,
                    "max_total_ideas": 4,
                    "max_revisions_per_idea": 2,
                    "max_runs_per_budget": 2,
                    "max_steps_per_idea": 12,
                },
                "concurrency": {
                    "max_llm_jobs": 2,
                    "max_run_jobs": 4,
                    "poll_interval_sec": 0.005,
                },
                "execution": {
                    "backend": "local",
                    "simulation": True,
                    "python_executable": sys.executable,
                },
                "gpu": {
                    "max_total_gpus": 2,
                    "max_gpus_per_run": 1,
                },
                "budgets": {
                    "B0": {"gpus": 1, "timeout_sec": 10},
                    "B1": {"gpus": 1, "timeout_sec": 10},
                    "B2": {"gpus": 1, "timeout_sec": 10},
                },
            }
        }
    )
    store = ResearchQueueStore(tmp_path)
    controller = ResearchQueueController(
        config=config,
        store=store,
        producer=StaticIdeaProducer(
            [
                _proposal("Positive path", "positive"),
                _proposal("Negative path", "negative"),
                _proposal("Revision path", "revise"),
                _proposal("Repeat path", "run_more"),
            ]
        ),
        preparer=SimulatedPreparationWorker(python_executable=sys.executable),
        reviewer=SimulatedReviewWorker(),
        run_backend=LocalRunBackend(slot_pool=GPUSlotPool(2)),
    )

    snapshot = asyncio.run(controller.run(max_seconds=10, until_idle=True))

    ideas = store.list_ideas()
    assert len(ideas) == 4
    assert all(idea.status is IdeaStatus.CONCLUDED for idea in ideas)
    conclusions = {idea.conclusion for idea in ideas}
    assert Conclusion.POSITIVE in conclusions
    assert Conclusion.NEGATIVE in conclusions
    assert all(
        (store.idea_dir(idea.idea_id) / "research_note.md").is_file() for idea in ideas
    )
    assert any(idea.current_revision == 2 for idea in ideas)
    assert any(
        len(
            [
                run
                for run in store.list_runs(idea_id=idea.idea_id)
                if run.budget.value == "B0"
            ]
        )
        >= 2
        for idea in ideas
    )
    assert snapshot["state"]["ideas"]["active"] == 0


def test_finished_idea_slot_refills_from_candidate(tmp_path) -> None:
    config = ResearchQueueConfig.from_mapping(
        {
            "research_queue": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "limits": {
                    "candidate_target": 2,
                    "generation_batch_size": 2,
                    "max_active_ideas": 1,
                    "max_total_ideas": 2,
                    "max_steps_per_idea": 6,
                },
                "concurrency": {"poll_interval_sec": 0.005},
                "execution": {
                    "simulation": True,
                    "python_executable": sys.executable,
                },
                "gpu": {
                    "max_total_gpus": 1,
                    "max_gpus_per_run": 1,
                },
                "budgets": {
                    "B0": {"gpus": 1, "timeout_sec": 10},
                    "B1": {"gpus": 1, "timeout_sec": 10},
                    "B2": {"gpus": 1, "timeout_sec": 10},
                },
            }
        }
    )
    store = ResearchQueueStore(tmp_path)
    controller = ResearchQueueController(
        config=config,
        store=store,
        producer=StaticIdeaProducer(
            [
                _proposal("First negative", "negative"),
                _proposal("Second negative", "negative"),
            ]
        ),
        preparer=SimulatedPreparationWorker(python_executable=sys.executable),
        reviewer=SimulatedReviewWorker(),
        run_backend=LocalRunBackend(slot_pool=GPUSlotPool(1)),
    )

    asyncio.run(controller.run(max_seconds=10, until_idle=True))

    events = [item["event"] for item in store.list_events(limit=200)]
    assert events.count("idea_admitted") == 2
    assert events.count("idea_concluded") == 2


def test_finite_static_producer_exhaustion_allows_until_idle(tmp_path) -> None:
    config = ResearchQueueConfig.from_mapping(
        {
            "research_queue": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "limits": {
                    "candidate_target": 3,
                    "generation_batch_size": 2,
                    "max_active_ideas": 1,
                    "max_total_ideas": 0,
                    "max_steps_per_idea": 6,
                },
                "concurrency": {"poll_interval_sec": 0.005},
                "execution": {
                    "simulation": True,
                    "python_executable": sys.executable,
                },
                "gpu": {
                    "max_total_gpus": 1,
                    "max_gpus_per_run": 1,
                },
                "budgets": {
                    "B0": {"gpus": 1, "timeout_sec": 10},
                    "B1": {"gpus": 1, "timeout_sec": 10},
                    "B2": {"gpus": 1, "timeout_sec": 10},
                },
            }
        }
    )
    store = ResearchQueueStore(tmp_path)
    controller = ResearchQueueController(
        config=config,
        store=store,
        producer=StaticIdeaProducer([_proposal("Only idea", "negative")]),
        preparer=SimulatedPreparationWorker(python_executable=sys.executable),
        reviewer=SimulatedReviewWorker(),
        run_backend=LocalRunBackend(slot_pool=GPUSlotPool(1)),
    )

    snapshot = asyncio.run(controller.run(max_seconds=5, until_idle=True))

    ideas = store.list_ideas()
    assert len(ideas) == 1
    assert ideas[0].status is IdeaStatus.CONCLUDED
    assert snapshot["producer_exhausted"] is True
    generation_events = [
        item
        for item in store.list_events(limit=200)
        if item["event"] == "idea_generation_completed"
    ]
    assert len(generation_events) == 1
    assert generation_events[0]["exhausted"] is True


def test_budget_escalation_changes_gpu_count_and_timeout(tmp_path) -> None:
    class FixedPreparationWorker:
        def __init__(self, requested_gpus: int) -> None:
            self.requested_gpus = requested_gpus

        def prepare(
            self,
            idea,
            *,
            revision,
            budget,
            previous_revision,
            feedback,
        ):
            del idea, previous_revision, feedback
            return PreparedRevision(
                revision=revision,
                command=(sys.executable, "experiment.py"),
                requested_gpus=self.requested_gpus,
                timeout_sec=1,
                plan={
                    "source_files": {
                        "experiment.py": (
                            "from pathlib import Path\n"
                            "Path('unused').write_text('unused')\n"
                        )
                    }
                },
            )

    class EscalatingReviewer:
        def review(self, idea, *, run, history, limits):
            del idea, history, limits
            if run.budget is BudgetLevel.B0:
                return ReviewDecision(
                    action=ReviewAction.ESCALATE,
                    reason="promote",
                    next_budget=BudgetLevel.B1,
                )
            return ReviewDecision(
                action=ReviewAction.CONCLUDE,
                reason="done",
                conclusion=Conclusion.POSITIVE,
            )

    class RecordingBackend:
        def __init__(self) -> None:
            self.runs = []

        async def run(self, run, *, revision_dir, output_dir, env):
            del revision_dir, output_dir, env
            self.runs.append(run)
            return RunResult(
                ok=True,
                metrics={"effect": 0.1},
                usage={"gpu_count": run.requested_gpus},
            )

        async def close(self):
            return None

        def snapshot(self):
            return {"runs": len(self.runs)}

    async def execute(requested_gpus: int, state_name: str):
        state_dir = tmp_path / state_name
        config = ResearchQueueConfig.from_mapping(
            {
                "research_queue": {
                    "enabled": True,
                    "state_dir": str(state_dir),
                    "limits": {
                        "candidate_target": 1,
                        "generation_batch_size": 1,
                        "max_active_ideas": 1,
                        "max_total_ideas": 1,
                        "max_steps_per_idea": 8,
                    },
                    "concurrency": {"poll_interval_sec": 0.001},
                    "execution": {"simulation": True},
                    "gpu": {
                        "max_total_gpus": 4,
                        "max_gpus_per_run": 4,
                    },
                    "budgets": {
                        "B0": {"gpus": 1, "timeout_sec": 10},
                        "B1": {"gpus": 3, "timeout_sec": 90},
                        "B2": {"gpus": 4, "timeout_sec": 180},
                    },
                }
            }
        )
        store = ResearchQueueStore(state_dir)
        backend = RecordingBackend()
        controller = ResearchQueueController(
            config=config,
            store=store,
            producer=StaticIdeaProducer([_proposal("Escalate", "positive")]),
            preparer=FixedPreparationWorker(requested_gpus),
            reviewer=EscalatingReviewer(),
            run_backend=backend,
        )
        await controller.run(max_seconds=5, until_idle=True)
        return backend.runs

    gpu_runs = asyncio.run(execute(1, "gpu"))
    assert [(run.requested_gpus, run.timeout_sec) for run in gpu_runs] == [
        (1, 10),
        (3, 90),
    ]

    cpu_runs = asyncio.run(execute(0, "cpu"))
    assert [(run.requested_gpus, run.timeout_sec) for run in cpu_runs] == [
        (0, 10),
        (0, 90),
    ]
