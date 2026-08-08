from __future__ import annotations

from researchclaw.research_queue.models import (
    BudgetLevel,
    IdeaProposal,
    IdeaRecord,
    RunRecord,
    RunStatus,
)
from researchclaw.research_queue.store import ResearchQueueStore


def test_store_persists_idea_run_and_events(tmp_path) -> None:
    store = ResearchQueueStore(tmp_path)
    store.initialize()
    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Store test",
            question="Question?",
            hypothesis="Hypothesis",
            treatment="A",
            control="B",
            primary_metric="score",
        )
    )
    store.upsert_idea(idea)
    run = RunRecord(
        run_id="run-1",
        idea_id=idea.idea_id,
        revision=1,
        budget=BudgetLevel.B0,
        requested_gpus=1,
        timeout_sec=10,
        command=("python", "experiment.py"),
        output_dir=str(tmp_path / "out"),
        status=RunStatus.RUNNING,
    )
    store.upsert_run(run)
    store.event("test_event", idea_id=idea.idea_id, run_id=run.run_id)

    assert store.get_idea(idea.idea_id) == idea
    assert store.get_run(run.run_id) == run
    assert store.list_events()[-1]["event"] == "test_event"
    assert store.events_path.is_file()
