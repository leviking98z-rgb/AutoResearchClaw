from __future__ import annotations

import json

from researchclaw.research_queue.models import (
    BudgetLevel,
    Conclusion,
    IdeaProposal,
    IdeaRecord,
    IdeaStatus,
    RunRecord,
    RunStatus,
)
from researchclaw.research_queue.portfolio import (
    build_portfolio_report,
    compare_portfolio_reports,
    write_portfolio_report,
)
from researchclaw.research_queue.store import ResearchQueueStore


def _concluded_idea(
    store: ResearchQueueStore,
    *,
    title: str,
    tokens: int,
    gpu_seconds: float,
) -> IdeaRecord:
    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title=title,
            question="Question?",
            hypothesis="Hypothesis.",
            treatment="Treatment.",
            control="Control.",
            primary_metric="metric",
        )
    )
    idea.status = IdeaStatus.CONCLUDED
    idea.conclusion = Conclusion.INCONCLUSIVE
    idea.total_tokens = tokens
    idea.gpu_seconds = gpu_seconds
    store.upsert_idea(idea)
    store.event("idea_admitted", idea_id=idea.idea_id)
    store.event("idea_selected_for_promotion", idea_id=idea.idea_id)
    return idea


def test_portfolio_report_counts_only_valid_conclusive_outcomes(tmp_path) -> None:
    store = ResearchQueueStore(tmp_path / "state")
    store.initialize()
    valid = _concluded_idea(
        store,
        title="Valid negative",
        tokens=100,
        gpu_seconds=10.0,
    )
    invalid = _concluded_idea(
        store,
        title="Invalid accept",
        tokens=200,
        gpu_seconds=20.0,
    )
    for idea, review in (
        (
            valid,
            {
                "execution_passed": True,
                "scientific_valid": True,
                "hypothesis_supported": False,
                "promotion_decision": "reject",
                "reason": "valid negative",
            },
        ),
        (
            invalid,
            {
                "execution_passed": True,
                "scientific_valid": False,
                "hypothesis_supported": True,
                "promotion_decision": "accept",
                "reason": "invalid evidence",
            },
        ),
    ):
        store.write_json_atomic(
            store.idea_dir(idea.idea_id) / "final_review.json",
            review,
        )
        store.event("benchmark_completed", idea_id=idea.idea_id)
        store.event(
            "idea_concluded",
            idea_id=idea.idea_id,
            conclusion=idea.conclusion.value,
        )
    store.upsert_run(
        RunRecord(
            run_id="run-b0",
            idea_id=valid.idea_id,
            revision=1,
            budget=BudgetLevel.B0,
            requested_gpus=0,
            timeout_sec=10,
            command=("python", "experiment.py"),
            output_dir=str(tmp_path / "run-b0"),
            status=RunStatus.SUCCEEDED,
        )
    )
    store.upsert_run(
        RunRecord(
            run_id="run-b2",
            idea_id=valid.idea_id,
            revision=1,
            budget=BudgetLevel.B2,
            requested_gpus=1,
            timeout_sec=10,
            command=("python", "benchmark.py"),
            output_dir=str(tmp_path / "run-b2"),
            status=RunStatus.SUCCEEDED,
        )
    )

    report = build_portfolio_report(store, system_id="candidate")

    assert report["summary"]["vco"] == 1
    assert report["summary"]["valid_negative"] == 1
    assert report["summary"]["valid_positive"] == 0
    assert report["summary"]["tokens_per_vco"] == 300
    assert report["summary"]["gpu_seconds_per_vco"] == 30
    assert report["summary"]["invalid_accepts"] == 1
    assert report["summary"]["false_accept_rate"] == 1.0
    assert report["funnel"]["reached_B0"] == 1
    assert report["funnel"]["reached_B2"] == 1

    output = tmp_path / "report"
    write_portfolio_report(store, output, system_id="candidate")
    assert json.loads((output / "summary.json").read_text())["vco"] == 1
    assert (output / "timeline.json").is_file()


def test_portfolio_comparison_reports_directional_improvement() -> None:
    baseline = {
        "summary": {
            "system_id": "baseline",
            "vco_at_window": 0,
            "ttfv_seconds": None,
            "tokens_per_vco": None,
            "gpu_seconds_per_vco": None,
            "false_accept_rate": 0.5,
        }
    }
    candidate = {
        "summary": {
            "system_id": "candidate",
            "vco_at_window": 2,
            "ttfv_seconds": 1200,
            "tokens_per_vco": 1000,
            "gpu_seconds_per_vco": 50,
            "false_accept_rate": 0.0,
        }
    }

    comparison = compare_portfolio_reports(baseline, candidate)

    assert comparison["metrics"]["vco_at_window"]["improved"] is True
    assert comparison["metrics"]["vco_at_window"]["absolute_delta"] == 2
    assert comparison["metrics"]["false_accept_rate"]["improved"] is True
