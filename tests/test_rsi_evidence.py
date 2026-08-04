from __future__ import annotations

import json
from pathlib import Path

import pytest

import researchclaw.rsi.evidence as evidence_module
from researchclaw.pipeline.stages import StageStatus
from researchclaw.rsi.evidence import (
    collect_evidence,
    compare_evidence,
    evidence_succeeded,
    export_evidence_json,
    load_stage_results,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _complete_run(run_dir: Path) -> None:
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "run_id": "rc-good",
            "stages_executed": 23,
            "stages_done": 23,
            "stages_failed": 0,
            "stages_paused": 0,
            "stages_blocked": 0,
            "final_status": "done",
            "degraded": False,
        },
    )
    _write_json(
        run_dir / "stage-14" / "experiment_summary.json",
        {
            "total_runs": 12,
            "total_conditions": 3,
            "metrics_summary": {
                "accuracy": {"mean": 0.92, "count": 12},
                "f1": {"mean": 0.90, "count": 12},
            },
            "condition_summaries": {
                "baseline": {"metrics": {"accuracy": {"mean": 0.84}}},
                "method": {"metrics": {"accuracy": {"mean": 0.92}}},
                "ablation": {"metrics": {"accuracy": {"mean": 0.88}}},
            },
            "best_run": {"status": "done", "metrics": {"accuracy": 0.95}},
            "paired_comparisons": [{"name": "method-baseline"}],
        },
    )
    paper = "\n\n".join(
        [
            "# A Careful Study",
            "## Abstract\n" + "abstract " * 250,
            "## Introduction\n" + "introduction " * 750,
            "## Related Work\n" + "related " * 550,
            "## Method\n" + "method " * 850,
            "## Experiments\n" + "experiment " * 700,
            "## Results\n" + "result " * 650,
            "## Discussion\n" + "discussion " * 500,
            "## Conclusion\n" + "conclusion " * 150,
        ]
    )
    _write_text(run_dir / "stage-17" / "paper_draft.md", paper)
    _write_json(
        run_dir / "stage-17" / "draft_quality.json",
        {
            "section_analysis": [
                {"heading": name, "status": "ok"}
                for name in (
                    "Abstract",
                    "Introduction",
                    "Related Work",
                    "Method",
                    "Experiments",
                    "Results",
                    "Discussion",
                    "Conclusion",
                )
            ],
            "overall_warnings": [],
            "revision_directives": [],
        },
    )
    _write_text(
        run_dir / "stage-18" / "reviews.md",
        "\n".join(
            [
                "# Reviewer A",
                "Strengths: grounded experiments and clear analysis. " * 30,
                "Recommendation: Accept",
                "# Reviewer B",
                "Strengths: reproducible and carefully scoped. " * 30,
                "Recommendation: Minor Revision",
                "# Reviewer C",
                "Strengths: strong statistical evidence. " * 30,
                "Recommendation: Accept",
            ]
        ),
    )
    _write_json(
        run_dir / "stage-20" / "quality_report.json",
        {
            "score_1_to_10": 9.0,
            "verdict": "proceed",
            "strengths": ["rigorous"],
            "weaknesses": [],
            "required_actions": [],
        },
    )
    _write_json(
        run_dir / "stage-20" / "fabrication_flags.json",
        {
            "fabrication_suspected": False,
            "has_real_data": True,
            "verified_values_count": 18,
        },
    )
    _write_json(
        run_dir / "stage-23" / "verification_report.json",
        {
            "summary": {
                "total": 20,
                "verified": 19,
                "suspicious": 1,
                "hallucinated": 0,
                "skipped": 0,
                "integrity_score": 0.95,
            },
            "results": [],
        },
    )


def test_collect_evidence_reconstructs_pipeline_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    stage = run_dir / "stage-20"
    stage.mkdir(parents=True)
    _write_json(
        run_dir / "pipeline_summary.json",
        {
            "run_id": "rc-1",
            "stages_failed": 1,
            "stages_paused": 0,
            "stages_blocked": 0,
            "final_status": "failed",
        },
    )
    _write_json(
        stage / "decision.json",
        {
            "status": "failed",
            "decision": "retry",
            "error": "quality below threshold",
            "output_artifacts": ["quality_report.json"],
            "evidence_refs": ["stage-20/quality_report.json"],
        },
    )
    _write_json(
        stage / "quality_report.json",
        {"score_1_to_10": "4.5", "weaknesses": ["weak ablation"]},
    )

    results = load_stage_results(run_dir)
    assert len(results) == 1
    assert results[0].status is StageStatus.FAILED
    assert results[0].error == "quality below threshold"

    evidence = collect_evidence(
        run_dir,
        pipeline_returncode=1,
        command=["researchclaw", "run"],
    )
    assert evidence["quality_score"] == 4.5
    assert evidence["failures"][0]["stage_name"] == "QUALITY_GATE"
    assert evidence["lessons"][0]["severity"] == "error"
    assert evidence["comparison"]["decision"] == "reject"
    assert evidence["composite_score"] < 20
    assert not evidence_succeeded(evidence)
    assert json.loads((run_dir / "rsi_evidence.json").read_text()) == evidence


def test_collects_all_requested_artifacts_and_scores_deterministically(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "complete"
    _complete_run(run_dir)

    first = collect_evidence(
        run_dir,
        pipeline_returncode=0,
        command=["researchclaw", "run"],
        topic_id="calibration-gates",
    )
    second = collect_evidence(
        run_dir,
        pipeline_returncode=0,
        command=["researchclaw", "run"],
    )

    structured = first["structured_evidence"]
    assert structured["pipeline"]["artifact"]["status"] == "ok"
    assert structured["experiment"]["total_runs"] == 12
    assert structured["experiment"]["metric_count"] == 2
    assert structured["paper"]["word_count"] > 4000
    assert structured["review"]["reviewer_count"] == 3
    assert structured["quality_gate"]["score_1_to_10"] == 9.0
    assert structured["citations"]["integrity_score"] == 0.95
    assert first["scorecard"] == second["scorecard"]
    assert first["composite_score"] == 93.12
    assert first["topic_id"] == "calibration-gates"
    assert first["comparison"]["decision"] == "accept"
    assert "establishes baseline" in first["comparison"]["reason"]
    assert evidence_succeeded(first)


def test_missing_and_malformed_artifacts_are_conservative(tmp_path: Path) -> None:
    run_dir = tmp_path / "damaged"
    run_dir.mkdir()
    _write_text(run_dir / "pipeline_summary.json", "{not json")
    _write_text(run_dir / "stage-14" / "experiment_summary.json", "[]")
    _write_text(run_dir / "stage-17" / "paper_draft.md", b"\xff".decode("latin1"))
    _write_text(run_dir / "stage-20" / "quality_report.json", '{"score": NaN}')
    _write_json(
        run_dir / "stage-23" / "verification_report.json",
        {"summary": "unexpected"},
    )

    evidence = collect_evidence(
        run_dir,
        pipeline_returncode=0,
        command=[],
    )
    structured = evidence["structured_evidence"]
    assert structured["pipeline"]["artifact"]["status"] == "malformed"
    assert structured["experiment"]["artifact"]["status"] == "malformed"
    assert structured["review"]["artifact"]["status"] == "missing"
    assert structured["quality_gate"]["score_1_to_10"] is None
    assert structured["citations"]["total"] is None
    assert evidence["composite_score"] < 5
    assert evidence["comparison"]["decision"] == "reject"
    assert "pipeline summary missing or malformed" in evidence["comparison"]["reason"]
    assert evidence["scorecard"]["missing_weight"] >= 0.75


def test_zero_citations_do_not_receive_perfect_credit(tmp_path: Path) -> None:
    run_dir = tmp_path / "zero-citations"
    _complete_run(run_dir)
    _write_json(
        run_dir / "stage-23" / "verification_report.json",
        {
            "summary": {
                "total": 0,
                "verified": 0,
                "suspicious": 0,
                "hallucinated": 0,
                "skipped": 0,
                "integrity_score": 1.0,
            },
            "note": "No references.bib found — nothing to verify.",
        },
    )

    evidence = collect_evidence(
        run_dir,
        pipeline_returncode=0,
        command=[],
    )
    citation_score = evidence["scorecard"]["components"]["citations"]["score"]
    assert citation_score == 0
    assert evidence["structured_evidence"]["citations"]["no_references"] is True


def test_fabrication_or_hallucinated_citations_force_rejection(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "unsafe"
    _complete_run(run_dir)
    _write_json(
        run_dir / "stage-20" / "fabrication_flags.json",
        {"fabrication_suspected": True, "has_real_data": False},
    )
    _write_json(
        run_dir / "stage-23" / "verification_report.json",
        {
            "summary": {
                "total": 20,
                "verified": 18,
                "suspicious": 1,
                "hallucinated": 1,
                "skipped": 0,
                "integrity_score": 0.9,
            }
        },
    )

    evidence = collect_evidence(
        run_dir,
        pipeline_returncode=0,
        command=[],
    )
    assert evidence["scorecard"]["components"]["quality_gate"]["score"] == 0
    assert any(
        "fabrication was suspected" in reason
        for reason in evidence["scorecard"]["hard_failures"]
    )
    assert (
        "citation verification found hallucinated references"
        in evidence["scorecard"]["hard_failures"]
    )
    assert evidence["comparison"]["decision"] == "reject"


def test_compare_evidence_accepts_only_clear_improvement() -> None:
    best = {"composite_score": 80.0, "scorecard": {"hard_failures": []}}
    candidate = {
        "composite_score": 80.5,
        "scorecard": {"hard_failures": []},
    }
    accepted = compare_evidence(candidate, best, min_improvement=0.25)
    assert accepted["decision"] == "accept"
    assert accepted["delta"] == 0.5

    rejected = compare_evidence(candidate, best, min_improvement=1.0)
    assert rejected["decision"] == "reject"
    assert "below required margin" in rejected["reason"]

    tied = compare_evidence(best, best)
    assert tied["decision"] == "reject"

    with pytest.raises(ValueError):
        compare_evidence(candidate, best, min_improvement=-1)


def test_safe_candidate_can_replace_unsafe_higher_scoring_incumbent() -> None:
    unsafe_best = {
        "composite_score": 99.0,
        "scorecard": {"hard_failures": ["hallucinated references"]},
    }
    safe_candidate = {
        "composite_score": 75.0,
        "scorecard": {"hard_failures": []},
    }
    comparison = compare_evidence(safe_candidate, unsafe_best)
    assert comparison["decision"] == "accept"
    assert "incumbent has hard failures" in comparison["reason"]


def test_export_is_atomic_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "evidence.json"
    target.parent.mkdir()
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(evidence_module, "atomic_write_json", fail_replace)
    with pytest.raises(OSError):
        export_evidence_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}


def test_export_evidence_json_writes_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "evidence.json"
    returned = export_evidence_json(target, {"score": 42, "说明": "保守评分"})
    assert returned == target
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "score": 42,
        "说明": "保守评分",
    }
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_evidence_succeeded_requires_done_summary() -> None:
    evidence = {
        "pipeline_returncode": 0,
        "pipeline_summary": {
            "stages_failed": 0,
            "stages_paused": 0,
            "stages_blocked": 0,
            "final_status": "done",
        },
    }
    assert evidence_succeeded(evidence)
    evidence["pipeline_summary"]["stages_blocked"] = 1
    assert not evidence_succeeded(evidence)
    evidence["pipeline_summary"]["stages_blocked"] = "broken"
    assert not evidence_succeeded(evidence)
