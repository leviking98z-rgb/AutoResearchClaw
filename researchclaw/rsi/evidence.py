"""Deterministic quality evidence and scorecards for RSI pipeline cycles.

The collector intentionally treats absent or malformed artifacts as missing
evidence.  It never infers that an experiment, review, quality gate, or
citation check succeeded merely because another artifact exists.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from researchclaw.evolution import EvolutionStore, LessonEntry, extract_lessons
from researchclaw.pipeline._helpers import StageResult
from researchclaw.pipeline.stages import Stage, StageStatus

from .storage import atomic_write_json

SCORE_VERSION = "rsi-evidence-v1"
SCORE_WEIGHTS: dict[str, float] = {
    "pipeline": 0.20,
    "experiment": 0.25,
    "paper": 0.15,
    "review": 0.10,
    "quality_gate": 0.20,
    "citations": 0.10,
}
DEFAULT_ACCEPT_MARGIN = 0.25

_BAD_STAGE_STATUSES = {
    StageStatus.FAILED,
    StageStatus.PAUSED,
    StageStatus.BLOCKED_APPROVAL,
    StageStatus.REJECTED,
}
_NEGATIVE_RECOMMENDATIONS = {"reject", "major", "major revision", "desk reject"}
_POSITIVE_RECOMMENDATIONS = {"accept", "minor", "minor revision", "weak accept"}
_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S+", re.MULTILINE)
_REVIEWER_RE = re.compile(r"(?im)^\s*#{1,6}\s*reviewer\b")
_RECOMMENDATION_RE = re.compile(r"(?im)\brecommendation\s*:\s*([^\n#]+)")
_CRITICAL_RE = re.compile(
    r"\b(critical|fabricat(?:ed|ion)|unsupported claim|desk reject)\b",
    re.IGNORECASE,
)


def _json_file(path: Path) -> tuple[dict[str, Any], str]:
    """Return ``(mapping, status)`` for a JSON artifact.

    Status is one of ``ok``, ``missing``, or ``malformed``.  Non-object JSON is
    malformed for the structured artifacts consumed here.
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, "missing"
    except (OSError, UnicodeError):
        return {}, "malformed"
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}, "malformed"
    if not isinstance(value, dict):
        return {}, "malformed"
    return value, "ok"


def _text_file(path: Path) -> tuple[str, str]:
    """Return ``(text, status)`` for a UTF-8 text artifact."""

    try:
        return path.read_text(encoding="utf-8"), "ok"
    except FileNotFoundError:
        return "", "missing"
    except (OSError, UnicodeError):
        return "", "malformed"


def _read_mapping(path: Path) -> dict[str, Any]:
    value, _ = _json_file(path)
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    return 0


def _strings(value: Any, *, limit: int = 50) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _normalize_score_0_10(value: Any) -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    return round(max(0.0, min(10.0, number)) * 10.0, 2)


def _normalize_ratio(value: Any) -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    if number > 1.0 and number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _status_entry(path: Path, status: str) -> dict[str, str]:
    return {"path": str(path), "status": status}


def load_stage_results(run_dir: Path) -> list[StageResult]:
    """Reconstruct ``StageResult`` objects from per-stage ``decision.json``."""

    results: list[StageResult] = []
    summary = _read_mapping(run_dir / "pipeline_summary.json")
    try:
        from_stage = int(summary.get("from_stage"))
        final_stage = int(summary.get("final_stage"))
    except (TypeError, ValueError):
        from_stage = final_stage = 0
    for stage_dir in sorted(run_dir.glob("stage-[0-9][0-9]")):
        decision = _read_mapping(stage_dir / "decision.json")
        try:
            stage_num = int(str(stage_dir.name).split("-", 1)[1])
            stage = Stage(stage_num)
            status = StageStatus(str(decision.get("status", "failed")))
        except (TypeError, ValueError):
            continue
        # A resumed run reuses earlier stage directories. Those historical
        # decisions are prerequisites/artifacts, not failures executed by the
        # current pipeline invocation. Restrict evidence to the summary's
        # authoritative execution window when available.
        if from_stage and final_stage and not from_stage <= stage_num <= final_stage:
            continue
        artifacts = decision.get("output_artifacts", ())
        evidence_refs = decision.get("evidence_refs", ())
        results.append(
            StageResult(
                stage=stage,
                status=status,
                artifacts=tuple(str(item) for item in artifacts or ()),
                error=(
                    str(decision["error"])
                    if decision.get("error") is not None
                    else None
                ),
                decision=str(decision.get("decision", "proceed")),
                evidence_refs=tuple(str(item) for item in evidence_refs or ()),
            )
        )
    return results


def load_lessons(run_dir: Path, results: list[StageResult]) -> list[LessonEntry]:
    """Load runner-produced lessons, or reconstruct them if unavailable."""

    store = EvolutionStore(run_dir / "evolution")
    lessons = store.load_all()
    if lessons:
        return lessons
    run_id = str(_read_mapping(run_dir / "pipeline_summary.json").get("run_id", ""))
    return extract_lessons(results, run_id=run_id, run_dir=run_dir)


def _quality_score_from_report(report: Mapping[str, Any]) -> float | None:
    for key in ("score_1_to_10", "score", "quality_score", "overall_score"):
        if key not in report:
            continue
        score = _finite_float(report[key])
        if score is not None:
            return score
    return None


def _artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    roots = [
        run_dir / "pipeline_summary.json",
        run_dir / "checkpoint.json",
        run_dir / "topic_candidates.json",
        run_dir / "selected_topic.json",
        run_dir / "topic_selection.md",
        run_dir / "iteration_summary.json",
        run_dir / "stage-14" / "experiment_summary.json",
        run_dir / "stage-17" / "paper_draft.md",
        run_dir / "stage-17" / "draft_quality.json",
        run_dir / "stage-17" / "paper_meta.json",
        run_dir / "stage-18" / "reviews.md",
        run_dir / "stage-20" / "quality_report.json",
        run_dir / "stage-20" / "fabrication_flags.json",
        run_dir / "stage-22" / "paper_final.md",
        run_dir / "stage-23" / "verification_report.json",
        run_dir / "deliverables" / "paper_final.md",
        run_dir / "deliverables" / "paper.tex",
        run_dir / "deliverables" / "references.bib",
    ]
    inventory: list[dict[str, Any]] = []
    for path in roots:
        if not path.exists():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        inventory.append(
            {
                "path": str(path.relative_to(run_dir)),
                "size_bytes": size,
                "kind": "directory" if path.is_dir() else "file",
            }
        )
    deliverables = run_dir / "deliverables"
    if deliverables.is_dir():
        for path in sorted(deliverables.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(run_dir))
            if any(item["path"] == relative for item in inventory):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            inventory.append({"path": relative, "size_bytes": size, "kind": "file"})
    return inventory


def _pipeline_evidence(
    summary: Mapping[str, Any],
    status: str,
    *,
    pipeline_returncode: int,
    stage_results: Sequence[StageResult],
) -> dict[str, Any]:
    counts: dict[str, int | None] = {}
    for key in (
        "stages_executed",
        "stages_done",
        "stages_failed",
        "stages_paused",
        "stages_blocked",
    ):
        counts[key] = _nonnegative_int(summary.get(key))

    result_counts = {
        "done": sum(result.status is StageStatus.DONE for result in stage_results),
        "failed": sum(result.status is StageStatus.FAILED for result in stage_results),
        "paused": sum(result.status is StageStatus.PAUSED for result in stage_results),
        "blocked": sum(
            result.status is StageStatus.BLOCKED_APPROVAL for result in stage_results
        ),
        "rejected": sum(
            result.status is StageStatus.REJECTED for result in stage_results
        ),
    }
    return {
        "artifact": _status_entry(Path("pipeline_summary.json"), status),
        "run_id": str(summary.get("run_id", "") or ""),
        "final_status": str(summary.get("final_status", "") or "").lower(),
        "degraded": bool(summary.get("degraded", False)),
        "pipeline_returncode": int(pipeline_returncode),
        **counts,
        "reconstructed_stage_counts": result_counts,
    }


def _best_experiment_path(run_dir: Path) -> Path:
    promoted = run_dir / "experiment_summary_best.json"
    if promoted.exists():
        return promoted
    return run_dir / "stage-14" / "experiment_summary.json"


def _experiment_evidence(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _best_experiment_path(run_dir)
    report, status = _json_file(path)
    metrics = report.get("metrics_summary")
    conditions = report.get("condition_summaries", report.get("condition_metrics", {}))
    best_run = report.get("best_run")
    total_runs = _nonnegative_int(report.get("total_runs"))
    total_conditions = _nonnegative_int(report.get("total_conditions"))
    if total_conditions is None and isinstance(conditions, Mapping):
        total_conditions = len(conditions)
    best_status = ""
    best_metrics_count = 0
    if isinstance(best_run, Mapping):
        best_status = str(best_run.get("status", "") or "").lower()
        best_metrics_count = _count(best_run.get("metrics"))
    ablation_warnings = _strings(report.get("ablation_warnings"))
    seed_warnings = _strings(report.get("seed_insufficiency_warnings"))
    evidence = {
        "artifact": _status_entry(path.relative_to(run_dir), status),
        "total_runs": total_runs,
        "total_conditions": total_conditions,
        "metric_count": _count(metrics),
        "condition_count": _count(conditions),
        "best_run_status": best_status,
        "best_run_metric_count": best_metrics_count,
        "paired_comparison_count": _count(report.get("paired_comparisons")),
        "ablation_warning_count": len(ablation_warnings),
        "seed_warning_count": len(seed_warnings),
        "warnings": ablation_warnings + seed_warnings,
        "analysis_quality": _finite_float(
            report.get("analysis_quality", report.get("quality_score"))
        ),
    }
    return evidence, report


def _paper_evidence(run_dir: Path) -> tuple[dict[str, Any], str]:
    paper_path = run_dir / "stage-17" / "paper_draft.md"
    paper, status = _text_file(paper_path)
    meta_path = run_dir / "stage-17" / "paper_meta.json"
    meta, meta_status = _json_file(meta_path)
    quality_path = run_dir / "stage-17" / "draft_quality.json"
    draft_quality, quality_status = _json_file(quality_path)
    headings = _HEADING_RE.findall(paper)
    warnings = _strings(draft_quality.get("overall_warnings"))
    section_analysis = draft_quality.get("section_analysis")
    blocked_outcome = str(meta.get("outcome", "") or "")
    return (
        {
            "artifact": _status_entry(paper_path.relative_to(run_dir), status),
            "meta_artifact": _status_entry(meta_path.relative_to(run_dir), meta_status),
            "draft_quality_artifact": _status_entry(
                quality_path.relative_to(run_dir), quality_status
            ),
            "word_count": len(_WORD_RE.findall(paper)),
            "character_count": len(paper),
            "heading_count": len(headings),
            "section_count": (
                len(section_analysis)
                if isinstance(section_analysis, list)
                else len(headings)
            ),
            "draft_warning_count": len(warnings),
            "draft_warnings": warnings,
            "blocked_outcome": blocked_outcome,
        },
        paper,
    )


def _recommendations(review: str) -> list[str]:
    values: list[str] = []
    for match in _RECOMMENDATION_RE.finditer(review):
        value = match.group(1).strip().strip("*_ .").lower()
        if value:
            values.append(value[:80])
    return values


def _review_evidence(run_dir: Path) -> tuple[dict[str, Any], str]:
    path = run_dir / "stage-18" / "reviews.md"
    review, status = _text_file(path)
    recommendations = _recommendations(review)
    negative = sum(
        any(token in value for token in _NEGATIVE_RECOMMENDATIONS)
        for value in recommendations
    )
    positive = sum(
        any(token in value for token in _POSITIVE_RECOMMENDATIONS)
        for value in recommendations
    )
    return (
        {
            "artifact": _status_entry(path.relative_to(run_dir), status),
            "word_count": len(_WORD_RE.findall(review)),
            "reviewer_count": len(_REVIEWER_RE.findall(review)),
            "recommendations": recommendations,
            "positive_recommendation_count": positive,
            "negative_recommendation_count": negative,
            "critical_issue_count": len(_CRITICAL_RE.findall(review)),
        },
        review,
    )


def _quality_gate_evidence(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "stage-20" / "quality_report.json"
    report, status = _json_file(path)
    score = _quality_score_from_report(report)
    fabrication_path = run_dir / "stage-20" / "fabrication_flags.json"
    fabrication, fabrication_status = _json_file(fabrication_path)
    return (
        {
            "artifact": _status_entry(path.relative_to(run_dir), status),
            "fabrication_artifact": _status_entry(
                fabrication_path.relative_to(run_dir), fabrication_status
            ),
            "score_1_to_10": score,
            "verdict": str(report.get("verdict", "") or "").lower(),
            "strengths": _strings(report.get("strengths")),
            "weaknesses": _strings(report.get("weaknesses")),
            "required_actions": _strings(report.get("required_actions")),
            "fabrication_suspected": (
                bool(fabrication.get("fabrication_suspected"))
                if fabrication_status == "ok"
                else None
            ),
            "has_real_data": (
                bool(fabrication.get("has_real_data"))
                if fabrication_status == "ok"
                else None
            ),
            "verified_values_count": (
                _nonnegative_int(fabrication.get("verified_values_count"))
                if fabrication_status == "ok"
                else None
            ),
        },
        report,
    )


def _citation_evidence(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "stage-23" / "verification_report.json"
    report, status = _json_file(path)
    summary = report.get("summary")
    source: Mapping[str, Any] = summary if isinstance(summary, Mapping) else report
    total = _nonnegative_int(source.get("total", source.get("total_references")))
    verified = _nonnegative_int(source.get("verified", source.get("verified_count")))
    suspicious = _nonnegative_int(
        source.get("suspicious", source.get("suspicious_count"))
    )
    hallucinated = _nonnegative_int(
        source.get("hallucinated", source.get("hallucinated_count"))
    )
    skipped = _nonnegative_int(source.get("skipped", source.get("skipped_count")))
    integrity = _normalize_ratio(
        source.get("integrity_score", source.get("verification_rate"))
    )
    if integrity is None and total is not None and total > 0 and verified is not None:
        integrity = max(0.0, min(1.0, verified / total))
    no_references = bool(
        total == 0
        and isinstance(report.get("note"), str)
        and "no reference" in str(report["note"]).lower()
    )
    return (
        {
            "artifact": _status_entry(path.relative_to(run_dir), status),
            "total": total,
            "verified": verified,
            "suspicious": suspicious,
            "hallucinated": hallucinated,
            "skipped": skipped,
            "integrity_score": integrity,
            "no_references": no_references,
        },
        report,
    )


def _component(
    *,
    name: str,
    score: float,
    available: bool,
    reasons: Sequence[str],
) -> dict[str, Any]:
    bounded = max(0.0, min(100.0, score))
    return {
        "name": name,
        "score": round(bounded, 2),
        "available": bool(available),
        "weight": SCORE_WEIGHTS[name],
        "weighted_points": round(bounded * SCORE_WEIGHTS[name], 2),
        "reasons": [str(reason) for reason in reasons if str(reason).strip()],
    }


def _score_pipeline(data: Mapping[str, Any]) -> dict[str, Any]:
    status = data.get("artifact", {}).get("status")
    if status != "ok":
        return _component(
            name="pipeline",
            score=0,
            available=False,
            reasons=[f"pipeline_summary.json is {status}; no completion credit"],
        )
    score = 0.0
    reasons: list[str] = []
    final_status = str(data.get("final_status", ""))
    if final_status == "done":
        score += 35
        reasons.append("pipeline final_status is done (+35)")
    else:
        reasons.append(f"pipeline final_status is {final_status or 'unknown'} (+0)")
    returncode = int(data.get("pipeline_returncode", 1) or 0)
    if returncode == 0:
        score += 20
        reasons.append("pipeline return code is zero (+20)")
    else:
        reasons.append(f"pipeline return code is {returncode} (+0)")

    failed = data.get("stages_failed")
    paused = data.get("stages_paused")
    blocked = data.get("stages_blocked")
    executed = data.get("stages_executed")
    done = data.get("stages_done")
    if failed == 0 and paused == 0 and blocked == 0:
        score += 25
        reasons.append("summary reports no failed, paused, or blocked stages (+25)")
    else:
        reasons.append(
            "summary has nonzero or unknown failed/paused/blocked stages (+0)"
        )
    if (
        isinstance(executed, int)
        and executed > 0
        and isinstance(done, int)
        and 0 <= done <= executed
    ):
        completion = done / executed
        points = 20 * completion
        score += points
        reasons.append(f"{done}/{executed} reported stages done (+{points:.2f})")
    else:
        reasons.append("stage completion ratio unavailable (+0)")
    if bool(data.get("degraded")):
        score -= 10
        reasons.append("pipeline is marked degraded (-10)")
    return _component(name="pipeline", score=score, available=True, reasons=reasons)


def _score_experiment(data: Mapping[str, Any]) -> dict[str, Any]:
    status = data.get("artifact", {}).get("status")
    if status != "ok":
        return _component(
            name="experiment",
            score=0,
            available=False,
            reasons=[f"experiment_summary.json is {status}; no experiment credit"],
        )
    score = 0.0
    reasons: list[str] = []
    total_runs = data.get("total_runs")
    if isinstance(total_runs, int) and total_runs > 0:
        points = min(25.0, 5.0 + 5.0 * math.log2(total_runs + 1))
        score += points
        reasons.append(f"{total_runs} experiment runs (+{points:.2f})")
    else:
        reasons.append("no positive total_runs evidence (+0)")
    metric_count = int(data.get("metric_count", 0) or 0)
    if metric_count > 0:
        points = min(25.0, 10.0 + 5.0 * metric_count)
        score += points
        reasons.append(f"{metric_count} summarized metrics (+{points:.2f})")
    else:
        reasons.append("metrics_summary is empty (+0)")
    condition_count = int(data.get("condition_count", 0) or 0)
    if condition_count > 0:
        points = min(20.0, 5.0 + 5.0 * condition_count)
        score += points
        reasons.append(f"{condition_count} experimental conditions (+{points:.2f})")
    else:
        reasons.append("no condition-level evidence (+0)")
    best_status = str(data.get("best_run_status", ""))
    best_metric_count = int(data.get("best_run_metric_count", 0) or 0)
    if best_metric_count > 0 and best_status != "failed":
        score += 15
        reasons.append("best run contains metrics and is not failed (+15)")
    else:
        reasons.append("best run success with metrics is not established (+0)")
    paired = int(data.get("paired_comparison_count", 0) or 0)
    if paired > 0:
        points = min(15.0, 5.0 + 2.5 * paired)
        score += points
        reasons.append(f"{paired} paired comparisons (+{points:.2f})")
    warning_count = int(data.get("ablation_warning_count", 0) or 0)
    warning_count += int(data.get("seed_warning_count", 0) or 0)
    if warning_count:
        penalty = min(30.0, 7.5 * warning_count)
        score -= penalty
        reasons.append(f"{warning_count} experiment rigor warnings (-{penalty:.2f})")
    return _component(name="experiment", score=score, available=True, reasons=reasons)


def _score_paper(data: Mapping[str, Any]) -> dict[str, Any]:
    status = data.get("artifact", {}).get("status")
    if status != "ok":
        return _component(
            name="paper",
            score=0,
            available=False,
            reasons=[f"stage-17 paper is {status}; no paper credit"],
        )
    score = 0.0
    reasons: list[str] = []
    words = int(data.get("word_count", 0) or 0)
    if words > 0:
        points = min(45.0, 45.0 * words / 4000.0)
        score += points
        reasons.append(f"paper has {words} words (+{points:.2f})")
    sections = int(data.get("section_count", 0) or 0)
    if sections > 0:
        points = min(35.0, 5.0 * sections)
        score += points
        reasons.append(f"paper has {sections} detected sections (+{points:.2f})")
    quality_status = data.get("draft_quality_artifact", {}).get("status")
    if quality_status == "ok":
        score += 20
        reasons.append("draft_quality.json is valid (+20)")
        warning_count = int(data.get("draft_warning_count", 0) or 0)
        if warning_count:
            penalty = min(35.0, 5.0 * warning_count)
            score -= penalty
            reasons.append(f"{warning_count} draft quality warnings (-{penalty:.2f})")
    else:
        reasons.append(f"draft_quality.json is {quality_status} (+0)")
    blocked_outcome = str(data.get("blocked_outcome", ""))
    if blocked_outcome:
        score = min(score, 10.0)
        reasons.append(f"paper_meta outcome is {blocked_outcome}; score capped at 10")
    return _component(name="paper", score=score, available=True, reasons=reasons)


def _score_review(data: Mapping[str, Any]) -> dict[str, Any]:
    status = data.get("artifact", {}).get("status")
    if status != "ok":
        return _component(
            name="review",
            score=0,
            available=False,
            reasons=[f"stage-18 reviews.md is {status}; no review credit"],
        )
    score = 0.0
    reasons: list[str] = []
    reviewer_count = int(data.get("reviewer_count", 0) or 0)
    if reviewer_count > 0:
        points = min(40.0, reviewer_count * 15.0)
        score += points
        reasons.append(f"{reviewer_count} named reviewers (+{points:.2f})")
    words = int(data.get("word_count", 0) or 0)
    if words > 0:
        points = min(30.0, 30.0 * words / 600.0)
        score += points
        reasons.append(f"review has {words} words (+{points:.2f})")
    recommendations = data.get("recommendations", [])
    if isinstance(recommendations, list) and recommendations:
        score += 30
        reasons.append("explicit reviewer recommendations are present (+30)")
    else:
        reasons.append("no explicit recommendation fields (+0)")
    negative = int(data.get("negative_recommendation_count", 0) or 0)
    critical = int(data.get("critical_issue_count", 0) or 0)
    if negative:
        penalty = min(35.0, 12.5 * negative)
        score -= penalty
        reasons.append(f"{negative} negative recommendations (-{penalty:.2f})")
    if critical:
        penalty = min(30.0, 7.5 * critical)
        score -= penalty
        reasons.append(f"{critical} critical issue markers (-{penalty:.2f})")
    return _component(name="review", score=score, available=True, reasons=reasons)


def _score_quality_gate(data: Mapping[str, Any]) -> dict[str, Any]:
    status = data.get("artifact", {}).get("status")
    normalized = _normalize_score_0_10(data.get("score_1_to_10"))
    if status != "ok" or normalized is None:
        detail = (
            f"quality_report.json is {status}"
            if status != "ok"
            else "quality score is missing or non-numeric"
        )
        return _component(
            name="quality_gate",
            score=0,
            available=False,
            reasons=[detail + "; no quality-gate credit"],
        )
    score = normalized
    reasons = [
        (
            f"reported quality score {data.get('score_1_to_10')}/10 maps to "
            f"{normalized:.2f}/100"
        )
    ]
    verdict = str(data.get("verdict", ""))
    if verdict in {"reject", "rejected", "fail", "failed"}:
        score = min(score, 25.0)
        reasons.append(f"quality verdict is {verdict}; score capped at 25")
    elif verdict in {"revise", "major revision", "major_revision"}:
        score = min(score, 60.0)
        reasons.append(f"quality verdict is {verdict}; score capped at 60")
    if data.get("fabrication_suspected") is True:
        score = 0.0
        reasons.append("fabrication_flags.json marks fabrication suspected; score is 0")
    elif data.get("has_real_data") is False:
        score = min(score, 20.0)
        reasons.append(
            "fabrication_flags.json reports no real data; score capped at 20"
        )
    return _component(name="quality_gate", score=score, available=True, reasons=reasons)


def _score_citations(data: Mapping[str, Any]) -> dict[str, Any]:
    status = data.get("artifact", {}).get("status")
    if status != "ok":
        return _component(
            name="citations",
            score=0,
            available=False,
            reasons=[f"citation verification report is {status}; no citation credit"],
        )
    total = data.get("total")
    integrity = data.get("integrity_score")
    if (
        not isinstance(total, int)
        or total <= 0
        or not isinstance(integrity, (int, float))
    ):
        reason = (
            "citation report explicitly contains no references"
            if data.get("no_references")
            else "positive citation count and integrity score are not both available"
        )
        return _component(
            name="citations", score=0, available=True, reasons=[reason + " (+0)"]
        )
    score = 100.0 * float(integrity)
    reasons = [
        f"citation integrity is {float(integrity):.3f} across {total} references"
    ]
    hallucinated = data.get("hallucinated")
    if isinstance(hallucinated, int) and hallucinated > 0:
        penalty = min(50.0, 10.0 * hallucinated)
        score -= penalty
        reasons.append(f"{hallucinated} hallucinated citations (-{penalty:.2f})")
    suspicious = data.get("suspicious")
    if isinstance(suspicious, int) and suspicious > 0:
        penalty = min(20.0, 2.0 * suspicious)
        score -= penalty
        reasons.append(f"{suspicious} suspicious citations (-{penalty:.2f})")
    return _component(name="citations", score=score, available=True, reasons=reasons)


def compute_composite_score(
    structured: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute a deterministic, fully explained 0-100 scorecard.

    Missing components score zero and their weight remains in the denominator.
    This is deliberately conservative: a partial run cannot look stronger just
    because difficult or failed stages did not produce an artifact.
    """

    components = {
        "pipeline": _score_pipeline(structured.get("pipeline", {})),
        "experiment": _score_experiment(structured.get("experiment", {})),
        "paper": _score_paper(structured.get("paper", {})),
        "review": _score_review(structured.get("review", {})),
        "quality_gate": _score_quality_gate(structured.get("quality_gate", {})),
        "citations": _score_citations(structured.get("citations", {})),
    }
    total = round(
        sum(component["weighted_points"] for component in components.values()), 2
    )
    available_weight = round(
        sum(
            float(component["weight"])
            for component in components.values()
            if component["available"]
        ),
        4,
    )
    hard_failures: list[str] = []
    pipeline = structured.get("pipeline", {})
    if pipeline.get("artifact", {}).get("status") != "ok":
        hard_failures.append("pipeline summary missing or malformed")
    elif str(pipeline.get("final_status", "")) != "done":
        hard_failures.append("pipeline did not finish with final_status=done")
    if int(pipeline.get("pipeline_returncode", 1) or 0) != 0:
        hard_failures.append("pipeline return code was nonzero")
    if structured.get("quality_gate", {}).get("fabrication_suspected") is True:
        hard_failures.append("fabrication was suspected")
    citations = structured.get("citations", {})
    if isinstance(citations.get("hallucinated"), int) and citations["hallucinated"] > 0:
        hard_failures.append("citation verification found hallucinated references")
    return {
        "version": SCORE_VERSION,
        "score": total,
        "maximum": 100.0,
        "weights": dict(SCORE_WEIGHTS),
        "components": components,
        "available_weight": available_weight,
        "missing_weight": round(1.0 - available_weight, 4),
        "hard_failures": hard_failures,
        "policy": (
            "Fixed-weight conservative score: missing or malformed evidence earns "
            "zero; component weights are never renormalized."
        ),
    }


def _score_from_evidence(evidence: Mapping[str, Any] | None) -> float | None:
    if not isinstance(evidence, Mapping):
        return None
    scorecard = evidence.get("scorecard")
    if isinstance(scorecard, Mapping):
        score = _finite_float(scorecard.get("score"))
        if score is not None:
            return score
    return _finite_float(evidence.get("composite_score"))


def _hard_failures_from_evidence(
    evidence: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(evidence, Mapping):
        return []
    scorecard = evidence.get("scorecard")
    if not isinstance(scorecard, Mapping):
        return []
    failures = scorecard.get("hard_failures")
    if not isinstance(failures, Sequence) or isinstance(
        failures, (str, bytes, bytearray)
    ):
        return []
    return [str(item) for item in failures if str(item).strip()]


def compare_evidence(
    candidate: Mapping[str, Any],
    best: Mapping[str, Any] | None,
    *,
    min_improvement: float = DEFAULT_ACCEPT_MARGIN,
) -> dict[str, Any]:
    """Compare a cycle candidate to the current best and explain the decision."""

    margin = _finite_float(min_improvement)
    if margin is None or margin < 0:
        raise ValueError("min_improvement must be a finite non-negative number")
    candidate_score = _score_from_evidence(candidate)
    best_score = _score_from_evidence(best)
    hard_failures = _hard_failures_from_evidence(candidate)
    best_hard_failures = _hard_failures_from_evidence(best)

    accepted = False
    reason = ""
    if candidate_score is None:
        reason = "rejected: candidate has no valid composite score"
    elif hard_failures:
        reason = "rejected: " + "; ".join(str(item) for item in hard_failures)
    elif best_score is None or best_hard_failures:
        accepted = True
        if best_hard_failures:
            reason = (
                f"accepted: incumbent has hard failures; candidate establishes "
                f"safe baseline at {candidate_score:.2f}"
            )
        else:
            reason = (
                f"accepted: no valid incumbent score; candidate establishes "
                f"baseline at {candidate_score:.2f}"
            )
    else:
        delta = candidate_score - best_score
        if delta >= margin:
            accepted = True
            reason = (
                f"accepted: candidate {candidate_score:.2f} exceeds best "
                f"{best_score:.2f} by {delta:.2f}, meeting margin {margin:.2f}"
            )
        else:
            reason = (
                f"rejected: candidate {candidate_score:.2f} improves on best "
                f"{best_score:.2f} by {delta:.2f}, below required margin "
                f"{margin:.2f}"
            )
    delta = (
        round(candidate_score - best_score, 2)
        if candidate_score is not None and best_score is not None
        else None
    )
    return {
        "decision": "accept" if accepted else "reject",
        "accepted": accepted,
        "candidate_score": candidate_score,
        "best_score": best_score,
        "delta": delta,
        "min_improvement": margin,
        "reason": reason,
    }


def export_evidence_json(path: Path, evidence: Mapping[str, Any]) -> Path:
    """Atomically export an evidence bundle as JSON."""

    atomic_write_json(path, dict(evidence))
    return path


def collect_evidence(
    run_dir: Path,
    *,
    pipeline_returncode: int,
    command: Iterable[str],
    best_evidence: Mapping[str, Any] | None = None,
    topic_id: str = "",
    min_improvement: float = DEFAULT_ACCEPT_MARGIN,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Collect, score, compare, and atomically export one pipeline run."""

    run_dir = Path(run_dir)
    summary, summary_status = _json_file(run_dir / "pipeline_summary.json")
    results = load_stage_results(run_dir)
    lessons = load_lessons(run_dir, results)
    experiment_evidence, experiment = _experiment_evidence(run_dir)
    selected_topic, selected_topic_status = _json_file(
        run_dir / "selected_topic.json"
    )
    paper_evidence, paper_text = _paper_evidence(run_dir)
    review_evidence, review_text = _review_evidence(run_dir)
    quality_evidence, quality = _quality_gate_evidence(run_dir)
    citation_evidence, verification = _citation_evidence(run_dir)
    failures = [
        {
            "stage": int(result.stage),
            "stage_name": result.stage.name,
            "status": result.status.value,
            "error": result.error,
            "decision": result.decision,
        }
        for result in results
        if result.status in _BAD_STAGE_STATUSES
    ]
    structured = {
        "pipeline": _pipeline_evidence(
            summary,
            summary_status,
            pipeline_returncode=pipeline_returncode,
            stage_results=results,
        ),
        "experiment": experiment_evidence,
        "paper": paper_evidence,
        "review": review_evidence,
        "quality_gate": quality_evidence,
        "citations": citation_evidence,
    }
    scorecard = compute_composite_score(structured)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "topic_id": str(topic_id or ""),
        "selected_topic": selected_topic,
        "selected_topic_artifact": _status_entry(
            Path("selected_topic.json"),
            selected_topic_status,
        ),
        "pipeline_returncode": int(pipeline_returncode),
        "command": [str(item) for item in command],
        "pipeline_summary": summary,
        "quality_report": quality,
        "quality_score": _quality_score_from_report(quality),
        "verification_report": verification,
        "experiment_summary": experiment,
        "paper": {
            "path": "stage-17/paper_draft.md",
            "text": paper_text,
            **paper_evidence,
        },
        "review": {
            "path": "stage-18/reviews.md",
            "text": review_text,
            **review_evidence,
        },
        "structured_evidence": structured,
        "scorecard": scorecard,
        "composite_score": scorecard["score"],
        "stage_results": [
            {
                "stage": int(result.stage),
                "stage_name": result.stage.name,
                "status": result.status.value,
                "error": result.error,
                "decision": result.decision,
                "artifacts": list(result.artifacts),
            }
            for result in results
        ],
        "failures": failures,
        "lessons": [lesson.to_dict() for lesson in lessons],
        "artifacts": _artifact_inventory(run_dir),
    }
    evidence["comparison"] = compare_evidence(
        evidence,
        best_evidence,
        min_improvement=min_improvement,
    )
    export_evidence_json(output_path or run_dir / "rsi_evidence.json", evidence)
    return evidence


def evidence_succeeded(evidence: Mapping[str, Any]) -> bool:
    """Return true only for a non-failing, non-paused pipeline outcome."""

    summary = evidence.get("pipeline_summary")
    if not isinstance(summary, Mapping):
        return False
    try:
        returncode = int(evidence.get("pipeline_returncode", 1) or 0)
        failed = int(summary.get("stages_failed", 0) or 0)
        paused = int(summary.get("stages_paused", 0) or 0)
        blocked = int(summary.get("stages_blocked", 0) or 0)
    except (TypeError, ValueError):
        return False
    if returncode != 0 or failed > 0 or paused > 0 or blocked > 0:
        return False
    return str(summary.get("final_status", "")).lower() == "done"
