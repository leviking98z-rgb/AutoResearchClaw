from __future__ import annotations

from researchclaw.autoresearch_v2.ideas import (
    IdeaAdmission,
    candidate_to_idea,
    evidence_adjusted_score,
    title_similarity,
    validate_candidate,
)


def _candidate(
    *,
    identifier: str = "calibration-gate",
    title: str = "Calibration-aware stopping for recursive self-improvement",
    family: str = "calibration",
) -> dict[str, object]:
    return {
        "id": identifier,
        "title": title,
        "family": family,
        "research_question": "Can calibration drift predict regression?",
        "falsifiable_hypothesis": (
            "Calibration-aware stopping improves held-out accuracy per token."
        ),
        "closest_prior_work": ["self-refinement", "selective prediction"],
        "novelty_gap": "Prospective collapse prediction is not tested.",
        "datasets": ["GSM8K", "MATH"],
        "models": ["Qwen2.5-7B-Instruct"],
        "compute": {"gpu_count": 1, "wall_clock_hours": 2},
        "primary_metric": "held-out accuracy per token",
        "baselines": ["single-pass no-self-improvement control"],
        "ablations": ["remove calibration features"],
        "failure_safety_tests": ["benchmark leakage", "regression rate"],
        "implementation_feasibility": "Open model and public data.",
        "licensing_feasibility": "Verify exact model and dataset licenses.",
        "information_gain_if_true": "Identifies a useful stopping signal.",
        "information_gain_if_false": "Rules out calibration drift.",
        "cheap_pilot": "One GPU, 50 examples, stop on futility.",
        "scores": {
            "novelty": 8.5,
            "scientific_importance": 9.0,
            "falsifiability": 9.0,
            "compute_tractability": 9.0,
            "reproducibility": 8.5,
            "meaningful_result_likelihood": 8.5,
            "risk": 2.0,
        },
    }


def test_high_quality_candidate_validates_and_scores() -> None:
    candidate = _candidate()
    assert validate_candidate(candidate) == []
    idea = candidate_to_idea(candidate)
    assert idea.score > 8
    assert idea.priority > 0.8
    assert idea.family == "calibration"


def test_candidate_without_control_is_rejected() -> None:
    candidate = _candidate()
    candidate["baselines"] = ["fixed-depth refinement"]
    assert "missing no-self-improvement control" in validate_candidate(candidate)


def test_admission_rejects_semantic_title_variant() -> None:
    original = candidate_to_idea(_candidate())
    variant = candidate_to_idea(
        _candidate(
            identifier="variant",
            title="Calibration aware stopping in recursive self improvement",
            family="calibration",
        )
    )
    decision = IdeaAdmission(duplicate_threshold=0.60).decide(
        variant,
        existing=[original],
    )
    assert not decision.admitted
    assert decision.duplicate_of == original.idea_id
    assert title_similarity(original.title, variant.title) >= 0.60


def test_admission_preserves_family_diversity() -> None:
    first = candidate_to_idea(_candidate(identifier="one"))
    second = candidate_to_idea(
        _candidate(
            identifier="two",
            title="Brier trajectory gates for model self-refinement",
        )
    )
    third = candidate_to_idea(
        _candidate(
            identifier="three",
            title="Conformal uncertainty gates for self-revision loops",
        )
    )
    decision = IdeaAdmission(max_same_family=2).decide(
        third,
        existing=[first, second],
    )
    assert not decision.admitted
    assert decision.reason == "family_quota"


def test_admission_can_require_grounded_novelty_evidence() -> None:
    idea = candidate_to_idea(_candidate())
    admission = IdeaAdmission(require_novelty_evidence=True)
    missing = admission.decide(idea, existing=[])
    assert not missing.admitted
    assert missing.reason == "novelty_evidence_unavailable"

    idea.candidate["novelty_evidence"] = {
        "available": True,
        "closest_papers": [{"title": "Grounded prior"}],
        "max_similarity": 0.3,
    }
    assert admission.decide(idea, existing=[]).admitted


def test_missing_novelty_evidence_reduces_self_reported_score() -> None:
    idea = candidate_to_idea(_candidate())
    original = idea.score
    idea.candidate["novelty_evidence"] = {
        "available": True,
        "closest_papers": [],
        "max_similarity": 0.0,
    }
    assert evidence_adjusted_score(idea) < original
