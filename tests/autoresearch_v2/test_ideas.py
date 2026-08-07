from __future__ import annotations

import json

from researchclaw.autoresearch_v2.ideas import (
    IdeaAdmission,
    LLMBoardIdeaGenerator,
    candidate_to_idea,
    designability_evidence,
    evidence_adjusted_score,
    research_mode_target_counts,
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
    assert idea.candidate["designability"]["designable"] is True
    assert idea.candidate["designability"]["penalty"] == 0
    assert {
        "measurable_unit",
        "public_dataset",
        "open_or_accessible_model",
        "independent_baseline",
        "cheap_compute_budget",
    } <= set(idea.candidate["designability"]["evidence"])


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


def test_board_keeps_valid_candidates_when_one_candidate_is_invalid() -> None:
    valid = _candidate(identifier="valid")
    invalid = _candidate(identifier="invalid")
    invalid["baselines"] = ["fixed-depth refinement"]

    class _Response:
        content = json.dumps({"candidates": [invalid, valid]})

    class _Client:
        def chat(self, *args, **kwargs):
            del args, kwargs
            return _Response()

    generator = LLMBoardIdeaGenerator(
        llm=_Client(),
        brief="RSI mechanisms",
    )

    ideas = generator.generate(count=1, existing=[])

    assert [idea.candidate["id"] for idea in ideas] == ["valid"]
    assert len(generator.last_rejections) == 1
    assert (
        "missing no-self-improvement control"
        in generator.last_rejections[0]["reason"]
    )


def test_board_prompt_binds_exact_executable_resources() -> None:
    prompts: list[str] = []
    candidate = _candidate()
    candidate["datasets"] = ["openai/gsm8k"]
    candidate["models"] = ["Qwen/Qwen2.5-1.5B-Instruct"]

    class _Response:
        content = json.dumps({"candidates": [candidate]})

    class _Client:
        def chat(self, messages, **kwargs):
            del kwargs
            prompts.append(messages[0]["content"])
            return _Response()

    generator = LLMBoardIdeaGenerator(
        llm=_Client(),
        brief="RSI mechanisms",
        available_models=("Qwen/Qwen2.5-1.5B-Instruct",),
        available_datasets=("openai/gsm8k",),
    )

    assert generator.generate(count=1, existing=[])
    assert "Qwen/Qwen2.5-1.5B-Instruct" in prompts[0]
    assert "openai/gsm8k" in prompts[0]
    assert "Never\npropose a nearby size" in prompts[0]
    assert "40% capability_improvement" in prompts[0]
    assert "mechanism-activation check" in prompts[0]


def test_admission_can_require_mechanism_activation_plan() -> None:
    idea = candidate_to_idea(_candidate())
    admission = IdeaAdmission(
        minimum_score=0,
        require_mechanism_activation_plan=True,
    )

    missing = admission.decide(idea, existing=[])
    assert not missing.admitted
    assert missing.reason == "mechanism_activation_plan_missing"

    idea.candidate["mechanism_activation"] = {
        "trigger": "the verifier rejects a proposed update",
        "measurement": "verifier_rejection_rate",
        "minimum_rate": 0.25,
        "behavioral_contrast": "treatment commits fewer rejected updates",
        "early_exit": "the gate is inactive under this workload",
    }
    assert admission.decide(idea, existing=[]).admitted


def test_research_mode_targets_are_balanced_and_exact() -> None:
    assert research_mode_target_counts(20) == {
        "capability_improvement": 8,
        "population_search": 5,
        "verifier_memory": 4,
        "stopping_rollback": 3,
    }
    assert sum(research_mode_target_counts(23).values()) == 23


def test_designability_penalizes_reviewable_budget_and_access_uncertainty() -> None:
    candidate = _candidate()
    candidate["datasets"] = ["Acme Reasoning Suite"]
    candidate["models"] = ["Acme Research Model"]
    candidate["compute"] = {"gpu_count": 2, "wall_clock_hours": 3}
    candidate["cheap_pilot"] = (
        "Run 3 seeds over 100 examples and report exact-match accuracy."
    )

    evidence = designability_evidence(candidate)
    idea = candidate_to_idea(candidate)

    assert evidence["designable"] is True
    assert evidence["blockers"] == []
    assert {
        "dataset_access_unverified",
        "model_access_unverified",
        "pilot_budget_above_cheap_target",
    } <= set(evidence["concerns"])
    assert evidence["penalty"] > 0
    assert idea.score < candidate["scores"]["novelty"]
    assert IdeaAdmission(minimum_score=0).decide(idea, existing=[]).admitted


def test_admission_rejects_pilot_without_measurable_resources_or_control() -> None:
    candidate = _candidate()
    candidate["datasets"] = ["data"]
    candidate["models"] = ["model"]
    candidate["primary_metric"] = ""
    candidate["baselines"] = ["our method", "remove one module ablation"]
    idea = candidate_to_idea(candidate)

    decision = IdeaAdmission(minimum_score=0).decide(idea, existing=[])

    assert not decision.admitted
    assert decision.reason.startswith("malformed:missing primary_metric")
    blockers = set(idea.candidate["designability"]["blockers"])
    assert {
        "missing_measurable_unit",
        "dataset_not_identified",
        "model_not_identified",
        "independent_baseline_missing",
    } <= blockers


def test_admission_rejects_unbounded_human_dependent_complex_pilot() -> None:
    candidate = _candidate()
    candidate["cheap_pilot"] = (
        "Recruit participants for a user study and have expert reviewers judge "
        "whether each answer is useful."
    )
    candidate["primary_metric"] = "expert reviewer preference"
    candidate["compute"] = {"gpu_count": 8, "wall_clock_hours": 8}
    idea = candidate_to_idea(candidate)

    decision = IdeaAdmission(minimum_score=0).decide(idea, existing=[])

    assert not decision.admitted
    blockers = set(idea.candidate["designability"]["blockers"])
    assert "external_human_dependency" in blockers
    assert "unbounded_complex_pilot" in blockers
    assert "hidden_human_or_review_dependency" in set(
        idea.candidate["designability"]["concerns"]
    )


def test_released_human_labels_do_not_trigger_brittle_human_rejection() -> None:
    candidate = _candidate()
    candidate["datasets"] = [
        "Public Helpfulness Benchmark with released human labels"
    ]
    candidate["cheap_pilot"] = (
        "Automatically score 100 examples against the released annotations."
    )
    candidate["primary_metric"] = "accuracy against existing labels"
    evidence = designability_evidence(candidate)
    idea = candidate_to_idea(candidate)

    assert evidence["designable"] is True
    assert "external_human_dependency" not in evidence["blockers"]
    assert "human_labels_already_available" in evidence["evidence"]
    assert IdeaAdmission(minimum_score=0).decide(idea, existing=[]).admitted
