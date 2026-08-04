from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.rsi.topic_selection import (
    MIN_TOPIC_CANDIDATES,
    normalize_topic_action,
    persist_topic_selection,
    select_topics,
    selection_with_candidate,
    validate_selection_document,
)


def _candidate(index: int, *, score: float | None = None) -> dict[str, object]:
    value = score if score is not None else 5.0 + index / 10
    return {
        "id": f"candidate-{index:02d}",
        "title": f"RSI candidate {index}",
        "research_question": f"Does RSI mechanism {index} improve held-out quality?",
        "falsifiable_hypothesis": (
            f"Mechanism {index} improves held-out accuracy over a fixed policy."
        ),
        "closest_prior_work": ["self-refinement", "verifier-guided agents"],
        "novelty_gap": "Prior work does not isolate this mechanism across cycles.",
        "datasets": ["GSM8K", "MBPP"],
        "models": ["Qwen open-weight model"],
        "compute": {
            "gpu_count": min(index, 32),
            "wall_clock_hours": 2 + index,
            "notes": "cheap pilot before scaling",
        },
        "primary_metric": "held-out accuracy improvement per 1M tokens",
        "baselines": ["no-self-improvement control", "fixed-iteration refinement"],
        "ablations": ["remove acceptance gate"],
        "failure_safety_tests": ["reward hacking", "regression", "data leakage"],
        "implementation_feasibility": "Uses public tasks and standard inference.",
        "licensing_feasibility": "Public benchmark and permissive model license.",
        "information_gain_if_true": "Identifies a causal RSI mechanism.",
        "information_gain_if_false": "Rules out a widely assumed mechanism.",
        "cheap_pilot": "Run 100 examples for three iterations on one GPU.",
        "scores": {
            "novelty": value,
            "scientific_importance": value,
            "falsifiability": value,
            "compute_tractability": value,
            "reproducibility": value,
            "meaningful_result_likelihood": value,
            "risk": 2.0,
        },
    }


def _document(count: int = MIN_TOPIC_CANDIDATES) -> dict[str, object]:
    return {
        "candidates": [_candidate(index) for index in range(1, count + 1)],
        "selected_candidate_id": f"candidate-{count:02d}",
        "selection_rationale": "Highest validated score and a cheap pilot.",
        "pivot_policy": "Pivot if held-out signal is absent after the pilot.",
    }


def test_validate_selection_requires_at_least_twelve_candidates() -> None:
    with pytest.raises(ValueError, match="at least 12"):
        validate_selection_document(_document(11))


def test_validate_selection_ranks_and_preserves_negative_result_value() -> None:
    selection = validate_selection_document(_document())

    assert len(selection.document["candidates"]) == 12
    assert selection.selected["id"] == "candidate-12"
    assert selection.selected["rank"] == 1
    assert selection.selected["information_gain_if_false"]
    assert selection.document["automatic_submission_enabled"] is False


def test_validate_selection_rejects_missing_no_rsi_control() -> None:
    document = _document()
    document["candidates"][0]["baselines"] = ["fixed-iteration refinement"]  # type: ignore[index]

    with pytest.raises(ValueError, match="incomplete or invalid"):
        validate_selection_document(document)


def test_select_topics_retries_invalid_json_then_persists(
    tmp_path: Path,
) -> None:
    responses = iter(["not-json", json.dumps(_document())])

    class FakeLLM:
        def chat(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(content=next(responses))

    selection = select_topics(
        llm=FakeLLM(),
        brief="Autonomously select an LLM RSI research topic.",
        cycle=1,
    )
    shared = tmp_path / "shared"
    run = tmp_path / "run"
    persist_topic_selection(
        shared_dir=shared,
        run_dir=run,
        selection=selection,
        cycle=1,
    )

    for root in (shared, run):
        assert (root / "topic_candidates.json").is_file()
        assert (root / "selected_topic.json").is_file()
        assert (root / "topic_selection.md").is_file()
        selected = json.loads(
            (root / "selected_topic.json").read_text(encoding="utf-8")
        )
        assert selected["title"] == "RSI candidate 12"
        assert selected["automatic_submission_enabled"] is False
    assert (shared / "topic_selections" / "selection-0001.json").is_file()
    assert (shared / "topic_selections" / "selection-0001.md").is_file()


def test_topic_action_is_fail_closed_and_requires_evidence() -> None:
    assert normalize_topic_action(None)["topic_action"] == "keep"
    assert normalize_topic_action({"topic_action": "unknown"})["topic_action"] == (
        "keep"
    )
    assert normalize_topic_action({"topic_action": "pivot"})["topic_action"] == (
        "keep"
    )
    assert normalize_topic_action({"topic_action": "refine"})["topic_action"] == (
        "keep"
    )
    pivot = normalize_topic_action(
        {
            "topic_action": "pivot",
            "pivot_reason": "The preregistered pilot signal stayed at chance.",
            "preferred_candidate_id": "candidate-03",
        }
    )
    assert pivot["topic_action"] == "pivot"
    assert pivot["preferred_candidate_id"] == "candidate-03"


def test_selection_with_candidate_reuses_validated_matrix() -> None:
    selection = validate_selection_document(_document())
    pivoted = selection_with_candidate(
        selection,
        candidate_id="candidate-03",
        rationale="Pilot invalidated the incumbent mechanism.",
    )

    assert pivoted.selected["id"] == "candidate-03"
    assert pivoted.document["selected_candidate_id"] == "candidate-03"
    assert "invalidated" in pivoted.document["selection_rationale"]
