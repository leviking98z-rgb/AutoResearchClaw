"""Structured autonomous topic selection for RSI research campaigns."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import atomic_write_json, atomic_write_text, utc_now

MIN_TOPIC_CANDIDATES = 12
TOPIC_SELECTION_SCHEMA_VERSION = 1
TOPIC_ACTIONS = frozenset({"keep", "refine", "pivot"})

_SCORE_FIELDS = (
    "novelty",
    "scientific_importance",
    "falsifiability",
    "compute_tractability",
    "reproducibility",
    "meaningful_result_likelihood",
)
_SCORE_WEIGHTS = {
    "novelty": 0.20,
    "scientific_importance": 0.20,
    "falsifiability": 0.15,
    "compute_tractability": 0.15,
    "reproducibility": 0.15,
    "meaningful_result_likelihood": 0.15,
}
_RISK_FIELD = "risk"

_SELECTION_SYSTEM = """\
You are the autonomous topic-selection board for an LLM recursive
self-improvement (RSI) research campaign. Generate and compare concrete,
falsifiable paper projects. Be skeptical about novelty, benchmark leakage,
reward hacking, confounding, licensing, and compute feasibility.

Return ONLY one JSON object matching the requested schema. Never fabricate
citations; record closest prior-work AREAS or search targets until primary
sources are verified. Automatic submission or public release is prohibited.
"""


@dataclass(frozen=True)
class TopicSelection:
    """Validated selection package persisted for one campaign cycle."""

    document: dict[str, Any]
    selected: dict[str, Any]

    @property
    def effective_topic(self) -> str:
        return str(self.selected["title"]).strip()


def _finite_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 10.0:
        return None
    return round(number, 3)


def _string_list(value: Any, *, minimum: int = 1) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        return []
    items = [str(item).strip() for item in value if str(item).strip()]
    return items if len(items) >= minimum else []


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _weighted_score(scores: Mapping[str, float]) -> float:
    positive = sum(scores[name] * _SCORE_WEIGHTS[name] for name in _SCORE_FIELDS)
    risk_penalty = scores[_RISK_FIELD] * 0.10
    return round(max(0.0, min(10.0, positive - risk_penalty)), 3)


def _validate_candidate(raw: Any, index: int) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None

    required_text = (
        "title",
        "research_question",
        "falsifiable_hypothesis",
        "novelty_gap",
        "primary_metric",
        "implementation_feasibility",
        "licensing_feasibility",
        "information_gain_if_true",
        "information_gain_if_false",
        "cheap_pilot",
    )
    text: dict[str, str] = {}
    for field in required_text:
        value = str(raw.get(field, "") or "").strip()
        if not value:
            return None
        text[field] = value

    required_lists = (
        "closest_prior_work",
        "datasets",
        "models",
        "baselines",
        "ablations",
        "failure_safety_tests",
    )
    lists: dict[str, list[str]] = {}
    for field in required_lists:
        value = _string_list(raw.get(field))
        if not value:
            return None
        lists[field] = value
    baseline_text = " ".join(lists["baselines"]).casefold()
    if not any(
        marker in baseline_text
        for marker in (
            "no-self-improvement",
            "no self-improvement",
            "no-rsi",
            "no rsi",
            "single-pass",
            "single pass",
            "fixed policy",
        )
    ):
        return None

    compute = raw.get("compute")
    if not isinstance(compute, Mapping):
        return None
    gpu_count = compute.get("gpu_count")
    wall_clock_hours = compute.get("wall_clock_hours")
    try:
        gpu_count_int = int(gpu_count)
        wall_clock_float = float(wall_clock_hours)
    except (TypeError, ValueError):
        return None
    if not 0 <= gpu_count_int <= 32 or not math.isfinite(wall_clock_float):
        return None
    if wall_clock_float <= 0:
        return None

    raw_scores = raw.get("scores")
    if not isinstance(raw_scores, Mapping):
        return None
    scores: dict[str, float] = {}
    for field in (*_SCORE_FIELDS, _RISK_FIELD):
        score = _finite_score(raw_scores.get(field))
        if score is None:
            return None
        scores[field] = score

    candidate_id = str(raw.get("id", "") or "").strip() or f"topic-{index:02d}"
    candidate = {
        "id": candidate_id,
        **text,
        **lists,
        "compute": {
            "gpu_count": gpu_count_int,
            "wall_clock_hours": round(wall_clock_float, 3),
            "notes": str(compute.get("notes", "") or "").strip(),
        },
        "scores": scores,
        "weighted_score": _weighted_score(scores),
    }
    return candidate


def validate_selection_document(
    raw: Mapping[str, Any],
    *,
    minimum_candidates: int = MIN_TOPIC_CANDIDATES,
) -> TopicSelection:
    """Validate, normalize, score, and rank an LLM selection response."""

    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, Sequence) or isinstance(
        candidates_raw, (str, bytes, bytearray)
    ):
        raise TypeError("topic selection must contain a candidates array")

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    for index, item in enumerate(candidates_raw, start=1):
        candidate = _validate_candidate(item, index)
        if candidate is None:
            raise ValueError(f"topic candidate {index} is incomplete or invalid")
        normalized_title = candidate["title"].casefold()
        if candidate["id"] in seen_ids or normalized_title in seen_titles:
            raise ValueError("topic candidate ids and titles must be unique")
        seen_ids.add(candidate["id"])
        seen_titles.add(normalized_title)
        candidates.append(candidate)

    if len(candidates) < minimum_candidates:
        raise ValueError(
            f"topic selection requires at least {minimum_candidates} candidates"
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item["weighted_score"]),
            float(item["scores"]["risk"]),
            str(item["id"]),
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank

    requested_id = str(raw.get("selected_candidate_id", "") or "").strip()
    selected = next(
        (candidate for candidate in ranked if candidate["id"] == requested_id),
        ranked[0],
    )
    # The selector may choose a lower-ranked topic for a defensible hard
    # constraint, but it must state why. Otherwise use the deterministic top.
    rationale = str(raw.get("selection_rationale", "") or "").strip()
    if selected["rank"] != 1 and not rationale:
        selected = ranked[0]

    document = {
        "schema_version": TOPIC_SELECTION_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "selection_method": (
            "LLM proposal with deterministic validation and weighted reranking"
        ),
        "score_scale": "0-10; risk is a penalty",
        "score_weights": {
            **_SCORE_WEIGHTS,
            "risk_penalty": 0.10,
        },
        "minimum_candidates": minimum_candidates,
        "candidates": ranked,
        "selected_candidate_id": selected["id"],
        "selection_rationale": rationale
        or (
            "Selected the highest validated weighted score after penalizing "
            "scientific, safety, implementation, and compute risk."
        ),
        "pivot_policy": str(raw.get("pivot_policy", "") or "").strip()
        or (
            "Run the selected cheap pilot first. Pivot to the next ranked "
            "candidate if novelty, feasibility, or the primary signal fails."
        ),
        "automatic_submission_enabled": False,
    }
    return TopicSelection(document=document, selected=selected)


def _selection_prompt(
    *,
    brief: str,
    previous_selection: Mapping[str, Any] | None,
    cycle: int,
) -> str:
    previous = (
        json.dumps(previous_selection, ensure_ascii=False, indent=2)[:10000]
        if previous_selection
        else "None; this is the initial autonomous selection."
    )
    return f"""\
Select a concrete research project for campaign cycle {cycle}.

Campaign meta-brief:
{brief[:16000]}

Previous accepted selection, if any:
{previous}

Generate at least {MIN_TOPIC_CANDIDATES} DISTINCT candidate projects about
recursive self-improvement / self-iterative optimization of LLMs or LLM
agents. Do not merely rename the same verifier-gating idea.

Required JSON schema:
{{
  "candidates": [
    {{
      "id": "short-stable-id",
      "title": "concrete paper title",
      "research_question": "one precise question",
      "falsifiable_hypothesis": "one primary testable hypothesis",
      "closest_prior_work": ["area or verified paper target"],
      "novelty_gap": "precise missing comparison or mechanism",
      "datasets": ["public dataset or benchmark"],
      "models": ["legal/open model or API-backed test subject"],
      "compute": {{
        "gpu_count": 1,
        "wall_clock_hours": 8,
        "notes": "pilot and scale plan"
      }},
      "primary_metric": "predeclared primary endpoint",
      "baselines": ["include a no-self-improvement control"],
      "ablations": ["mechanism-isolating ablation"],
      "failure_safety_tests": ["collapse, regression, hacking, leakage test"],
      "implementation_feasibility": "dependencies and engineering assessment",
      "licensing_feasibility": "data/model/license assessment",
      "information_gain_if_true": "what is learned",
      "information_gain_if_false": "what is learned",
      "cheap_pilot": "small discriminating pilot before scale",
      "scores": {{
        "novelty": 0,
        "scientific_importance": 0,
        "falsifiability": 0,
        "compute_tractability": 0,
        "reproducibility": 0,
        "meaningful_result_likelihood": 0,
        "risk": 0
      }}
    }}
  ],
  "selected_candidate_id": "id",
  "selection_rationale": "compare the leading candidates and explain choice",
  "pivot_policy": "observable pilot outcomes that trigger the next candidate"
}}

All scores must be numeric from 0 to 10. Compute must fit at most 32 H20 GPUs.
Preserve negative-result value. Do not include publication or submission steps.
"""


def select_topics(
    *,
    llm: Any,
    brief: str,
    cycle: int,
    previous_selection: Mapping[str, Any] | None = None,
    attempts: int = 2,
) -> TopicSelection:
    """Ask the configured model to generate and select a validated topic matrix."""

    error = ""
    for attempt in range(1, max(1, attempts) + 1):
        prompt = _selection_prompt(
            brief=brief,
            previous_selection=previous_selection,
            cycle=cycle,
        )
        if error:
            prompt += (
                "\n\nYour previous response failed validation. Correct every "
                f"schema issue. Validation error: {error}"
            )
        response = llm.chat(
            [{"role": "user", "content": prompt}],
            system=_SELECTION_SYSTEM,
            json_mode=True,
            max_tokens=16000,
            temperature=0.35 if attempt == 1 else 0.15,
        )
        raw = _json_object(response.content)
        try:
            return validate_selection_document(raw)
        except (TypeError, ValueError) as exc:
            error = str(exc)
    raise ValueError(f"autonomous topic selection failed validation: {error}")


def render_selection_markdown(selection: TopicSelection) -> str:
    """Render a concise, human-reviewable selection report."""

    selected = selection.selected
    rows = [
        "# Autonomous RSI Topic Selection",
        "",
        f"**Selected:** {selected['title']}",
        "",
        f"**Primary hypothesis:** {selected['falsifiable_hypothesis']}",
        "",
        f"**Primary metric:** {selected['primary_metric']}",
        "",
        f"**Selection rationale:** {selection.document['selection_rationale']}",
        "",
        "## Ranked candidate matrix",
        "",
        "| Rank | Candidate | Score | Risk | Pilot GPUs | Pilot hours |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for candidate in selection.document["candidates"]:
        rows.append(
            "| {rank} | {title} | {score:.3f} | {risk:.1f} | {gpus} | "
            "{hours:g} |".format(
                rank=candidate["rank"],
                title=str(candidate["title"]).replace("|", "\\|"),
                score=candidate["weighted_score"],
                risk=candidate["scores"]["risk"],
                gpus=candidate["compute"]["gpu_count"],
                hours=candidate["compute"]["wall_clock_hours"],
            )
        )
    rows.extend(
        [
            "",
            "## Selected pilot",
            "",
            selected["cheap_pilot"],
            "",
            "## Pivot policy",
            "",
            selection.document["pivot_policy"],
            "",
            "## Safety boundary",
            "",
            "Local artifacts only. Automatic submission and public release remain disabled.",
            "",
        ]
    )
    return "\n".join(rows)


def persist_topic_selection(
    *,
    shared_dir: Path,
    run_dir: Path,
    selection: TopicSelection,
    cycle: int | None = None,
) -> None:
    """Persist campaign-shared and cycle-local autonomous selection artifacts."""

    selected = dict(selection.selected)
    selected.update(
        {
            "schema_version": TOPIC_SELECTION_SCHEMA_VERSION,
            "selected_at": selection.document["generated_at"],
            "selection_rationale": selection.document["selection_rationale"],
            "pivot_policy": selection.document["pivot_policy"],
            "automatic_submission_enabled": False,
        }
    )
    markdown = render_selection_markdown(selection)
    for root in (shared_dir, run_dir):
        atomic_write_json(root / "topic_candidates.json", selection.document)
        atomic_write_json(root / "selected_topic.json", selected)
        atomic_write_text(root / "topic_selection.md", markdown)
    if cycle is not None:
        history = shared_dir / "topic_selections"
        atomic_write_json(
            history / f"selection-{cycle:04d}.json",
            selection.document,
        )
        atomic_write_text(
            history / f"selection-{cycle:04d}.md",
            markdown,
        )


def topic_selection_from_records(
    *,
    candidates: Sequence[Mapping[str, Any]],
    selected_candidate_id: str,
    selection_rationale: str,
    pivot_policy: str,
) -> TopicSelection:
    """Build a validated selection from deterministic local candidate records."""

    return validate_selection_document(
        {
            "candidates": [dict(candidate) for candidate in candidates],
            "selected_candidate_id": selected_candidate_id,
            "selection_rationale": selection_rationale,
            "pivot_policy": pivot_policy,
        }
    )


def selection_from_artifacts(
    *,
    candidates_path: Path,
    selected_path: Path | None = None,
) -> TopicSelection:
    """Load and revalidate a previously persisted candidate matrix."""

    raw = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("persisted topic candidate matrix is not a JSON object")
    if selected_path is not None and selected_path.is_file():
        selected_raw = json.loads(selected_path.read_text(encoding="utf-8"))
        if isinstance(selected_raw, dict) and selected_raw.get("id"):
            raw["selected_candidate_id"] = selected_raw["id"]
            raw["selection_rationale"] = str(
                selected_raw.get(
                    "selection_rationale",
                    raw.get("selection_rationale", ""),
                )
                or ""
            )
    return validate_selection_document(raw)


def normalize_topic_action(
    diagnosis: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Return a fail-closed topic action from a cycle diagnosis.

    ``keep`` is the safe default. ``pivot`` is accepted only with an
    evidence-based reason, while ``refine`` additionally requires a concrete
    revised topic specification. The legacy ``brief_patch`` field is ignored
    so campaign policy cannot be repurposed as a topic mutation.
    """

    raw = diagnosis if isinstance(diagnosis, Mapping) else {}
    action = str(raw.get("topic_action", "keep") or "keep").strip().casefold()
    if action not in TOPIC_ACTIONS:
        action = "keep"
    reason = str(raw.get("pivot_reason", "") or "").strip()
    preferred = str(raw.get("preferred_candidate_id", "") or "").strip()
    patch = str(raw.get("topic_patch", "") or "").strip()
    if action == "pivot" and not reason:
        action = "keep"
        preferred = ""
    if action == "refine" and not patch:
        action = "keep"
    return {
        "topic_action": action,
        "pivot_reason": reason,
        "preferred_candidate_id": preferred,
        "topic_patch": patch,
    }


def selection_with_candidate(
    selection: TopicSelection,
    *,
    candidate_id: str,
    rationale: str,
) -> TopicSelection:
    """Return *selection* with one validated matrix candidate selected."""

    selected = next(
        (
            candidate
            for candidate in selection.document["candidates"]
            if candidate["id"] == candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"unknown preferred topic candidate: {candidate_id}")
    document = dict(selection.document)
    document["generated_at"] = utc_now()
    document["selected_candidate_id"] = selected["id"]
    document["selection_rationale"] = rationale.strip() or (
        f"Pivoted to validated candidate {selected['id']}."
    )
    return TopicSelection(document=document, selected=dict(selected))
