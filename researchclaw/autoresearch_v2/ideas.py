"""High-quality continuous Idea generation and deterministic admission."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .literature import LiteratureContext, LiteratureProvider, semantic_similarity
from .models import IdeaRecord, stable_id

MIN_IDEA_CANDIDATES = 12
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SCORE_FIELDS = (
    "novelty",
    "scientific_importance",
    "falsifiability",
    "compute_tractability",
    "reproducibility",
    "meaningful_result_likelihood",
)
_WEIGHTS = {
    "novelty": 0.22,
    "scientific_importance": 0.20,
    "falsifiability": 0.16,
    "compute_tractability": 0.14,
    "reproducibility": 0.13,
    "meaningful_result_likelihood": 0.15,
}


class IdeaGenerator(Protocol):
    def generate(
        self,
        *,
        count: int,
        existing: Iterable[IdeaRecord],
    ) -> list[IdeaRecord]: ...


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.casefold()))


def title_similarity(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def infer_family(candidate: Mapping[str, Any]) -> str:
    explicit = str(candidate.get("family", "") or "").strip()
    if explicit:
        return explicit.casefold().replace(" ", "-")
    text = " ".join(
        str(candidate.get(key, "") or "")
        for key in ("title", "research_question", "novelty_gap")
    ).casefold()
    for marker, family in (
        ("verifier", "verifier"),
        ("calibrat", "calibration"),
        ("memory", "memory"),
        ("population", "population"),
        ("credit assignment", "credit-assignment"),
        ("self-train", "self-training"),
        ("reward", "reward-modeling"),
        ("reflection", "reflection"),
        ("tool", "tool-use"),
    ):
        if marker in text:
            return family
    return "other"


def validate_candidate(candidate: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "title",
        "research_question",
        "falsifiable_hypothesis",
        "novelty_gap",
        "primary_metric",
        "cheap_pilot",
        "information_gain_if_false",
        "implementation_feasibility",
        "licensing_feasibility",
    ):
        if not str(candidate.get(field, "") or "").strip():
            errors.append(f"missing {field}")
    for field in (
        "closest_prior_work",
        "datasets",
        "models",
        "baselines",
        "ablations",
        "failure_safety_tests",
    ):
        value = candidate.get(field)
        if not isinstance(value, list) or not any(str(x).strip() for x in value):
            errors.append(f"missing {field}")
    baselines = " ".join(
        str(item) for item in candidate.get("baselines", [])
    ).casefold()
    if not any(
        marker in baselines
        for marker in (
            "no-self-improvement",
            "no self-improvement",
            "single-pass",
            "single pass",
            "fixed policy",
        )
    ):
        errors.append("missing no-self-improvement control")
    compute = candidate.get("compute")
    if not isinstance(compute, Mapping):
        errors.append("missing compute")
    else:
        try:
            gpus = int(compute.get("gpu_count", -1))
            hours = float(compute.get("wall_clock_hours", -1))
        except (TypeError, ValueError):
            errors.append("invalid compute")
        else:
            if not 0 <= gpus <= 32 or not math.isfinite(hours) or hours <= 0:
                errors.append("compute outside supported bounds")
    scores = candidate.get("scores")
    if not isinstance(scores, Mapping):
        errors.append("missing scores")
    else:
        for field in (*_SCORE_FIELDS, "risk"):
            try:
                score = float(scores.get(field))
            except (TypeError, ValueError):
                errors.append(f"invalid score {field}")
                continue
            if not math.isfinite(score) or not 0 <= score <= 10:
                errors.append(f"invalid score {field}")
    return errors


def weighted_score(candidate: Mapping[str, Any]) -> float:
    scores = candidate["scores"]
    positive = sum(float(scores[name]) * _WEIGHTS[name] for name in _SCORE_FIELDS)
    risk = float(scores["risk"]) * 0.12
    return round(max(0.0, min(10.0, positive - risk)), 4)


def candidate_to_idea(candidate: Mapping[str, Any]) -> IdeaRecord:
    title = str(candidate["title"]).strip()
    score = weighted_score(candidate)
    return IdeaRecord(
        idea_id=stable_id(
            str(candidate.get("id", "") or title),
            prefix="idea",
        ),
        title=title,
        research_question=str(candidate["research_question"]).strip(),
        falsifiable_hypothesis=str(
            candidate["falsifiable_hypothesis"]
        ).strip(),
        primary_metric=str(candidate["primary_metric"]).strip(),
        candidate={**dict(candidate), "weighted_score": score},
        score=score,
        family=infer_family(candidate),
        priority=round(score / 10.0, 4),
    )


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reason: str
    duplicate_of: str = ""


class IdeaAdmission:
    def __init__(
        self,
        *,
        duplicate_threshold: float = 0.72,
        max_same_family: int = 2,
        minimum_score: float = 6.0,
        semantic_duplicate_threshold: float = 0.995,
    ) -> None:
        self.duplicate_threshold = duplicate_threshold
        self.max_same_family = max_same_family
        self.minimum_score = minimum_score
        self.semantic_duplicate_threshold = semantic_duplicate_threshold

    def decide(
        self,
        idea: IdeaRecord,
        *,
        existing: Iterable[IdeaRecord],
    ) -> AdmissionDecision:
        errors = validate_candidate(idea.candidate)
        if errors:
            return AdmissionDecision(False, "malformed:" + ";".join(errors))
        if idea.score < self.minimum_score:
            return AdmissionDecision(False, "score_below_threshold")
        current = list(existing)
        for other in current:
            similarity = title_similarity(idea.title, other.title)
            semantic = semantic_similarity(
                (
                    f"{idea.title} {idea.research_question} "
                    f"{idea.falsifiable_hypothesis}"
                ),
                (
                    f"{other.title} {other.research_question} "
                    f"{other.falsifiable_hypothesis}"
                ),
            )
            if (
                idea.idea_id == other.idea_id
                or idea.title.casefold() == other.title.casefold()
                or similarity >= self.duplicate_threshold
                or semantic >= self.semantic_duplicate_threshold
            ):
                return AdmissionDecision(
                    False,
                    (
                        f"duplicate:title_similarity={similarity:.3f};"
                        f"semantic_similarity={semantic:.3f}"
                    ),
                    other.idea_id,
                )
        same_family = sum(other.family == idea.family for other in current)
        if same_family >= self.max_same_family:
            return AdmissionDecision(False, "family_quota")
        return AdmissionDecision(True, "admitted")


class StaticIdeaGenerator:
    def __init__(self, candidates: Iterable[Mapping[str, Any]]) -> None:
        self._candidates = [dict(candidate) for candidate in candidates]
        self._index = 0

    def generate(
        self,
        *,
        count: int,
        existing: Iterable[IdeaRecord],
    ) -> list[IdeaRecord]:
        del existing
        selected = self._candidates[self._index : self._index + count]
        self._index += len(selected)
        return [candidate_to_idea(candidate) for candidate in selected]


class LLMBoardIdeaGenerator:
    """Generate a structured candidate board using the decision model only."""

    def __init__(
        self,
        *,
        llm: Any,
        brief: str,
        literature: LiteratureProvider | None = None,
    ) -> None:
        self.llm = llm
        self.brief = brief
        self.literature = literature
        self.round = 0
        self.last_context = LiteratureContext()

    def generate(
        self,
        *,
        count: int,
        existing: Iterable[IdeaRecord],
    ) -> list[IdeaRecord]:
        self.round += 1
        archive = [
            {
                "title": idea.title,
                "family": idea.family,
                "status": idea.status.value,
                "exit_reason": idea.exit_reason,
            }
            for idea in existing
        ][-80:]
        requested = max(MIN_IDEA_CANDIDATES, count)
        self.last_context = (
            self.literature.context_for_board(self.brief, existing)
            if self.literature is not None
            else LiteratureContext()
        )
        prompt = self._prompt(
            requested=requested,
            archive=archive,
            literature=self.last_context,
        )
        response = self.llm.chat(
            [{"role": "user", "content": prompt}],
            system=(
                "You are a skeptical autonomous research portfolio board. "
                "Return only JSON. Prefer mechanistic, falsifiable, high-"
                "information-gain projects; avoid cosmetic prompt variants."
            ),
            json_mode=True,
            max_tokens=24000,
            temperature=0.35,
        )
        raw = self._json_object(response.content)
        candidates = raw.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < requested:
            raise ValueError(
                f"idea board requires at least {requested} candidates"
            )
        ideas: list[IdeaRecord] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError("candidate must be an object")
            errors = validate_candidate(candidate)
            if errors:
                raise ValueError("invalid candidate: " + "; ".join(errors))
            idea = candidate_to_idea(candidate)
            idea.candidate["literature_context"] = {
                "queries": list(self.last_context.queries),
                "available": self.last_context.available,
                "refreshed": self.last_context.refreshed,
                "error": self.last_context.error,
                "papers": list(self.last_context.papers),
            }
            if self.literature is not None:
                idea.candidate["novelty_evidence"] = (
                    self.literature.evidence_for_idea(idea)
                )
            ideas.append(idea)
        return sorted(ideas, key=lambda item: (-item.score, item.idea_id))

    def _prompt(
        self,
        *,
        requested: int,
        archive: list[dict[str, Any]],
        literature: LiteratureContext,
    ) -> str:
        return f"""\
Generate {requested} DISTINCT candidate research projects for round {self.round}.

Mission:
{self.brief[:16000]}

Prior/active archive to avoid repeating:
{json.dumps(archive, ensure_ascii=False, indent=2)[:16000]}

Recent and reusable literature evidence from InfoHub:
{json.dumps(list(literature.papers), ensure_ascii=False, indent=2)[:24000]}

Each candidate must test a mechanism of LLM/agent recursive self-improvement,
not merely benchmark a prompt. Favor ideas whose cheap pilot can falsify the
premise in <=2 GPU-hours before scale-up. Preserve valuable negative results.

Return:
{{
  "candidates": [{{
    "id": "stable-id",
    "title": "specific title",
    "research_question": "precise question",
    "falsifiable_hypothesis": "one primary hypothesis",
    "closest_prior_work": ["specific paper IDs/titles from evidence where relevant"],
    "novelty_gap": "exact gap",
    "datasets": ["public benchmark"],
    "models": ["open/legal model"],
    "compute": {{"gpu_count": 1, "wall_clock_hours": 2}},
    "primary_metric": "predeclared endpoint",
    "baselines": ["must include no-self-improvement or single-pass control"],
    "ablations": ["mechanism-isolating ablation"],
    "failure_safety_tests": ["leakage/regression/reward-hacking check"],
    "implementation_feasibility": "dependencies",
    "licensing_feasibility": "license assessment",
    "information_gain_if_true": "what is learned",
    "information_gain_if_false": "what is learned",
    "cheap_pilot": "bounded discriminating pilot and explicit early-stop",
    "scores": {{
      "novelty": 0, "scientific_importance": 0, "falsifiability": 0,
      "compute_tractability": 0, "reproducibility": 0,
      "meaningful_result_likelihood": 0, "risk": 0
    }}
  }}]
}}
All scores are 0-10. Do not select only one winner; produce a diverse board.
"""

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```\w*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        value = json.loads(cleaned)
        return dict(value) if isinstance(value, Mapping) else {}
