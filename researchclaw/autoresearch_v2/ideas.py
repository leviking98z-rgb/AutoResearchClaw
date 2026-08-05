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
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
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
_PUBLIC_DATASET_MARKERS = (
    "arc",
    "bbh",
    "bigbench",
    "big-bench",
    "cifar",
    "commonsenseqa",
    "drop",
    "evalplus",
    "gsm8k",
    "hellaswag",
    "humaneval",
    "imagenet",
    "math",
    "mbpp",
    "mmlu",
    "openwebtext",
    "public",
    "squad",
    "truthfulqa",
    "wikitext",
)
_OPEN_MODEL_MARKERS = (
    "apache",
    "bloom",
    "falcon",
    "gemma",
    "gpt2",
    "gpt-2",
    "huggingface",
    "llama",
    "mistral",
    "mit license",
    "open weights",
    "open-weight",
    "phi",
    "qwen",
    "smollm",
    "t5",
    "tinyllama",
)
_BOUND_MARKERS = (
    "batch",
    "budget",
    "epoch",
    "example",
    "gpu",
    "hour",
    "iteration",
    "minute",
    "sample",
    "seed",
    "step",
    "task",
    "token",
)
_AUTOMATED_MEASUREMENT_MARKERS = (
    "accuracy",
    "auc",
    "benchmark",
    "brier",
    "calibration",
    "cost",
    "ece",
    "error",
    "exact match",
    "f1",
    "latency",
    "loss",
    "pass@",
    "precision",
    "recall",
    "regression",
    "reward",
    "runtime",
    "score",
    "throughput",
    "token",
)
_GENERIC_RESOURCE_TOKENS = frozenset(
    {
        "a",
        "an",
        "benchmark",
        "corpus",
        "data",
        "dataset",
        "datasets",
        "model",
        "models",
        "private",
        "proprietary",
        "public",
        "test",
        "unknown",
    }
)
_HUMAN_DEPENDENCY_PATTERNS = (
    re.compile(
        r"\b(?:human|expert|clinician|lawyer|teacher|annotator|reviewer)"
        r"(?:[- ](?:rated|judged|graded|scored|reviewed|annotated|labeled))?\b"
    ),
    re.compile(r"\bmanual(?:ly)? (?:review|rating|grading|scoring|annotation|label)"),
    re.compile(r"\b(?:user|participant) study\b"),
    re.compile(r"\b(?:irb|ethics approval|institutional review board)\b"),
    re.compile(r"\b(?:interview|survey|focus group)\b"),
)
_HUMAN_REPLACEMENT_MARKERS = (
    "already labeled",
    "automatic",
    "automated",
    "deterministic",
    "existing labels",
    "gold label",
    "gold labels",
    "public labels",
    "released annotations",
)


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


def _items(candidate: Mapping[str, Any], field: str) -> list[str]:
    value = candidate.get(field)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _has_marker(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_public_or_concrete(
    items: list[str],
    *,
    markers: tuple[str, ...],
) -> tuple[bool, bool]:
    text = " ".join(items).casefold()
    if _has_marker(text, markers):
        return True, True
    # Named resources are still useful design evidence even when this small
    # deterministic checker does not recognize their catalog entry. Generic
    # placeholders such as "a dataset" must not receive the same credit.
    concrete = any(
        bool(re.search(r"\d|[-_/]", item))
        or len(_tokens(item) - _GENERIC_RESOURCE_TOKENS) >= 2
        for item in items
    )
    return False, concrete


def designability_evidence(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Return a conservative, deterministic cheap-pilot designability audit.

    The audit rewards positive evidence and only emits hard blockers when
    multiple independent signals make the proposed pilot non-executable. It is
    intentionally not a domain whitelist: unfamiliar but concrete resources
    remain reviewable instead of being rejected by brittle keyword matching.
    """

    pilot = str(candidate.get("cheap_pilot", "") or "").strip()
    metric = str(candidate.get("primary_metric", "") or "").strip()
    datasets = _items(candidate, "datasets")
    models = _items(candidate, "models")
    baselines = _items(candidate, "baselines")
    ablations = _items(candidate, "ablations")
    implementation = str(
        candidate.get("implementation_feasibility", "") or ""
    ).strip()
    licensing = str(candidate.get("licensing_feasibility", "") or "").strip()
    pilot_text = f"{pilot} {metric}".casefold()
    resource_text = " ".join(
        (*datasets, *models, implementation, licensing)
    ).casefold()
    full_text = " ".join(
        (
            pilot,
            metric,
            *datasets,
            *models,
            *baselines,
            *ablations,
            implementation,
            licensing,
        )
    ).casefold()

    evidence: list[str] = []
    concerns: list[str] = []
    blockers: list[str] = []

    has_metric = bool(metric)
    has_bound = bool(_NUMBER_RE.search(pilot_text)) and _has_marker(
        pilot_text, _BOUND_MARKERS
    )
    has_automated_measurement = _has_marker(
        pilot_text, _AUTOMATED_MEASUREMENT_MARKERS
    )
    if has_metric and (has_bound or has_automated_measurement):
        evidence.append("measurable_unit")
    elif not has_metric:
        blockers.append("missing_measurable_unit")
    else:
        concerns.append("pilot_unit_not_operationalized")

    dataset_public, dataset_concrete = _looks_public_or_concrete(
        datasets,
        markers=_PUBLIC_DATASET_MARKERS,
    )
    if dataset_public:
        evidence.append("public_dataset")
    elif dataset_concrete:
        evidence.append("concrete_dataset_needs_access_verification")
        concerns.append("dataset_access_unverified")
    else:
        blockers.append("dataset_not_identified")

    model_open, model_concrete = _looks_public_or_concrete(
        models,
        markers=_OPEN_MODEL_MARKERS,
    )
    if model_open or _has_marker(resource_text, _OPEN_MODEL_MARKERS):
        evidence.append("open_or_accessible_model")
    elif model_concrete:
        evidence.append("concrete_model_needs_access_verification")
        concerns.append("model_access_unverified")
    else:
        blockers.append("model_not_identified")

    baseline_texts = [item.casefold() for item in baselines]
    no_improvement_controls = (
        "no-self-improvement",
        "no self-improvement",
        "single-pass",
        "single pass",
        "fixed policy",
    )
    has_independent_baseline = any(
        _has_marker(item, no_improvement_controls)
        or _has_marker(
            item,
            (
                "baseline model",
                "best published",
                "current method",
                "heuristic",
                "prior method",
                "random policy",
                "standard method",
                "state of the art",
            ),
        )
        for item in baseline_texts
    )
    if has_independent_baseline:
        evidence.append("independent_baseline")
    else:
        blockers.append("independent_baseline_missing")

    compute = candidate.get("compute")
    gpu_count: int | None = None
    wall_clock_hours: float | None = None
    if isinstance(compute, Mapping):
        try:
            gpu_count = int(compute.get("gpu_count", -1))
            wall_clock_hours = float(compute.get("wall_clock_hours", -1))
        except (TypeError, ValueError):
            pass
    if (
        gpu_count is None
        or wall_clock_hours is None
        or gpu_count < 0
        or not math.isfinite(wall_clock_hours)
        or wall_clock_hours <= 0
    ):
        blockers.append("compute_budget_invalid")
    else:
        gpu_hours = gpu_count * wall_clock_hours
        if gpu_hours <= 2.0 and gpu_count <= 2:
            evidence.append("cheap_compute_budget")
        elif gpu_hours <= 8.0 and gpu_count <= 4:
            concerns.append("pilot_budget_above_cheap_target")
        else:
            concerns.append("pilot_budget_complex")
            if not has_bound:
                blockers.append("unbounded_complex_pilot")

    hidden_human_dependencies = sorted(
        {
            pattern.pattern
            for pattern in _HUMAN_DEPENDENCY_PATTERNS
            if pattern.search(full_text)
        }
    )
    has_human_replacement = _has_marker(full_text, _HUMAN_REPLACEMENT_MARKERS)
    if hidden_human_dependencies and not has_human_replacement:
        concerns.append("hidden_human_or_review_dependency")
        # Human studies or approvals are not a cheap autonomous pilot unless a
        # bounded, already-released annotation source replaces live recruitment.
        if not has_bound or any(
            marker in full_text
            for marker in (
                "participant study",
                "user study",
                "irb",
                "ethics approval",
                "institutional review board",
            )
        ):
            blockers.append("external_human_dependency")
    elif hidden_human_dependencies:
        evidence.append("human_labels_already_available")

    raw_penalty = 0.0
    concern_penalties = {
        "pilot_unit_not_operationalized": 0.45,
        "dataset_access_unverified": 0.20,
        "model_access_unverified": 0.20,
        "pilot_budget_above_cheap_target": 0.45,
        "pilot_budget_complex": 0.90,
        "hidden_human_or_review_dependency": 0.90,
    }
    blocker_penalties = {
        "missing_measurable_unit": 1.25,
        "dataset_not_identified": 1.00,
        "model_not_identified": 1.00,
        "independent_baseline_missing": 1.25,
        "compute_budget_invalid": 1.25,
        "unbounded_complex_pilot": 1.25,
        "external_human_dependency": 1.50,
    }
    raw_penalty += sum(concern_penalties.get(item, 0.25) for item in concerns)
    raw_penalty += sum(blocker_penalties.get(item, 0.75) for item in blockers)
    penalty = round(min(3.0, raw_penalty), 4)
    return {
        "version": 1,
        "designable": not blockers,
        "penalty": penalty,
        "evidence": sorted(set(evidence)),
        "concerns": sorted(set(concerns)),
        "blockers": sorted(set(blockers)),
        "facts": {
            "gpu_count": gpu_count,
            "wall_clock_hours": wall_clock_hours,
            "gpu_hours": (
                round(gpu_count * wall_clock_hours, 4)
                if gpu_count is not None
                and wall_clock_hours is not None
                and gpu_count >= 0
                and math.isfinite(wall_clock_hours)
                and wall_clock_hours > 0
                else None
            ),
            "datasets": datasets,
            "models": models,
            "baselines": baselines,
            "pilot_has_numeric_bound": has_bound,
            "hidden_human_signals": len(hidden_human_dependencies),
        },
    }


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
    designability = candidate.get("designability")
    if not isinstance(designability, Mapping):
        designability = designability_evidence(candidate)
    try:
        penalty = float(designability.get("penalty", 0.0) or 0.0)
    except (TypeError, ValueError):
        penalty = 0.0
    if not math.isfinite(penalty) or penalty < 0:
        penalty = 0.0
    return round(max(0.0, min(10.0, positive - risk - penalty)), 4)


def candidate_to_idea(candidate: Mapping[str, Any]) -> IdeaRecord:
    title = str(candidate["title"]).strip()
    enriched = dict(candidate)
    enriched["designability"] = designability_evidence(candidate)
    score = weighted_score(enriched)
    enriched["weighted_score"] = score
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
        candidate=enriched,
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
        semantic_duplicate_threshold: float = 0.72,
        require_novelty_evidence: bool = False,
    ) -> None:
        self.duplicate_threshold = duplicate_threshold
        self.max_same_family = max_same_family
        self.minimum_score = minimum_score
        self.semantic_duplicate_threshold = semantic_duplicate_threshold
        self.require_novelty_evidence = require_novelty_evidence

    def decide(
        self,
        idea: IdeaRecord,
        *,
        existing: Iterable[IdeaRecord],
    ) -> AdmissionDecision:
        errors = validate_candidate(idea.candidate)
        if errors:
            return AdmissionDecision(False, "malformed:" + ";".join(errors))
        designability = idea.candidate.get("designability")
        if not isinstance(designability, Mapping):
            designability = designability_evidence(idea.candidate)
            idea.candidate["designability"] = designability
        blockers = designability.get("blockers", [])
        if (
            designability.get("designable") is False
            or isinstance(blockers, list)
            and blockers
        ):
            reason = ",".join(str(item) for item in blockers if str(item))
            return AdmissionDecision(
                False,
                "not_designable:" + (reason or "deterministic_gate"),
            )
        if idea.score < self.minimum_score:
            return AdmissionDecision(False, "score_below_threshold")
        if self.require_novelty_evidence:
            evidence = idea.candidate.get("novelty_evidence", {})
            if not isinstance(evidence, Mapping):
                return AdmissionDecision(False, "novelty_evidence_missing")
            if evidence.get("available") is not True:
                return AdmissionDecision(
                    False,
                    "novelty_evidence_unavailable",
                )
            if not evidence.get("closest_papers"):
                return AdmissionDecision(
                    False,
                    "novelty_evidence_empty",
                )
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
                or (
                    idea.family == other.family
                    and semantic >= self.semantic_duplicate_threshold
                    and similarity >= self.duplicate_threshold * 0.5
                )
            ):
                return AdmissionDecision(
                    False,
                    (
                        f"duplicate:title_similarity={similarity:.3f};"
                        f"semantic_similarity={semantic:.3f}"
                    ),
                    other.idea_id,
                )
        from .models import ACTIVE_IDEA_STATUSES, IdeaStatus

        portfolio_statuses = {
            IdeaStatus.RESERVOIR,
            *ACTIVE_IDEA_STATUSES,
        }
        same_family = sum(
            other.family == idea.family
            and other.status in portfolio_statuses
            for other in current
        )
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
        utility_llm: Any | None = None,
    ) -> None:
        self.llm = llm
        self.utility_llm = utility_llm
        self.brief = brief
        self.literature = literature
        self.round = 0
        self.last_context = LiteratureContext()
        self.last_rejections: list[dict[str, Any]] = []

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
                "research_question": idea.research_question,
                "falsifiable_hypothesis": idea.falsifiable_hypothesis,
                "primary_metric": idea.primary_metric,
                "novelty_gap": str(
                    idea.candidate.get("novelty_gap", "") or ""
                ),
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
        if self.utility_llm is not None:
            prompt += (
                "\n\nUtility-tier literature map:\n"
                + json.dumps(
                    self._utility_literature_map(self.last_context),
                    ensure_ascii=False,
                    indent=2,
                )[:12000]
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
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("idea board returned no candidate list")
        self.last_rejections = []
        ideas: list[IdeaRecord] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                self.last_rejections.append(
                    {
                        "index": index,
                        "reason": "candidate must be an object",
                    }
                )
                continue
            errors = validate_candidate(candidate)
            if errors:
                self.last_rejections.append(
                    {
                        "index": index,
                        "id": str(candidate.get("id", "") or ""),
                        "title": str(candidate.get("title", "") or ""),
                        "reason": "; ".join(errors),
                    }
                )
                continue
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
            idea.score = evidence_adjusted_score(idea)
            idea.priority = round(idea.score / 10.0, 4)
            idea.candidate["weighted_score"] = idea.score
            ideas.append(idea)
        if not ideas:
            reasons = "; ".join(
                str(item.get("reason", "") or "")
                for item in self.last_rejections[:3]
            )
            raise ValueError(
                "idea board returned no valid candidates"
                + (f": {reasons}" if reasons else "")
            )
        return sorted(
            ideas,
            key=lambda item: (-item.score, item.idea_id),
        )[:requested]

    def _utility_literature_map(
        self,
        literature: LiteratureContext,
    ) -> dict[str, Any]:
        response = self.utility_llm.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Extract a compact research landscape from these "
                        "papers. Return JSON with mechanisms, open_gaps, "
                        "avoid_duplicates, and suggested_short_queries:\n"
                        + json.dumps(
                            list(literature.papers),
                            ensure_ascii=False,
                        )[:24000]
                    ),
                }
            ],
            system=(
                "You are a low-cost literature organizer. Return only JSON; "
                "do not select the final research idea."
            ),
            json_mode=True,
            max_tokens=4000,
            temperature=0.1,
        )
        try:
            value = self._json_object(response.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return value

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


def evidence_adjusted_score(idea: IdeaRecord) -> float:
    """Temper self-reported board scores with durable novelty evidence."""

    score = float(idea.score)
    evidence = idea.candidate.get("novelty_evidence", {})
    if not isinstance(evidence, Mapping):
        return round(max(0.0, min(10.0, score - 0.5)), 4)
    papers = evidence.get("closest_papers", [])
    if not isinstance(papers, list) or not papers:
        # Missing evidence is uncertainty, never evidence of high novelty.
        return round(max(0.0, min(10.0, score - 0.75)), 4)
    try:
        similarity = float(evidence.get("max_similarity", 0.0) or 0.0)
    except (TypeError, ValueError):
        similarity = 0.0
    if similarity >= 0.72:
        score -= 1.25
    elif similarity >= 0.50:
        score -= 0.55
    # A grounded nearest-work set is itself valuable even when similarity is
    # low: the idea can now be designed and reviewed against real precedents.
    score += min(0.25, len(papers) * 0.025)
    return round(max(0.0, min(10.0, score)), 4)
