"""Deterministic candidate validation, deduplication, and family quotas."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .config import FactoryConfig
from .models import ACTIVE_IDEA_STATUSES, Idea, IdeaStatus, normalize_title

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(normalize_title(value)))


def title_similarity(left: str, right: str) -> float:
    """Cheap deterministic Jaccard screen; semantic embeddings are a later tier."""

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    admitted: bool
    reason_code: str
    detail: str = ""


class AdmissionController:
    def __init__(
        self,
        config: FactoryConfig,
        *,
        duplicate_threshold: float = 0.90,
    ) -> None:
        self.config = config
        self.duplicate_threshold = duplicate_threshold

    def validate_candidate(self, candidate: Mapping[str, Any]) -> list[str]:
        required_text = (
            "title",
            "research_question",
            "falsifiable_hypothesis",
            "primary_metric",
            "cheap_pilot",
            "information_gain_if_false",
        )
        errors = [
            f"missing {name}"
            for name in required_text
            if not str(candidate.get(name, "") or "").strip()
        ]
        baselines = candidate.get("baselines", ())
        if not isinstance(baselines, (list, tuple)):
            baselines = ()
        baseline_text = " ".join(str(item) for item in baselines).casefold()
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
            errors.append("missing no-self-improvement control")
        compute = candidate.get("compute")
        if not isinstance(compute, Mapping):
            errors.append("missing compute estimate")
        else:
            try:
                gpu_count = int(compute.get("gpu_count", -1))
                wall_hours = float(compute.get("wall_clock_hours", -1))
            except (TypeError, ValueError):
                errors.append("invalid compute estimate")
            else:
                if gpu_count < 0 or gpu_count > 32 or wall_hours <= 0:
                    errors.append("compute estimate outside supported bounds")
        return errors

    def decide(
        self,
        idea: Idea,
        *,
        existing: Iterable[Idea],
    ) -> AdmissionDecision:
        errors = self.validate_candidate(idea.candidate)
        if errors:
            return AdmissionDecision(False, "MALFORMED", "; ".join(errors))

        existing_list = list(existing)
        for other in existing_list:
            if other.idea_id == idea.idea_id:
                return AdmissionDecision(False, "DUPLICATE_ID", other.idea_id)
            if other.normalized_title == idea.normalized_title:
                return AdmissionDecision(False, "DUPLICATE", other.idea_id)
            similarity = title_similarity(other.title, idea.title)
            if similarity >= self.duplicate_threshold:
                return AdmissionDecision(
                    False,
                    "POSSIBLE_DUPLICATE",
                    f"{other.idea_id} title_similarity={similarity:.3f}",
                )

        active_same_family = sum(
            other.family == idea.family
            and other.status in ACTIVE_IDEA_STATUSES
            for other in existing_list
        )
        if (
            active_same_family
            >= self.config.population.max_same_family_active
        ):
            return AdmissionDecision(
                False,
                "FAMILY_QUOTA",
                idea.family,
            )
        return AdmissionDecision(True, "ADMITTED")

    @staticmethod
    def archive_rejection(idea: Idea, decision: AdmissionDecision) -> Idea:
        idea.status = IdeaStatus.REJECTED
        idea.exit_reason = decision.reason_code
        return idea
