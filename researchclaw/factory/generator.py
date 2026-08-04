"""Continuous candidate generators and topic-selection adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from .models import Idea, stable_key


class CandidateGenerator(Protocol):
    def generate(self, *, count: int, context: Mapping[str, Any]) -> list[Idea]: ...


def infer_family(candidate: Mapping[str, Any]) -> str:
    explicit = str(candidate.get("family", "") or "").strip()
    if explicit:
        return stable_key(explicit, length=8).rsplit("-", 1)[0]
    title = str(candidate.get("title", "") or "").casefold()
    rules = (
        ("verifier", "verifier-gating"),
        ("calibration", "calibration"),
        ("memory", "memory"),
        ("self-train", "self-training"),
        ("self train", "self-training"),
        ("reward", "reward-modeling"),
        ("reflection", "reflection"),
        ("tool", "tool-use"),
        ("distill", "distillation"),
    )
    return next((family for marker, family in rules if marker in title), "other")


def candidate_to_idea(
    candidate: Mapping[str, Any],
    *,
    source: str = "de_novo",
    parent_ids: Sequence[str] = (),
) -> Idea:
    title = str(candidate.get("title", "") or "").strip()
    raw_id = str(candidate.get("id", "") or "").strip()
    idea_id = f"idea-{stable_key(raw_id or title, length=12)}"
    weighted = float(candidate.get("weighted_score", 0.0) or 0.0)
    priority = max(0.0, min(1.0, weighted / 10.0))
    return Idea(
        idea_id=idea_id,
        title=title,
        research_question=str(candidate.get("research_question", "") or ""),
        falsifiable_hypothesis=str(
            candidate.get("falsifiable_hypothesis", "") or ""
        ),
        primary_metric=str(candidate.get("primary_metric", "") or ""),
        family=infer_family(candidate),
        source=source,
        parent_ids=tuple(parent_ids),
        priority=priority,
        candidate=dict(candidate),
    )


class StaticCandidateGenerator:
    """Deterministic finite/cycling source for tests and offline operation."""

    def __init__(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        repeat: bool = False,
    ) -> None:
        self._candidates = [dict(item) for item in candidates]
        self._index = 0
        self._repeat = repeat

    def generate(self, *, count: int, context: Mapping[str, Any]) -> list[Idea]:
        del context
        generated: list[Idea] = []
        while len(generated) < count and self._candidates:
            if self._index >= len(self._candidates):
                if not self._repeat:
                    break
                self._index = 0
            raw = dict(self._candidates[self._index])
            # Repeated deterministic sources must still create distinct
            # proposals; the admission layer will reject equivalent titles.
            generated.append(candidate_to_idea(raw))
            self._index += 1
        return generated


class CallableCandidateGenerator:
    def __init__(
        self,
        fn: Callable[[int, Mapping[str, Any]], Iterable[Idea]],
    ) -> None:
        self._fn = fn

    def generate(self, *, count: int, context: Mapping[str, Any]) -> list[Idea]:
        return list(self._fn(count, context))


class LLMCandidateGenerator:
    """Reuse the validated RSI topic board while retaining every candidate."""

    def __init__(
        self,
        *,
        llm: Any,
        brief: str,
        minimum_batch_size: int = 12,
    ) -> None:
        self.llm = llm
        self.brief = brief
        self.minimum_batch_size = minimum_batch_size
        self.cycle = 0
        self.previous_selection: dict[str, Any] | None = None

    def generate(self, *, count: int, context: Mapping[str, Any]) -> list[Idea]:
        del context
        try:
            from researchclaw.rsi.topic_selection import select_topics
        except ImportError as exc:
            raise RuntimeError(
                "LLM candidate generation requires the ResearchClaw RSI "
                "topic-selection module; use --simulation-candidates or "
                "install the production RSI components"
            ) from exc
        self.cycle += 1
        selection = select_topics(
            llm=self.llm,
            brief=self.brief,
            cycle=self.cycle,
            previous_selection=self.previous_selection,
        )
        self.previous_selection = selection.document
        ideas = [
            candidate_to_idea(candidate)
            for candidate in selection.document["candidates"]
        ]
        # select_topics currently validates at least 12, so ``count`` is a
        # lower desired batch size rather than a truncation request.  Keeping
        # all candidates preserves the board's comparisons and fills the
        # reservoir efficiently.
        minimum = max(count, self.minimum_batch_size)
        return ideas[: max(minimum, len(ideas))]
