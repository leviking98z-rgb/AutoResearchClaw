"""Persistent InfoHub context for Idea generation and novelty evidence."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Protocol

from .config import LiteratureConfig
from .models import IdeaRecord

_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_STOP = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "using",
    "based",
    "large",
    "language",
    "model",
    "models",
    "research",
}


@dataclass(frozen=True, slots=True)
class LiteratureContext:
    papers: tuple[dict[str, Any], ...] = ()
    queries: tuple[str, ...] = ()
    available: bool = False
    refreshed: bool = False
    error: str = ""


class LiteratureProvider(Protocol):
    def context_for_board(
        self,
        brief: str,
        existing: Iterable[IdeaRecord],
    ) -> LiteratureContext: ...

    def evidence_for_idea(self, idea: IdeaRecord) -> dict[str, Any]: ...


class InfoHubLiteratureProvider:
    def __init__(self, config: LiteratureConfig) -> None:
        from researchclaw.literature.infohub import InfoHubClient

        self.client = InfoHubClient(
            enabled=config.enabled,
            mode=config.mode,
            base_url=config.url,
            repo_path=config.repo,
            timeout_sec=config.timeout_sec,
            search_limit=config.search_limit,
            collect_days=config.collect_days,
            collect_platforms=config.collect_platforms,
        )
        self.config = config

    def context_for_board(
        self,
        brief: str,
        existing: Iterable[IdeaRecord],
    ) -> LiteratureContext:
        queries = self._board_queries(brief, existing)
        papers: list[Any] = []
        errors: list[str] = []
        available = False
        for query in queries:
            result = self.client.search(
                query,
                limit=self.config.search_limit,
            )
            papers.extend(result.papers)
            available = available or result.available
            if result.error:
                errors.append(f"{query}: {result.error}")
        refreshed = False
        if (
            self.config.refresh_on_low_results
            and len(papers) < self.config.min_results
        ):
            refreshed_result = self.client.collect(
                queries,
                limit_per_query=max(
                    1, self.config.search_limit // max(1, len(queries))
                ),
            )
            refreshed = refreshed_result.available
            available = available or refreshed_result.available
            papers.extend(refreshed_result.papers)
            if refreshed_result.error:
                errors.append(refreshed_result.error)
        from researchclaw.literature.infohub import deduplicate_papers

        unique = deduplicate_papers(papers)[: self.config.search_limit]
        return LiteratureContext(
            papers=tuple(_paper_summary(paper) for paper in unique),
            queries=tuple(queries),
            available=available,
            refreshed=refreshed,
            error="; ".join(errors),
        )

    def evidence_for_idea(self, idea: IdeaRecord) -> dict[str, Any]:
        queries = [
            idea.title,
            f"{idea.research_question} {idea.falsifiable_hypothesis}",
        ]
        papers: list[Any] = []
        errors: list[str] = []
        available = False
        for query in queries:
            result = self.client.search(
                query,
                limit=self.config.search_limit,
            )
            available = available or result.available
            papers.extend(result.papers)
            if result.error:
                errors.append(f"{query}: {result.error}")
        ranked = sorted(
            (
                (
                    semantic_similarity(
                        f"{idea.title} {idea.research_question} "
                        f"{idea.falsifiable_hypothesis}",
                        f"{paper.title} {paper.abstract}",
                    ),
                    paper,
                )
                for paper in papers
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        top = [
            {**_paper_summary(paper), "similarity": round(score, 4)}
            for score, paper in ranked[:10]
        ]
        max_similarity = top[0]["similarity"] if top else 0.0
        return {
            "queries": queries,
            "available": available,
            "error": "; ".join(errors),
            "closest_papers": top,
            "max_similarity": max_similarity,
            "novelty_risk": (
                "high"
                if max_similarity >= 0.72
                else "medium"
                if max_similarity >= 0.50
                else "low"
            ),
        }

    @staticmethod
    def _board_queries(
        brief: str,
        existing: Iterable[IdeaRecord],
    ) -> list[str]:
        queries = [
            "recursive self improvement large language model agents",
            "self refinement language model verifier calibration",
        ]
        keywords = [
            word
            for word in _tokens(brief)
            if word not in _STOP
        ][:10]
        if keywords:
            queries.append(" ".join(keywords))
        families = sorted(
            {
                idea.family.replace("-", " ")
                for idea in existing
                if idea.family and idea.family != "other"
            }
        )
        queries.extend(
            f"large language model self improvement {family}"
            for family in families[-3:]
        )
        return list(dict.fromkeys(queries))[:6]


def semantic_similarity(left: str, right: str) -> float:
    a = set(_tokens(left))
    b = set(_tokens(right))
    jaccard = len(a & b) / len(a | b) if a and b else 0.0
    sequence = SequenceMatcher(None, left.casefold(), right.casefold()).ratio()
    # Jaccard over research descriptions is often inflated by domain boilerplate
    # (benchmark/model/metric terms). Require close phrasing as corroboration
    # instead of treating shared vocabulary alone as a semantic duplicate.
    return round(jaccard * sequence, 4)


def _tokens(value: str) -> list[str]:
    return [
        token
        for token in _WORD_RE.findall(value.casefold())
        if token not in _STOP
    ]


def _paper_summary(paper: Any) -> dict[str, Any]:
    return {
        "paper_id": str(getattr(paper, "paper_id", "") or ""),
        "title": str(getattr(paper, "title", "") or ""),
        "year": int(getattr(paper, "year", 0) or 0),
        "abstract": str(getattr(paper, "abstract", "") or "")[:2000],
        "venue": str(getattr(paper, "venue", "") or ""),
        "citation_count": int(
            getattr(paper, "citation_count", 0) or 0
        ),
        "doi": str(getattr(paper, "doi", "") or ""),
        "arxiv_id": str(getattr(paper, "arxiv_id", "") or ""),
        "url": str(getattr(paper, "url", "") or ""),
        "source": str(getattr(paper, "source", "") or ""),
    }
