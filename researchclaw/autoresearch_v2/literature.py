"""Persistent InfoHub context for Idea generation and novelty evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from .config import LiteratureConfig
from .models import IdeaRecord

_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
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
_QUERY_EXPANSIONS = {
    "calibration": "uncertainty calibration selective prediction",
    "credit": "counterfactual credit assignment causal attribution",
    "memory": "agent memory reflection experience retrieval",
    "poison": "memory poisoning attack defense correction",
    "population": "population search evolutionary agent optimization",
    "reflection": "self reflection self correction iterative refinement",
    "rollback": "rollback regression gate checkpoint reversible update",
    "selfplay": "self play curriculum task generation learnability",
    "verifier": "verifier feedback reward hacking overfitting",
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
        queries = self._idea_queries(idea)
        papers, available, errors = self._search_queries(queries)
        refreshed = False
        if (
            self.config.refresh_on_low_results
            and len(papers) < self.config.min_results
        ):
            refresh = self.client.collect(
                queries,
                limit_per_query=max(
                    1,
                    self.config.search_limit // max(1, len(queries)),
                ),
            )
            refreshed = refresh.available
            available = available or refresh.available
            papers.extend(refresh.papers)
            if refresh.error:
                errors.append(refresh.error)
            # The durable library may rank newly collected records differently
            # from the adapter response. Search again so evidence is exactly
            # what later runs will reuse from InfoHub.
            searched, searched_available, searched_errors = (
                self._search_queries(queries)
            )
            papers.extend(searched)
            available = available or searched_available
            errors.extend(searched_errors)
        from researchclaw.literature.infohub import deduplicate_papers

        papers = deduplicate_papers(papers)
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
            "refreshed": refreshed,
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

    def _search_queries(
        self,
        queries: Iterable[str],
    ) -> tuple[list[Any], bool, list[str]]:
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
        return papers, available, errors

    @staticmethod
    def _idea_queries(idea: IdeaRecord) -> list[str]:
        text = " ".join(
            (
                idea.title,
                idea.research_question,
                idea.falsifiable_hypothesis,
                str(idea.candidate.get("novelty_gap", "") or ""),
            )
        )
        tokens = [
            token
            for token in _tokens(text)
            if token not in _STOP
        ]
        weighted: list[str] = []
        for token in tokens:
            weighted.extend(
                _QUERY_EXPANSIONS.get(token, token).split()
            )
        unique_weighted = list(dict.fromkeys(weighted))
        # InfoHub FTS and academic endpoints are much more reliable on a few
        # focused concepts than on an entire generated hypothesis sentence.
        primary = " ".join(unique_weighted[:8])
        family = idea.family.replace("-", " ")
        family_query = (
            f"large language model agent {family} "
            "self improvement"
        )
        queries = [
            primary,
            family_query,
            " ".join(unique_weighted[-8:]),
        ]
        return [
            query
            for query in dict.fromkeys(
                " ".join(query.split()).strip() for query in queries
            )
            if query
        ][:4]

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
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    # Cosine-style lexical overlap gives useful non-zero novelty ranking for
    # differently phrased papers while remaining conservative for admission.
    return round(intersection / math.sqrt(len(a) * len(b)), 4)


def _tokens(value: str) -> list[str]:
    value = _CAMEL_RE.sub(" ", value)
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
