from __future__ import annotations

from dataclasses import dataclass

from researchclaw.autoresearch_v2.config import LiteratureConfig
from researchclaw.autoresearch_v2.literature import (
    InfoHubLiteratureProvider,
    semantic_similarity,
)
from researchclaw.autoresearch_v2.models import IdeaRecord
from researchclaw.literature.infohub import InfoHubResult
from researchclaw.literature.models import Paper


def _idea() -> IdeaRecord:
    return IdeaRecord(
        idea_id="memory",
        title="Quarantine and Recovery from Poisoned Self-Written Memories",
        research_question=(
            "Can contradiction-triggered quarantine stop poisoned agent memory?"
        ),
        falsifiable_hypothesis=(
            "Quarantine recovers accuracy after memory poisoning."
        ),
        primary_metric="accuracy",
        family="memory",
        candidate={"novelty_gap": "recovery after poisoned reflection memory"},
    )


@dataclass
class _Client:
    collected: bool = False

    def search(self, query: str, *, limit: int) -> InfoHubResult:
        del limit
        if self.collected and "memory" in query:
            return InfoHubResult(
                papers=(
                    Paper(
                        paper_id="prior",
                        title="Memory Poisoning Attacks on LLM Agents",
                        abstract=(
                            "Persistent agent memory can be poisoned and "
                            "requires defensive recovery mechanisms."
                        ),
                    ),
                ),
                available=True,
            )
        return InfoHubResult(available=True)

    def collect(self, queries, *, limit_per_query: int) -> InfoHubResult:
        del queries, limit_per_query
        self.collected = True
        return InfoHubResult(available=True, new_items=1)


def test_novelty_evidence_refreshes_and_requeries_infohub() -> None:
    provider = InfoHubLiteratureProvider.__new__(
        InfoHubLiteratureProvider
    )
    provider.config = LiteratureConfig(
        min_results=1,
        refresh_on_low_results=True,
    )
    provider.client = _Client()
    evidence = provider.evidence_for_idea(_idea())
    assert evidence["refreshed"] is True
    assert evidence["closest_papers"]
    assert evidence["max_similarity"] > 0


def test_semantic_similarity_handles_camel_case_and_paraphrase() -> None:
    score = semantic_similarity(
        "MemMorph memory poisoning recovery for LLM agents",
        "Memory poisoning attacks and defenses in language-model agents",
    )
    assert score > 0
