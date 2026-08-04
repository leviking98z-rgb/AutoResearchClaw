from __future__ import annotations

import json

from researchclaw.literature.infohub import (
    InfoHubClient,
    deduplicate_papers,
    paper_identity,
)
from researchclaw.literature.models import Paper


def test_infohub_http_search_converts_rows(
    monkeypatch,
) -> None:
    client = InfoHubClient(base_url="http://infohub.invalid")

    def fake_json(path, **kwargs):
        assert path == "/api/find"
        assert kwargs["params"]["q"] == "self improving agents"
        return {
            "items": [
                {
                    "platform": "arxiv",
                    "post_id": "2601.12345v2",
                    "title": "Self Improving Agents",
                    "author": "Ada Lovelace and Alan Turing",
                    "url": "https://arxiv.org/abs/2601.12345",
                    "ts": "2026-01-02",
                    "abstract": "A persistent agent improvement method.",
                }
            ]
        }

    monkeypatch.setattr(client, "_http_json", fake_json)
    result = client.search("self improving agents")

    assert result.available
    assert len(result.papers) == 1
    paper = result.papers[0]
    assert paper.arxiv_id == "2601.12345"
    assert paper.year == 2026
    assert paper.source == "infohub:arxiv"
    assert [author.name for author in paper.authors] == [
        "Ada Lovelace",
        "Alan Turing",
    ]


def test_infohub_collect_previews_then_ingests(monkeypatch) -> None:
    client = InfoHubClient()
    calls: list[tuple[str, dict]] = []

    def fake_json(path, **kwargs):
        calls.append((path, kwargs))
        if path == "/api/preview":
            return {
                "cards": [
                    {
                        "post": {
                            "platform": "arxiv",
                            "post_id": "2602.00001v1",
                            "title": "Calibration Gates",
                            "url": "https://arxiv.org/abs/2602.00001",
                            "author": "Researcher",
                            "publish_ts": 1769904000,
                            "abstract": "Gate regressions during iterative optimization.",
                        }
                    }
                ],
                "errs": [],
            }
        assert path == "/api/ingest"
        return {"new": 1, "total": 1}

    monkeypatch.setattr(client, "_http_json", fake_json)
    result = client.collect(["calibration gates"])

    assert result.available
    assert result.new_items == 1
    assert result.total_items == 1
    assert len(result.papers) == 1
    assert [call[0] for call in calls] == ["/api/preview", "/api/ingest"]


def test_infohub_ingest_serializes_papers(monkeypatch) -> None:
    client = InfoHubClient()
    payloads: list[dict] = []

    def fake_json(path, **kwargs):
        assert path == "/api/ingest"
        payloads.append(kwargs["payload"])
        return {"new": 1, "total": 1}

    monkeypatch.setattr(client, "_http_json", fake_json)
    result = client.ingest_papers(
        [
            Paper(
                paper_id="s2-1",
                title="Evidence Gated Self Improvement",
                year=2026,
                abstract="Measured evidence.",
                doi="10.1234/example",
                url="https://doi.org/10.1234/example",
                source="openalex",
            )
        ],
        keyword="researchclaw:test",
    )

    assert result.available
    assert result.new_items == 1
    post = payloads[0]["posts"][0]
    assert post["post_id"] == "10.1234/example"
    assert post["keyword"] == "researchclaw:test"
    assert json.loads(post["raw"])["source"] == "openalex"


def test_deduplicate_prefers_more_complete_record() -> None:
    sparse = Paper(paper_id="1", title="Same Paper", doi="10.1/x")
    rich = Paper(
        paper_id="2",
        title="Same Paper",
        doi="10.1/x",
        abstract="Full abstract",
        citation_count=4,
    )
    result = deduplicate_papers([sparse, rich])

    assert result == [rich]
    assert paper_identity(rich) == "doi:10.1/x"


def test_infohub_failure_is_nonfatal(monkeypatch) -> None:
    client = InfoHubClient()

    def fail(*_args, **_kwargs):
        raise TimeoutError("down")

    monkeypatch.setattr(client, "_http_json", fail)
    result = client.search("query")

    assert not result.available
    assert result.papers == ()
    assert "TimeoutError" in result.error
