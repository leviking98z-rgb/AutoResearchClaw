"""InfoHub-backed persistent literature memory.

The ResearchClaw query cache is intentionally short lived.  InfoHub is the
durable layer: every paper discovered by the research pipeline is upserted into
the shared library, and later runs query that library before touching external
academic APIs.

The adapter supports two deployment modes:

``http``
    Uses the long-running InfoHub web service.  This is the preferred production
    mode because the service owns a warm SQLite connection.

``local``
    Imports a local InfoHub checkout and talks to its Store directly.  This is
    useful for tests and single-machine/offline deployments.

All failures are non-fatal to the research pipeline; callers can fall back to
the existing OpenAlex/Semantic Scholar/arXiv clients.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_ARXIV_RE = re.compile(
    r"(?i)(?:arxiv:|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5})(?:v\d+)?"
)
_DOI_RE = re.compile(r"(?i)(?:doi\.org/|doi:\s*)?(10\.\d{4,9}/[-._;()/:a-z0-9]+)")


@dataclass(frozen=True)
class InfoHubResult:
    """One InfoHub operation result, suitable for logs and stage metadata."""

    papers: tuple[Paper, ...] = ()
    available: bool = False
    new_items: int = 0
    total_items: int = 0
    error: str = ""
    mode: str = ""


def _int_year(value: Any) -> int:
    if isinstance(value, (int, float)):
        raw = int(value)
        if raw > 10_000:
            try:
                return datetime.fromtimestamp(raw, UTC).year
            except (OverflowError, OSError, ValueError):
                return 0
        return raw if 1800 <= raw <= 2200 else 0
    text = str(value or "")
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else 0


def _author_names(value: Any) -> tuple[Author, ...]:
    if isinstance(value, list):
        names: list[Author] = []
        for item in value:
            if isinstance(item, Mapping):
                name = str(item.get("name", "") or "").strip()
                affiliation = str(item.get("affiliation", "") or "").strip()
            else:
                name = str(item or "").strip()
                affiliation = ""
            if name:
                names.append(Author(name=name, affiliation=affiliation))
        return tuple(names)
    text = str(value or "").strip()
    if not text:
        return ()
    # InfoHub has a single display-author column.  Preserve the string rather
    # than guessing aggressively when delimiters are ambiguous.
    parts = [part.strip() for part in re.split(r"\s+and\s+|[;|]", text) if part.strip()]
    return tuple(Author(name=part) for part in parts)


def _extract_arxiv_id(*values: Any) -> str:
    for value in values:
        match = _ARXIV_RE.search(str(value or ""))
        if match:
            return match.group(1)
    return ""


def _extract_doi(*values: Any) -> str:
    for value in values:
        match = _DOI_RE.search(str(value or ""))
        if match:
            return match.group(1).rstrip(".,;)").lower()
    return ""


def _paper_from_row(row: Mapping[str, Any]) -> Paper | None:
    title = str(row.get("title", "") or "").strip()
    if not title or title == "(无标题)":
        return None
    platform = str(row.get("platform", "infohub") or "infohub").strip()
    post_id = str(row.get("post_id", "") or "").strip()
    url = str(row.get("url", "") or "").strip()
    abstract = str(
        row.get("abstract", "") or row.get("snippet", "") or ""
    ).strip()
    raw = row.get("raw", "")
    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    arxiv_id = _extract_arxiv_id(post_id, url, raw_text)
    doi = _extract_doi(url, raw_text)
    return Paper(
        paper_id=post_id or hashlib.sha256(title.lower().encode()).hexdigest()[:20],
        title=title,
        authors=_author_names(row.get("authors") or row.get("author")),
        year=_int_year(row.get("year") or row.get("publish_ts") or row.get("ts")),
        abstract=abstract,
        venue=str(row.get("venue", "") or ""),
        citation_count=int(row.get("citation_count", 0) or 0),
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        source=f"infohub:{platform}",
    )


def paper_identity(paper: Paper) -> str:
    """Stable cross-source identity used for persistent deduplication."""

    if paper.doi:
        return f"doi:{paper.doi.lower().strip()}"
    arxiv_id = _extract_arxiv_id(paper.arxiv_id, paper.url, paper.paper_id)
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    title = re.sub(r"[^a-z0-9]+", " ", paper.title.lower()).strip()
    return f"title:{hashlib.sha256(title.encode()).hexdigest()}"


def deduplicate_papers(papers: Iterable[Paper]) -> list[Paper]:
    """Deduplicate while preferring the most complete representation."""

    best: dict[str, Paper] = {}
    order: list[str] = []
    for paper in papers:
        key = paper_identity(paper)
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = paper
            order.append(key)
            continue
        incumbent_score = (
            bool(incumbent.abstract),
            len(incumbent.abstract),
            bool(incumbent.doi),
            bool(incumbent.arxiv_id),
            incumbent.citation_count,
        )
        candidate_score = (
            bool(paper.abstract),
            len(paper.abstract),
            bool(paper.doi),
            bool(paper.arxiv_id),
            paper.citation_count,
        )
        if candidate_score > incumbent_score:
            best[key] = paper
    return [best[key] for key in order]


class InfoHubClient:
    """Small resilient client for the persistent InfoHub library."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        mode: str = "http",
        base_url: str = "http://127.0.0.1:8077",
        repo_path: str = "/root/servers/infohub",
        timeout_sec: float = 20.0,
        search_limit: int = 60,
        collect_days: int = 3650,
        collect_platforms: Sequence[str] = ("arxiv", "scholar", "bing"),
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = (mode or "http").strip().lower()
        self.base_url = base_url.rstrip("/")
        self.repo_path = Path(repo_path).expanduser()
        self.timeout_sec = max(0.1, float(timeout_sec))
        self.search_limit = max(1, int(search_limit))
        self.collect_days = max(1, int(collect_days))
        self.collect_platforms = tuple(
            str(value).strip() for value in collect_platforms if str(value).strip()
        )

    @classmethod
    def from_config(cls, config: Any) -> InfoHubClient:
        lit = getattr(config, "literature_search", config)
        return cls(
            enabled=getattr(lit, "infohub_enabled", True),
            mode=getattr(lit, "infohub_mode", "http"),
            base_url=getattr(lit, "infohub_url", "http://127.0.0.1:8077"),
            repo_path=getattr(lit, "infohub_repo", "/root/servers/infohub"),
            timeout_sec=getattr(lit, "infohub_timeout_sec", 20.0),
            search_limit=getattr(lit, "infohub_search_limit", 60),
            collect_days=getattr(lit, "infohub_collect_days", 3650),
            collect_platforms=getattr(
                lit,
                "infohub_collect_platforms",
                ("arxiv", "scholar", "bing"),
            ),
        )

    def _http_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(
                {key: value for key, value in params.items() if value not in (None, "")}
            )
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
            decoded = json.loads(response.read().decode("utf-8"))
        if not isinstance(decoded, dict):
            raise TypeError(f"InfoHub returned {type(decoded).__name__}, expected object")
        if decoded.get("error"):
            raise RuntimeError(str(decoded["error"]))
        return decoded

    def _local_modules(self) -> tuple[Any, Any, Any]:
        root = str(self.repo_path.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from infohub.collect import ingest, search_external  # type: ignore[import-not-found]
        from infohub.db import Store  # type: ignore[import-not-found]

        return Store, search_external, ingest

    def health(self) -> InfoHubResult:
        if not self.enabled:
            return InfoHubResult(error="disabled", mode=self.mode)
        try:
            if self.mode == "local":
                Store, _, _ = self._local_modules()
                store = Store()
                try:
                    stats = store.stats()
                finally:
                    store.close()
            else:
                stats = self._http_json("/api/stats")
            return InfoHubResult(
                available=True,
                total_items=int(stats.get("total", 0) or 0),
                mode=self.mode,
            )
        except Exception as exc:  # noqa: BLE001
            return InfoHubResult(error=f"{type(exc).__name__}: {exc}", mode=self.mode)

    def search(self, query: str, *, limit: int | None = None) -> InfoHubResult:
        if not self.enabled or not query.strip():
            return InfoHubResult(error="disabled or empty query", mode=self.mode)
        wanted = max(1, int(limit or self.search_limit))
        try:
            if self.mode == "local":
                Store, _, _ = self._local_modules()
                store = Store()
                try:
                    rows = store.query(text=query, limit=wanted)
                    raw_rows = [dict(row) for row in rows]
                finally:
                    store.close()
            else:
                payload = self._http_json(
                    "/api/find", params={"q": query, "k": wanted}
                )
                raw_rows = [
                    row for row in payload.get("items", []) if isinstance(row, dict)
                ]
            papers = [
                paper for row in raw_rows if (paper := _paper_from_row(row)) is not None
            ]
            return InfoHubResult(
                papers=tuple(deduplicate_papers(papers)),
                available=True,
                total_items=len(raw_rows),
                mode=self.mode,
            )
        except Exception as exc:  # noqa: BLE001
            return InfoHubResult(error=f"{type(exc).__name__}: {exc}", mode=self.mode)

    def collect(
        self,
        queries: Sequence[str],
        *,
        limit_per_query: int = 20,
    ) -> InfoHubResult:
        cleaned = [query.strip() for query in queries if query.strip()]
        if not self.enabled or not cleaned:
            return InfoHubResult(error="disabled or empty query", mode=self.mode)
        joined = "\n".join(cleaned)
        platforms = ",".join(self.collect_platforms)
        try:
            if self.mode == "local":
                Store, search_external, ingest = self._local_modules()
                posts, errors = search_external(
                    joined,
                    platforms,
                    self.collect_days,
                    max(1, int(limit_per_query)),
                )
                store = Store()
                try:
                    new_count = int(ingest(store, posts))
                finally:
                    store.close()
                raw_posts = [post.to_row() for post in posts]
                if errors:
                    logger.warning("InfoHub collect partial failures: %s", errors)
            else:
                preview = self._http_json(
                    "/api/preview",
                    payload={
                        "query": joined,
                        "platforms": platforms,
                        "days": self.collect_days,
                        "limit": max(1, int(limit_per_query)),
                    },
                )
                posts = [
                    card.get("post")
                    for card in preview.get("cards", [])
                    if isinstance(card, dict) and isinstance(card.get("post"), dict)
                ]
                ingest = self._http_json("/api/ingest", payload={"posts": posts})
                new_count = int(ingest.get("new", 0) or 0)
                raw_posts = posts
                errors = preview.get("errs", [])
                if errors:
                    logger.warning("InfoHub collect partial failures: %s", errors)
            papers = [
                paper
                for row in raw_posts
                if isinstance(row, Mapping)
                if (paper := _paper_from_row(row)) is not None
            ]
            return InfoHubResult(
                papers=tuple(deduplicate_papers(papers)),
                available=True,
                new_items=new_count,
                total_items=len(raw_posts),
                mode=self.mode,
            )
        except Exception as exc:  # noqa: BLE001
            return InfoHubResult(error=f"{type(exc).__name__}: {exc}", mode=self.mode)

    def ingest_papers(self, papers: Sequence[Paper], *, keyword: str = "") -> InfoHubResult:
        """Upsert papers found by any backend into the durable InfoHub library."""

        if not self.enabled or not papers:
            return InfoHubResult(error="disabled or empty papers", mode=self.mode)
        normalized_papers = deduplicate_papers(papers)
        posts: list[dict[str, Any]] = []
        row_papers: list[Paper] = []
        now = int(time.time())
        for paper in normalized_papers:
            platform = "arxiv" if paper.arxiv_id else "scholar"
            post_id = (
                paper.arxiv_id
                or paper.doi
                or paper.paper_id
                or hashlib.sha256(paper.title.lower().encode()).hexdigest()[:20]
            )
            publish_ts = 0
            if paper.year:
                publish_ts = int(
                    datetime(paper.year, 1, 1, tzinfo=UTC).timestamp()
                )
            row_papers.append(paper)
            posts.append(
                {
                    "platform": platform,
                    "post_id": post_id,
                    "title": paper.title,
                    "snippet": paper.abstract[:2000],
                    "url": paper.url,
                    "author": " and ".join(author.name for author in paper.authors),
                    "author_id": "",
                    "publish_ts": publish_ts,
                    "fetched_ts": now,
                    "like": 0,
                    "comment": 0,
                    "collect": 0,
                    "view": int(paper.citation_count),
                    "keyword": keyword or "researchclaw",
                    "raw": json.dumps(
                        {
                            "doi": paper.doi,
                            "arxiv_id": paper.arxiv_id,
                            "venue": paper.venue,
                            "citation_count": paper.citation_count,
                            "source": paper.source,
                        },
                        ensure_ascii=False,
                    ),
                    "abstract": paper.abstract or None,
                    "labels": None,
                    "summary": None,
                    "embedding": None,
                    "dup_of": None,
                }
            )
        try:
            if self.mode == "local":
                Store, _, ingest = self._local_modules()
                from infohub.model import Post  # type: ignore[import-not-found]

                store = Store()
                try:
                    new_count = int(ingest(store, [Post.from_row(row) for row in posts]))
                    # ``upsert`` owns collection fields; snapshot normally owns
                    # abstract.  Preserve known academic abstracts explicitly.
                    for paper, row in zip(row_papers, posts):
                        if paper.abstract:
                            store.set_abstract(
                                str(row["platform"]),
                                str(row["post_id"]),
                                paper.abstract,
                            )
                finally:
                    store.close()
            else:
                response = self._http_json("/api/ingest", payload={"posts": posts})
                new_count = int(response.get("new", 0) or 0)
                # The public ingest endpoint stores collection fields and the
                # abstract as snippet.  InfoHub's FTS/search UI still sees the
                # content, while direct-local mode additionally fills the
                # dedicated abstract column.
            return InfoHubResult(
                papers=tuple(normalized_papers),
                available=True,
                new_items=new_count,
                total_items=len(posts),
                mode=self.mode,
            )
        except Exception as exc:  # noqa: BLE001
            return InfoHubResult(error=f"{type(exc).__name__}: {exc}", mode=self.mode)


def query_persistent_memory(
    config: Any,
    queries: Sequence[str],
    *,
    limit_per_query: int | None = None,
) -> InfoHubResult:
    """Search InfoHub for all queries and return one deduplicated union."""

    client = InfoHubClient.from_config(config)
    all_papers: list[Paper] = []
    available = False
    errors: list[str] = []
    for query in queries:
        result = client.search(query, limit=limit_per_query)
        available = available or result.available
        all_papers.extend(result.papers)
        if result.error:
            errors.append(f"{query}: {result.error}")
    return InfoHubResult(
        papers=tuple(deduplicate_papers(all_papers)),
        available=available,
        total_items=len(all_papers),
        error="; ".join(errors),
        mode=client.mode,
    )
