"""Best-effort InfoHub projection of AutoResearch's durable scientific state.

SQLite and immutable attempt directories remain the source of truth. This
adapter only creates a searchable, human-readable Research Note in InfoHub.
Failures are returned to the Controller for audit and never block research.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .attestation import build_tree_manifest
from .config import ResearchMemoryConfig
from .models import IdeaRecord
from .store import V2Store

_MAX_EVENTS = 80
_MAX_MANIFEST_FILES = 500


@dataclass(frozen=True, slots=True)
class ResearchMemoryResult:
    ok: bool
    external_id: str
    error: str = ""


class ResearchMemory(Protocol):
    def reconcile(self, idea: IdeaRecord) -> ResearchMemoryResult: ...


class NullResearchMemory:
    def reconcile(self, idea: IdeaRecord) -> ResearchMemoryResult:
        return ResearchMemoryResult(
            ok=True,
            external_id=idea.idea_id,
        )


class InfoHubResearchMemory:
    """Project one Idea into one idempotently updated InfoHub Research Note."""

    def __init__(
        self,
        *,
        config: ResearchMemoryConfig,
        system_id: str,
        store: V2Store,
    ) -> None:
        self.config = config
        self.system_id = str(system_id).strip() or "autoresearch-v2"
        self.store = store

    def reconcile(self, idea: IdeaRecord) -> ResearchMemoryResult:
        external_id = (
            f"autoresearch-v2:{self.system_id}:{idea.idea_id}"
        )
        if not self.config.enabled:
            return ResearchMemoryResult(ok=True, external_id=external_id)
        payload = self._payload(idea, external_id=external_id)
        request = urllib.request.Request(
            self.config.url.rstrip("/")
            + "/api/research-note/upsert",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_sec,
            ) as response:
                body = response.read().decode("utf-8", errors="replace")
                if not 200 <= int(response.status) < 300:
                    return ResearchMemoryResult(
                        ok=False,
                        external_id=external_id,
                        error=f"HTTP {response.status}: {body[:500]}",
                    )
                if body.strip():
                    parsed = json.loads(body)
                    if (
                        isinstance(parsed, Mapping)
                        and parsed.get("ok") is False
                    ):
                        return ResearchMemoryResult(
                            ok=False,
                            external_id=external_id,
                            error=str(parsed.get("error", "upsert failed")),
                        )
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            return ResearchMemoryResult(
                ok=False,
                external_id=external_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        return ResearchMemoryResult(ok=True, external_id=external_id)

    def _payload(
        self,
        idea: IdeaRecord,
        *,
        external_id: str,
    ) -> dict[str, Any]:
        current = self.store.current_dir(idea.idea_id)
        plan = _read_json(current / "plan.json")
        pilot_metrics = _read_json(
            current / "artifacts" / "pilot" / "metrics.json"
        )
        scale_metrics = _read_json(
            current / "artifacts" / "scale" / "metrics.json"
        )
        final_review = _read_json(current / "final_review.json")
        paper = _read_text(current / "paper.md")
        events = self.store.list_events(
            idea_id=idea.idea_id,
            limit=_MAX_EVENTS,
        )
        manifest = _artifact_manifest(current)
        metadata = {
            "schema": "autoresearch_v2.research_memory",
            "version": 1,
            "system_id": self.system_id,
            "idea_id": idea.idea_id,
            "family": idea.family,
            "score": idea.score,
            "priority": idea.priority,
            "status": idea.status.value,
            "final_outcome": idea.candidate.get("final_outcome", ""),
            "exit_reason": idea.exit_reason,
            "cost": {
                "llm_tokens": idea.llm_tokens_spent,
                "llm_calls": idea.llm_calls,
                "gpu_seconds": idea.gpu_seconds_spent,
            },
            "metrics": {
                "pilot": pilot_metrics,
                "scale": scale_metrics,
            },
            "plan": plan,
            "final_review": final_review,
            "artifacts": manifest,
            "source_of_truth": {
                "database": str(self.store.db_path),
                "current_dir": str(current),
            },
            "timestamps": {
                "created_at": idea.created_at,
                "updated_at": idea.updated_at,
                "last_progress_at": idea.last_progress_at,
            },
        }
        tags = _tags(idea)
        return {
            "external_id": external_id,
            "title": idea.title,
            "content": _markdown(
                idea,
                plan=plan,
                pilot_metrics=pilot_metrics,
                scale_metrics=scale_metrics,
                final_review=final_review,
                paper=paper,
                events=events,
                manifest=manifest,
            ),
            "kind": "research",
            "source": "autoresearch-v2",
            "status": idea.status.value,
            "tags": tags,
            "metadata": metadata,
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _artifact_manifest(current: Path) -> list[dict[str, Any]]:
    if not current.is_dir():
        return []
    try:
        manifest = build_tree_manifest(
            current,
            exclude=(
                "__pycache__",
                ".pytest_cache",
                "artifacts/pilot/_migration_backup",
                "artifacts/scale/_migration_backup",
            ),
        )
    except (OSError, ValueError):
        return []
    files = manifest.get("files", [])
    if not isinstance(files, list):
        return []
    result: list[dict[str, Any]] = []
    for item in files[:_MAX_MANIFEST_FILES]:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path", "") or "")
        result.append(
            {
                "path": path,
                "sha256": str(item.get("sha256", "") or ""),
                "size": int(item.get("size_bytes", 0) or 0),
                "kind": _artifact_kind(path),
            }
        )
    return result


def _artifact_kind(path: str) -> str:
    name = Path(path).name
    if name == "paper.md":
        return "paper"
    if name == "plan.json":
        return "plan"
    if name in {"metrics.json", "runtime_evidence.json"}:
        return "evidence"
    if name.endswith("_review.json") or name == "report.json":
        return "review"
    if Path(path).suffix == ".py":
        return "code"
    return "artifact"


def _tags(idea: IdeaRecord) -> list[str]:
    values = [
        "RSI",
        idea.family,
        idea.status.value,
        str(idea.candidate.get("final_outcome", "") or ""),
    ]
    return list(
        dict.fromkeys(
            value.strip().replace(" ", "-")
            for value in values
            if value and value.strip()
        )
    )


def _markdown(
    idea: IdeaRecord,
    *,
    plan: Mapping[str, Any],
    pilot_metrics: Mapping[str, Any],
    scale_metrics: Mapping[str, Any],
    final_review: Mapping[str, Any],
    paper: str,
    events: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> str:
    sections = [
        f"# {idea.title}",
        "## Status",
        f"- State: `{idea.status.value}`",
        f"- Final outcome: `{idea.candidate.get('final_outcome', '')}`",
        f"- Exit reason: {idea.exit_reason or '—'}",
        f"- Family: `{idea.family}`",
        "",
        "## Research question",
        idea.research_question,
        "",
        "## Falsifiable hypothesis",
        idea.falsifiable_hypothesis,
        "",
        "## Literature context",
        _json_block(idea.candidate.get("literature_context", {})),
        "",
        "## Experimental plan",
        _json_block(plan),
        "",
        "## Current result",
        "### Pilot",
        _json_block(pilot_metrics),
        "### Scale",
        _json_block(scale_metrics),
        "",
        "## Decision history",
        _event_lines(events),
        "",
        "## Final review",
        _json_block(final_review),
        "",
        "## Paper",
        paper.strip() or "_Not generated yet._",
        "",
        "## Artifact manifest",
        _manifest_lines(manifest),
        "",
        "## Source of truth",
        (
            "This note is a searchable projection. AutoResearch SQLite and "
            "immutable attempt/current directories remain canonical."
        ),
    ]
    return "\n".join(sections).rstrip() + "\n"


def _json_block(value: Any) -> str:
    if not value:
        return "_Not available yet._"
    return (
        "```json\n"
        + json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```"
    )


def _event_lines(events: list[dict[str, Any]]) -> str:
    if not events:
        return "_No events recorded._"
    return "\n".join(
        (
            f"- `{event.get('timestamp', '')}` "
            f"**{event.get('event_type', '')}**"
            + (
                f" — {event.get('reason')}"
                if event.get("reason")
                else ""
            )
        )
        for event in events
    )


def _manifest_lines(manifest: list[dict[str, Any]]) -> str:
    if not manifest:
        return "_No accepted artifacts yet._"
    return "\n".join(
        (
            f"- `{item['path']}` — {item['kind']}, "
            f"{item['size']} bytes, sha256 `{item['sha256']}`"
        )
        for item in manifest
    )
