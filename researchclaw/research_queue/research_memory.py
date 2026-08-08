"""Best-effort InfoHub projection for the lightweight Research Queue."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import ResearchMemoryConfig
from .models import IdeaRecord
from .store import ResearchQueueStore


@dataclass(frozen=True, slots=True)
class ResearchMemoryResult:
    ok: bool
    external_id: str
    error: str = ""


class ResearchMemory(Protocol):
    def reconcile(self, idea: IdeaRecord) -> ResearchMemoryResult: ...


class NullResearchMemory:
    def reconcile(self, idea: IdeaRecord) -> ResearchMemoryResult:
        return ResearchMemoryResult(ok=True, external_id=idea.idea_id)


class InfoHubResearchMemory:
    def __init__(
        self,
        *,
        config: ResearchMemoryConfig,
        system_id: str,
        store: ResearchQueueStore,
    ) -> None:
        self.config = config
        self.system_id = system_id
        self.store = store

    def reconcile(self, idea: IdeaRecord) -> ResearchMemoryResult:
        external_id = f"research-queue:{self.system_id}:{idea.idea_id}"
        if not self.config.enabled:
            return ResearchMemoryResult(ok=True, external_id=external_id)
        payload = self._payload(idea, external_id=external_id)
        request = urllib.request.Request(
            self.config.url.rstrip("/") + "/api/research-note/upsert",
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
                    if isinstance(parsed, Mapping) and parsed.get("ok") is False:
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
        idea_dir = self.store.idea_dir(idea.idea_id)
        note = _read_text(idea_dir / "research_note.md")
        final_review = _read_json(idea_dir / "final_review.json")
        benchmark_result = _read_json(idea_dir / "benchmark" / "result.json")
        manifest = _manifest(idea_dir)
        metadata = {
            "schema": "research_queue.research_memory",
            "version": 1,
            "system_id": self.system_id,
            "idea": idea.to_dict(),
            "research_spec": (
                idea.research_spec.to_dict()
                if idea.research_spec is not None
                else {}
            ),
            "final_review": final_review,
            "benchmark_result": benchmark_result,
            "artifacts": manifest,
            "source_of_truth": {
                "database": str(self.store.db_path),
                "idea_dir": str(idea_dir),
            },
        }
        tags = list(
            dict.fromkeys(
                [
                    "AutoResearch",
                    "ResearchQueue",
                    idea.status.value,
                    idea.conclusion.value if idea.conclusion else "",
                    *idea.tags,
                ]
            )
        )
        return {
            "external_id": external_id,
            "title": idea.title,
            "content": note or f"# {idea.title}\n",
            "kind": "research",
            "source": "research-queue",
            "status": idea.status.value,
            "tags": [item for item in tags if item],
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


def _manifest(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    values: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        relative = path.relative_to(root).as_posix()
        values.append(
            {
                "path": relative,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "kind": _artifact_kind(relative),
            }
        )
    return values


def _artifact_kind(path: str) -> str:
    name = Path(path).name
    if name == "research_note.md":
        return "note"
    if name in {"research_spec.json", "benchmark-config.yaml"}:
        return "plan"
    if name in {"final_review.json", "result.json"}:
        return "evidence"
    if Path(path).suffix == ".py":
        return "code"
    return "artifact"
