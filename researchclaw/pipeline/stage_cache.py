"""Content-addressed cache for deterministic/expensive pipeline stages.

Each stage cache entry contains:

* a fingerprint of the stage's effective inputs and relevant configuration;
* a manifest of produced files with content hashes;
* a reusable payload stored outside individual run directories.

The cache is intentionally conservative: a missing file, hash mismatch,
software schema bump, prompt/config change, or upstream input change causes a
miss and normal execution.  Hits are copied into the current run atomically
enough for the sequential pipeline and are recorded in structured logs.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from researchclaw.pipeline.contracts import CONTRACTS
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = 1
CACHEABLE_STAGES = frozenset(
    {
        Stage.LITERATURE_COLLECT,
        Stage.LITERATURE_SCREEN,
        Stage.KNOWLEDGE_EXTRACT,
        Stage.SYNTHESIS,
        Stage.HYPOTHESIS_GEN,
    }
)

_CONFIG_FIELDS: dict[Stage, tuple[str, ...]] = {
    Stage.LITERATURE_COLLECT: (
        "research.topic",
        "research.daily_paper_count",
        "literature_search",
        "web_search",
        "llm.roles.literature_researcher",
        "prompts",
    ),
    Stage.LITERATURE_SCREEN: (
        "research.topic",
        "research.domains",
        "research.quality_threshold",
        "llm.roles.literature_researcher",
        "prompts",
    ),
    Stage.KNOWLEDGE_EXTRACT: (
        "research.topic",
        "llm.roles.literature_researcher",
        "prompts",
    ),
    Stage.SYNTHESIS: (
        "research.topic",
        "llm.roles.idea_scientist",
        "prompts",
    ),
    Stage.HYPOTHESIS_GEN: (
        "research.topic",
        "literature_search",
        "llm.roles.idea_scientist",
        "prompts",
    ),
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _get_path(root: Any, dotted: str) -> Any:
    current = root
    for part in dotted.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            break
    return _jsonable(current)


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and not child.name.endswith(".tmp"):
                yield child


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_input(run_dir: Path, input_name: str) -> Path | None:
    target = input_name.rstrip("/")
    for stage_dir in sorted(run_dir.glob("stage-*"), reverse=True):
        candidate = stage_dir / target
        if input_name.endswith("/") and candidate.is_dir():
            return candidate
        if not input_name.endswith("/") and candidate.is_file():
            return candidate
    return None


def stage_input_fingerprint(
    *,
    stage: Stage,
    run_dir: Path,
    config: Any,
) -> tuple[str, dict[str, Any]]:
    """Compute the stage fingerprint and its human-auditable components."""

    contract = CONTRACTS[stage]
    input_names = list(contract.input_files)
    # Some stage implementations consume auxiliary artifacts in addition to
    # the minimal contract file.  Fingerprint those effective inputs as well.
    if stage == Stage.LITERATURE_COLLECT and "queries.json" not in input_names:
        input_names.append("queries.json")
    inputs: list[dict[str, Any]] = []
    for input_name in input_names:
        path = _find_input(run_dir, input_name)
        if path is None:
            inputs.append({"name": input_name, "missing": True})
            continue
        records = []
        root = path if path.is_dir() else path.parent
        for file_path in _iter_files(path):
            records.append(
                {
                    "path": str(file_path.relative_to(root)),
                    "size": file_path.stat().st_size,
                    "sha256": _hash_file(file_path),
                }
            )
        inputs.append({"name": input_name, "files": records})
    config_view = {
        field: _get_path(config, field) for field in _CONFIG_FIELDS.get(stage, ())
    }
    try:
        package_version = importlib.metadata.version("researchclaw")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source"
    components = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "package_version": package_version,
        "stage": int(stage),
        "stage_name": stage.name,
        "inputs": inputs,
        "config": config_view,
    }
    encoded = json.dumps(
        components, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), components


def _cache_root(run_dir: Path, config: Any) -> Path | None:
    if not bool(
        getattr(getattr(config, "runtime", None), "stage_cache_enabled", True)
    ):
        return None
    configured = str(
        getattr(getattr(config, "runtime", None), "stage_cache_dir", "") or ""
    ).strip()
    if configured:
        return Path(configured).expanduser()
    raw = os.environ.get("RESEARCHCLAW_STAGE_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    # RSI campaigns place runs under ``<campaign>/runs/cycle-*``.
    if run_dir.parent.name == "runs":
        return run_dir.parent.parent / "shared" / "stage-cache"
    return run_dir / ".stage-cache"


def _manifest_valid(entry: Path, manifest: Mapping[str, Any]) -> bool:
    if int(manifest.get("schema_version", 0) or 0) != CACHE_SCHEMA_VERSION:
        return False
    payload = entry / "payload"
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return False
    for record in files:
        if not isinstance(record, Mapping):
            return False
        relative = str(record.get("path", "") or "")
        source = payload / relative
        if not relative or not source.is_file():
            return False
        if _hash_file(source) != record.get("sha256"):
            return False
    return True


def restore_stage_cache(
    *,
    stage: Stage,
    stage_dir: Path,
    run_dir: Path,
    config: Any,
) -> dict[str, Any] | None:
    """Restore a valid stage entry; return metadata on hit."""

    if stage not in CACHEABLE_STAGES:
        return None
    root = _cache_root(run_dir, config)
    if root is None:
        return None
    fingerprint, components = stage_input_fingerprint(
        stage=stage, run_dir=run_dir, config=config
    )
    entry = root / f"stage-{int(stage):02d}" / fingerprint
    try:
        manifest = json.loads((entry / "manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return None
    if not isinstance(manifest, dict) or not _manifest_valid(entry, manifest):
        logger.warning("Invalid stage cache entry ignored: %s", entry)
        return None
    payload = entry / "payload"
    stage_dir.mkdir(parents=True, exist_ok=True)
    restored = {str(record["path"]) for record in manifest["files"]}
    for existing in sorted(stage_dir.rglob("*"), reverse=True):
        if existing.is_file():
            relative = str(existing.relative_to(stage_dir))
            if relative not in restored:
                existing.unlink()
        elif existing.is_dir():
            try:
                existing.rmdir()
            except OSError:
                pass
    for record in manifest["files"]:
        relative = str(record["path"])
        source = payload / relative
        destination = stage_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    hit = {
        "hit": True,
        "stage": int(stage),
        "fingerprint": fingerprint,
        "cache_entry": str(entry),
        "source_run_id": manifest.get("run_id"),
        "source_stage_dir": manifest.get("source_stage_dir"),
        "files": len(manifest["files"]),
        "fingerprint_components": components,
    }
    (stage_dir / "cache_restore.json").write_text(
        json.dumps(hit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return hit


def save_stage_cache(
    *,
    stage: Stage,
    stage_dir: Path,
    run_dir: Path,
    run_id: str,
    config: Any,
    artifacts: Iterable[str],
) -> dict[str, Any] | None:
    """Publish stage outputs into the shared content-addressed cache."""

    if stage not in CACHEABLE_STAGES:
        return None
    root = _cache_root(run_dir, config)
    if root is None:
        return None
    fingerprint, components = stage_input_fingerprint(
        stage=stage, run_dir=run_dir, config=config
    )
    parent = root / f"stage-{int(stage):02d}"
    entry = parent / fingerprint
    if (entry / "manifest.json").is_file():
        return {
            "saved": False,
            "fingerprint": fingerprint,
            "cache_entry": str(entry),
            "reason": "already_exists",
        }
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=parent))
    payload = temporary / "payload"
    payload.mkdir()
    copied: dict[str, dict[str, Any]] = {}
    try:
        for artifact in artifacts:
            source = stage_dir / artifact.rstrip("/")
            if not source.exists():
                continue
            for file_path in _iter_files(source):
                relative = str(file_path.relative_to(stage_dir))
                if relative in {"decision.json", "stage_health.json", "cache_restore.json"}:
                    continue
                destination = payload / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, destination)
                copied[relative] = {
                    "path": relative,
                    "size": file_path.stat().st_size,
                    "sha256": _hash_file(file_path),
                }
        if not copied:
            shutil.rmtree(temporary, ignore_errors=True)
            return None
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "stage": int(stage),
            "stage_name": stage.name,
            "fingerprint": fingerprint,
            "fingerprint_components": components,
            "run_id": run_id,
            "source_stage_dir": str(stage_dir),
            "files": [copied[key] for key in sorted(copied)],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        try:
            temporary.replace(entry)
        except FileExistsError:
            shutil.rmtree(temporary, ignore_errors=True)
        return {
            "saved": True,
            "fingerprint": fingerprint,
            "cache_entry": str(entry),
            "files": len(copied),
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def cached_result(stage: Stage) -> Any:
    """Build a StageResult lazily to avoid an executor import cycle."""

    from researchclaw.pipeline._helpers import StageResult

    contract = CONTRACTS[stage]
    artifacts = tuple(contract.output_files)
    return StageResult(
        stage=stage,
        status=StageStatus.DONE,
        artifacts=artifacts,
        decision="cache_hit",
        evidence_refs=tuple(f"stage-{int(stage):02d}/{item}" for item in artifacts),
    )
