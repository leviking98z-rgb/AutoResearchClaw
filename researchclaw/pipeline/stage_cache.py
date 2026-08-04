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
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchclaw.pipeline.contracts import CONTRACTS
from researchclaw.pipeline.stages import Stage, StageStatus

logger = logging.getLogger(__name__)

# Schema 2 is the first version whose entries are published only after the
# output contract, PRM, approval gate, and HITL review all accept the stage.
# Reject schema-1 entries because older builds could save pre-verdict output.
CACHE_SCHEMA_VERSION = 2
CACHEABLE_STAGES = frozenset(
    {
        Stage.LITERATURE_COLLECT,
        Stage.LITERATURE_SCREEN,
        Stage.KNOWLEDGE_EXTRACT,
        Stage.SYNTHESIS,
        Stage.HYPOTHESIS_GEN,
    }
)
LITERATURE_FRESHNESS_STAGES = frozenset(
    {
        Stage.LITERATURE_COLLECT,
        Stage.HYPOTHESIS_GEN,
    }
)

_AUXILIARY_INPUTS: dict[Stage, tuple[str, ...]] = {
    Stage.LITERATURE_COLLECT: ("queries.json",),
    Stage.KNOWLEDGE_EXTRACT: ("web_context.md",),
    Stage.HYPOTHESIS_GEN: ("candidates.jsonl", "hitl_guidance.md"),
}

_CONFIG_FIELDS: dict[Stage, tuple[str, ...]] = {
    Stage.LITERATURE_COLLECT: (
        "research.topic",
        "research.daily_paper_count",
        "literature_search",
        "web_search",
        "llm.provider",
        "llm.base_url",
        "llm.wire_api",
        "llm.primary_model",
        "llm.fallback_models",
        "llm.timeout_sec",
        "llm.roles.literature_researcher",
        "prompts",
        "project.profile",
        "skills",
    ),
    Stage.LITERATURE_SCREEN: (
        "research.topic",
        "research.domains",
        "research.quality_threshold",
        "llm.provider",
        "llm.primary_model",
        "llm.fallback_models",
        "llm.roles.literature_researcher",
        "prompts",
        "project.profile",
        "skills",
    ),
    Stage.KNOWLEDGE_EXTRACT: (
        "research.topic",
        "llm.provider",
        "llm.primary_model",
        "llm.fallback_models",
        "llm.roles.literature_researcher",
        "prompts",
        "project.profile",
        "skills",
    ),
    Stage.SYNTHESIS: (
        "research.topic",
        "llm.provider",
        "llm.primary_model",
        "llm.fallback_models",
        "llm.roles.idea_scientist",
        "prompts",
        "project.profile",
        "skills",
    ),
    Stage.HYPOTHESIS_GEN: (
        "research.topic",
        "literature_search",
        "llm.provider",
        "llm.primary_model",
        "llm.fallback_models",
        "llm.roles.idea_scientist",
        "prompts",
        "project.profile",
        "skills",
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


def _path_content_view(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        return value
    try:
        return {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _hash_file(path),
        }
    except OSError:
        return value


def _prompt_config_view(config: Any) -> dict[str, Any]:
    prompts = getattr(config, "prompts", None)
    if prompts is None:
        return {}
    custom_file = getattr(prompts, "custom_file", "")
    raw_extra = getattr(prompts, "extra_prompts", ())
    if isinstance(raw_extra, Mapping):
        extra_items = raw_extra.items()
    else:
        extra_items = raw_extra or ()
    extra: list[dict[str, Any]] = []
    for item in extra_items:
        try:
            name, value = item
        except (TypeError, ValueError):
            continue
        extra.append(
            {
                "stage": str(name),
                "value": _path_content_view(str(value)),
            }
        )
    return {
        "custom_file": _path_content_view(str(custom_file)),
        "extra_prompts": sorted(extra, key=lambda row: row["stage"]),
    }


def _find_input(
    run_dir: Path,
    input_name: str,
    *,
    current_stage: Stage,
) -> Path | None:
    def _stage_sort_key(path: Path) -> tuple[str, int]:
        name = path.name
        if "_v" in name:
            base, _, version = name.rpartition("_v")
            try:
                return (base, -int(version))
            except ValueError:
                return (name, -999)
        return (name, 0)

    target = input_name.rstrip("/")
    allow_current_stage = input_name == "hitl_guidance.md"
    for stage_dir in sorted(
        run_dir.glob("stage-*"),
        key=_stage_sort_key,
        reverse=True,
    ):
        match = re.match(r"stage-(\d+)", stage_dir.name)
        if match is None:
            continue
        stage_number = int(match.group(1))
        if stage_number > int(current_stage):
            continue
        if stage_number == int(current_stage) and not allow_current_stage:
            continue
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
    for auxiliary in _AUXILIARY_INPUTS.get(stage, ()):
        if auxiliary not in input_names:
            input_names.append(auxiliary)
    inputs: list[dict[str, Any]] = []
    for input_name in input_names:
        path = _find_input(run_dir, input_name, current_stage=stage)
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
    config_view["prompts_content"] = _prompt_config_view(config)
    config_view["environment"] = {
        "ARC_ABL_DISABLE_DEBATE": os.environ.get("ARC_ABL_DISABLE_DEBATE", ""),
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


def _entry_age_sec(entry: Path, manifest: Mapping[str, Any]) -> float:
    created_at = str(manifest.get("created_at", "") or "").strip()
    if created_at:
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            return max(0.0, (datetime.now(UTC) - created).total_seconds())
        except ValueError:
            pass
    try:
        return max(0.0, datetime.now(UTC).timestamp() - entry.stat().st_mtime)
    except OSError:
        return float("inf")


def _freshness_ttl_hours(stage: Stage, config: Any) -> float:
    if stage not in LITERATURE_FRESHNESS_STAGES:
        return 0.0
    raw = getattr(
        getattr(config, "runtime", None),
        "stage_cache_literature_ttl_hours",
        24.0,
    )
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 24.0


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
    ttl_hours = _freshness_ttl_hours(stage, config)
    age_sec = _entry_age_sec(entry, manifest)
    if ttl_hours > 0 and age_sec > ttl_hours * 3600:
        logger.info(
            "Expired stage cache entry ignored: stage=%s age=%.1fh ttl=%.1fh",
            stage.name,
            age_sec / 3600,
            ttl_hours,
        )
        return None
    payload = entry / "payload"
    stage_dir.mkdir(parents=True, exist_ok=True)
    restored = {str(record["path"]) for record in manifest["files"]}
    for relative in sorted(restored):
        existing = stage_dir / relative
        if existing.is_file() or existing.is_symlink():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)
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
        "artifacts": manifest.get("artifacts", []),
        "files": len(manifest["files"]),
        "age_sec": round(age_sec, 3),
        "ttl_hours": ttl_hours,
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
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "stage": int(stage),
            "stage_name": stage.name,
            "fingerprint": fingerprint,
            "fingerprint_components": components,
            "run_id": run_id,
            "source_stage_dir": str(stage_dir),
            "artifacts": sorted(
                {
                    str(artifact).rstrip("/")
                    for artifact in artifacts
                    if str(artifact).strip()
                }
            ),
            "files": [copied[key] for key in sorted(copied)],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        replaced = False
        if entry.exists():
            existing_manifest: dict[str, Any] | None = None
            try:
                loaded = json.loads(
                    (entry / "manifest.json").read_text(encoding="utf-8")
                )
                if isinstance(loaded, dict):
                    existing_manifest = loaded
            except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
                pass
            ttl_hours = _freshness_ttl_hours(stage, config)
            existing_valid = (
                existing_manifest is not None
                and _manifest_valid(entry, existing_manifest)
            )
            existing_fresh = (
                existing_valid
                and (
                    ttl_hours <= 0
                    or _entry_age_sec(entry, existing_manifest)
                    <= ttl_hours * 3600
                )
            )
            if existing_fresh:
                shutil.rmtree(temporary, ignore_errors=True)
                return {
                    "saved": False,
                    "fingerprint": fingerprint,
                    "cache_entry": str(entry),
                    "reason": "already_exists",
                }
            backup = parent / f".{fingerprint}.stale-{os.getpid()}"
            shutil.rmtree(backup, ignore_errors=True)
            try:
                entry.replace(backup)
                temporary.replace(entry)
                replaced = True
            finally:
                shutil.rmtree(backup, ignore_errors=True)
        else:
            try:
                temporary.replace(entry)
            except FileExistsError:
                shutil.rmtree(temporary, ignore_errors=True)
                return {
                    "saved": False,
                    "fingerprint": fingerprint,
                    "cache_entry": str(entry),
                    "reason": "concurrent_writer",
                }
        return {
            "saved": True,
            "fingerprint": fingerprint,
            "cache_entry": str(entry),
            "files": len(copied),
            "replaced": replaced,
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def cached_result(stage: Stage, cache_details: Mapping[str, Any] | None = None) -> Any:
    """Build a StageResult lazily to avoid an executor import cycle."""

    from researchclaw.pipeline._helpers import StageResult

    contract = CONTRACTS[stage]
    artifacts: tuple[str, ...] = tuple(contract.output_files)
    if cache_details is not None:
        manifest_path = Path(str(cache_details.get("cache_entry", ""))) / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
            if isinstance(raw_artifacts, list) and raw_artifacts:
                artifacts = tuple(str(item) for item in raw_artifacts if str(item))
        except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
            pass
    return StageResult(
        stage=stage,
        status=StageStatus.DONE,
        artifacts=artifacts,
        decision="cache_hit",
        evidence_refs=tuple(f"stage-{int(stage):02d}/{item}" for item in artifacts),
    )
