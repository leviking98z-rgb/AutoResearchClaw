from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from researchclaw.pipeline.stage_cache import (
    CACHE_SCHEMA_VERSION,
    cached_result,
    restore_stage_cache,
    save_stage_cache,
    stage_input_fingerprint,
)
from researchclaw.pipeline.stages import Stage


def _config(cache_dir: Path, *, topic: str = "RSI") -> SimpleNamespace:
    return SimpleNamespace(
        runtime=SimpleNamespace(
            stage_cache_enabled=True,
            stage_cache_dir=str(cache_dir),
            stage_cache_literature_ttl_hours=24,
        ),
        research=SimpleNamespace(
            topic=topic,
            daily_paper_count=20,
            domains=("ml",),
            quality_threshold=8.0,
        ),
        literature_search=SimpleNamespace(sources=("arxiv",)),
        web_search=SimpleNamespace(enabled=False),
        llm=SimpleNamespace(
            roles={
                "literature_researcher": SimpleNamespace(model="fast"),
                "research_synthesizer": SimpleNamespace(model="strong"),
                "idea_scientist": SimpleNamespace(model="strong"),
            }
        ),
        prompts=SimpleNamespace(custom_file="", extra_prompts=()),
    )


def _stage3(run_dir: Path, query: str = "self improvement") -> None:
    stage = run_dir / "stage-03"
    stage.mkdir(parents=True)
    (stage / "search_plan.yaml").write_text("strategy: academic\n")
    (stage / "queries.json").write_text(json.dumps({"queries": [query]}))


def test_stage_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "cycle-1"
    second = tmp_path / "cycle-2"
    _stage3(first)
    _stage3(second)
    stage4 = first / "stage-04"
    stage4.mkdir()
    (stage4 / "candidates.jsonl").write_text('{"title":"paper"}\n')
    (stage4 / "references.bib").write_text("@article{x}\n")
    config = _config(cache_dir)

    saved = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=first,
        run_id="first",
        config=config,
        artifacts=("candidates.jsonl", "references.bib"),
    )
    restored_dir = second / "stage-04"
    restored_dir.mkdir()
    (restored_dir / "stale.txt").write_text("remove me")
    hit = restore_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=restored_dir,
        run_dir=second,
        config=config,
    )

    assert saved is not None
    assert hit is not None
    assert (restored_dir / "candidates.jsonl").read_text() == '{"title":"paper"}\n'
    assert (restored_dir / "stale.txt").read_text() == "remove me"
    assert json.loads((restored_dir / "cache_restore.json").read_text())["hit"]


def test_input_change_invalidates_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first = tmp_path / "cycle-1"
    second = tmp_path / "cycle-2"
    _stage3(first, "query one")
    _stage3(second, "query two")
    stage4 = first / "stage-04"
    stage4.mkdir()
    (stage4 / "candidates.jsonl").write_text("{}\n")
    config = _config(cache_dir)
    save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=first,
        run_id="first",
        config=config,
        artifacts=("candidates.jsonl",),
    )

    assert (
        restore_stage_cache(
            stage=Stage.LITERATURE_COLLECT,
            stage_dir=second / "stage-04",
            run_dir=second,
            config=config,
        )
        is None
    )


def test_config_change_changes_fingerprint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    first, _ = stage_input_fingerprint(
        stage=Stage.LITERATURE_COLLECT,
        run_dir=run_dir,
        config=_config(tmp_path / "cache", topic="topic one"),
    )
    second, _ = stage_input_fingerprint(
        stage=Stage.LITERATURE_COLLECT,
        run_dir=run_dir,
        config=_config(tmp_path / "cache", topic="topic two"),
    )

    assert first != second


def test_prompt_file_content_change_invalidates_fingerprint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("version one")
    config = _config(tmp_path / "cache")
    config.prompts.custom_file = str(prompt_file)
    first, _ = stage_input_fingerprint(
        stage=Stage.LITERATURE_COLLECT,
        run_dir=run_dir,
        config=config,
    )
    prompt_file.write_text("version two")
    second, _ = stage_input_fingerprint(
        stage=Stage.LITERATURE_COLLECT,
        run_dir=run_dir,
        config=config,
    )

    assert first != second


def test_hypothesis_fingerprint_includes_candidates_and_guidance(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    stage4 = run_dir / "stage-04"
    stage4.mkdir(parents=True)
    candidates = stage4 / "candidates.jsonl"
    candidates.write_text('{"title":"one"}\n')
    stage7 = run_dir / "stage-07"
    stage7.mkdir()
    (stage7 / "synthesis.md").write_text("synthesis")
    stage8 = run_dir / "stage-08"
    stage8.mkdir()
    guidance = stage8 / "hitl_guidance.md"
    guidance.write_text("prefer robust hypotheses")
    config = _config(tmp_path / "cache")

    first, _ = stage_input_fingerprint(
        stage=Stage.HYPOTHESIS_GEN,
        run_dir=run_dir,
        config=config,
    )
    candidates.write_text('{"title":"two"}\n')
    second, _ = stage_input_fingerprint(
        stage=Stage.HYPOTHESIS_GEN,
        run_dir=run_dir,
        config=config,
    )
    guidance.write_text("prefer efficient hypotheses")
    third, _ = stage_input_fingerprint(
        stage=Stage.HYPOTHESIS_GEN,
        run_dir=run_dir,
        config=config,
    )

    assert first != second
    assert second != third


def test_fingerprint_prefers_canonical_stage_over_versioned_copy(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    canonical = run_dir / "stage-07"
    canonical.mkdir(parents=True)
    (canonical / "synthesis.md").write_text("canonical")
    versioned = run_dir / "stage-07_v1"
    versioned.mkdir()
    (versioned / "synthesis.md").write_text("old version")
    fingerprint, components = stage_input_fingerprint(
        stage=Stage.HYPOTHESIS_GEN,
        run_dir=run_dir,
        config=_config(tmp_path / "cache"),
    )

    assert fingerprint
    synthesis = next(
        item for item in components["inputs"] if item["name"] == "synthesis.md"
    )
    assert synthesis["files"][0]["sha256"] == (
        hashlib.sha256(b"canonical").hexdigest()
    )


def test_corrupt_payload_is_cache_miss(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()
    (stage4 / "candidates.jsonl").write_text("{}\n")
    config = _config(cache_dir)
    saved = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="first",
        config=config,
        artifacts=("candidates.jsonl",),
    )
    assert saved is not None
    entry = Path(saved["cache_entry"])
    (entry / "payload" / "candidates.jsonl").write_text("corrupt")

    assert (
        restore_stage_cache(
            stage=Stage.LITERATURE_COLLECT,
            stage_dir=tmp_path / "other" / "stage-04",
            run_dir=run_dir,
            config=config,
        )
        is None
    )


def test_expired_literature_cache_is_miss(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()
    (stage4 / "candidates.jsonl").write_text("{}\n")
    config = _config(cache_dir)
    saved = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="first",
        config=config,
        artifacts=("candidates.jsonl",),
    )
    assert saved is not None
    manifest_path = Path(saved["cache_entry"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = (
        datetime.now(UTC) - timedelta(hours=25)
    ).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest))

    assert (
        restore_stage_cache(
            stage=Stage.LITERATURE_COLLECT,
            stage_dir=tmp_path / "other" / "stage-04",
            run_dir=run_dir,
            config=config,
        )
        is None
    )


def test_expired_literature_cache_can_be_rebuilt(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()
    output = stage4 / "candidates.jsonl"
    output.write_text('{"version":1}\n')
    config = _config(cache_dir)
    first = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="first",
        config=config,
        artifacts=("candidates.jsonl",),
    )
    assert first is not None
    manifest_path = Path(first["cache_entry"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = (
        datetime.now(UTC) - timedelta(hours=25)
    ).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest))
    output.write_text('{"version":2}\n')

    second = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="second",
        config=config,
        artifacts=("candidates.jsonl",),
    )

    assert second is not None
    assert second["saved"] is True
    assert second["replaced"] is True
    restored = tmp_path / "restored" / "stage-04"
    hit = restore_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=restored,
        run_dir=run_dir,
        config=config,
    )
    assert hit is not None
    assert (restored / "candidates.jsonl").read_text() == '{"version":2}\n'


def test_old_cache_schema_is_rejected_and_rebuilt(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()
    output = stage4 / "candidates.jsonl"
    output.write_text('{"new":true}\n')
    config = _config(cache_dir)
    saved = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="first",
        config=config,
        artifacts=("candidates.jsonl",),
    )
    assert saved is not None
    manifest_path = Path(saved["cache_entry"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = CACHE_SCHEMA_VERSION - 1
    manifest_path.write_text(json.dumps(manifest))

    assert restore_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=tmp_path / "old" / "stage-04",
        run_dir=run_dir,
        config=config,
    ) is None
    rebuilt = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="rebuilt",
        config=config,
        artifacts=("candidates.jsonl",),
    )
    assert rebuilt is not None
    assert rebuilt["saved"] is True
    assert rebuilt["replaced"] is True
    assert json.loads(manifest_path.read_text())["schema_version"] == (
        CACHE_SCHEMA_VERSION
    )


def test_cached_result_uses_manifest_artifact_list(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()
    (stage4 / "candidates.jsonl").write_text("{}\n")
    (stage4 / "references.bib").write_text("@article{x}\n")
    saved = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="first",
        config=_config(cache_dir),
        artifacts=("candidates.jsonl", "references.bib"),
    )
    assert saved is not None

    result = cached_result(Stage.LITERATURE_COLLECT, saved)

    assert result.artifacts == ("candidates.jsonl", "references.bib")


def test_zero_ttl_disables_literature_cache_expiry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    run_dir = tmp_path / "run"
    _stage3(run_dir)
    stage4 = run_dir / "stage-04"
    stage4.mkdir()
    (stage4 / "candidates.jsonl").write_text("{}\n")
    config = _config(cache_dir)
    config.runtime.stage_cache_literature_ttl_hours = 0
    saved = save_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=stage4,
        run_dir=run_dir,
        run_id="first",
        config=config,
        artifacts=("candidates.jsonl",),
    )
    assert saved is not None
    manifest_path = Path(saved["cache_entry"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = (
        datetime.now(UTC) - timedelta(days=365)
    ).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest))

    hit = restore_stage_cache(
        stage=Stage.LITERATURE_COLLECT,
        stage_dir=tmp_path / "other" / "stage-04",
        run_dir=run_dir,
        config=config,
    )
    assert hit is not None
    assert hit["ttl_hours"] == 0
