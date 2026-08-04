from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from researchclaw.pipeline.stage_cache import (
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
    assert not (restored_dir / "stale.txt").exists()
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

