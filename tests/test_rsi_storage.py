from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.rsi.storage import (
    CampaignStore,
    EventLog,
    atomic_write_json,
    atomic_write_text,
    cleanup_atomic_temp_files,
)


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"generation": 1, "payload": "old"})
    atomic_write_json(target, {"generation": 2, "payload": ["new"]})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "generation": 2,
        "payload": ["new"],
    }
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_atomic_write_preserves_old_state_if_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    atomic_write_json(target, {"generation": 1})

    def fail_replace(_source: object, _target: object) -> None:
        raise OSError("simulated crash before rename")

    monkeypatch.setattr("researchclaw.rsi.storage.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated crash"):
        atomic_write_json(target, {"generation": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": 1}
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_atomic_write_text_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    atomic_write_text(target, "generation: 1\n")
    atomic_write_text(target, "generation: 2\npayload: complete\n")

    assert target.read_text(encoding="utf-8") == (
        "generation: 2\npayload: complete\n"
    )
    assert not list(tmp_path.glob(".config.yaml.*.tmp"))


def test_cleanup_atomic_temp_files_removes_only_hidden_tmp_artifacts(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "runs" / "cycle-0001"
    nested.mkdir(parents=True)
    stale = nested / ".config.provenance.json.deadbeef.tmp"
    stale.write_text("partial", encoding="utf-8")
    legitimate = nested / "results.tmp"
    legitimate.write_text("keep", encoding="utf-8")

    removed = cleanup_atomic_temp_files(tmp_path)

    assert removed == [stale]
    assert not stale.exists()
    assert legitimate.read_text(encoding="utf-8") == "keep"


def test_cleanup_atomic_temp_files_skips_deep_generated_run_trees(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "cycle-0001"
    deep = run_dir / "stage-10" / "generated" / "vendor"
    deep.mkdir(parents=True)
    top_level_stale = run_dir / ".state.deadbeef.tmp"
    top_level_stale.write_text("partial", encoding="utf-8")
    deep_tmp = deep / ".cache.deadbeef.tmp"
    deep_tmp.write_text("generated tree content", encoding="utf-8")

    removed = cleanup_atomic_temp_files(tmp_path)

    assert removed == [top_level_stale]
    assert not top_level_stale.exists()
    assert deep_tmp.read_text(encoding="utf-8") == "generated tree content"


def test_event_log_is_append_only_jsonl(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    log.append("campaign_created", campaign_id="campaign-1")
    log.append("cycle_started", campaign_id="campaign-1", cycle=1, run="one")

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["type"] for line in lines] == [
        "campaign_created",
        "cycle_started",
    ]
    assert log.read_all()[1]["run"] == "one"


def test_campaign_store_pause_compatibility_and_heartbeat(
    tmp_path: Path,
) -> None:
    store = CampaignStore(tmp_path / "campaign")
    store.initialize()
    store.set_control("pause", "operator review")

    assert store.control_requested("pause")
    assert (store.root / "pause.request.json").is_file()

    store.write_heartbeat(
        {
            "campaign_id": "campaign",
            "supervisor_pid": 123,
            "phase": "pipeline",
        }
    )
    assert json.loads(store.heartbeat_path.read_text())["phase"] == "pipeline"
    assert json.loads(store.supervisor_heartbeat_path.read_text())["phase"] == (
        "pipeline"
    )

    store.clear_control("pause")
    assert not store.control_requested("pause")
    assert not (store.root / "pause.request.json").exists()
