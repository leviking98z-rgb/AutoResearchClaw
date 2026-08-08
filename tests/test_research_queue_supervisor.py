from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from researchclaw.research_queue.supervisor import (
    AcceptanceSupervisor,
    SupervisorConfig,
    collect_canary_report,
)


def _write_state(root: Path, artifact_root: Path) -> None:
    root.mkdir(parents=True)
    artifact_root.mkdir(parents=True)
    conn = sqlite3.connect(root / "research_queue.db")
    conn.executescript(
        """
        CREATE TABLE ideas (
          idea_id TEXT PRIMARY KEY, status TEXT, priority REAL,
          updated_at TEXT, data_json TEXT
        );
        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY, idea_id TEXT, status TEXT,
          budget TEXT, revision INTEGER, updated_at TEXT, data_json TEXT
        );
        CREATE TABLE events (
          seq INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT,
          event_type TEXT, idea_id TEXT, run_id TEXT, payload_json TEXT
        );
        """
    )
    idea = {
        "idea_id": "idea-1",
        "status": "concluded",
        "conclusion": "negative",
    }
    run = {
        "run_id": "run-1",
        "idea_id": "idea-1",
        "status": "succeeded",
        "budget": "B0",
        "result": {"usage": {"budget_parameters": {"seeds": 3}}},
    }
    conn.execute(
        "INSERT INTO ideas VALUES (?,?,?,?,?)",
        ("idea-1", "concluded", 0.5, "now", json.dumps(idea)),
    )
    conn.execute(
        "INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
        ("run-1", "idea-1", "succeeded", "B0", 1, "now", json.dumps(run)),
    )
    events = [
        ("2026-01-01T00:00:00+00:00", "controller_started", {}),
        ("2026-01-01T00:00:01+00:00", "idea_generation_started", {}),
        ("2026-01-01T02:00:01+00:00", "controller_stopped", {}),
    ]
    for timestamp, event, payload in events:
        conn.execute(
            "INSERT INTO events(timestamp,event_type,payload_json) VALUES (?,?,?)",
            (timestamp, event, json.dumps(payload)),
        )
    conn.commit()
    conn.close()
    revision = artifact_root / "ideas" / "idea-1" / "revisions" / "revision-001"
    revision.mkdir(parents=True)
    (revision / "experiment.py").write_text("import json\nimport numpy\n")
    note = artifact_root / "ideas" / "idea-1" / "research_note.md"
    note.write_text("# note\n")
    audit = root / "llm-audit" / "worker"
    audit.mkdir(parents=True)
    (audit / "calls.jsonl").write_text(
        json.dumps(
            {
                "model": "test",
                "outcome": "success",
                "total_tokens": 10,
            }
        )
        + "\n"
    )


def test_collect_canary_report_enforces_deterministic_gates(
    tmp_path,
    monkeypatch,
) -> None:
    state = tmp_path / "state"
    artifacts = tmp_path / "artifacts"
    _write_state(state, artifacts)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "research_queue": {
                    "artifact_dir": str(artifacts),
                    "limits": {
                        "max_total_tokens": 100,
                        "generation_max_batches": 1,
                    },
                    "execution": {"allowed_python_imports": ["numpy"]},
                }
            }
        )
    )
    monkeypatch.setattr(
        "researchclaw.research_queue.supervisor._resource_status",
        lambda _path: {"snapshot": {"allocations": [], "queue": []}},
    )
    monkeypatch.setattr(
        "researchclaw.research_queue.supervisor._systemctl_properties",
        lambda _unit: {"ActiveState": "inactive", "Result": "success"},
    )
    report = collect_canary_report(
        state_dir=state,
        config_path=config,
        unit="test.service",
        cb_path=tmp_path / "cb",
    )
    assert report["passed"]
    assert report["audit_total_tokens"] == 10
    assert report["checks"]["all_succeeded_runs_attest_budget"]


def test_collect_canary_report_rejects_materialized_import(
    tmp_path,
    monkeypatch,
) -> None:
    state = tmp_path / "state"
    artifacts = tmp_path / "artifacts"
    _write_state(state, artifacts)
    experiment = next(artifacts.glob("ideas/*/revisions/*/experiment.py"))
    experiment.write_text("import sklearn\n")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "research_queue": {
                    "artifact_dir": str(artifacts),
                    "limits": {
                        "max_total_tokens": 100,
                        "generation_max_batches": 1,
                    },
                    "execution": {"allowed_python_imports": ["numpy"]},
                }
            }
        )
    )
    monkeypatch.setattr(
        "researchclaw.research_queue.supervisor._resource_status",
        lambda _path: {"snapshot": {"allocations": [], "queue": []}},
    )
    monkeypatch.setattr(
        "researchclaw.research_queue.supervisor._systemctl_properties",
        lambda _unit: {"ActiveState": "inactive", "Result": "success"},
    )
    report = collect_canary_report(
        state_dir=state,
        config_path=config,
        unit="test.service",
        cb_path=tmp_path / "cb",
    )
    assert not report["passed"]
    assert report["disallowed_imports"][0]["imports"] == ["sklearn"]


def test_resource_id_parser_accepts_clusterbridge_text(tmp_path) -> None:
    config = SupervisorConfig(
        state_dir=tmp_path / "state",
        canary_state_dir=tmp_path / "canary",
        canary_config=tmp_path / "canary.yaml",
        canary_unit="canary.service",
        canary_report=tmp_path / "report.json",
        benchmark_config=tmp_path / "benchmark.yaml",
        benchmark_output_dir=tmp_path / "benchmark-output",
        benchmark_remote_script=tmp_path / "run.sh",
    )
    supervisor = AcceptanceSupervisor(config)
    assert supervisor._resource_ids_from_output(
        "request req-1\nstatus: allocated\nallocation: alloc-2\n"
    ) == {"request_id": "req-1", "allocation_id": "alloc-2"}
    assert supervisor._resource_ids_from_output(
        "allocation alloc-3\nstatus: active\n"
    ) == {"allocation_id": "alloc-3"}
