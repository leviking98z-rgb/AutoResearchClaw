from __future__ import annotations

import json
from pathlib import Path

from researchclaw.pipeline.event_log import EventLog, EventType, create_event
from researchclaw.pipeline.runner import _write_observability_summary


def test_pipeline_event_log_is_append_only_jsonl(tmp_path: Path) -> None:
    log = EventLog(tmp_path)
    log.append(create_event(EventType.STAGE_START, run_id="r1", stage="SYNTHESIS"))
    log.append(
        create_event(
            EventType.STAGE_END,
            run_id="r1",
            stage="SYNTHESIS",
            elapsed_sec=1.2,
        )
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "pipeline_events.jsonl").read_text().splitlines()
    ]
    assert [row["type"] for row in rows] == ["stage_start", "stage_end"]
    assert all("timestamp" in row for row in rows)


def test_observability_summary_aggregates_pipeline_and_role_audits(
    tmp_path: Path,
) -> None:
    log = EventLog(tmp_path)
    log.append(
        create_event(
            EventType.STAGE_END,
            run_id="r1",
            stage="SYNTHESIS",
            elapsed_sec=2.5,
        )
    )
    log.append(
        create_event(
            EventType.CACHE_HIT,
            run_id="r1",
            stage="SYNTHESIS",
        )
    )
    log.append(
        create_event(
            EventType.LITERATURE_MEMORY,
            run_id="r1",
            stage="LITERATURE_COLLECT",
            infohub={
                "memory_hits": 12,
                "collected": 2,
                "new_items": 1,
                "persisted": 3,
            },
            total_candidates=14,
        )
    )
    audit = tmp_path / "audit" / "llm-idea_scientist.jsonl"
    audit.parent.mkdir()
    audit.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "status": "ok",
                        "elapsed_sec": 1.25,
                        "response_model": "model-a",
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "retries": 1,
                        "fallback_count": 1,
                        "truncated": False,
                    }
                ),
                json.dumps(
                    {
                        "status": "error",
                        "elapsed_sec": 0.75,
                        "requested_model": "model-a",
                        "retries": 0,
                        "fallback_count": 0,
                        "truncated": True,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = _write_observability_summary(tmp_path)

    assert summary["pipeline"]["stage_count"] == 1
    assert summary["pipeline"]["cache_hits"] == 1
    assert summary["pipeline"]["literature_memory_events"] == 1
    assert summary["literature"]["memory_hits"] == 12
    assert summary["literature"]["new_items"] == 1
    assert summary["llm"]["calls"] == 2
    assert summary["llm"]["errors"] == 1
    assert summary["llm"]["total_tokens"] == 15
    assert summary["llm"]["roles"]["idea_scientist"]["models"] == {
        "model-a": 2
    }
    assert (tmp_path / "observability_summary.json").is_file()
