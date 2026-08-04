from __future__ import annotations

import json
from pathlib import Path

from researchclaw.pipeline.event_log import EventLog, EventType, create_event


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

