"""Structured, append-only pipeline event log.

Human-readable ``pipeline.log`` remains useful for operators.  This JSONL log
is optimized for retrospective analysis: stage latency, cache effectiveness,
failure clusters, source outages, and resume behavior can be aggregated without
scraping console text.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    STAGE_FAIL = "stage_fail"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_STORE = "cache_store"
    LITERATURE_MEMORY = "literature_memory"
    RESUME = "resume"


class EventLog:
    """Durable JSONL journal with fsync for crash-safe retrospection."""

    def __init__(self, log_dir: Path, filename: str = "pipeline_events.jsonl") -> None:
        self.path = Path(log_dir) / filename
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return event


def create_event(event_type: EventType | str, **payload: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "type": event_type.value if isinstance(event_type, EventType) else str(event_type),
        **payload,
    }
