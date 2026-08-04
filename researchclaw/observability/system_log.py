"""Unified operational event journal for cross-layer retrospectives."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from researchclaw.factory.io import append_jsonl, utc_now_ms

_CORRELATION_ENV = {
    "factory_id": "RESEARCHCLAW_FACTORY_ID",
    "idea_id": "RESEARCHCLAW_IDEA_ID",
    "work_item_id": "RESEARCHCLAW_WORK_ITEM_ID",
    "attempt": "RESEARCHCLAW_WORK_ITEM_ATTEMPT",
    "run_id": "RESEARCHCLAW_RUN_ID",
}


def correlation_from_env() -> dict[str, Any]:
    """Return stable correlation identifiers propagated to worker processes."""

    result: dict[str, Any] = {}
    for field, environment_name in _CORRELATION_ENV.items():
        value = os.environ.get(environment_name, "").strip()
        if not value:
            continue
        if field == "attempt":
            try:
                result[field] = int(value)
            except ValueError:
                result[field] = value
        else:
            result[field] = value
    return result


class OperationalEventLogger:
    """Append-only JSONL logger with explicit component/event semantics."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        component: str,
        durable: bool = False,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).expanduser() if path else None
        self.component = str(component).strip() or "unknown"
        self.durable = durable
        self.context = {
            **correlation_from_env(),
            **dict(context or {}),
        }

    def bind(self, **context: Any) -> OperationalEventLogger:
        """Return a child logger sharing the journal and adding context."""

        return OperationalEventLogger(
            self.path,
            component=self.component,
            durable=self.durable,
            context={**self.context, **context},
        )

    def emit(
        self,
        event: str,
        *,
        level: str = "INFO",
        outcome: str = "",
        reason_code: str = "",
        **payload: Any,
    ) -> dict[str, Any]:
        record = {
            "schema_version": 1,
            "timestamp": utc_now_ms(),
            "level": str(level).upper(),
            "component": self.component,
            "event": str(event),
            **self.context,
            **payload,
        }
        if outcome:
            record["outcome"] = str(outcome)
        if reason_code:
            record["reason_code"] = str(reason_code)
        if self.path is None:
            return record
        return append_jsonl(
            self.path,
            record,
            durable=self.durable,
        )
