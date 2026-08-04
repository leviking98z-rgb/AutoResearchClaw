"""Read-only/control dashboard for the durable multi-Idea Factory."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import FactoryConfig
from .io import tail_jsonl, tail_text_lines
from .models import IdeaStatus, LeaseStatus
from .observability import build_factory_observability
from .store import FactoryStore


class ControlRequest(BaseModel):
    reason: str = Field(default="", max_length=500)


class FactoryDashboard:
    def __init__(
        self,
        store: FactoryStore,
        *,
        control_enabled: bool = True,
    ) -> None:
        self.store = store
        self.control_enabled = control_enabled

    def collect(self) -> dict[str, Any]:
        snapshot = self.store.snapshot()
        ideas = self.store.list_ideas()
        reservoir = self.store.load_reservoir()
        items = self.store.list_work_items()
        leases = self.store.list_leases()
        item_counts: dict[str, dict[str, int]] = {}
        for item in items:
            current = item_counts.setdefault(item.idea_id, {})
            current[item.status.value] = current.get(item.status.value, 0) + 1
        lanes = {
            "reservoir": [idea.to_dict() for idea in reservoir],
            "screening": [],
            "build": [],
            "pilot": [],
            "validation": [],
            "paper": [],
            "completed": [],
            "rejected": [],
        }
        lane_by_status = {
            IdeaStatus.SCREENING: "screening",
            IdeaStatus.BUILDING: "build",
            IdeaStatus.SMOKE: "pilot",
            IdeaStatus.PILOT: "pilot",
            IdeaStatus.REPAIR: "pilot",
            IdeaStatus.VALIDATING: "validation",
            IdeaStatus.PAPER: "paper",
            IdeaStatus.COMPLETED: "completed",
            IdeaStatus.COMPLETED_NEGATIVE: "completed",
            IdeaStatus.PARKED: "rejected",
            IdeaStatus.REJECTED: "rejected",
            IdeaStatus.FAILED: "rejected",
        }
        for idea in ideas:
            record = idea.to_dict()
            record["work_items"] = item_counts.get(idea.idea_id, {})
            lane = lane_by_status.get(idea.status, "rejected")
            lanes[lane].append(record)
        for values in lanes.values():
            values.sort(key=lambda value: (-value.get("priority", 0), value["idea_id"]))
        return {
            "factory": snapshot,
            "observability": build_factory_observability(self.store),
            "lanes": lanes,
            "gpu": {
                "allocated": sum(
                    lease.allocated_gpus
                    for lease in leases
                    if lease.status
                    in {LeaseStatus.ADMITTED, LeaseStatus.RUNNING}
                ),
                "leases": [lease.to_dict() for lease in leases],
            },
            "controls": {
                "enabled": self.control_enabled,
                "can_pause": self.control_enabled
                and not self.store.control_requested("pause"),
                "can_resume": self.control_enabled
                and self.store.control_requested("pause"),
                "can_stop": self.control_enabled,
            },
        }

    @staticmethod
    def _timestamp_age_sec(value: object) -> float | None:
        try:
            observed = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - observed).total_seconds())

    def health(self, *, stale_after_sec: float = 120.0) -> dict[str, Any]:
        """Return an operational health verdict, not only HTTP liveness."""

        state = self.store.load_state()
        age_sec = self._timestamp_age_sec(state.get("updated_at"))
        running_items = [
            item
            for item in self.store.list_work_items()
            if item.status.value in {"admitted", "running"}
        ]
        active_leases = [
            lease
            for lease in self.store.list_leases()
            if lease.status in {LeaseStatus.ADMITTED, LeaseStatus.RUNNING}
        ]
        stale = (
            str(state.get("status", "")).casefold() == "running"
            and (age_sec is None or age_sec > stale_after_sec)
        )
        reasons: list[str] = []
        if stale:
            reasons.append("factory_tick_stale")
        status = "degraded" if reasons else "ok"
        return {
            "status": status,
            "factory_status": str(state.get("status", "unknown")),
            "tick": int(state.get("tick", 0) or 0),
            "tick_age_sec": round(age_sec, 3) if age_sec is not None else None,
            "stale_after_sec": float(stale_after_sec),
            "running_work_items": len(running_items),
            "active_leases": len(active_leases),
            "reasons": reasons,
        }

    def events(self, limit: int = 200) -> list[dict[str, Any]]:
        return tail_jsonl(
            self.store.events_path,
            limit=max(1, min(limit, 2000)),
        )

    def idea_events(
        self,
        idea_id: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if not idea_id or any(
            token in idea_id for token in ("/", "\\", "..")
        ):
            raise ValueError("invalid idea_id")
        path = self.store.idea_dir(idea_id) / "events.jsonl"
        return tail_jsonl(path, limit=max(1, min(limit, 2000)))

    def idea_log(
        self,
        idea_id: str,
        source: str,
        *,
        item_id: str = "",
        attempt: int = 1,
        limit: int = 400,
    ) -> str:
        if not idea_id or any(
            token in idea_id for token in ("/", "\\", "..")
        ):
            raise ValueError("invalid idea_id")
        allowed = {
            "pipeline": (
                self.store.idea_dir(idea_id)
                / "runs"
                / "pipeline"
                / "pipeline.log"
            ),
            "pipeline_events": (
                self.store.idea_dir(idea_id)
                / "runs"
                / "pipeline"
                / "pipeline_events.jsonl"
            ),
            "operational_events": (
                self.store.idea_dir(idea_id)
                / "operational_events.jsonl"
            ),
        }
        if source in {"worker_stdout", "worker_stderr"}:
            if not item_id or any(
                token in item_id for token in ("/", "\\", "..")
            ):
                raise ValueError("invalid item_id")
            suffix = "stdout.log" if source == "worker_stdout" else "stderr.log"
            path = (
                self.store.idea_dir(idea_id)
                / "workers"
                / item_id
                / f"attempt-{max(1, int(attempt)):02d}"
                / suffix
            )
        else:
            path = allowed.get(source)
            if path is None:
                raise ValueError("invalid log source")
        return "\n".join(
            tail_text_lines(
                path,
                limit=max(1, min(limit, 5000)),
            )
        )


def create_dashboard_app(
    factory_root: str | Path,
    *,
    factory_id: str = "research-factory",
    control_enabled: bool = True,
) -> FastAPI:
    store = FactoryStore(factory_root, factory_id=factory_id)
    store.initialize()
    dashboard = FactoryDashboard(store, control_enabled=control_enabled)
    app = FastAPI(title="ResearchClaw Factory")
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health(stale_after_sec: float = 120.0) -> dict[str, Any]:
        return dashboard.health(
            stale_after_sec=max(1.0, min(stale_after_sec, 86_400.0))
        )

    @app.get("/api/dashboard")
    def status() -> dict[str, Any]:
        return dashboard.collect()

    @app.get("/api/events")
    def events(limit: int = 200) -> dict[str, Any]:
        return {"events": dashboard.events(limit)}

    @app.get("/api/ideas/{idea_id}/events")
    def idea_events(idea_id: str, limit: int = 200) -> dict[str, Any]:
        try:
            events = dashboard.idea_events(idea_id, limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"idea_id": idea_id, "events": events}

    @app.get("/api/ideas/{idea_id}/logs")
    def idea_logs(
        idea_id: str,
        source: str = "pipeline",
        item_id: str = "",
        attempt: int = 1,
        limit: int = 400,
    ) -> PlainTextResponse:
        try:
            text = dashboard.idea_log(
                idea_id,
                source,
                item_id=item_id,
                attempt=attempt,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PlainTextResponse(text)

    def require_control() -> None:
        if not control_enabled:
            raise HTTPException(status_code=403, detail="controls disabled")

    @app.post("/api/control/pause")
    def pause(request: ControlRequest) -> dict[str, Any]:
        require_control()
        path = store.set_control("pause", request.reason or "dashboard pause")
        return {"accepted": True, "path": str(path)}

    @app.post("/api/control/resume")
    def resume(request: ControlRequest) -> dict[str, Any]:
        del request
        require_control()
        store.clear_control("pause")
        return {"accepted": True}

    @app.post("/api/control/stop")
    def stop(request: ControlRequest) -> dict[str, Any]:
        require_control()
        path = store.set_control("stop", request.reason or "dashboard stop")
        return {"accepted": True, "path": str(path)}

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="researchclaw-factory-dashboard")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args(argv)
    config = FactoryConfig.from_file(args.config)
    import uvicorn

    uvicorn.run(
        create_dashboard_app(
            config.root,
            factory_id=config.factory_id,
            control_enabled=not args.no_control,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
