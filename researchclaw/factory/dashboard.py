"""Read-only/control dashboard for the durable multi-Idea Factory."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import FactoryConfig
from .models import IdeaStatus, LeaseStatus
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

    def events(self, limit: int = 200) -> list[dict[str, Any]]:
        try:
            lines = self.store.events_path.read_text(encoding="utf-8").splitlines()
        except (FileNotFoundError, OSError):
            return []
        events: list[dict[str, Any]] = []
        for line in lines[-max(1, min(limit, 2000)) :]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, Mapping):
                events.append(dict(value))
        return events


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
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/dashboard")
    def status() -> dict[str, Any]:
        return dashboard.collect()

    @app.get("/api/events")
    def events(limit: int = 200) -> dict[str, Any]:
        return {"events": dashboard.events(limit)}

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
