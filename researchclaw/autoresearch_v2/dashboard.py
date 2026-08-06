"""Operational dashboard and controls for AutoResearch v2."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import V2Config
from .models import IdeaStatus
from .store import V2Store
from .usage import UsageMonitor


class ControlRequest(BaseModel):
    reason: str = Field(default="", max_length=500)


def configured_gpu_total(config: V2Config) -> int:
    """Return the GPU capacity represented by the active backend config."""
    if not config.gpu.enabled:
        return 0
    if config.gpu.mode == "resource_manager":
        return max(0, int(config.gpu.resource_manager.desired_gpus))
    if not config.gpu.pool_config:
        return 0
    try:
        from researchclaw.factory.pool_config import PoolConfigSummary

        return max(
            0,
            int(
                PoolConfigSummary.from_file(
                    config.gpu.pool_config
                ).expected_total_gpus
            ),
        )
    except Exception:  # noqa: BLE001
        return 0


class V2Dashboard:
    def __init__(
        self,
        store: V2Store,
        *,
        gpu_total: int = 0,
        target_utilization: float = 0.9,
        control_enabled: bool = True,
        usage_monitor: UsageMonitor | None = None,
        usage_cache_ttl_sec: float = 10.0,
    ) -> None:
        self.store = store
        self.gpu_total = max(0, int(gpu_total))
        self.target_utilization = float(target_utilization)
        self.control_enabled = control_enabled
        self.usage_monitor = usage_monitor
        self.usage_cache_ttl_sec = max(
            0.0,
            float(usage_cache_ttl_sec),
        )
        self._usage_cache: dict[
            tuple[int | None, int | None],
            tuple[datetime, dict[str, Any]],
        ] = {}
        self._usage_lock = threading.Lock()

    def collect(self) -> dict[str, Any]:
        ideas = self.store.list_ideas()
        jobs = self.store.list_jobs()
        attempts = self.store.list_attempts()
        jobs_by_idea: dict[str, list[dict[str, Any]]] = {}
        attempts_by_idea: dict[str, list[dict[str, Any]]] = {}
        for job in jobs:
            jobs_by_idea.setdefault(job.idea_id, []).append(job.to_dict())
        for attempt in attempts:
            attempts_by_idea.setdefault(
                attempt.idea_id, []
            ).append(attempt.to_dict())
        lanes = {
            "reservoir": [],
            "design": [],
            "build": [],
            "pilot": [],
            "scale": [],
            "report": [],
            "completed": [],
            "rejected": [],
        }
        lane_for = {
            IdeaStatus.RESERVOIR: "reservoir",
            IdeaStatus.NEW: "design",
            IdeaStatus.DESIGNING: "design",
            IdeaStatus.BUILDING: "build",
            IdeaStatus.PILOTING: "pilot",
            IdeaStatus.SCALING: "scale",
            IdeaStatus.REPORTING: "report",
            IdeaStatus.RETRYABLE: "build",
            IdeaStatus.COMPLETED: "completed",
            IdeaStatus.COMPLETED_NEGATIVE: "completed",
            IdeaStatus.REJECTED: "rejected",
            IdeaStatus.QUARANTINED: "rejected",
        }
        for idea in ideas:
            record = idea.to_dict()
            record["jobs"] = jobs_by_idea.get(idea.idea_id, [])
            record["attempts"] = attempts_by_idea.get(idea.idea_id, [])
            lanes[lane_for[idea.status]].append(record)
        for values in lanes.values():
            values.sort(
                key=lambda value: (
                    -float(value.get("priority", 0)),
                    str(value["idea_id"]),
                )
            )
        running_gpu = [
            job
            for job in jobs
            if job.status.value == "running" and job.requires_gpu
        ]
        allocated = sum(
            int(job.result.get("allocated_gpus", 0) or 0)
            for job in running_gpu
        )
        ideas_by_status: dict[str, int] = {}
        jobs_by_status: dict[str, int] = {}
        for idea in ideas:
            ideas_by_status[idea.status.value] = (
                ideas_by_status.get(idea.status.value, 0) + 1
            )
        for job in jobs:
            jobs_by_status[job.status.value] = (
                jobs_by_status.get(job.status.value, 0) + 1
            )
        last_event = self.store.list_events(limit=1)
        writer = self.store.writer_status()
        if self.store.control_requested("stop"):
            controller_status = "stopping"
        elif self.store.control_requested("pause"):
            controller_status = "paused"
        elif writer["state"] == "live":
            controller_status = "running"
        else:
            controller_status = "stopped"
        return {
            "controller": {
                "timestamp": (
                    last_event[-1]["timestamp"]
                    if last_event
                    else ""
                ),
                "status": controller_status,
                "process": writer,
                "ideas_total": len(ideas),
                "ideas_by_status": ideas_by_status,
                "jobs_total": len(jobs),
                "jobs_by_status": jobs_by_status,
                "llm_tokens_total": sum(
                    idea.llm_tokens_spent for idea in ideas
                ),
                "gpu_hours_total": sum(
                    idea.gpu_seconds_spent for idea in ideas
                )
                / 3600.0,
            },
            "lanes": lanes,
            "gpu": {
                "total": self.gpu_total,
                "allocated": allocated,
                "utilization": (
                    allocated / self.gpu_total
                    if self.gpu_total
                    else 0.0
                ),
                "target_utilization": self.target_utilization,
                "jobs": [job.to_dict() for job in running_gpu],
            },
            "controls": {
                "enabled": self.control_enabled,
                "can_pause": self.control_enabled
                and writer["state"] == "live"
                and not self.store.control_requested("pause"),
                "can_resume": self.control_enabled
                and writer["state"] == "live"
                and self.store.control_requested("pause"),
                "can_stop": self.control_enabled
                and writer["state"] == "live",
            },
        }

    def usage(
        self,
        *,
        hours: int | None = None,
        bucket_minutes: int | None = None,
    ) -> dict[str, Any]:
        if self.usage_monitor is None:
            return {
                "enabled": False,
                "generated_at": datetime.now(UTC).isoformat(
                    timespec="milliseconds"
                ),
            }
        key = (hours, bucket_minutes)
        now = datetime.now(UTC)
        with self._usage_lock:
            cached = self._usage_cache.get(key)
            if (
                cached is not None
                and (now - cached[0]).total_seconds()
                < self.usage_cache_ttl_sec
            ):
                return cached[1]
            payload = {
                "enabled": True,
                **self.usage_monitor.collect(
                    hours=hours,
                    bucket_minutes=bucket_minutes,
                ),
            }
            self._usage_cache[key] = (now, payload)
            return payload

    def health(self, stale_after_sec: float = 120.0) -> dict[str, Any]:
        events = self.store.list_events(limit=500)
        ticks = [
            event
            for event in events
            if event.get("event_type") == "controller_tick"
        ]
        age_sec: float | None = None
        if ticks:
            try:
                observed = datetime.fromisoformat(ticks[-1]["timestamp"])
                if observed.tzinfo is None:
                    observed = observed.replace(tzinfo=UTC)
                age_sec = max(
                    0.0,
                    (datetime.now(UTC) - observed).total_seconds(),
                )
            except (TypeError, ValueError):
                pass
        stale = age_sec is None or age_sec > stale_after_sec
        writer = self.store.writer_status()
        reasons = ["controller_tick_stale"] if stale else []
        if writer["state"] == "stale":
            reasons.append("controller_lock_stale")
        return {
            "status": "degraded" if reasons else "ok",
            "tick_age_sec": age_sec,
            "stale_after_sec": stale_after_sec,
            "reasons": reasons,
            "writer_lock_present": writer["state"] != "missing",
            "writer_lock_state": writer["state"],
            "writer_pid": writer["pid"],
        }

    def log(
        self,
        idea_id: str,
        attempt_id: str,
        stream: str,
        limit: int,
    ) -> str:
        if stream not in {"stdout", "stderr", "attempt"}:
            raise ValueError("invalid stream")
        attempt = self.store.get_attempt(attempt_id)
        if attempt is None or attempt.idea_id != idea_id:
            raise ValueError("attempt not found")
        path = (
            self.store.attempt_dir(attempt) / f"{stream}.log"
            if stream in {"stdout", "stderr"}
            else self.store.attempt_dir(attempt) / "attempt.json"
        )
        try:
            lines = path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except FileNotFoundError:
            lines = []
        return "\n".join(lines[-max(1, min(limit, 5000)) :])


def create_dashboard_app(
    config: V2Config,
    *,
    control_enabled: bool = True,
) -> FastAPI:
    store = V2Store(
        config.root,
        db_path=config.database_path,
        db_backup_path=config.database_backup_path,
        backup_interval_sec=config.storage.backup_interval_sec,
    )
    # Dashboard is a live read-only companion to the controller. It must not
    # perform crash-recovery cleanup while controller worker threads are in
    # the middle of an atomic current-directory swap.
    store.initialize(recover_filesystem=False)
    gpu_total = configured_gpu_total(config)
    usage_monitor = (
        UsageMonitor(
            store=store,
            budgets=config.budgets,
            config=config.usage_monitoring,
            gpu_total=gpu_total,
        )
        if config.usage_monitoring.enabled
        else None
    )
    dashboard = V2Dashboard(
        store,
        gpu_total=gpu_total,
        target_utilization=config.gpu.target_utilization,
        control_enabled=control_enabled,
        usage_monitor=usage_monitor,
    )
    app = FastAPI(title="AutoResearch v2")
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health(stale_after_sec: float = 120.0) -> dict[str, Any]:
        return dashboard.health(
            max(1.0, min(stale_after_sec, 86_400.0))
        )

    @app.get("/api/dashboard")
    def status() -> dict[str, Any]:
        return dashboard.collect()

    @app.get("/api/usage")
    def usage(
        hours: int | None = None,
        bucket_minutes: int | None = None,
    ) -> dict[str, Any]:
        return dashboard.usage(
            hours=hours,
            bucket_minutes=bucket_minutes,
        )

    @app.get("/api/events")
    def events(
        limit: int = 200,
        after_seq: int = 0,
    ) -> dict[str, Any]:
        return {
            "events": store.list_events(
                limit=limit,
                after_seq=after_seq,
            )
        }

    @app.get("/api/ideas/{idea_id}")
    def idea(idea_id: str) -> dict[str, Any]:
        record = store.get_idea(idea_id)
        if record is None:
            raise HTTPException(status_code=404, detail="idea not found")
        return {
            "idea": record.to_dict(),
            "jobs": [
                job.to_dict()
                for job in store.list_jobs(idea_id=idea_id)
            ],
            "attempts": [
                attempt.to_dict()
                for attempt in store.list_attempts(idea_id=idea_id)
            ],
            "events": store.list_events(
                idea_id=idea_id,
                limit=500,
            ),
        }

    @app.get("/api/ideas/{idea_id}/logs")
    def logs(
        idea_id: str,
        attempt_id: str,
        stream: str = "stdout",
        limit: int = 500,
    ) -> PlainTextResponse:
        try:
            text = dashboard.log(
                idea_id,
                attempt_id,
                stream,
                limit,
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
        path = store.set_control(
            "pause",
            request.reason or "dashboard pause",
        )
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
        path = store.set_control(
            "stop",
            request.reason or "dashboard stop",
        )
        return {"accepted": True, "path": str(path)}

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoresearch-v2-dashboard")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8120)
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args(argv)
    config = V2Config.from_file(args.config)
    import uvicorn

    uvicorn.run(
        create_dashboard_app(
            config,
            control_enabled=not args.no_control,
        ),
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
