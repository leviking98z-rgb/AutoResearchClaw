"""Unified local and ClusterBridge Run backends for B0/B1/B2."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from researchclaw.autoresearch_v2.config import (
    GPUConfig,
    GPUResourceManagerConfig,
)
from researchclaw.autoresearch_v2.elastic_gpu import (
    ResourceManagedGPUManager,
)
from researchclaw.autoresearch_v2.models import (
    JobKind,
)
from researchclaw.autoresearch_v2.models import (
    JobRecord as V2JobRecord,
)

from .config import ResearchQueueConfig
from .models import RunRecord, RunResult


class RunBackend(Protocol):
    async def run(
        self,
        run: RunRecord,
        *,
        revision_dir: Path,
        output_dir: Path,
        env: Mapping[str, str],
    ) -> RunResult: ...

    async def close(self) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class SlotLease:
    gpus: int


class GPUSlotPool:
    """Simple prototype GPU counter with condition-variable backfill."""

    def __init__(self, total_gpus: int) -> None:
        self.total_gpus = max(0, int(total_gpus))
        self._used = 0
        self._condition = asyncio.Condition()

    @property
    def used(self) -> int:
        return self._used

    @property
    def available(self) -> int:
        return max(0, self.total_gpus - self._used)

    async def acquire(self, gpus: int) -> SlotLease:
        requested = max(0, int(gpus))
        if requested > self.total_gpus:
            raise ValueError(
                f"run requests {requested} GPUs but limit is {self.total_gpus}"
            )
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._used + requested <= self.total_gpus
            )
            self._used += requested
        return SlotLease(requested)

    async def release(self, lease: SlotLease) -> None:
        async with self._condition:
            self._used = max(0, self._used - max(0, lease.gpus))
            self._condition.notify_all()


class LocalRunBackend:
    """Execute generated experiments as local subprocesses."""

    def __init__(self, *, slot_pool: GPUSlotPool) -> None:
        self.slot_pool = slot_pool
        self.running = 0

    async def run(
        self,
        run: RunRecord,
        *,
        revision_dir: Path,
        output_dir: Path,
        env: Mapping[str, str],
    ) -> RunResult:
        lease = await self.slot_pool.acquire(run.requested_gpus)
        self.running += 1
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        stdout_path = output_dir / "stdout.log"
        stderr_path = output_dir / "stderr.log"
        try:
            command = _resolve_command(run.command, revision_dir)
            process_env = {
                **os.environ,
                **{str(key): str(value) for key, value in env.items()},
                "CUDA_VISIBLE_DEVICES": _local_cuda_devices(run.requested_gpus),
            }
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=revision_dir,
                env=process_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=run.timeout_sec,
                )
            except TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
                stdout_path.write_bytes(stdout)
                stderr_path.write_bytes(stderr)
                return RunResult(
                    ok=False,
                    error=f"run timed out after {run.timeout_sec:.1f}s",
                    returncode=-1,
                    usage={
                        "gpu_count": run.requested_gpus,
                        "gpu_seconds": run.requested_gpus
                        * (time.monotonic() - started),
                    },
                )
            stdout_path.write_bytes(stdout)
            stderr_path.write_bytes(stderr)
            result = _load_result(output_dir)
            result.returncode = int(process.returncode or 0)
            result.ok = result.ok and process.returncode == 0
            if process.returncode != 0 and not result.error:
                result.error = (
                    f"experiment exited with return code {process.returncode}"
                )
            result.artifacts = sorted(
                set(
                    result.artifacts
                    + [
                        str(stdout_path),
                        str(stderr_path),
                    ]
                )
            )
            result.usage = {
                **result.usage,
                "gpu_count": run.requested_gpus,
                "gpu_seconds": run.requested_gpus * (time.monotonic() - started),
            }
            return result
        except (OSError, ValueError) as exc:
            return RunResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                returncode=-1,
                usage={
                    "gpu_count": run.requested_gpus,
                    "gpu_seconds": run.requested_gpus * (time.monotonic() - started),
                },
            )
        finally:
            self.running = max(0, self.running - 1)
            await self.slot_pool.release(lease)

    async def close(self) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "local",
            "running": self.running,
            "total_gpus": self.slot_pool.total_gpus,
            "used_gpus": self.slot_pool.used,
            "available_gpus": self.slot_pool.available,
        }


class ClusterBridgeRunBackend:
    """Small adapter over the existing elastic ClusterBridge implementation."""

    def __init__(
        self,
        *,
        manager: ResourceManagedGPUManager,
        slot_pool: GPUSlotPool,
        poll_interval_sec: float,
    ) -> None:
        self.manager = manager
        self.slot_pool = slot_pool
        self.poll_interval_sec = max(0.1, float(poll_interval_sec))
        self._active: dict[str, tuple[Any, int]] = {}
        self._broker_reconcile_lock = asyncio.Lock()
        self._completed_payloads: dict[str, dict[str, Any]] = {}
        self.manager.bootstrap(required_gpus=0)

    async def run(
        self,
        run: RunRecord,
        *,
        revision_dir: Path,
        output_dir: Path,
        env: Mapping[str, str],
    ) -> RunResult:
        lease = await self.slot_pool.acquire(run.requested_gpus)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            broker = await self._wait_for_broker(required_gpus=run.requested_gpus)
            command = _remote_command(
                run.command,
                revision_dir=revision_dir,
                output_dir=output_dir,
                env=env,
            )
            job = V2JobRecord(
                job_id=run.run_id,
                idea_id=run.idea_id,
                kind=JobKind.PILOT,
                requires_gpu=True,
                min_gpus=run.requested_gpus,
                preferred_gpus=run.requested_gpus,
                max_gpus=run.requested_gpus,
                timeout_sec=run.timeout_sec,
                command=command,
                expected_output_dir=str(output_dir),
                attempt_id=run.run_id,
            )
            decision = broker.submit(
                job,
                priorities={run.idea_id: 1.0},
            )
            if not decision.admitted:
                return RunResult(
                    ok=False,
                    error=f"GPU broker rejected run: {decision.reason}",
                    returncode=-1,
                )
            self._active[run.run_id] = (broker, run.requested_gpus)
            while True:
                payload = await self._take_completed_payload(
                    broker,
                    run_id=run.run_id,
                )
                if payload is not None:
                    result = _result_from_broker_payload(
                        payload,
                        output_dir=output_dir,
                    )
                    elapsed = time.monotonic() - started
                    result.usage = {
                        **result.usage,
                        "gpu_count": run.requested_gpus,
                        "gpu_seconds": run.requested_gpus * elapsed,
                    }
                    return result
                await asyncio.sleep(self.poll_interval_sec)
        except Exception as exc:  # noqa: BLE001
            return RunResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                returncode=-1,
                usage={
                    "gpu_count": run.requested_gpus,
                    "gpu_seconds": run.requested_gpus * (time.monotonic() - started),
                },
            )
        finally:
            self._active.pop(run.run_id, None)
            async with self._broker_reconcile_lock:
                self._completed_payloads.pop(run.run_id, None)
            await self.slot_pool.release(lease)
            self._reconcile_demand()

    async def _take_completed_payload(
        self,
        broker: Any,
        *,
        run_id: str,
    ) -> dict[str, Any] | None:
        """Harvest broker results once and route them to the owning Run."""

        async with self._broker_reconcile_lock:
            payload = self._completed_payloads.pop(run_id, None)
            if payload is not None:
                return payload
            for completed_id, completed_payload in broker.reconcile():
                self._completed_payloads[str(completed_id)] = dict(completed_payload)
            return self._completed_payloads.pop(run_id, None)

    async def _wait_for_broker(self, *, required_gpus: int) -> Any:
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            active_gpus = sum(value[1] for value in self._active.values())
            demand = min(
                self.slot_pool.total_gpus,
                max(required_gpus, active_gpus + required_gpus),
            )
            self.manager.reconcile(
                required_gpus=demand,
                pending_gpu_jobs=1,
                running_gpu_jobs=len(self._active),
                force=True,
            )
            broker = self.manager.broker
            if broker is not None:
                return broker
            await asyncio.sleep(self.poll_interval_sec)
        raise TimeoutError("timed out waiting for ClusterBridge allocation")

    def _reconcile_demand(self) -> None:
        active_gpus = sum(value[1] for value in self._active.values())
        self.manager.reconcile(
            required_gpus=active_gpus,
            pending_gpu_jobs=0,
            running_gpu_jobs=len(self._active),
            force=True,
        )

    async def close(self) -> None:
        self.manager.reconcile(
            required_gpus=0,
            pending_gpu_jobs=0,
            running_gpu_jobs=0,
            force=True,
        )
        await asyncio.sleep(0)
        self.manager.close()
        async with self._broker_reconcile_lock:
            self._completed_payloads.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "clusterbridge",
            "logical_total_gpus": self.slot_pool.total_gpus,
            "logical_used_gpus": self.slot_pool.used,
            "active_runs": len(self._active),
            "routed_results": len(self._completed_payloads),
            "resource_manager": self.manager.snapshot(),
        }


def build_run_backend(config: ResearchQueueConfig) -> RunBackend:
    slot_pool = GPUSlotPool(config.gpu.max_total_gpus)
    if config.execution.backend == "local":
        return LocalRunBackend(slot_pool=slot_pool)
    resource = dict(config.gpu.resource_manager)
    resource.setdefault("max_gpus", config.gpu.max_total_gpus)
    resource.setdefault(
        "project",
        "ResearchQueuePrototype",
    )
    resource.setdefault(
        "purpose",
        "Continuous Research Queue prototype runs",
    )
    resource.setdefault("release_on_shutdown", True)
    gpu_config = GPUConfig(
        enabled=True,
        mode="resource_manager",
        reserved_gpus=0,
        target_utilization=1.0,
        max_share_per_idea=1.0,
        pilot_max_gpus=config.gpu.max_gpus_per_run,
        scale_max_gpus=config.gpu.max_gpus_per_run,
        resource_manager=GPUResourceManagerConfig(
            owner=str(resource.get("owner", "") or ""),
            cb_command=str(
                resource.get(
                    "cb_command",
                    "/root/shared/.clusters/.tools/clusterbridge.sh",
                )
            ),
            project=str(resource["project"]),
            purpose=str(resource["purpose"]),
            max_gpus=int(resource["max_gpus"]),
            duration_min=int(resource.get("duration_min", 120)),
            renew_ttl_min=int(resource.get("renew_ttl_min", 120)),
            renew_interval_sec=float(resource.get("renew_interval_sec", 300.0)),
            reconcile_interval_sec=float(resource.get("reconcile_interval_sec", 2.0)),
            allow_cross_cluster=bool(resource.get("allow_cross_cluster", True)),
            gpu_type=str(resource.get("gpu_type", "") or ""),
            priority=str(resource.get("priority", "normal") or "normal"),
            release_on_shutdown=bool(resource.get("release_on_shutdown", True)),
            log_root=str(
                resource.get(
                    "log_root",
                    "/root/shared/.clusters/.tmp/research-queue-prototype",
                )
            ),
            ray_command=str(
                resource.get(
                    "ray_command",
                    "/opt/conda/envs/torch-base/bin/ray",
                )
            ),
            ray_python=str(
                resource.get(
                    "ray_python",
                    "/opt/conda/envs/torch-base/bin/python3",
                )
            ),
            ray_port=int(resource.get("ray_port", 6379)),
            command_timeout_sec=float(resource.get("command_timeout_sec", 180.0)),
            prepare_timeout_sec=float(resource.get("prepare_timeout_sec", 900.0)),
        ),
    )
    task_env = {
        name: value
        for name in config.gpu.pass_env
        if (value := os.environ.get(name)) is not None
    }
    manager = ResourceManagedGPUManager(
        gpu_config,
        task_env=task_env,
        task_namespace=config.system_id,
    )
    return ClusterBridgeRunBackend(
        manager=manager,
        slot_pool=slot_pool,
        poll_interval_sec=config.gpu.poll_interval_sec,
    )


def _resolve_command(
    command: tuple[str, ...],
    revision_dir: Path,
) -> tuple[str, ...]:
    if not command:
        raise ValueError("empty command")
    resolved = list(command)
    executable = shutil.which(resolved[0])
    if executable:
        resolved[0] = executable
    if len(resolved) > 1:
        candidate = revision_dir / resolved[1]
        if candidate.exists():
            resolved[1] = str(candidate)
    return tuple(resolved)


def _local_cuda_devices(count: int) -> str:
    if count <= 0:
        return ""
    return ",".join(str(index) for index in range(count))


def _load_result(output_dir: Path) -> RunResult:
    path = output_dir / "result.json"
    if not path.is_file():
        return RunResult(
            ok=False,
            error=f"experiment did not write {path}",
            returncode=-1,
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return RunResult(
            ok=False,
            error=f"invalid result.json: {exc}",
            returncode=-1,
        )
    if not isinstance(value, Mapping):
        return RunResult(
            ok=False,
            error="result.json must be an object",
            returncode=-1,
        )
    status = str(value.get("status", "ok") or "ok").casefold()
    return RunResult(
        ok=status in {"ok", "success", "succeeded", "passed"},
        metrics=dict(value.get("metrics", {}) or {}),
        artifacts=[str(item) for item in value.get("artifacts", ()) or ()],
        usage=dict(value.get("usage", {}) or {}),
        error=str(value.get("error", "") or ""),
        returncode=int(value.get("returncode", 0) or 0),
    )


def _remote_command(
    command: tuple[str, ...],
    *,
    revision_dir: Path,
    output_dir: Path,
    env: Mapping[str, str],
) -> str:
    exports = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items())
    argv = " ".join(shlex.quote(item) for item in command)
    return (
        "set -euo pipefail; "
        f"mkdir -p {shlex.quote(str(output_dir))}; "
        f"cd {shlex.quote(str(revision_dir))}; "
        f"env {exports} {argv}"
    )


def _result_from_broker_payload(
    payload: Mapping[str, Any],
    *,
    output_dir: Path,
) -> RunResult:
    local = _load_result(output_dir)
    returncode_raw = payload.get("returncode", -1)
    returncode = int(-1 if returncode_raw is None else returncode_raw)
    if local.ok and returncode == 0:
        local.returncode = 0
        return local
    return RunResult(
        ok=False,
        metrics=local.metrics,
        artifacts=local.artifacts,
        usage=local.usage,
        error=str(
            payload.get("error")
            or local.error
            or payload.get("stderr")
            or f"remote run failed with {returncode}"
        ),
        returncode=returncode,
    )
