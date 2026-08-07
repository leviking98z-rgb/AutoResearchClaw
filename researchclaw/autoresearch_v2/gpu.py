"""Logical GPU scheduling over one existing asynchronous pool owner."""

from __future__ import annotations

import concurrent.futures
import hashlib
import math
import os
import re
import socket
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .gpu_lease import (
    SharedGPULeaseRegistry,
    shared_registry_path,
    stable_pool_identity,
)
from .models import JobRecord

_MAX_POOL_TASK_ID_LENGTH = 128
_TASK_NAMESPACE_LENGTH = 12


class AsyncGPUPool(Protocol):
    def submit_task(
        self,
        command: str,
        *,
        timeout_sec: float,
        env: Mapping[str, str] | None = None,
        task_id: str | None = None,
        require_ready: bool = True,
        num_gpus: int = 0,
        num_cpus: int = 1,
    ) -> Any: ...

    def probe_task(self, task_id: str) -> Any: ...

    def collect_task(self, task_id: str) -> Any: ...

    def cancel_task(self, task_id: str) -> Any: ...


@dataclass(slots=True)
class GPULease:
    task_id: str
    idea_id: str
    job_id: str
    allocated_gpus: int
    state: str = "running"
    probe_failures: int = 0


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    admitted: bool
    allocated_gpus: int = 0
    reason: str = ""
    task_id: str = ""


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def stable_task_namespace(value: str) -> str:
    """Return a short, safe namespace for one controller system id."""

    text = str(value or "").strip()
    if not text:
        return ""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("._-")
    if not slug or not slug[0].isalnum():
        slug = "run"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    prefix = slug[: _TASK_NAMESPACE_LENGTH - len(digest) - 1]
    return f"{prefix}-{digest}" if prefix else digest


def _new_pool_task_id(
    job: JobRecord,
    *,
    task_namespace: str,
    attempt_number: int | None = None,
) -> str:
    """Build a deterministic pool id without truncation collisions."""

    number = (
        job.attempt + 1
        if attempt_number is None
        else max(1, int(attempt_number))
    )
    base = f"{job.job_id}-attempt-{number:02d}"
    candidate = f"{task_namespace}-{base}" if task_namespace else base
    if len(candidate) <= _MAX_POOL_TASK_ID_LENGTH:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:16]
    prefix_length = _MAX_POOL_TASK_ID_LENGTH - len(digest) - 1
    return f"{candidate[:prefix_length].rstrip('._-')}-{digest}"


class AdaptiveGPUScheduler:
    """Fair-share scheduler with malleable allocation and short-job backfill."""

    def __init__(
        self,
        *,
        total_gpus: int,
        reserved_gpus: int = 0,
        max_share_per_idea: float = 0.5,
        target_utilization: float = 0.9,
    ) -> None:
        self.total_gpus = max(0, int(total_gpus))
        self.reserved_gpus = max(0, int(reserved_gpus))
        self.max_share_per_idea = float(max_share_per_idea)
        self.target_utilization = float(target_utilization)

    @property
    def usable_gpus(self) -> int:
        return max(0, self.total_gpus - self.reserved_gpus)

    def used(self, leases: Iterable[GPULease]) -> int:
        return sum(
            lease.allocated_gpus
            for lease in leases
            if lease.state
            in {
                "reserved",
                "submitted",
                "probe_pending",
                "running",
                "probe_degraded",
                "orphaned",
            }
        )

    def available(self, leases: Iterable[GPULease]) -> int:
        return max(0, self.usable_gpus - self.used(leases))

    def target_gpus(self, *, pending: bool) -> int:
        if not pending:
            return 0
        return min(
            self.usable_gpus,
            max(1, math.ceil(self.usable_gpus * self.target_utilization)),
        )

    def allocate(
        self,
        job: JobRecord,
        *,
        leases: Iterable[GPULease],
    ) -> AllocationDecision:
        if not job.requires_gpu:
            return AllocationDecision(True, 0, "non_gpu")
        current = list(leases)
        available = self.available(current)
        idea_cap = max(
            job.min_gpus,
            int(self.usable_gpus * self.max_share_per_idea),
        )
        used_by_idea = sum(
            lease.allocated_gpus
            for lease in current
            if lease.idea_id == job.idea_id
            and lease.state
            in {
                "submitted",
                "probe_pending",
                "running",
                "probe_degraded",
                "orphaned",
            }
        )
        cap = min(
            available,
            max(0, idea_cap - used_by_idea),
            job.max_gpus or job.preferred_gpus or job.min_gpus,
        )
        if cap < job.min_gpus:
            return AllocationDecision(False, 0, "insufficient_capacity")
        preferred = job.preferred_gpus or job.min_gpus
        allocated = max(job.min_gpus, min(preferred, cap))
        return AllocationDecision(True, allocated, "admitted")

    @staticmethod
    def order(
        jobs: Iterable[JobRecord],
        priorities: Mapping[str, float],
    ) -> list[JobRecord]:
        """Information-value priority plus shortest-job backfill."""

        return sorted(
            jobs,
            key=lambda job: (
                -float(priorities.get(job.idea_id, 0.0)),
                job.preferred_gpus or job.min_gpus,
                job.timeout_sec,
                job.created_at,
                job.job_id,
            ),
        )


class GPUBroker:
    """Single logical owner for every physical asynchronous GPU task."""

    def __init__(
        self,
        *,
        pool: AsyncGPUPool,
        scheduler: AdaptiveGPUScheduler,
        probe_failure_threshold: int = 3,
        lease_registry: SharedGPULeaseRegistry | None = None,
        owner_id: str | None = None,
        lease_heartbeat_interval_sec: float | None = None,
        task_env: Mapping[str, str] | None = None,
        task_namespace: str = "",
        manage_pool_keepalive: bool = True,
        reconcile_timeout_sec: float = 0.0,
        probe_interval_sec: float = 0.0,
        allocation_id: str = "",
        pool_id: str = "",
    ) -> None:
        self.pool = pool
        self.scheduler = scheduler
        config = getattr(pool, "config", None)
        self.allocation_id = str(allocation_id or "").strip()
        self.pool_id = str(
            pool_id or getattr(config, "pool_id", "") or ""
        ).strip()
        self.manage_pool_keepalive = bool(manage_pool_keepalive)
        self.probe_failure_threshold = max(
            1,
            int(probe_failure_threshold),
        )
        self.task_env = {
            str(key): str(value)
            for key, value in (task_env or {}).items()
        }
        self.task_namespace = stable_task_namespace(task_namespace)
        self.leases: dict[str, GPULease] = {}
        # Reconciliation runs on the Controller heartbeat thread. Keep it
        # strictly non-blocking by default: background probes are harvested on
        # later ticks instead of making every tick wait for ClusterBridge.
        self.reconcile_timeout_sec = max(0.0, float(reconcile_timeout_sec))
        self.probe_interval_sec = max(0.0, float(probe_interval_sec))
        self.lease_registry = lease_registry
        self.owner_id = owner_id or (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        )
        self._lease_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None
        if self.lease_registry is not None:
            self.lease_registry.heartbeat(
                self.owner_id,
                prune_expired_orphans=False,
            )
            interval = lease_heartbeat_interval_sec
            if interval is None:
                interval = min(
                    30.0,
                    max(1.0, self.lease_registry.owner_ttl_sec / 3.0),
                )
            if interval > 0:
                self._lease_thread = threading.Thread(
                    target=self._lease_heartbeat_loop,
                    args=(float(interval),),
                    name=f"gpu-lease-{self.owner_id[-12:]}",
                    daemon=True,
                )
                self._lease_thread.start()
        # Pool probes use ClusterBridge subprocesses. Reconcile them in
        # parallel so tick latency is bounded by one transport call instead of
        # growing linearly with the number of running GPU jobs.
        self._probe_workers = max(
            1,
            min(32, int(getattr(scheduler, "total_gpus", 1) or 1)),
        )
        self._probe_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._probe_workers,
            thread_name_prefix="autoresearch-v2-gpu-probe",
        )
        self._probe_futures: dict[
            str,
            concurrent.futures.Future[Any],
        ] = {}
        self._probe_not_before: dict[str, float] = {}
        self._submit_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._probe_workers,
            thread_name_prefix="autoresearch-v2-gpu-submit",
        )
        self._submit_futures: dict[
            str,
            concurrent.futures.Future[None],
        ] = {}
        self._closed = False
        if self.manage_pool_keepalive and hasattr(
            self.pool,
            "start_keepalive",
        ):
            self.pool.start_keepalive()

    def submit(
        self,
        job: JobRecord,
        *,
        priorities: Mapping[str, float],
    ) -> AllocationDecision:
        del priorities
        if self._closed:
            raise RuntimeError("GPU broker is closed")
        if not job.command:
            return AllocationDecision(False, 0, "command_missing")
        existing = self.leases.get(job.job_id)
        if existing is not None:
            return AllocationDecision(
                True,
                existing.allocated_gpus,
                "existing_lease",
                existing.task_id,
            )
        task_id = (
            job.submitted_task_id
            or self.task_id_for(job)
        )
        if self.lease_registry is not None:
            self._refresh_global_leases()
            reservation = self.lease_registry.reserve(
                owner_id=self.owner_id,
                task_id=task_id,
                idea_id=job.idea_id,
                job_id=job.job_id,
                min_gpus=job.min_gpus,
                preferred_gpus=job.preferred_gpus,
                max_gpus=job.max_gpus,
                prune_expired_orphans=False,
            )
            decision = AllocationDecision(
                reservation.admitted,
                reservation.allocated_gpus,
                reservation.reason,
                task_id if reservation.admitted else "",
            )
        else:
            decision = self.scheduler.allocate(
                job,
                leases=self.leases.values(),
            )
        if not decision.admitted:
            return decision
        lease = GPULease(
            task_id=task_id,
            idea_id=job.idea_id,
            job_id=job.job_id,
            allocated_gpus=decision.allocated_gpus,
            state="reserved",
        )
        self.leases[job.job_id] = lease
        try:
            self._submit_futures[job.job_id] = self._submit_executor.submit(
                self._submit_reserved_task,
                job,
                lease,
            )
        except Exception:
            self.leases.pop(job.job_id, None)
            if self.lease_registry is not None:
                self.lease_registry.release(self.owner_id, task_id)
            raise
        return AllocationDecision(
            True,
            decision.allocated_gpus,
            "submitting",
            task_id,
        )

    def _submit_reserved_task(
        self,
        job: JobRecord,
        lease: GPULease,
    ) -> None:
        try:
            self.pool.submit_task(
                job.command,
                timeout_sec=job.timeout_sec,
                task_id=lease.task_id,
                num_gpus=lease.allocated_gpus,
                num_cpus=2,
                env={
                    **self.task_env,
                    "AUTORESEARCH_V2_IDEA_ID": job.idea_id,
                    "AUTORESEARCH_V2_JOB_ID": job.job_id,
                    "AUTORESEARCH_V2_ATTEMPT_ID": job.attempt_id,
                    "AUTORESEARCH_V2_GPU_COUNT": str(
                        lease.allocated_gpus
                    ),
                    "AUTORESEARCH_V2_OUTPUT_DIR": job.expected_output_dir,
                },
            )
        except Exception as submit_error:
            # submit_task uses deterministic IDs. A transport error may occur
            # after durable creation, so probe before permitting a duplicate.
            try:
                probe = self.pool.probe_task(lease.task_id)
            except Exception:  # noqa: BLE001
                raise submit_error
            state = str(_value(probe, "state", "unknown"))
            if state not in {
                "submitted",
                "running",
                "finished",
                "timed_out",
                "lost",
            }:
                raise
        if self.lease_registry is not None:
            self.lease_registry.mark_state(
                self.owner_id,
                lease.task_id,
                "submitted",
            )

    def task_id_for(
        self,
        job: JobRecord,
        *,
        attempt_number: int | None = None,
    ) -> str:
        """Return the namespaced deterministic id for one new attempt."""

        return _new_pool_task_id(
            job,
            task_namespace=self.task_namespace,
            attempt_number=attempt_number,
        )

    def task_exists(self, task_id: str) -> bool:
        """Return transport-free evidence that this pool owns ``task_id``."""

        if not task_id:
            return False
        task_exists = getattr(self.pool, "task_exists", None)
        if not callable(task_exists):
            # Legacy/test pools cannot prove absence. Preserve the historical
            # adoption behavior rather than turning uncertainty into loss.
            return True
        try:
            return bool(task_exists(task_id))
        except Exception:  # noqa: BLE001
            return False

    def adopt(
        self,
        job: JobRecord,
        *,
        task_id: str,
        allocated_gpus: int,
    ) -> bool:
        if not task_id:
            raise ValueError("cannot adopt a GPU job without a task id")
        if not self.task_exists(task_id):
            return False
        if self.lease_registry is not None:
            self.lease_registry.adopt(
                owner_id=self.owner_id,
                task_id=task_id,
                idea_id=job.idea_id,
                job_id=job.job_id,
                allocated_gpus=allocated_gpus,
            )
        self.leases[job.job_id] = GPULease(
            task_id=task_id,
            idea_id=job.idea_id,
            job_id=job.job_id,
            allocated_gpus=max(1, int(allocated_gpus)),
            state="running",
        )
        return True

    def reconcile(self) -> list[tuple[str, dict[str, Any]]]:
        if self._closed:
            return []
        self._refresh_global_leases()
        completed: list[tuple[str, dict[str, Any]]] = []
        leases = list(self.leases.items())
        probes: dict[
            str,
            concurrent.futures.Future[Any],
        ] = {}
        terminal_jobs: set[str] = set()
        for job_id, lease in leases:
            submit_future = self._submit_futures.get(job_id)
            if submit_future is not None:
                if not submit_future.done():
                    continue
                self._submit_futures.pop(job_id, None)
                try:
                    submit_future.result()
                except Exception as exc:  # noqa: BLE001
                    completed.append(
                        (
                            job_id,
                            self._complete_payload(
                                {
                                    "returncode": -1,
                                    "elapsed_sec": 0.0,
                                    "pool_state": "submission_failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "failure_class": "infrastructure_transient",
                                    "consume_attempt": False,
                                },
                                lease,
                                "submission_failed",
                            ),
                        )
                    )
                    terminal_jobs.add(job_id)
                    self._release_lease(job_id, lease)
                    continue
                lease.state = "submitted"
            cached = self._collect_cached_task(lease.task_id)
            if cached is not None:
                completed.append(
                    (
                        job_id,
                        self._result_payload(cached, lease, "finished"),
                    )
                )
                terminal_jobs.add(job_id)
                self._release_lease(job_id, lease)
                continue
            if lease.state == "reserved":
                continue
            future = self._probe_futures.get(job_id)
            if future is not None and not future.done():
                probes[job_id] = future
                continue
            if future is None:
                if time.monotonic() < self._probe_not_before.get(job_id, 0.0):
                    continue
                future = self._probe_executor.submit(
                    self.pool.probe_task,
                    lease.task_id,
                )
                self._probe_futures[job_id] = future
            probes[job_id] = future
        unfinished = [
            future
            for future in probes.values()
            if not future.done()
        ]
        if unfinished and self.reconcile_timeout_sec > 0:
            concurrent.futures.wait(
                tuple(unfinished),
                timeout=self.reconcile_timeout_sec,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
        for job_id, lease in leases:
            if job_id in terminal_jobs:
                continue
            future = probes.get(job_id)
            if future is None or not future.done():
                # A healthy in-flight probe is not degradation. Preserve the
                # last confirmed task state and expose "probe_pending" only for
                # a never-confirmed reservation/submission.
                if lease.state in {"reserved", "submitted"}:
                    lease.state = "probe_pending"
                continue
            self._probe_futures.pop(job_id, None)
            self._probe_not_before[job_id] = (
                time.monotonic() + self.probe_interval_sec
            )
            try:
                probe = future.result()
            except Exception:  # noqa: BLE001
                lease.probe_failures += 1
                lease.state = "probe_degraded"
                if self.lease_registry is not None:
                    self.lease_registry.mark_state(
                        self.owner_id,
                        lease.task_id,
                        lease.state,
                    )
                # Transport/probe failure is not scientific task failure.
                # Keep the durable lease and retry asynchronously; otherwise
                # a congested ClusterBridge queue consumes retries and can
                # launch duplicate GPU attempts while the original task lives.
                continue
            state = str(_value(probe, "state", "unknown"))
            lease.state = state
            lease.probe_failures = 0
            if state in {"submitted", "running"}:
                if self.lease_registry is not None:
                    self.lease_registry.mark_state(
                        self.owner_id,
                        lease.task_id,
                        state,
                    )
                continue
            try:
                result = self.pool.collect_task(lease.task_id)
            except Exception as exc:  # noqa: BLE001
                payload = {
                    "returncode": -1,
                    "elapsed_sec": float(
                        _value(probe, "elapsed_sec", 0.0) or 0.0
                    ),
                    "pool_state": state,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            else:
                payload = {
                    "returncode": int(_value(result, "returncode", -1)),
                    "elapsed_sec": float(
                        _value(result, "elapsed_sec", 0.0) or 0.0
                    ),
                    "stdout": str(_value(result, "stdout", "") or ""),
                    "stderr": str(_value(result, "stderr", "") or ""),
                    "stdout_path": str(
                        _value(result, "stdout_path", "") or ""
                    ),
                    "stderr_path": str(
                        _value(result, "stderr_path", "") or ""
                    ),
                    "result_path": str(
                        _value(result, "result_path", "") or ""
                    ),
                    "remote_dir": str(
                        _value(result, "remote_dir", "") or ""
                    ),
                    "timed_out": bool(
                        _value(result, "timed_out", False)
                    ),
                    "trusted_gpu_evidence": _value(
                        result,
                        "trusted_gpu_evidence",
                        None,
                    ),
                }
            completed.append(
                (
                    job_id,
                    self._complete_payload(payload, lease, state),
                )
            )
            self._release_lease(job_id, lease)
        return completed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._lease_stop.set()
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=5.0)
        self._submit_executor.shutdown(wait=False, cancel_futures=False)
        self._probe_executor.shutdown(wait=False, cancel_futures=True)
        self._detach_workers_for_process_exit(self._submit_executor)
        self._detach_workers_for_process_exit(self._probe_executor)
        if self.lease_registry is not None:
            for lease in self.leases.values():
                submit_future = self._submit_futures.get(lease.job_id)
                if submit_future is not None:
                    if not submit_future.done():
                        self.lease_registry.detach(
                            self.owner_id,
                            lease.task_id,
                        )
                        continue
                    try:
                        submit_future.result()
                    except Exception:  # noqa: BLE001
                        self.lease_registry.release(
                            self.owner_id,
                            lease.task_id,
                        )
                        continue
                cached = self._collect_cached_task(lease.task_id)
                if cached is not None:
                    state = "finished"
                else:
                    future = self._probe_futures.get(lease.job_id)
                    if future is None or not future.done():
                        state = "unknown"
                    else:
                        try:
                            state = str(
                                _value(
                                    future.result(),
                                    "state",
                                    "unknown",
                                )
                            )
                        except Exception:  # noqa: BLE001
                            state = "unknown"
                if state in {"submitted", "running"}:
                    self.lease_registry.detach(
                        self.owner_id,
                        lease.task_id,
                    )
                elif state in {"finished", "timed_out", "lost", "cancelled"}:
                    self.lease_registry.release(
                        self.owner_id,
                        lease.task_id,
                    )
                else:
                    self.lease_registry.detach(
                        self.owner_id,
                        lease.task_id,
                    )
            self.lease_registry.close_owner(self.owner_id)
        # v2 adopts but does not release the physical pool. It does own the
        # keepalive thread it started for the lifetime of this controller.
        if self.manage_pool_keepalive and hasattr(
            self.pool,
            "stop_keepalive",
        ):
            self.pool.stop_keepalive()

    def cancel(self, job_id: str) -> None:
        lease = self.leases.pop(job_id, None)
        submit_future = self._submit_futures.pop(job_id, None)
        future = self._probe_futures.pop(job_id, None)
        self._probe_not_before.pop(job_id, None)
        if future is not None:
            future.cancel()
        if lease is not None:
            if submit_future is not None and not submit_future.done():
                cancelled = submit_future.cancel()
                if not cancelled:
                    try:
                        submit_future.result(timeout=0.05)
                    except concurrent.futures.TimeoutError:
                        pass
                    except Exception:  # noqa: BLE001
                        cancelled = True
                if not submit_future.done():
                    if self.lease_registry is not None:
                        self.lease_registry.detach(
                            self.owner_id,
                            lease.task_id,
                        )
                    return
            if submit_future is not None:
                try:
                    submit_future.result()
                except Exception:  # noqa: BLE001
                    if self.lease_registry is not None:
                        self.lease_registry.release(
                            self.owner_id,
                            lease.task_id,
                        )
                    return
            try:
                self.pool.cancel_task(lease.task_id)
            except Exception:
                if self.lease_registry is not None:
                    self.lease_registry.detach(
                        self.owner_id,
                        lease.task_id,
                    )
                raise
            if self.lease_registry is not None:
                self.lease_registry.release(
                    self.owner_id,
                    lease.task_id,
                )

    def snapshot(self, *, pending_jobs: int = 0) -> dict[str, Any]:
        local_leases = list(self.leases.values())
        global_leases = (
            self.lease_registry.list_leases()
            if self.lease_registry is not None
            else []
        )
        leases = (
            [
                GPULease(
                    lease.task_id,
                    lease.idea_id,
                    lease.job_id,
                    lease.allocated_gpus,
                    lease.state,
                )
                for lease in global_leases
            ]
            if global_leases
            else local_leases
        )
        used = self.scheduler.used(leases)
        usable = self.scheduler.usable_gpus
        return {
            "total_gpus": self.scheduler.total_gpus,
            "reserved_gpus": self.scheduler.reserved_gpus,
            "usable_gpus": usable,
            "allocated_gpus": used,
            "available_gpus": max(0, usable - used),
            "utilization": (used / usable) if usable else 0.0,
            "target_utilization": self.scheduler.target_utilization,
            "target_gpus": self.scheduler.target_gpus(
                pending=bool(pending_jobs)
            ),
            "pending_jobs": max(0, int(pending_jobs)),
            "allocation_id": self.allocation_id,
            "pool_id": self.pool_id,
            "leases": [
                {
                    "task_id": lease.task_id,
                    "idea_id": lease.idea_id,
                    "job_id": lease.job_id,
                    "allocated_gpus": lease.allocated_gpus,
                    "state": lease.state,
                    "probe_failures": lease.probe_failures,
                }
                for lease in leases
            ],
        }

    def _refresh_global_leases(self) -> None:
        if self.lease_registry is None:
            return
        self.lease_registry.heartbeat(
            self.owner_id,
            prune_expired_orphans=False,
        )
        # Stale owners are inspected using only cached/asynchronous evidence.
        # Unknown expired work remains reserved rather than being deleted and
        # accidentally submitted twice while a remote GPU task is still alive.
        self.lease_registry.reap_stale(
            self._reap_probe_task,
            release_unverified_expired=False,
        )

    def _collect_cached_task(self, task_id: str) -> Any | None:
        collect_cached = getattr(self.pool, "collect_cached_task", None)
        if not callable(collect_cached):
            return None
        try:
            return collect_cached(task_id)
        except Exception:  # noqa: BLE001
            return None

    def _reap_probe_task(self, task_id: str) -> Any:
        cached = self._collect_cached_task(task_id)
        if cached is not None:
            return {"state": "finished"}
        for job_id, lease in self.leases.items():
            if lease.task_id != task_id:
                continue
            future = self._probe_futures.get(job_id)
            if future is not None and future.done():
                return future.result()
            break
        task_exists = getattr(self.pool, "task_exists", None)
        if callable(task_exists) and not task_exists(task_id):
            # No durable task metadata means this was an abandoned reservation,
            # not a detached remote task that still needs protection.
            return {"state": "lost"}
        # Shared lease reaping runs on the Controller heartbeat path. Never
        # perform a new ClusterBridge RPC here: uncertain work stays reserved
        # until a cached or asynchronous Broker probe resolves it.
        raise TimeoutError("no cached asynchronous GPU probe result")

    @staticmethod
    def _complete_payload(
        payload: dict[str, Any],
        lease: GPULease,
        state: str,
    ) -> dict[str, Any]:
        payload["task_id"] = lease.task_id
        payload["allocated_gpus"] = lease.allocated_gpus
        payload["pool_state"] = payload.get("pool_state", state)
        return payload

    def _result_payload(
        self,
        result: Any,
        lease: GPULease,
        state: str,
    ) -> dict[str, Any]:
        return self._complete_payload(
            {
                "returncode": int(_value(result, "returncode", -1)),
                "elapsed_sec": float(
                    _value(result, "elapsed_sec", 0.0) or 0.0
                ),
                "stdout": str(_value(result, "stdout", "") or ""),
                "stderr": str(_value(result, "stderr", "") or ""),
                "stdout_path": str(
                    _value(result, "stdout_path", "") or ""
                ),
                "stderr_path": str(
                    _value(result, "stderr_path", "") or ""
                ),
                "result_path": str(
                    _value(result, "result_path", "") or ""
                ),
                "remote_dir": str(
                    _value(result, "remote_dir", "") or ""
                ),
                "timed_out": bool(
                    _value(result, "timed_out", False)
                ),
                "trusted_gpu_evidence": _value(
                    result,
                    "trusted_gpu_evidence",
                    None,
                ),
            },
            lease,
            state,
        )

    def _release_lease(self, job_id: str, lease: GPULease) -> None:
        self._submit_futures.pop(job_id, None)
        future = self._probe_futures.pop(job_id, None)
        self._probe_not_before.pop(job_id, None)
        if future is not None:
            future.cancel()
        if self.lease_registry is not None:
            self.lease_registry.release(
                self.owner_id,
                lease.task_id,
            )
        self.leases.pop(job_id, None)

    @staticmethod
    def _detach_workers_for_process_exit(
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        """Do not let slow ClusterBridge probes pin interpreter shutdown."""

        try:
            from concurrent.futures import thread as thread_module
        except ImportError:
            return
        for worker in tuple(executor._threads):
            thread_module._threads_queues.pop(worker, None)

    def _lease_heartbeat_loop(self, interval_sec: float) -> None:
        while not self._lease_stop.wait(interval_sec):
            try:
                if self.lease_registry is not None:
                    self.lease_registry.heartbeat(
                        self.owner_id,
                        prune_expired_orphans=False,
                    )
            except Exception:  # noqa: BLE001, S112
                # Scheduling paths retry the heartbeat synchronously and fail
                # closed if the shared registry remains unavailable.
                continue


def build_clusterbridge_broker(
    pool_config: str,
    *,
    reserved_gpus: int,
    max_share_per_idea: float,
    target_utilization: float,
    probe_failure_threshold: int = 3,
    task_env: Mapping[str, str] | None = None,
    restore_state: bool = True,
    task_namespace: str = "",
) -> GPUBroker:
    """Adopt a prepared ClusterBridge/Ray pool without owning its lifecycle."""

    from researchclaw.experiment.clusterbridge_pool import (
        ClusterBridgePool,
        PoolNotClaimedError,
        PoolNotReadyError,
    )
    from researchclaw.factory.pool_config import PoolConfigSummary

    summary = PoolConfigSummary.from_file(pool_config)
    pool = ClusterBridgePool.from_file(
        summary.config_path,
        restore_state=restore_state,
    )
    if not pool.claimed:
        raise PoolNotClaimedError(
            "v2 requires an already claimed and prepared shared GPU pool; "
            "it will not claim, prepare, or release physical nodes"
        )
    if not pool.prepared:
        raise PoolNotReadyError(
            "v2 requires an already claimed and prepared shared GPU pool; "
            "it will not claim, prepare, or release physical nodes"
        )
    return build_clusterbridge_broker_from_pool(
        pool,
        total_gpus=summary.expected_total_gpus,
        reserved_gpus=reserved_gpus,
        max_share_per_idea=max_share_per_idea,
        target_utilization=target_utilization,
        probe_failure_threshold=probe_failure_threshold,
        task_env=task_env,
        task_namespace=task_namespace,
    )


def build_clusterbridge_broker_from_pool(
    pool: AsyncGPUPool,
    *,
    total_gpus: int,
    reserved_gpus: int,
    max_share_per_idea: float,
    target_utilization: float,
    probe_failure_threshold: int = 3,
    task_env: Mapping[str, str] | None = None,
    lease_registry_path: str | Path | None = None,
    task_namespace: str = "",
    manage_pool_keepalive: bool = True,
    reconcile_timeout_sec: float = 0.0,
    probe_interval_sec: float = 0.0,
) -> GPUBroker:
    """Build a broker around an already claimed and prepared pool object."""

    config = getattr(pool, "config", None)
    nodes = [
        {
            "address": node.address,
            "ray_ip": node.ray_ip,
            "gpu_ids": list(node.gpu_ids),
        }
        for node in getattr(config, "nodes", ())
    ]
    pool_id = stable_pool_identity(
        nodes=nodes,
        expected_total_gpus=total_gpus,
    )
    state_dir = Path(getattr(pool, "state_dir", Path.cwd())).resolve()
    registry_path = (
        Path(lease_registry_path).expanduser().resolve()
        if lease_registry_path is not None
        else shared_registry_path(state_dir, pool_id)
    )
    lease_registry = SharedGPULeaseRegistry(
        registry_path,
        pool_id=pool_id,
        total_gpus=total_gpus,
        reserved_gpus=reserved_gpus,
        max_share_per_idea=max_share_per_idea,
        owner_ttl_sec=max(
            60.0,
            float(getattr(config, "renew_interval_sec", 120.0)) * 2.0,
        ),
    )
    return GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(
            total_gpus=total_gpus,
            reserved_gpus=reserved_gpus,
            max_share_per_idea=max_share_per_idea,
            target_utilization=target_utilization,
        ),
        probe_failure_threshold=probe_failure_threshold,
        lease_registry=lease_registry,
        task_env=task_env,
        task_namespace=task_namespace,
        manage_pool_keepalive=manage_pool_keepalive,
        reconcile_timeout_sec=reconcile_timeout_sec,
        probe_interval_sec=probe_interval_sec,
        pool_id=str(getattr(config, "pool_id", "") or ""),
    )


def clusterbridge_capacity(pool_config: str) -> int:
    """Read configured GPU capacity without adopting or probing a live pool."""

    from researchclaw.factory.pool_config import PoolConfigSummary

    return PoolConfigSummary.from_file(pool_config).expected_total_gpus
