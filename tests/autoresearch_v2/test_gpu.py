from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from researchclaw.autoresearch_v2.gpu import (
    AdaptiveGPUScheduler,
    GPUBroker,
    GPULease,
)
from researchclaw.autoresearch_v2.gpu_lease import SharedGPULeaseRegistry
from researchclaw.autoresearch_v2.models import JobKind, JobRecord


def _job(
    idea: str,
    gpus: tuple[int, int, int],
    *,
    timeout: float = 100,
) -> JobRecord:
    return JobRecord(
        job_id=f"{idea}-pilot",
        idea_id=idea,
        kind=JobKind.PILOT,
        requires_gpu=True,
        min_gpus=gpus[0],
        preferred_gpus=gpus[1],
        max_gpus=gpus[2],
        timeout_sec=timeout,
    )


def test_scheduler_malleably_backfills_available_capacity() -> None:
    scheduler = AdaptiveGPUScheduler(
        total_gpus=8,
        reserved_gpus=1,
        max_share_per_idea=0.5,
    )
    leases = [
        GPULease("task-a", "idea-a", "job-a", 3),
        GPULease("task-b", "idea-b", "job-b", 2),
    ]
    decision = scheduler.allocate(
        _job("idea-c", (1, 4, 4)),
        leases=leases,
    )
    assert decision.admitted
    assert decision.allocated_gpus == 2


def test_scheduler_enforces_per_idea_fair_share() -> None:
    scheduler = AdaptiveGPUScheduler(
        total_gpus=8,
        max_share_per_idea=0.5,
    )
    leases = [GPULease("task-a", "idea-a", "job-a", 4)]
    decision = scheduler.allocate(
        _job("idea-a", (1, 2, 2)),
        leases=leases,
    )
    assert not decision.admitted
    assert decision.reason == "insufficient_capacity"


def test_short_job_backfill_breaks_equal_priority_ties() -> None:
    scheduler = AdaptiveGPUScheduler(total_gpus=8)
    long = _job("long", (1, 4, 4), timeout=1000)
    short = _job("short", (1, 1, 1), timeout=10)
    ordered = scheduler.order(
        [long, short],
        priorities={"long": 0.8, "short": 0.8},
    )
    assert [job.idea_id for job in ordered] == ["short", "long"]


def test_broker_owns_keepalive_but_never_releases_adopted_pool() -> None:
    class Pool:
        def __init__(self) -> None:
            self.started = 0
            self.stopped = 0
            self.released = 0

        def start_keepalive(self) -> None:
            self.started += 1

        def stop_keepalive(self) -> None:
            self.stopped += 1

        def release(self) -> None:
            self.released += 1

    pool = Pool()
    broker = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(total_gpus=8),
    )
    assert pool.started == 1
    broker.close()
    assert pool.stopped == 1
    assert pool.released == 0


def test_broker_forwards_trusted_task_environment() -> None:
    class Pool:
        def __init__(self) -> None:
            self.request: dict[str, object] = {}

        def submit_task(self, command: str, **kwargs):
            self.request = {"command": command, **kwargs}
            return {"task_id": kwargs["task_id"]}

    pool = Pool()
    broker = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(total_gpus=1),
        task_env={
            "https_proxy": "http://proxy.invalid:3128",
            "HF_HOME": "/root/.cache/huggingface",
        },
    )
    job = _job("idea-env", (1, 1, 1))
    job.command = "true"
    decision = broker.submit(job, priorities={})
    assert decision.admitted
    environment = pool.request["env"]
    assert environment["https_proxy"] == "http://proxy.invalid:3128"
    assert environment["HF_HOME"] == "/root/.cache/huggingface"
    assert environment["AUTORESEARCH_V2_IDEA_ID"] == "idea-env"


def test_task_namespace_isolates_identical_jobs_between_runs() -> None:
    pool = _SharedPool()
    first = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(total_gpus=2),
        task_namespace="rsi-canary-a",
    )
    second = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(total_gpus=2),
        task_namespace="rsi-canary-b",
    )
    first_job = _job("same-idea", (1, 1, 1))
    second_job = _job("same-idea", (1, 1, 1))
    first_job.command = second_job.command = "true"

    first_task = first.submit(first_job, priorities={}).task_id
    second_task = second.submit(second_job, priorities={}).task_id

    assert first_task != second_task
    assert len(first_task) <= 128
    assert len(second_task) <= 128
    assert first_task in pool.requests
    assert second_task in pool.requests


def test_existing_submitted_task_id_is_preserved_across_namespace_change() -> None:
    pool = _SharedPool()
    broker = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(total_gpus=1),
        task_namespace="new-run",
    )
    job = _job("idea-resume", (1, 1, 1))
    job.command = "true"
    job.submitted_task_id = "legacy-task-id"

    decision = broker.submit(job, priorities={})

    assert decision.task_id == "legacy-task-id"
    assert "legacy-task-id" in pool.requests


def test_long_job_id_gets_bounded_collision_resistant_task_id() -> None:
    pool = _SharedPool()
    broker = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(total_gpus=1),
        task_namespace="rsi-canary-long",
    )
    job = _job("idea-" + ("x" * 180), (1, 1, 1))
    job.command = "true"

    task_id = broker.submit(job, priorities={}).task_id

    assert len(task_id) <= 128
    assert task_id in pool.requests


def test_explicit_attempt_number_produces_distinct_task_ids() -> None:
    broker = GPUBroker(
        pool=_SharedPool(),
        scheduler=AdaptiveGPUScheduler(total_gpus=1),
        task_namespace="retry-run",
    )
    job = _job("idea-retry", (1, 1, 1))

    first = broker.task_id_for(job, attempt_number=1)
    second = broker.task_id_for(job, attempt_number=2)

    assert first != second
    assert first.endswith("-attempt-01")
    assert second.endswith("-attempt-02")


def test_transient_probe_failure_keeps_lease() -> None:
    class Pool:
        def __init__(self) -> None:
            self.calls = 0

        def probe_task(self, task_id: str):
            del task_id
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary bridge timeout")
            return {"state": "running"}

    broker = GPUBroker(
        pool=Pool(),
        scheduler=AdaptiveGPUScheduler(total_gpus=2),
        probe_failure_threshold=2,
    )
    broker.leases["job"] = GPULease(
        "task",
        "idea",
        "job",
        1,
    )
    assert broker.reconcile() == []
    assert "job" in broker.leases
    assert broker.leases["job"].probe_failures == 1
    assert broker.reconcile() == []
    assert broker.leases["job"].probe_failures == 0


def test_reconcile_probes_multiple_gpu_jobs_concurrently() -> None:
    barrier = threading.Barrier(2)

    class Pool:
        def probe_task(self, task_id: str):
            del task_id
            barrier.wait(timeout=2)
            return {"state": "running"}

    broker = GPUBroker(
        pool=Pool(),
        scheduler=AdaptiveGPUScheduler(total_gpus=2),
    )
    broker.leases = {
        "job-a": GPULease("task-a", "idea-a", "job-a", 1),
        "job-b": GPULease("task-b", "idea-b", "job-b", 1),
    }

    assert broker.reconcile() == []
    assert set(broker.leases) == {"job-a", "job-b"}


class _SharedPool:
    def __init__(self) -> None:
        self.requests: dict[str, dict[str, object]] = {}
        self.states: dict[str, str] = {}

    def submit_task(self, command: str, **kwargs):
        task_id = str(kwargs["task_id"])
        self.requests[task_id] = {"command": command, **kwargs}
        self.states[task_id] = "running"
        return {"task_id": task_id}

    def probe_task(self, task_id: str):
        return {"state": self.states.get(task_id, "lost")}

    def collect_task(self, task_id: str):
        self.states[task_id] = "finished"
        return {"returncode": 0, "elapsed_sec": 1.0}

    def cancel_task(self, task_id: str):
        self.states[task_id] = "cancelled"
        return {"returncode": 130}


def _registry(path: Path, *, clock=lambda: 100.0):
    return SharedGPULeaseRegistry(
        path,
        pool_id="shared-test-pool",
        total_gpus=4,
        max_share_per_idea=1.0,
        owner_ttl_sec=10.0,
        clock=clock,
    )


def _broker(
    pool: _SharedPool,
    registry: SharedGPULeaseRegistry,
    owner: str,
) -> GPUBroker:
    return GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(
            total_gpus=4,
            max_share_per_idea=1.0,
        ),
        lease_registry=registry,
        owner_id=owner,
        lease_heartbeat_interval_sec=0,
    )


def test_global_registry_prevents_two_controllers_from_overallocating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "leases.sqlite3"
    pool = _SharedPool()
    broker_a = _broker(pool, _registry(path), "controller-a")
    broker_b = _broker(pool, _registry(path), "controller-b")
    job_a = _job("idea-a", (3, 3, 3))
    job_b = _job("idea-b", (2, 3, 3))
    job_a.command = job_b.command = "true"

    with ThreadPoolExecutor(max_workers=2) as executor:
        decisions = list(
            executor.map(
                lambda pair: pair[0].submit(pair[1], priorities={}),
                ((broker_a, job_a), (broker_b, job_b)),
            )
        )

    assert sum(item.allocated_gpus for item in decisions if item.admitted) <= 4
    assert sum(1 for item in decisions if item.admitted) == 1
    assert sum(lease.allocated_gpus for lease in _registry(path).list_leases()) <= 4


def test_reconcile_releases_global_capacity_for_another_controller(
    tmp_path: Path,
) -> None:
    path = tmp_path / "leases.sqlite3"
    pool = _SharedPool()
    broker_a = _broker(pool, _registry(path), "controller-a")
    broker_b = _broker(pool, _registry(path), "controller-b")
    first = _job("idea-a", (4, 4, 4))
    second = _job("idea-b", (1, 2, 2))
    first.command = second.command = "true"

    assert broker_a.submit(first, priorities={}).admitted
    assert not broker_b.submit(second, priorities={}).admitted
    pool.states["idea-a-pilot-attempt-01"] = "finished"
    assert broker_a.reconcile()[0][0] == first.job_id
    assert broker_b.submit(second, priorities={}).admitted


def test_cancel_releases_global_capacity(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    pool = _SharedPool()
    broker_a = _broker(pool, _registry(path), "controller-a")
    broker_b = _broker(pool, _registry(path), "controller-b")
    first = _job("idea-a", (4, 4, 4))
    second = _job("idea-b", (4, 4, 4))
    first.command = second.command = "true"

    assert broker_a.submit(first, priorities={}).admitted
    broker_a.cancel(first.job_id)
    assert broker_b.submit(second, priorities={}).admitted


def test_stale_owner_running_task_stays_reserved_until_terminal_probe(
    tmp_path: Path,
) -> None:
    now = [100.0]
    path = tmp_path / "leases.sqlite3"
    registry = _registry(path, clock=lambda: now[0])
    reservation = registry.reserve(
        owner_id="crashed",
        task_id="task-a",
        idea_id="idea-a",
        job_id="job-a",
        min_gpus=4,
        preferred_gpus=4,
        max_gpus=4,
    )
    assert reservation.admitted
    now[0] = 111.0

    assert registry.reap_stale(lambda _: {"state": "running"}) == []
    blocked = registry.reserve(
        owner_id="new",
        task_id="task-b",
        idea_id="idea-b",
        job_id="job-b",
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
    )
    assert not blocked.admitted
    now[0] = 122.0
    assert registry.reap_stale(lambda _: {"state": "finished"}) == ["task-a"]
    admitted = registry.reserve(
        owner_id="new",
        task_id="task-b",
        idea_id="idea-b",
        job_id="job-b",
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
    )
    assert admitted.admitted


def test_adopt_transfers_existing_task_accounting_to_new_owner(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "leases.sqlite3")
    assert registry.reserve(
        owner_id="old",
        task_id="task-a",
        idea_id="idea-a",
        job_id="job-a",
        min_gpus=2,
        preferred_gpus=2,
        max_gpus=2,
    ).admitted

    registry.adopt(
        owner_id="new",
        task_id="task-a",
        idea_id="idea-a",
        job_id="job-a",
        allocated_gpus=2,
    )

    leases = registry.list_leases()
    assert len(leases) == 1
    assert leases[0].owner_id == "new"
    assert leases[0].allocated_gpus == 2


def test_orphaned_reservation_expires_even_if_owner_is_still_heartbeating(
    tmp_path: Path,
) -> None:
    now = [100.0]
    registry = _registry(
        tmp_path / "leases.sqlite3",
        clock=lambda: now[0],
    )
    assert registry.reserve(
        owner_id="controller",
        task_id="uncertain-submit",
        idea_id="idea-a",
        job_id="job-a",
        min_gpus=4,
        preferred_gpus=4,
        max_gpus=4,
    ).admitted
    registry.detach("controller", "uncertain-submit")
    now[0] = 111.0
    registry.heartbeat("controller")

    assert registry.reap_stale(lambda _: {"state": "lost"}) == [
        "uncertain-submit"
    ]
    assert registry.list_leases() == []


def test_expired_orphan_does_not_block_reservation_when_probe_is_unavailable(
    tmp_path: Path,
) -> None:
    now = [100.0]
    registry = _registry(
        tmp_path / "leases.sqlite3",
        clock=lambda: now[0],
    )
    assert registry.reserve(
        owner_id="crashed",
        task_id="missing-task",
        idea_id="idea-a",
        job_id="job-a",
        min_gpus=4,
        preferred_gpus=4,
        max_gpus=4,
    ).admitted
    registry.detach("crashed", "missing-task")
    now[0] = 111.0

    reservation = registry.reserve(
        owner_id="new",
        task_id="task-b",
        idea_id="idea-b",
        job_id="job-b",
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
    )

    assert reservation.admitted
    assert reservation.allocated_gpus == 1


def test_registry_rejects_capacity_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "leases.sqlite3"
    _registry(path)

    try:
        SharedGPULeaseRegistry(
            path,
            pool_id="shared-test-pool",
            total_gpus=8,
            owner_ttl_sec=10.0,
        )
    except ValueError as exc:
        assert "capacity mismatch" in str(exc)
    else:
        raise AssertionError("capacity mismatch must fail closed")
