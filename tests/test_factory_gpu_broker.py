from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from researchclaw.factory.config import FactoryConfig
from researchclaw.factory.gpu_broker import GPUBroker
from researchclaw.factory.models import ResourceRequest, WorkItem, WorkKind
from researchclaw.factory.scheduler import FactoryScheduler
from researchclaw.factory.store import FactoryStore


@dataclass
class Result:
    returncode: int = 0
    stdout_path: str = "stdout.log"
    stderr_path: str = "stderr.log"
    elapsed_sec: float = 1.0
    timed_out: bool = False


class FakePool:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.submit_kwargs: list[dict[str, object]] = []
        self.finished = False

    def submit_task(self, command: str, **kwargs):
        del command
        self.submitted.append(kwargs["task_id"])
        self.submit_kwargs.append(dict(kwargs))
        return {"task_id": kwargs["task_id"]}

    def probe_task(self, task_id: str):
        del task_id
        return {"state": "finished" if self.finished else "running"}

    def collect_task(self, task_id: str):
        del task_id
        return Result()

    def cancel_task(self, task_id: str):
        del task_id
        return Result(returncode=130)


class SubmitFailPool(FakePool):
    def submit_task(self, command: str, **kwargs):
        del command, kwargs
        raise RuntimeError("submission failed")

    def probe_task(self, task_id: str):
        del task_id
        raise RuntimeError("task not created")


class CollectFailPool(FakePool):
    def collect_task(self, task_id: str):
        del task_id
        raise RuntimeError("collection failed")


class AmbiguousSubmitPool(FakePool):
    def __init__(self) -> None:
        super().__init__()
        self.created: set[str] = set()

    def submit_task(self, command: str, **kwargs):
        del command
        self.created.add(kwargs["task_id"])
        raise RuntimeError("response lost after launch")

    def probe_task(self, task_id: str):
        if task_id in self.created:
            return {"state": "running"}
        raise RuntimeError("missing")


def test_gpu_broker_is_single_submission_and_releases_lease(
    tmp_path: Path,
) -> None:
    config = FactoryConfig.from_mapping(
        {"factory": {"scheduler": {"reserved_gpus": 0}}}
    )
    store = FactoryStore(tmp_path)
    store.initialize()
    item = WorkItem(
        item_id="idea-a-pilot",
        idea_id="idea-a",
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        command="python train.py",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=2,
            max_gpus=2,
        ),
    )
    store.save_work_item(item)
    pool = FakePool()
    broker = GPUBroker(
        pool=pool,
        store=store,
        scheduler=FactoryScheduler(config, total_gpus=4),
    )
    assert broker.submit(item).admitted is True
    restored = store.get_work_item(item.item_id)
    assert restored is not None
    assert restored.attempt == 1
    assert len(pool.submitted) == 1
    submitted_env = pool.submit_kwargs[0]["env"]
    assert isinstance(submitted_env, dict)
    assert submitted_env["RESEARCHCLAW_FACTORY_ID"] == store.factory_id
    assert submitted_env["RESEARCHCLAW_WORK_ITEM_ATTEMPT"] == "1"
    assert broker.reconcile() == []
    pool.finished = True
    changed = broker.reconcile()
    assert changed[0].status.value == "succeeded"
    assert store.list_leases()[0].status.value == "released"


def test_gpu_submit_failure_rolls_back_lease(tmp_path: Path) -> None:
    import pytest

    config = FactoryConfig.from_mapping(
        {"factory": {"scheduler": {"reserved_gpus": 0}}}
    )
    store = FactoryStore(tmp_path)
    store.initialize()
    item = WorkItem(
        item_id="idea-submit-fail",
        idea_id="idea-submit-fail",
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        command="python train.py",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=1,
            max_gpus=1,
        ),
    )
    store.save_work_item(item)
    broker = GPUBroker(
        pool=SubmitFailPool(),
        store=store,
        scheduler=FactoryScheduler(config, total_gpus=2),
    )

    with pytest.raises(RuntimeError, match="submission failed"):
        broker.submit(item)
    assert store.list_leases()[0].status.value == "released"
    assert item.attempt == 1


def test_gpu_collect_failure_isolated_as_failed_item(tmp_path: Path) -> None:
    config = FactoryConfig.from_mapping(
        {"factory": {"scheduler": {"reserved_gpus": 0}}}
    )
    store = FactoryStore(tmp_path)
    store.initialize()
    item = WorkItem(
        item_id="idea-collect-fail",
        idea_id="idea-collect-fail",
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        command="python train.py",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=1,
            max_gpus=1,
        ),
    )
    store.save_work_item(item)
    pool = CollectFailPool()
    broker = GPUBroker(
        pool=pool,
        store=store,
        scheduler=FactoryScheduler(config, total_gpus=2),
    )
    broker.submit(item)
    pool.finished = True

    changed = broker.reconcile()

    assert changed[0].status.value == "failed"
    assert changed[0].result["failure_reason"] == "GPU_COLLECT_FAILED"
    assert store.list_leases()[0].status.value == "released"


def test_existing_admitted_lease_is_adopted_without_self_starvation(
    tmp_path: Path,
) -> None:
    config = FactoryConfig.from_mapping(
        {"factory": {"scheduler": {"reserved_gpus": 0}}}
    )
    store = FactoryStore(tmp_path)
    store.initialize()
    item = WorkItem(
        item_id="idea-admitted",
        idea_id="idea-admitted",
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        command="python train.py",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=1,
            max_gpus=1,
        ),
    )
    store.save_work_item(item)
    scheduler = FactoryScheduler(config, total_gpus=1)
    lease = scheduler.lease_for(item, 1)
    lease.status = lease.status.ADMITTED
    store.save_leases([lease])
    pool = FakePool()
    broker = GPUBroker(pool=pool, store=store, scheduler=scheduler)

    submission = broker.submit(item)

    assert submission.admitted is True
    assert pool.submitted == [lease.pool_task_id]
    assert store.list_leases()[0].status.value == "running"
    assert item.attempt == 1


def test_existing_admitted_lease_adopts_its_attempt_on_restart(
    tmp_path: Path,
) -> None:
    config = FactoryConfig.from_mapping(
        {"factory": {"scheduler": {"reserved_gpus": 0}}}
    )
    store = FactoryStore(tmp_path)
    store.initialize()
    item = WorkItem(
        item_id="idea-restart-admitted",
        idea_id="idea-restart-admitted",
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        command="python train.py",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=1,
            max_gpus=1,
        ),
    )
    scheduler = FactoryScheduler(config, total_gpus=1)
    item.attempt = 1
    lease = scheduler.lease_for(item, 1)
    lease.status = lease.status.ADMITTED
    store.save_work_item(item)
    store.save_leases([lease])
    pool = FakePool()
    broker = GPUBroker(pool=pool, store=store, scheduler=scheduler)

    submission = broker.submit(item)

    assert submission.admitted is True
    assert item.attempt == 2
    assert pool.submitted == [f"{item.item_id}-attempt-02"]


def test_ambiguous_submit_adopts_live_deterministic_task(
    tmp_path: Path,
) -> None:
    config = FactoryConfig.from_mapping(
        {"factory": {"scheduler": {"reserved_gpus": 0}}}
    )
    store = FactoryStore(tmp_path)
    store.initialize()
    item = WorkItem(
        item_id="idea-ambiguous",
        idea_id="idea-ambiguous",
        kind=WorkKind.GPU_EXPERIMENT,
        profile="pilot",
        command="python train.py",
        resources=ResourceRequest(
            min_gpus=1,
            preferred_gpus=1,
            max_gpus=1,
        ),
    )
    store.save_work_item(item)
    pool = AmbiguousSubmitPool()
    broker = GPUBroker(
        pool=pool,
        store=store,
        scheduler=FactoryScheduler(config, total_gpus=1),
    )

    submission = broker.submit(item)

    assert submission.admitted is True
    assert submission.reason == "SUBMISSION_ADOPTED_AFTER_ERROR"
    assert item.attempt == 1
    assert store.list_leases()[0].status.value == "running"
