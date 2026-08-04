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
        self.finished = False

    def submit_task(self, command: str, **kwargs):
        del command
        self.submitted.append(kwargs["task_id"])
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
    assert len(pool.submitted) == 1
    assert broker.reconcile() == []
    pool.finished = True
    changed = broker.reconcile()
    assert changed[0].status.value == "succeeded"
    assert store.list_leases()[0].status.value == "released"
