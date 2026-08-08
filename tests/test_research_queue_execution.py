from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

from researchclaw.research_queue.execution import (
    ClusterBridgeRunBackend,
    GPUSlotPool,
    LocalRunBackend,
)
from researchclaw.research_queue.models import BudgetLevel, RunRecord


def test_local_backend_runs_real_result_contract(tmp_path) -> None:
    revision = tmp_path / "revision"
    output = tmp_path / "output"
    revision.mkdir()
    (revision / "experiment.py").write_text(
        """
import json, os, pathlib
out = pathlib.Path(os.environ["RESEARCH_QUEUE_OUTPUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(json.dumps({
    "status": "ok",
    "metrics": {"effect": 0.25},
    "artifacts": []
}))
""".strip()
        + "\n",
        encoding="utf-8",
    )
    run = RunRecord(
        run_id="run-local",
        idea_id="idea-local",
        revision=1,
        budget=BudgetLevel.B0,
        requested_gpus=1,
        timeout_sec=10,
        command=(sys.executable, "experiment.py"),
        output_dir=str(output),
    )
    backend = LocalRunBackend(slot_pool=GPUSlotPool(2))

    result = asyncio.run(
        backend.run(
            run,
            revision_dir=revision,
            output_dir=output,
            env={
                "RESEARCH_QUEUE_OUTPUT_DIR": str(output),
                "RESEARCH_QUEUE_BUDGET": "B0",
            },
        )
    )

    assert result.ok
    assert result.metrics["effect"] == 0.25
    assert result.usage["gpu_count"] == 1
    assert json.loads((output / "result.json").read_text())["status"] == "ok"


def test_gpu_slot_pool_bounds_concurrency() -> None:
    async def scenario() -> None:
        pool = GPUSlotPool(2)
        first = await pool.acquire(2)
        started = asyncio.Event()

        async def waiter() -> None:
            lease = await pool.acquire(1)
            started.set()
            await pool.release(lease)

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        assert not started.is_set()
        await pool.release(first)
        await asyncio.wait_for(started.wait(), timeout=1)
        await task

    asyncio.run(scenario())


def test_clusterbridge_backend_routes_parallel_results_to_owning_run(
    tmp_path,
) -> None:
    class FakeBroker:
        def __init__(self) -> None:
            self.submitted: list[str] = []
            self.reconcile_calls = 0

        def submit(self, job, *, priorities):
            del priorities
            self.submitted.append(job.job_id)
            return SimpleNamespace(admitted=True, reason="accepted")

        def reconcile(self):
            self.reconcile_calls += 1
            if len(self.submitted) < 2:
                return []
            completed = [
                (
                    job_id,
                    {
                        "returncode": 0,
                        "marker": job_id,
                    },
                )
                for job_id in reversed(self.submitted)
            ]
            self.submitted.clear()
            return completed

    class FakeManager:
        def __init__(self, broker) -> None:
            self.broker = broker
            self.demands: list[dict[str, object]] = []
            self.closed = False

        def bootstrap(self, *, required_gpus=0):
            self.demands.append({"required_gpus": required_gpus})

        def reconcile(self, **kwargs):
            self.demands.append(dict(kwargs))
            return False

        def close(self):
            self.closed = True

        def snapshot(self):
            return {"closed": self.closed}

    async def scenario() -> None:
        broker = FakeBroker()
        manager = FakeManager(broker)
        backend = ClusterBridgeRunBackend(
            manager=manager,
            slot_pool=GPUSlotPool(2),
            poll_interval_sec=0.001,
        )
        revision = tmp_path / "revision"
        revision.mkdir()
        runs = [
            RunRecord(
                run_id=f"run-{index}",
                idea_id=f"idea-{index}",
                revision=1,
                budget=BudgetLevel.B0,
                requested_gpus=1,
                timeout_sec=10,
                command=(sys.executable, "experiment.py"),
                output_dir=str(tmp_path / f"output-{index}"),
            )
            for index in (1, 2)
        ]
        for run in runs:
            output = tmp_path / f"output-{run.run_id[-1]}"
            output.mkdir()
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "metrics": {"owner": run.run_id},
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )

        results = await asyncio.gather(
            *[
                backend.run(
                    run,
                    revision_dir=revision,
                    output_dir=tmp_path / f"output-{run.run_id[-1]}",
                    env={},
                )
                for run in runs
            ]
        )

        assert [result.metrics["owner"] for result in results] == [
            "run-1",
            "run-2",
        ]
        assert all(result.ok for result in results)
        assert backend.slot_pool.used == 0
        assert backend.snapshot()["routed_results"] == 0
        await backend.close()
        assert manager.closed

    asyncio.run(scenario())
