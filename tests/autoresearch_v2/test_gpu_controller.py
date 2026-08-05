from __future__ import annotations

import json
from pathlib import Path

from researchclaw.autoresearch_v2.config import V2Config
from researchclaw.autoresearch_v2.controller import V2Controller
from researchclaw.autoresearch_v2.gpu import AdaptiveGPUScheduler, GPUBroker
from researchclaw.autoresearch_v2.ideas import StaticIdeaGenerator
from researchclaw.autoresearch_v2.jobs import SimulatedJobExecutor
from researchclaw.autoresearch_v2.models import IdeaStatus, JobKind
from researchclaw.autoresearch_v2.store import V2Store


def _candidate(index: int) -> dict[str, object]:
    return {
        "id": f"gpu-{index}",
        "title": f"GPU mechanism experiment {index}",
        "family": f"family-{index}",
        "research_question": "Does it work?",
        "falsifiable_hypothesis": "It improves accuracy.",
        "closest_prior_work": ["self refinement"],
        "novelty_gap": "mechanism gap",
        "datasets": ["GSM8K"],
        "models": ["Qwen"],
        "compute": {"gpu_count": 2, "wall_clock_hours": 0.1},
        "primary_metric": "accuracy",
        "baselines": ["single-pass no-self-improvement"],
        "ablations": ["remove mechanism"],
        "failure_safety_tests": ["leakage"],
        "implementation_feasibility": "public stack",
        "licensing_feasibility": "verify",
        "information_gain_if_true": "positive",
        "information_gain_if_false": "negative",
        "cheap_pilot": "two GPUs",
        "scores": {
            "novelty": 8,
            "scientific_importance": 8,
            "falsifiability": 9,
            "compute_tractability": 9,
            "reproducibility": 9,
            "meaningful_result_likelihood": 8,
            "risk": 2,
        },
    }


class _Pool:
    def __init__(self) -> None:
        self.requests: dict[str, dict[str, object]] = {}

    def submit_task(self, command: str, **kwargs):
        self.requests[kwargs["task_id"]] = {
            "command": command,
            **kwargs,
        }
        return {"task_id": kwargs["task_id"]}

    def probe_task(self, task_id: str):
        return {"state": "finished"}

    def collect_task(self, task_id: str):
        request = self.requests[task_id]
        output_dir = Path(
            str(request.get("env", {}).get("AUTORESEARCH_V2_OUTPUT_DIR", ""))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "result_valid": True,
                    "metrics": {"accuracy": 0.7},
                    "success_probability": 0.99,
                    "decision": "promote",
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "runtime_evidence.json").write_text(
            json.dumps(
                {
                    "model_loaded": "Qwen-test",
                    "datasets_loaded": ["GSM8K"],
                    "examples_processed": 10,
                    "seeds": [0],
                    "gpu_count": int(request["num_gpus"]),
                    "gate_decision": "promote",
                    "metrics": {"accuracy": 0.7},
                }
            ),
            encoding="utf-8",
        )
        return {"returncode": 0, "elapsed_sec": 10.0}

    def cancel_task(self, task_id: str):
        return {"returncode": 130, "elapsed_sec": 1.0}


def test_controller_submits_multiple_gpu_ideas_into_one_pool(
    tmp_path: Path,
) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "population": {
                    "reservoir_low_watermark": 1,
                    "reservoir_target": 3,
                    "generation_batch_size": 3,
                    "active_idea_target": 3,
                    "max_active_ideas": 3,
                    "max_same_family": 2,
                },
                "concurrency": {
                    "max_llm_jobs": 3,
                    "max_cpu_jobs": 3,
                    "max_gpu_jobs": 3,
                    "poll_interval_sec": 0.001,
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "reserved_gpus": 0,
                    "pilot_max_gpus": 2,
                    "scale_max_gpus": 2,
                },
            }
        }
    )
    store = V2Store(tmp_path)
    pool = _Pool()
    broker = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(
            total_gpus=6,
            max_share_per_idea=0.5,
        ),
    )
    controller = V2Controller(
        config=config,
        store=store,
        generator=StaticIdeaGenerator([_candidate(i) for i in range(3)]),
        executors={
            JobKind.DESIGN: SimulatedJobExecutor(),
            JobKind.BUILD: SimulatedJobExecutor(),
            JobKind.REPORT: SimulatedJobExecutor(),
        },
        gpu_broker=broker,
        sleep=lambda _: None,
    )
    controller.initialize()
    for _ in range(80):
        controller.tick()
        import time

        time.sleep(0.002)
        if sum("-pilot-attempt-" in task_id for task_id in pool.requests) >= 3:
            break

    pilot_requests = {
        task_id: request
        for task_id, request in pool.requests.items()
        if "-pilot-attempt-" in task_id
    }
    assert len(pilot_requests) == 3
    assert sum(
        int(request["num_gpus"]) for request in pilot_requests.values()
    ) == 6
    for request in pilot_requests.values():
        assert request["command"]
    controller.tick()
    assert any(
        idea.status in {IdeaStatus.SCALING, IdeaStatus.REPORTING}
        for idea in store.list_ideas()
    )
    controller._pool.shutdown(wait=True)


def test_controller_enforces_max_gpu_jobs(tmp_path: Path) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "population": {
                    "reservoir_low_watermark": 1,
                    "reservoir_target": 3,
                    "generation_batch_size": 3,
                    "active_idea_target": 3,
                    "max_active_ideas": 3,
                    "max_same_family": 2,
                },
                "concurrency": {
                    "max_llm_jobs": 3,
                    "max_cpu_jobs": 3,
                    "max_gpu_jobs": 1,
                    "poll_interval_sec": 0.001,
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "pilot_max_gpus": 2,
                    "scale_max_gpus": 2,
                },
            }
        }
    )

    class RunningPool(_Pool):
        def probe_task(self, task_id: str):
            del task_id
            return {"state": "running"}

    store = V2Store(tmp_path)
    pool = RunningPool()
    controller = V2Controller(
        config=config,
        store=store,
        generator=StaticIdeaGenerator([_candidate(i) for i in range(3)]),
        executors={
            JobKind.DESIGN: SimulatedJobExecutor(),
            JobKind.BUILD: SimulatedJobExecutor(),
            JobKind.REPORT: SimulatedJobExecutor(),
        },
        gpu_broker=GPUBroker(
            pool=pool,
            scheduler=AdaptiveGPUScheduler(total_gpus=6),
        ),
        sleep=lambda _: None,
    )
    controller.initialize()
    import time

    for _ in range(80):
        controller.tick()
        time.sleep(0.002)
        if pool.requests:
            break
    for _ in range(5):
        controller.tick()
    assert len(pool.requests) == 1
    assert controller.snapshot()["running_gpu_jobs"] == 1
    controller._pool.shutdown(wait=True)


def test_missing_gpu_artifacts_retry_without_mutating_current(
    tmp_path: Path,
) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "population": {
                    "reservoir_low_watermark": 1,
                    "reservoir_target": 1,
                    "generation_batch_size": 1,
                    "active_idea_target": 1,
                    "max_active_ideas": 1,
                    "max_same_family": 2,
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                },
                "budgets": {
                    "max_job_attempts": 3,
                },
            }
        }
    )

    class MissingArtifactPool(_Pool):
        def collect_task(self, task_id: str):
            del task_id
            return {"returncode": 0, "elapsed_sec": 1.0}

    store = V2Store(tmp_path)
    controller = V2Controller(
        config=config,
        store=store,
        generator=StaticIdeaGenerator([_candidate(0)]),
        executors={
            JobKind.DESIGN: SimulatedJobExecutor(),
            JobKind.BUILD: SimulatedJobExecutor(),
            JobKind.REPORT: SimulatedJobExecutor(),
        },
        gpu_broker=GPUBroker(
            pool=MissingArtifactPool(),
            scheduler=AdaptiveGPUScheduler(total_gpus=2),
        ),
        sleep=lambda _: None,
    )
    original_dispatch = controller._dispatch
    controller.initialize()
    import time

    pilot_submitted = False
    for _ in range(80):
        controller._dispatch = (
            (lambda: None) if pilot_submitted else original_dispatch
        )
        controller.tick()
        pilot_jobs = [
            job
            for job in store.list_jobs()
            if job.kind is JobKind.PILOT
        ]
        if pilot_jobs and pilot_jobs[0].status.value == "running":
            pilot_submitted = True
        time.sleep(0.002)
        if (
            pilot_jobs
            and pilot_jobs[0].status.value == "retry_wait"
            and pilot_jobs[0].attempt == 1
        ):
            break
    idea = store.list_ideas()[0]
    assert idea.status is IdeaStatus.PILOTING
    assert not (
        store.current_dir(idea.idea_id) / "artifacts" / "pilot"
    ).exists()
    controller._pool.shutdown(wait=True)
