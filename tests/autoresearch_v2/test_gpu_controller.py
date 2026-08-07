from __future__ import annotations

import json
import sys
from pathlib import Path

from researchclaw.autoresearch_v2.config import V2Config
from researchclaw.autoresearch_v2.controller import V2Controller
from researchclaw.autoresearch_v2.gpu import AdaptiveGPUScheduler, GPUBroker
from researchclaw.autoresearch_v2.ideas import (
    StaticIdeaGenerator,
    candidate_to_idea,
)
from researchclaw.autoresearch_v2.jobs import SimulatedJobExecutor
from researchclaw.autoresearch_v2.models import (
    AttemptStatus,
    IdeaStatus,
    JobKind,
    JobRecord,
    JobStatus,
)
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


class _NoopGenerator:
    def generate(self, *, count, existing):
        del count, existing
        return []


def test_gpu_job_remains_ready_when_configured_pool_is_unavailable(
    tmp_path: Path,
) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path / "state"),
                "population": {
                    "reservoir_low_watermark": 1,
                    "reservoir_target": 1,
                    "generation_batch_size": 1,
                    "active_idea_target": 1,
                    "max_active_ideas": 1,
                    "max_same_family": 1,
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": str(tmp_path / "pool.yaml"),
                    "shared_workspace_root": str(tmp_path),
                },
            }
        }
    )
    store = V2Store(config.root)
    controller = V2Controller(
        config=config,
        store=store,
        generator=_NoopGenerator(),
        gpu_broker=None,
        configured_gpu_capacity=8,
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.PILOTING
    idea.current_job_id = f"{idea.idea_id}-pilot"
    store.save_idea(idea)
    job = JobRecord(
        job_id=idea.current_job_id,
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.READY,
        requires_gpu=True,
        min_gpus=1,
        preferred_gpus=2,
        max_gpus=2,
    )
    store.save_job(job)

    controller._dispatch()

    durable = store.get_job(job.job_id)
    assert durable is not None
    assert durable.status is JobStatus.READY
    assert store.list_attempts(job_id=job.job_id) == []
    snapshot = controller.snapshot()
    assert snapshot["gpu"] == {
        "total_gpus": 8,
        "allocated_gpus": 0,
        "available_gpus": 0,
        "utilization": 0.0,
        "target_utilization": config.gpu.target_utilization,
        "pending_jobs": 1,
        "leases": [],
        "state": "unavailable",
    }
    controller.close()


def test_remote_smoke_infrastructure_classifier_reads_stdout() -> None:
    classify = V2Controller._remote_smoke_infrastructure_code

    assert (
        classify(
            result={"timed_out": True},
            stdout="xet-core CAS client Status Code: 500",
            stderr="",
            returncode=124,
        )
        == "dependency_hub_5xx"
    )
    assert (
        classify(
            result={"timed_out": True},
            stdout="Fetching 4 files from huggingface",
            stderr="",
            returncode=124,
        )
        == "dependency_download_timeout"
    )
    assert (
        classify(
            result={},
            stdout="",
            stderr="LocalEntryNotFoundError: offline cache miss",
            returncode=1,
        )
        == "dependency_cache_miss"
    )
    assert (
        classify(
            result={"pool_state": "probe_failed"},
            stdout="",
            stderr="",
            returncode=-1,
        )
        == "gpu_task_lost"
    )
    assert (
        classify(
            result={"pool_state": "lost", "error": "remote task lost"},
            stdout="",
            stderr="LocalEntryNotFoundError: offline cache miss",
            returncode=1,
        )
        == "dependency_cache_miss"
    )


def test_controller_runs_build_smoke_on_gpu_pool_before_pilot(
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
                "concurrency": {
                    "max_llm_jobs": 1,
                    "max_cpu_jobs": 1,
                    "max_gpu_jobs": 1,
                    "poll_interval_sec": 0.001,
                },
                "execution": {
                    "python_executable": sys.executable,
                    "smoke_environment": "gpu_pool",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
                    "pilot_max_gpus": 1,
                    "scale_max_gpus": 1,
                },
            }
        }
    )

    class SmokeAwarePool(_Pool):
        def collect_task(self, task_id: str):
            request = self.requests[task_id]
            output_dir = Path(
                str(
                    request.get("env", {}).get(
                        "AUTORESEARCH_V2_OUTPUT_DIR",
                        "",
                    )
                )
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            if output_dir.name != "smoke":
                return super().collect_task(task_id)
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "result_valid": True,
                        "metrics": {"smoke_forward_pass": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            (output_dir / "runtime_evidence.json").write_text(
                json.dumps(
                    {
                        "model_loaded": "Qwen-test",
                        "datasets_loaded": ["GSM8K"],
                        "examples_processed": 1,
                        "seeds": [0],
                        "gpu_count": 1,
                        "gate_decision": "promote",
                        "metrics": {"smoke_forward_pass": 1.0},
                        "cuda_executed": True,
                        "gpu_uuid": "GPU-test-uuid",
                        "peak_gpu_memory_mb": 128.0,
                    }
                ),
                encoding="utf-8",
            )
            return {
                "returncode": 0,
                "elapsed_sec": 1.0,
                "stdout": "smoke-ok\n",
                "stderr": "",
                "trusted_gpu_evidence": {
                    "schema": "autoresearch_v2.trusted_gpu_evidence",
                    "version": 1,
                    "task_id": task_id,
                    "ray_task_id": "ray-task-test",
                    "ray_node_id": "ray-node-test",
                    "ray_actor_id": "",
                    "hostname": "gpu-node",
                    "cuda_visible_devices": "0",
                    "allocated_gpus": 1,
                    "returncode": 0,
                    "gpu_uuids": ["GPU-test-uuid"],
                    "gpu_names": ["Test GPU"],
                    "peak_gpu_memory_mb": 128.0,
                    "peak_gpu_utilization_percent": 50.0,
                    "samples": [],
                },
            }

    store = V2Store(tmp_path)
    pool = SmokeAwarePool()
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
            pool=pool,
            scheduler=AdaptiveGPUScheduler(total_gpus=1),
        ),
        sleep=lambda _: None,
    )
    controller.initialize()

    import time

    for _ in range(120):
        controller.tick()
        time.sleep(0.002)
        pilot_requests = [
            request
            for request in pool.requests.values()
            if Path(
                str(
                    request.get("env", {}).get(
                        "AUTORESEARCH_V2_OUTPUT_DIR",
                        "",
                    )
                )
            ).name
            == "pilot"
        ]
        if pilot_requests:
            break

    requests = list(pool.requests.values())
    output_names = [
        Path(
            str(
                request.get("env", {}).get(
                    "AUTORESEARCH_V2_OUTPUT_DIR",
                    "",
                )
            )
        ).name
        for request in requests
    ]
    assert "smoke" in output_names
    assert "pilot" in output_names
    smoke_index = output_names.index("smoke")
    pilot_index = output_names.index("pilot")
    assert smoke_index < pilot_index
    smoke_request = requests[smoke_index]
    assert sys.executable in str(smoke_request["command"])
    assert "shell=False" in str(smoke_request["command"])

    builds = [
        job
        for job in store.list_jobs()
        if job.kind is JobKind.BUILD and job.result.get("remote_smoke")
    ]
    assert len(builds) == 1
    assert builds[0].status.value == "succeeded"
    smoke_attempts = [
        attempt
        for job in builds
        for attempt in store.list_attempts(job_id=job.job_id)
        if attempt.status.value == "accepted"
    ]
    assert len(smoke_attempts) == 1
    assert smoke_attempts[0].validation["remote_smoke"]["verified"] is True
    assert smoke_attempts[0].validation["remote_smoke"][
        "attestation_sha256"
    ]
    trusted_path = Path(
        smoke_attempts[0].validation["remote_smoke"][
            "trusted_gpu_evidence_path"
        ]
    )
    assert trusted_path.is_file()
    assert json.loads(trusted_path.read_text())["gpu_uuids"] == [
        "GPU-test-uuid"
    ]
    smoke_contracts = []
    for path in tmp_path.rglob("execution_contract.json"):
        contract = json.loads(path.read_text(encoding="utf-8"))
        if contract.get("mode") == "smoke":
            smoke_contracts.append(contract)
    assert len(smoke_contracts) == 1
    assert smoke_contracts[0]["resource_limits"]["min_gpus"] == 1
    controller._pool.shutdown(wait=True)


def test_remote_smoke_rejects_missing_trusted_gpu_telemetry(
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
                "concurrency": {
                    "max_llm_jobs": 1,
                    "max_cpu_jobs": 1,
                    "max_gpu_jobs": 1,
                    "poll_interval_sec": 0.001,
                },
                "execution": {
                    "python_executable": sys.executable,
                    "smoke_environment": "gpu_pool",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
                    "pilot_max_gpus": 1,
                    "scale_max_gpus": 1,
                },
            }
        }
    )

    class UntrustedOnlyPool(_Pool):
        def collect_task(self, task_id: str):
            request = self.requests[task_id]
            output_dir = Path(
                str(
                    request.get("env", {}).get(
                        "AUTORESEARCH_V2_OUTPUT_DIR",
                        "",
                    )
                )
            )
            if output_dir.name != "smoke":
                return super().collect_task(task_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "result_valid": True,
                        "metrics": {"smoke_forward_pass": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            # Generated code can claim perfect GPU evidence; without the
            # trusted Ray wrapper telemetry the Controller must reject it.
            (output_dir / "runtime_evidence.json").write_text(
                json.dumps(
                    {
                        "model_loaded": "Qwen-test",
                        "datasets_loaded": ["GSM8K"],
                        "examples_processed": 1,
                        "seeds": [0],
                        "gpu_count": 1,
                        "gate_decision": "promote",
                        "metrics": {"smoke_forward_pass": 1.0},
                        "cuda_executed": True,
                        "gpu_uuid": "GPU-forged",
                        "peak_gpu_memory_mb": 9999.0,
                    }
                ),
                encoding="utf-8",
            )
            return {
                "returncode": 0,
                "elapsed_sec": 1.0,
                "stdout": "claimed-ok\n",
                "stderr": "",
            }

    store = V2Store(tmp_path)
    pool = UntrustedOnlyPool()
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
            pool=pool,
            scheduler=AdaptiveGPUScheduler(total_gpus=1),
        ),
        sleep=lambda _: None,
    )
    controller.initialize()

    import time

    for _ in range(120):
        controller.tick()
        time.sleep(0.002)
        smoke_jobs = [
            job
            for job in store.list_jobs()
            if job.kind is JobKind.BUILD
            and job.result.get("remote_smoke")
        ]
        if smoke_jobs and smoke_jobs[0].status is JobStatus.FAILED:
            break

    smoke_jobs = [
        job
        for job in store.list_jobs()
        if job.kind is JobKind.BUILD and job.result.get("remote_smoke")
    ]
    assert len(smoke_jobs) == 1
    assert smoke_jobs[0].status is JobStatus.FAILED
    attempts = store.list_attempts(job_id=smoke_jobs[0].job_id)
    assert attempts
    assert any(
        "trusted GPU evidence" in error
        for attempt in attempts
        for error in attempt.validation["errors"]
    )
    assert not any(
        Path(
            str(
                request.get("env", {}).get(
                    "AUTORESEARCH_V2_OUTPUT_DIR",
                    "",
                )
            )
        ).name
        == "pilot"
        for request in pool.requests.values()
    )
    controller._pool.shutdown(wait=True)


def test_unchanged_remote_smoke_repair_is_blocked_before_gpu_submit(
    tmp_path: Path,
) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
                },
            }
        }
    )
    store = V2Store(tmp_path)
    pool = _Pool()
    controller = V2Controller(
        config=config,
        store=store,
        generator=_NoopGenerator(),
        gpu_broker=GPUBroker(
            pool=pool,
            scheduler=AdaptiveGPUScheduler(total_gpus=1),
        ),
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.PILOTING
    current = store.current_dir(idea.idea_id)
    current.mkdir(parents=True, exist_ok=True)
    (current / "plan.json").write_text(
        json.dumps({"required_runtime_evidence": []}),
        encoding="utf-8",
    )
    (current / "main.py").write_text("print('same source')\n")
    (current / "build.json").write_text(
        json.dumps(
            {
                "files": {"main.py": "print('same source')\n"},
                "commands": {
                    "smoke": ["python", "main.py"],
                    "pilot": ["python", "main.py"],
                    "scale": ["python", "main.py"],
                },
            }
        ),
        encoding="utf-8",
    )
    prior = controller._implementation_failure_fingerprint(
        idea_id=idea.idea_id,
        errors=["same runtime failure"],
    )
    idea.candidate["_autoresearch_v2_last_implementation_failure"] = {
        **prior,
        "errors": ["same runtime failure"],
    }
    job = JobRecord(
        job_id=f"{idea.idea_id}-pilot",
        idea_id=idea.idea_id,
        kind=JobKind.BUILD,
        status=JobStatus.READY,
        requires_gpu=True,
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
        result={"remote_smoke": True, "next_kind": "pilot"},
    )
    idea.current_job_id = job.job_id
    store.save_idea(idea)
    store.save_job(job)

    controller._dispatch()

    durable_idea = store.get_idea(idea.idea_id)
    durable_job = store.get_job(job.job_id)
    assert durable_idea is not None
    assert durable_job is not None
    assert durable_idea.status is IdeaStatus.QUARANTINED
    assert durable_idea.exit_reason == "remote_smoke_no_progress"
    assert durable_job.status is JobStatus.FAILED
    assert durable_job.result["diagnostics"][
        "blocked_before_gpu_submit"
    ] is True
    assert store.list_attempts(job_id=job.job_id) == []
    assert pool.requests == {}
    controller.close()


def test_unchanged_pilot_contract_failure_is_blocked_before_gpu_submit(
    tmp_path: Path,
) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
                },
            }
        }
    )
    store = V2Store(tmp_path)
    pool = _Pool()
    controller = V2Controller(
        config=config,
        store=store,
        generator=_NoopGenerator(),
        gpu_broker=GPUBroker(
            pool=pool,
            scheduler=AdaptiveGPUScheduler(total_gpus=1),
        ),
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.PILOTING
    current = store.current_dir(idea.idea_id)
    current.mkdir(parents=True, exist_ok=True)
    (current / "plan.json").write_text(
        json.dumps({"required_runtime_evidence": []}),
        encoding="utf-8",
    )
    (current / "main.py").write_text("print('same pilot source')\n")
    (current / "build.json").write_text(
        json.dumps(
            {
                "files": {"main.py": "print('same pilot source')\n"},
                "commands": {
                    "smoke": ["python", "main.py"],
                    "pilot": ["python", "main.py"],
                    "scale": ["python", "main.py"],
                },
            }
        ),
        encoding="utf-8",
    )
    prior = controller._implementation_failure_fingerprint(
        idea_id=idea.idea_id,
        errors=["same pilot runtime contract failure"],
    )
    idea.candidate["_autoresearch_v2_last_implementation_failure"] = {
        **prior,
        "errors": ["same pilot runtime contract failure"],
    }
    job = JobRecord(
        job_id=f"{idea.idea_id}-pilot",
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.READY,
        requires_gpu=True,
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
    )
    idea.current_job_id = job.job_id
    store.save_idea(idea)
    store.save_job(job)

    controller._dispatch()

    durable_idea = store.get_idea(idea.idea_id)
    durable_job = store.get_job(job.job_id)
    assert durable_idea is not None
    assert durable_job is not None
    assert durable_idea.status is IdeaStatus.QUARANTINED
    assert durable_idea.exit_reason == "gpu_implementation_no_progress"
    assert durable_job.status is JobStatus.FAILED
    assert durable_job.result["diagnostics"][
        "blocked_before_gpu_submit"
    ] is True
    assert store.list_attempts(job_id=job.job_id) == []
    assert pool.requests == {}
    controller.close()


def test_attestation_key_creation_is_race_safe(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "shared-controller.key"
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path / "state"),
                "execution": {
                    "attestation_key_file": str(key_path),
                },
            }
        }
    )
    controllers = [
        V2Controller(
            config=config,
            store=V2Store(tmp_path / f"state-{index}"),
            generator=StaticIdeaGenerator([]),
            executors={},
        )
        for index in range(2)
    ]

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        keys = list(
            executor.map(
                lambda controller: controller._attestation_key(),
                controllers,
            )
        )

    assert keys[0] == keys[1]
    assert len(keys[0]) >= 32
    assert key_path.is_file()
    for controller in controllers:
        controller._pool.shutdown(wait=True)


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
                "execution": {
                    "smoke_environment": "local",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
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
        task_namespace="controller-namespace-test",
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
    for _ in range(200):
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
    assert all(task_id.startswith("con-") for task_id in pilot_requests)
    assert sum(
        int(request["num_gpus"]) for request in pilot_requests.values()
    ) == 6
    for request in pilot_requests.values():
        assert request["command"]
        assert "execution_contract.json" in str(request["command"])
        assert "shell=False" in str(request["command"])
    for task_id in pilot_requests:
        job = next(
            job
            for job in store.list_jobs()
            if job.submitted_task_id == task_id
        )
        assert job.result["task_id"] == task_id
        assert any(
            event["event_type"] == "gpu_job_submitted"
            and event["job_id"] == job.job_id
            and event["task_id"] == task_id
            for event in store.list_events(limit=5000)
        )
    contract_paths = list(
        tmp_path.rglob("execution_contract.json")
    )
    assert len(contract_paths) >= 3
    contract = json.loads(contract_paths[0].read_text(encoding="utf-8"))
    assert isinstance(contract["argv"], list)
    assert contract["argv"][0] == "python"
    assert contract["argv"][1].endswith(".py")
    assert "python -c" in str(next(iter(pilot_requests.values()))["command"])
    controller.tick()
    assert any(
        idea.status in {IdeaStatus.SCALING, IdeaStatus.REPORTING}
        for idea in store.list_ideas()
    )
    controller._pool.shutdown(wait=True)


def test_elastic_recovery_defers_gpu_adoption_until_broker_attaches(
    tmp_path: Path,
) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "gpu": {
                    "enabled": True,
                    "mode": "resource_manager",
                    "shared_workspace_root": str(tmp_path),
                    "resource_manager": {
                        "owner": "test-owner",
                    },
                },
            }
        }
    )
    store = V2Store(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.PILOTING
    idea.current_job_id = f"{idea.idea_id}-pilot"
    store.save_idea(idea)
    job = JobRecord(
        job_id=idea.current_job_id,
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.RUNNING,
        attempt=1,
        requires_gpu=True,
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
        attempt_id=f"{idea.current_job_id}-attempt-01",
        submitted_task_id="legacy-running-task",
        result={"allocated_gpus": 1},
    )
    store.save_job(job)

    class DelayedManager:
        broker = None
        configured_capacity = 1

        def reconcile(self) -> bool:
            return False

        def snapshot(self) -> dict[str, object]:
            return {}

    manager = DelayedManager()
    controller = V2Controller(
        config=config,
        store=store,
        generator=_NoopGenerator(),
        gpu_manager=manager,
        sleep=lambda _: None,
    )
    controller.initialize()

    durable = store.get_job(job.job_id)
    assert durable is not None
    assert durable.status is JobStatus.RUNNING
    assert durable.submitted_task_id == "legacy-running-task"
    assert any(
        event["event_type"] == "gpu_job_adoption_deferred"
        for event in store.list_events()
    )

    pool = _Pool()
    broker = GPUBroker(
        pool=pool,
        scheduler=AdaptiveGPUScheduler(total_gpus=1),
        task_namespace="new-run",
    )
    manager.broker = broker
    controller._reconcile_gpu_manager()

    assert broker.leases[job.job_id].task_id == "legacy-running-task"
    assert any(
        event["event_type"] == "gpu_job_adopted"
        and event.get("reconnect") is True
        for event in store.list_events()
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
                "execution": {
                    "smoke_environment": "local",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
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
                "execution": {
                    "smoke_environment": "local",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
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


def test_gpu_submission_failure_does_not_exhaust_scientific_attempts(
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
                "execution": {
                    "smoke_environment": "local",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
                },
                "budgets": {
                    "max_job_attempts": 1,
                },
            }
        }
    )

    class SubmissionFailurePool(_Pool):
        def submit_task(self, command: str, **kwargs):
            del command, kwargs
            raise RuntimeError("temporary resource-manager outage")

        def probe_task(self, task_id: str):
            del task_id
            raise RuntimeError("task was never created")

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
            pool=SubmissionFailurePool(),
            scheduler=AdaptiveGPUScheduler(total_gpus=1),
        ),
        sleep=lambda _: None,
    )
    controller.initialize()
    import time

    for _ in range(100):
        controller.tick()
        time.sleep(0.002)
        gpu_jobs = [
            job for job in store.list_jobs() if job.requires_gpu
        ]
        if (
            gpu_jobs
            and gpu_jobs[0].status is JobStatus.RETRY_WAIT
            and gpu_jobs[0].result.get("reason") == "gpu_submission_failed"
        ):
            break

    job = next(item for item in store.list_jobs() if item.requires_gpu)
    idea = store.get_idea(job.idea_id)
    assert idea is not None
    assert job.status is JobStatus.RETRY_WAIT
    assert idea.status is IdeaStatus.PILOTING
    assert job.result["failure_class"] == "infrastructure_transient"
    assert job.result["consume_attempt"] is False
    assert job.result["infrastructure_retries"] == 1
    assert job.attempt_limit == 2
    controller.close()


def test_runtime_contract_failure_with_raw_artifacts_returns_to_build(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "execution": {
                    "smoke_environment": "local",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(tmp_path),
                },
            }
        }
    )
    class ContractFailurePool(_Pool):
        def collect_task(self, task_id: str):
            del task_id
            return {
                "returncode": 1,
                "elapsed_sec": 1.0,
                "stdout": "",
                "stderr": "runtime contract failed",
            }

    store = V2Store(tmp_path)
    pool = ContractFailurePool()
    controller = V2Controller(
        config=config,
        store=store,
        generator=_NoopGenerator(),
        gpu_broker=GPUBroker(
            pool=pool,
            scheduler=AdaptiveGPUScheduler(total_gpus=1),
        ),
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.PILOTING
    current = store.current_dir(idea.idea_id)
    current.mkdir(parents=True, exist_ok=True)
    (current / "plan.json").write_text(
        json.dumps({"required_runtime_evidence": []}),
        encoding="utf-8",
    )
    (current / "main.py").write_text("print('pilot')\n", encoding="utf-8")
    (current / "build.json").write_text(
        json.dumps(
            {
                "files": {"main.py": "print('pilot')\n"},
                "commands": {
                    "smoke": ["python", "main.py"],
                    "pilot": ["python", "main.py"],
                    "scale": ["python", "main.py"],
                },
            }
        ),
        encoding="utf-8",
    )
    job = JobRecord(
        job_id=f"{idea.idea_id}-pilot",
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.RUNNING,
        requires_gpu=True,
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
        result={"allocated_gpus": 1},
    )
    attempt = store.create_attempt(job)
    attempt.status = AttemptStatus.RUNNING
    job.attempt = attempt.number
    job.attempt_id = attempt.attempt_id
    idea.current_job_id = job.job_id
    candidate = store.snapshot_current(attempt)
    output_dir = candidate / "artifacts" / "pilot"
    raw_dir = output_dir / "_raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "metrics.json").write_text("{}\n", encoding="utf-8")
    (raw_dir / "runtime_evidence.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    job.expected_output_dir = str(output_dir)
    store.save_idea(idea)
    store.save_job(job)
    store.save_attempt(attempt)
    job.submitted_task_id = "task-runtime-contract"
    store.save_job(job)
    controller.gpu_broker.adopt(
        job,
        task_id="task-runtime-contract",
        allocated_gpus=1,
    )
    monkeypatch.setattr(
        controller,
        "_normalize_controller_runtime_artifacts",
        lambda **kwargs: {"error": "deterministic runtime contract failure"},
    )
    monkeypatch.setattr(
        "researchclaw.autoresearch_v2.controller.validate_experiment_artifacts",
        lambda output: {
            "ok": False,
            "errors": ["missing canonical runtime"],
            "metrics": {},
            "runtime_evidence": {},
            "files": [],
        },
    )
    monkeypatch.setattr(
        controller,
        "_attest_gpu_execution",
        lambda **kwargs: {"errors": []},
    )
    monkeypatch.setattr(
        "researchclaw.autoresearch_v2.controller."
        "validate_runtime_against_contract",
        lambda **kwargs: [],
    )

    controller._collect_gpu_finished()

    durable_idea = store.get_idea(idea.idea_id)
    durable_job = store.get_job(job.job_id)
    assert durable_idea is not None
    assert durable_job is not None
    assert durable_idea.status is IdeaStatus.BUILDING
    assert durable_job.status is JobStatus.FAILED
    assert durable_job.result["decision"] == "repair_build"
    assert durable_job.result["reason"] == "pilot_runtime_contract_invalid"
    assert any(
        event["event_type"] == "gpu_implementation_returned_to_build"
        for event in store.list_events(limit=50)
    )
    controller.close()
