from __future__ import annotations

import time
from pathlib import Path

from researchclaw.autoresearch_v2.config import V2Config
from researchclaw.autoresearch_v2.controller import V2Controller
from researchclaw.autoresearch_v2.ideas import StaticIdeaGenerator
from researchclaw.autoresearch_v2.jobs import JobOutcome, SimulatedJobExecutor
from researchclaw.autoresearch_v2.models import IdeaStatus, JobKind, JobStatus
from researchclaw.autoresearch_v2.store import V2Store


def _candidate(index: int) -> dict[str, object]:
    family = ["calibration", "memory", "verifier", "population"][index % 4]
    return {
        "id": f"idea-{index}",
        "title": f"Distinct {family} mechanism study {index}",
        "family": family,
        "research_question": f"Does mechanism {index} improve RSI?",
        "falsifiable_hypothesis": f"Mechanism {index} improves accuracy.",
        "closest_prior_work": ["self-refinement"],
        "novelty_gap": f"Gap {index}",
        "datasets": ["GSM8K"],
        "models": ["Qwen2.5-7B"],
        "compute": {"gpu_count": 1, "wall_clock_hours": 0.1},
        "primary_metric": "accuracy",
        "baselines": ["single-pass no-self-improvement"],
        "ablations": ["remove mechanism"],
        "failure_safety_tests": ["leakage"],
        "implementation_feasibility": "public stack",
        "licensing_feasibility": "verify exact licenses",
        "information_gain_if_true": "mechanism works",
        "information_gain_if_false": "mechanism is ruled out",
        "cheap_pilot": "one GPU, ten examples",
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


def _config(root: Path) -> V2Config:
    return V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(root),
                "population": {
                    "reservoir_low_watermark": 2,
                    "reservoir_target": 8,
                    "generation_batch_size": 6,
                    "active_idea_target": 4,
                    "max_active_ideas": 6,
                    "max_same_family": 2,
                },
                "concurrency": {
                    "max_llm_jobs": 4,
                    "max_cpu_jobs": 6,
                    "max_gpu_jobs": 4,
                    "poll_interval_sec": 0.001,
                },
            }
        }
    )


def test_multi_idea_closed_loop_completes_concurrently(tmp_path: Path) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(i) for i in range(8)]),
        executors={kind: SimulatedJobExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.run(max_ticks=200)
    snapshot = controller.snapshot()

    assert snapshot["ideas_by_status"].get("completed", 0) >= 3
    assert (
        snapshot["ideas_by_status"].get("completed", 0)
        + snapshot["ideas_by_status"].get("reporting", 0)
        >= 4
    )
    assert snapshot["jobs_by_status"].get("succeeded", 0) >= 19
    assert all(
        (controller.store.current_dir(idea.idea_id) / "paper.md").is_file()
        for idea in controller.store.list_ideas()
        if idea.status is IdeaStatus.COMPLETED
    )


class _SlowExecutor(SimulatedJobExecutor):
    def execute(self, **kwargs):
        time.sleep(0.05)
        return super().execute(**kwargs)


class _VerySlowExecutor(SimulatedJobExecutor):
    def execute(self, **kwargs):
        time.sleep(2)
        return super().execute(**kwargs)


class _InterruptedExecutor:
    def execute(self, **kwargs):
        del kwargs
        time.sleep(0.05)
        return JobOutcome(False, "retry", "interrupted fixture", {})


def test_controller_reaches_configured_parallelism(tmp_path: Path) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(i) for i in range(8)]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.tick()
    assert controller.snapshot()["running_futures"] == 4
    controller.request_stop()
    controller._pool.shutdown(wait=True)


class _AlwaysFail:
    def execute(self, **kwargs):
        del kwargs
        return JobOutcome(False, "retry", "deterministic failure", {})


def test_bounded_retry_quarantines_one_idea_without_stopping_others(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    controller = V2Controller(
        config=config,
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(i) for i in range(4)]),
        executors={
            JobKind.DESIGN: SimulatedJobExecutor(),
            JobKind.BUILD: _AlwaysFail(),
            JobKind.PILOT: SimulatedJobExecutor(),
            JobKind.SCALE: SimulatedJobExecutor(),
            JobKind.REPORT: SimulatedJobExecutor(),
        },
        sleep=lambda _: None,
    )
    controller.run(max_ticks=25)
    ideas = controller.store.list_ideas()

    assert any(idea.status is IdeaStatus.QUARANTINED for idea in ideas)
    assert all(
        job.attempt <= job.attempt_limit
        for job in controller.store.list_jobs()
        if job.status is JobStatus.FAILED
    )


def test_restart_recovers_running_job_without_spending_retry_budget(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    controller = V2Controller(
        config=_config(tmp_path),
        store=store,
        generator=StaticIdeaGenerator([_candidate(1)]),
        executors={kind: _InterruptedExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.tick()
    running = store.list_jobs(statuses={JobStatus.RUNNING})
    assert running
    controller._pool.shutdown(wait=True)

    restarted = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        sleep=lambda _: None,
    )
    restarted.initialize()
    recovered = restarted.store.get_job(running[0].job_id)
    assert recovered is not None
    assert recovered.status is JobStatus.RETRY_WAIT
    assert recovered.attempt == 0
    assert recovered.attempt_id == ""
    attempt = restarted.store.get_attempt(running[0].attempt_id)
    assert attempt is not None
    assert attempt.status.value == "failed"
    assert attempt.error == "controller_interrupted"
    restarted._pool.shutdown(wait=True)


def test_repeated_restart_interruption_never_quarantines_or_spends_budget(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    controller = V2Controller(
        config=_config(tmp_path),
        store=store,
        generator=StaticIdeaGenerator([_candidate(2)]),
        executors={kind: _InterruptedExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.tick()
    job = store.list_jobs(statuses={JobStatus.RUNNING})[0]
    job.attempt = job.attempt_limit
    attempt = store.get_attempt(job.attempt_id)
    assert attempt is not None
    attempt.number = job.attempt_limit
    store.save_attempt(attempt)
    store.save_job(job)
    controller._pool.shutdown(wait=True)

    for _ in range(3):
        restarted = V2Controller(
            config=_config(tmp_path),
            store=V2Store(tmp_path),
            generator=StaticIdeaGenerator([]),
            executors={
                kind: _InterruptedExecutor()
                for kind in JobKind
            },
            sleep=lambda _: None,
        )
        restarted.initialize()
        recovered = restarted.store.get_job(job.job_id)
        idea = restarted.store.get_idea(job.idea_id)
        assert recovered is not None
        assert recovered.status is JobStatus.RETRY_WAIT
        assert recovered.attempt == job.attempt_limit - 1
        assert idea is not None
        assert idea.status is IdeaStatus.DESIGNING
        assert idea.current_job_id == job.job_id
        assert idea.exit_reason == ""

        # Simulate another process interruption after the refunded attempt is
        # re-dispatched. Every restart must preserve the same scientific count.
        restarted.tick()
        running = restarted.store.get_job(job.job_id)
        assert running is not None and running.status is JobStatus.RUNNING
        assert running.attempt == job.attempt_limit
        job = running
        restarted._pool.shutdown(wait=True)


def test_request_stop_prevents_new_admission_and_drains_started_work(
    tmp_path: Path,
) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(i) for i in range(8)]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.tick()
    started = {
        job.job_id
        for job in controller.store.list_jobs()
        if job.status is JobStatus.RUNNING
    }
    assert started

    controller.request_stop(reason="SIGTERM")
    controller.tick()
    assert {
        job.job_id
        for job in controller.store.list_jobs()
        if job.status in {JobStatus.READY, JobStatus.RUNNING}
    } <= started
    controller._drain_local_futures()
    assert all(
        job.status is JobStatus.SUCCEEDED
        for job in controller.store.list_jobs()
        if job.job_id in started
    )
    controller.close()


def test_run_service_stop_does_not_wait_for_non_cancellable_llm_work(
    tmp_path: Path,
) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(i) for i in range(4)]),
        executors={kind: _VerySlowExecutor() for kind in JobKind},
        sleep=lambda _: controller.request_stop(reason="SIGTERM"),
    )

    started = time.monotonic()
    controller.run()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert controller.store._writer_lock_stream is None
    assert controller.store.list_jobs(statuses={JobStatus.RUNNING})
    from concurrent.futures import thread as thread_module

    assert all(
        worker not in thread_module._threads_queues
        for worker in controller._pool._threads
    )


def test_max_ticks_does_not_schedule_unbounded_downstream_work(
    tmp_path: Path,
) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(i) for i in range(4)]),
        executors={kind: SimulatedJobExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.run(max_ticks=1)
    # The final tick may finish its already-submitted Design jobs, but must not
    # turn a bounded canary into a full lifecycle drain.
    assert all(
        job.kind is JobKind.DESIGN
        for job in controller.store.list_jobs()
    )


def test_restart_reconciles_already_accepted_attempt(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    controller = V2Controller(
        config=_config(tmp_path),
        store=store,
        generator=StaticIdeaGenerator([_candidate(0)]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.tick()
    job = store.list_jobs(statuses={JobStatus.RUNNING})[0]
    attempt = store.get_attempt(job.attempt_id)
    assert attempt is not None
    candidate = store.snapshot_current(attempt)
    (candidate / "plan.json").write_text("{}", encoding="utf-8")
    store.commit_candidate(attempt)
    controller._pool.shutdown(wait=True)

    restarted = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        sleep=lambda _: None,
    )
    restarted.initialize()
    recovered = restarted.store.get_job(job.job_id)
    idea = restarted.store.get_idea(job.idea_id)
    assert recovered is not None and recovered.status is JobStatus.SUCCEEDED
    assert idea is not None and idea.status is IdeaStatus.BUILDING
    restarted._pool.shutdown(wait=True)
