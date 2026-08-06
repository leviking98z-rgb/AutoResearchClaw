from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import researchclaw.autoresearch_v2.controller as controller_module
from researchclaw.autoresearch_v2.config import V2Config
from researchclaw.autoresearch_v2.controller import V2Controller
from researchclaw.autoresearch_v2.ideas import (
    StaticIdeaGenerator,
    candidate_to_idea,
)
from researchclaw.autoresearch_v2.jobs import JobOutcome, SimulatedJobExecutor
from researchclaw.autoresearch_v2.models import (
    AttemptRecord,
    AttemptStatus,
    IdeaStatus,
    JobKind,
    JobRecord,
    JobStatus,
)
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


class _ControlledExecutor:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def execute(self, **kwargs):
        self.started.set()
        assert self.release.wait(timeout=5)
        return SimulatedJobExecutor().execute(**kwargs)


class _InterruptedExecutor:
    def execute(self, **kwargs):
        del kwargs
        time.sleep(0.05)
        return JobOutcome(False, "retry", "interrupted fixture", {})


class _BlockingIdeaGenerator:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def generate(self, *, count, existing):
        del count, existing
        self.started.set()
        assert self.release.wait(timeout=5)
        return [candidate_to_idea(_candidate(99))]


def test_production_idea_generation_never_blocks_job_dispatch_or_tick(
    tmp_path: Path,
) -> None:
    generator = _BlockingIdeaGenerator()
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=generator,
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    active = candidate_to_idea(_candidate(0))
    active.status = IdeaStatus.DESIGNING
    controller.store.save_idea(active)

    started = time.monotonic()
    snapshot = controller.tick()
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert snapshot["idea_generation_running"] is True
    assert generator.started.wait(timeout=1)
    assert controller.store.list_jobs(statuses={JobStatus.RUNNING})

    generator.release.set()
    controller._idea_generation.future.result(timeout=2)
    controller._collect_idea_generation()
    controller._pool.shutdown(wait=True)
    controller.close()


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


def test_dispatch_cancels_non_current_and_inactive_jobs(
    tmp_path: Path,
) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()

    active = candidate_to_idea(_candidate(0))
    active.status = IdeaStatus.DESIGNING
    active.current_job_id = f"{active.idea_id}-design-current"
    controller.store.save_idea(active)
    current = JobRecord(
        job_id=active.current_job_id,
        idea_id=active.idea_id,
        kind=JobKind.DESIGN,
        status=JobStatus.READY,
    )
    superseded = JobRecord(
        job_id=f"{active.idea_id}-design-old",
        idea_id=active.idea_id,
        kind=JobKind.DESIGN,
        status=JobStatus.RETRY_WAIT,
    )
    controller.store.save_job(current)
    controller.store.save_job(superseded)

    inactive = candidate_to_idea(_candidate(1))
    inactive.status = IdeaStatus.QUARANTINED
    inactive.current_job_id = f"{inactive.idea_id}-build"
    controller.store.save_idea(inactive)
    stale = JobRecord(
        job_id=inactive.current_job_id,
        idea_id=inactive.idea_id,
        kind=JobKind.BUILD,
        status=JobStatus.READY,
    )
    controller.store.save_job(stale)

    controller._dispatch()

    assert controller.store.get_job(current.job_id).status is JobStatus.RUNNING
    durable_superseded = controller.store.get_job(superseded.job_id)
    durable_stale = controller.store.get_job(stale.job_id)
    assert durable_superseded.status is JobStatus.CANCELLED
    assert durable_superseded.result["reason"] == "superseded_job"
    assert durable_stale.status is JobStatus.CANCELLED
    assert durable_stale.result["reason"] == "inactive_idea"
    assert controller.store.get_idea(inactive.idea_id).current_job_id == ""

    controller.request_stop()
    controller._pool.shutdown(wait=True)


def test_initialize_cancels_stale_running_job_before_recovery(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    inactive = candidate_to_idea(_candidate(0))
    inactive.status = IdeaStatus.REJECTED
    inactive.current_job_id = f"{inactive.idea_id}-pilot"
    store.save_idea(inactive)
    stale = JobRecord(
        job_id=inactive.current_job_id,
        idea_id=inactive.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.RUNNING,
        requires_gpu=False,
        attempt=1,
        attempt_id=f"{inactive.current_job_id}-attempt-01",
    )
    store.save_job(stale)
    attempt = AttemptRecord(
        attempt_id=stale.attempt_id,
        idea_id=inactive.idea_id,
        job_id=stale.job_id,
        number=1,
        status=AttemptStatus.RUNNING,
    )
    store.save_attempt(attempt)

    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        executors={kind: SimulatedJobExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()

    durable = controller.store.get_job(stale.job_id)
    durable_attempt = controller.store.get_attempt(stale.attempt_id)
    assert durable.status is JobStatus.CANCELLED
    assert durable.result["reason"] == "inactive_idea"
    assert durable_attempt.status is AttemptStatus.CANCELLED
    assert controller.store.get_idea(inactive.idea_id).current_job_id == ""
    assert not controller.store.list_jobs(statuses={JobStatus.RETRY_WAIT})
    controller.close()


def test_completed_local_future_cannot_resurrect_cancelled_job(
    tmp_path: Path,
) -> None:
    executor = _ControlledExecutor()
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        executors={JobKind.DESIGN: executor},
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.DESIGNING
    idea.current_job_id = f"{idea.idea_id}-design"
    controller.store.save_idea(idea)
    job = JobRecord(
        job_id=idea.current_job_id,
        idea_id=idea.idea_id,
        kind=JobKind.DESIGN,
        status=JobStatus.READY,
    )
    controller.store.save_job(job)
    controller._dispatch()
    assert executor.started.wait(timeout=1)

    idea = controller.store.get_idea(idea.idea_id)
    idea.status = IdeaStatus.QUARANTINED
    controller.store.save_idea(idea)
    controller._reconcile_current_jobs()
    assert controller.store.get_job(job.job_id).status is JobStatus.CANCELLED

    executor.release.set()
    controller._running[job.job_id].future.result(timeout=2)
    controller._collect_finished()

    durable_job = controller.store.get_job(job.job_id)
    durable_idea = controller.store.get_idea(idea.idea_id)
    durable_attempt = controller.store.get_attempt(
        f"{job.job_id}-attempt-01"
    )
    assert durable_job.status is JobStatus.CANCELLED
    assert durable_idea.status is IdeaStatus.QUARANTINED
    assert durable_attempt.status is AttemptStatus.CANCELLED
    controller.close()


class _AlwaysFail:
    def execute(self, **kwargs):
        del kwargs
        return JobOutcome(False, "retry", "deterministic failure", {})


class _InfrastructureFail:
    def execute(self, **kwargs):
        del kwargs
        return JobOutcome(
            False,
            "retry",
            "temporary infrastructure failure",
            {
                "failure_class": "infrastructure_transient",
                "consume_attempt": False,
            },
        )


def test_infrastructure_retry_does_not_spend_scientific_attempt(
    tmp_path: Path,
) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(0)]),
        executors={kind: _InfrastructureFail() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.tick()
    for _ in range(50):
        controller._collect_finished()
        jobs = controller.store.list_jobs()
        if jobs and jobs[0].status is JobStatus.RETRY_WAIT:
            break
        time.sleep(0.002)
    job = controller.store.list_jobs()[0]
    assert job.status is JobStatus.RETRY_WAIT
    assert job.attempt == 1
    assert job.attempt_limit == 2
    assert job.result["infrastructure_retries"] == 1
    controller.request_stop()
    controller._pool.shutdown(wait=True)


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


def test_design_job_uses_one_outer_attempt(tmp_path: Path) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([_candidate(0)]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.tick()

    design = next(
        job
        for job in controller.store.list_jobs()
        if job.kind is JobKind.DESIGN
    )
    assert design.attempt_limit == 1
    controller.request_stop()
    controller._pool.shutdown(wait=True)


def test_report_job_uses_one_outer_attempt(tmp_path: Path) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.REPORTING
    controller.store.save_idea(idea)
    controller._ensure_jobs()

    report = next(
        job
        for job in controller.store.list_jobs()
        if job.kind is JobKind.REPORT
    )
    assert report.attempt_limit == 1
    controller.request_stop()
    controller._pool.shutdown(wait=True)


def test_completed_negative_report_preserves_terminal_negative_status(
    tmp_path: Path,
) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    idea.status = IdeaStatus.REPORTING
    idea.candidate["final_outcome"] = "informative_negative"
    job = JobRecord(
        job_id=f"{idea.idea_id}-report",
        idea_id=idea.idea_id,
        kind=JobKind.REPORT,
    )
    attempt = AttemptRecord(
        attempt_id=f"{job.job_id}-attempt-01",
        idea_id=idea.idea_id,
        job_id=job.job_id,
        number=1,
    )

    controller._apply_outcome(
        idea,
        job,
        attempt,
        JobOutcome(True, "complete", "paper_package_generated", {}),
    )

    assert idea.status is IdeaStatus.COMPLETED_NEGATIVE
    assert job.status is JobStatus.SUCCEEDED
    controller._pool.shutdown(wait=True)


def test_research_memory_http_never_blocks_controller_tick(
    tmp_path: Path,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class _Memory:
        def reconcile(self, idea):
            started.set()
            assert release.wait(timeout=5)
            return SimpleNamespace(
                ok=True,
                external_id=f"memory:{idea.idea_id}",
                error="",
            )

    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "research_memory": {
                    "reconcile_interval_ticks": 1,
                },
            }
        }
    )
    controller = V2Controller(
        config=config,
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        research_memory=_Memory(),
        sleep=lambda _: None,
    )
    controller.initialize()
    controller.store.save_idea(candidate_to_idea(_candidate(0)))

    before = time.monotonic()
    controller.tick()
    elapsed = time.monotonic() - before

    assert elapsed < 0.5
    assert started.wait(timeout=1)
    release.set()
    assert controller._research_memory_sync is not None
    controller._research_memory_sync.future.result(timeout=2)
    controller._collect_research_memory_sync()
    controller.close()


def test_research_memory_reconcile_is_idempotent_until_idea_changes(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class _Memory:
        def reconcile(self, idea):
            calls.append(idea.status.value)
            return SimpleNamespace(
                ok=True,
                external_id=f"memory:{idea.idea_id}",
                error="",
            )

    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "research_memory": {
                    "reconcile_interval_ticks": 1,
                },
            }
        }
    )
    controller = V2Controller(
        config=config,
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        research_memory=_Memory(),
        sleep=lambda _: None,
    )
    controller.initialize()
    idea = candidate_to_idea(_candidate(0))
    controller.store.save_idea(idea)

    controller._reconcile_research_memory()
    controller._reconcile_research_memory()
    assert calls == ["new"]

    idea.status = IdeaStatus.REPORTING
    controller.store.save_idea(idea)
    controller._reconcile_research_memory()
    assert calls == ["new", "reporting"]
    controller._pool.shutdown(wait=True)


def test_detached_gateway_calls_count_toward_llm_capacity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    monkeypatch.setattr(
        controller_module,
        "_live_model_gateway_calls",
        lambda: 3,
    )
    controller._simulation_mode = False

    assert controller._running_llm_count() == 3
    controller._pool.shutdown(wait=True)


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


def test_restart_refund_removes_incomplete_candidate_workspace(
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
    running = store.list_jobs(statuses={JobStatus.RUNNING})[0]
    attempt = store.get_attempt(running.attempt_id)
    assert attempt is not None
    candidate = store.prepare_candidate(attempt)
    (candidate / "partial.json").write_text("{}", encoding="utf-8")
    controller._pool.shutdown(wait=True)

    restarted = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        executors={kind: _SlowExecutor() for kind in JobKind},
        sleep=lambda _: None,
    )
    restarted.initialize()

    assert not candidate.exists()
    recovered = restarted.store.get_job(running.job_id)
    assert recovered is not None
    assert recovered.attempt == 0
    restarted.tick()
    redispatched = restarted.store.get_job(running.job_id)
    assert redispatched is not None
    assert redispatched.status is JobStatus.RUNNING
    assert redispatched.attempt_id == running.attempt_id
    restarted.request_stop()
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
