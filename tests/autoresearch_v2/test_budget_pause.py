from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import researchclaw.autoresearch_v2.controller as controller_module
from researchclaw.autoresearch_v2.config import V2Config
from researchclaw.autoresearch_v2.controller import V2Controller
from researchclaw.autoresearch_v2.gpu import AdaptiveGPUScheduler, GPUBroker
from researchclaw.autoresearch_v2.ideas import (
    StaticIdeaGenerator,
    candidate_to_idea,
)
from researchclaw.autoresearch_v2.models import (
    IdeaRecord,
    IdeaStatus,
    JobKind,
    JobRecord,
    JobStatus,
)
from researchclaw.autoresearch_v2.store import V2Store


class _ClockDateTime(datetime):
    current = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        value = cls.current
        return value if tz is None else value.astimezone(tz)


class _Pool:
    def start_keepalive(self) -> None:
        pass

    def stop_keepalive(self) -> None:
        pass


def _config(root: Path) -> V2Config:
    return V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(root),
                "gpu": {
                    "enabled": True,
                    "pool_config": "unused-in-test",
                    "shared_workspace_root": str(root),
                },
                "budgets": {
                    "max_wall_clock_hours_per_idea": 1,
                    "max_no_progress_hours": 1,
                },
            }
        }
    )


def _candidate() -> dict[str, object]:
    return {
        "id": "budget-pause",
        "title": "GPU outage budget pause",
        "family": "calibration",
        "research_question": "Does the experiment work?",
        "falsifiable_hypothesis": "It improves accuracy.",
        "closest_prior_work": ["self refinement"],
        "novelty_gap": "budget pause semantics",
        "datasets": ["GSM8K"],
        "models": ["Qwen"],
        "compute": {"gpu_count": 1, "wall_clock_hours": 0.1},
        "primary_metric": "accuracy",
        "baselines": ["single pass"],
        "ablations": ["remove mechanism"],
        "failure_safety_tests": ["leakage"],
        "implementation_feasibility": "public stack",
        "licensing_feasibility": "verified",
        "information_gain_if_true": "supports mechanism",
        "information_gain_if_false": "rules out mechanism",
        "cheap_pilot": "one GPU",
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


def _active_gpu_idea(store: V2Store) -> tuple[IdeaRecord, JobRecord]:
    idea = candidate_to_idea(_candidate())
    idea.status = IdeaStatus.PILOTING
    idea.created_at = (
        _ClockDateTime.current - timedelta(minutes=30)
    ).isoformat()
    idea.last_progress_at = idea.created_at
    job = JobRecord(
        job_id=f"{idea.idea_id}-pilot",
        idea_id=idea.idea_id,
        kind=JobKind.PILOT,
        status=JobStatus.READY,
        requires_gpu=True,
        min_gpus=1,
        preferred_gpus=1,
        max_gpus=1,
        timeout_sec=60,
    )
    idea.current_job_id = job.job_id
    store.save_idea(idea)
    store.save_job(job)
    return idea, job


def _record_gpu_unavailable(store: V2Store) -> None:
    store.event(
        "gpu_broker_unavailable",
        error="lease unavailable",
        configured_gpu_capacity=8,
    )
    unavailable = store.list_events(limit=1)[0]
    with store.connect() as conn:
        conn.execute(
            "UPDATE events SET timestamp=? WHERE seq=?",
            (_ClockDateTime.current.isoformat(), unavailable["seq"]),
        )


def _broker() -> GPUBroker:
    return GPUBroker(
        pool=_Pool(),
        scheduler=AdaptiveGPUScheduler(total_gpus=1),
    )


def test_gpu_outage_pauses_no_progress_and_wall_clock_budgets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _ClockDateTime.current = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(controller_module, "datetime", _ClockDateTime)
    monkeypatch.setattr(
        controller_module,
        "utc_now",
        lambda: _ClockDateTime.current.isoformat(timespec="milliseconds"),
    )
    store = V2Store(tmp_path)
    store.initialize()
    idea, job = _active_gpu_idea(store)
    _record_gpu_unavailable(store)
    controller = V2Controller(
        config=_config(tmp_path),
        store=store,
        generator=StaticIdeaGenerator([]),
        configured_gpu_capacity=8,
        sleep=lambda _: None,
    )
    controller.initialize()

    _ClockDateTime.current += timedelta(hours=12)
    controller._enforce_liveness_budgets()
    durable = store.get_idea(idea.idea_id)
    assert durable is not None
    assert durable.status is IdeaStatus.PILOTING
    assert controller._budget_block_reason(durable, job) == ""
    assert controller.snapshot()["idea_budget_paused"] is True
    assert len(
        [
            event
            for event in store.list_events(limit=50)
            if event["event_type"] == "idea_budget_pause_started"
        ]
    ) == 1
    controller.close()


def test_gpu_budget_pause_survives_restart_and_resumes_exactly_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _ClockDateTime.current = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(controller_module, "datetime", _ClockDateTime)
    monkeypatch.setattr(
        controller_module,
        "utc_now",
        lambda: _ClockDateTime.current.isoformat(timespec="milliseconds"),
    )
    store = V2Store(tmp_path)
    store.initialize()
    idea, job = _active_gpu_idea(store)
    _record_gpu_unavailable(store)
    first = V2Controller(
        config=_config(tmp_path),
        store=store,
        generator=StaticIdeaGenerator([]),
        configured_gpu_capacity=8,
        sleep=lambda _: None,
    )
    first.initialize()
    first.close()

    _ClockDateTime.current += timedelta(hours=12)
    restarted = V2Controller(
        config=_config(tmp_path),
        store=V2Store(tmp_path),
        generator=StaticIdeaGenerator([]),
        gpu_broker=_broker(),
        configured_gpu_capacity=8,
        sleep=lambda _: None,
    )
    restarted.initialize()
    durable = restarted.store.get_idea(idea.idea_id)
    assert durable is not None
    assert restarted._budget_block_reason(durable, job) == ""
    restarted._enforce_liveness_budgets()
    durable = restarted.store.get_idea(idea.idea_id)
    assert durable is not None
    assert durable.status is IdeaStatus.PILOTING

    restarted._sync_gpu_budget_pause_state()
    events = restarted.store.list_events(limit=50)
    assert len(
        [
            event
            for event in events
            if event["event_type"] == "idea_budget_pause_started"
        ]
    ) == 1
    assert len(
        [
            event
            for event in events
            if event["event_type"] == "idea_budget_pause_resumed"
        ]
    ) == 1

    _ClockDateTime.current += timedelta(minutes=31)
    durable = restarted.store.get_idea(idea.idea_id)
    assert durable is not None
    assert (
        restarted._budget_block_reason(durable, job)
        == "wall_clock_budget_exhausted"
    )
    restarted._enforce_liveness_budgets()
    durable = restarted.store.get_idea(idea.idea_id)
    assert durable is not None
    assert durable.status is IdeaStatus.QUARANTINED
    assert durable.exit_reason == "no_progress_timeout"
    restarted.close()


def test_gpu_unavailable_startup_event_backfills_pause_start(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _ClockDateTime.current = datetime(2026, 8, 6, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(controller_module, "datetime", _ClockDateTime)
    monkeypatch.setattr(
        controller_module,
        "utc_now",
        lambda: _ClockDateTime.current.isoformat(timespec="milliseconds"),
    )
    store = V2Store(tmp_path)
    store.initialize()
    idea, job = _active_gpu_idea(store)
    _record_gpu_unavailable(store)
    _ClockDateTime.current += timedelta(hours=12)

    controller = V2Controller(
        config=_config(tmp_path),
        store=store,
        generator=StaticIdeaGenerator([]),
        configured_gpu_capacity=8,
        sleep=lambda _: None,
    )
    controller.initialize()
    durable = store.get_idea(idea.idea_id)
    assert durable is not None
    assert controller._budget_block_reason(durable, job) == ""
    controller._enforce_liveness_budgets()
    durable = store.get_idea(idea.idea_id)
    assert durable is not None
    assert durable.status is IdeaStatus.PILOTING
    controller.close()
