from __future__ import annotations

import json
from pathlib import Path

import yaml

from researchclaw.factory.actor import (
    PipelineIdeaWorker,
    SimulatedIdeaWorker,
    work_item_for_idea,
)
from researchclaw.factory.config import FactoryConfig
from researchclaw.factory.generator import (
    StaticCandidateGenerator,
    candidate_to_idea,
)
from researchclaw.factory.models import (
    BudgetTier,
    IdeaStatus,
    WorkItemStatus,
    WorkKind,
)
from researchclaw.factory.orchestrator import FactoryOrchestrator
from researchclaw.factory.store import FactoryStore


def _candidate(index: int) -> dict[str, object]:
    return {
        "id": f"candidate-{index}",
        "title": f"Distinct RSI mechanism study {index}",
        "research_question": f"question {index}",
        "falsifiable_hypothesis": f"hypothesis {index}",
        "primary_metric": "accuracy",
        "cheap_pilot": "three-seed pilot",
        "information_gain_if_false": "maps a boundary condition",
        "baselines": ["no-self-improvement control"],
        "compute": {"gpu_count": 1, "wall_clock_hours": 1},
        "weighted_score": 8.0,
    }


def _config(root: Path) -> FactoryConfig:
    return FactoryConfig.from_mapping(
        {
            "factory": {
                "enabled": True,
                "state_dir": str(root),
                "reservoir": {
                    "low_watermark": 2,
                    "target_size": 6,
                    "generation_batch_size": 6,
                    "generation_interval_sec": 0.001,
                },
                "population": {
                    "max_active_ideas": 3,
                    "max_screening_ideas": 3,
                    "max_pilot_ideas": 3,
                    "max_validation_ideas": 3,
                    "max_paper_ideas": 3,
                    "max_same_family_active": 3,
                },
                "scheduler": {"llm_slots": 3, "poll_interval_sec": 0.001},
                "worker": {"simulation": True, "simulation_delay_sec": 0},
            }
        }
    )


def test_factory_keeps_multiple_ideas_active_and_refills(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    generator = StaticCandidateGenerator(
        [_candidate(index) for index in range(9)]
    )
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=generator,
        worker=SimulatedIdeaWorker(delay_sec=0),
        sleep=lambda _: None,
    )
    orchestrator.initialize()
    snapshots = [orchestrator.tick() for _ in range(8)]

    assert max(
        sum(
            counts[status.value]
            for status in (
                IdeaStatus.SCREENING,
                IdeaStatus.BUILDING,
                IdeaStatus.PILOT,
                IdeaStatus.VALIDATING,
                IdeaStatus.PAPER,
            )
        )
        for counts in (item["ideas_by_status"] for item in snapshots)
    ) == 3
    assert snapshots[-1]["ideas_by_status"]["completed"] >= 3
    assert sum(
        snapshots[-1]["ideas_by_status"][status.value]
        for status in (
            IdeaStatus.SCREENING,
            IdeaStatus.BUILDING,
            IdeaStatus.PILOT,
            IdeaStatus.VALIDATING,
            IdeaStatus.PAPER,
        )
    ) >= 1
    store.release_writer_lock()


def test_restart_does_not_duplicate_active_work_items(tmp_path: Path) -> None:
    config = _config(tmp_path)
    generator = StaticCandidateGenerator([_candidate(1), _candidate(2)])
    first = FactoryOrchestrator(
        config=config,
        store=FactoryStore(tmp_path),
        generator=generator,
        worker=SimulatedIdeaWorker(delay_sec=100),
    )
    first.initialize()
    first.tick()
    before = {
        item.item_id
        for item in first.store.list_work_items()
    }
    first.store.release_writer_lock()

    restarted = FactoryOrchestrator(
        config=config,
        store=FactoryStore(tmp_path),
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )
    restarted.initialize()
    restarted.tick()
    after = {
        item.item_id
        for item in restarted.store.list_work_items()
    }
    assert after == before
    restarted.store.release_writer_lock()


def test_later_generation_duplicate_id_does_not_overwrite_active_idea(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    original = candidate_to_idea(_candidate(1))
    original.status = IdeaStatus.SCREENING
    store.save_idea(original)
    later = _candidate(1)
    later["title"] = "Different title from a later generation"
    generator = StaticCandidateGenerator([later])
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=generator,
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    orchestrator._refill_reservoir()
    orchestrator._admit_ideas()

    assert store.load_reservoir() == []
    assert store.get_idea(original.idea_id) == original
    events = [
        json.loads(line)
        for line in store.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["type"] == "candidate_deduplicated"
        and event["idea_id"] == original.idea_id
        and event["title"] == later["title"]
        and event["reason_code"] == "duplicate_id"
        for event in events
    )


def test_duplicate_id_admission_rejection_never_overwrites_active_idea(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    original = candidate_to_idea(_candidate(1))
    original.status = IdeaStatus.SCREENING
    store.save_idea(original)
    duplicate = _candidate(1)
    duplicate["title"] = "Different title already queued in the reservoir"
    duplicate_idea = candidate_to_idea(duplicate)
    store.save_reservoir([duplicate_idea])
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    orchestrator._admit_ideas()

    assert store.load_reservoir() == []
    assert store.get_idea(original.idea_id) == original
    events = [
        json.loads(line)
        for line in store.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        event["type"] == "admission_rejected"
        and event["idea_id"] == original.idea_id
        and event["reason_code"] == "DUPLICATE_ID"
        and event["detail"] == original.idea_id
        for event in events
    )


def test_pipeline_worker_writes_per_idea_scientific_contract(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """project: {name: test, mode: full-auto}
research:
  topic: stale topic
  selected_topic_file: /tmp/stale-topic.json
runtime: {timezone: UTC}
notifications: {channel: stdout, target: ''}
knowledge_base: {backend: markdown, root: ''}
llm:
  provider: openai-compatible
  base_url: http://example.invalid
  api_key_env: TEST_KEY
""",
        encoding="utf-8",
    )
    config = FactoryConfig.from_mapping(
        {
            "factory": {
                "worker": {
                    "pipeline_config": str(base),
                }
            }
        }
    )
    raw = _candidate(1)
    raw["title"] = "RSI verifier calibration"
    raw["falsifiable_hypothesis"] = "Calibration reduces false accepts"
    idea = candidate_to_idea(raw)
    idea.status = IdeaStatus.SCREENING
    item = work_item_for_idea(idea)
    worker = PipelineIdeaWorker(config)
    idea_dir = tmp_path / "idea"

    command = worker._command(idea=idea, item=item, idea_dir=idea_dir)

    attempt_contract = (
        idea_dir
        / "contract"
        / item.item_id
        / "attempt-01"
    )
    selected = json.loads(
        (attempt_contract / "selected_topic.json").read_text(
            encoding="utf-8"
        )
    )
    generated = yaml.safe_load(
        (attempt_contract / "pipeline.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert selected["title"] == idea.title
    assert selected["falsifiable_hypothesis"] == idea.falsifiable_hypothesis
    assert generated["research"]["topic"] == idea.title
    assert generated["research"]["selected_topic_file"] == str(
        attempt_contract / "selected_topic.json"
    )
    assert generated["experiment"]["time_budget_sec"] == 3600
    assert str(attempt_contract / "pipeline.yaml") in command
    assert (
        idea_dir / "contract" / "selected_topic.json"
    ).exists()
    assert (idea_dir / "contract" / "pipeline.yaml").exists()


def test_pipeline_worker_preserves_contract_for_each_retry_attempt(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """project: {name: test, mode: full-auto}
research: {topic: stale topic}
runtime: {timezone: UTC}
notifications: {channel: stdout, target: ''}
knowledge_base: {backend: markdown, root: ''}
llm:
  provider: openai-compatible
  base_url: http://example.invalid
  api_key_env: TEST_KEY
""",
        encoding="utf-8",
    )
    config = FactoryConfig.from_mapping(
        {"factory": {"worker": {"pipeline_config": str(base)}}}
    )
    idea = candidate_to_idea(_candidate(11))
    idea.status = IdeaStatus.SCREENING
    item = work_item_for_idea(idea)
    worker = PipelineIdeaWorker(config)
    idea_dir = tmp_path / "idea"

    item.attempt = 1
    first = worker._command(idea=idea, item=item, idea_dir=idea_dir)
    item.attempt = 2
    second = worker._command(idea=idea, item=item, idea_dir=idea_dir)

    assert first != second
    assert any("attempt-01/pipeline.yaml" in arg for arg in first)
    assert any("attempt-02/pipeline.yaml" in arg for arg in second)
    assert (
        idea_dir
        / "contract"
        / item.item_id
        / "attempt-01"
        / "pipeline.yaml"
    ).exists()
    assert (
        idea_dir
        / "contract"
        / item.item_id
        / "attempt-02"
        / "pipeline.yaml"
    ).exists()


def test_terminal_failed_work_item_gets_new_round(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(1))
    idea.status = IdeaStatus.SCREENING
    store.save_idea(idea)
    first = work_item_for_idea(idea)
    first.status = WorkItemStatus.FAILED
    store.save_work_item(first)
    idea.current_item_id = first.item_id
    store.save_idea(idea)
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    orchestrator._ensure_work_items()

    assert {
        item.item_id for item in store.list_work_items()
    } == {first.item_id, f"{first.item_id}-round-02"}


def test_zero_remaining_gpu_budget_rejects_without_starting_worker(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(1))
    idea.status = IdeaStatus.PILOT
    idea.budget_tier = BudgetTier.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    store.save_work_item(item)
    idea.current_item_id = item.item_id
    store.save_idea(idea)
    ledger = store.load_budget(idea.idea_id)
    ledger.gpu_seconds_by_tier[BudgetTier.PILOT.value] = (
        config.budgets.pilot_gpu_hours * 3600.0
    )
    store.save_budget(ledger)
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    orchestrator._dispatch_ready_items()

    rejected = store.get_idea(idea.idea_id)
    rejected_item = store.get_work_item(item.item_id)
    assert rejected is not None
    assert rejected.status is IdeaStatus.REJECTED
    assert rejected.exit_reason == "GPU_BUDGET_EXHAUSTED"
    assert rejected_item is not None
    assert rejected_item.status is WorkItemStatus.CANCELLED
    assert rejected_item.attempt == 0
    assert rejected_item.result["remaining_gpu_seconds"] == 0.0


def test_gpu_request_larger_than_remaining_budget_is_rejected(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(2))
    idea.status = IdeaStatus.PILOT
    idea.budget_tier = BudgetTier.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    store.save_work_item(item)
    ledger = store.load_budget(idea.idea_id)
    ledger.gpu_seconds_by_tier[BudgetTier.PILOT.value] = (
        config.budgets.pilot_gpu_hours * 3600.0 - 30.0
    )
    store.save_budget(ledger)
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    orchestrator._dispatch_ready_items()

    rejected_item = store.get_work_item(item.item_id)
    assert rejected_item is not None
    assert rejected_item.status is WorkItemStatus.CANCELLED
    assert rejected_item.result["requested_gpu_seconds"] == 3600.0
    assert rejected_item.result["remaining_gpu_seconds"] == 30.0


def test_pipeline_gpu_driver_keeps_gpu_kind_and_is_reconciled(
    tmp_path: Path,
) -> None:
    class DriverBroker:
        def can_admit_driver(self, item):
            del item
            return True, 1, "DRIVER_LEASE_ADMITTED"

        def reconcile(self):
            return []

        def release_driver(self, item_id):
            del item_id
            return 2.0

    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(3))
    idea.status = IdeaStatus.PILOT
    idea.budget_tier = BudgetTier.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    store.save_work_item(item)
    worker = SimulatedIdeaWorker(delay_sec=0)
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=worker,
        gpu_broker=DriverBroker(),
    )

    orchestrator._dispatch_ready_items()
    running = store.get_work_item(item.item_id)
    assert running is not None
    assert running.kind is WorkKind.GPU_EXPERIMENT
    assert running.status is WorkItemStatus.RUNNING
    assert (
        running.metadata["gpu_execution"]
        == "pipeline_driver_over_shared_pool"
    )

    orchestrator._reconcile_workers()
    succeeded = store.get_work_item(item.item_id)
    assert succeeded is not None
    assert succeeded.status is WorkItemStatus.SUCCEEDED


def test_cancel_pipeline_gpu_driver_stops_worker_and_releases_lease(
    tmp_path: Path,
) -> None:
    class DriverBroker:
        def __init__(self):
            self.released: list[str] = []

        def release_driver(self, item_id):
            self.released.append(item_id)
            return 3.0

    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(4))
    idea.status = IdeaStatus.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    item.status = WorkItemStatus.RUNNING
    item.attempt = 1
    item.metadata["gpu_execution"] = "pipeline_driver_over_shared_pool"
    store.save_work_item(item)
    worker = SimulatedIdeaWorker(delay_sec=100)
    worker.start(idea=idea, item=item, idea_dir=store.idea_dir(idea.idea_id))
    broker = DriverBroker()
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=worker,
        gpu_broker=broker,
    )

    orchestrator.cancel_active()

    cancelled = store.get_work_item(item.item_id)
    assert cancelled is not None
    assert cancelled.status is WorkItemStatus.CANCELLED
    assert cancelled.result["elapsed_sec"] == 3.0
    assert item.item_id in worker.cancelled
    assert broker.released == [item.item_id]


def test_failed_gpu_attempt_is_charged_before_retry(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(5))
    idea.status = IdeaStatus.PILOT
    idea.budget_tier = BudgetTier.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    item.status = WorkItemStatus.RUNNING
    item.attempt = 1
    item.metadata["allocated_gpus"] = 2
    item.result = {
        "elapsed_sec": 30.0,
        "returncode": 1,
        "timed_out": False,
    }
    store.save_work_item(item)
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    orchestrator._fail_item(item, reason="GPU_TASK_FAILED")

    failed = store.get_work_item(item.item_id)
    ledger = store.load_budget(idea.idea_id)
    assert failed is not None
    assert failed.status is WorkItemStatus.RETRY_WAIT
    assert failed.result["returncode"] == 1
    assert failed.result["failure_reason"] == "GPU_TASK_FAILED"
    assert ledger.gpu_seconds_by_tier[BudgetTier.PILOT.value] == 60.0


def test_retry_attempts_are_accounted_once_each(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(6))
    idea.status = IdeaStatus.PILOT
    idea.budget_tier = BudgetTier.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    item.status = WorkItemStatus.RUNNING
    item.metadata["allocated_gpus"] = 1
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    item.attempt = 1
    item.result = {"elapsed_sec": 10.0}
    orchestrator._record_terminal_gpu_usage(item)
    orchestrator._record_terminal_gpu_usage(item)
    item.attempt = 2
    item.result = {"elapsed_sec": 20.0}
    orchestrator._record_terminal_gpu_usage(item)

    ledger = store.load_budget(idea.idea_id)
    assert ledger.gpu_seconds_by_tier[BudgetTier.PILOT.value] == 30.0
    assert sorted(ledger.usage_records) == [
        f"{item.item_id}:attempt-01",
        f"{item.item_id}:attempt-02",
    ]


def test_failed_gpu_submit_consumes_retry_attempt_without_fake_usage(
    tmp_path: Path,
) -> None:
    class SubmitFailBroker:
        def submit(self, item):
            item.attempt += 1
            raise RuntimeError("submission failed")

    config = FactoryConfig.from_mapping(
        {
            "factory": {
                "enabled": True,
                "state_dir": str(tmp_path),
                "scheduler": {
                    "llm_slots": 3,
                    "reserved_gpus": 0,
                },
                "worker": {"simulation": False},
            }
        }
    )
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(7))
    idea.status = IdeaStatus.PILOT
    idea.budget_tier = BudgetTier.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    item.command = "python train.py"
    store.save_work_item(item)
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
        gpu_broker=SubmitFailBroker(),
    )

    orchestrator._dispatch_ready_items()

    failed_submit = store.get_work_item(item.item_id)
    ledger = store.load_budget(idea.idea_id)
    assert failed_submit is not None
    assert failed_submit.attempt == 1
    assert failed_submit.status is WorkItemStatus.RETRY_WAIT
    assert ledger.usage_records == {}

    failed_submit.attempt = 2
    failed_submit.metadata["allocated_gpus"] = 1
    failed_submit.result = {"elapsed_sec": 12.0}
    orchestrator._record_terminal_gpu_usage(failed_submit)

    ledger = store.load_budget(idea.idea_id)
    assert sorted(ledger.usage_records) == [
        f"{item.item_id}:attempt-02",
    ]
    assert ledger.gpu_seconds_by_tier[BudgetTier.PILOT.value] == 12.0


def test_global_work_item_failed_event_has_failure_dimensions(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = FactoryStore(tmp_path)
    store.initialize()
    idea = candidate_to_idea(_candidate(8))
    idea.status = IdeaStatus.PILOT
    store.save_idea(idea)
    item = work_item_for_idea(idea)
    item.attempt = item.attempt_limit
    item.status = WorkItemStatus.RUNNING
    store.save_work_item(item)
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=StaticCandidateGenerator([]),
        worker=SimulatedIdeaWorker(delay_sec=100),
    )

    orchestrator._fail_item(item, reason="GPU_TASK_FAILED")

    events = [
        row
        for row in (
            json.loads(line)
            for line in store.events_path.read_text(
                encoding="utf-8"
            ).splitlines()
        )
        if row["type"] == "work_item_failed"
    ]
    assert events == [
        {
            "attempt": item.attempt_limit,
            "factory_id": store.factory_id,
            "failure_reason": "GPU_TASK_FAILED",
            "idea_id": idea.idea_id,
            "item_id": item.item_id,
            "kind": item.kind.value,
            "profile": item.profile,
            "status": WorkItemStatus.FAILED.value,
            "timestamp": events[0]["timestamp"],
            "type": "work_item_failed",
        }
    ]
