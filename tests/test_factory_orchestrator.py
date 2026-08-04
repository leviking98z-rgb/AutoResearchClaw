from __future__ import annotations

from pathlib import Path

from researchclaw.factory.actor import SimulatedIdeaWorker
from researchclaw.factory.config import FactoryConfig
from researchclaw.factory.generator import StaticCandidateGenerator
from researchclaw.factory.models import IdeaStatus
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
