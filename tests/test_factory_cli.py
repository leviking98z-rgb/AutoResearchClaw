from __future__ import annotations

import json
from pathlib import Path

from researchclaw.factory.cli import main


def _config(tmp_path: Path, *, enabled: bool) -> Path:
    path = tmp_path / "factory.yaml"
    path.write_text(
        "\n".join(
            [
                "factory:",
                f"  enabled: {'true' if enabled else 'false'}",
                f"  state_dir: {tmp_path / 'state'}",
                "  worker:",
                "    simulation: true",
                "    simulation_delay_sec: 0",
                "  reservoir:",
                "    low_watermark: 1",
                "    target_size: 1",
                "    generation_batch_size: 1",
                "    generation_interval_sec: 0.001",
                "  population:",
                "    max_active_ideas: 1",
                "    max_screening_ideas: 1",
                "    max_pilot_ideas: 1",
                "    max_validation_ideas: 1",
                "    max_paper_ideas: 1",
                "    max_same_family_active: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_cli_refuses_disabled_factory(tmp_path: Path) -> None:
    config = _config(tmp_path, enabled=False)
    assert main(["-c", str(config), "start", "--once"]) == 2


def test_cli_simulation_start_status_pause_resume(tmp_path: Path) -> None:
    config = _config(tmp_path, enabled=True)
    candidates = tmp_path / "candidates.json"
    candidates.write_text(
        json.dumps(
            [
                {
                    "id": "candidate",
                    "title": "RSI simulation candidate",
                    "research_question": "question",
                    "falsifiable_hypothesis": "hypothesis",
                    "primary_metric": "accuracy",
                    "cheap_pilot": "pilot",
                    "information_gain_if_false": "boundary",
                    "baselines": ["no-self-improvement control"],
                    "compute": {"gpu_count": 1, "wall_clock_hours": 1},
                    "weighted_score": 8,
                }
            ]
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "-c",
                str(config),
                "start",
                "--once",
                "--simulation-candidates",
                str(candidates),
            ]
        )
        == 0
    )
    assert main(["-c", str(config), "status"]) == 0
    assert main(["-c", str(config), "pause", "test"]) == 0
    assert main(["-c", str(config), "resume"]) == 0
