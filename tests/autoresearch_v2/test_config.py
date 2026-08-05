from __future__ import annotations

import pytest

from researchclaw.autoresearch_v2.config import V2Config


def test_config_accepts_parallel_defaults() -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "population": {
                    "active_idea_target": 6,
                    "max_active_ideas": 8,
                },
                "concurrency": {
                    "max_llm_jobs": 4,
                    "max_cpu_jobs": 8,
                    "max_gpu_jobs": 6,
                },
            }
        }
    )
    assert config.population.active_idea_target == 6
    assert config.concurrency.max_gpu_jobs == 6


def test_config_rejects_impossible_target() -> None:
    with pytest.raises(ValueError, match="active_idea_target"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "population": {
                        "active_idea_target": 9,
                        "max_active_ideas": 8,
                    }
                }
            }
        )
