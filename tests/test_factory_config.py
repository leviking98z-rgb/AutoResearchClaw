from __future__ import annotations

import pytest

from researchclaw.factory.config import FactoryConfig


def test_factory_disabled_by_default() -> None:
    config = FactoryConfig.from_mapping({"factory": {}})
    assert config.enabled is False
    assert config.reservoir.target_size == 24
    assert config.population.max_active_ideas == 10


def test_factory_config_rejects_invalid_watermarks() -> None:
    with pytest.raises(ValueError, match="target_size"):
        FactoryConfig.from_mapping(
            {
                "factory": {
                    "reservoir": {
                        "low_watermark": 12,
                        "target_size": 4,
                    }
                }
            }
        )


def test_gpu_broker_requires_pool_config() -> None:
    with pytest.raises(ValueError, match="pool_config"):
        FactoryConfig.from_mapping(
            {"factory": {"gpu": {"enabled": True}}}
        )
