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
    assert config.budgets.max_job_attempts == 3
    assert config.budgets.max_design_revisions == 2
    assert config.research_memory.enabled is True
    assert config.research_memory.reconcile_interval_ticks == 15
    assert config.usage_monitoring.enabled is True
    assert config.usage_monitoring.history_hours == 168
    assert config.database_path == config.root / "autoresearch.db"
    assert config.database_backup_path is None


def test_config_accepts_local_database_and_shared_backup(tmp_path) -> None:
    state = tmp_path / "shared-state"
    local_database = tmp_path / "local" / "autoresearch.db"
    backup = state / "autoresearch.db.backup"
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "state_dir": str(state),
                "storage": {
                    "database_path": str(local_database),
                    "database_backup_path": str(backup),
                    "backup_interval_sec": 12.5,
                },
            }
        }
    )

    assert config.database_path == local_database.resolve()
    assert config.database_backup_path == backup.resolve()
    assert config.storage.backup_interval_sec == 12.5


def test_config_defaults_backup_beside_shared_state_for_local_database(
    tmp_path,
) -> None:
    state = tmp_path / "shared-state"
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "state_dir": str(state),
                "storage": {
                    "database_path": str(
                        tmp_path / "local" / "autoresearch.db"
                    ),
                },
            }
        }
    )

    assert (
        config.database_backup_path
        == (state / "autoresearch.db.backup").resolve()
    )


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


def test_execution_smoke_environment_is_validated() -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "execution": {
                    "smoke_environment": "GPU_POOL",
                }
            }
        }
    )
    assert config.execution.smoke_environment == "gpu_pool"

    with pytest.raises(ValueError, match="smoke_environment"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "execution": {
                        "smoke_environment": "somewhere",
                    }
                }
            }
        )


def test_offline_gpu_dependencies_require_cache_dir(tmp_path) -> None:
    with pytest.raises(ValueError, match="gpu_cache_dir"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "state_dir": str(tmp_path / "runs" / "state"),
                    "execution": {
                        "gpu_dependency_mode": "offline",
                    },
                    "gpu": {
                        "enabled": True,
                        "pool_config": str(tmp_path / "pool.yaml"),
                        "shared_workspace_root": str(tmp_path / "runs"),
                    },
                }
            }
        )

    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "state_dir": str(tmp_path / "runs" / "state"),
                "execution": {
                    "gpu_dependency_mode": "OFFLINE",
                    "gpu_cache_dir": "/data/cache/autoresearch-v2/huggingface",
                    "gpu_cache_archive": "/root/sync/autoresearch-cache.tar",
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": str(tmp_path / "pool.yaml"),
                    "shared_workspace_root": str(tmp_path / "runs"),
                },
            }
        }
    )
    assert config.execution.gpu_dependency_mode == "offline"
    assert config.execution.gpu_cache_dir.endswith("huggingface")
    assert config.execution.gpu_cache_archive.endswith(".tar")


def test_attestation_key_cannot_live_in_generated_candidate(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="outside the Controller state"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "state_dir": str(tmp_path),
                    "execution": {
                        "attestation_key_file": str(
                            tmp_path
                            / "ideas"
                            / "idea-1"
                            / "current"
                            / "controller.key"
                        ),
                    },
                }
            }
        )


def test_attempt_and_revision_budgets_must_be_positive() -> None:
    with pytest.raises(ValueError, match="attempt and revision budgets"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "budgets": {
                        "max_design_revisions": 0,
                    }
                }
            }
        )


def test_research_memory_interval_must_be_positive() -> None:
    with pytest.raises(ValueError, match="reconcile_interval_ticks"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "research_memory": {
                        "reconcile_interval_ticks": 0,
                    }
                }
            }
        )


def test_storage_rejects_invalid_backup_policy(tmp_path) -> None:
    database = tmp_path / "autoresearch.db"
    with pytest.raises(ValueError, match="backup_interval_sec"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "storage": {
                        "database_path": str(database),
                        "backup_interval_sec": 0,
                    }
                }
            }
        )

    with pytest.raises(ValueError, match="must differ"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "storage": {
                        "database_path": str(database),
                        "database_backup_path": str(database),
                    }
                }
            }
        )


def test_usage_monitoring_accepts_rates_and_validates_thresholds() -> None:
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "usage_monitoring": {
                    "warning_threshold": 0.4,
                    "critical_threshold": 0.9,
                    "single_call_token_warning": 123_456,
                    "monthly_token_budget": 10_000,
                    "model_prices": {
                        "claude-sonnet-5": {
                            "input_per_million_usd": 3,
                            "output_per_million_usd": 15,
                        }
                    },
                }
            }
        }
    )
    assert config.usage_monitoring.monthly_token_budget == 10_000
    assert config.usage_monitoring.single_call_token_warning == 123_456
    assert (
        config.usage_monitoring.model_prices["claude-sonnet-5"][
            "output_per_million_usd"
        ]
        == 15
    )

    with pytest.raises(ValueError, match="thresholds"):
        V2Config.from_mapping(
            {
                "autoresearch_v2": {
                    "usage_monitoring": {
                        "warning_threshold": 0.9,
                        "critical_threshold": 0.8,
                    }
                }
            }
        )
