"""Strict, standalone configuration for Factory mode.

Factory configuration intentionally lives outside :mod:`researchclaw.config`
so adding the opt-in control plane cannot alter legacy single-Idea behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _positive_float(
    value: object,
    name: str,
    *,
    allow_zero: bool = False,
) -> float:
    import math

    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    minimum_ok = result >= 0 if allow_zero else result > 0
    if not math.isfinite(result) or not minimum_ok:
        relation = ">=" if allow_zero else ">"
        raise ValueError(f"{name} must be finite and {relation} 0")
    return result


@dataclass(frozen=True, slots=True)
class ReservoirConfig:
    low_watermark: int = 12
    target_size: int = 24
    generation_batch_size: int = 6
    generation_interval_sec: float = 300.0
    retry_backoff_sec: float = 60.0

    def __post_init__(self) -> None:
        if self.low_watermark < 0:
            raise ValueError("low_watermark cannot be negative")
        if self.target_size < self.low_watermark:
            raise ValueError("target_size must be >= low_watermark")
        if self.generation_batch_size < 1:
            raise ValueError("generation_batch_size must be positive")


@dataclass(frozen=True, slots=True)
class PopulationConfig:
    max_active_ideas: int = 10
    max_screening_ideas: int = 4
    max_pilot_ideas: int = 6
    max_validation_ideas: int = 3
    max_paper_ideas: int = 2
    max_same_family_active: int = 2


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    llm_slots: int = 2
    gpu_target_utilization: float = 0.90
    reserved_gpus: int = 2
    pilot_max_gpus_per_idea: int = 4
    validation_max_gpus_per_idea: int = 8
    max_gpu_share_per_idea: float = 0.50
    backfill_enabled: bool = True
    checkpoint_preemption: bool = False
    poll_interval_sec: float = 5.0


@dataclass(frozen=True, slots=True)
class GPUConfig:
    enabled: bool = False
    pool_config: str = ""
    restore_state: bool = True
    claim_on_start: bool = False
    prepare_on_start: bool = False
    release_on_shutdown: bool = False


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    desk_llm_calls: int = 12
    smoke_gpu_hours: float = 0.5
    pilot_gpu_hours: float = 4.0
    validation_gpu_hours: float = 32.0
    scale_gpu_hours: float = 0.0
    max_wall_clock_hours: float = 72.0
    max_engineering_repairs: int = 2
    max_no_progress_rounds: int = 2


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    minimum_seeds: int = 3
    minimum_effect_size: float = 0.03
    success_probability: float = 0.95
    futility_probability: float = 0.95
    maximum_gpu_hours: float = 12.0


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    python: str = ""
    pipeline_config: str = ""
    skip_preflight: bool = True
    auto_approve: bool = True
    graceful_shutdown_sec: float = 30.0
    simulation: bool = False
    simulation_delay_sec: float = 0.01


@dataclass(frozen=True, slots=True)
class FactoryConfig:
    enabled: bool = False
    factory_id: str = "research-factory"
    state_dir: str = "research-factory"
    topic_brief: str = ""
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    early_stopping: EarlyStoppingConfig = field(
        default_factory=EarlyStoppingConfig
    )
    worker: WorkerConfig = field(default_factory=WorkerConfig)

    @property
    def root(self) -> Path:
        return Path(self.state_dir).expanduser().resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> FactoryConfig:
        top = _mapping(raw, "config")
        data = _mapping(top.get("factory", top), "factory")

        reservoir_raw = _mapping(data.get("reservoir"), "factory.reservoir")
        reservoir = ReservoirConfig(
            low_watermark=_positive_int(
                reservoir_raw.get("low_watermark", 12),
                "factory.reservoir.low_watermark",
                allow_zero=True,
            ),
            target_size=_positive_int(
                reservoir_raw.get("target_size", 24),
                "factory.reservoir.target_size",
                allow_zero=True,
            ),
            generation_batch_size=_positive_int(
                reservoir_raw.get("generation_batch_size", 6),
                "factory.reservoir.generation_batch_size",
            ),
            generation_interval_sec=_positive_float(
                reservoir_raw.get("generation_interval_sec", 300),
                "factory.reservoir.generation_interval_sec",
            ),
            retry_backoff_sec=_positive_float(
                reservoir_raw.get("retry_backoff_sec", 60),
                "factory.reservoir.retry_backoff_sec",
            ),
        )

        population_raw = _mapping(
            data.get("population"), "factory.population"
        )
        population = PopulationConfig(
            **{
                name: _positive_int(
                    population_raw.get(name, default),
                    f"factory.population.{name}",
                )
                for name, default in {
                    "max_active_ideas": 10,
                    "max_screening_ideas": 4,
                    "max_pilot_ideas": 6,
                    "max_validation_ideas": 3,
                    "max_paper_ideas": 2,
                    "max_same_family_active": 2,
                }.items()
            }
        )

        scheduler_raw = _mapping(
            data.get("scheduler"), "factory.scheduler"
        )
        scheduler = SchedulerConfig(
            llm_slots=_positive_int(
                scheduler_raw.get("llm_slots", 2),
                "factory.scheduler.llm_slots",
            ),
            gpu_target_utilization=_positive_float(
                scheduler_raw.get("gpu_target_utilization", 0.90),
                "factory.scheduler.gpu_target_utilization",
            ),
            reserved_gpus=_positive_int(
                scheduler_raw.get("reserved_gpus", 2),
                "factory.scheduler.reserved_gpus",
                allow_zero=True,
            ),
            pilot_max_gpus_per_idea=_positive_int(
                scheduler_raw.get("pilot_max_gpus_per_idea", 4),
                "factory.scheduler.pilot_max_gpus_per_idea",
            ),
            validation_max_gpus_per_idea=_positive_int(
                scheduler_raw.get("validation_max_gpus_per_idea", 8),
                "factory.scheduler.validation_max_gpus_per_idea",
            ),
            max_gpu_share_per_idea=_positive_float(
                scheduler_raw.get("max_gpu_share_per_idea", 0.50),
                "factory.scheduler.max_gpu_share_per_idea",
            ),
            backfill_enabled=bool(
                scheduler_raw.get("backfill_enabled", True)
            ),
            checkpoint_preemption=bool(
                scheduler_raw.get("checkpoint_preemption", False)
            ),
            poll_interval_sec=_positive_float(
                scheduler_raw.get("poll_interval_sec", 5),
                "factory.scheduler.poll_interval_sec",
            ),
        )
        if scheduler.gpu_target_utilization > 1:
            raise ValueError("gpu_target_utilization cannot exceed 1")
        if scheduler.max_gpu_share_per_idea > 1:
            raise ValueError("max_gpu_share_per_idea cannot exceed 1")

        gpu_raw = _mapping(data.get("gpu"), "factory.gpu")
        gpu = GPUConfig(
            enabled=bool(gpu_raw.get("enabled", False)),
            pool_config=str(gpu_raw.get("pool_config", "") or ""),
            restore_state=bool(gpu_raw.get("restore_state", True)),
            claim_on_start=bool(gpu_raw.get("claim_on_start", False)),
            prepare_on_start=bool(gpu_raw.get("prepare_on_start", False)),
            release_on_shutdown=bool(
                gpu_raw.get("release_on_shutdown", False)
            ),
        )
        if gpu.enabled and not gpu.pool_config:
            raise ValueError(
                "factory.gpu.pool_config is required when GPU Broker is enabled"
            )

        budgets_raw = _mapping(data.get("budgets"), "factory.budgets")
        budgets = BudgetConfig(
            desk_llm_calls=_positive_int(
                budgets_raw.get("desk_llm_calls", 12),
                "factory.budgets.desk_llm_calls",
            ),
            smoke_gpu_hours=_positive_float(
                budgets_raw.get("smoke_gpu_hours", 0.5),
                "factory.budgets.smoke_gpu_hours",
                allow_zero=True,
            ),
            pilot_gpu_hours=_positive_float(
                budgets_raw.get("pilot_gpu_hours", 4),
                "factory.budgets.pilot_gpu_hours",
                allow_zero=True,
            ),
            validation_gpu_hours=_positive_float(
                budgets_raw.get("validation_gpu_hours", 32),
                "factory.budgets.validation_gpu_hours",
                allow_zero=True,
            ),
            scale_gpu_hours=_positive_float(
                budgets_raw.get("scale_gpu_hours", 0),
                "factory.budgets.scale_gpu_hours",
                allow_zero=True,
            ),
            max_wall_clock_hours=_positive_float(
                budgets_raw.get("max_wall_clock_hours", 72),
                "factory.budgets.max_wall_clock_hours",
            ),
            max_engineering_repairs=_positive_int(
                budgets_raw.get("max_engineering_repairs", 2),
                "factory.budgets.max_engineering_repairs",
                allow_zero=True,
            ),
            max_no_progress_rounds=_positive_int(
                budgets_raw.get("max_no_progress_rounds", 2),
                "factory.budgets.max_no_progress_rounds",
                allow_zero=True,
            ),
        )

        stopping_raw = _mapping(
            data.get("early_stopping"), "factory.early_stopping"
        )
        early_stopping = EarlyStoppingConfig(
            minimum_seeds=_positive_int(
                stopping_raw.get("minimum_seeds", 3),
                "factory.early_stopping.minimum_seeds",
            ),
            minimum_effect_size=_positive_float(
                stopping_raw.get("minimum_effect_size", 0.03),
                "factory.early_stopping.minimum_effect_size",
                allow_zero=True,
            ),
            success_probability=_positive_float(
                stopping_raw.get("success_probability", 0.95),
                "factory.early_stopping.success_probability",
            ),
            futility_probability=_positive_float(
                stopping_raw.get("futility_probability", 0.95),
                "factory.early_stopping.futility_probability",
            ),
            maximum_gpu_hours=_positive_float(
                stopping_raw.get("maximum_gpu_hours", 12),
                "factory.early_stopping.maximum_gpu_hours",
            ),
        )
        if (
            early_stopping.success_probability > 1
            or early_stopping.futility_probability > 1
        ):
            raise ValueError("stopping probabilities cannot exceed 1")

        worker_raw = _mapping(data.get("worker"), "factory.worker")
        worker = WorkerConfig(
            python=str(worker_raw.get("python", "") or ""),
            pipeline_config=str(
                worker_raw.get("pipeline_config", "") or ""
            ),
            skip_preflight=bool(worker_raw.get("skip_preflight", True)),
            auto_approve=bool(worker_raw.get("auto_approve", True)),
            graceful_shutdown_sec=_positive_float(
                worker_raw.get("graceful_shutdown_sec", 30),
                "factory.worker.graceful_shutdown_sec",
            ),
            simulation=bool(worker_raw.get("simulation", False)),
            simulation_delay_sec=_positive_float(
                worker_raw.get("simulation_delay_sec", 0.01),
                "factory.worker.simulation_delay_sec",
                allow_zero=True,
            ),
        )

        enabled = data.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("factory.enabled must be a YAML boolean")
        return cls(
            enabled=enabled,
            factory_id=str(data.get("factory_id", "research-factory")),
            state_dir=str(data.get("state_dir", "research-factory")),
            topic_brief=str(data.get("topic_brief", "") or ""),
            reservoir=reservoir,
            population=population,
            scheduler=scheduler,
            gpu=gpu,
            budgets=budgets,
            early_stopping=early_stopping,
            worker=worker,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> FactoryConfig:
        value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("Factory config file must contain a mapping")
        return cls.from_mapping(value)
