"""Configuration for the lightweight Research Queue prototype."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import BudgetLevel


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class BudgetSpec:
    gpus: int
    timeout_sec: float
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpus": self.gpus,
            "timeout_sec": self.timeout_sec,
            "parameters": self.parameters,
        }


@dataclass(frozen=True, slots=True)
class QueueLimits:
    candidate_target: int = 6
    generation_batch_size: int = 2
    max_active_ideas: int = 4
    max_total_ideas: int = 0
    max_revisions_per_idea: int = 2
    max_runs_per_budget: int = 2
    max_steps_per_idea: int = 10
    max_infra_retries: int = 1
    duplicate_threshold: float = 0.78


@dataclass(frozen=True, slots=True)
class QueueConcurrency:
    max_llm_jobs: int = 2
    max_run_jobs: int = 4
    poll_interval_sec: float = 0.1


@dataclass(frozen=True, slots=True)
class QueueModels:
    researchclaw_config: str = "config.rsi.yaml"
    decision_role: str = "research_director"
    worker_role: str = "coding_engineer"
    utility_role: str = "literature_researcher"


@dataclass(frozen=True, slots=True)
class QueueExecution:
    backend: str = "local"
    python_executable: str = "python"
    simulation: bool = True


@dataclass(frozen=True, slots=True)
class QueueGPU:
    max_total_gpus: int = 4
    max_gpus_per_run: int = 2
    poll_interval_sec: float = 1.0
    pass_env: tuple[str, ...] = ()
    resource_manager: dict[str, Any] = field(default_factory=dict)


def _default_budgets() -> dict[BudgetLevel, BudgetSpec]:
    return {
        BudgetLevel.B0: BudgetSpec(
            gpus=1,
            timeout_sec=120.0,
            parameters={"examples": 8, "seeds": 1},
        ),
        BudgetLevel.B1: BudgetSpec(
            gpus=2,
            timeout_sec=600.0,
            parameters={"examples": 32, "seeds": 2},
        ),
        BudgetLevel.B2: BudgetSpec(
            gpus=4,
            timeout_sec=1800.0,
            parameters={"examples": 64, "seeds": 3},
        ),
    }


@dataclass(frozen=True, slots=True)
class ResearchQueueConfig:
    enabled: bool = False
    system_id: str = "research-queue-prototype"
    state_dir: str = "workspace/research-queue-prototype"
    artifact_dir: str = ""
    brief: str = ""
    brief_file: str = ""
    limits: QueueLimits = field(default_factory=QueueLimits)
    concurrency: QueueConcurrency = field(default_factory=QueueConcurrency)
    models: QueueModels = field(default_factory=QueueModels)
    execution: QueueExecution = field(default_factory=QueueExecution)
    gpu: QueueGPU = field(default_factory=QueueGPU)
    budgets: dict[BudgetLevel, BudgetSpec] = field(default_factory=_default_budgets)

    @property
    def root(self) -> Path:
        return Path(self.state_dir).expanduser().resolve()

    @property
    def artifact_root(self) -> Path:
        value = self.artifact_dir or self.state_dir
        return Path(value).expanduser().resolve()

    def budget(self, level: BudgetLevel) -> BudgetSpec:
        return self.budgets[level]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> ResearchQueueConfig:
        base = Path(base_dir or ".").expanduser().resolve()
        top = _mapping(raw, "config")
        data = _mapping(
            top.get("research_queue", top),
            "research_queue",
        )
        limits_raw = _mapping(data.get("limits"), "limits")
        limits = QueueLimits(
            candidate_target=int(limits_raw.get("candidate_target", 6)),
            generation_batch_size=int(limits_raw.get("generation_batch_size", 2)),
            max_active_ideas=int(limits_raw.get("max_active_ideas", 4)),
            max_total_ideas=int(limits_raw.get("max_total_ideas", 0)),
            max_revisions_per_idea=int(limits_raw.get("max_revisions_per_idea", 2)),
            max_runs_per_budget=int(limits_raw.get("max_runs_per_budget", 2)),
            max_steps_per_idea=int(limits_raw.get("max_steps_per_idea", 10)),
            max_infra_retries=int(limits_raw.get("max_infra_retries", 1)),
            duplicate_threshold=float(limits_raw.get("duplicate_threshold", 0.78)),
        )
        concurrency_raw = _mapping(
            data.get("concurrency"),
            "concurrency",
        )
        concurrency = QueueConcurrency(
            max_llm_jobs=int(concurrency_raw.get("max_llm_jobs", 2)),
            max_run_jobs=int(concurrency_raw.get("max_run_jobs", 4)),
            poll_interval_sec=float(concurrency_raw.get("poll_interval_sec", 0.1)),
        )
        models_raw = _mapping(data.get("models"), "models")
        model_config = str(
            models_raw.get("researchclaw_config", "config.rsi.yaml")
            or "config.rsi.yaml"
        )
        if not Path(model_config).expanduser().is_absolute():
            model_config = str((base / model_config).resolve())
        models = QueueModels(
            researchclaw_config=model_config,
            decision_role=str(models_raw.get("decision_role", "research_director")),
            worker_role=str(models_raw.get("worker_role", "coding_engineer")),
            utility_role=str(models_raw.get("utility_role", "literature_researcher")),
        )
        execution_raw = _mapping(data.get("execution"), "execution")
        execution = QueueExecution(
            backend=str(execution_raw.get("backend", "local") or "local")
            .strip()
            .casefold(),
            python_executable=str(
                execution_raw.get("python_executable", "python") or "python"
            ),
            simulation=bool(execution_raw.get("simulation", True)),
        )
        gpu_raw = _mapping(data.get("gpu"), "gpu")
        pass_env_raw = gpu_raw.get("pass_env", ())
        if isinstance(pass_env_raw, str):
            pass_env = tuple(
                item.strip() for item in pass_env_raw.split(",") if item.strip()
            )
        else:
            pass_env = tuple(
                str(item).strip() for item in (pass_env_raw or ()) if str(item).strip()
            )
        gpu = QueueGPU(
            max_total_gpus=int(gpu_raw.get("max_total_gpus", 4)),
            max_gpus_per_run=int(gpu_raw.get("max_gpus_per_run", 2)),
            poll_interval_sec=float(gpu_raw.get("poll_interval_sec", 1.0)),
            pass_env=pass_env,
            resource_manager=_mapping(
                gpu_raw.get("resource_manager"),
                "gpu.resource_manager",
            ),
        )
        budgets = _default_budgets()
        budgets_raw = _mapping(data.get("budgets"), "budgets")
        for level in BudgetLevel:
            item = _mapping(
                budgets_raw.get(level.value),
                f"budgets.{level.value}",
            )
            if not item:
                continue
            default = budgets[level]
            budgets[level] = BudgetSpec(
                gpus=int(item.get("gpus", default.gpus)),
                timeout_sec=float(item.get("timeout_sec", default.timeout_sec)),
                parameters=_mapping(
                    item.get("parameters", default.parameters),
                    f"budgets.{level.value}.parameters",
                ),
            )
        state_dir = str(
            data.get("state_dir", "workspace/research-queue-prototype")
            or "workspace/research-queue-prototype"
        )
        if not Path(state_dir).expanduser().is_absolute():
            state_dir = str((base / state_dir).resolve())
        artifact_dir = str(data.get("artifact_dir", "") or "").strip()
        if artifact_dir and not Path(artifact_dir).expanduser().is_absolute():
            artifact_dir = str((base / artifact_dir).resolve())
        brief_file = str(data.get("brief_file", "") or "").strip()
        if brief_file and not Path(brief_file).expanduser().is_absolute():
            brief_file = str((base / brief_file).resolve())
        config = cls(
            enabled=bool(data.get("enabled", False)),
            system_id=str(
                data.get("system_id", "research-queue-prototype")
                or "research-queue-prototype"
            ),
            state_dir=state_dir,
            artifact_dir=artifact_dir,
            brief=str(data.get("brief", "") or "").strip(),
            brief_file=brief_file,
            limits=limits,
            concurrency=concurrency,
            models=models,
            execution=execution,
            gpu=gpu,
            budgets=budgets,
        )
        config.validate()
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> ResearchQueueConfig:
        config_path = Path(path).expanduser().resolve()
        value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(value, Mapping):
            raise TypeError("research queue config must be a mapping")
        return cls.from_mapping(value, base_dir=config_path.parent)

    def resolved_brief(self) -> str:
        if self.brief_file:
            return Path(self.brief_file).read_text(encoding="utf-8").strip()
        return self.brief.strip()

    def validate(self) -> None:
        if self.limits.candidate_target < 1:
            raise ValueError("candidate_target must be positive")
        if self.limits.generation_batch_size < 1:
            raise ValueError("generation_batch_size must be positive")
        if self.limits.max_active_ideas < 1:
            raise ValueError("max_active_ideas must be positive")
        if self.limits.max_revisions_per_idea < 1:
            raise ValueError("max_revisions_per_idea must be positive")
        if self.limits.max_runs_per_budget < 1:
            raise ValueError("max_runs_per_budget must be positive")
        if self.limits.max_steps_per_idea < 1:
            raise ValueError("max_steps_per_idea must be positive")
        if self.concurrency.max_llm_jobs < 1:
            raise ValueError("max_llm_jobs must be positive")
        if self.concurrency.max_run_jobs < 1:
            raise ValueError("max_run_jobs must be positive")
        if self.gpu.max_total_gpus < 0:
            raise ValueError("max_total_gpus cannot be negative")
        if self.gpu.max_gpus_per_run < 0:
            raise ValueError("max_gpus_per_run cannot be negative")
        if self.gpu.max_gpus_per_run > self.gpu.max_total_gpus:
            raise ValueError("max_gpus_per_run cannot exceed max_total_gpus")
        if self.execution.backend not in {"local", "clusterbridge"}:
            raise ValueError("execution.backend must be local or clusterbridge")
        for level, budget in self.budgets.items():
            if budget.gpus < 0:
                raise ValueError(f"{level.value}.gpus cannot be negative")
            if budget.timeout_sec <= 0:
                raise ValueError(f"{level.value}.timeout_sec must be positive")
