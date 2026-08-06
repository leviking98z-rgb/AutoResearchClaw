"""Small, independent configuration model for AutoResearch v2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _mapping(value: object, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True, slots=True)
class PopulationConfig:
    reservoir_low_watermark: int = 12
    reservoir_target: int = 24
    generation_batch_size: int = 12
    active_idea_target: int = 8
    max_active_ideas: int = 10
    max_same_family: int = 2


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    minimum_score: float = 6.0
    duplicate_threshold: float = 0.72
    semantic_duplicate_threshold: float = 0.72
    require_novelty_evidence: bool = True


@dataclass(frozen=True, slots=True)
class ConcurrencyConfig:
    max_llm_jobs: int = 4
    max_cpu_jobs: int = 8
    max_gpu_jobs: int = 8
    poll_interval_sec: float = 2.0


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_build_attempts: int = 2
    max_job_attempts: int = 3
    max_design_revisions: int = 2
    pilot_gpu_hours: float = 2.0
    scale_gpu_hours: float = 32.0
    max_llm_tokens_per_idea: int = 2_000_000
    max_wall_clock_hours_per_idea: float = 72.0
    max_no_progress_hours: float = 12.0


@dataclass(frozen=True, slots=True)
class RetentionConfig:
    event_jsonl_max_mb: int = 256
    llm_audit_max_mb: int = 256
    keep_failed_attempts_per_job: int = 4
    maintenance_interval_ticks: int = 300


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Controller-owned policy for executing generated experiment projects."""

    python_executable: str = "python"
    attestation_key_file: str = ""
    attestation_key_id: str = "autoresearch-v2-controller"
    smoke_timeout_sec: float = 300.0
    smoke_environment: str = "auto"
    allowed_env_keys: tuple[str, ...] = (
        "AUTORESEARCH_V2_ATTEMPT_ID",
        "AUTORESEARCH_V2_GPU_COUNT",
        "AUTORESEARCH_V2_IDEA_ID",
        "AUTORESEARCH_V2_JOB_ID",
        "AUTORESEARCH_V2_OUTPUT_DIR",
    )


@dataclass(frozen=True, slots=True)
class GPUConfig:
    enabled: bool = False
    pool_config: str = ""
    reserved_gpus: int = 0
    target_utilization: float = 0.90
    max_share_per_idea: float = 0.50
    pilot_max_gpus: int = 2
    scale_max_gpus: int = 8
    probe_failure_threshold: int = 3
    shared_workspace_root: str = (
        "/root/shared/.clusters/.workdir/autoresearch-v2/runs"
    )


@dataclass(frozen=True, slots=True)
class LiteratureConfig:
    enabled: bool = True
    mode: str = "http"
    url: str = "http://127.0.0.1:8077"
    repo: str = "/root/servers/infohub"
    timeout_sec: float = 20.0
    search_limit: int = 30
    collect_days: int = 3650
    collect_platforms: tuple[str, ...] = ("arxiv", "scholar", "bing")
    refresh_on_low_results: bool = True
    min_results: int = 8


@dataclass(frozen=True, slots=True)
class ModelConfig:
    researchclaw_config: str = "config.rsi.yaml"
    decision_role: str = "research_director"
    worker_role: str = "coding_engineer"
    utility_role: str = "literature_researcher"


@dataclass(frozen=True, slots=True)
class V2Config:
    enabled: bool = False
    system_id: str = "autoresearch-v2"
    state_dir: str = "workspace/autoresearch-v2"
    topic_brief: str = ""
    population: PopulationConfig = field(default_factory=PopulationConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    concurrency: ConcurrencyConfig = field(default_factory=ConcurrencyConfig)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)

    @property
    def root(self) -> Path:
        return Path(self.state_dir).expanduser().resolve()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> V2Config:
        top = _mapping(raw, "config")
        data = _mapping(
            top.get("autoresearch_v2", top.get("v2", top)),
            "autoresearch_v2",
        )
        population = PopulationConfig(
            **{
                key: int(value)
                for key, value in _mapping(
                    data.get("population"), "population"
                ).items()
            }
        )
        admission_raw = _mapping(data.get("admission"), "admission")
        admission = AdmissionConfig(
            minimum_score=float(
                admission_raw.get("minimum_score", 6.0)
            ),
            duplicate_threshold=float(
                admission_raw.get("duplicate_threshold", 0.72)
            ),
            semantic_duplicate_threshold=float(
                admission_raw.get(
                    "semantic_duplicate_threshold",
                    0.72,
                )
            ),
            require_novelty_evidence=bool(
                admission_raw.get("require_novelty_evidence", True)
            ),
        )
        concurrency_raw = _mapping(data.get("concurrency"), "concurrency")
        concurrency = ConcurrencyConfig(
            max_llm_jobs=int(concurrency_raw.get("max_llm_jobs", 4)),
            max_cpu_jobs=int(concurrency_raw.get("max_cpu_jobs", 8)),
            max_gpu_jobs=int(concurrency_raw.get("max_gpu_jobs", 8)),
            poll_interval_sec=float(
                concurrency_raw.get("poll_interval_sec", 2.0)
            ),
        )
        budgets_raw = _mapping(data.get("budgets"), "budgets")
        budgets = BudgetConfig(
            max_build_attempts=int(
                budgets_raw.get("max_build_attempts", 2)
            ),
            max_job_attempts=int(budgets_raw.get("max_job_attempts", 3)),
            max_design_revisions=int(
                budgets_raw.get("max_design_revisions", 2)
            ),
            pilot_gpu_hours=float(
                budgets_raw.get("pilot_gpu_hours", 2.0)
            ),
            scale_gpu_hours=float(
                budgets_raw.get("scale_gpu_hours", 32.0)
            ),
            max_llm_tokens_per_idea=int(
                budgets_raw.get("max_llm_tokens_per_idea", 2_000_000)
            ),
            max_wall_clock_hours_per_idea=float(
                budgets_raw.get("max_wall_clock_hours_per_idea", 72.0)
            ),
            max_no_progress_hours=float(
                budgets_raw.get("max_no_progress_hours", 12.0)
            ),
        )
        retention_raw = _mapping(data.get("retention"), "retention")
        retention = RetentionConfig(
            event_jsonl_max_mb=int(
                retention_raw.get("event_jsonl_max_mb", 256)
            ),
            llm_audit_max_mb=int(
                retention_raw.get("llm_audit_max_mb", 256)
            ),
            keep_failed_attempts_per_job=int(
                retention_raw.get(
                    "keep_failed_attempts_per_job",
                    4,
                )
            ),
            maintenance_interval_ticks=int(
                retention_raw.get("maintenance_interval_ticks", 300)
            ),
        )
        execution_raw = _mapping(data.get("execution"), "execution")
        env_keys_raw = execution_raw.get(
            "allowed_env_keys",
            ExecutionConfig().allowed_env_keys,
        )
        if isinstance(env_keys_raw, str):
            allowed_env_keys = tuple(
                value.strip()
                for value in env_keys_raw.split(",")
                if value.strip()
            )
        else:
            allowed_env_keys = tuple(
                str(value).strip()
                for value in (env_keys_raw or ())
                if str(value).strip()
            )
        execution = ExecutionConfig(
            python_executable=str(
                execution_raw.get("python_executable", "python")
                or "python"
            ),
            attestation_key_file=str(
                execution_raw.get("attestation_key_file", "") or ""
            ),
            attestation_key_id=str(
                execution_raw.get(
                    "attestation_key_id",
                    "autoresearch-v2-controller",
                )
                or "autoresearch-v2-controller"
            ),
            smoke_timeout_sec=float(
                execution_raw.get("smoke_timeout_sec", 300.0)
            ),
            smoke_environment=str(
                execution_raw.get("smoke_environment", "auto") or "auto"
            )
            .strip()
            .casefold(),
            allowed_env_keys=(
                allowed_env_keys or ExecutionConfig().allowed_env_keys
            ),
        )
        gpu_raw = _mapping(data.get("gpu"), "gpu")
        gpu = GPUConfig(
            enabled=bool(gpu_raw.get("enabled", False)),
            pool_config=str(gpu_raw.get("pool_config", "") or ""),
            reserved_gpus=int(gpu_raw.get("reserved_gpus", 0)),
            target_utilization=float(
                gpu_raw.get("target_utilization", 0.90)
            ),
            max_share_per_idea=float(
                gpu_raw.get("max_share_per_idea", 0.50)
            ),
            pilot_max_gpus=int(gpu_raw.get("pilot_max_gpus", 2)),
            scale_max_gpus=int(gpu_raw.get("scale_max_gpus", 8)),
            probe_failure_threshold=int(
                gpu_raw.get("probe_failure_threshold", 3)
            ),
            shared_workspace_root=str(
                gpu_raw.get(
                    "shared_workspace_root",
                    "/root/shared/.clusters/.workdir/autoresearch-v2/runs",
                )
                or ""
            ),
        )
        models_raw = _mapping(data.get("models"), "models")
        models = ModelConfig(
            researchclaw_config=str(
                models_raw.get("researchclaw_config", "config.rsi.yaml")
            ),
            decision_role=str(
                models_raw.get("decision_role", "research_director")
            ),
            worker_role=str(
                models_raw.get("worker_role", "coding_engineer")
            ),
            utility_role=str(
                models_raw.get("utility_role", "literature_researcher")
            ),
        )
        literature_raw = _mapping(data.get("literature"), "literature")
        platforms_raw = literature_raw.get(
            "collect_platforms", ("arxiv", "scholar", "bing")
        )
        if isinstance(platforms_raw, str):
            platforms = tuple(
                value.strip()
                for value in platforms_raw.split(",")
                if value.strip()
            )
        else:
            platforms = tuple(
                str(value).strip()
                for value in (platforms_raw or ())
                if str(value).strip()
            )
        literature = LiteratureConfig(
            enabled=bool(literature_raw.get("enabled", True)),
            mode=str(literature_raw.get("mode", "http") or "http"),
            url=str(
                literature_raw.get("url", "http://127.0.0.1:8077")
                or "http://127.0.0.1:8077"
            ),
            repo=str(
                literature_raw.get("repo", "/root/servers/infohub")
                or "/root/servers/infohub"
            ),
            timeout_sec=float(literature_raw.get("timeout_sec", 20.0)),
            search_limit=int(literature_raw.get("search_limit", 30)),
            collect_days=int(literature_raw.get("collect_days", 3650)),
            collect_platforms=platforms or ("arxiv", "scholar", "bing"),
            refresh_on_low_results=bool(
                literature_raw.get("refresh_on_low_results", True)
            ),
            min_results=int(literature_raw.get("min_results", 8)),
        )
        config = cls(
            enabled=bool(data.get("enabled", False)),
            system_id=str(data.get("system_id", "autoresearch-v2")),
            state_dir=str(
                data.get("state_dir", "workspace/autoresearch-v2")
            ),
            topic_brief=str(data.get("topic_brief", "") or ""),
            population=population,
            admission=admission,
            concurrency=concurrency,
            budgets=budgets,
            retention=retention,
            execution=execution,
            gpu=gpu,
            models=models,
            literature=literature,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 < self.population.active_idea_target <= (
            self.population.max_active_ideas
        ):
            raise ValueError(
                "active_idea_target must be within max_active_ideas"
            )
        if self.population.reservoir_target < (
            self.population.reservoir_low_watermark
        ):
            raise ValueError(
                "reservoir_target must be >= reservoir_low_watermark"
            )
        if not 0 <= self.admission.minimum_score <= 10:
            raise ValueError("admission.minimum_score must be in [0,10]")
        if not 0 < self.admission.duplicate_threshold <= 1:
            raise ValueError(
                "admission.duplicate_threshold must be in (0,1]"
            )
        if not 0 < self.admission.semantic_duplicate_threshold <= 1:
            raise ValueError(
                "admission.semantic_duplicate_threshold must be in (0,1]"
            )
        if min(
            self.concurrency.max_llm_jobs,
            self.concurrency.max_cpu_jobs,
            self.concurrency.max_gpu_jobs,
        ) < 1:
            raise ValueError("concurrency limits must be positive")
        if not 0 < self.gpu.target_utilization <= 1:
            raise ValueError("gpu.target_utilization must be in (0,1]")
        if not 0 < self.gpu.max_share_per_idea <= 1:
            raise ValueError("gpu.max_share_per_idea must be in (0,1]")
        if self.gpu.probe_failure_threshold < 1:
            raise ValueError(
                "gpu.probe_failure_threshold must be positive"
            )
        if self.gpu.enabled and not self.gpu.pool_config:
            raise ValueError("gpu.pool_config is required when GPU is enabled")
        if self.gpu.enabled and not self.gpu.shared_workspace_root:
            raise ValueError(
                "gpu.shared_workspace_root is required when GPU is enabled"
            )
        if self.gpu.enabled:
            state = self.root
            shared = Path(
                self.gpu.shared_workspace_root
            ).expanduser().resolve()
            if not state.is_relative_to(shared):
                raise ValueError(
                    "state_dir must be inside gpu.shared_workspace_root so "
                    "immutable attempts are visible to ClusterBridge nodes"
                )
        if min(
            self.budgets.pilot_gpu_hours,
            self.budgets.scale_gpu_hours,
            self.budgets.max_wall_clock_hours_per_idea,
            self.budgets.max_no_progress_hours,
        ) <= 0:
            raise ValueError("budget hours must be positive")
        if min(
            self.budgets.max_build_attempts,
            self.budgets.max_job_attempts,
            self.budgets.max_design_revisions,
        ) < 1:
            raise ValueError("attempt and revision budgets must be positive")
        if min(
            self.retention.event_jsonl_max_mb,
            self.retention.llm_audit_max_mb,
            self.retention.keep_failed_attempts_per_job,
            self.retention.maintenance_interval_ticks,
        ) < 1:
            raise ValueError("retention limits must be positive")
        if self.execution.smoke_timeout_sec <= 0:
            raise ValueError("execution.smoke_timeout_sec must be positive")
        if self.execution.smoke_environment not in {
            "auto",
            "local",
            "gpu_pool",
        }:
            raise ValueError(
                "execution.smoke_environment must be one of "
                "auto, local, gpu_pool"
            )
        if not self.execution.python_executable.strip():
            raise ValueError("execution.python_executable is required")
        if not self.execution.attestation_key_id.strip():
            raise ValueError("execution.attestation_key_id is required")
        invalid_env_keys = [
            key
            for key in self.execution.allowed_env_keys
            if not key
            or not (key[0].isalpha() or key[0] == "_")
            or any(
                not (character.isalnum() or character == "_")
                for character in key
            )
        ]
        if invalid_env_keys:
            raise ValueError(
                "execution.allowed_env_keys contains invalid names: "
                + ", ".join(invalid_env_keys)
            )
        state_root = self.root
        key_file = self.execution.attestation_key_file.strip()
        if key_file:
            key_path = Path(key_file).expanduser().resolve()
            shared_root = Path(
                self.gpu.shared_workspace_root
            ).expanduser().resolve()
            if (
                key_path == state_root
                or key_path.is_relative_to(state_root)
                or key_path == shared_root
                or key_path.is_relative_to(shared_root)
            ):
                raise ValueError(
                    "execution.attestation_key_file must be outside the "
                    "Controller state and GPU-visible shared workspace"
                )

    @classmethod
    def from_file(cls, path: str | Path) -> V2Config:
        value = yaml.safe_load(
            Path(path).expanduser().read_text(encoding="utf-8")
        )
        if not isinstance(value, Mapping):
            raise TypeError("v2 config must be a mapping")
        return cls.from_mapping(value)
