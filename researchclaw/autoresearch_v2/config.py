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
    max_gpu_submissions_per_tick: int = 2
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
class StorageConfig:
    """Hot database placement and durable shared backup policy."""

    database_path: str = ""
    database_backup_path: str = ""
    backup_interval_sec: float = 60.0


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Controller-owned policy for executing generated experiment projects."""

    python_executable: str = "python"
    attestation_key_file: str = ""
    attestation_key_id: str = "autoresearch-v2-controller"
    smoke_timeout_sec: float = 300.0
    smoke_environment: str = "auto"
    gpu_dependency_mode: str = "online"
    gpu_cache_dir: str = ""
    gpu_cache_archive: str = ""
    # In offline mode these are the exact immutable resources staged on every
    # GPU node.  Keeping the manifest in config lets the Idea board and typed
    # protocol compiler fail before spending LLM/GPU attempts on cache misses.
    available_models: tuple[str, ...] = ()
    available_datasets: tuple[str, ...] = ()
    allowed_env_keys: tuple[str, ...] = (
        "AUTORESEARCH_V2_ATTEMPT_ID",
        "AUTORESEARCH_V2_GPU_COUNT",
        "AUTORESEARCH_V2_IDEA_ID",
        "AUTORESEARCH_V2_JOB_ID",
        "AUTORESEARCH_V2_OUTPUT_DIR",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_ENABLE_HF_TRANSFER",
    )


@dataclass(frozen=True, slots=True)
class GPUResourceManagerConfig:
    """Elastic ClusterBridge allocation policy.

    The resource manager owns the node lease.  AutoResearch materializes a
    ClusterBridge/Ray pool from the returned allocation and may reconnect it
    without restarting the controller.
    """

    owner: str = ""
    cb_command: str = (
        "/root/shared/.clusters/.tools/clusterbridge.sh"
    )
    project: str = "AutoResearchClaw-v2"
    purpose: str = "AutoResearch v2 elastic GPU pool"
    # Retained for backward-compatible config parsing. Runtime request size is
    # driven only by durable Job demand and capped by max_gpus.
    min_gpus: int = 0
    desired_gpus: int = 1
    max_gpus: int = 64
    duration_min: int = 1440
    renew_ttl_min: int = 1440
    renew_interval_sec: float = 900.0
    reconcile_interval_sec: float = 15.0
    allow_cross_cluster: bool = True
    gpu_type: str = ""
    priority: str = "normal"
    release_on_shutdown: bool = False
    preferred_allocation_id: str = ""
    log_root: str = (
        "/root/shared/.clusters/.tmp/autoresearch-v2/elastic-pools"
    )
    ray_command: str = "/opt/conda/envs/torch-base/bin/ray"
    ray_python: str = "/opt/conda/envs/torch-base/bin/python3"
    ray_port: int = 6379
    command_timeout_sec: float = 180.0
    prepare_timeout_sec: float = 900.0


@dataclass(frozen=True, slots=True)
class GPUConfig:
    enabled: bool = False
    mode: str = "static_pool"
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
    resource_manager: GPUResourceManagerConfig = field(
        default_factory=GPUResourceManagerConfig
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
class ResearchMemoryConfig:
    """Best-effort projection of durable research state into InfoHub."""

    enabled: bool = True
    url: str = "http://127.0.0.1:8077"
    timeout_sec: float = 10.0
    reconcile_interval_ticks: int = 15


@dataclass(frozen=True, slots=True)
class UsageMonitoringConfig:
    """Usage accounting, alerting, and optional estimated pricing."""

    enabled: bool = True
    history_hours: int = 168
    bucket_minutes: int = 60
    warning_threshold: float = 0.50
    critical_threshold: float = 0.80
    token_burn_warning_per_hour: int = 1_000_000
    single_call_token_warning: int = 100_000
    gpu_idle_warning_minutes: int = 30
    monthly_token_budget: int = 0
    monthly_gpu_hours_budget: float = 0.0
    monthly_cost_budget_usd: float = 0.0
    gpu_hour_cost_usd: float = 0.0
    model_prices: dict[str, dict[str, float]] = field(default_factory=dict)


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
    storage: StorageConfig = field(default_factory=StorageConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)
    research_memory: ResearchMemoryConfig = field(
        default_factory=ResearchMemoryConfig
    )
    usage_monitoring: UsageMonitoringConfig = field(
        default_factory=UsageMonitoringConfig
    )

    @property
    def root(self) -> Path:
        return Path(self.state_dir).expanduser().resolve()

    @property
    def database_path(self) -> Path:
        configured = self.storage.database_path.strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return self.root / "autoresearch.db"

    @property
    def database_backup_path(self) -> Path | None:
        configured = self.storage.database_backup_path.strip()
        if configured:
            return Path(configured).expanduser().resolve()
        if self.database_path != self.root / "autoresearch.db":
            return self.root / "autoresearch.db.backup"
        return None

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
            max_gpu_submissions_per_tick=int(
                concurrency_raw.get("max_gpu_submissions_per_tick", 2)
            ),
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
        storage_raw = _mapping(data.get("storage"), "storage")
        storage = StorageConfig(
            database_path=str(
                storage_raw.get("database_path", "") or ""
            ),
            database_backup_path=str(
                storage_raw.get("database_backup_path", "") or ""
            ),
            backup_interval_sec=float(
                storage_raw.get("backup_interval_sec", 60.0)
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

        def _string_tuple(raw: object) -> tuple[str, ...]:
            if isinstance(raw, str):
                values = raw.split(",")
            else:
                values = raw or ()
            return tuple(
                str(value).strip()
                for value in values
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
            gpu_dependency_mode=str(
                execution_raw.get("gpu_dependency_mode", "online")
                or "online"
            )
            .strip()
            .casefold(),
            gpu_cache_dir=str(
                execution_raw.get("gpu_cache_dir", "") or ""
            ).strip(),
            gpu_cache_archive=str(
                execution_raw.get("gpu_cache_archive", "") or ""
            ).strip(),
            available_models=_string_tuple(
                execution_raw.get("available_models", ())
            ),
            available_datasets=_string_tuple(
                execution_raw.get("available_datasets", ())
            ),
            allowed_env_keys=(
                allowed_env_keys or ExecutionConfig().allowed_env_keys
            ),
        )
        gpu_raw = _mapping(data.get("gpu"), "gpu")
        resource_manager_raw = _mapping(
            gpu_raw.get("resource_manager"),
            "gpu.resource_manager",
        )
        resource_manager = GPUResourceManagerConfig(
            owner=str(resource_manager_raw.get("owner", "") or ""),
            cb_command=str(
                resource_manager_raw.get(
                    "cb_command",
                    "/root/shared/.clusters/.tools/clusterbridge.sh",
                )
                or ""
            ),
            project=str(
                resource_manager_raw.get(
                    "project",
                    "AutoResearchClaw-v2",
                )
                or ""
            ),
            purpose=str(
                resource_manager_raw.get(
                    "purpose",
                    "AutoResearch v2 elastic GPU pool",
                )
                or ""
            ),
            min_gpus=int(resource_manager_raw.get("min_gpus", 0)),
            desired_gpus=int(
                resource_manager_raw.get("desired_gpus", 1)
            ),
            max_gpus=int(resource_manager_raw.get("max_gpus", 64)),
            duration_min=int(
                resource_manager_raw.get("duration_min", 1440)
            ),
            renew_ttl_min=int(
                resource_manager_raw.get("renew_ttl_min", 1440)
            ),
            renew_interval_sec=float(
                resource_manager_raw.get("renew_interval_sec", 900.0)
            ),
            reconcile_interval_sec=float(
                resource_manager_raw.get(
                    "reconcile_interval_sec",
                    15.0,
                )
            ),
            allow_cross_cluster=bool(
                resource_manager_raw.get("allow_cross_cluster", True)
            ),
            gpu_type=str(
                resource_manager_raw.get("gpu_type", "") or ""
            ),
            priority=str(
                resource_manager_raw.get("priority", "normal")
                or "normal"
            ),
            release_on_shutdown=bool(
                resource_manager_raw.get("release_on_shutdown", False)
            ),
            preferred_allocation_id=str(
                resource_manager_raw.get(
                    "preferred_allocation_id",
                    "",
                )
                or ""
            ).strip(),
            log_root=str(
                resource_manager_raw.get(
                    "log_root",
                    "/root/shared/.clusters/.tmp/"
                    "autoresearch-v2/elastic-pools",
                )
                or ""
            ),
            ray_command=str(
                resource_manager_raw.get(
                    "ray_command",
                    "/opt/conda/envs/torch-base/bin/ray",
                )
                or ""
            ),
            ray_python=str(
                resource_manager_raw.get(
                    "ray_python",
                    "/opt/conda/envs/torch-base/bin/python3",
                )
                or ""
            ),
            ray_port=int(resource_manager_raw.get("ray_port", 6379)),
            command_timeout_sec=float(
                resource_manager_raw.get("command_timeout_sec", 180.0)
            ),
            prepare_timeout_sec=float(
                resource_manager_raw.get("prepare_timeout_sec", 900.0)
            ),
        )
        gpu = GPUConfig(
            enabled=bool(gpu_raw.get("enabled", False)),
            mode=str(gpu_raw.get("mode", "static_pool") or "static_pool")
            .strip()
            .casefold(),
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
            resource_manager=resource_manager,
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
        research_memory_raw = _mapping(
            data.get("research_memory"),
            "research_memory",
        )
        research_memory = ResearchMemoryConfig(
            enabled=bool(research_memory_raw.get("enabled", True)),
            url=str(
                research_memory_raw.get(
                    "url",
                    literature.url,
                )
                or literature.url
            ),
            timeout_sec=float(
                research_memory_raw.get("timeout_sec", 10.0)
            ),
            reconcile_interval_ticks=int(
                research_memory_raw.get(
                    "reconcile_interval_ticks",
                    15,
                )
            ),
        )
        usage_raw = _mapping(
            data.get("usage_monitoring"),
            "usage_monitoring",
        )
        prices_raw = _mapping(
            usage_raw.get("model_prices"),
            "usage_monitoring.model_prices",
        )
        model_prices: dict[str, dict[str, float]] = {}
        for raw_model, raw_price in prices_raw.items():
            price = _mapping(
                raw_price,
                f"usage_monitoring.model_prices[{raw_model!r}]",
            )
            model_prices[str(raw_model)] = {
                "input_per_million_usd": float(
                    price.get("input_per_million_usd", 0.0)
                ),
                "output_per_million_usd": float(
                    price.get("output_per_million_usd", 0.0)
                ),
            }
        usage_monitoring = UsageMonitoringConfig(
            enabled=bool(usage_raw.get("enabled", True)),
            history_hours=int(usage_raw.get("history_hours", 168)),
            bucket_minutes=int(usage_raw.get("bucket_minutes", 60)),
            warning_threshold=float(
                usage_raw.get("warning_threshold", 0.50)
            ),
            critical_threshold=float(
                usage_raw.get("critical_threshold", 0.80)
            ),
            token_burn_warning_per_hour=int(
                usage_raw.get(
                    "token_burn_warning_per_hour",
                    1_000_000,
                )
            ),
            single_call_token_warning=int(
                usage_raw.get("single_call_token_warning", 100_000)
            ),
            gpu_idle_warning_minutes=int(
                usage_raw.get("gpu_idle_warning_minutes", 30)
            ),
            monthly_token_budget=int(
                usage_raw.get("monthly_token_budget", 0)
            ),
            monthly_gpu_hours_budget=float(
                usage_raw.get("monthly_gpu_hours_budget", 0.0)
            ),
            monthly_cost_budget_usd=float(
                usage_raw.get("monthly_cost_budget_usd", 0.0)
            ),
            gpu_hour_cost_usd=float(
                usage_raw.get("gpu_hour_cost_usd", 0.0)
            ),
            model_prices=model_prices,
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
            storage=storage,
            execution=execution,
            gpu=gpu,
            models=models,
            literature=literature,
            research_memory=research_memory,
            usage_monitoring=usage_monitoring,
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
            self.concurrency.max_gpu_submissions_per_tick,
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
        if self.gpu.mode not in {"static_pool", "resource_manager"}:
            raise ValueError(
                "gpu.mode must be static_pool or resource_manager"
            )
        if (
            self.gpu.enabled
            and self.gpu.mode == "static_pool"
            and not self.gpu.pool_config
        ):
            raise ValueError(
                "gpu.pool_config is required in static_pool mode"
            )
        if self.gpu.enabled and self.gpu.mode == "resource_manager":
            elastic = self.gpu.resource_manager
            if not elastic.owner.strip():
                raise ValueError(
                    "gpu.resource_manager.owner is required in "
                    "resource_manager mode"
                )
            if elastic.max_gpus <= 0:
                raise ValueError(
                    "gpu.resource_manager.max_gpus must be positive"
                )
            if min(
                elastic.duration_min,
                elastic.renew_ttl_min,
                elastic.renew_interval_sec,
                elastic.reconcile_interval_sec,
                elastic.command_timeout_sec,
                elastic.prepare_timeout_sec,
            ) <= 0:
                raise ValueError(
                    "gpu.resource_manager durations must be positive"
                )
            if (
                elastic.renew_interval_sec
                >= elastic.renew_ttl_min * 60
            ):
                raise ValueError(
                    "gpu.resource_manager.renew_interval_sec must be "
                    "shorter than renew_ttl_min"
                )
            if not all(
                value.strip()
                for value in (
                    elastic.cb_command,
                    elastic.project,
                    elastic.purpose,
                    elastic.log_root,
                    elastic.ray_command,
                    elastic.ray_python,
                )
            ):
                raise ValueError(
                    "gpu.resource_manager command, project, purpose, "
                    "log_root, and Ray fields are required"
                )
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
        if self.storage.backup_interval_sec <= 0:
            raise ValueError(
                "storage.backup_interval_sec must be positive"
            )
        backup_path = self.database_backup_path
        if (
            backup_path is not None
            and backup_path == self.database_path
        ):
            raise ValueError(
                "storage.database_backup_path must differ from "
                "storage.database_path"
            )
        if self.execution.smoke_timeout_sec <= 0:
            raise ValueError("execution.smoke_timeout_sec must be positive")
        if self.research_memory.timeout_sec <= 0:
            raise ValueError(
                "research_memory.timeout_sec must be positive"
            )
        if self.research_memory.reconcile_interval_ticks < 1:
            raise ValueError(
                "research_memory.reconcile_interval_ticks must be positive"
            )
        if (
            self.research_memory.enabled
            and not self.research_memory.url.strip()
        ):
            raise ValueError(
                "research_memory.url is required when enabled"
            )
        usage = self.usage_monitoring
        if usage.history_hours < 1 or usage.bucket_minutes < 1:
            raise ValueError(
                "usage_monitoring history and bucket sizes must be positive"
            )
        if not 0 < usage.warning_threshold < usage.critical_threshold <= 1:
            raise ValueError(
                "usage_monitoring thresholds must satisfy "
                "0 < warning < critical <= 1"
            )
        if min(
            usage.token_burn_warning_per_hour,
            usage.single_call_token_warning,
        ) < 0:
            raise ValueError(
                "usage_monitoring token thresholds must be non-negative"
            )
        if usage.gpu_idle_warning_minutes < 0:
            raise ValueError(
                "usage_monitoring GPU idle threshold must be non-negative"
            )
        if min(
            usage.monthly_token_budget,
            usage.monthly_gpu_hours_budget,
            usage.monthly_cost_budget_usd,
            usage.gpu_hour_cost_usd,
        ) < 0:
            raise ValueError(
                "usage_monitoring budgets and prices must be non-negative"
            )
        for model, prices in usage.model_prices.items():
            if not str(model).strip() or min(prices.values()) < 0:
                raise ValueError(
                    "usage_monitoring model prices must use non-empty "
                    "models and non-negative values"
                )
        if self.execution.smoke_environment not in {
            "auto",
            "local",
            "gpu_pool",
        }:
            raise ValueError(
                "execution.smoke_environment must be one of "
                "auto, local, gpu_pool"
            )
        if self.execution.gpu_dependency_mode not in {
            "online",
            "offline",
        }:
            raise ValueError(
                "execution.gpu_dependency_mode must be online or offline"
            )
        if (
            self.execution.gpu_dependency_mode == "offline"
            and self.gpu.enabled
            and not self.execution.gpu_cache_dir
        ):
            raise ValueError(
                "execution.gpu_cache_dir is required for offline GPU "
                "dependencies"
            )
        if (
            self.execution.gpu_cache_archive
            and not self.execution.gpu_cache_dir
        ):
            raise ValueError(
                "execution.gpu_cache_dir is required when "
                "gpu_cache_archive is configured"
            )
        if (
            self.execution.gpu_dependency_mode == "offline"
            and self.gpu.enabled
            and not self.execution.available_models
        ):
            raise ValueError(
                "execution.available_models is required for offline GPU "
                "dependencies"
            )
        if (
            self.execution.gpu_dependency_mode == "offline"
            and self.gpu.enabled
            and not self.execution.available_datasets
        ):
            raise ValueError(
                "execution.available_datasets is required for offline GPU "
                "dependencies"
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
