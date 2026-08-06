"""Runtime assembly for production and simulation deployments."""

from __future__ import annotations

import os
from typing import Any

from researchclaw.experiment.clusterbridge_pool import ClusterPoolError

from .config import V2Config
from .controller import V2Controller
from .elastic_gpu import ResourceManagedGPUManager
from .gates import LLMDecisionGate
from .gpu import build_clusterbridge_broker, clusterbridge_capacity
from .ideas import IdeaGenerator, LLMBoardIdeaGenerator
from .jobs import (
    BuildJobExecutor,
    DesignJobExecutor,
    ExperimentJobExecutor,
    ReportJobExecutor,
)
from .literature import InfoHubLiteratureProvider
from .llm import RoleRouter, StructuredRole
from .models import JobKind
from .protocols import validate_protocol_draft
from .research_memory import InfoHubResearchMemory
from .store import V2Store
from .validation import validate_build_output


def build_store(config: V2Config) -> V2Store:
    """Build the canonical store shared by controller, CLI, and dashboard."""

    return V2Store(
        config.root,
        db_path=config.database_path,
        db_backup_path=config.database_backup_path,
        backup_interval_sec=config.storage.backup_interval_sec,
    )


def _validate_report(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(value.get("title", "") or "").strip():
        errors.append("missing title")
    if not isinstance(value.get("claims"), list):
        errors.append("claims must be a list")
    if not isinstance(value.get("limitations"), list):
        errors.append("limitations must be a list")
    if not str(value.get("paper_markdown", "") or "").strip():
        errors.append("missing paper_markdown")
    return errors


def build_production_controller(
    config: V2Config,
    *,
    generator: IdeaGenerator | None = None,
) -> V2Controller:
    store = build_store(config)
    router = RoleRouter(
        config.models.researchclaw_config,
        audit_root=config.root / "llm-audit",
        decision_role=config.models.decision_role,
        worker_role=config.models.worker_role,
        utility_role=config.models.utility_role,
    )
    literature = InfoHubLiteratureProvider(config.literature)
    research_memory = InfoHubResearchMemory(
        config=config.research_memory,
        system_id=config.system_id,
        store=store,
    )
    idea_generator = generator or LLMBoardIdeaGenerator(
        llm=router.decision,
        brief=config.topic_brief,
        literature=literature,
        utility_llm=router.utility,
    )
    decision_gate = LLMDecisionGate(client=router.decision)
    design = DesignJobExecutor(
        StructuredRole(
            client=router.worker,
            system=(
                "You are a rigorous experiment designer. Return only JSON; "
                "make the cheapest experiment scientifically discriminating."
            ),
            validator=validate_protocol_draft,
        ),
        decision_gate=decision_gate,
        max_revisions=config.budgets.max_design_revisions,
    )
    build = BuildJobExecutor(
        StructuredRole(
            client=router.worker,
            system=(
                "You are a senior ML systems engineer. Return only complete "
                "JSON project snapshots, never patches."
            ),
            validator=validate_build_output,
        ),
        smoke_timeout_sec=config.execution.smoke_timeout_sec,
        python_executable=config.execution.python_executable,
        execute_smoke_locally=(
            config.execution.smoke_environment == "local"
            or (
                config.execution.smoke_environment == "auto"
                and not config.gpu.enabled
            )
        ),
    )
    report = ReportJobExecutor(
        StructuredRole(
            client=router.worker,
            system=(
                "You are an evidence-bound scientific writer. Return only "
                "JSON and never invent results or citations."
            ),
            validator=_validate_report,
        ),
        decision_gate=decision_gate,
    )
    task_env = {
        key: value
        for key in config.execution.allowed_env_keys
        if (value := os.environ.get(key)) is not None
        and not (
            key in {
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            }
            and str(value).strip().casefold()
            in {"1", "true", "yes", "on"}
        )
    }
    configured_gpu_capacity = 0
    broker = None
    gpu_manager = None
    gpu_broker_error = ""
    if config.gpu.enabled:
        if config.gpu.mode == "resource_manager":
            configured_gpu_capacity = (
                config.gpu.resource_manager.desired_gpus
            )
            gpu_manager = ResourceManagedGPUManager(
                config.gpu,
                task_env=task_env,
            )
            gpu_manager.bootstrap()
            broker = gpu_manager.broker
            manager_snapshot = gpu_manager.snapshot()
            if broker is None and manager_snapshot.get("last_error"):
                gpu_broker_error = str(manager_snapshot["last_error"])
        else:
            configured_gpu_capacity = clusterbridge_capacity(
                config.gpu.pool_config
            )
            try:
                broker = build_clusterbridge_broker(
                    config.gpu.pool_config,
                    reserved_gpus=config.gpu.reserved_gpus,
                    max_share_per_idea=config.gpu.max_share_per_idea,
                    target_utilization=config.gpu.target_utilization,
                    probe_failure_threshold=(
                        config.gpu.probe_failure_threshold
                    ),
                    task_env=task_env,
                )
            except ClusterPoolError as exc:
                # Idea generation, Design, and Build do not require a live
                # physical allocation. GPU jobs remain READY while elastic
                # deployments hot-reconnect or static deployments restart.
                gpu_broker_error = f"{type(exc).__name__}: {exc}"
    controller = V2Controller(
        config=config,
        store=store,
        generator=idea_generator,
        executors={
            JobKind.DESIGN: design,
            JobKind.BUILD: build,
            JobKind.PILOT: ExperimentJobExecutor(
                decision_gate=decision_gate
            ),
            JobKind.SCALE: ExperimentJobExecutor(
                decision_gate=decision_gate
            ),
            JobKind.REPORT: report,
        },
        gpu_broker=broker,
        gpu_manager=gpu_manager,
        configured_gpu_capacity=configured_gpu_capacity,
        research_memory=research_memory,
    )
    if gpu_broker_error:
        controller.store.initialize(recover_filesystem=False)
        controller.store.event(
            "gpu_broker_unavailable",
            error=gpu_broker_error,
            configured_gpu_capacity=configured_gpu_capacity,
        )
    return controller
