"""Runtime assembly for production and simulation deployments."""

from __future__ import annotations

from typing import Any

from researchclaw.experiment.clusterbridge_pool import ClusterPoolError

from .config import V2Config
from .controller import V2Controller
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
from .store import V2Store
from .validation import validate_build_output


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
    store = V2Store(config.root)
    router = RoleRouter(
        config.models.researchclaw_config,
        audit_root=config.root / "llm-audit",
        decision_role=config.models.decision_role,
        worker_role=config.models.worker_role,
        utility_role=config.models.utility_role,
    )
    literature = InfoHubLiteratureProvider(config.literature)
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
    configured_gpu_capacity = (
        clusterbridge_capacity(config.gpu.pool_config)
        if config.gpu.enabled
        else 0
    )
    broker = None
    gpu_broker_error = ""
    if config.gpu.enabled:
        try:
            broker = build_clusterbridge_broker(
                config.gpu.pool_config,
                reserved_gpus=config.gpu.reserved_gpus,
                max_share_per_idea=config.gpu.max_share_per_idea,
                target_utilization=config.gpu.target_utilization,
                probe_failure_threshold=(
                    config.gpu.probe_failure_threshold
                ),
            )
        except ClusterPoolError as exc:
            # Idea generation, Design, and Build do not require a live physical
            # allocation. Keep the Controller useful while the GPU pool is
            # released; GPU jobs remain READY until a restart can adopt a
            # freshly claimed/prepared pool.
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
        configured_gpu_capacity=configured_gpu_capacity,
    )
    if gpu_broker_error:
        controller.store.initialize()
        controller.store.event(
            "gpu_broker_unavailable",
            error=gpu_broker_error,
            configured_gpu_capacity=configured_gpu_capacity,
        )
    return controller
