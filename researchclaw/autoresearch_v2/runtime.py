"""Runtime assembly for production and simulation deployments."""

from __future__ import annotations

from typing import Any

from .config import V2Config
from .controller import V2Controller
from .gates import LLMDecisionGate
from .gpu import build_clusterbridge_broker
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
from .store import V2Store
from .validation import validate_build_output, validate_plan


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
            validator=validate_plan,
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
        )
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
    broker = (
        build_clusterbridge_broker(
            config.gpu.pool_config,
            reserved_gpus=config.gpu.reserved_gpus,
            max_share_per_idea=config.gpu.max_share_per_idea,
            target_utilization=config.gpu.target_utilization,
            probe_failure_threshold=(
                config.gpu.probe_failure_threshold
            ),
        )
        if config.gpu.enabled
        else None
    )
    return V2Controller(
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
    )
