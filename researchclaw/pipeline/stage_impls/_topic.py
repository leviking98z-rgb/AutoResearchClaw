"""Stages 1-2: Topic initialization and problem decomposition."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.hardware import detect_hardware, ensure_torch_available
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._domain import _detect_domain
from researchclaw.pipeline._helpers import (
    StageResult,
    _bind_stage_role,
    _get_evolution_overlay,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _execute_topic_init(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    llm = _bind_stage_role(llm, Stage.TOPIC_INIT)
    topic = config.research.topic
    selected_topic_path = Path(
        str(getattr(config.research, "selected_topic_file", "") or "")
    ).expanduser()
    selected_topic: dict[str, Any] = {}
    if str(selected_topic_path) not in {"", "."}:
        try:
            loaded = json.loads(selected_topic_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                selected_topic = loaded
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            logger.warning(
                "Configured selected topic artifact could not be read: %s",
                selected_topic_path,
            )
    if selected_topic:
        topic = str(selected_topic.get("title", topic) or topic).strip()
        (stage_dir / "selected_topic.json").write_text(
            json.dumps(selected_topic, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (stage_dir / "topic.md").write_text(topic + "\n", encoding="utf-8")
    domains = (
        ", ".join(config.research.domains) if config.research.domains else "general"
    )
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "topic_init")
        sp = _pm.for_stage(
            "topic_init",
            evolution_overlay=_overlay,
            topic=topic,
            domains=domains,
            project_name=config.project.name,
            quality_threshold=config.research.quality_threshold,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        goal_md = resp.content
    else:
        goal_md = f"""# Research Goal

## Topic
{topic}

## Scope
Investigate the topic with emphasis on reproducible methods and measurable outcomes.

## SMART Goal
- Specific: Build a focused research plan for {topic}
- Measurable: Produce literature shortlist, hypotheses, experiment plan, and final paper
- Achievable: Complete through staged pipeline with gate checks
- Relevant: Aligned with project {config.project.name}
- Time-bound: Constrained by pipeline execution budget

## Constraints
- Quality threshold: {config.research.quality_threshold}
- Daily paper target: {config.research.daily_paper_count}

## Success Criteria
- At least 2 falsifiable hypotheses
- Executable experiment code and results analysis
- Revised paper passing quality gate

## Generated
{_utcnow_iso()}
"""
    if selected_topic:
        primary_hypothesis = str(
            selected_topic.get("falsifiable_hypothesis", "") or ""
        ).strip()
        primary_metric = str(
            selected_topic.get("primary_metric", "") or ""
        ).strip()
        goal_md = (
            "# Autonomous Selection Contract\n\n"
            f"- **Selected topic:** {topic}\n"
            f"- **Primary hypothesis:** {primary_hypothesis}\n"
            f"- **Primary metric:** {primary_metric}\n"
            f"- **Candidate ID:** {selected_topic.get('id', '')}\n\n"
            "The concrete topic above is authoritative for every downstream "
            "stage. The broader campaign brief is policy context only.\n\n"
            + goal_md
        )
    (stage_dir / "goal.md").write_text(goal_md, encoding="utf-8")

    # --- Hardware detection (GPU / MPS / CPU) ---
    # When using ssh_remote, detect hardware on the remote host instead of locally
    _ssh_cfg = config.experiment.ssh_remote if config.experiment.mode == "ssh_remote" else None
    _cb_cfg = (
        config.experiment.clusterbridge
        if config.experiment.mode == "clusterbridge"
        else None
    )
    hw = detect_hardware(
        ssh_config=_ssh_cfg,
        clusterbridge_config=_cb_cfg,
    )
    if config.experiment.mode == "clusterbridge_pool":
        # The control machine is intentionally GPU-less; the prepared pool is
        # the experiment hardware.  Describe it directly so later code
        # generation does not fall back to CPU-only guidance.
        from researchclaw.cluster import ClusterBridgePoolConfig
        from researchclaw.hardware import HardwareProfile

        try:
            _pool_cfg = ClusterBridgePoolConfig.from_file(
                config.experiment.clusterbridge_pool.config_file
            )
            _head = _pool_cfg.head_node
            _cb_probe = type(
                "_ClusterBridgeProbe",
                (),
                {
                    "node": _head.address,
                    "cb_command": _pool_cfg.cb_command,
                },
            )()
            _head_hw = detect_hardware(clusterbridge_config=_cb_probe)
            if _head_hw.has_gpu:
                hw = HardwareProfile(
                    has_gpu=True,
                    gpu_type="cuda",
                    gpu_name=(
                        f"{_head_hw.gpu_name}; Ray pool "
                        f"{_pool_cfg.configured_gpu_count} GPUs/"
                        f"{len(_pool_cfg.nodes)} nodes"
                    ),
                    vram_mb=_head_hw.vram_mb,
                    tier="high",
                    warning="",
                )
        except Exception as _pool_hw_error:  # noqa: BLE001
            logger.warning(
                "ClusterBridge pool hardware detection failed: %s",
                _pool_hw_error,
            )
    (stage_dir / "hardware_profile.json").write_text(
        json.dumps(hw.to_dict(), indent=2), encoding="utf-8"
    )
    if hw.warning:
        logger.warning("Hardware advisory: %s", hw.warning)
    else:
        logger.info("Hardware detected: %s (%s, %s MB VRAM)", hw.gpu_name, hw.gpu_type, hw.vram_mb)

    # --- Optionally ensure PyTorch is available ---
    if hw.has_gpu and config.experiment.mode == "sandbox":
        torch_ok = ensure_torch_available(config.experiment.sandbox.python_path, hw.gpu_type)
        if torch_ok:
            logger.info("PyTorch is available for sandbox experiments")
        else:
            logger.warning("PyTorch could not be installed; sandbox will use CPU-only packages")
    elif hw.has_gpu and config.experiment.mode == "docker":
        logger.info("Docker sandbox: PyTorch pre-installed in container image")
    elif hw.has_gpu and config.experiment.mode == "clusterbridge":
        logger.info(
            "ClusterBridge sandbox: PyTorch/CUDA are provided by the remote node"
        )
    elif config.experiment.mode == "clusterbridge_pool":
        logger.info(
            "ClusterBridge pool sandbox: PyTorch/CUDA are provided by the "
            "prepared multi-node Ray pool"
        )

    artifacts = ["goal.md", "hardware_profile.json"]
    evidence_refs = ["stage-01/goal.md", "stage-01/hardware_profile.json"]
    if selected_topic:
        artifacts.extend(("selected_topic.json", "topic.md"))
        evidence_refs.extend(
            ("stage-01/selected_topic.json", "stage-01/topic.md")
        )
    return StageResult(
        stage=Stage.TOPIC_INIT,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(evidence_refs),
    )


def _execute_problem_decompose(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    llm = _bind_stage_role(llm, Stage.PROBLEM_DECOMPOSE)
    goal_text = _read_prior_artifact(run_dir, "goal.md") or ""
    topic = (_read_prior_artifact(run_dir, "topic.md") or config.research.topic).strip()
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "problem_decompose")
        sp = _pm.for_stage(
            "problem_decompose",
            evolution_overlay=_overlay,
            topic=topic,
            goal_text=goal_text,
        )
        resp = llm.chat(
            [{"role": "user", "content": sp.user}],
            system=sp.system,
        )
        body = resp.content
    else:
        body = f"""# Problem Decomposition

## Source
Derived from `goal.md` for topic: {topic}

## Sub-questions
1. Which problem settings and benchmarks define current SOTA?
2. Which methodological gaps remain unresolved?
3. Which hypotheses are testable under realistic constraints?
4. Which datasets and metrics best discriminate method quality?
5. Which failure modes can invalidate expected gains?

## Priority Ranking
1. Problem framing and benchmark setup
2. Gap identification and hypothesis formulation
3. Experiment and metric design
4. Failure analysis and robustness checks

## Risks
- Ambiguous task definition
- Dataset leakage or metric mismatch

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "problem_tree.md").write_text(body, encoding="utf-8")

    # IMP-35: Topic/title quality pre-evaluation
    # Quick LLM check: is the topic well-scoped for a conference paper?
    if llm is not None:
        try:
            _eval_resp = llm.chat(
                [
                    {
                        "role": "user",
                        "content": (
                            "Evaluate this research topic for a top ML conference paper. "
                            "Score 1-10 on: (a) novelty, (b) specificity, (c) feasibility. "
                            "If overall score < 5, suggest a refined topic.\n\n"
                            f"Topic: {topic}\n\n"
                            "Reply as JSON: {\"novelty\": N, \"specificity\": N, "
                            "\"feasibility\": N, \"overall\": N, \"suggestion\": \"...\"}"
                        ),
                    }
                ],
                system=(
                    f"You are a senior {_detect_domain(topic, config.research.domains)[1]} "
                    f"researcher evaluating research topic quality."
                ),
            )
            _eval_data = _safe_json_loads(_eval_resp.content, {})
            if isinstance(_eval_data, dict):
                overall = _eval_data.get("overall", 10)
                if isinstance(overall, (int, float)) and overall < 5:
                    logger.warning(
                        "IMP-35: Topic quality score %s/10 — consider refining: %s",
                        overall,
                        _eval_data.get("suggestion", ""),
                    )
                else:
                    logger.info("IMP-35: Topic quality score %s/10", overall)
                (stage_dir / "topic_evaluation.json").write_text(
                    json.dumps(_eval_data, indent=2), encoding="utf-8"
                )
        except Exception:  # noqa: BLE001
            logger.debug("IMP-35: Topic evaluation skipped (non-blocking)")

    return StageResult(
        stage=Stage.PROBLEM_DECOMPOSE,
        status=StageStatus.DONE,
        artifacts=("problem_tree.md",),
        evidence_refs=("stage-02/problem_tree.md",),
    )
