"""Stage 9: Experiment design."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _bind_stage_role,
    _build_context_preamble,
    _chat_with_prompt,
    _extract_yaml_block,
    _get_evolution_overlay,
    _load_hardware_profile,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


def _normalize_plan_field(value: Any) -> list:
    """Normalize a plan field (baselines, proposed_methods, ablations, datasets)
    from any shape the LLM might produce into a flat list of items.

    Handles: list[str], list[dict], dict[str, Any], str, None.
    When the input is a dict, we preserve the full structure by converting each
    key-value pair into a dict item (with at least a 'name' key), rather than
    discarding either keys or values.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        result = []
        for k, v in value.items():
            if isinstance(v, dict):
                # e.g. {"baseline_1": {"params": ...}} -> {"name": "baseline_1", "params": ...}
                item = dict(v)
                item.setdefault("name", str(k))
                result.append(item)
            else:
                # e.g. {"baseline_1": "description"} -> {"name": "baseline_1", "description": str(v)}
                result.append({"name": str(k), "description": str(v) if v else ""})
        return result
    if isinstance(value, list):
        return list(value)
    return [value]


def _plan_field_names(items: list) -> list[str]:
    """Extract string names from a normalized plan field for display/dedup."""
    result = []
    for item in items:
        if isinstance(item, dict):
            result.append(item.get("name", str(item)))
        else:
            result.append(str(item))
    return result


_RSI_TOPIC_MARKERS = (
    "recursive self-improvement",
    "self-improvement",
    "self improvement",
    "self-iterative",
    "self iterative",
    "acceptance gate",
    "accept/reject",
    "rollback",
    "llm agent",
    "language model",
    "calibration-aware",
    "calibration aware",
)
_VISION_DOMAIN_ADAPTATION_MARKERS = (
    "cifar",
    "imagenet",
    "mnist",
    "fashionmnist",
    "svhn",
    "office-home",
    "office31",
    "visda",
    "resnet",
    "convnet",
    "convolutional",
    "image classification",
    "computer vision",
    "domain adaptation",
    "domain discriminator",
    "feature alignment",
    "dann",
    "coral",
)
_RAW_SEMANTIC_CONTRACT_FIELDS = (
    "research_question",
    "falsifiable_hypothesis",
    "primary_metric",
    "datasets",
    "models",
)
_SEMANTIC_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "data",
        "dataset",
        "datasets",
        "do",
        "does",
        "during",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "may",
        "metric",
        "model",
        "models",
        "of",
        "on",
        "or",
        "our",
        "relative",
        "research",
        "than",
        "that",
        "the",
        "their",
        "this",
        "to",
        "using",
        "versus",
        "via",
        "we",
        "whether",
        "which",
        "with",
    }
)
_SEMANTIC_GENERIC_OVERLAP_TOKENS = frozenset(
    {
        "accuracy",
        "average",
        "error",
        "loss",
        "mean",
        "performance",
        "rate",
        "result",
        "score",
    }
)


def _selected_topic_contract(
    run_dir: Path,
    config: RCConfig,
) -> dict[str, Any]:
    """Load the authoritative autonomous topic, if one exists."""
    raw = _read_prior_artifact(run_dir, "selected_topic.json")
    selected = _safe_json_loads(raw or "{}", {})
    if not isinstance(selected, dict):
        selected = {}
    selected.setdefault("title", config.research.topic)
    return selected


_SELECTED_TOPIC_CONTRACT_FIELDS = (
    "id",
    "title",
    "research_question",
    "falsifiable_hypothesis",
    "datasets",
    "models",
    "primary_metric",
    "baselines",
    "ablations",
    "failure_safety_tests",
    "cheap_pilot",
    "pilot_envelope",
    "compute",
    "pivot_policy",
)


def _has_authoritative_selected_topic(
    selected_topic: dict[str, Any],
) -> bool:
    """Return whether Stage 1 supplied a concrete autonomous topic contract."""

    return bool(str(selected_topic.get("id", "") or "").strip())


def _selected_topic_prompt_contract(
    selected_topic: dict[str, Any],
    fallback_topic: str,
) -> str:
    """Render the selected topic as a mandatory Stage-9 prompt contract."""

    if not _has_authoritative_selected_topic(selected_topic):
        return ""
    contract = {
        key: selected_topic.get(key)
        for key in _SELECTED_TOPIC_CONTRACT_FIELDS
        if selected_topic.get(key) not in (None, "", [], {})
    }
    contract.setdefault("title", fallback_topic)
    return (
        "## MANDATORY SELECTED-TOPIC CONTRACT\n"
        "This JSON object is the authoritative scientific specification for "
        "the experiment. The broader campaign brief and generic dataset "
        "examples are policy/context only.\n"
        "You MUST preserve its title, research question, falsifiable "
        "hypothesis, primary metric, declared datasets, models, baselines, "
        "ablations, cheap pilot, compute constraints, safety tests, and pivot "
        "policy. You may add implementation detail, but MUST NOT substitute "
        "an unrelated benchmark, task, model family, or metric.\n"
        "Return the hypothesis and primary metric explicitly in the YAML.\n"
        f"```json\n{json.dumps(contract, ensure_ascii=False, indent=2)}\n```\n"
    )


def _merge_unique_plan_items(primary: Any, secondary: Any) -> list[Any]:
    """Merge plan fields while preserving authoritative selected items first."""

    merged: list[Any] = []
    seen: set[str] = set()
    for item in (
        *_normalize_plan_field(primary),
        *_normalize_plan_field(secondary),
    ):
        if isinstance(item, dict):
            identity = str(
                item.get("name")
                or item.get("title")
                or item.get("id")
                or item
            ).strip().casefold()
        else:
            identity = str(item).strip().casefold()
        if not identity or identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
    return merged


def _apply_selected_topic_contract(
    plan: dict[str, Any],
    selected_topic: dict[str, Any],
    fallback_topic: str,
) -> dict[str, Any]:
    """Anchor a generated plan to the authoritative autonomous selection.

    Stage 9 may enrich implementation detail, but it is not allowed to replace
    the selected scientific question with a convenient benchmark.  Dataset and
    model identity are therefore replaced by the selected values, while
    baselines and ablations keep the selected controls first and may retain
    compatible additions proposed by the designer.
    """

    if not _has_authoritative_selected_topic(selected_topic):
        plan.setdefault("topic", fallback_topic)
        return plan

    anchored = dict(plan)
    title = str(selected_topic.get("title", fallback_topic) or fallback_topic)
    anchored["topic"] = title
    anchored["title"] = title

    for field in (
        "research_question",
        "falsifiable_hypothesis",
        "primary_metric",
        "cheap_pilot",
        "pivot_policy",
    ):
        value = selected_topic.get(field)
        if value not in (None, ""):
            anchored[field] = value

    for field in ("datasets", "models"):
        selected = _normalize_plan_field(selected_topic.get(field))
        if selected:
            anchored[field] = selected

    for field in ("baselines", "ablations", "failure_safety_tests"):
        selected = selected_topic.get(field)
        if _normalize_plan_field(selected):
            anchored[field] = _merge_unique_plan_items(
                selected,
                anchored.get(field),
            )

    compute = selected_topic.get("compute")
    if isinstance(compute, dict) and compute:
        generated_compute = anchored.get("compute_budget")
        merged_compute = (
            dict(generated_compute)
            if isinstance(generated_compute, dict)
            else {}
        )
        merged_compute.update(compute)
        anchored["compute_budget"] = merged_compute
        anchored["selected_compute"] = dict(compute)

    pilot_envelope = selected_topic.get("pilot_envelope")
    if isinstance(pilot_envelope, dict) and pilot_envelope:
        anchored["pilot_envelope"] = dict(pilot_envelope)

    selected_models = _normalize_plan_field(selected_topic.get("models"))
    if selected_models:
        anchored["models"] = selected_models

    primary_metric = str(
        selected_topic.get("primary_metric", "") or ""
    ).strip()
    if primary_metric:
        anchored["metrics"] = _merge_unique_plan_items(
            [primary_metric],
            anchored.get("metrics"),
        )

    anchored["selected_topic_id"] = str(selected_topic.get("id", "") or "")
    anchored["selected_topic_contract"] = {
        key: selected_topic.get(key)
        for key in _SELECTED_TOPIC_CONTRACT_FIELDS
        if selected_topic.get(key) not in (None, "", [], {})
    }
    return anchored


def _selected_topic_declares_benchmarks(
    selected_topic: dict[str, Any],
) -> bool:
    """Whether BenchmarkAgent discovery would duplicate an explicit contract."""

    return bool(
        _has_authoritative_selected_topic(selected_topic)
        and _normalize_plan_field(selected_topic.get("datasets"))
        and _normalize_plan_field(selected_topic.get("baselines"))
    )


def _write_semantic_alignment(
    stage_dir: Path,
    plan: dict[str, Any],
    selected_topic: dict[str, Any],
    fallback_topic: str,
    *,
    phase: str,
    persist: bool = True,
) -> dict[str, Any]:
    report = _validate_experiment_semantics(
        plan,
        selected_topic,
        fallback_topic,
    )
    report["phase"] = phase
    if not persist:
        return report
    report_path = (
        stage_dir / "semantic_alignment.json"
        if phase == "final"
        else stage_dir / f"semantic_alignment_{phase}.json"
    )
    report["report_path"] = report_path.name
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _semantic_misalignment_result(
    stage_dir: Path,
    semantic_report: dict[str, Any],
) -> StageResult:
    phase_report_name = str(
        semantic_report.get("report_path") or "semantic_alignment.json"
    )
    canonical_report_name = "semantic_alignment.json"
    if phase_report_name != canonical_report_name:
        canonical_report = dict(semantic_report)
        canonical_report["report_path"] = canonical_report_name
        (stage_dir / canonical_report_name).write_text(
            json.dumps(canonical_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    logger.error(
        "Stage 9 BLOCKED during %s: experiment plan is semantically "
        "misaligned with selected topic %r: %s",
        semantic_report.get("phase", "validation"),
        semantic_report["selected_topic_title"],
        "; ".join(semantic_report["reasons"]),
    )
    return StageResult(
        stage=Stage.EXPERIMENT_DESIGN,
        status=StageStatus.PAUSED,
        artifacts=tuple(
            dict.fromkeys(
                (canonical_report_name, phase_report_name)
            )
        ),
        error=(
            "Experiment plan is not semantically aligned with the "
            "authoritative selected_topic contract"
        ),
        evidence_refs=(f"stage-09/{canonical_report_name}",),
        decision="semantic_misalignment",
    )


def _flatten_semantic_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(
            f"{key} {_flatten_semantic_text(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_semantic_text(item) for item in value)
    return str(value or "")


def _semantic_tokens(value: Any) -> set[str]:
    """Return stable content tokens for a selected-topic contract field."""

    text = _flatten_semantic_text(value).casefold()
    tokens: set[str] = set()
    for raw_token in re.findall(r"[a-z0-9]+", text):
        if len(raw_token) <= 1 or raw_token in _SEMANTIC_STOPWORDS:
            continue
        token = raw_token
        if token.startswith("calibrat"):
            token = "calibrat"
        elif token.startswith("accept"):
            token = "accept"
        elif token.startswith("regress"):
            token = "regress"
        elif token.startswith("iterat"):
            token = "iterat"
        elif token.startswith("improv"):
            token = "improv"
        elif token.endswith("ies") and len(token) > 5:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.add(token)
    return tokens


def _raw_contract_field_alignment(
    raw_value: Any,
    selected_value: Any,
) -> dict[str, Any]:
    """Assess one explicitly generated field against its authoritative value.

    This intentionally uses a conservative lexical contradiction test rather
    than requiring exact copies. Compatible enrichments can add detail or a
    subset/superset of declared datasets/models, while unrelated replacements
    have no meaningful selected-topic tokens in common.
    """

    raw_text = _flatten_semantic_text(raw_value).strip()
    selected_text = _flatten_semantic_text(selected_value).strip()
    raw_tokens = _semantic_tokens(raw_value)
    selected_tokens = _semantic_tokens(selected_value)
    shared_tokens = raw_tokens & selected_tokens
    shared_specific_tokens = (
        shared_tokens - _SEMANTIC_GENERIC_OVERLAP_TOKENS
    )
    raw_specific_tokens = raw_tokens - _SEMANTIC_GENERIC_OVERLAP_TOKENS
    selected_specific_tokens = (
        selected_tokens - _SEMANTIC_GENERIC_OVERLAP_TOKENS
    )
    comparable = bool(raw_text and selected_text and selected_tokens)
    raw_lower = raw_text.casefold()
    selected_lower = selected_text.casefold()
    contains_authoritative_text = (
        raw_lower in selected_lower or selected_lower in raw_lower
    )
    compatible = (
        not comparable
        or contains_authoritative_text
        or bool(shared_specific_tokens)
        or (
            (not raw_specific_tokens or not selected_specific_tokens)
            and bool(shared_tokens)
        )
    )
    return {
        "explicit": bool(raw_text),
        "comparable": comparable,
        "compatible": compatible,
        "shared_tokens": sorted(shared_tokens),
        "shared_specific_tokens": sorted(shared_specific_tokens),
        "raw_tokens": sorted(raw_tokens),
        "selected_tokens": sorted(selected_tokens),
    }


def _generated_plan_drift(
    plan: dict[str, Any],
    selected_topic: dict[str, Any],
    fallback_topic: str,
) -> dict[str, Any]:
    """Detect raw Stage-9 drift without requiring a complete raw contract.

    The LLM is allowed to omit fields that the authoritative selected-topic
    contract will fill.  It is not allowed to explicitly replace declared
    question, hypothesis, metric, datasets, or models with an unrelated task.
    """

    validation_plan = {
        **plan,
        "topic": selected_topic.get("title", fallback_topic),
        "title": selected_topic.get("title", fallback_topic),
        "research_question": selected_topic.get("research_question", ""),
        "falsifiable_hypothesis": selected_topic.get(
            "falsifiable_hypothesis", ""
        ),
        "primary_metric": selected_topic.get("primary_metric", ""),
    }

    selected_report = _validate_experiment_semantics(
        validation_plan,
        selected_topic,
        fallback_topic,
    )
    raw_field_alignment: dict[str, dict[str, Any]] = {}
    explicit_drift_fields: list[str] = []
    if _has_authoritative_selected_topic(selected_topic):
        for field in _RAW_SEMANTIC_CONTRACT_FIELDS:
            raw_value = plan.get(field)
            selected_value = selected_topic.get(field)
            if not _flatten_semantic_text(raw_value).strip():
                continue
            if not _flatten_semantic_text(selected_value).strip():
                continue
            alignment = _raw_contract_field_alignment(
                raw_value,
                selected_value,
            )
            raw_field_alignment[field] = alignment
            if not alignment["compatible"]:
                explicit_drift_fields.append(field)

    if explicit_drift_fields:
        selected_report["reasons"].append(
            "Raw plan explicitly replaces authoritative selected-topic "
            "fields with unrelated content: "
            + ", ".join(explicit_drift_fields)
        )
        selected_report["aligned"] = False
    selected_report["raw_field_alignment"] = raw_field_alignment
    selected_report["explicit_drift_fields"] = explicit_drift_fields
    selected_report["raw_plan_check"] = True
    return selected_report


def _validate_experiment_semantics(
    plan: dict[str, Any],
    selected_topic: dict[str, Any],
    fallback_topic: str,
) -> dict[str, Any]:
    """Hard-check that Stage 9 still tests the selected scientific question.

    The special-case drift detector is intentionally narrow: it protects LLM
    recursive-self-improvement topics from the recurrent CIFAR/DANN/CORAL
    substitution while leaving genuinely visual/domain-adaptation topics alone.
    Generic topics still receive a lightweight selected-field overlap check.
    """
    topic_fields = (
        "title",
        "research_question",
        "falsifiable_hypothesis",
        "datasets",
        "models",
        "primary_metric",
        "baselines",
        "ablations",
    )
    topic_text = " ".join(
        _flatten_semantic_text(selected_topic.get(field))
        for field in topic_fields
    ).strip()
    if not topic_text:
        topic_text = fallback_topic
    plan_text = _flatten_semantic_text(
        {
            key: plan.get(key)
            for key in (
                "topic",
                "title",
                "research_question",
                "falsifiable_hypothesis",
                "primary_metric",
                "objectives",
                "datasets",
                "models",
                "tasks",
                "baselines",
                "proposed_methods",
                "ablations",
                "metrics",
                "cheap_pilot",
                "failure_safety_tests",
                "pivot_policy",
            )
        }
    )
    topic_lower = topic_text.casefold()
    plan_lower = plan_text.casefold()

    is_rsi_topic = any(marker in topic_lower for marker in _RSI_TOPIC_MARKERS)
    topic_is_vision = any(
        marker in topic_lower for marker in _VISION_DOMAIN_ADAPTATION_MARKERS
    )
    forbidden_drift_terms = sorted(
        marker
        for marker in _VISION_DOMAIN_ADAPTATION_MARKERS
        if marker in plan_lower and marker not in topic_lower
    )

    anchor_groups: dict[str, tuple[str, ...]] = {}
    if is_rsi_topic:
        anchor_groups = {
            "llm_or_agent_subject": (
                "llm",
                "language model",
                "agent",
                "model response",
                "task trace",
            ),
            "self_improvement_process": (
                "self-improvement",
                "self improvement",
                "self-iterative",
                "recursive",
                "iteration",
                "mutation",
                "candidate update",
            ),
            "acceptance_or_calibration_endpoint": (
                "accept",
                "gate",
                "calibrat",
                "regression",
                "rollback",
                "expected calibration error",
                "ece",
            ),
        }
    else:
        for field in ("datasets", "models", "primary_metric"):
            raw = selected_topic.get(field)
            values = raw if isinstance(raw, list) else [raw]
            markers = tuple(
                str(item).strip().casefold()
                for item in values
                if str(item or "").strip()
            )
            if markers:
                anchor_groups[field] = markers

    matched_anchor_groups = sorted(
        name
        for name, markers in anchor_groups.items()
        if any(marker in plan_lower for marker in markers)
    )
    missing_anchor_groups = sorted(
        set(anchor_groups) - set(matched_anchor_groups)
    )

    reasons: list[str] = []
    if is_rsi_topic and not topic_is_vision and forbidden_drift_terms:
        reasons.append(
            "RSI/LLM topic drifted into computer-vision or domain-adaptation "
            "datasets, models, tasks, or metrics"
        )
    if is_rsi_topic and len(matched_anchor_groups) < 2:
        reasons.append(
            "Plan does not preserve enough LLM self-improvement, acceptance, "
            "calibration, rollback, or regression endpoints"
        )
    if (
        not is_rsi_topic
        and selected_topic.get("id")
        and anchor_groups
        and not matched_anchor_groups
    ):
        reasons.append(
            "Plan does not overlap the selected topic's declared datasets, "
            "models, or primary metric"
        )

    required_field_matches: dict[str, bool] = {}
    if selected_topic.get("id"):
        for field in (
            "research_question",
            "falsifiable_hypothesis",
            "primary_metric",
        ):
            expected = _flatten_semantic_text(
                selected_topic.get(field)
            ).strip().casefold()
            if expected:
                required_field_matches[field] = (
                    expected in _flatten_semantic_text(
                        plan.get(field)
                    ).strip().casefold()
                )
        missing_required_fields = sorted(
            field
            for field, matched in required_field_matches.items()
            if not matched
        )
        if missing_required_fields:
            reasons.append(
                "Plan does not preserve the selected topic's authoritative "
                + ", ".join(missing_required_fields)
            )
    else:
        missing_required_fields = []

    return {
        "aligned": not reasons,
        "selected_topic_id": str(selected_topic.get("id", "") or ""),
        "selected_topic_title": str(
            selected_topic.get("title", fallback_topic) or fallback_topic
        ),
        "is_rsi_topic": is_rsi_topic,
        "topic_is_vision": topic_is_vision,
        "matched_anchor_groups": matched_anchor_groups,
        "missing_anchor_groups": missing_anchor_groups,
        "required_field_matches": required_field_matches,
        "missing_required_fields": missing_required_fields,
        "forbidden_drift_terms": forbidden_drift_terms,
        "reasons": reasons,
    }


def _execute_experiment_design(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    llm = _bind_stage_role(llm, Stage.EXPERIMENT_DESIGN)
    hypotheses = _read_prior_artifact(run_dir, "hypotheses.md") or ""
    _selected_topic = _selected_topic_contract(run_dir, config)
    _authoritative_topic = _has_authoritative_selected_topic(_selected_topic)
    _selected_contract_prompt = _selected_topic_prompt_contract(
        _selected_topic,
        config.research.topic,
    )
    _design_topic = str(
        _selected_topic.get("title", config.research.topic)
        or config.research.topic
    )
    preamble = _build_context_preamble(
        config, run_dir, include_goal=True, include_hypotheses=True
    )
    plan: dict[str, Any] | None = None

    # ── Domain detection ──────────────────────────────────────────────────
    # Detect the research domain early so we can adapt experiment design
    # and code generation. For ML domains, existing behavior is unchanged.
    _domain_profile = None
    try:
        from researchclaw.domains.detector import detect_domain as _detect_domain_adv
        _domain_profile = _detect_domain_adv(
            topic=_design_topic,
            hypotheses=hypotheses,
        )
        logger.info(
            "Domain detected: %s (%s)",
            _domain_profile.display_name,
            _domain_profile.domain_id,
        )
        # Persist domain profile for Stage 10
        import json as _json_dd
        (stage_dir / "domain_profile.json").write_text(
            _json_dd.dumps({
                "domain_id": _domain_profile.domain_id,
                "display_name": _domain_profile.display_name,
                "experiment_paradigm": _domain_profile.experiment_paradigm,
                "core_libraries": _domain_profile.core_libraries,
                "gpu_required": _domain_profile.gpu_required,
            }, indent=2),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.debug("Domain detection unavailable", exc_info=True)

    # --- Domain-specific experiment design context (YAML-driven overlay) ---
    # For ML and HEP, the active prompt bank is already domain-native so we
    # leave this empty. For other profiles (biology, physics, economics, …)
    # the GenericPromptAdapter injects YAML-defined guidance here.
    _domain_design_context = ""
    if _domain_profile is not None:
        try:
            from researchclaw.domains.prompt_adapter import get_adapter as _get_prompt_adapter
            _adapter = _get_prompt_adapter(_domain_profile)
            _design_blocks = _adapter.get_experiment_design_blocks(
                {"topic": _design_topic}
            )
            if _design_blocks.experiment_design_context:
                _domain_design_context = (
                    "## Domain-Specific Experiment Guidelines\n"
                    + _design_blocks.experiment_design_context
                    + "\n\n"
                )
                if _design_blocks.statistical_test_guidance:
                    _domain_design_context += (
                        "## Statistical Analysis Guidance\n"
                        + _design_blocks.statistical_test_guidance + "\n\n"
                    )
                logger.info(
                    "ExperimentDesign: injecting YAML-driven domain context for %s",
                    _domain_profile.domain_id,
                )
        except Exception:  # noqa: BLE001
            logger.debug("Domain experiment design context unavailable", exc_info=True)

    if llm is not None:
        _pm = prompts or PromptManager()
        # Pass dataset_guidance block for experiment design
        try:
            _dg_block = _pm.block("dataset_guidance")
        except (KeyError, Exception):  # noqa: BLE001
            _dg_block = ""
        # I-08: Inject RL step guidance for RL topics
        _rl_kws = ("reinforcement learning", "ppo", "sac", "td3", "ddpg",
                    "dqn", "mujoco", "continuous control", "actor-critic",
                    "policy gradient", "exploration bonus")
        _is_rl_topic = any(kw in _design_topic.lower() for kw in _rl_kws)
        if _is_rl_topic:
            try:
                _dg_block += _pm.block("rl_step_guidance")
            except Exception:  # noqa: BLE001
                pass
            # Improvement G: For RL with short budget, constrain to classic control
            if config.experiment.time_budget_sec <= 3600:
                _dg_block += (
                    "\n\n## RL TIME CONSTRAINT (MANDATORY):\n"
                    f"Your time budget is {config.experiment.time_budget_sec}s (≤ 3600s).\n"
                    "You MUST use ONLY classic control environments: "
                    "CartPole-v1, Pendulum-v1, MountainCar-v0, Acrobot-v1, LunarLander-v3.\n"
                    "Do NOT use MuJoCo (HalfCheetah, Hopper, Walker2d, Ant, Humanoid) — "
                    "they require >5000s for meaningful training.\n"
                )
            if config.experiment.time_budget_sec <= 1800:
                _dg_block += (
                    "Time budget ≤ 1800s: use ONLY CartPole-v1 or Pendulum-v1 "
                    "(the simplest environments).\n"
                )
        # F-01: Inject framework docs for experiment design
        try:
            from researchclaw.data import detect_frameworks, load_framework_docs
            _fw_ids = detect_frameworks(_design_topic, hypotheses)
            if _fw_ids:
                _fw_docs = load_framework_docs(_fw_ids, max_chars=4000)
                if _fw_docs:
                    _dg_block += _fw_docs
        except Exception:  # noqa: BLE001
            pass
        # Compute guidance must reflect the actual execution backend.  Factory
        # mode passes the per-Idea lease through RESEARCHCLAW_GPU_REQUEST.
        _gpu_request_raw = os.environ.get("RESEARCHCLAW_GPU_REQUEST", "")
        try:
            _allocated_gpus = max(0, int(_gpu_request_raw or 0))
        except ValueError:
            _allocated_gpus = 0
        _experiment_mode = str(config.experiment.mode)
        if _experiment_mode == "clusterbridge_pool":
            try:
                from researchclaw.cluster import ClusterBridgePoolConfig

                _pool = ClusterBridgePoolConfig.from_file(
                    config.experiment.clusterbridge_pool.config_file
                )
                _pool_gpus = _pool.configured_gpu_count
                _pool_nodes = len(_pool.nodes)
            except Exception:  # noqa: BLE001
                _pool_gpus = _allocated_gpus or 1
                _pool_nodes = 1
            _idea_gpus = _allocated_gpus or min(1, _pool_gpus)
            _hw_profile_str = (
                f"- Backend: prepared ClusterBridge/Ray pool\n"
                f"- Pool capacity: {_pool_gpus} GPUs across {_pool_nodes} nodes\n"
                f"- Current Idea allocation: {_idea_gpus} GPU(s)\n"
                "- CPU: shared cluster CPUs"
            )
            _gpu_execution_guidance = (
                f"- This Idea may use at most {_idea_gpus} GPU(s); do not "
                "assume ownership of the whole pool.\n"
                "- Parallelize independent conditions/seeds through Ray tasks "
                "with explicit num_gpus resource requests.\n"
                "- Keep each individual training task single-GPU unless the "
                "selected-topic contract explicitly requires model parallelism.\n"
                "- Never create, stop, or reconfigure the shared Ray cluster."
            )
        else:
            _idea_gpus = _allocated_gpus or 1
            _hw_profile_str = (
                f"- GPU allocation: {_idea_gpus} GPU(s)\n"
                "- GPU model/VRAM: use the detected Stage-1 hardware profile\n"
                "- CPU: local/shared server"
            )
            _gpu_execution_guidance = (
                f"- Design experiments that fit the allocated {_idea_gpus} "
                "GPU(s).\n"
                "- Do not use multi-node execution unless the configured "
                "experiment backend explicitly supports it."
            )
        _per_condition_sec = int(config.experiment.time_budget_sec * 0.7 / 6)
        _selected_datasets = _plan_field_names(
            _normalize_plan_field(_selected_topic.get("datasets"))
        )
        _tier1 = (
            ", ".join(_selected_datasets)
            if _authoritative_topic and _selected_datasets
            else "CIFAR-10, CIFAR-100, MNIST, FashionMNIST, STL-10, SVHN"
        )

        _overlay = _get_evolution_overlay(run_dir, "experiment_design")
        sp = _pm.for_stage(
            "experiment_design",
            evolution_overlay=_overlay,
            preamble=preamble,
            hypotheses=hypotheses,
            dataset_guidance=_dg_block,
            domain_design_context=_domain_design_context,
            time_budget_sec=config.experiment.time_budget_sec,
            metric_key=config.experiment.metric_key,
            metric_direction=config.experiment.metric_direction,
            hardware_profile=_hw_profile_str,
            gpu_execution_guidance=_gpu_execution_guidance,
            per_condition_budget_sec=_per_condition_sec,
            available_tier1_datasets=_tier1,
        )
        if _selected_contract_prompt:
            sp = type(sp)(
                system=sp.system,
                user=f"{_selected_contract_prompt}\n\n{sp.user}",
                json_mode=sp.json_mode,
                max_tokens=sp.max_tokens,
            )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        raw_yaml = _extract_yaml_block(resp.content)
        try:
            parsed = yaml.safe_load(raw_yaml)
        except yaml.YAMLError:
            parsed = None
        # Fallback: reasoning models sometimes emit the YAML without fences
        # or wrapped in prose. Try parsing the whole response as YAML.
        if not isinstance(parsed, dict):
            try:
                parsed = yaml.safe_load(resp.content)
            except yaml.YAMLError:
                pass
        # Last fallback: try to find any YAML-like dict in the response
        if not isinstance(parsed, dict):
            import re as _re_yaml

            # Look for lines starting with known keys
            _yaml_lines = []
            _capturing = False
            for line in resp.content.splitlines():
                if _re_yaml.match(
                    r"^(baselines|proposed_methods|ablations|datasets|"
                    r"metrics|objectives|risks|compute_budget)\s*:",
                    line,
                ):
                    _capturing = True
                if _capturing:
                    if line.strip() == "" or line.startswith("```"):
                        continue
                    if line.startswith("#") or line.startswith("**"):
                        continue
                    _yaml_lines.append(line)
            if _yaml_lines:
                try:
                    parsed = yaml.safe_load("\n".join(_yaml_lines))
                except yaml.YAMLError:
                    pass
        if isinstance(parsed, dict):
            plan = parsed
        else:
            logger.warning(
                "Stage 09: LLM response could not be parsed as YAML "
                "(len=%d, first 200 chars: %s). Content extraction method "
                "returned: %s",
                len(resp.content),
                resp.content[:200],
                raw_yaml[:200] if raw_yaml else "<empty>",
            )
            # BUG-12: Retry with a stricter, shorter prompt
            if llm is not None:
                logger.info("Stage 09: Retrying with strict YAML-only prompt...")
                _retry_prompt = (
                    "Output ONLY valid YAML. No prose, no markdown fences, no explanation.\n"
                    f"{_selected_contract_prompt}\n"
                    f"Topic: {_design_topic}\n"
                    "Required keys: baselines, proposed_methods, ablations, "
                    "datasets, models, metrics, objectives, risks, compute_budget, "
                    "research_question, falsifiable_hypothesis, primary_metric, "
                    "cheap_pilot, pivot_policy.\n"
                    "List-like keys map to lists; contract fields must be copied "
                    "exactly from the mandatory selected-topic contract."
                )
                _retry_resp = _chat_with_prompt(
                    llm,
                    "You output ONLY valid YAML. Nothing else.",
                    _retry_prompt,
                    max_tokens=4096,
                )
                try:
                    _retry_parsed = yaml.safe_load(_retry_resp.content)
                    if isinstance(_retry_parsed, dict):
                        plan = _retry_parsed
                        logger.info("Stage 09: Strict YAML retry succeeded.")
                except yaml.YAMLError:
                    pass

    # BUG-12: Fallback 4 — extract method/baseline names from Stage 8 hypotheses
    if plan is None:
        _hyp_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        if _hyp_text:
            import re as _re_hyp
            # Extract method-like names from hypothesis text
            _method_candidates = _re_hyp.findall(
                r"(?:proposed|our|novel|new)\s+(?:method|approach|algorithm|framework|model)[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            _baseline_candidates = _re_hyp.findall(
                r"(?:baseline|compare|existing|standard|traditional)\s+(?:method|approach|model)?[:\s]+[\"']?([A-Za-z][\w-]+)",
                _hyp_text, _re_hyp.IGNORECASE,
            )
            if _method_candidates or _baseline_candidates:
                logger.info(
                    "Stage 09: Extracted names from hypotheses: methods=%s, baselines=%s",
                    _method_candidates[:3], _baseline_candidates[:3],
                )
                plan = {
                    "topic": _design_topic,
                    "generated": _utcnow_iso(),
                    "objectives": ["Evaluate hypotheses with controlled experiments"],
                    "datasets": ["primary_dataset"],
                    "baselines": _baseline_candidates[:3] or ["baseline_1", "baseline_2"],
                    "proposed_methods": _method_candidates[:3] or ["proposed_method"],
                    "ablations": ["without_key_component", "simplified_version"],
                    "metrics": [config.experiment.metric_key, "secondary_metric"],
                    "risks": ["validity threats", "confounding variables"],
                    "compute_budget": {"max_gpu": 1, "max_hours": 4},
                }

    if plan is None:
        # BUG-12: Use domain-aware names instead of fully generic placeholders
        _topic_prefix = _design_topic.split()[0] if _design_topic else "method"
        logger.warning(
            "Stage 09: LLM failed to produce valid experiment plan YAML. "
            "Using topic-derived fallback."
        )
        plan = {
            "topic": _design_topic,
            "generated": _utcnow_iso(),
            "objectives": ["Evaluate hypotheses with controlled experiments"],
            "datasets": ["primary_dataset", "secondary_dataset"],
            "baselines": [f"{_topic_prefix}_baseline_1", f"{_topic_prefix}_baseline_2"],
            "proposed_methods": [f"{_topic_prefix}_proposed", f"{_topic_prefix}_variant"],
            "ablations": ["without_key_component", "simplified_version"],
            "metrics": [config.experiment.metric_key, "secondary_metric"],
            "risks": ["validity threats", "confounding variables"],
            "compute_budget": {"max_gpu": 1, "max_hours": 4},
        }

    _generated_plan = dict(plan)
    plan = _apply_selected_topic_contract(
        plan,
        _selected_topic,
        config.research.topic,
    )

    # Schema-deficit guard: when the LLM returned a parseable dict that
    # bypassed every fallback cascade (because plan was never None) but
    # lacks any actual experiment content, pause rather than silently
    # advancing a content-empty plan to code generation.  Use
    # _normalize_plan_field so the guard accepts every shape the rest of
    # this file already supports (str, dict, list[str], list[dict]).
    _required_any = ("baselines", "proposed_methods", "ablations")
    _normalized = {k: _normalize_plan_field(plan.get(k)) for k in _required_any}
    if not any(_normalized.values()):
        (stage_dir / "plan_meta.json").write_text(
            json.dumps(
                {
                    "outcome": "model_response_schema_deficient",
                    "missing_required_keys": [
                        k for k in _required_any if not _normalized[k]
                    ],
                    "received_keys": sorted(plan.keys()),
                    "note": (
                        "Experiment plan parsed but lacked baselines, proposed_methods, "
                        "and ablations. Pipeline paused; refine the prompt or rerun stage."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.warning(
            "Stage 9: model plan parsed but missing required content keys — pausing pipeline"
        )
        return StageResult(
            stage=Stage.EXPERIMENT_DESIGN,
            status=StageStatus.PAUSED,
            artifacts=("plan_meta.json",),
            error="Experiment plan missing baselines/proposed_methods/ablations",
            evidence_refs=("stage-09/plan_meta.json",),
            decision="schema_deficient",
        )

    # Reject explicit raw-plan drift before BenchmarkAgent or code generation
    # can spend more calls. Missing raw fields are filled by the authoritative
    # contract and are not themselves a reason to fail.
    _pre_benchmark_semantic_report = _generated_plan_drift(
        _generated_plan,
        _selected_topic,
        config.research.topic,
    )
    _pre_benchmark_semantic_report["phase"] = "pre_benchmark"
    _pre_benchmark_semantic_report["report_path"] = (
        "semantic_alignment_pre_benchmark.json"
    )
    (stage_dir / _pre_benchmark_semantic_report["report_path"]).write_text(
        json.dumps(
            _pre_benchmark_semantic_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if not _pre_benchmark_semantic_report["aligned"]:
        return _semantic_misalignment_result(
            stage_dir,
            _pre_benchmark_semantic_report,
        )

    # ── BA: BenchmarkAgent — intelligent dataset/baseline selection ──────
    _benchmark_plan = None
    # BUG-40: Skip BenchmarkAgent for non-ML domains — it has no relevant
    # benchmarks for physics/chemistry/mathematics/etc. and would inject
    # wrong datasets (e.g., CIFAR-10 for PDE topics).
    _ba_domain_profile = _domain_profile
    if _ba_domain_profile is None:
        try:
            from researchclaw.domains.detector import detect_domain as _detect_domain_adv
            _ba_domain_profile = _detect_domain_adv(
                topic=_design_topic,
                hypotheses=hypotheses,
            )
        except Exception:  # noqa: BLE001
            logger.debug("BenchmarkAgent domain detection unavailable", exc_info=True)
    _ba_domain_id = (
        _ba_domain_profile.domain_id
        if _ba_domain_profile is not None
        else "generic"
    )
    _ba_domain_ok = _ba_domain_id.startswith("ml_")
    _ba_contract_complete = _selected_topic_declares_benchmarks(
        _selected_topic
    )
    if not _ba_domain_ok:
        logger.info(
            "BenchmarkAgent skipped: domain profile '%s' is not an ML profile (topic: %s)",
            _ba_domain_id, _design_topic[:80],
        )
    elif _ba_contract_complete:
        logger.info(
            "BenchmarkAgent skipped: selected topic %s already declares "
            "authoritative datasets and baselines",
            _selected_topic.get("id"),
        )
    if (
        _ba_domain_ok
        and not _ba_contract_complete
        and config.experiment.benchmark_agent.enabled
        and config.experiment.mode
        in ("sandbox", "docker", "clusterbridge", "clusterbridge_pool")
        and llm is not None
    ):
        try:
            from researchclaw.agents.benchmark_agent import BenchmarkOrchestrator
            from researchclaw.agents.benchmark_agent.orchestrator import (
                BenchmarkAgentConfig as _BACfg,
            )

            _ba_cfg_raw = config.experiment.benchmark_agent
            _ba_cfg = _BACfg(
                enabled=_ba_cfg_raw.enabled,
                enable_hf_search=_ba_cfg_raw.enable_hf_search,
                max_hf_results=_ba_cfg_raw.max_hf_results,
                enable_web_search=_ba_cfg_raw.enable_web_search,
                max_web_results=_ba_cfg_raw.max_web_results,
                web_search_min_local=_ba_cfg_raw.web_search_min_local,
                tier_limit=_ba_cfg_raw.tier_limit,
                min_benchmarks=_ba_cfg_raw.min_benchmarks,
                min_baselines=_ba_cfg_raw.min_baselines,
                prefer_cached=_ba_cfg_raw.prefer_cached,
                max_iterations=_ba_cfg_raw.max_iterations,
            )

            _hw = _load_hardware_profile(run_dir)
            _ba = BenchmarkOrchestrator(
                llm,
                config=_ba_cfg,
                gpu_memory_mb=(
                    _hw.get("gpu_memory_mb", 49000) if _hw else 49000
                ),
                time_budget_sec=config.experiment.time_budget_sec,
                network_policy=(
                    config.experiment.docker.network_policy
                    if config.experiment.mode == "docker"
                    else "full"
                ),
                stage_dir=stage_dir / "benchmark_agent",
            )
            _benchmark_plan = _ba.orchestrate({
                "topic": _design_topic,
                "hypothesis": hypotheses,
                "experiment_plan": (
                    _flatten_semantic_text(plan.get("objectives", ""))
                    if isinstance(plan, dict)
                    else ""
                ),
            })

            # Inject BenchmarkAgent selections into experiment plan
            if isinstance(plan, dict) and _benchmark_plan.selected_benchmarks:
                _candidate_plan = dict(plan)
                _candidate_plan["datasets"] = [
                    b.get("name", "Unknown") for b in _benchmark_plan.selected_benchmarks
                ]
                # Normalize existing baselines — LLM may emit dict, list of
                # dicts, or list of strings.
                _baselines_from_plan = _plan_field_names(
                    _normalize_plan_field(plan.get("baselines", []))
                )
                _candidate_plan["baselines"] = [
                    bl.get("name", "Unknown") for bl in _benchmark_plan.selected_baselines
                ] + _baselines_from_plan
                # Deduplicate baselines
                _candidate_plan["baselines"] = list(
                    dict.fromkeys(_candidate_plan["baselines"])
                )
                _raw_benchmark_semantic_report = _write_semantic_alignment(
                    stage_dir,
                    _candidate_plan,
                    _selected_topic,
                    config.research.topic,
                    phase="benchmark_candidate",
                    persist=False,
                )
                if not _raw_benchmark_semantic_report["aligned"]:
                    logger.warning(
                        "BenchmarkAgent suggestions discarded because they "
                        "violate the selected-topic contract: %s",
                        "; ".join(
                            _raw_benchmark_semantic_report["reasons"]
                        ),
                    )
                    _benchmark_plan = None
                else:
                    _candidate_plan = _apply_selected_topic_contract(
                        _candidate_plan,
                        _selected_topic,
                        config.research.topic,
                    )
                    _benchmark_semantic_report = _write_semantic_alignment(
                        stage_dir,
                        _candidate_plan,
                        _selected_topic,
                        config.research.topic,
                        phase="post_benchmark",
                    )
                    if not _benchmark_semantic_report["aligned"]:
                        logger.warning(
                            "BenchmarkAgent suggestions discarded because "
                            "the anchored plan is still misaligned: %s",
                            "; ".join(_benchmark_semantic_report["reasons"]),
                        )
                        _benchmark_plan = None
                    else:
                        plan = _candidate_plan

            if _benchmark_plan is not None:
                logger.info(
                    "BenchmarkAgent: %d benchmarks, %d baselines selected "
                    "(%d LLM calls, %.1fs)",
                    len(_benchmark_plan.selected_benchmarks),
                    len(_benchmark_plan.selected_baselines),
                    _benchmark_plan.total_llm_calls,
                    _benchmark_plan.elapsed_sec,
                )
        except Exception as _ba_exc:
            logger.warning("BenchmarkAgent failed (non-fatal): %s", _ba_exc)

    # Save benchmark plan for code_generation stage
    if _benchmark_plan is not None:
        try:
            (stage_dir / "benchmark_plan.json").write_text(
                json.dumps(_benchmark_plan.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    plan = _apply_selected_topic_contract(
        plan,
        _selected_topic,
        config.research.topic,
    )

    # BUG-R41-09: Enforce condition count limit based on time budget.
    # Too many conditions (30+) guarantee timeouts and wasted compute.
    _time_budget = getattr(
        getattr(config, "experiment", None), "time_budget_sec", 3600
    )
    _max_conditions = 8  # default for budgets ≤ 3600s
    if _time_budget > 3600:
        _max_conditions = 12
    if _time_budget > 7200:
        _max_conditions = 20

    _baselines = _normalize_plan_field(plan.get("baselines", []))
    _proposed = _normalize_plan_field(plan.get("proposed_methods", []))
    _ablations = _normalize_plan_field(plan.get("ablations", []))
    _total = len(_baselines) + len(_proposed) + len(_ablations)

    if _total > _max_conditions:
        logger.warning(
            "Stage 9: Plan has %d conditions (limit %d for %ds budget). "
            "Trimming to fit.",
            _total, _max_conditions, _time_budget,
        )
        # Keep all proposed methods (up to max), trim baselines and ablations
        _proposed_count = min(len(_proposed), max(1, _max_conditions - 4))
        _remaining = max(0, _max_conditions - _proposed_count)
        _baseline_budget = max(1, _remaining // 2)
        _ablation_budget = max(0, _remaining - _baseline_budget)
        if len(_proposed) > _proposed_count:
            plan["proposed_methods"] = _proposed[:_proposed_count]
            logger.info(
                "Stage 9: Trimmed proposed methods %d → %d",
                len(_proposed), _proposed_count,
            )

        if len(_baselines) > _baseline_budget:
            plan["baselines"] = _baselines[:_baseline_budget]
            logger.info(
                "Stage 9: Trimmed baselines %d → %d",
                len(_baselines), _baseline_budget,
            )
        if len(_ablations) > _ablation_budget:
            plan["ablations"] = _ablations[:_ablation_budget]
            logger.info(
                "Stage 9: Trimmed ablations %d → %d",
                len(_ablations), _ablation_budget,
            )

    # --- HITL: Read human guidance if available ---
    guidance_file = stage_dir / "hitl_guidance.md"
    if guidance_file.exists():
        try:
            guidance = guidance_file.read_text(encoding="utf-8").strip()
            if guidance and llm is not None and isinstance(plan, dict):
                logger.info("Applying HITL guidance to experiment design")
                resp = llm.chat(
                    [{"role": "user", "content": (
                        f"The human researcher provided this guidance for "
                        f"the experiment design:\n\n{guidance}\n\n"
                        f"Current experiment plan:\n"
                        f"```yaml\n{yaml.dump(plan, default_flow_style=False)}\n```\n\n"
                        f"Update the YAML plan to incorporate the guidance. "
                        f"Return ONLY the updated YAML."
                    )}],
                    max_tokens=4096,
                )
                updated = _extract_yaml_block(resp.content)
                try:
                    parsed_update = yaml.safe_load(updated)
                    if isinstance(parsed_update, dict):
                        plan = _apply_selected_topic_contract(
                            parsed_update,
                            _selected_topic,
                            config.research.topic,
                        )
                except yaml.YAMLError:
                    pass
        except Exception:
            logger.debug("HITL guidance application failed (non-blocking)")

    # Production scientific gate: the executable plan must still test the
    # authoritative selected topic. Run after every automatic and HITL rewrite
    # so no later mutation can reintroduce an unrelated benchmark.
    _semantic_report = _write_semantic_alignment(
        stage_dir,
        plan,
        _selected_topic,
        config.research.topic,
        phase="final",
    )
    if not _semantic_report["aligned"]:
        return _semantic_misalignment_result(
            stage_dir,
            _semantic_report,
        )

    # --- HITL: Baseline Navigator data persistence ---
    try:
        from researchclaw.hitl.workshops.baseline import BaselineNavigator, BaselineCandidate

        nav = BaselineNavigator(run_dir, llm_client=llm)
        if isinstance(plan, dict):
            baselines = plan.get("baselines", [])
            if isinstance(baselines, list):
                for b in baselines:
                    if isinstance(b, dict):
                        nav.baselines.append(BaselineCandidate(
                            name=b.get("name", str(b)),
                            description=b.get("description", ""),
                        ))
                    elif isinstance(b, str):
                        nav.baselines.append(BaselineCandidate(name=b))
            metrics = plan.get("metrics", [])
            if isinstance(metrics, list):
                nav.metrics = [str(m) for m in metrics]
        nav.save()
    except Exception:
        pass

    (stage_dir / "exp_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.EXPERIMENT_DESIGN,
        status=StageStatus.DONE,
        artifacts=("exp_plan.yaml", "semantic_alignment.json"),
        evidence_refs=(
            "stage-09/exp_plan.yaml",
            "stage-09/semantic_alignment.json",
        ),
    )
