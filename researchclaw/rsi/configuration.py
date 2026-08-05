"""Campaign-local ResearchClaw config generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .storage import CampaignStore, atomic_write_json, atomic_write_text

_PROMPT_STAGES = (
    "topic_init",
    "problem_decompose",
    "search_strategy",
    "literature_collect",
    "literature_screen",
    "knowledge_extract",
    "synthesis",
    "hypothesis_gen",
    "experiment_design",
    "code_generation",
    "resource_planning",
    "result_analysis",
    "research_decision",
    "paper_outline",
    "paper_draft",
    "peer_review",
    "paper_revision",
    "quality_gate",
    "knowledge_archive",
    "export_publish",
)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    return value


def _mapping(root: dict[str, Any], key: str) -> dict[str, Any]:
    value = root.get(key)
    if not isinstance(value, dict):
        value = {}
        root[key] = value
    return value


def _campaign_knowledge_text(
    path: Path,
    *,
    max_entries: int = 30,
    max_chars: int = 12000,
) -> str:
    """Render recent accepted A-Evolve knowledge into prompt-safe text."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeError):
        return ""
    rendered: list[str] = []
    total = 0
    for raw in lines[-max_entries:]:
        try:
            item = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "campaign knowledge")).strip()
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        section = f"- **{name}**: {content}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[:remaining].rstrip() + " …"
        rendered.append(section)
        total += len(section)
    return "\n".join(rendered)


def prepare_cycle_config(
    *,
    base_config: Path,
    output_path: Path,
    store: CampaignStore,
    topic: str,
    campaign_brief: str | None = None,
    selected_topic_path: Path | None = None,
    autonomous_topic_selection: bool = False,
    model: str,
    bridge_url: str,
    api_key_env: str,
    timeout_sec: int,
    cycle: int | None = None,
) -> Path:
    """Write a self-contained cycle config with campaign memory attached."""

    data = load_yaml_mapping(base_config)
    base_dir = base_config.parent.resolve()

    research = _mapping(data, "research")
    research["topic"] = topic
    if campaign_brief:
        research["campaign_brief"] = campaign_brief
    research["autonomous_topic_selection"] = bool(autonomous_topic_selection)
    if selected_topic_path is not None:
        research["selected_topic_file"] = str(selected_topic_path)

    project = _mapping(data, "project")
    # Full-auto avoids approval gates, but publishing remains disabled below.
    project["mode"] = "full-auto"

    llm = _mapping(data, "llm")
    llm.update(
        {
            "provider": "openai-compatible",
            "base_url": bridge_url,
            "wire_api": "chat_completions",
            "api_key_env": api_key_env,
            "api_key": "",
            "primary_model": model,
            "fallback_models": [],
            "timeout_sec": timeout_sec,
        }
    )
    tiers = llm.get("model_tiers")
    has_model_tiers = isinstance(tiers, dict) and any(
        isinstance(value, str) and bool(value.strip())
        or isinstance(value, dict)
        and bool(str(value.get("model", "") or "").strip())
        for value in tiers.values()
    )
    roles = llm.get("roles")
    if isinstance(roles, dict):
        for role_data in roles.values():
            if not isinstance(role_data, dict):
                continue
            # ``--model`` remains the backward-compatible default model.
            # Three-tier routing, when present, is authoritative and survives
            # cycle materialization. Without tiers, preserve explicit legacy
            # role models.
            has_alternate_backend = any(
                str(role_data.get(key, "") or "").strip()
                for key in ("provider", "base_url", "api_key_env", "api_key")
            )
            has_explicit_model = bool(
                str(
                    role_data.get("model", role_data.get("primary_model", ""))
                    or ""
                ).strip()
            )
            if (
                not has_model_tiers
                and not has_alternate_backend
                and not has_explicit_model
            ):
                role_data["model"] = model
                role_data["fallback_models"] = []
                role_data["timeout_sec"] = timeout_sec

    cli_agent = _mapping(_mapping(data, "experiment"), "cli_agent")
    cli_agent["provider"] = "llm"

    security = _mapping(data, "security")
    security["allow_publish_without_approval"] = False

    hitl = _mapping(data, "hitl")
    hitl["enabled"] = False

    metaclaw = _mapping(data, "metaclaw_bridge")
    metaclaw["skills_dir"] = str(store.shared_skills_dir)

    skills = _mapping(data, "skills")
    custom_dirs = [str(value) for value in skills.get("custom_dirs", []) or []]
    shared_skills = str(store.shared_skills_dir)
    if shared_skills not in custom_dirs:
        custom_dirs.append(shared_skills)
    skills["custom_dirs"] = custom_dirs

    prompts = _mapping(data, "prompts")
    custom_file = str(prompts.get("custom_file", "") or "").strip()
    if custom_file:
        custom_path = Path(custom_file).expanduser()
        if not custom_path.is_absolute():
            prompts["custom_file"] = str((base_dir / custom_path).resolve())
    extras = prompts.get("extra_prompts")
    if not isinstance(extras, dict):
        extras = {}
    guidance_path = str(store.shared_prompt_path)
    try:
        guidance_text = store.shared_prompt_path.read_text(
            encoding="utf-8"
        ).strip()
    except FileNotFoundError:
        guidance_text = ""
    knowledge_path = store.shared_dir / "knowledge_entries.jsonl"
    knowledge_text = _campaign_knowledge_text(knowledge_path)
    selected_topic_context = ""
    if selected_topic_path is not None:
        try:
            selected_topic_context = selected_topic_path.read_text(
                encoding="utf-8"
            ).strip()
        except (FileNotFoundError, OSError, UnicodeError):
            selected_topic_context = ""
    topic_patch_context = ""
    try:
        topic_patch_data = yaml.safe_load(
            store.shared_topic_patch_path.read_text(encoding="utf-8")
        )
        if isinstance(topic_patch_data, dict):
            topic_patch_context = str(
                topic_patch_data.get("topic_patch", "") or ""
            ).strip()
    except (FileNotFoundError, OSError, UnicodeError, yaml.YAMLError):
        topic_patch_context = ""
    repair_patch_context = ""
    repair_patch_meta: dict[str, Any] = {}
    try:
        repair_patch_data = yaml.safe_load(
            store.shared_repair_patch_path.read_text(encoding="utf-8")
        )
        if isinstance(repair_patch_data, dict):
            expires_after = int(repair_patch_data.get("expires_after_cycle", 0))
            is_active = cycle is None or cycle <= expires_after
            if is_active:
                repair_patch_context = str(
                    repair_patch_data.get("repair_prompt_patch", "") or ""
                ).strip()
                repair_patch_meta = repair_patch_data
    except (
        FileNotFoundError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        yaml.YAMLError,
    ):
        repair_patch_context = ""
        repair_patch_meta = {}
    repair_stages = {
        "experiment_design",
        "code_generation",
        "resource_planning",
        "result_analysis",
    }
    for stage in _PROMPT_STAGES:
        prior = str(extras.get(stage, "") or "").strip()
        merged = store.shared_prompts_dir / f"{stage}.md"
        text = ""
        if prior:
            prior_path = Path(prior).expanduser()
            if not prior_path.is_absolute():
                prior_path = (base_dir / prior_path).resolve()
            if prior_path.is_file():
                text = prior_path.read_text(encoding="utf-8").strip()
            elif prior:
                text = prior
        sections = []
        if text:
            sections.append(text)
        if selected_topic_context:
            sections.append(
                "## Authoritative Autonomous Topic Selection\n\n"
                f"{selected_topic_context}\n\n"
                "Use this concrete selected topic, primary hypothesis, and "
                "primary metric. Do not replace it with a new topic inside "
                "the ordinary 23-stage pipeline. The pipeline must implement "
                "and evaluate this selected hypothesis directly."
            )
        if topic_patch_context:
            sections.append(
                "## Accepted Bounded Topic Refinement\n\n"
                f"{topic_patch_context}\n\n"
                "This refines the incumbent question without replacing its "
                "candidate identity or weakening campaign policy."
            )
        if repair_patch_context and stage in repair_stages:
            failure_signature = str(
                repair_patch_meta.get("failure_signature", "unknown")
            )
            recovery_action = str(
                repair_patch_meta.get("recovery_action", "auto_repair")
            )
            sections.append(
                "## Transient Failed-Cycle Engineering Repair\n\n"
                f"Failure signature: `{failure_signature}`\n\n"
                f"Recovery action: `{recovery_action}`\n\n"
                f"{repair_patch_context}\n\n"
                "This repair is temporary and may only fix implementation, "
                "dependency, data-loading, resource, or reproducibility "
                "defects. It must not lower or bypass quality gates, alter "
                "the hypothesis to manufacture success, fabricate evidence, "
                "disable verification, or relax safety/publication policy."
            )
        if campaign_brief:
            sections.append(
                "## Campaign Meta-Brief and Safety Policy\n\n"
                f"{campaign_brief}\n\n"
                "The meta-brief supplies scientific and safety constraints; "
                "it is not the concrete research topic. Autonomous topic "
                "selection and cross-cycle campaign iteration are already "
                "implemented by the outer RSI supervisor. Do not require "
                "experiment code to reimplement the supervisor, candidate "
                "selection board, or the whole meta-research system."
            )
        if guidance_text:
            sections.append(
                "## Campaign RSI Guidance\n\n"
                f"{guidance_text}\n\n"
                f"_Materialized from {guidance_path} for this cycle._"
            )
        if knowledge_text:
            sections.append(
                "## Accepted Cross-Cycle Knowledge\n\n"
                f"{knowledge_text}\n\n"
                f"_Materialized from {knowledge_path} for this cycle._"
            )
        atomic_write_text(
            merged,
            "\n\n".join(sections).strip() + "\n",
        )
        extras[stage] = str(merged)
    prompts["extra_prompts"] = extras

    knowledge_base = _mapping(data, "knowledge_base")
    kb_root = str(knowledge_base.get("root", "") or "").strip()
    if kb_root:
        kb_path = Path(kb_root).expanduser()
        if not kb_path.is_absolute():
            knowledge_base["root"] = str((base_dir / kb_path).resolve())

    # Normalize other path-bearing settings because the generated config lives
    # under the campaign run directory, not next to the original YAML.
    memory = _mapping(data, "memory")
    memory_dir = str(memory.get("store_dir", "") or "").strip()
    if memory_dir:
        memory_path = Path(memory_dir).expanduser()
        if not memory_path.is_absolute():
            memory["store_dir"] = str((base_dir / memory_path).resolve())
    raw_builtin = str(skills.get("builtin_dir", "") or "").strip()
    if raw_builtin:
        builtin_path = Path(raw_builtin).expanduser()
        if not builtin_path.is_absolute():
            skills["builtin_dir"] = str((base_dir / builtin_path).resolve())
    for key in ("custom_dirs", "external_dirs"):
        normalized: list[str] = []
        for raw_value in skills.get(key, []) or []:
            path = Path(str(raw_value)).expanduser()
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            normalized.append(str(path))
        if key == "custom_dirs" and shared_skills not in normalized:
            normalized.append(shared_skills)
        skills[key] = normalized

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_path,
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
    )
    # A small machine-readable provenance sidecar is useful for audits.
    atomic_write_json(
        output_path.with_suffix(".provenance.json"),
        {
            "base_config": str(base_config),
            "topic": topic,
            "campaign_brief": campaign_brief or topic,
            "selected_topic_file": (
                str(selected_topic_path) if selected_topic_path is not None else ""
            ),
            "autonomous_topic_selection": bool(autonomous_topic_selection),
            "model": model,
            "bridge_url": bridge_url,
            "shared_brief": str(store.shared_brief_path),
            "shared_skills": str(store.shared_skills_dir),
            "shared_prompt": str(store.shared_prompt_path),
            "automatic_submission_enabled": False,
        },
    )
    return output_path
