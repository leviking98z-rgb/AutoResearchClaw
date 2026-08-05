"""Role-aware LLM routing for the autonomous research pipeline.

Roles are first-class execution identities rather than prompt labels.  Each
role may select a provider/model, own an isolated backend session, apply
role-level generation defaults, and emit an auditable per-call trace.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchclaw.config import RCConfig, RoleConfig
from researchclaw.factory.io import append_jsonl
from researchclaw.llm.client import LLMResponse
from researchclaw.pipeline.stages import Stage

logger = logging.getLogger(__name__)


DEFAULT_STAGE_ROLES: dict[Stage, str] = {
    Stage.TOPIC_INIT: "research_director",
    Stage.PROBLEM_DECOMPOSE: "research_director",
    Stage.SEARCH_STRATEGY: "literature_researcher",
    Stage.LITERATURE_COLLECT: "literature_researcher",
    Stage.LITERATURE_SCREEN: "literature_researcher",
    Stage.KNOWLEDGE_EXTRACT: "literature_researcher",
    Stage.SYNTHESIS: "idea_scientist",
    Stage.HYPOTHESIS_GEN: "idea_scientist",
    Stage.EXPERIMENT_DESIGN: "experiment_designer",
    Stage.CODE_GENERATION: "coding_engineer",
    Stage.RESOURCE_PLANNING: "compute_operator",
    Stage.EXPERIMENT_RUN: "compute_operator",
    Stage.ITERATIVE_REFINE: "coding_engineer",
    Stage.RESULT_ANALYSIS: "result_analyst",
    Stage.RESEARCH_DECISION: "research_director",
    Stage.PAPER_OUTLINE: "paper_writer",
    Stage.PAPER_DRAFT: "paper_writer",
    Stage.PEER_REVIEW: "skeptical_reviewer",
    Stage.PAPER_REVISION: "paper_writer",
    Stage.QUALITY_GATE: "skeptical_reviewer",
    Stage.KNOWLEDGE_ARCHIVE: "research_director",
    Stage.EXPORT_PUBLISH: "paper_writer",
    Stage.CITATION_VERIFY: "citation_auditor",
}

KNOWN_ROLES: frozenset[str] = frozenset(
    set(DEFAULT_STAGE_ROLES.values())
    | {
        "topic_selector",
        "campaign_director",
        "mutation_proposer",
        "mutation_auditor",
    }
)

ROLE_MODEL_TIERS: dict[str, str] = {
    # Expensive, consequential judgments.
    "topic_selector": "decision",
    "campaign_director": "decision",
    "research_director": "decision",
    "skeptical_reviewer": "decision",
    "mutation_auditor": "decision",
    # Main scientific and engineering work.
    "idea_scientist": "worker",
    "experiment_designer": "worker",
    "coding_engineer": "worker",
    "result_analyst": "worker",
    "paper_writer": "worker",
    "mutation_proposer": "worker",
    # High-volume extraction, organization, and operational narration.
    "literature_researcher": "utility",
    "compute_operator": "utility",
    "citation_auditor": "utility",
}

_SAFE_ROLE_RE = re.compile(r"[^a-z0-9_.-]+")


def normalize_role_name(role: str) -> str:
    """Return a stable, filesystem-safe role identifier."""

    value = _SAFE_ROLE_RE.sub("_", str(role or "").strip().casefold()).strip("_")
    return value or "default"


def role_for_stage(stage: Stage) -> str:
    """Return the default execution role for a pipeline stage."""

    return DEFAULT_STAGE_ROLES[stage]


def model_tier_for_role(role: str) -> str:
    """Return the configured three-tier class for *role*."""

    return ROLE_MODEL_TIERS.get(normalize_role_name(role), "worker")


@dataclass(frozen=True)
class ResolvedRole:
    """Fully resolved role policy after global-config inheritance."""

    name: str
    config: RoleConfig
    provider: str
    base_url: str
    wire_api: str
    api_key_env: str
    api_key: str
    model: str
    fallback_models: tuple[str, ...]
    temperature: float | None
    max_tokens: int | None
    timeout_sec: int
    isolated_session: bool
    session_name: str
    tools: tuple[str, ...]


def resolve_role(config: RCConfig, role: str) -> ResolvedRole:
    """Resolve a role against the global LLM defaults and three tiers."""

    name = normalize_role_name(role)
    role_cfg = config.llm.roles.get(name, RoleConfig())
    tier_name = model_tier_for_role(name)
    tier_cfg = getattr(config.llm.model_tiers, tier_name)
    inherited_model = tier_cfg.model or config.llm.primary_model
    inherited_fallbacks = (
        tier_cfg.fallback_models if tier_cfg.model else config.llm.fallback_models
    )
    # Once three-tier routing is enabled, the tier is the model authority.
    # Per-role model fields from older configs are intentionally ignored so a
    # deployment cannot silently grow back into N independently routed models.
    tiered = any(
        candidate.model
        for candidate in (
            config.llm.model_tiers.decision,
            config.llm.model_tiers.worker,
            config.llm.model_tiers.utility,
        )
    )
    fallback_models = (
        inherited_fallbacks
        if tiered or role_cfg.fallback_models is None
        else role_cfg.fallback_models
    )
    return ResolvedRole(
        name=name,
        config=role_cfg,
        provider=role_cfg.provider or config.llm.provider,
        base_url=role_cfg.base_url or config.llm.base_url,
        wire_api=role_cfg.wire_api or config.llm.wire_api,
        api_key_env=role_cfg.api_key_env or config.llm.api_key_env,
        api_key=role_cfg.api_key or config.llm.api_key,
        model=inherited_model if tiered else (role_cfg.model or inherited_model),
        fallback_models=tuple(fallback_models),
        temperature=role_cfg.temperature,
        max_tokens=role_cfg.max_tokens,
        timeout_sec=(
            role_cfg.timeout_sec
            if role_cfg.timeout_sec is not None
            else config.llm.timeout_sec
        ),
        isolated_session=role_cfg.isolated_session,
        session_name=(
            role_cfg.session_name
            or (
                f"{config.llm.acp.session_name}-{name}"
                if role_cfg.isolated_session
                else config.llm.acp.session_name
            )
        ),
        tools=role_cfg.tools,
    )


def _role_rc_config(config: RCConfig, resolved: ResolvedRole) -> RCConfig:
    """Build a lightweight RCConfig copy for the role's backend factory."""

    acp = replace(
        config.llm.acp,
        session_name=resolved.session_name,
        timeout_sec=resolved.timeout_sec,
    )
    llm = replace(
        config.llm,
        provider=resolved.provider,
        base_url=resolved.base_url,
        wire_api=resolved.wire_api,
        api_key_env=resolved.api_key_env,
        api_key=resolved.api_key,
        primary_model=resolved.model,
        fallback_models=resolved.fallback_models,
        timeout_sec=resolved.timeout_sec,
        acp=acp,
    )
    return replace(config, llm=llm)


def _hash_messages(messages: list[dict[str, str]], system: str | None) -> str:
    payload = json.dumps(
        {"system": system or "", "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RoleLLMClient:
    """LLM-compatible facade bound to one research role."""

    def __init__(
        self,
        backend: Any,
        resolved: ResolvedRole,
        *,
        audit_path: Path | None = None,
        stage: Stage | None = None,
        backend_cache_key: tuple[object, ...] | None = None,
    ) -> None:
        self._backend = backend
        self._backend_factory: Any | None = None
        self._backend_cache: dict[tuple[object, ...], Any] = {}
        self._backend_cache_key = (
            backend_cache_key or self._backend_key(resolved)
        )
        self.resolved_role = resolved
        self.role = resolved.name
        self.stage = stage
        self.audit_path = audit_path
        self.config = getattr(backend, "config", None)
        self._audit_lock = threading.Lock()

    @property
    def backend(self) -> Any:
        return self._backend

    def for_role(
        self,
        role: str,
        *,
        stage: Stage | None = None,
    ) -> RoleLLMClient:
        """Return a role-bound facade over the same injected backend."""

        name = normalize_role_name(role)
        resolved = resolve_role_from_global(self.resolved_role, name)
        role_policies = getattr(self, "_role_policies", None)
        if isinstance(role_policies, dict) and name in role_policies:
            resolved = role_policies[name]
        backend = self._backend
        if (
            callable(self._backend_factory)
            and self._requires_distinct_backend(resolved)
        ):
            cache_key = self._backend_key(resolved)
            if cache_key not in self._backend_cache:
                self._backend_cache[cache_key] = self._backend_factory(name)
            backend = self._backend_cache[cache_key]
        else:
            cache_key = self._backend_cache_key
        audit_path = None
        if self.audit_path is not None:
            audit_path = (
                self.audit_path.parent
                / f"llm-{resolved.name}.jsonl"
            )
        child = RoleLLMClient(
            backend,
            resolved,
            audit_path=audit_path,
            stage=stage if stage is not None else self.stage,
            backend_cache_key=cache_key,
        )
        child._backend_factory = self._backend_factory
        child._backend_cache = self._backend_cache
        if isinstance(role_policies, dict):
            child._role_policies = role_policies  # type: ignore[attr-defined]
        return child

    def for_stage(self, stage: Stage) -> RoleLLMClient:
        """Return a sibling facade using the canonical role for ``stage``."""

        return self.for_role(role_for_stage(stage), stage=stage)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
        system: str | None = None,
        strip_thinking: bool = False,
    ) -> LLMResponse:
        """Call the bound backend using role defaults and append an audit row."""

        requested_model = model or self.resolved_role.model or None
        backend_primary_model = self._backend_primary_model()
        # Native role backends are constructed with the role's own primary
        # model and fallback chain.  Passing that same model as a call-level
        # override would make LLMClient skip its fallback chain, so only pass
        # ``model=`` for an explicit caller override or when an injected/shared
        # backend is configured for a different primary model.
        effective_model = (
            requested_model
            if model is not None
            or not backend_primary_model
            or requested_model != backend_primary_model
            else None
        )
        effective_tokens = (
            max_tokens
            if max_tokens is not None
            else self.resolved_role.max_tokens
        )
        effective_temperature = (
            temperature
            if temperature is not None
            else self.resolved_role.temperature
        )
        started = time.monotonic()
        request_hash = _hash_messages(messages, system)
        try:
            response = self._backend.chat(
                messages,
                model=effective_model,
                max_tokens=effective_tokens,
                temperature=effective_temperature,
                json_mode=json_mode,
                system=system,
                strip_thinking=strip_thinking,
            )
        except Exception as exc:
            self._write_audit(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "role": self.role,
                    "stage": int(self.stage) if self.stage is not None else None,
                    "provider": self.resolved_role.provider,
                    "requested_model": requested_model or "",
                    "backend_model_override": effective_model or "",
                    "configured_fallback_models": list(
                        self.resolved_role.fallback_models
                    ),
                    "attempted_models": list(
                        getattr(exc, "_researchclaw_attempted_models", ()) or ()
                    ),
                    "attempts": int(
                        getattr(exc, "_researchclaw_attempts", 1) or 1
                    ),
                    "retries": int(
                        getattr(exc, "_researchclaw_retries", 0) or 0
                    ),
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc)[:500],
                    "elapsed_sec": round(time.monotonic() - started, 6),
                    "request_sha256": request_hash,
                    "message_count": len(messages),
                    "max_tokens": effective_tokens,
                    "temperature": effective_temperature,
                    "json_mode": bool(json_mode),
                }
            )
            raise

        self._write_audit(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "role": self.role,
                "stage": int(self.stage) if self.stage is not None else None,
                "provider": self.resolved_role.provider,
                "requested_model": requested_model or "",
                "backend_model_override": effective_model or "",
                "response_model": getattr(response, "model", ""),
                "configured_fallback_models": list(
                    self.resolved_role.fallback_models
                ),
                "attempted_models": list(
                    getattr(response, "attempted_models", ()) or ()
                ),
                "attempts": int(getattr(response, "attempts", 1) or 1),
                "retries": int(getattr(response, "retries", 0) or 0),
                "fallback_count": int(
                    getattr(response, "fallback_count", 0) or 0
                ),
                "status": "ok",
                "elapsed_sec": round(time.monotonic() - started, 6),
                "request_sha256": request_hash,
                "message_count": len(messages),
                "max_tokens": effective_tokens,
                "temperature": effective_temperature,
                "json_mode": bool(json_mode),
                "prompt_tokens": int(getattr(response, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(response, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(getattr(response, "total_tokens", 0) or 0),
                "finish_reason": str(getattr(response, "finish_reason", "") or ""),
                "truncated": bool(getattr(response, "truncated", False)),
            }
        )
        return response

    def preflight(self) -> tuple[bool, str]:
        return self._backend.preflight()

    def close(self) -> None:
        seen: set[int] = set()
        for backend in (self._backend, *self._backend_cache.values()):
            identity = id(backend)
            if identity in seen:
                continue
            seen.add(identity)
            backend_close = getattr(backend, "close", None)
            if callable(backend_close):
                backend_close()

    def _requires_distinct_backend(self, target: ResolvedRole) -> bool:
        current = self.resolved_role
        return any(
            (
                target.provider != current.provider,
                target.base_url != current.base_url,
                target.wire_api != current.wire_api,
                target.api_key_env != current.api_key_env,
                target.api_key != current.api_key,
                target.model != current.model,
                target.fallback_models != current.fallback_models,
                target.timeout_sec != current.timeout_sec,
                (
                    target.provider == "acp"
                    and target.session_name != current.session_name
                ),
            )
        )

    def _backend_primary_model(self) -> str:
        """Return the primary model configured on the concrete backend."""

        config = getattr(self._backend, "config", None)
        return str(
            getattr(config, "primary_model", "")
            or getattr(config, "model", "")
            or ""
        ).strip()

    @staticmethod
    def _backend_key(target: ResolvedRole) -> tuple[object, ...]:
        """Return the concrete backend/session identity for cache reuse."""

        return (
            target.provider,
            target.base_url,
            target.wire_api,
            target.api_key_env,
            target.api_key,
            target.model,
            target.fallback_models,
            target.timeout_sec,
            target.session_name if target.provider == "acp" else "",
        )

    def _write_audit(self, record: dict[str, Any]) -> None:
        if self.audit_path is None:
            return
        try:
            with self._audit_lock:
                append_jsonl(
                    self.audit_path,
                    record,
                    durable=True,
                )
        except Exception:
            logger.warning(
                "Could not write role LLM audit log: %s",
                self.audit_path,
                exc_info=True,
            )


def create_role_llm_client(
    config: RCConfig,
    role: str,
    *,
    run_dir: Path | None = None,
    stage: Stage | None = None,
    backend: Any | None = None,
) -> RoleLLMClient:
    """Create an isolated, auditable LLM client for one role."""

    from researchclaw.llm import create_llm_client

    resolved = resolve_role(config, role)
    role_config = _role_rc_config(config, resolved)
    selected_backend = backend
    if selected_backend is None:
        selected_backend = create_llm_client(role_config)
    audit_path = None
    if run_dir is not None:
        audit_path = Path(run_dir) / "audit" / f"llm-{resolved.name}.jsonl"
    client = RoleLLMClient(
        selected_backend,
        resolved,
        audit_path=audit_path,
        stage=stage,
    )
    client._backend_cache[client._backend_cache_key] = selected_backend
    client._role_policies = {  # type: ignore[attr-defined]
        name: resolve_role(config, name)
        for name in set(config.llm.roles) | set(KNOWN_ROLES)
    }
    if backend is None:
        client._backend_factory = lambda name: create_llm_client(  # type: ignore[attr-defined]
            _role_rc_config(config, resolve_role(config, name))
        )
    return client


def create_stage_llm_client(
    config: RCConfig,
    stage: Stage,
    *,
    run_dir: Path | None = None,
) -> RoleLLMClient:
    """Create the configured role client for a pipeline stage."""

    return create_role_llm_client(
        config,
        role_for_stage(stage),
        run_dir=run_dir,
        stage=stage,
    )


def bind_role_llm_client(
    backend: Any,
    config: RCConfig,
    role: str,
    *,
    run_dir: Path | None = None,
    stage: Stage | None = None,
) -> RoleLLMClient:
    """Bind an injected/test backend to role policy without replacing it."""

    resolved = resolve_role(config, role)
    audit_path = None
    if run_dir is not None:
        audit_path = Path(run_dir) / "audit" / f"llm-{resolved.name}.jsonl"
    client = RoleLLMClient(
        backend,
        resolved,
        audit_path=audit_path,
        stage=stage,
    )
    client._backend_cache[client._backend_cache_key] = backend
    client._role_policies = {  # type: ignore[attr-defined]
        name: resolve_role(config, name)
        for name in set(config.llm.roles) | set(KNOWN_ROLES)
    }
    return client


def resolve_role_from_global(current: ResolvedRole, role: str) -> ResolvedRole:
    """Best-effort sibling-role binding when only a role client is available.

    The executor normally creates the exact stage client.  This helper is for
    nested specialist agents (for example the coding engineer inside the
    experiment-design stage) where the original RCConfig is not carried by the
    client.  It preserves the shared backend while changing the auditable role
    identity; role-specific backend/model overrides still require construction
    through :func:`create_role_llm_client`.
    """

    name = normalize_role_name(role)
    return replace(
        current,
        name=name,
        session_name=f"{current.session_name.rsplit('-', 1)[0]}-{name}",
    )
