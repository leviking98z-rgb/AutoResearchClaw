"""Structured role calls with schema validation and bounded retries."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, Mapping):
        raise TypeError("model response must be one JSON object")
    return dict(value)


@dataclass(frozen=True, slots=True)
class StructuredResult:
    value: dict[str, Any]
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    attempts: int


class StructuredValidationError(ValueError):
    """Bounded structured generation failed but produced repairable JSON."""

    def __init__(
        self,
        message: str,
        *,
        previous_value: Mapping[str, Any] | None,
        errors: list[str],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        attempts: int,
    ) -> None:
        super().__init__(message)
        self.previous_value = (
            dict(previous_value)
            if previous_value is not None
            else None
        )
        self.errors = tuple(str(item) for item in errors)
        self.model = str(model)
        self.prompt_tokens = max(0, int(prompt_tokens))
        self.completion_tokens = max(0, int(completion_tokens))
        self.total_tokens = max(0, int(total_tokens))
        self.attempts = max(0, int(attempts))


class StructuredRole:
    def __init__(
        self,
        *,
        client: Any,
        system: str,
        validator: Callable[[Mapping[str, Any]], list[str]],
        max_attempts: int = 2,
    ) -> None:
        self.client = client
        self.system = system
        self.validator = validator
        self.max_attempts = max(1, int(max_attempts))

    def call(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        retry_context: Callable[
            [Mapping[str, Any], list[str]], str
        ]
        | None = None,
    ) -> StructuredResult:
        error = ""
        total_prompt = 0
        total_completion = 0
        total_tokens = 0
        last_model = ""
        previous_value: dict[str, Any] | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = prompt
            if error:
                repair = ""
                if retry_context is not None and previous_value is not None:
                    repair = retry_context(
                        previous_value,
                        [
                            item.strip()
                            for item in error.split(";")
                            if item.strip()
                        ],
                    )
                if not repair:
                    repair = (
                        "Return a complete corrected JSON object. Preserve all "
                        "valid fields from your previous response and change "
                        "only what the validation errors require."
                    )
                request += (
                    "\n\nYour previous response was rejected by deterministic "
                    f"validation. Validation errors: {error}\n\n{repair}"
                )
            response = self.client.chat(
                [{"role": "user", "content": request}],
                system=self.system,
                json_mode=True,
                max_tokens=max_tokens,
                temperature=temperature if attempt == 1 else min(0.15, temperature),
            )
            total_prompt += int(getattr(response, "prompt_tokens", 0) or 0)
            total_completion += int(
                getattr(response, "completion_tokens", 0) or 0
            )
            total_tokens += int(getattr(response, "total_tokens", 0) or 0)
            last_model = str(getattr(response, "model", "") or "")
            try:
                value = parse_json_object(response.content)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                error = f"invalid_json:{exc}"
                previous_value = None
                continue
            errors = self.validator(value)
            if errors:
                error = "; ".join(errors)
                previous_value = value
                continue
            return StructuredResult(
                value=value,
                model=last_model,
                prompt_tokens=total_prompt,
                completion_tokens=total_completion,
                total_tokens=total_tokens,
                attempts=attempt,
            )
        errors = [
            item.strip()
            for item in error.split(";")
            if item.strip()
        ]
        raise StructuredValidationError(
            f"structured role failed validation: {error}",
            previous_value=previous_value,
            errors=errors or [error],
            model=last_model,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_tokens,
            attempts=self.max_attempts,
        )

    def repair(
        self,
        previous_value: Mapping[str, Any],
        errors: list[str],
        *,
        retry_context: Callable[
            [Mapping[str, Any], list[str]], str
        ],
        max_tokens: int,
        temperature: float,
    ) -> StructuredResult:
        """Run a bounded structured repair from the rejected JSON itself.

        ``call`` intentionally raises after exhausting its local attempts, but
        the caller may still own a broader revision budget.  Exposing this
        method lets that outer loop continue from the exact rejected object
        instead of discarding it and asking the model to redesign from
        scratch.
        """

        return self.call(
            retry_context(previous_value, errors),
            max_tokens=max_tokens,
            temperature=temperature,
            retry_context=retry_context,
        )


class RoleRouter:
    """Create exactly three model-tier clients without importing old control."""

    def __init__(
        self,
        researchclaw_config: str | Path,
        *,
        audit_root: str | Path,
        decision_role: str,
        worker_role: str,
        utility_role: str,
    ) -> None:
        from researchclaw.config import RCConfig
        from researchclaw.llm import create_llm_client

        config = RCConfig.load(
            Path(researchclaw_config).expanduser(),
            check_paths=False,
        )
        root = Path(audit_root)
        tiers = config.llm.model_tiers

        def build(tier_name: str, role: str) -> _AuditedClient:
            tier = getattr(tiers, tier_name)
            llm = replace(
                config.llm,
                primary_model=tier.model or config.llm.primary_model,
                fallback_models=(
                    tier.fallback_models
                    if tier.model
                    else config.llm.fallback_models
                ),
            )
            backend = create_llm_client(replace(config, llm=llm))
            return _AuditedClient(
                backend=backend,
                role=role,
                tier=tier_name,
                audit_path=root / tier_name / "calls.jsonl",
            )

        self.decision = build("decision", decision_role)
        self.worker = build("worker", worker_role)
        self.utility = build("utility", utility_role)


class _AuditedClient:
    def __init__(
        self,
        *,
        backend: Any,
        role: str,
        tier: str,
        audit_path: Path,
    ) -> None:
        self.backend = backend
        self.role = role
        self.tier = tier
        self.audit_path = audit_path
        self._lock = threading.Lock()

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        started = datetime.now(UTC)
        try:
            response = self.backend.chat(messages, **kwargs)
        except Exception as exc:
            self._append(
                {
                    "timestamp": started.isoformat(),
                    "role": self.role,
                    "tier": self.tier,
                    "outcome": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        self._append(
            {
                "timestamp": started.isoformat(),
                "role": self.role,
                "tier": self.tier,
                "outcome": "success",
                "model": str(getattr(response, "model", "") or ""),
                "prompt_tokens": int(
                    getattr(response, "prompt_tokens", 0) or 0
                ),
                "completion_tokens": int(
                    getattr(response, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(
                    getattr(response, "total_tokens", 0) or 0
                ),
            }
        )
        return response

    def _append(self, value: Mapping[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    dict(value),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
