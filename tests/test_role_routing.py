from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from researchclaw.config import RCConfig, validate_config
from researchclaw.experiment.code_agent import LlmCodeAgent
from researchclaw.llm.client import LLMResponse
from researchclaw.llm.roles import (
    DEFAULT_STAGE_ROLES,
    bind_role_llm_client,
    resolve_role,
    role_for_stage,
)
from researchclaw.pipeline.stages import Stage
from researchclaw.prompts import PromptManager


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            content="ok",
            model=str(kwargs.get("model") or "backend-default"),
            prompt_tokens=3,
            completion_tokens=2,
            total_tokens=5,
            finish_reason="stop",
        )


class ConfiguredRecordingBackend(RecordingBackend):
    def __init__(
        self,
        primary_model: str,
        fallback_models: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.config = type(
            "BackendConfig",
            (),
            {
                "primary_model": primary_model,
                "fallback_models": list(fallback_models),
            },
        )()


class MetadataRecordingBackend(RecordingBackend):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, **kwargs})
        return LLMResponse(
            content="ok",
            model="fallback-model",
            prompt_tokens=8,
            completion_tokens=5,
            total_tokens=13,
            finish_reason="length",
            truncated=True,
            attempts=3,
            retries=1,
            fallback_count=1,
            attempted_models=("primary-model", "fallback-model"),
        )


def _config(tmp_path: Path) -> RCConfig:
    data = {
        "project": {"name": "roles", "mode": "full-auto"},
        "research": {"topic": "role routing"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "console"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://global.example/v1",
            "api_key_env": "GLOBAL_KEY",
            "api_key": "global-key",
            "primary_model": "global-model",
            "fallback_models": ["global-fallback"],
            "timeout_sec": 300,
            "roles": {
                "idea_scientist": {
                    "model": "idea-model",
                    "fallback_models": ["idea-fallback"],
                    "temperature": 0.85,
                    "max_tokens": 9000,
                    "timeout_sec": 420,
                    "tools": ["literature"],
                },
                "coding_engineer": {
                    "provider": "openai-compatible",
                    "base_url": "https://code.example/v1",
                    "api_key_env": "CODE_KEY",
                    "model": "code-model",
                    "temperature": 0.1,
                    "max_tokens": 32000,
                },
                "skeptical_reviewer": {
                    "model": "review-model",
                    "temperature": 0.0,
                    "isolated_session": True,
                },
            },
        },
        "experiment": {"mode": "simulated"},
    }
    return RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)


def test_stage_role_map_covers_every_stage() -> None:
    assert set(DEFAULT_STAGE_ROLES) == set(Stage)
    assert role_for_stage(Stage.HYPOTHESIS_GEN) == "idea_scientist"
    assert role_for_stage(Stage.CODE_GENERATION) == "coding_engineer"
    assert role_for_stage(Stage.PEER_REVIEW) == "skeptical_reviewer"
    assert role_for_stage(Stage.CITATION_VERIFY) == "citation_auditor"


def test_role_config_parses_and_inherits_global_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    idea = resolve_role(config, "idea_scientist")
    code = resolve_role(config, "coding_engineer")

    assert idea.provider == "openai-compatible"
    assert idea.base_url == "https://global.example/v1"
    assert idea.api_key_env == "GLOBAL_KEY"
    assert idea.model == "idea-model"
    assert idea.fallback_models == ("idea-fallback",)
    assert idea.temperature == pytest.approx(0.85)
    assert idea.max_tokens == 9000
    assert idea.timeout_sec == 420
    assert idea.tools == ("literature",)

    assert code.base_url == "https://code.example/v1"
    assert code.api_key_env == "CODE_KEY"
    assert code.api_key == "global-key"
    assert code.model == "code-model"


def test_role_can_explicitly_disable_global_fallbacks(tmp_path: Path) -> None:
    base = _config(tmp_path)
    data = {
        "project": {"name": "roles", "mode": "full-auto"},
        "research": {"topic": "role routing"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "console"},
        "knowledge_base": {"backend": "markdown", "root": str(tmp_path / "kb")},
        "openclaw_bridge": {},
        "llm": {
            "provider": base.llm.provider,
            "base_url": base.llm.base_url,
            "api_key_env": "GLOBAL_KEY",
            "api_key": "global-key",
            "primary_model": "global-model",
            "fallback_models": ["global-fallback"],
            "roles": {
                "idea_scientist": {"fallback_models": []},
                "paper_writer": {},
            },
        },
        "experiment": {"mode": "simulated"},
    }
    config = RCConfig.from_dict(data, project_root=tmp_path, check_paths=False)

    assert resolve_role(config, "idea_scientist").fallback_models == ()
    assert resolve_role(config, "paper_writer").fallback_models == (
        "global-fallback",
    )


def test_role_client_applies_defaults_and_explicit_call_overrides(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    backend = RecordingBackend()
    client = bind_role_llm_client(
        backend,
        config,
        "idea_scientist",
        run_dir=tmp_path,
        stage=Stage.HYPOTHESIS_GEN,
    )

    client.chat([{"role": "user", "content": "generate"}])
    first = backend.calls[-1]
    assert first["model"] == "idea-model"
    assert first["max_tokens"] == 9000
    assert first["temperature"] == pytest.approx(0.85)

    client.chat(
        [{"role": "user", "content": "strict"}],
        max_tokens=123,
        temperature=0.0,
    )
    second = backend.calls[-1]
    assert second["max_tokens"] == 123
    assert second["temperature"] == 0.0


def test_native_role_backend_keeps_its_fallback_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    backend = ConfiguredRecordingBackend(
        "idea-model",
        ("idea-fallback",),
    )
    monkeypatch.setattr(
        "researchclaw.llm.create_llm_client",
        lambda _config: backend,
    )
    from researchclaw.llm.roles import create_role_llm_client

    client = create_role_llm_client(config, "idea_scientist")
    client.chat([{"role": "user", "content": "generate"}])

    assert backend.calls[-1]["model"] is None
    assert backend.config.fallback_models == ["idea-fallback"]


def test_explicit_call_model_still_overrides_native_role_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    backend = ConfiguredRecordingBackend("idea-model", ("idea-fallback",))
    monkeypatch.setattr(
        "researchclaw.llm.create_llm_client",
        lambda _config: backend,
    )
    from researchclaw.llm.roles import create_role_llm_client

    client = create_role_llm_client(config, "idea_scientist")
    client.chat(
        [{"role": "user", "content": "generate"}],
        model="one-off-model",
    )

    assert backend.calls[-1]["model"] == "one-off-model"


def test_role_client_writes_audit_without_prompt_content(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = RecordingBackend()
    client = bind_role_llm_client(
        backend,
        config,
        "skeptical_reviewer",
        run_dir=tmp_path,
        stage=Stage.PEER_REVIEW,
    )
    secret_prompt = "private writer scratchpad must not enter audit"
    client.chat([{"role": "user", "content": secret_prompt}])

    audit_path = tmp_path / "audit" / "llm-skeptical_reviewer.jsonl"
    row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["role"] == "skeptical_reviewer"
    assert row["stage"] == int(Stage.PEER_REVIEW)
    assert row["requested_model"] == "review-model"
    assert row["total_tokens"] == 5
    assert row["attempts"] == 1
    assert row["retries"] == 0
    assert row["fallback_count"] == 0
    assert row["truncated"] is False
    assert row["json_mode"] is False
    assert len(row["request_sha256"]) == 64
    assert secret_prompt not in audit_path.read_text(encoding="utf-8")
    pipeline_rows = [
        json.loads(line)
        for line in (tmp_path / "pipeline_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert pipeline_rows[-1]["type"] == "llm_call"
    assert pipeline_rows[-1]["role"] == "skeptical_reviewer"


def test_role_audit_records_retry_and_fallback_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = MetadataRecordingBackend()
    client = bind_role_llm_client(
        backend,
        config,
        "idea_scientist",
        run_dir=tmp_path,
        stage=Stage.HYPOTHESIS_GEN,
    )

    client.chat([{"role": "user", "content": "generate"}], json_mode=True)

    row = json.loads(
        (tmp_path / "audit" / "llm-idea_scientist.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert row["attempted_models"] == ["primary-model", "fallback-model"]
    assert row["attempts"] == 3
    assert row["retries"] == 1
    assert row["fallback_count"] == 1
    assert row["response_model"] == "fallback-model"
    assert row["truncated"] is True
    assert row["json_mode"] is True


def test_for_stage_rebinds_role_and_defaults(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = RecordingBackend()
    idea = bind_role_llm_client(
        backend,
        config,
        "idea_scientist",
        run_dir=tmp_path,
        stage=Stage.HYPOTHESIS_GEN,
    )
    reviewer = idea.for_stage(Stage.PEER_REVIEW)
    reviewer.chat([{"role": "user", "content": "review"}])

    assert reviewer.role == "skeptical_reviewer"
    assert reviewer.stage == Stage.PEER_REVIEW
    assert backend.calls[-1]["model"] == "review-model"
    assert backend.calls[-1]["temperature"] == 0.0
    assert (tmp_path / "audit" / "llm-skeptical_reviewer.jsonl").exists()


def test_role_with_alternate_provider_gets_distinct_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    idea_backend = RecordingBackend()
    code_backend = RecordingBackend()
    created: list[str] = []

    def fake_create(role_config: RCConfig) -> RecordingBackend:
        created.append(role_config.llm.base_url)
        return (
            code_backend
            if role_config.llm.base_url == "https://code.example/v1"
            else idea_backend
        )

    monkeypatch.setattr("researchclaw.llm.create_llm_client", fake_create)
    from researchclaw.llm.roles import create_role_llm_client

    idea = create_role_llm_client(config, "idea_scientist", run_dir=tmp_path)
    coding = idea.for_stage(Stage.CODE_GENERATION)
    coding.chat([{"role": "user", "content": "implement"}])

    assert created == [
        "https://global.example/v1",
        "https://code.example/v1",
    ]
    assert idea.backend is idea_backend
    assert coding.backend is code_backend
    assert code_backend.calls[-1]["model"] == "code-model"


def test_role_with_different_model_gets_distinct_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    idea_backend = ConfiguredRecordingBackend(
        "idea-model",
        ("idea-fallback",),
    )
    review_backend = ConfiguredRecordingBackend(
        "review-model",
        ("global-fallback",),
    )
    created: list[str] = []

    def fake_create(role_config: RCConfig) -> ConfiguredRecordingBackend:
        created.append(role_config.llm.primary_model)
        if role_config.llm.primary_model == "review-model":
            return review_backend
        return idea_backend

    monkeypatch.setattr("researchclaw.llm.create_llm_client", fake_create)
    from researchclaw.llm.roles import create_role_llm_client

    idea = create_role_llm_client(config, "idea_scientist")
    reviewer = idea.for_stage(Stage.PEER_REVIEW)
    reviewer.chat([{"role": "user", "content": "review"}])

    assert created == ["idea-model", "review-model"]
    assert reviewer.backend is review_backend
    assert review_backend.calls[-1]["model"] is None


def test_roles_with_same_backend_policy_reuse_one_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _config(tmp_path)
    roles = dict(base.llm.roles)
    shared_policy = replace(
        roles["idea_scientist"],
        model="shared-specialist-model",
        fallback_models=("shared-fallback",),
        temperature=0.7,
    )
    roles["idea_scientist"] = shared_policy
    roles["experiment_designer"] = replace(
        shared_policy,
        temperature=0.2,
        max_tokens=12000,
    )
    config = replace(base, llm=replace(base.llm, roles=roles))
    shared_backend = ConfiguredRecordingBackend(
        "shared-specialist-model",
        ("shared-fallback",),
    )
    created: list[str] = []

    def fake_create(role_config: RCConfig) -> ConfiguredRecordingBackend:
        created.append(role_config.llm.primary_model)
        return shared_backend

    monkeypatch.setattr("researchclaw.llm.create_llm_client", fake_create)
    from researchclaw.llm.roles import create_role_llm_client

    idea = create_role_llm_client(config, "idea_scientist")
    designer = idea.for_stage(Stage.EXPERIMENT_DESIGN)
    assert designer.backend is shared_backend
    assert created == ["shared-specialist-model"]


def test_acp_isolated_roles_keep_distinct_backend_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _config(tmp_path)
    roles = {
        "idea_scientist": replace(
            base.llm.roles["idea_scientist"],
            provider="acp",
            base_url="",
            api_key_env="",
            api_key="",
            model="shared-acp-model",
            fallback_models=(),
            isolated_session=True,
        ),
        "experiment_designer": replace(
            base.llm.roles["idea_scientist"],
            provider="acp",
            base_url="",
            api_key_env="",
            api_key="",
            model="shared-acp-model",
            fallback_models=(),
            isolated_session=True,
        ),
    }
    config = replace(
        base,
        llm=replace(
            base.llm,
            provider="acp",
            primary_model="shared-acp-model",
            fallback_models=(),
            roles=roles,
        ),
    )
    created_sessions: list[str] = []

    def fake_create(role_config: RCConfig) -> RecordingBackend:
        created_sessions.append(role_config.llm.acp.session_name)
        return RecordingBackend()

    monkeypatch.setattr("researchclaw.llm.create_llm_client", fake_create)
    from researchclaw.llm.roles import create_role_llm_client

    idea = create_role_llm_client(config, "idea_scientist")
    designer = idea.for_stage(Stage.EXPERIMENT_DESIGN)

    assert designer.backend is not idea.backend
    assert created_sessions == [
        "researchclaw-idea_scientist",
        "researchclaw-experiment_designer",
    ]


def test_llm_code_agent_honors_cli_model_override(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    roles = dict(base.llm.roles)
    roles["coding_engineer"] = replace(
        roles["coding_engineer"],
        provider="",
        base_url="",
        api_key_env="",
        model="",
    )
    config = replace(
        base,
        llm=replace(base.llm, roles=roles),
        experiment=replace(
            base.experiment,
            cli_agent=replace(
                base.experiment.cli_agent,
                model="dedicated-cli-code-model",
            ),
        ),
    )
    backend = RecordingBackend()
    agent = LlmCodeAgent(backend, PromptManager(), config)

    result = agent.generate(
        exp_plan="objective: smoke test",
        topic="role routing",
        metric_key="accuracy",
        pkg_hint="numpy",
        compute_budget="cpu",
        extra_guidance="",
        workdir=tmp_path,
    )

    assert not result.error
    assert backend.calls[-1]["model"] == "dedicated-cli-code-model"


@pytest.mark.parametrize(
    ("roles", "expected"),
    [
        ([], "llm.roles must be a mapping"),
        (
            {"idea_scientist": {"temperature": 3.0}},
            "llm.roles.idea_scientist.temperature must be between 0 and 2",
        ),
        (
            {"coding_engineer": {"max_tokens": 0}},
            "llm.roles.coding_engineer.max_tokens must be a positive integer",
        ),
        (
            {"paper_writer": {"tools": "filesystem"}},
            "llm.roles.paper_writer.tools must be a list",
        ),
    ],
)
def test_role_config_validation(roles: object, expected: str) -> None:
    data: dict[str, Any] = {
        "project": {"name": "roles", "mode": "full-auto"},
        "research": {"topic": "role routing"},
        "runtime": {"timezone": "UTC"},
        "notifications": {"channel": "console"},
        "knowledge_base": {"backend": "markdown", "root": "kb"},
        "openclaw_bridge": {},
        "llm": {
            "provider": "openai-compatible",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "KEY",
            "roles": roles,
        },
        "experiment": {"mode": "simulated"},
    }
    result = validate_config(data, check_paths=False)
    assert result.ok is False
    assert expected in result.errors
