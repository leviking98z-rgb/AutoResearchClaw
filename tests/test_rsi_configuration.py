from __future__ import annotations

import json
from pathlib import Path

import yaml

from researchclaw.rsi.configuration import prepare_cycle_config
from researchclaw.rsi.storage import CampaignStore


def test_prepare_cycle_config_wires_bridge_memory_and_safety(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    extra = base_dir / "experiment.md"
    extra.write_text("Always run a baseline.", encoding="utf-8")
    base = base_dir / "config.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "test", "mode": "docs-first"},
                "research": {"topic": "old"},
                "knowledge_base": {"backend": "markdown", "root": "kb"},
                "llm": {
                    "provider": "acp",
                    "roles": {
                        "idea_scientist": {
                            "model": "old-model",
                            "temperature": 0.8,
                        },
                        "coding_engineer": {
                            "provider": "openai-compatible",
                            "base_url": "https://code.example/v1",
                            "model": "dedicated-code-model",
                        },
                    },
                },
                "security": {"allow_publish_without_approval": True},
                "experiment": {
                    "cli_agent": {"provider": "cursor"},
                    "clusterbridge_pool": {
                        "config_file": "config.cluster32.yaml"
                    },
                },
                "prompts": {
                    "extra_prompts": {"experiment_design": "experiment.md"}
                },
                "hitl": {"enabled": True},
                "skills": {"custom_dirs": ["skills"]},
                "memory": {"store_dir": "memory"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    store = CampaignStore(tmp_path / "campaign")
    store.initialize()
    store.shared_prompt_path.write_text("campaign guidance", encoding="utf-8")
    (store.shared_dir / "knowledge_entries.jsonl").write_text(
        '{"name":"validated-batch","content":"Batch 128 was stable on H20."}\n',
        encoding="utf-8",
    )

    output = prepare_cycle_config(
        base_config=base,
        output_path=store.runs_dir / "cycle-0001" / "config.yaml",
        store=store,
        topic="new falsifiable topic",
        model="codebuddy/deepseek-v4-pro-ioa",
        bridge_url="http://127.0.0.1:8787/v1",
        api_key_env="BRIDGE_LOCAL_API_KEY",
        timeout_sec=1800,
    )
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert data["project"]["mode"] == "full-auto"
    assert data["research"]["topic"] == "new falsifiable topic"
    assert data["llm"]["provider"] == "openai-compatible"
    assert data["llm"]["primary_model"] == "codebuddy/deepseek-v4-pro-ioa"
    assert data["llm"]["fallback_models"] == []
    assert (
        data["llm"]["roles"]["idea_scientist"]["model"]
        == "old-model"
    )
    assert data["llm"]["roles"]["idea_scientist"]["temperature"] == 0.8
    assert (
        data["llm"]["roles"]["coding_engineer"]["model"]
        == "dedicated-code-model"
    )
    assert data["experiment"]["cli_agent"]["provider"] == "llm"
    assert data["security"]["allow_publish_without_approval"] is False
    assert data["hitl"]["enabled"] is False
    assert str(store.shared_skills_dir) in data["skills"]["custom_dirs"]
    assert Path(data["knowledge_base"]["root"]).is_absolute()
    assert Path(data["memory"]["store_dir"]).is_absolute()
    assert data["experiment"]["clusterbridge_pool"]["config_file"] == str(
        (base_dir / "config.cluster32.yaml").resolve()
    )

    merged = Path(data["prompts"]["extra_prompts"]["experiment_design"])
    assert merged.is_file()
    merged_text = merged.read_text(encoding="utf-8")
    assert "Always run a baseline." in merged_text
    assert "campaign guidance" in merged_text
    assert "Batch 128 was stable on H20." in merged_text
    assert str(store.shared_prompt_path) in merged_text
    paper_guidance = Path(data["prompts"]["extra_prompts"]["paper_draft"])
    assert paper_guidance.is_file()
    assert "campaign guidance" in paper_guidance.read_text(encoding="utf-8")
    assert "experiment_run" not in data["prompts"]["extra_prompts"]
    assert "iterative_refine" not in data["prompts"]["extra_prompts"]
    assert "citation_verify" not in data["prompts"]["extra_prompts"]


def test_prepare_cycle_config_preserves_three_tier_routing(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "test", "mode": "docs-first"},
                "research": {"topic": "placeholder"},
                "knowledge_base": {"backend": "markdown", "root": "kb"},
                "llm": {
                    "provider": "openai-compatible",
                    "model_tiers": {
                        "decision": {"model": "codebuddy/gpt-5.6-sol"},
                        "worker": {"model": "codebuddy/claude-sonnet-5"},
                        "utility": {"model": "codebuddy/claude-haiku-4.5"},
                    },
                    "roles": {
                        "topic_selector": {"temperature": 0.2},
                        "coding_engineer": {"temperature": 0.1},
                        "literature_researcher": {"temperature": 0.0},
                    },
                },
                "experiment": {"cli_agent": {"provider": "llm"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    store = CampaignStore(tmp_path / "campaign")
    store.initialize()

    output = prepare_cycle_config(
        base_config=base,
        output_path=store.runs_dir / "cycle-0001" / "config.yaml",
        store=store,
        topic="tiered topic",
        model="legacy-default-model",
        bridge_url="http://127.0.0.1:8787/v1",
        api_key_env="BRIDGE_LOCAL_API_KEY",
        timeout_sec=1800,
    )
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert data["llm"]["primary_model"] == "legacy-default-model"
    assert data["llm"]["model_tiers"]["decision"]["model"] == (
        "codebuddy/gpt-5.6-sol"
    )
    assert data["llm"]["model_tiers"]["worker"]["model"] == (
        "codebuddy/claude-sonnet-5"
    )
    assert data["llm"]["model_tiers"]["utility"]["model"] == (
        "codebuddy/claude-haiku-4.5"
    )
    assert "model" not in data["llm"]["roles"]["topic_selector"]
    assert "model" not in data["llm"]["roles"]["coding_engineer"]
    assert "model" not in data["llm"]["roles"]["literature_researcher"]


def test_prepare_cycle_config_separates_meta_brief_from_selected_topic(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "test", "mode": "docs-first"},
                "research": {"topic": "placeholder"},
                "knowledge_base": {"backend": "markdown", "root": "kb"},
                "llm": {"provider": "acp"},
                "experiment": {"cli_agent": {"provider": "llm"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    store = CampaignStore(tmp_path / "campaign")
    store.initialize()
    selected_path = store.runs_dir / "cycle-0001" / "selected_topic.json"
    selected_path.parent.mkdir(parents=True)
    selected_path.write_text('{"title":"Concrete RSI topic"}', encoding="utf-8")
    store.shared_topic_patch_path.write_text(
        '{"topic_action":"refine","topic_patch":"Evaluate the same gate at '
        'a fixed matched-token cap."}',
        encoding="utf-8",
    )

    output = prepare_cycle_config(
        base_config=base,
        output_path=selected_path.parent / "config.yaml",
        store=store,
        topic="Concrete RSI topic",
        campaign_brief="Broad RSI meta-brief",
        selected_topic_path=selected_path,
        autonomous_topic_selection=True,
        model="codebuddy/deepseek-v4-pro-ioa",
        bridge_url="http://127.0.0.1:8787/v1",
        api_key_env="BRIDGE_LOCAL_API_KEY",
        timeout_sec=1800,
    )

    data = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert data["research"]["topic"] == "Concrete RSI topic"
    assert data["research"]["campaign_brief"] == "Broad RSI meta-brief"
    assert data["research"]["selected_topic_file"] == str(selected_path)
    assert data["research"]["autonomous_topic_selection"] is True
    topic_prompt = Path(data["prompts"]["extra_prompts"]["topic_init"])
    prompt_text = topic_prompt.read_text(encoding="utf-8")
    assert "Concrete RSI topic" in prompt_text
    assert "Broad RSI meta-brief" in prompt_text
    assert "fixed matched-token cap" in prompt_text
    assert "Accepted Bounded Topic Refinement" in prompt_text
    assert "not the concrete research topic" in prompt_text
    assert "already implemented by the outer RSI supervisor" in prompt_text
    assert "Do not require experiment code to reimplement" in prompt_text


def test_prepare_cycle_config_materializes_only_unexpired_repair(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "test"},
                "research": {"topic": "placeholder"},
                "knowledge_base": {"backend": "markdown", "root": "kb"},
                "llm": {"provider": "openai-compatible"},
                "experiment": {"cli_agent": {"provider": "llm"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    store = CampaignStore(tmp_path / "campaign")
    store.initialize()
    store.shared_repair_patch_path.write_text(
        json.dumps(
            {
                "failure_signature": "sig-1",
                "source_cycle": 2,
                "expires_after_cycle": 4,
                "recovery_action": "regenerate",
                "repair_prompt_patch": "Use the verified dataset cache path.",
            }
        ),
        encoding="utf-8",
    )

    active = prepare_cycle_config(
        base_config=base,
        output_path=tmp_path / "active.yaml",
        store=store,
        topic="topic",
        model="model",
        bridge_url="http://bridge/v1",
        api_key_env="KEY",
        timeout_sec=10,
        cycle=4,
    )
    active_data = yaml.safe_load(active.read_text(encoding="utf-8"))
    active_text = Path(
        active_data["prompts"]["extra_prompts"]["code_generation"]
    ).read_text(encoding="utf-8")
    assert "Transient Failed-Cycle Engineering Repair" in active_text
    assert "verified dataset cache path" in active_text
    topic_text = Path(
        active_data["prompts"]["extra_prompts"]["topic_init"]
    ).read_text(encoding="utf-8")
    gate_text = Path(
        active_data["prompts"]["extra_prompts"]["quality_gate"]
    ).read_text(encoding="utf-8")
    export_text = Path(
        active_data["prompts"]["extra_prompts"]["export_publish"]
    ).read_text(encoding="utf-8")
    assert "Transient Failed-Cycle Engineering Repair" not in topic_text
    assert "Transient Failed-Cycle Engineering Repair" not in gate_text
    assert "Transient Failed-Cycle Engineering Repair" not in export_text

    expired = prepare_cycle_config(
        base_config=base,
        output_path=tmp_path / "expired.yaml",
        store=store,
        topic="topic",
        model="model",
        bridge_url="http://bridge/v1",
        api_key_env="KEY",
        timeout_sec=10,
        cycle=5,
    )
    expired_data = yaml.safe_load(expired.read_text(encoding="utf-8"))
    expired_text = Path(
        expired_data["prompts"]["extra_prompts"]["code_generation"]
    ).read_text(encoding="utf-8")
    assert "Transient Failed-Cycle Engineering Repair" not in expired_text
