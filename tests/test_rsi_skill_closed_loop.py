from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from researchclaw.pipeline import _helpers
from researchclaw.rsi.configuration import prepare_cycle_config
from researchclaw.rsi.storage import CampaignStore


def test_cycle_two_prompt_uses_campaign_skill_registry(tmp_path: Path) -> None:
    """A skill promoted after cycle 1 must reach a real cycle-2 overlay."""

    store = CampaignStore(tmp_path / "campaign")
    store.initialize()
    skill_dir = store.shared_skills_dir / "arc-cycle-one"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: arc-cycle-one
description: Campaign lesson for experiment design.
metadata:
  category: experiment
  trigger-keywords: "experiment_design"
  applicable-stages: "9"
  priority: "1"
---

CAMPAIGN_SKILL_FROM_CYCLE_ONE
""",
        encoding="utf-8",
    )

    base = tmp_path / "config.yaml"
    base.write_text(
        """
project:
  name: skill-loop
research:
  topic: placeholder
runtime:
  timezone: UTC
notifications:
  channel: none
knowledge_base:
  backend: markdown
  root: .
llm:
  provider: openai-compatible
  base_url: http://127.0.0.1:8787/v1
  api_key_env: BRIDGE_LOCAL_API_KEY
experiment:
  mode: simulated
""",
        encoding="utf-8",
    )
    generated = prepare_cycle_config(
        base_config=base,
        output_path=store.runs_dir / "cycle-0002" / "config.yaml",
        store=store,
        topic="test experiment design",
        model="codebuddy/deepseek-v4-pro-ioa",
        bridge_url="http://127.0.0.1:8787/v1",
        api_key_env="BRIDGE_LOCAL_API_KEY",
        timeout_sec=1800,
    )

    import yaml

    data = yaml.safe_load(generated.read_text(encoding="utf-8"))
    config = SimpleNamespace(
        skills=SimpleNamespace(
            custom_dirs=tuple(data["skills"]["custom_dirs"]),
            external_dirs=tuple(data["skills"]["external_dirs"]),
            max_skills_per_stage=3,
        )
    )
    _helpers._skill_registries.clear()
    _helpers._active_skill_registry_key = None
    _helpers._get_skill_registry(config)

    overlay = _helpers._get_evolution_overlay(
        store.runs_dir / "cycle-0002",
        "experiment_design",
    )

    assert "CAMPAIGN_SKILL_FROM_CYCLE_ONE" in overlay


def test_generated_generic_experiment_keyword_matches_real_stage(
    tmp_path: Path,
) -> None:
    store = CampaignStore(tmp_path / "campaign")
    store.initialize()
    skill_dir = store.shared_skills_dir / "arc-generated"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: arc-generated
description: Generated experiment lesson.
metadata:
  category: a-evolve
  trigger-keywords: "research,pipeline,quality,experiment,obs-1"
  applicable-stages: "9"
  priority: "2"
---

GENERIC_EXPERIMENT_SKILL
""",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        skills=SimpleNamespace(
            custom_dirs=(str(store.shared_skills_dir),),
            external_dirs=(),
            max_skills_per_stage=3,
        )
    )
    _helpers._skill_registries.clear()
    _helpers._active_skill_registry_key = None
    _helpers._get_skill_registry(config)

    overlay = _helpers._get_evolution_overlay(
        store.runs_dir / "cycle-0002",
        "experiment_design",
        config=config,
        topic="",
    )

    assert "GENERIC_EXPERIMENT_SKILL" in overlay
