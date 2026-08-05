from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from researchclaw.rsi.cli import _load_run_policy, _recorded_process_is_alive
from researchclaw.rsi.storage import CampaignStore
from researchclaw.rsi.supervisor import CampaignSupervisor, SupervisorOptions


def _base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "rsi-test", "mode": "docs-first"},
                "research": {
                    "topic": "placeholder",
                    "domains": [],
                    "quality_threshold": 5.0,
                },
                "runtime": {"timezone": "UTC"},
                "notifications": {"channel": "console"},
                "knowledge_base": {"backend": "markdown", "root": "kb"},
                "openclaw_bridge": {},
                "llm": {"provider": "openai-compatible"},
                "literature_search": {},
                "security": {},
                "experiment": {"cli_agent": {"provider": "llm"}},
                "export": {},
                "prompts": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


class FakeProcess:
    _pid = 42000

    def __init__(self, returncode: int = 0) -> None:
        type(self)._pid += 1
        self.pid = type(self)._pid
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _pipeline_factory(
    summaries: list[dict[str, object]],
    returncodes: list[int],
    seen_commands: list[list[str]],
):
    calls = {"index": 0}

    def factory(command, **_kwargs):
        index = calls["index"]
        calls["index"] += 1
        seen_commands.append(list(command))
        run_dir = Path(command[command.index("--output") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "pipeline_summary.json").write_text(
            json.dumps(summaries[index]),
            encoding="utf-8",
        )
        return FakeProcess(returncodes[index])

    return factory


def _options(
    tmp_path: Path,
    *,
    single_cycle: bool = False,
    max_cycles: int = 0,
    max_failures: int = 3,
) -> SupervisorOptions:
    return SupervisorOptions(
        campaign_dir=tmp_path / "campaign",
        repo_root=tmp_path,
        base_config=_base_config(tmp_path),
        topic="Does method X improve metric Y over baseline Z?",
        single_cycle=single_cycle,
        max_cycles=max_cycles,
        max_consecutive_failures=max_failures,
        backoff_initial_sec=0,
        backoff_max_sec=0,
        heartbeat_interval_sec=0.01,
        control_poll_sec=0.01,
    )


def _autonomous_options(
    tmp_path: Path,
    *,
    single_cycle: bool = True,
) -> SupervisorOptions:
    options = _options(tmp_path, single_cycle=single_cycle)
    data = yaml.safe_load(options.base_config.read_text(encoding="utf-8"))
    data["research"]["autonomous_topic_selection"] = True
    options.base_config.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return options


def _topic_candidate(index: int) -> dict[str, object]:
    score = 6.0 + index / 10
    return {
        "id": f"topic-{index:02d}",
        "title": f"Autonomous RSI topic {index}",
        "research_question": f"Does mechanism {index} transfer?",
        "falsifiable_hypothesis": (
            f"Mechanism {index} improves held-out accuracy."
        ),
        "closest_prior_work": ["self-refinement"],
        "novelty_gap": "The transfer mechanism is not isolated.",
        "datasets": ["GSM8K"],
        "models": ["open 7B model"],
        "compute": {
            "gpu_count": 1,
            "wall_clock_hours": 2,
            "notes": "cheap pilot",
        },
        "primary_metric": "held-out accuracy",
        "baselines": ["single-pass no-self-improvement control"],
        "ablations": ["remove gate"],
        "failure_safety_tests": ["regression"],
        "implementation_feasibility": "Public tasks and local inference.",
        "licensing_feasibility": "Permissive public assets.",
        "information_gain_if_true": "Identifies a useful mechanism.",
        "information_gain_if_false": "Rules out the mechanism.",
        "cheap_pilot": "Run 50 examples.",
        "scores": {
            "novelty": score,
            "scientific_importance": score,
            "falsifiability": score,
            "compute_tractability": score,
            "reproducibility": score,
            "meaningful_result_likelihood": score,
            "risk": 2.0,
        },
    }


def _topic_document() -> dict[str, object]:
    return {
        "candidates": [_topic_candidate(index) for index in range(1, 13)],
        "selected_candidate_id": "topic-12",
        "selection_rationale": "Highest validated score.",
        "pivot_policy": "Pivot if the pilot signal is absent.",
    }


def test_autonomous_selection_uses_selected_topic_in_pipeline_config(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    summary = {
        "run_id": "rc-topic",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }

    class TopicLLM:
        def chat(self, *_args, **_kwargs):
            return type("Response", (), {"content": json.dumps(_topic_document())})()

    supervisor = CampaignSupervisor(
        _autonomous_options(tmp_path),
        popen_factory=_pipeline_factory([summary], [0], commands),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: TopicLLM(),
        diagnosis_fn=lambda **_kwargs: {
            "summary": "Keep the incumbent.",
            "topic_action": "keep",
            "topic_patch": "",
            "pivot_reason": "",
            "preferred_candidate_id": "",
            "prompt_patch": "",
            "stop_recommended": False,
        },
        aevolve_fn=lambda **_kwargs: [],
    )

    assert supervisor.run() == 0
    store = CampaignStore(tmp_path / "campaign")
    run_dir = store.runs_dir / "cycle-0001"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["research"]["topic"] == "Autonomous RSI topic 12"
    assert config["research"]["campaign_brief"].startswith("Does method X")
    assert (run_dir / "topic_candidates.json").is_file()
    assert (run_dir / "selected_topic.json").is_file()
    assert store.shared_topic_candidates_path.is_file()
    assert store.shared_selected_topic_path.is_file()
    assert len(commands) == 1


def test_autonomous_selection_failure_does_not_start_pipeline(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    invalid = _topic_document()
    invalid["candidates"] = invalid["candidates"][:11]  # type: ignore[index]

    class InvalidTopicLLM:
        def chat(self, *_args, **_kwargs):
            return type("Response", (), {"content": json.dumps(invalid)})()

    supervisor = CampaignSupervisor(
        replace(
            _autonomous_options(tmp_path),
            max_consecutive_failures=1,
        ),
        popen_factory=_pipeline_factory([], [], commands),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: InvalidTopicLLM(),
        diagnosis_fn=lambda **_kwargs: {},
        aevolve_fn=lambda **_kwargs: [],
    )

    assert supervisor.run() == 1
    assert commands == []
    state = CampaignStore(tmp_path / "campaign").load_state()
    assert "at least 12" in state["last_error"]


def test_autonomous_next_cycle_keeps_incumbent_without_selector_call(
    tmp_path: Path,
) -> None:
    options = _autonomous_options(tmp_path)
    summary = {
        "run_id": "rc-topic",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    calls = {"selector": 0}

    class TopicLLM:
        def chat(self, *_args, **_kwargs):
            calls["selector"] += 1
            return type("Response", (), {"content": json.dumps(_topic_document())})()

    diagnosis = {
        "summary": "Keep the incumbent.",
        "topic_action": "keep",
        "topic_patch": "",
        "pivot_reason": "",
        "preferred_candidate_id": "",
        "prompt_patch": "",
        "stop_recommended": False,
    }
    first = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary], [0], []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: TopicLLM(),
        diagnosis_fn=lambda **_kwargs: diagnosis,
        aevolve_fn=lambda **_kwargs: [],
    )
    assert first.run() == 0

    second = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary], [0], []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: (_ for _ in ()).throw(
            AssertionError("selector must not be called for topic_action=keep")
        ),
        diagnosis_fn=lambda **_kwargs: diagnosis,
        aevolve_fn=lambda **_kwargs: [],
    )
    assert second.run() == 0
    assert calls["selector"] == 1
    selected = json.loads(
        (
            CampaignStore(tmp_path / "campaign").runs_dir
            / "cycle-0002"
            / "selected_topic.json"
        ).read_text(encoding="utf-8")
    )
    assert selected["id"] == "topic-12"


def test_topic_pivot_uses_preferred_existing_candidate_and_local_baseline(
    tmp_path: Path,
) -> None:
    supervisor = CampaignSupervisor(_autonomous_options(tmp_path))
    supervisor.initialize()
    store = supervisor.store
    from researchclaw.rsi.topic_selection import (
        persist_topic_selection,
        validate_selection_document,
    )

    persist_topic_selection(
        shared_dir=store.shared_dir,
        run_dir=store.runs_dir / "cycle-0001",
        selection=validate_selection_document(_topic_document()),
        cycle=1,
    )
    prior_evidence = store.runs_dir / "cycle-0001" / "rsi_evidence.json"
    prior_evidence.write_text(
        json.dumps({"topic_id": "topic-12", "composite_score": 90.0}),
        encoding="utf-8",
    )
    supervisor.state.update(
        {
            "best_evidence_path": str(prior_evidence),
            "best_run_dir": str(prior_evidence.parent),
            "pending_topic_action": {
                "topic_action": "pivot",
                "topic_patch": "",
                "pivot_reason": "The incumbent pilot signal stayed at chance.",
                "preferred_candidate_id": "topic-11",
            },
        }
    )
    supervisor.state = store.save_state(supervisor.state)

    assert supervisor._load_best_evidence("topic-11") is None
    from researchclaw.rsi.topic_selection import (
        selection_from_artifacts,
        selection_with_candidate,
    )

    incumbent = selection_from_artifacts(
        candidates_path=store.shared_topic_candidates_path,
        selected_path=store.shared_selected_topic_path,
    )
    pivoted = selection_with_candidate(
        incumbent,
        candidate_id="topic-11",
        rationale="The incumbent pilot signal stayed at chance.",
    )
    assert pivoted.selected["id"] == "topic-11"


def test_single_cycle_runs_diagnosis_aevolve_and_checkpoints(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    summary = {
        "run_id": "rc-test",
        "stages_executed": 23,
        "stages_done": 23,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    diagnosis = {
        "summary": "Need a stronger ablation.",
        "strengths": ["reproducible"],
        "weaknesses": ["ablation"],
        "next_cycle_priorities": ["add ablation"],
        "brief_patch": "Refined brief for cycle two",
        "prompt_patch": "Always compare against the strongest baseline.",
        "stop_recommended": False,
        "stop_reason": "",
    }

    seen_cancel_event: list[threading.Event] = []

    def aevolve(**kwargs):
        seen_cancel_event.append(kwargs["cancel_event"])
        skill = kwargs["skills_dir"] / "arc-aevolve-test"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            "---\nname: arc-aevolve-test\ndescription: test\n---\nbody\n",
            encoding="utf-8",
        )
        return ["arc-aevolve-test"]

    supervisor = CampaignSupervisor(
        _options(tmp_path, single_cycle=True),
        popen_factory=_pipeline_factory([summary], [0], commands),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: diagnosis,
        aevolve_fn=aevolve,
    )
    assert supervisor.run() == 0
    assert seen_cancel_event == [supervisor._cancel_event]

    store = CampaignStore(tmp_path / "campaign")
    state = store.load_state()
    assert state["status"] == "paused_single_cycle"
    assert state["next_cycle"] == 2
    assert state["completed_cycles"] == 1
    assert state["successful_cycles"] == 1
    assert state["consecutive_failures"] == 0
    assert state["last_mutations"] == ["arc-aevolve-test"]
    assert store.shared_brief_path.read_text().strip() == (
        "Does method X improve metric Y over baseline Z?"
    )
    assert "strongest baseline" in store.shared_prompt_path.read_text()
    assert (store.runs_dir / "cycle-0001" / "rsi_evidence.json").is_file()
    assert (store.diagnostics_dir / "cycle-0001.json").is_file()
    assert (
        store.shared_skills_dir / "arc-aevolve-test" / "SKILL.md"
    ).is_file()
    assert "--auto-approve" in commands[0]
    assert "--no-graceful-degradation" in commands[0]
    assert "--topic" not in commands[0]
    policy = json.loads(store.policy_path.read_text(encoding="utf-8"))
    assert policy["model"] == "codebuddy/claude-sonnet-5"
    assert policy["continuous"] is False

    events = [event["type"] for event in store.log.read_all()]
    assert events == [
        "campaign_created",
        "supervisor_started",
        "cycle_started",
        "cycle_completed",
        "single_cycle_complete",
    ]


def test_consecutive_failures_trigger_durable_pause(tmp_path: Path) -> None:
    commands: list[list[str]] = []
    failed_summary = {
        "run_id": "rc-fail",
        "stages_executed": 1,
        "stages_done": 0,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 1,
        "final_status": "failed",
    }
    supervisor = CampaignSupervisor(
        _options(tmp_path, max_failures=2),
        popen_factory=_pipeline_factory(
            [failed_summary, failed_summary], [1, 1], commands
        ),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: {
            "summary": "failed",
            "brief_patch": "",
            "prompt_patch": "",
            "stop_recommended": False,
        },
        aevolve_fn=lambda **_kwargs: [],
    )
    assert supervisor.run() == 1

    store = CampaignStore(tmp_path / "campaign")
    state = store.load_state()
    assert state["status"] == "paused_failure_threshold"
    assert state["completed_cycles"] == 2
    assert state["failed_cycles"] == 2
    assert state["consecutive_failures"] == 2
    assert state["next_cycle"] == 3
    assert store.control_requested("pause")
    assert (store.root / "pause.request.json").is_file()
    assert len(commands) == 2


def test_resume_continues_from_next_cycle(tmp_path: Path) -> None:
    options = _options(tmp_path, single_cycle=True)
    first_commands: list[list[str]] = []
    summary = {
        "run_id": "rc-ok",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    diagnosis = {
        "summary": "ok",
        "brief_patch": "",
        "prompt_patch": "",
        "stop_recommended": False,
    }
    first = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary], [0], first_commands),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: diagnosis,
        aevolve_fn=lambda **_kwargs: [],
    )
    assert first.run() == 0

    store = CampaignStore(options.campaign_dir)
    store.clear_control("pause")
    store.clear_control("stop")
    second_commands: list[list[str]] = []
    second = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary], [0], second_commands),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: diagnosis,
        aevolve_fn=lambda **_kwargs: [],
    )
    assert second.run() == 0

    state = store.load_state()
    assert state["completed_cycles"] == 2
    assert state["next_cycle"] == 3
    assert (store.runs_dir / "cycle-0001").is_dir()
    assert (store.runs_dir / "cycle-0002").is_dir()


def test_reconcile_accepts_early_identity_of_current_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = CampaignSupervisor(_options(tmp_path, single_cycle=True))
    supervisor.initialize()
    supervisor.state.update(
        {
            "status": "running",
            "pid": 55001,
            "supervisor_start_ticks": 777,
            "active_cycle": None,
            "active_run_dir": None,
        }
    )
    supervisor.state = supervisor.store.save_state(supervisor.state)
    monkeypatch.setattr(
        "researchclaw.rsi.supervisor._process_identity_matches",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("current supervisor must not be treated as stale")
        ),
    )

    supervisor._reconcile_interrupted_execution(
        current_pid=55001,
        current_start_ticks=777,
    )

    assert supervisor.state["status"] == "running"
    assert supervisor.state["pid"] == 55001


def test_lessons_are_shared_across_cycles_without_duplication(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path, max_cycles=2)
    commands: list[list[str]] = []
    summary = {
        "run_id": "rc-ok",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    lesson_one = json.dumps({"description": "cycle one lesson"}) + "\n"
    lesson_two = json.dumps({"description": "cycle two lesson"}) + "\n"
    aevolve_calls = {"count": 0}

    def aevolve(**kwargs):
        aevolve_calls["count"] += 1
        path = kwargs["run_dir"] / "evolution" / "lessons.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(lesson_one if aevolve_calls["count"] == 1 else lesson_two)
        return []

    supervisor = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary, summary], [0, 0], commands),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: {
            "summary": "continue",
            "brief_patch": "",
            "prompt_patch": "",
            "stop_recommended": False,
        },
        aevolve_fn=aevolve,
    )
    assert supervisor.run() == 0

    store = CampaignStore(options.campaign_dir)
    shared = (store.shared_dir / "lessons.jsonl").read_text(encoding="utf-8")
    # Cycle two tied the incumbent score and was rejected, so its candidate
    # learning must remain run-local instead of changing campaign behavior.
    assert shared == lesson_one
    cycle_two = (
        store.runs_dir / "cycle-0002" / "evolution" / "lessons.jsonl"
    ).read_text(encoding="utf-8")
    assert cycle_two == lesson_one + lesson_two


def test_knowledge_is_seeded_and_promoted_without_duplication(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path, max_cycles=2)
    summary = {
        "run_id": "rc-ok",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    entry_one = json.dumps({"name": "one", "content": "first"}) + "\n"
    entry_two = json.dumps({"name": "two", "content": "second"}) + "\n"
    calls = {"count": 0}

    def aevolve(**kwargs):
        calls["count"] += 1
        path = kwargs["run_dir"] / "evolution" / "knowledge_entries.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry_one if calls["count"] == 1 else entry_two)
        return []

    supervisor = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary, summary], [0, 0], []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: {
            "summary": "continue",
            "brief_patch": "",
            "prompt_patch": "",
            "stop_recommended": False,
        },
        aevolve_fn=aevolve,
    )

    assert supervisor.run() == 0

    store = CampaignStore(options.campaign_dir)
    shared = (
        store.shared_dir / "knowledge_entries.jsonl"
    ).read_text(encoding="utf-8")
    assert shared == entry_one
    cycle_two = (
        store.runs_dir
        / "cycle-0002"
        / "evolution"
        / "knowledge_entries.jsonl"
    ).read_text(encoding="utf-8")
    assert cycle_two == entry_one + entry_two


def test_rejected_cycle_does_not_promote_diagnosis_or_skill(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path, max_cycles=2)
    summary = {
        "run_id": "rc-ok",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    calls = {"count": 0}

    def diagnosis(**_kwargs):
        calls["count"] += 1
        return {
            "summary": "candidate",
            "brief_patch": f"BRIEF-{calls['count']}",
            "prompt_patch": f"PROMPT-{calls['count']}",
            "stop_recommended": False,
        }

    def aevolve(**kwargs):
        skill = (
            kwargs["skills_dir"]
            / f"arc-cycle-{calls['count']}"
        )
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text(
            f"---\nname: {skill.name}\ndescription: test\n---\n"
            f"SKILL-{calls['count']}\n",
            encoding="utf-8",
        )
        return [skill.name]

    supervisor = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary, summary], [0, 0], []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=diagnosis,
        aevolve_fn=aevolve,
    )

    assert supervisor.run() == 0

    store = CampaignStore(options.campaign_dir)
    assert store.shared_brief_path.read_text().strip() == (
        "Does method X improve metric Y over baseline Z?"
    )
    guidance = store.shared_prompt_path.read_text(encoding="utf-8")
    assert "PROMPT-1" in guidance
    assert "PROMPT-2" not in guidance
    assert (store.shared_skills_dir / "arc-cycle-1").is_dir()
    assert not (store.shared_skills_dir / "arc-cycle-2").exists()
    assert (
        store.runs_dir
        / "cycle-0002"
        / "evolution"
        / "candidate_skills"
        / "arc-cycle-2"
    ).is_dir()


def test_plateau_threshold_pauses_campaign(tmp_path: Path) -> None:
    options = replace(
        _options(tmp_path, max_cycles=10),
        max_no_improvement_cycles=2,
    )
    summary = {
        "run_id": "rc-ok",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    supervisor = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary] * 3, [0] * 3, []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: {
            "summary": "plateau",
            "brief_patch": "",
            "prompt_patch": "",
            "stop_recommended": False,
        },
        aevolve_fn=lambda **_kwargs: [],
    )

    assert supervisor.run() == 1
    state = CampaignStore(options.campaign_dir).load_state()
    assert state["status"] == "paused_no_improvement"
    assert state["completed_cycles"] == 3
    assert state["consecutive_no_improvement"] == 2


def test_continuous_mode_ignores_cycle_plateau_and_stop_recommendation(
    tmp_path: Path,
) -> None:
    options = replace(
        _options(tmp_path, max_cycles=1),
        continuous=True,
        max_no_improvement_cycles=1,
        max_consecutive_failures=1,
    )
    summary = {
        "run_id": "rc-ok",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    calls = {"count": 0}

    def diagnosis(**_kwargs):
        calls["count"] += 1
        return {
            "summary": "continue",
            "brief_patch": "",
            "prompt_patch": "",
            "stop_recommended": True,
            "stop_reason": "model requested stop",
        }

    supervisor = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary, summary], [0, 0], []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=diagnosis,
        aevolve_fn=lambda **_kwargs: [],
    )

    original_run_cycle = supervisor._run_cycle

    def bounded_run_cycle(cycle: int):
        result = original_run_cycle(cycle)
        if cycle == 2:
            supervisor.store.set_control("stop", "bounded test")
        return result

    supervisor._run_cycle = bounded_run_cycle  # type: ignore[method-assign]
    assert supervisor.run() == 0

    state = CampaignStore(options.campaign_dir).load_state()
    assert calls["count"] == 2
    assert state["completed_cycles"] == 2
    assert state["status"] == "stopped"


def test_continuous_mode_pauses_on_repeated_failure_signature(
    tmp_path: Path,
) -> None:
    options = replace(
        _options(tmp_path, max_cycles=1),
        continuous=True,
        max_consecutive_failures=3,
    )
    failed = {
        "run_id": "same-failure",
        "stages_executed": 1,
        "stages_done": 0,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 1,
        "final_status": "failed",
    }
    supervisor = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([failed] * 3, [1] * 3, []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=lambda **_kwargs: {
            "summary": "repair",
            "repair_prompt_patch": "Recreate the missing runtime dependency.",
            "stop_recommended": False,
        },
        aevolve_fn=lambda **_kwargs: [],
    )

    assert supervisor.run() == 1
    store = CampaignStore(options.campaign_dir)
    state = store.load_state()
    assert state["status"] == "paused_failure_threshold"
    assert state["consecutive_same_failure"] == 3
    assert state["failure_recovery_action"] == "quarantine"
    events = store.log.read_all()
    recovery = [
        event
        for event in events
        if event["type"] == "failure_repair_evaluated"
    ]
    assert [event["recovery_action"] for event in recovery] == [
        "auto_repair",
        "regenerate",
        "quarantine",
    ]
    threshold = [
        event for event in events if event["type"] == "failure_threshold_reached"
    ][-1]
    assert threshold["threshold_kind"] == "same_failure_signature"


def test_failure_signature_normalizes_paths_and_numbers() -> None:
    first, _ = CampaignSupervisor._failure_signature(
        {
            "pipeline_summary": {"final_status": "failed"},
            "failures": [
                {
                    "stage": 10,
                    "stage_name": "CODE_GENERATION",
                    "status": "failed",
                    "error": "Missing /tmp/run-123/data at attempt 2",
                }
            ],
        },
        returncode=1,
        topic_id="idea-a",
    )
    second, _ = CampaignSupervisor._failure_signature(
        {
            "pipeline_summary": {"final_status": "failed"},
            "failures": [
                {
                    "stage": 10,
                    "stage_name": "CODE_GENERATION",
                    "status": "failed",
                    "error": "Missing /tmp/run-999/data at attempt 8",
                }
            ],
        },
        returncode=1,
        topic_id="idea-a",
    )
    different_idea, _ = CampaignSupervisor._failure_signature(
        {
            "pipeline_summary": {"final_status": "failed"},
            "failures": [
                {
                    "stage": 10,
                    "stage_name": "CODE_GENERATION",
                    "status": "failed",
                    "error": "Missing /tmp/run-999/data at attempt 8",
                }
            ],
        },
        returncode=1,
        topic_id="idea-b",
    )

    assert first == second
    assert first != different_idea


def test_load_run_policy_prefers_durable_policy_file(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "campaign")
    store.initialize()
    store.policy_path.write_text(
        json.dumps(
            {
                "continuous": True,
                "max_cycles": 0,
                "llm_timeout_sec": 2222,
                "pipeline_extra_args": ["--from-stage", "9"],
            }
        ),
        encoding="utf-8",
    )

    policy = _load_run_policy(
        store,
        {
            "model": "codebuddy/deepseek-v4-pro-ioa",
            "bridge_url": "http://127.0.0.1:8787/v1",
            "run_policy": {"max_cycles": 7},
        },
    )

    assert policy["continuous"] is True
    assert policy["max_cycles"] == 0
    assert policy["llm_timeout_sec"] == 2222
    assert policy["pipeline_extra_args"] == ["--from-stage", "9"]


def test_restart_removes_stale_atomic_temp_files(tmp_path: Path) -> None:
    options = replace(
        _options(tmp_path, single_cycle=True),
        dry_run=True,
        resume_existing=True,
    )
    first = CampaignSupervisor(options)
    first.initialize()
    stale = (
        first.store.runs_dir
        / "cycle-0001"
        / ".config.provenance.json.killed.tmp"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("partial", encoding="utf-8")

    restarted = CampaignSupervisor(options)
    assert restarted.run() == 0

    assert not stale.exists()
    events = restarted.store.log.read_all()
    cleanup = next(
        event
        for event in events
        if event["type"] == "stale_atomic_temps_removed"
    )
    assert cleanup["paths"] == [
        "runs/cycle-0001/.config.provenance.json.killed.tmp"
    ]


def test_diagnosis_is_interruptible_by_campaign_control(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path, single_cycle=True)
    summary = {
        "run_id": "rc-ok",
        "stages_executed": 1,
        "stages_done": 1,
        "stages_paused": 0,
        "stages_blocked": 0,
        "stages_failed": 0,
        "final_status": "done",
    }
    entered = threading.Event()

    def diagnosis(**kwargs):
        entered.set()
        kwargs["cancel_event"].wait(timeout=2)
        raise InterruptedError("cancelled")

    supervisor = CampaignSupervisor(
        options,
        popen_factory=_pipeline_factory([summary], [0], []),
        sleep=lambda _seconds: None,
        llm_factory=lambda _path: object(),
        diagnosis_fn=diagnosis,
        aevolve_fn=lambda **_kwargs: [],
    )

    def request_pause() -> None:
        assert entered.wait(timeout=2)
        supervisor.store.set_control("pause", "test")
        supervisor._cancel_event.set()

    controller = threading.Thread(target=request_pause)
    controller.start()
    assert supervisor.run() == 0
    controller.join(timeout=2)
    assert CampaignStore(options.campaign_dir).load_state()["status"] == "paused"


def test_terminate_child_cancels_recorded_pool_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    run_dir = supervisor.store.runs_dir / "cycle-0001"
    metadata = run_dir / "work" / ".clusterbridge_pool_task.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "task_id": "rc-pool-cancel-test",
                "state": "starting",
            }
        ),
        encoding="utf-8",
    )
    supervisor.state.update(
        {"active_run_dir": str(run_dir), "active_cycle": 1}
    )
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append([str(part) for part in argv])
        return type(
            "Result",
            (),
            {"returncode": 0, "stdout": "{}", "stderr": ""},
        )()

    monkeypatch.setattr(subprocess, "run", fake_run)
    supervisor._cancel_remote_pool_tasks("pause")

    assert calls
    assert calls[0][-2:] == ["cancel-task", "rc-pool-cancel-test"]
    events = supervisor.store.log.read_all()
    assert events[-1]["type"] == "remote_pool_task_cancelled"


def test_reconcile_interrupted_execution_terminates_matching_local_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    run_dir = supervisor.store.runs_dir / "cycle-0001"
    run_dir.mkdir(parents=True)
    supervisor.state.update(
        {
            "status": "running",
            "phase": "pipeline",
            "pid": 41001,
            "supervisor_start_ticks": 10,
            "active_cycle": 1,
            "active_run_dir": str(run_dir),
            "active_child_pid": 42001,
            "active_child_start_ticks": 20,
        }
    )
    supervisor.state = supervisor.store.save_state(supervisor.state)
    ticks = {41001: None, 42001: 20}
    monkeypatch.setattr(
        "researchclaw.rsi.supervisor._process_start_ticks",
        lambda pid: ticks.get(pid),
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        supervisor,
        "_terminate_orphan_process_group",
        lambda pid: terminated.append(pid) or "terminated",
    )
    cancelled: list[tuple[str, Path | None]] = []
    monkeypatch.setattr(
        supervisor,
        "_cancel_remote_pool_tasks",
        lambda reason, *, run_dir=None: cancelled.append((reason, run_dir)),
    )

    supervisor._reconcile_interrupted_execution()

    assert terminated == [42001]
    assert cancelled == [("supervisor_restart_reconciliation", run_dir)]
    assert supervisor.state["status"] == "interrupted"
    assert supervisor.state["active_child_pid"] is None
    assert supervisor.state["next_cycle"] == 1
    assert supervisor.store.log.read_all()[-1]["type"] == (
        "interrupted_execution_reconciled"
    )


def test_reconcile_interrupted_execution_does_not_kill_reused_pid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = CampaignSupervisor(_options(tmp_path))
    supervisor.initialize()
    supervisor.state.update(
        {
            "status": "running",
            "pid": 41002,
            "supervisor_start_ticks": 10,
            "active_cycle": 1,
            "active_run_dir": None,
            "active_child_pid": 42002,
            "active_child_start_ticks": 20,
        }
    )
    supervisor.state = supervisor.store.save_state(supervisor.state)
    ticks = {41002: None, 42002: 999}
    monkeypatch.setattr(
        "researchclaw.rsi.supervisor._process_start_ticks",
        lambda pid: ticks.get(pid),
    )

    def forbidden(_pid: int) -> str:
        pytest.fail("a PID with mismatched start ticks must not be signalled")

    monkeypatch.setattr(supervisor, "_terminate_orphan_process_group", forbidden)

    supervisor._reconcile_interrupted_execution()

    event = supervisor.store.log.read_all()[-1]
    assert event["local_outcome"] == "already_exited_or_reused"


def test_pipeline_persists_child_identity_for_crash_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    supervisor._transition(
        status="running",
        phase="pipeline",
        active_cycle=1,
        active_run_dir=str(supervisor.store.runs_dir / "cycle-0001"),
    )
    monkeypatch.setattr(
        "researchclaw.rsi.supervisor._process_start_ticks",
        lambda pid: 777 if pid == 42077 else None,
    )

    class RecordedProcess(FakeProcess):
        def __init__(self) -> None:
            self.pid = 42077
            self.returncode = 0

    supervisor._popen_factory = lambda *_args, **_kwargs: RecordedProcess()
    monkeypatch.setattr(
        "researchclaw.rsi.supervisor.ensure_runtime_dependencies",
        lambda **_kwargs: [],
    )
    monkeypatch.setattr(
        "researchclaw.rsi.supervisor.dependencies_satisfied",
        lambda _results: True,
    )
    run_dir = supervisor.store.runs_dir / "cycle-0001"
    run_dir.mkdir(parents=True)

    assert supervisor._run_pipeline(1, run_dir, ["true"]) == 0
    persisted = supervisor.store.load_state()
    assert persisted["active_child_pid"] == 42077
    assert persisted["active_child_start_ticks"] == 777


def test_pipeline_checks_runtime_dependencies_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = _options(tmp_path)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    supervisor._transition(
        status="running",
        phase="pipeline",
        active_cycle=1,
        active_run_dir=str(supervisor.store.runs_dir / "cycle-0001"),
    )
    checked: list[str] = []

    class Result:
        module = "arxiv"
        requirement = "arxiv>=2.1,<5"
        status = "present"
        detail = "version 4.0.1"

        def to_dict(self):
            return {
                "module": self.module,
                "requirement": self.requirement,
                "status": self.status,
                "detail": self.detail,
            }

    def fake_ensure(**kwargs):
        checked.append(str(kwargs["python_executable"]))
        return [Result()]

    monkeypatch.setattr(
        "researchclaw.rsi.supervisor.ensure_runtime_dependencies",
        fake_ensure,
    )
    run_dir = supervisor.store.runs_dir / "cycle-0001"
    run_dir.mkdir(parents=True)
    supervisor._popen_factory = lambda *_args, **_kwargs: FakeProcess(0)

    assert supervisor._run_pipeline(1, run_dir, ["/chosen/python", "run"]) == 0
    assert checked == ["/chosen/python"]
    assert "runtime dependency arxiv: present" in (
        run_dir / "pipeline.log"
    ).read_text(encoding="utf-8")


def test_recorded_supervisor_liveness_rejects_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "researchclaw.rsi.cli._process_start_ticks",
        lambda _pid: 999,
    )

    assert not _recorded_process_is_alive(
        {"pid": 12345, "supervisor_start_ticks": 111}
    )
    assert _recorded_process_is_alive(
        {"pid": 12345, "supervisor_start_ticks": 999}
    )
    assert not _recorded_process_is_alive({"pid": 12345})


def test_existing_pause_request_prevents_pipeline_start(tmp_path: Path) -> None:
    options = _options(tmp_path, single_cycle=True)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    CampaignStore(options.campaign_dir).set_control("pause", "review")

    def forbidden(*_args, **_kwargs):
        pytest.fail("pipeline should not start while paused")

    resumed = CampaignSupervisor(options, popen_factory=forbidden)
    assert resumed.run() == 0
    assert CampaignStore(options.campaign_dir).load_state()["status"] == "paused"


def test_interrupted_cycle_reuses_checkpoint_with_pipeline_resume(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path, single_cycle=True)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    run_dir = supervisor.store.runs_dir / "cycle-0001"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.json").write_text("{}", encoding="utf-8")
    config_path = _base_config(tmp_path)

    command = supervisor._pipeline_command(run_dir, config_path)

    assert "--resume" in command


def test_next_failed_cycle_copies_completed_prefix_and_resumes(
    tmp_path: Path,
) -> None:
    options = _options(tmp_path, single_cycle=True)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    previous = supervisor.store.runs_dir / "cycle-0001"
    (previous / "stage-01").mkdir(parents=True)
    (previous / "stage-08").mkdir(parents=True)
    (previous / "stage-09").mkdir(parents=True)
    (previous / "stage-01" / "goal.md").write_text("goal")
    (previous / "stage-08" / "hypotheses.md").write_text("hypothesis")
    (previous / "stage-09" / "partial.txt").write_text("must not be copied")
    (previous / "checkpoint.json").write_text(
        json.dumps(
            {
                "last_completed_stage": 8,
                "last_completed_name": "HYPOTHESIS_GEN",
                "run_id": "old",
            }
        )
    )
    (previous / "pipeline_summary.json").write_text(
        json.dumps({"final_status": "paused", "final_stage": 9})
    )
    supervisor.state.update(
        {
            "last_run_dir": str(previous),
            "pending_topic_action": {"topic_action": "keep"},
        }
    )
    supervisor.state = supervisor.store.save_state(supervisor.state)
    target = supervisor.store.runs_dir / "cycle-0002"
    target.mkdir()

    source_cycle = supervisor._seed_failed_cycle_resume(target, 2)

    assert source_cycle == 1
    assert (target / "stage-01" / "goal.md").read_text() == "goal"
    assert (target / "stage-08" / "hypotheses.md").read_text() == "hypothesis"
    assert not (target / "stage-09").exists()
    assert supervisor._checkpoint_next_stage_name(target) == "EXPERIMENT_DESIGN"
    manifest = json.loads((target / "resume_manifest.json").read_text())
    assert manifest["last_completed_stage"] == 8
    command = supervisor._pipeline_command(target, _base_config(tmp_path))
    assert "--resume" in command


def test_topic_pivot_does_not_reuse_failed_prefix(tmp_path: Path) -> None:
    supervisor = CampaignSupervisor(_options(tmp_path, single_cycle=True))
    supervisor.initialize()
    previous = supervisor.store.runs_dir / "cycle-0001"
    previous.mkdir(parents=True)
    (previous / "checkpoint.json").write_text(
        json.dumps({"last_completed_stage": 8})
    )
    (previous / "pipeline_summary.json").write_text(
        json.dumps({"final_status": "failed"})
    )
    supervisor.state.update(
        {
            "last_run_dir": str(previous),
            "pending_topic_action": {
                "topic_action": "pivot",
                "pivot_reason": "pilot falsified the incumbent",
            },
        }
    )
    supervisor.state = supervisor.store.save_state(supervisor.state)
    target = supervisor.store.runs_dir / "cycle-0002"
    target.mkdir()

    assert supervisor._seed_failed_cycle_resume(target, 2) is None
    assert not (target / "checkpoint.json").exists()


def test_dry_run_single_cycle_avoids_llm(tmp_path: Path) -> None:
    options = replace(_options(tmp_path, single_cycle=True), dry_run=True)

    def forbidden_llm(_path: Path):
        pytest.fail("dry-run must not construct an LLM client")

    supervisor = CampaignSupervisor(options, llm_factory=forbidden_llm)
    assert supervisor.run() == 0
    state = CampaignStore(options.campaign_dir).load_state()
    assert state["completed_cycles"] == 1
    assert state["successful_cycles"] == 1
    assert state["status"] == "paused_single_cycle"


def test_control_files_are_reported_in_status(tmp_path: Path) -> None:
    options = _options(tmp_path)
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    store = CampaignStore(options.campaign_dir)
    store.set_control("stop", "done")

    from researchclaw.rsi.supervisor import campaign_status

    status = campaign_status(options.campaign_dir)
    assert status["stop_requested"] is True
    assert status["pause_requested"] is False
    assert status["automatic_submission_enabled"] is False


def test_foreground_supervisor_sets_bridge_placeholder_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = replace(
        _options(tmp_path, single_cycle=True),
        dry_run=True,
        api_key_env="RSI_TEST_BRIDGE_KEY",
    )
    monkeypatch.delenv("RSI_TEST_BRIDGE_KEY", raising=False)

    assert CampaignSupervisor(options).run() == 0

    assert os.environ["RSI_TEST_BRIDGE_KEY"] == "local-bridge"


def test_foreground_supervisor_replaces_empty_bridge_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = replace(
        _options(tmp_path, single_cycle=True),
        dry_run=True,
        api_key_env="RSI_TEST_EMPTY_BRIDGE_KEY",
    )
    monkeypatch.setenv("RSI_TEST_EMPTY_BRIDGE_KEY", "")

    assert CampaignSupervisor(options).run() == 0

    assert os.environ["RSI_TEST_EMPTY_BRIDGE_KEY"] == "local-bridge"


def test_campaign_lock_rejects_second_supervisor(tmp_path: Path) -> None:
    options = replace(_options(tmp_path, single_cycle=True), dry_run=True)
    first = CampaignSupervisor(options)
    first.store.root.mkdir(parents=True, exist_ok=True)
    first._acquire_campaign_lock()
    try:
        second = CampaignSupervisor(options)
        with pytest.raises(RuntimeError, match="lock is already held"):
            second.run()
    finally:
        first._release_campaign_lock()


def test_resume_existing_clears_pause_only_after_lock_acquired(
    tmp_path: Path,
) -> None:
    options = replace(
        _options(tmp_path, single_cycle=True),
        dry_run=True,
        resume_existing=True,
    )
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    store = CampaignStore(options.campaign_dir)
    store.set_control("pause", "resume me")

    lock_holder = CampaignSupervisor(options)
    lock_holder._acquire_campaign_lock()
    try:
        contender = CampaignSupervisor(options)
        with pytest.raises(RuntimeError, match="lock is already held"):
            contender.run()
        assert store.control_requested("pause") is True
    finally:
        lock_holder._release_campaign_lock()

    assert CampaignSupervisor(options).run() == 0
    assert store.control_requested("pause") is False
