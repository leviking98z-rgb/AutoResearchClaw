from __future__ import annotations

import importlib.util
import fcntl
import json
import shutil
import sys
from enum import IntEnum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "safe_cycle_rollback.py"


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("safe_cycle_rollback", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rollback = _load_helper()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_quiescent_cycle(tmp_path: Path) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    run = campaign / "runs" / "cycle-0002"
    run.mkdir(parents=True)
    (campaign / "control").mkdir()
    _write_json(
        campaign / "state.json",
        {
            "status": "paused",
            "phase": "idle",
            "active_cycle": 2,
            "next_cycle": 2,
            "completed_cycles": 1,
            "active_run_dir": str(run),
            "pid": None,
            "supervisor_start_ticks": None,
            "active_child_pid": None,
            "active_child_start_ticks": None,
        },
    )
    _write_json(campaign / "control" / "pause", {"reason": "test"})
    (campaign / "supervisor.lock").touch()

    required = rollback._REQUIRED_PRIOR_OUTPUTS
    for stage in range(1, 9):
        stage_dir = run / f"stage-{stage:02d}"
        stage_dir.mkdir()
        for relative in required[stage]:
            target = stage_dir / relative.rstrip("/")
            if relative.endswith("/"):
                target.mkdir()
                (target / "card.md").write_text("content", encoding="utf-8")
            else:
                target.write_text("content", encoding="utf-8")
        _write_json(
            stage_dir / "decision.json",
            {
                "stage_id": f"{stage:02d}-test",
                "run_id": "rc-original",
                "status": "done",
            },
        )

    for stage in range(9, 15):
        stage_dir = run / f"stage-{stage:02d}"
        stage_dir.mkdir()
        (stage_dir / "artifact.txt").write_text("invalid", encoding="utf-8")
    versioned = run / "stage-14_repair_v1"
    versioned.mkdir()
    _write_json(
        versioned / ".clusterbridge_pool_task.json",
        {"task_id": "rc-pool-test", "state": "finished", "returncode": 1},
    )
    _write_json(
        run / "checkpoint.json",
        {
            "last_completed_stage": 15,
            "last_completed_name": "RESEARCH_DECISION",
            "run_id": "rc-original",
        },
    )
    (run / "experiment_diagnosis.json").write_text("{}", encoding="utf-8")
    hitl = run / "hitl"
    hitl.mkdir()
    (hitl / "baseline_navigator.json").write_text(
        '{"baselines": ["invalid"]}',
        encoding="utf-8",
    )
    (run / "pipeline.log").write_text("preserve me", encoding="utf-8")
    (run / "selected_topic.json").write_text("{}", encoding="utf-8")
    return campaign, run


def test_plan_fails_closed_when_pipeline_process_is_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    state = json.loads((campaign / "state.json").read_text(encoding="utf-8"))
    state["status"] = "running"
    state["phase"] = "pipeline"
    state["active_child_pid"] = 4242
    state["active_child_start_ticks"] = 777
    _write_json(campaign / "state.json", state)
    monkeypatch.setattr(
        rollback,
        "_process_start_ticks",
        lambda pid: 777 if pid == 4242 else None,
    )

    plan = rollback.build_plan(
        campaign_dir=campaign,
        run_dir=run,
        from_stage=9,
    )

    assert plan.safe_to_apply is False
    assert any("pipeline_child pid 4242 is alive" in item for item in plan.blockers)


def test_campaign_guard_lock_refuses_when_supervisor_lock_is_held(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    lock_path = campaign / "supervisor.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(rollback.SafetyError, match="supervisor.lock is held"):
            with rollback._campaign_guard_lock(campaign):
                pytest.fail("guard lock unexpectedly acquired")


def test_plan_fails_closed_for_nonterminal_pool_task(tmp_path: Path) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    metadata = next(run.rglob(".clusterbridge_pool_task.json"))
    _write_json(
        metadata,
        {"task_id": "rc-pool-test", "state": "starting"},
    )

    plan = rollback.build_plan(
        campaign_dir=campaign,
        run_dir=run,
        from_stage=9,
    )

    assert plan.safe_to_apply is False
    assert any("rc-pool-test: starting" in item for item in plan.blockers)


def test_plan_fails_closed_if_campaign_already_finalized_cycle2(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    state = json.loads((campaign / "state.json").read_text(encoding="utf-8"))
    state.update(
        {
            "active_cycle": None,
            "active_run_dir": None,
            "next_cycle": 3,
            "completed_cycles": 2,
        }
    )
    _write_json(campaign / "state.json", state)

    plan = rollback.build_plan(
        campaign_dir=campaign,
        run_dir=run,
        from_stage=9,
    )

    assert plan.campaign_identity_preserved is False
    assert any(
        "no longer identifies this run as the unfinished active cycle" in blocker
        for blocker in plan.blockers
    )


def test_plan_restores_retained_stage_lineage_over_resumed_cli_run_id(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    _write_json(
        run / "checkpoint.json",
        {
            "last_completed_stage": 15,
            "last_completed_name": "RESEARCH_DECISION",
            "run_id": "rc-resumed-cli",
        },
    )

    plan = rollback.build_plan(
        campaign_dir=campaign,
        run_dir=run,
        from_stage=9,
    )

    assert plan.checkpoint_run_id == "rc-resumed-cli"
    assert plan.run_id == "rc-original"
    assert plan.safe_to_apply is True
    assert any("checkpoint run_id differs" in warning for warning in plan.warnings)


def test_plan_accepts_paused_unfinished_cycle_with_cleared_active_identity(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    state = json.loads((campaign / "state.json").read_text(encoding="utf-8"))
    state.update(
        {
            "active_cycle": None,
            "active_run_dir": None,
            "next_cycle": 2,
            "completed_cycles": 1,
        }
    )
    _write_json(campaign / "state.json", state)

    plan = rollback.build_plan(
        campaign_dir=campaign,
        run_dir=run,
        from_stage=9,
    )

    assert plan.campaign_identity_preserved is True
    assert plan.safe_to_apply is True


def test_run_phase_refuses_when_checkpoint_does_not_match_boundary(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)

    with pytest.raises(rollback.SafetyError, match="checkpoint stage is 15"):
        rollback.run_pipeline_phase(
            campaign_dir=campaign,
            run_dir=run,
            from_stage=9,
            to_stage=9,
            skip_preflight=True,
        )


def test_run_phase_refuses_prior_stage_run_id_mismatch(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    for child in list(run.iterdir()):
        if child.is_dir() and child.name.startswith("stage-"):
            match = rollback._STAGE_DIR_RE.fullmatch(child.name)
            if match is not None and int(match.group(1)) >= 9:
                shutil.rmtree(child)
    for name in rollback._ROOT_ARTIFACTS:
        (run / name).unlink(missing_ok=True)
    _write_json(
        run / "checkpoint.json",
        {
            "last_completed_stage": 8,
            "last_completed_name": "HYPOTHESIS_GEN",
            "run_id": "rc-original",
        },
    )
    decision = json.loads(
        (run / "stage-08" / "decision.json").read_text(encoding="utf-8")
    )
    decision["run_id"] = "rc-other"
    _write_json(run / "stage-08" / "decision.json", decision)

    with pytest.raises(rollback.SafetyError, match="inconsistent run_ids"):
        rollback.run_pipeline_phase(
            campaign_dir=campaign,
            run_dir=run,
            from_stage=9,
            to_stage=9,
            skip_preflight=True,
        )


def test_run_phase_initial_boundary_reaches_runtime_without_stage9_artifacts(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    for child in list(run.iterdir()):
        if child.is_dir() and child.name.startswith("stage-"):
            match = rollback._STAGE_DIR_RE.fullmatch(child.name)
            if match is not None and int(match.group(1)) >= 9:
                shutil.rmtree(child)
    for name in rollback._ROOT_ARTIFACTS:
        (run / name).unlink(missing_ok=True)
    _write_json(
        run / "checkpoint.json",
        {
            "last_completed_stage": 8,
            "last_completed_name": "HYPOTHESIS_GEN",
            "run_id": "rc-original",
        },
    )
    with pytest.raises(rollback.SafetyError, match="cannot import|config.yaml"):
        rollback.run_pipeline_phase(
            campaign_dir=campaign,
            run_dir=run,
            from_stage=9,
            to_stage=9,
            skip_preflight=True,
        )


def test_run_phase_followup_reaches_runtime_after_quiescence_checks(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    for child in list(run.iterdir()):
        if child.name.startswith("stage-09") or child.name.startswith("stage-1"):
            if child.is_dir():
                shutil.rmtree(child)
    _write_json(
        run / "checkpoint.json",
        {
            "last_completed_stage": 9,
            "last_completed_name": "EXPERIMENT_DESIGN",
            "run_id": "rc-original",
        },
    )
    stage9 = run / "stage-09"
    stage9.mkdir()
    _write_json(
        stage9 / "decision.json",
        {"stage_id": "09-test", "run_id": "rc-original", "status": "done"},
    )

    with pytest.raises(rollback.SafetyError, match="cannot import|config.yaml"):
        rollback.run_pipeline_phase(
            campaign_dir=campaign,
            run_dir=run,
            from_stage=10,
            to_stage=11,
            skip_preflight=True,
            allow_followup=True,
        )


def test_run_phase_does_not_report_success_when_intermediate_stage_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    for child in list(run.iterdir()):
        if child.is_dir() and child.name.startswith("stage-"):
            match = rollback._STAGE_DIR_RE.fullmatch(child.name)
            if match is not None and int(match.group(1)) >= 9:
                shutil.rmtree(child)
    for name in rollback._ROOT_ARTIFACTS:
        (run / name).unlink(missing_ok=True)
    _write_json(
        run / "checkpoint.json",
        {
            "last_completed_stage": 9,
            "last_completed_name": "EXPERIMENT_DESIGN",
            "run_id": "rc-original",
        },
    )
    stage9 = run / "stage-09"
    stage9.mkdir()
    _write_json(
        stage9 / "decision.json",
        {"stage_id": "09-test", "run_id": "rc-original", "status": "done"},
    )
    (run / "config.yaml").write_text("project: {}\n", encoding="utf-8")

    class FakeStatus:
        DONE = "done"
        FAILED = "failed"
        PAUSED = "paused"
        BLOCKED_APPROVAL = "blocked"

    class FakeStage(IntEnum):
        EXPERIMENT_DESIGN = 9
        CODE_GENERATION = 10
        RESOURCE_PLANNING = 11

    fake_config = SimpleNamespace(
        llm=SimpleNamespace(api_key_env=""),
        knowledge_base=SimpleNamespace(root=""),
    )
    fake_modules = {
        "researchclaw.adapters": SimpleNamespace(AdapterBundle=lambda: object()),
        "researchclaw.config": SimpleNamespace(
            RCConfig=SimpleNamespace(load=lambda *_args, **_kwargs: fake_config)
        ),
        "researchclaw.llm": SimpleNamespace(create_llm_client=lambda _cfg: None),
        "researchclaw.pipeline.runner": SimpleNamespace(
            execute_pipeline=lambda **_kwargs: [
                SimpleNamespace(stage=10, status=FakeStatus.DONE)
            ]
        ),
        "researchclaw.pipeline.stages": SimpleNamespace(
            Stage=FakeStage,
            StageStatus=FakeStatus,
        ),
    }
    for name, module in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    result = rollback.run_pipeline_phase(
        campaign_dir=campaign,
        run_dir=run,
        from_stage=10,
        to_stage=11,
        skip_preflight=True,
        allow_followup=True,
    )

    assert result["final_stage"] == 10
    assert result["expected_stages"] == 2
    assert result["stages_executed"] == 1
    assert result["failed"] == 0
    assert result["success"] is False


def test_archive_phase_summary_moves_summary_outside_canonical_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "cycle-0002"
    run.mkdir()
    summary = run / "pipeline_summary.json"
    summary.write_text('{"final_stage": 9}', encoding="utf-8")
    monkeypatch.setattr(rollback, "_timestamp_slug", lambda: "20260804T120000Z")

    destination = rollback._archive_phase_summary(run, 10, 11)

    assert destination == (
        run
        / "rollback_backups"
        / "phase_summaries"
        / "before-stage-10-to-11-20260804T120000Z.json"
    )
    assert not summary.exists()
    assert destination.read_text(encoding="utf-8") == '{"final_stage": 9}'
    assert not list(run.glob("stage-*"))


def test_rewrite_final_pipeline_summary_marks_recovered_cycle_complete(
    tmp_path: Path,
) -> None:
    run = tmp_path / "cycle-0002"
    run.mkdir()
    _write_json(
        run / "pipeline_summary.json",
        {
            "run_id": "rc-original",
            "stages_executed": 8,
            "stages_done": 8,
            "stages_paused": 0,
            "stages_blocked": 0,
            "stages_failed": 0,
            "from_stage": 16,
            "final_stage": 23,
            "final_status": "done",
        },
    )

    rollback._rewrite_final_pipeline_summary(
        run,
        run_id="rc-original",
        from_stage=16,
        to_stage=23,
    )

    summary = json.loads(
        (run / "pipeline_summary.json").read_text(encoding="utf-8")
    )
    assert summary["from_stage"] == 1
    assert summary["stages_executed"] == 23
    assert summary["stages_done"] == 23
    assert summary["recovered_from_stage"] == 16
    assert summary["bounded_final_phase_stages"] == 8
    assert summary["bounded_final_phase_done"] == 8
    assert summary["summary_scope"] == "cycle_recovery_full_pipeline"


def test_rewrite_final_pipeline_summary_refuses_incomplete_phase(
    tmp_path: Path,
) -> None:
    run = tmp_path / "cycle-0002"
    run.mkdir()
    _write_json(
        run / "pipeline_summary.json",
        {
            "run_id": "rc-original",
            "stages_executed": 7,
            "stages_done": 7,
            "stages_paused": 0,
            "stages_blocked": 0,
            "stages_failed": 0,
            "from_stage": 16,
            "final_stage": 23,
            "final_status": "done",
        },
    )

    with pytest.raises(rollback.SafetyError, match="incomplete final"):
        rollback._rewrite_final_pipeline_summary(
            run,
            run_id="rc-original",
            from_stage=16,
            to_stage=23,
        )


def test_apply_archives_stage_artifacts_outside_stage_globs_and_rewinds_checkpoint(
    tmp_path: Path,
) -> None:
    campaign, run = _build_quiescent_cycle(tmp_path)
    backup = run / "rollback_backups" / "from-stage-09-test"
    plan = rollback.build_plan(
        campaign_dir=campaign,
        run_dir=run,
        from_stage=9,
        backup_dir=backup,
    )

    assert plan.safe_to_apply is True
    result = rollback.apply_plan(plan)

    assert result == backup
    assert not (run / "stage-09").exists()
    assert not (run / "stage-14_repair_v1").exists()
    assert (backup / "stage-09" / "artifact.txt").read_text() == "invalid"
    assert (backup / "stage-14_repair_v1").is_dir()
    assert (backup / "hitl" / "baseline_navigator.json").is_file()
    assert not (run / "hitl").exists()
    assert not list(run.glob("stage-09*"))
    assert not list(run.glob("stage-14*"))

    checkpoint = json.loads((run / "checkpoint.json").read_text(encoding="utf-8"))
    assert checkpoint["last_completed_stage"] == 8
    assert checkpoint["last_completed_name"] == "HYPOTHESIS_GEN"
    assert checkpoint["run_id"] == "rc-original"
    assert checkpoint["rollback_from_stage"] == 9
    assert checkpoint["rollback_backup"] == (
        "rollback_backups/from-stage-09-test"
    )

    manifest = json.loads(
        (backup / "rollback_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert {Path(item["source"]).name for item in manifest["moves"]} >= {
        "stage-09",
        "stage-14_repair_v1",
        "checkpoint.json",
        "experiment_diagnosis.json",
    }
    assert (run / "stage-08" / "hypotheses.md").is_file()
    assert (run / "pipeline.log").read_text(encoding="utf-8") == "preserve me"
    assert (run / "selected_topic.json").is_file()
