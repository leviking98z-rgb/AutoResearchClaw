from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from researchclaw.rsi import cli
from researchclaw.rsi.storage import CampaignStore
from researchclaw.rsi.supervisor import CampaignSupervisor, SupervisorOptions


def _base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "rsi-cli-test", "mode": "docs-first"},
                "research": {"topic": "placeholder"},
                "knowledge_base": {"backend": "markdown", "root": "kb"},
                "llm": {"provider": "openai-compatible"},
                "security": {},
                "experiment": {"cli_agent": {"provider": "llm"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _initialize_campaign(tmp_path: Path) -> tuple[Path, Path, CampaignStore]:
    campaign_dir = tmp_path / "campaigns" / "durable-policy"
    base_config = _base_config(tmp_path)
    options = SupervisorOptions(
        campaign_dir=campaign_dir,
        repo_root=tmp_path,
        base_config=base_config,
        topic="Does method X improve metric Y over baseline Z?",
        continuous=True,
        max_cycles=0,
        dry_run=True,
        max_consecutive_failures=9,
        max_no_improvement_cycles=11,
        backoff_initial_sec=4.5,
        backoff_max_sec=88.0,
        heartbeat_interval_sec=17.0,
        control_poll_sec=0.25,
        api_key_env="RSI_POLICY_KEY",
        llm_timeout_sec=2222,
        skip_preflight=True,
        no_aevolve=True,
        pipeline_extra_args=("--from-stage", "9"),
    )
    supervisor = CampaignSupervisor(options)
    supervisor.initialize()
    store = CampaignStore(campaign_dir)
    state = store.load_state()
    state.update({"status": "paused", "pid": None})
    store.save_state(state)
    store.set_control("pause", "test pause")
    return campaign_dir, base_config, store


def test_resume_reconstructs_exact_persisted_submit_argv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign_dir, base_config, _store = _initialize_campaign(tmp_path)
    captured: list[list[str]] = []

    def fake_submit(argv):
        captured.append(list(argv))
        return 0

    monkeypatch.setattr(cli, "submit_main", fake_submit)

    assert cli.resume_main([str(campaign_dir)]) == 0
    assert captured == [
        [
            "Does method X improve metric Y over baseline Z?",
            "--campaign-id",
            "durable-policy",
            "--campaign-root",
            str(campaign_dir.parent),
            "--config",
            str(base_config),
            "--max-cycles",
            "0",
            "--max-consecutive-failures",
            "9",
            "--max-no-improvement-cycles",
            "11",
            "--backoff-initial-sec",
            "4.5",
            "--backoff-max-sec",
            "88.0",
            "--heartbeat-interval-sec",
            "17.0",
            "--control-poll-sec",
            "0.25",
            "--model",
            "codebuddy/claude-sonnet-5",
            "--bridge-url",
            "http://127.0.0.1:8787/v1",
            "--api-key-env",
            "RSI_POLICY_KEY",
            "--llm-timeout-sec",
            "2222",
            "--_resume-existing",
            "--continuous",
            "--dry-run",
            "--skip-preflight",
            "--no-aevolve",
            "--pipeline-arg=--from-stage",
            "--pipeline-arg=9",
        ]
    ]


def test_background_resume_marks_spawned_supervisor_as_resume_existing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign_dir, base_config, store = _initialize_campaign(tmp_path)
    spawned: list[list[str]] = []

    class FakeChild:
        pid = 54321

    def fake_popen(argv, **_kwargs):
        spawned.append([str(value) for value in argv])
        return FakeChild()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        cli,
        "_wait_for_supervisor_start",
        lambda **_kwargs: (True, "running"),
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="detached\n",
            stderr="",
        ),
    )

    assert (
        cli.submit_main(
            [
                "Does method X improve metric Y over baseline Z?",
                "--campaign-id",
                campaign_dir.name,
                "--campaign-root",
                str(campaign_dir.parent),
                "--config",
                str(base_config),
                "--continuous",
                "--dry-run",
                "--skip-preflight",
                "--no-aevolve",
                "--pipeline-arg=--from-stage",
                "--_resume-existing",
            ]
        )
        == 0
    )

    assert len(spawned) == 1
    assert "--_run-supervisor" in spawned[0]
    assert "--_startup-handshake" in spawned[0]
    assert "--_supervisor-log" in spawned[0]
    assert str(campaign_dir / "supervisor.log") in spawned[0]
    assert "--_resume-existing" in spawned[0]
    assert "--pipeline-arg=--from-stage" in spawned[0]
    assert store.control_requested("pause") is True


def test_startup_handshake_accepts_new_state_ownership_with_stale_heartbeat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign_dir, _base_config_path, store = _initialize_campaign(tmp_path)

    class FakeChild:
        pid = 54321

        def poll(self):
            state = store.load_state()
            state.update(
                {
                    "status": "running",
                    "pid": self.pid,
                    "supervisor_start_ticks": 777,
                }
            )
            store.save_state(state)
            store.write_heartbeat(
                {
                    "supervisor_pid": 11111,
                    "status": "paused",
                    "timestamp": "2020-01-01T00:00:00+00:00",
                }
            )

    monkeypatch.setattr(cli, "_process_start_ticks", lambda _pid: 777)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    started, detail = cli._wait_for_supervisor_start(
        campaign_dir=campaign_dir,
        child=FakeChild(),  # type: ignore[arg-type]
        timeout_sec=0.1,
    )

    assert started is True
    assert "state_pid=54321" in detail


def test_stop_requests_supervisor_and_campaign_monitor_shutdown(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    campaign_dir, _base_config_path, store = _initialize_campaign(tmp_path)
    state = store.load_state()
    state.update(
        {
            "status": "running",
            "pid": 44001,
            "supervisor_start_ticks": 101,
        }
    )
    store.save_state(state)
    store.clear_control("pause")
    store.root.joinpath("rsi-monitor.pid").write_text(
        json.dumps({"pid": 44002, "start_ticks": 202}),
        encoding="utf-8",
    )
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == 0:
            raise ProcessLookupError

    monkeypatch.setattr(cli.os, "kill", fake_kill)
    monkeypatch.setattr(
        cli,
        "_process_start_ticks",
        lambda pid: {44001: 101, 44002: 202}.get(pid),
    )
    monkeypatch.setattr(cli, "_monitor_process_start_ticks", lambda _pid: 202)

    assert (
        cli.stop_main(
            [str(campaign_dir), "--reason", "finished lifecycle test"]
        )
        == 0
    )

    assert signals == [
        (44001, cli.signal.SIGTERM),
        (44002, cli.signal.SIGTERM),
        (44002, 0),
    ]
    assert store.control_requested("stop") is True
    output = json.loads(capsys.readouterr().out)
    assert output["stop_requested"] is True
    assert output["monitor"] == {
        "requested": True,
        "pid": 44002,
        "reason": "exited",
    }
    assert store.log.read_all()[-1]["type"] == "monitor_stop_attempted"


def test_monitor_stop_escalates_after_grace_period(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    campaign.joinpath("rsi-monitor.pid").write_text(
        json.dumps({"pid": 55002, "start_ticks": 303}),
        encoding="utf-8",
    )
    signals: list[tuple[int, int]] = []
    probes = {"count": 0}

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if signum == 0:
            probes["count"] += 1
            if probes["count"] >= 2:
                raise ProcessLookupError

    monkeypatch.setattr(cli.os, "kill", fake_kill)
    monkeypatch.setattr(cli, "_monitor_process_start_ticks", lambda _pid: 303)

    result = cli._stop_monitor(
        campaign,
        wait_timeout_sec=0,
        kill_timeout_sec=1,
    )

    assert result == {"requested": True, "pid": 55002, "reason": "sigkill"}
    assert signals == [
        (55002, cli.signal.SIGTERM),
        (55002, cli.signal.SIGKILL),
        (55002, 0),
        (55002, 0),
    ]
    assert not campaign.joinpath("rsi-monitor.pid").exists()


def test_stop_refuses_reused_supervisor_and_monitor_pids(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    campaign_dir, _base_config_path, store = _initialize_campaign(tmp_path)
    state = store.load_state()
    state.update(
        {
            "status": "crashed",
            "pid": 66001,
            "supervisor_start_ticks": 111,
        }
    )
    store.save_state(state)
    store.root.joinpath("rsi-monitor.pid").write_text(
        json.dumps({"pid": 66002, "start_ticks": 222}),
        encoding="utf-8",
    )
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        cli,
        "_process_start_ticks",
        lambda pid: {66001: 999}.get(pid),
    )
    monkeypatch.setattr(
        cli,
        "_monitor_process_start_ticks",
        lambda _pid: 999,
    )
    monkeypatch.setattr(
        cli.os,
        "kill",
        lambda pid, signum: signals.append((pid, signum)),
    )

    assert cli.stop_main([str(campaign_dir)]) == 0

    assert signals == []
    output = json.loads(capsys.readouterr().out)
    assert output["monitor"] == {
        "requested": False,
        "pid": 66002,
        "reason": "identity_mismatch",
    }
    assert not store.root.joinpath("rsi-monitor.pid").exists()


def test_stop_signals_monitor_watchdog_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    campaign.joinpath("rsi-monitor.pid").write_text(
        json.dumps({"pid": 77001, "start_ticks": 444}),
        encoding="utf-8",
    )
    campaign.joinpath("rsi-monitor.pid.watchdog.pid").write_text(
        "77002\n",
        encoding="utf-8",
    )
    signals: list[tuple[int, int]] = []

    def fake_kill(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if pid == 77001 and signum == 0:
            raise ProcessLookupError

    monkeypatch.setattr(cli, "_monitor_process_start_ticks", lambda _pid: 444)
    monkeypatch.setattr(cli.os, "kill", fake_kill)

    result = cli._stop_monitor(campaign)

    assert result == {"requested": True, "pid": 77001, "reason": "exited"}
    assert signals == [
        (77001, cli.signal.SIGTERM),
        (77002, cli.signal.SIGTERM),
        (77001, 0),
    ]
    assert campaign.joinpath("rsi-monitor.pid.watchdog.stop").exists()
