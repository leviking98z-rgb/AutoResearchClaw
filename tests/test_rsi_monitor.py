"""Tests for the persistent RSI monitor and its operations scripts."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = ROOT / "researchclaw" / "rsi" / "monitor.py"


def _load_monitor() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "researchclaw_rsi_monitor_tests",
        MONITOR_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


monitor = _load_monitor()


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode()
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


class FakeRunner:
    def __init__(
        self,
        *,
        nodes: tuple[str, ...] = ("10.0.0.1",),
        restart_returncode: int = 0,
    ) -> None:
        self.nodes = nodes
        self.restart_returncode = restart_returncode
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        del kwargs
        argv = [str(part) for part in args]
        with self._lock:
            self.calls.append(argv)
        if argv[-1] == "list":
            rows = "\n".join(
                f"{node:<16} yes      8    test_cluster"
                f"                                 owner/research     "
                "2026-08-04T05:03:49Z(599m)"
                for node in self.nodes
            )
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="",
                stderr=(
                    "NODE             ALIVE    GPUs CLUSTER"
                    "                                      CLAIM"
                    "              EXPIRES\n"
                    f"{rows}\n"
                    f"-- alive {len(self.nodes)} / claimed {len(self.nodes)} --\n"
                ),
            )
        if argv[-1] == "claims":
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout="CLAIM details\n",
                stderr="",
            )
        if "nvidia-smi " in argv[-1]:
            rows = "\n".join(
                f"{index}, NVIDIA H20, 97871, 1024, 25" for index in range(8)
            )
            return subprocess.CompletedProcess(argv, 0, stdout=rows, stderr="")
        if argv and argv[0] == "restart-supervisor":
            return subprocess.CompletedProcess(
                argv,
                self.restart_returncode,
                stdout="pid=4321\n" if self.restart_returncode == 0 else "",
                stderr="boom" if self.restart_returncode else "",
            )
        raise AssertionError(f"unexpected command: {argv}")


def _write_campaign_files(
    campaign_dir: Path,
    *,
    now: datetime,
    heartbeat_pid: int = 1234,
    heartbeat_at: datetime | None = None,
    state_status: str = "running",
) -> None:
    campaign_dir.mkdir(parents=True, exist_ok=True)
    run_dir = campaign_dir / "runs" / "cycle-1"
    run_dir.mkdir(parents=True)
    heartbeat = {
        "pid": heartbeat_pid,
        "timestamp": monitor.utc_iso(heartbeat_at or now),
        "cycle": 1,
    }
    (campaign_dir / "supervisor_heartbeat.json").write_text(
        json.dumps(heartbeat),
        encoding="utf-8",
    )
    (campaign_dir / "campaign_state.json").write_text(
        json.dumps(
            {
                "status": state_status,
                "cycle": 1,
                "run_dir": str(run_dir),
                "updated_at": monitor.utc_iso(now),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "last_completed_stage": 12,
                "timestamp": monitor.utc_iso(now),
            }
        ),
        encoding="utf-8",
    )


def _make_monitor(
    campaign_dir: Path,
    *,
    now: datetime,
    runner: FakeRunner,
    pid_alive=lambda pid: True,
    restart_command: tuple[str, ...] = (),
    restart_policy: Any | None = None,
) -> Any:
    config = monitor.MonitorConfig(
        campaign_dir=campaign_dir,
        nodes=runner.nodes,
        expected_gpus_per_node=8,
        restart_command=restart_command,
        restart_policy=restart_policy or monitor.RestartPolicy(),
    )
    return monitor.CampaignMonitor(
        config,
        runner=runner,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        sleep=lambda seconds: None,
        pid_alive=pid_alive,
    )


def test_poll_once_collects_all_health_and_writes_atomic_snapshot(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    runner = FakeRunner(nodes=("10.0.0.1", "10.0.0.2"))
    watchdog = _make_monitor(tmp_path, now=now, runner=runner)

    snapshot = watchdog.poll_once()

    assert snapshot["overall"] == "degraded"
    assert snapshot["checks"]["supervisor"]["status"] == "ok"
    assert snapshot["checks"]["bridge"]["status"] == "ok"
    assert snapshot["checks"]["campaign"]["status"] == "ok"
    assert snapshot["checks"]["checkpoint"]["data"]["last_completed_stage"] == 12
    assert snapshot["checks"]["progress"]["status"] == "ok"
    assert snapshot["checks"]["cluster"]["status"] == "ok"
    assert snapshot["checks"]["gpu"]["data"]["total_gpus"] == 16
    assert snapshot["checks"]["pool"]["status"] == "degraded"
    persisted = json.loads((tmp_path / "monitor_snapshot.json").read_text())
    assert persisted["generated_at"] == monitor.utc_iso(now)
    assert not list(tmp_path.glob(".monitor_snapshot.json.*.tmp"))


def test_atomic_write_json_retries_transient_missing_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "monitor_heartbeat.json"
    real_replace = monitor.os.replace
    calls = 0

    def flaky_replace(src, dst):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FileNotFoundError(src)
        return real_replace(src, dst)

    monkeypatch.setattr(monitor.os, "replace", flaky_replace)
    monitor.atomic_write_json(target, {"status": "ok"})

    assert calls == 2
    assert json.loads(target.read_text()) == {"status": "ok"}
    assert not list(tmp_path.glob(".monitor_heartbeat.json.*.tmp"))


def test_stale_heartbeat_and_dead_pid_trigger_restart(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    stale = now - timedelta(hours=1)
    _write_campaign_files(
        tmp_path,
        now=now,
        heartbeat_at=stale,
        heartbeat_pid=9999,
    )
    runner = FakeRunner()
    watchdog = _make_monitor(
        tmp_path,
        now=now,
        runner=runner,
        pid_alive=lambda pid: False,
        restart_command=("restart-supervisor", "--resume", str(tmp_path)),
        restart_policy=monitor.RestartPolicy(
            initial_delay_sec=30,
            multiplier=2,
            max_delay_sec=300,
            max_attempts=3,
            reset_after_healthy_sec=60,
        ),
    )

    snapshot = watchdog.poll_once()

    assert snapshot["checks"]["supervisor"]["status"] == "fail"
    assert snapshot["restart"]["action"] == "started"
    assert snapshot["restart"]["last_pid"] == 4321
    assert snapshot["restart"]["consecutive_failures"] == 1
    assert snapshot["restart"]["next_attempt_at"] == "2026-08-04T03:00:30+00:00"
    assert any(call[0] == "restart-supervisor" for call in runner.calls)


def test_restart_backoff_survives_new_monitor_process(tmp_path: Path) -> None:
    first_now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(
        tmp_path,
        now=first_now,
        heartbeat_at=first_now - timedelta(hours=1),
    )
    first_runner = FakeRunner()
    first = _make_monitor(
        tmp_path,
        now=first_now,
        runner=first_runner,
        pid_alive=lambda pid: False,
        restart_command=("restart-supervisor",),
        restart_policy=monitor.RestartPolicy(
            initial_delay_sec=60,
            multiplier=2,
            max_delay_sec=300,
            max_attempts=3,
        ),
    )
    assert first.poll_once()["restart"]["action"] == "started"

    second_now = first_now + timedelta(seconds=10)
    second_runner = FakeRunner()
    second = _make_monitor(
        tmp_path,
        now=second_now,
        runner=second_runner,
        pid_alive=lambda pid: False,
        restart_command=("restart-supervisor",),
        restart_policy=monitor.RestartPolicy(
            initial_delay_sec=60,
            multiplier=2,
            max_delay_sec=300,
            max_attempts=3,
        ),
    )
    snapshot = second.poll_once()

    assert snapshot["restart"]["action"] == "backoff"
    assert snapshot["restart"]["remaining_sec"] == 50.0
    assert not any(call[0] == "restart-supervisor" for call in second_runner.calls)


def test_pause_suppresses_restart_but_monitor_keeps_polling(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(
        tmp_path,
        now=now,
        heartbeat_at=now - timedelta(hours=1),
    )
    pause_path = monitor.create_pause_request(
        tmp_path,
        reason="inspect evidence",
        requested_by="test",
        now=now,
    )
    runner = FakeRunner()
    watchdog = _make_monitor(
        tmp_path,
        now=now,
        runner=runner,
        pid_alive=lambda pid: False,
        restart_command=("restart-supervisor",),
    )

    snapshot = watchdog.poll_once()

    assert pause_path.exists()
    assert snapshot["paused"] is True
    assert snapshot["pause"]["request"]["semantics"] == "cooperative_pause_not_stop"
    assert snapshot["restart"]["action"] == "suppressed_paused"
    assert not any(call[0] == "restart-supervisor" for call in runner.calls)
    assert (tmp_path / "monitor_snapshot.json").exists()


def test_stalled_pipeline_progress_triggers_restart(tmp_path: Path) -> None:
    now = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    run_dir = tmp_path / "runs" / "cycle-1"
    state_path = tmp_path / "campaign_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "phase": "pipeline",
            "active_run_dir": str(run_dir),
            "active_child_pid": 5678,
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    stale_epoch = (now - timedelta(hours=5)).timestamp()
    for path in (run_dir / "checkpoint.json",):
        os.utime(path, (stale_epoch, stale_epoch))

    runner = FakeRunner()
    config = monitor.MonitorConfig(
        campaign_dir=tmp_path,
        nodes=runner.nodes,
        expected_gpus_per_node=8,
        pipeline_progress_stale_sec=60,
        restart_command=("restart-supervisor",),
    )
    watchdog = monitor.CampaignMonitor(
        config,
        runner=runner,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        sleep=lambda seconds: None,
        pid_alive=lambda pid: True,
    )

    snapshot = watchdog.poll_once()

    assert snapshot["checks"]["supervisor"]["status"] == "ok"
    assert snapshot["checks"]["progress"]["status"] == "fail"
    assert snapshot["restart"]["action"] == "started"


def test_stalled_progress_can_restart_despite_pool_probe_failure(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)
    runner = FakeRunner()
    watchdog = _make_monitor(
        tmp_path,
        now=now,
        runner=runner,
        restart_command=("restart-supervisor",),
    )
    checks = {
        name: monitor.Check(name, status, name, monitor.utc_iso(now))
        for name, status in {
            "supervisor": "ok",
            "campaign": "degraded",
            "progress": "fail",
            "bridge": "ok",
            "cluster": "fail",
            "gpu": "ok",
            "pool": "fail",
            "lease": "ok",
        }.items()
    }

    restart = watchdog._maybe_restart(
        now,
        supervisor=checks["supervisor"],
        campaign=checks["campaign"],
        progress=checks["progress"],
        pause_requested=False,
        dependencies=(
            checks["bridge"],
            checks["cluster"],
            checks["gpu"],
            checks["pool"],
            checks["lease"],
        ),
    )

    assert restart["action"] == "started"


def test_gpu_timeout_degrades_when_fresh_pool_snapshot_confirms_resources(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 8, 0, tzinfo=UTC)
    runner = FakeRunner(nodes=("10.0.0.1",))
    _write_campaign_files(tmp_path, now=now)
    (tmp_path / "monitor_snapshot.json").write_text(
        json.dumps(
            {
                "generated_at": monitor.utc_iso(now),
                "checks": {"pool": {"status": "ok"}},
            }
        ),
        encoding="utf-8",
    )
    watchdog = _make_monitor(tmp_path, now=now, runner=runner)
    watchdog._run_commands_parallel = lambda *args, **kwargs: {
        "10.0.0.1": {
            "argv": [],
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "elapsed_sec": 20,
            "error": "TimeoutExpired",
        }
    }
    watchdog._read_resource_snapshot = lambda _now: {
        "fresh": True,
        "error": None,
        "allocations": [
            {
                "status": "active",
                "owner": None,
                "nodes": ["10.0.0.1"],
            }
        ],
    }
    watchdog._read_claim_records = lambda _now: {
        "error": None,
        "nodes": {
            "10.0.0.1": {
                "active": True,
                "owner": None,
                "purpose": None,
            }
        },
    }

    check = watchdog._check_gpus(now)

    assert check.status == "degraded"
    assert check.data["fallback_used"] is True


def test_cb_and_gpu_parsers() -> None:
    cb_text = """
NODE             ALIVE    GPUs CLUSTER                                      CLAIM              EXPIRES
28.83.2.169      yes      8    unirl_video_zw5                              owner/purpose      2026-08-03T22:02:33Z(178m)
28.83.50.39      STALE(99s) 8  unirl_video_zw6                              unclaimed          -
-- alive 1 / claimed 1 --
"""
    parsed = monitor.parse_cb_list(cb_text)
    assert parsed["28.83.2.169"]["alive"] == "yes"
    assert parsed["28.83.2.169"]["gpus"] == 8
    assert parsed["28.83.50.39"]["alive"] == "STALE(99s)"
    assert parsed["28.83.50.39"]["claim"] == "unclaimed"

    gpus = monitor.parse_nvidia_smi(
        "0, NVIDIA H20, 97871, 1024, 25\n"
        "1, NVIDIA H20, 97871, 2048, 50\n"
        "[cb hook] informational stderr accidentally merged\n"
    )
    assert [gpu["index"] for gpu in gpus] == [0, 1]
    assert gpus[1]["utilization_gpu_percent"] == 50


def test_cluster_check_rejects_claim_owned_by_another_session(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    runner = FakeRunner()
    config = monitor.MonitorConfig(
        campaign_dir=tmp_path,
        nodes=runner.nodes,
        expected_gpus_per_node=8,
        expected_claim_owner="expected-owner",
        expected_claim_purpose="research",
    )
    watchdog = monitor.CampaignMonitor(
        config,
        runner=runner,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        sleep=lambda seconds: None,
        pid_alive=lambda pid: True,
    )

    snapshot = watchdog.poll_once()

    assert snapshot["checks"]["cluster"]["status"] == "fail"
    assert snapshot["checks"]["cluster"]["data"]["claim_mismatches"]


def test_dependency_failure_suppresses_supervisor_restart(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(
        tmp_path,
        now=now,
        heartbeat_at=now - timedelta(hours=1),
    )
    runner = FakeRunner()
    config = monitor.MonitorConfig(
        campaign_dir=tmp_path,
        nodes=runner.nodes,
        expected_gpus_per_node=8,
        expected_claim_owner="wrong-owner",
        restart_command=("restart-supervisor",),
    )
    watchdog = monitor.CampaignMonitor(
        config,
        runner=runner,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        sleep=lambda seconds: None,
        pid_alive=lambda pid: False,
    )

    snapshot = watchdog.poll_once()

    assert snapshot["restart"]["action"] == "suppressed_dependency_failure"
    assert "cluster" in snapshot["restart"]["failed_dependencies"]
    assert not any(call[0] == "restart-supervisor" for call in runner.calls)


def test_slow_cb_list_does_not_block_gpu_probes(tmp_path: Path) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    nodes = ("10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4")
    base = FakeRunner(nodes=nodes)
    gpu_completed = threading.Event()
    gpu_count = 0
    lock = threading.Lock()
    list_observed_gpu_completion = False
    cb_root = tmp_path / "cluster"
    cb_command = cb_root / ".tools" / "clusterbridge.sh"
    cb_command.parent.mkdir(parents=True)
    cb_command.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        nonlocal gpu_count, list_observed_gpu_completion
        argv = [str(part) for part in args]
        if argv[-1] == "list":
            list_observed_gpu_completion = gpu_completed.wait(timeout=0.5)
            raise subprocess.TimeoutExpired(argv, 0.05)
        result = base(argv, **kwargs)
        if "nvidia-smi " in argv[-1]:
            with lock:
                gpu_count += 1
                if gpu_count == len(nodes):
                    gpu_completed.set()
        return result

    config = monitor.MonitorConfig(
        campaign_dir=tmp_path,
        cb_command=str(cb_command),
        nodes=nodes,
        expected_gpus_per_node=8,
        cluster_probe_timeout_sec=0.05,
        gpu_probe_timeout_sec=0.5,
        external_probe_deadline_sec=0.6,
        resource_snapshot_path=tmp_path / "missing-resource-snapshot.json",
    )
    watchdog = monitor.CampaignMonitor(
        config,
        runner=runner,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        pid_alive=lambda pid: True,
    )

    snapshot = watchdog.poll_once()

    assert list_observed_gpu_completion is True
    assert gpu_completed.is_set()
    assert snapshot["checks"]["gpu"]["status"] == "ok"
    assert snapshot["checks"]["gpu"]["data"]["total_gpus"] == 32
    assert snapshot["checks"]["cluster"]["status"] == "fail"
    assert "TimeoutExpired" in snapshot["checks"]["cluster"]["data"]["list"][
        "error"
    ]


def test_one_slow_gpu_node_does_not_serialize_other_nodes(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    nodes = ("10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4")
    slow_node = nodes[0]
    base = FakeRunner(nodes=nodes)
    fast_completed: set[str] = set()
    fast_nodes_done = threading.Event()
    slow_observed_fast_completion = False
    lock = threading.Lock()

    def runner(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        nonlocal slow_observed_fast_completion
        argv = [str(part) for part in args]
        if "nvidia-smi " in argv[-1]:
            node = argv[2]
            if node == slow_node:
                slow_observed_fast_completion = fast_nodes_done.wait(
                    timeout=0.5
                )
                raise subprocess.TimeoutExpired(argv, 0.05)
            else:
                with lock:
                    fast_completed.add(node)
                    if fast_completed == set(nodes[1:]):
                        fast_nodes_done.set()
        return base(argv, **kwargs)

    config = monitor.MonitorConfig(
        campaign_dir=tmp_path,
        nodes=nodes,
        expected_gpus_per_node=8,
        cluster_probe_timeout_sec=0.5,
        gpu_probe_timeout_sec=0.05,
        external_probe_deadline_sec=0.2,
    )
    watchdog = monitor.CampaignMonitor(
        config,
        runner=runner,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        pid_alive=lambda pid: True,
    )

    snapshot = watchdog.poll_once()
    gpu = snapshot["checks"]["gpu"]

    assert slow_observed_fast_completion is True
    assert fast_completed == set(nodes[1:])
    assert gpu["status"] == "fail"
    assert gpu["data"]["failed_nodes"] == [slow_node]
    assert "TimeoutExpired" in gpu["data"]["nodes"][slow_node]["error"]
    assert gpu["data"]["total_gpus"] == 24


def test_resource_snapshot_only_degrades_transport_failure(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    nodes = ("10.0.0.1", "10.0.0.2")
    cb_root = tmp_path / "cluster"
    cb_command = cb_root / ".tools" / "clusterbridge.sh"
    cb_command.parent.mkdir(parents=True)
    cb_command.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    for node in nodes:
        claim_path = cb_root / ".bridge" / node / "claim.json"
        claim_path.parent.mkdir(parents=True)
        claim_path.write_text(
            json.dumps(
                {
                    "owner": "owner",
                    "purpose": "research",
                    "expires_at": monitor.utc_iso(now + timedelta(hours=1)),
                }
            ),
            encoding="utf-8",
        )
    resource_snapshot = tmp_path / "resource-snapshot.json"
    resource_snapshot.write_text(
        json.dumps(
            {
                "updated_at": monitor.utc_iso(now),
                "summary": {
                    "active_allocations": 2,
                    "allocated_nodes": 2,
                    "allocated_gpus": 16,
                },
                "allocations": [
                    {
                        "id": f"alloc-{index}",
                        "status": "active",
                        "nodes": [node],
                        "gpu_count": 8,
                    }
                    for index, node in enumerate(nodes)
                ],
            }
        ),
        encoding="utf-8",
    )
    base = FakeRunner(nodes=nodes)

    def list_failure(
        args: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        argv = [str(part) for part in args]
        if argv[-1] == "list":
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr="cephfs list path temporarily slow",
            )
        return base(argv, **kwargs)

    config = monitor.MonitorConfig(
        campaign_dir=tmp_path,
        cb_command=str(cb_command),
        nodes=nodes,
        expected_gpus_per_node=8,
        expected_claim_owner="owner",
        expected_claim_purpose="research",
        resource_snapshot_path=resource_snapshot,
    )
    watchdog = monitor.CampaignMonitor(
        config,
        runner=list_failure,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        pid_alive=lambda pid: True,
    )

    cluster = watchdog.poll_once()["checks"]["cluster"]

    assert cluster["status"] == "degraded"
    assert cluster["data"]["fallback_used"] is True
    assert cluster["data"]["resource_snapshot"]["fresh"] is True

    first_claim = cb_root / ".bridge" / nodes[0] / "claim.json"
    first_claim.write_text(
        json.dumps(
            {
                "owner": "different-owner",
                "purpose": "research",
                "expires_at": monitor.utc_iso(now + timedelta(hours=1)),
            }
        ),
        encoding="utf-8",
    )
    mismatched_claim = monitor.CampaignMonitor(
        config,
        runner=list_failure,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        pid_alive=lambda pid: True,
    ).poll_once()["checks"]["cluster"]

    assert mismatched_claim["status"] == "fail"
    assert mismatched_claim["data"]["fallback_used"] is False
    first_claim.write_text(
        json.dumps(
            {
                "owner": "owner",
                "purpose": "research",
                "expires_at": monitor.utc_iso(now + timedelta(hours=1)),
            }
        ),
        encoding="utf-8",
    )

    class ExplicitDeadRunner(FakeRunner):
        def __call__(
            self,
            args: list[str],
            **kwargs: Any,
        ) -> subprocess.CompletedProcess:
            argv = [str(part) for part in args]
            if argv[-1] == "list":
                rows = "\n".join(
                    (
                        f"{node:<16} "
                        f"{'STALE(99s)' if node == nodes[0] else 'yes':<8} "
                        "8    test_cluster"
                        "                                 owner/research     "
                        "2026-08-04T05:03:49Z(599m)"
                    )
                    for node in nodes
                )
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="",
                    stderr=(
                        "NODE             ALIVE    GPUs CLUSTER"
                        "                                      CLAIM"
                        "              EXPIRES\n"
                        f"{rows}\n"
                        "-- alive 1 / claimed 2 --\n"
                    ),
                )
            return super().__call__(argv, **kwargs)

    explicit_dead = monitor.CampaignMonitor(
        config,
        runner=ExplicitDeadRunner(nodes=nodes),
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        pid_alive=lambda pid: True,
    ).poll_once()["checks"]["cluster"]

    assert explicit_dead["status"] == "fail"
    assert explicit_dead["data"]["unhealthy_nodes"] == [nodes[0]]


def test_consecutive_full_green_polls_remain_ok(tmp_path: Path) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    nodes = ("10.0.0.1", "10.0.0.2")
    runner = FakeRunner(nodes=nodes)
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text("clusterbridge_pool: {}\n", encoding="utf-8")
    lease_heartbeat = tmp_path / "lease-heartbeat.json"
    lease_heartbeat.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "status": "ok",
                "timestamp": monitor.utc_iso(now),
            }
        ),
        encoding="utf-8",
    )
    original_call = runner.__call__

    def all_green_runner(
        args: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        argv = [str(part) for part in args]
        if "researchclaw.rsi.pool_probe" in argv:
            payload = {
                "pool_id": "test-pool",
                "claimed": True,
                "prepared": True,
                "ray_started": True,
                "resources": {
                    "total_gpu": 16,
                    "available_gpu": 16,
                    "alive_nodes": 2,
                },
            }
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(payload),
                stderr="",
            )
        return original_call(argv, **kwargs)

    config = monitor.MonitorConfig(
        campaign_dir=tmp_path,
        nodes=nodes,
        expected_gpus_per_node=8,
        pool_config=pool_config,
        lease_heartbeat_path=lease_heartbeat,
    )
    watchdog = monitor.CampaignMonitor(
        config,
        runner=all_green_runner,
        urlopen=lambda request, timeout: FakeResponse({"status": "ok"}),
        clock=lambda: now,
        monotonic=lambda: 10.0,
        pid_alive=lambda pid: True,
    )

    first = watchdog.poll_once()
    second = watchdog.poll_once()

    assert first["overall"] == "ok"
    assert second["overall"] == "ok"
    assert all(
        check["status"] == "ok" for check in second["checks"].values()
    )


def test_run_forever_is_bounded_for_tests(tmp_path: Path) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(tmp_path, now=now)
    runner = FakeRunner()
    watchdog = _make_monitor(tmp_path, now=now, runner=runner)

    assert watchdog.run_forever(max_iterations=2) == 2
    assert sum(call[-1] == "list" for call in runner.calls) == 2


@pytest.mark.parametrize(
    "state_status",
    [
        "completed",
        "stopped",
        "paused",
        "paused_single_cycle",
        "paused_failure",
        "paused_failure_threshold",
    ],
)
def test_terminal_campaign_does_not_require_active_checkpoint(
    tmp_path: Path,
    state_status: str,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(
        tmp_path,
        now=now,
        heartbeat_pid=os.getpid(),
        state_status=state_status,
    )
    run_dir = tmp_path / "runs" / "cycle-1"
    (run_dir / "checkpoint.json").unlink()
    runner = FakeRunner()
    watchdog = _make_monitor(
        tmp_path,
        now=now,
        runner=runner,
        pid_alive=lambda pid: True,
    )

    snapshot = watchdog.poll_once()

    assert snapshot["checks"]["checkpoint"]["status"] == "ok"
    assert "not required" in snapshot["checks"]["checkpoint"]["detail"]


def test_terminal_campaign_with_dead_supervisor_is_healthy_and_not_restarted(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 4, 3, 0, tzinfo=UTC)
    _write_campaign_files(
        tmp_path,
        now=now,
        heartbeat_at=now - timedelta(hours=1),
        heartbeat_pid=9999,
        state_status="completed",
    )
    runner = FakeRunner()
    watchdog = _make_monitor(
        tmp_path,
        now=now,
        runner=runner,
        pid_alive=lambda pid: False,
        restart_command=("restart-supervisor",),
    )

    snapshot = watchdog.poll_once()

    assert snapshot["checks"]["supervisor"]["status"] == "ok"
    assert "not required" in snapshot["checks"]["supervisor"]["detail"]
    assert snapshot["restart"]["action"] == "not_needed"
    assert not any(call[0] == "restart-supervisor" for call in runner.calls)


@pytest.mark.parametrize(
    "name",
    ["rsi-monitor", "rsi-monitor-launch", "rsi-daemon", "rsi-pause"],
)
def test_operations_scripts_are_valid_and_executable(name: str) -> None:
    script = ROOT / "bin" / name
    assert script.exists()
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert os.access(script, os.X_OK)


def test_pause_script_creates_request_without_stopping_processes(
    tmp_path: Path,
) -> None:
    script = ROOT / "bin" / "rsi-pause"
    result = subprocess.run(
        [str(script), str(tmp_path), "manual review"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "control" / "pause").read_text())
    assert payload["reason"] == "manual review"
    assert "GPU claims remain active" in result.stdout


def test_daemon_foreground_mode_is_test_safe(tmp_path: Path) -> None:
    # Stub every external probe so the detached wrapper can be exercised in a
    # bounded foreground mode without touching a real Bridge or cluster.
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "exec \"$RSI_REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    campaign = tmp_path / "campaign"
    _write_campaign_files(
        campaign,
        now=datetime.now(UTC),
        heartbeat_pid=os.getpid(),
    )
    # --max-iterations proves daemon mode is bounded. A deliberately invalid
    # Bridge URL and /bin/false cb yield a health snapshot, not a hanging job.
    env = {
        **os.environ,
        "RSI_DAEMON_FOREGROUND": "1",
        "RSI_PYTHON": str(fake_python),
        "RSI_REAL_PYTHON": os.environ.get(
            "PYTHON",
            str(ROOT / ".venv" / "bin" / "python"),
        ),
    }
    result = subprocess.run(
        [
            str(ROOT / "bin" / "rsi-daemon"),
            str(campaign),
            "--max-iterations",
            "1",
            "--bridge-health-url",
            "http://127.0.0.1:1/health",
            "--cb-command",
            "/bin/false",
            "--bridge-timeout-sec",
            "0.1",
            "--command-timeout-sec",
            "0.1",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (campaign / "monitor_snapshot.json").exists()
    assert not (campaign / "rsi-monitor.pid").exists()


def test_daemon_rejects_existing_json_pid_identity(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().split()
    (campaign / "rsi-monitor.pid").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "start_ticks": int(fields[21]),
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(ROOT / "bin" / "rsi-daemon"),
            str(campaign),
            "--max-iterations",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 1
    assert f"already running with PID {os.getpid()}" in result.stderr


def test_monitor_launcher_restarts_failed_child_until_stopped(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "monitor.pid"
    count_file = tmp_path / "count"
    child = tmp_path / "child.sh"
    child.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f"n=$(cat {count_file!s} 2>/dev/null || echo 0)\n"
        "n=$((n+1))\n"
        f"printf '%s\\n' \"$n\" > {count_file!s}\n"
        "exit 7\n",
        encoding="utf-8",
    )
    child.chmod(0o755)
    env = {**os.environ, "RSI_MONITOR_RESTART_BACKOFF_SEC": "0.05"}
    proc = subprocess.Popen(
        [
            str(ROOT / "bin" / "rsi-monitor-launch"),
            str(pid_file),
            str(child),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                if int(count_file.read_text()) >= 2:
                    break
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(0.02)
        else:
            pytest.fail("monitor launcher did not restart the failed child")
        os.kill(proc.pid, signal.SIGTERM)
        assert proc.wait(timeout=5) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert int(count_file.read_text()) >= 2
    assert not (tmp_path / "monitor.pid.watchdog.pid").exists()
