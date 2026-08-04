"""Tests for the shared-CephFS ClusterBridge GPU sandbox."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

from researchclaw.config import ClusterBridgeConfig, ExperimentConfig
from researchclaw.experiment.clusterbridge_sandbox import ClusterBridgeSandbox
from researchclaw.experiment.factory import create_sandbox


def test_clusterbridge_command_isolates_network_and_gpu(tmp_path: Path) -> None:
    cfg = ClusterBridgeConfig(
        node="10.0.0.1",
        gpu_ids=(2,),
        timeout_sec=123,
    )
    sandbox = ClusterBridgeSandbox(cfg, tmp_path / "work")
    command = sandbox._build_remote_command(
        Path("/root/shared/rc-test"),
        entry_point="main.py",
        args=["--seed", "7"],
        env_overrides={"RC_TEST": "ok"},
        timeout_sec=45,
    )
    assert "unshare --net env" in command
    assert "CUDA_VISIBLE_DEVICES=2" in command
    assert "RC_TEST=ok" in command
    assert "timeout -k 30 45s" in command
    assert "main.py --seed 7" in command


def test_clusterbridge_run_parses_remote_metrics(tmp_path: Path) -> None:
    cfg = ClusterBridgeConfig(
        node="10.0.0.1",
        shared_root=str(tmp_path / "shared"),
        cleanup_remote=False,
        timeout_sec=30,
    )
    sandbox = ClusterBridgeSandbox(cfg, tmp_path / "work")

    def fake_run(*args, **kwargs):
        del args, kwargs
        run_dirs = list((tmp_path / "shared").iterdir())
        assert len(run_dirs) == 1
        (run_dirs[0] / "results.json").write_text(
            '{"metrics": {"secondary_metric": 2.5}}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(
            args=["cb"],
            returncode=0,
            stdout="primary_metric: 1.25\n",
            stderr="",
        )

    with mock.patch("subprocess.run", side_effect=fake_run):
        result = sandbox.run("print('unused')", timeout_sec=10)

    assert result.returncode == 0
    assert result.metrics["primary_metric"] == 1.25
    assert result.metrics["secondary_metric"] == 2.5


def test_clusterbridge_factory_requires_node(tmp_path: Path) -> None:
    config = ExperimentConfig(
        mode="clusterbridge",
        clusterbridge=ClusterBridgeConfig(node=""),
    )
    try:
        create_sandbox(config, tmp_path)
    except RuntimeError as exc:
        assert "clusterbridge.node" in str(exc)
    else:
        raise AssertionError("create_sandbox should require a ClusterBridge node")
