"""Mocked tests for the multi-node ClusterBridge / Ray execution pool."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from researchclaw.cluster import (
    BridgeResult,
    ClusterBridgeClient,
    ClusterBridgePoolConfig,
    ClusterNode,
    LeaseKeepalive,
    RayPoolConfig,
    UnsafeForceClaimError,
)
from researchclaw.experiment.clusterbridge_pool import (
    ClusterBridgePool,
    PoolNotClaimedError,
    PoolTaskConflict,
    PoolTaskNotFinished,
    PoolTaskTimeout,
    RayResourceError,
    RayResources,
)


def _config(
    tmp_path: Path,
    *,
    gpu_ids: tuple[int, ...] = (0, 1),
    node_count: int = 2,
) -> ClusterBridgePoolConfig:
    nodes = tuple(
        ClusterNode(address=f"10.0.0.{index + 1}", gpu_ids=gpu_ids)
        for index in range(node_count)
    )
    return ClusterBridgePoolConfig(
        nodes=nodes,
        cb_command="/root/shared/.clusters/.tools/clusterbridge.sh",
        pool_id="test-pool",
        log_root=str(tmp_path),
        claim_ttl_min=10,
        renew_interval_sec=60,
        command_timeout_sec=5,
        expected_total_gpus=len(gpu_ids) * node_count,
        ray=RayPoolConfig(
            start_timeout_sec=5,
            resource_timeout_sec=5,
            poll_interval_sec=0.01,
        ),
    )


def _result(stdout: str = "", stderr: str = "") -> BridgeResult:
    return BridgeResult(
        argv=("bash", "cb"),
        returncode=0,
        stdout=stdout,
        stderr=stderr,
        elapsed_sec=0.01,
    )


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.run_handler = lambda node, command: _result()
        self.claim_owner = "test-owner"
        self.claim_purpose = "AutoResearchClaw multi-node experiments"
        self.claims_active = True

    def list_nodes(self) -> BridgeResult:
        self.calls.append(("list",))
        return _result("nodes")

    def claim_records(self, nodes) -> dict[str, dict[str, object]]:
        self.calls.append(
            ("claim_records", tuple(node.address for node in nodes))
        )
        return {
            node.address: {
                "active": self.claims_active,
                "owner": self.claim_owner,
                "purpose": self.claim_purpose,
            }
            for node in nodes
        }

    def claim(
        self,
        nodes,
        *,
        purpose: str,
        ttl_min: int,
        force: bool = False,
    ) -> BridgeResult:
        self.calls.append(
            (
                "claim",
                tuple(node.address for node in nodes),
                purpose,
                ttl_min,
                force,
            )
        )
        return _result()

    def renew(self, nodes, *, ttl_min: int) -> BridgeResult:
        self.calls.append(
            ("renew", tuple(node.address for node in nodes), ttl_min)
        )
        return _result()

    def release(self, nodes, *, force: bool = False) -> BridgeResult:
        self.calls.append(
            ("release", tuple(node.address for node in nodes), force)
        )
        return _result()

    def run_node(
        self,
        node: ClusterNode,
        command: str,
        *,
        timeout_sec: float | None = None,
    ) -> BridgeResult:
        self.calls.append(("run", node.address, command, timeout_sec))
        return self.run_handler(node, command)

    def detach(self, command, *, timeout_sec=None) -> BridgeResult:
        self.calls.append(("detach", tuple(command), timeout_sec))
        return _result()


class ClaimFailClient(FakeClient):
    def claim(
        self,
        nodes,
        *,
        purpose: str,
        ttl_min: int,
        force: bool = False,
    ) -> BridgeResult:
        super().claim(
            nodes,
            purpose=purpose,
            ttl_min=ttl_min,
            force=force,
        )
        raise RuntimeError("partial claim failure")


def test_config_loads_32_gpu_pool(tmp_path: Path) -> None:
    config_path = tmp_path / "pool.yaml"
    config_path.write_text(
        """
clusterbridge_pool:
  pool_id: research-32
  log_root: /root/shared/.clusters/.tmp/research-32
  expected_total_gpus: 32
  nodes:
    - address: 10.0.0.1
      gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
    - address: 10.0.0.2
      gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
    - address: 10.0.0.3
      gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
    - address: 10.0.0.4
      gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  ray:
    head_node: 10.0.0.2
    port: 6380
""",
        encoding="utf-8",
    )

    config = ClusterBridgePoolConfig.from_file(config_path)

    assert config.configured_gpu_count == 32
    assert config.expected_total_gpus == 32
    assert config.head_node.address == "10.0.0.2"
    assert config.ray.port == 6380
    assert config.allow_force_claim is False


def test_config_rejects_wrong_expected_gpu_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        ClusterBridgePoolConfig(
            nodes=(ClusterNode("10.0.0.1", (0, 1)),),
            expected_total_gpus=32,
            log_root=str(tmp_path),
            claim_ttl_min=10,
            renew_interval_sec=60,
        )


def test_config_rejects_duplicate_ray_ips(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate Ray"):
        ClusterBridgePoolConfig(
            nodes=(
                ClusterNode("10.0.0.1", (0,), ray_ip="192.0.2.1"),
                ClusterNode("10.0.0.2", (0,), ray_ip="192.0.2.1"),
            ),
            expected_total_gpus=2,
            log_root=str(tmp_path),
            claim_ttl_min=10,
            renew_interval_sec=60,
        )


def test_config_rejects_string_force_claim_flag(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="YAML boolean"):
        ClusterBridgePoolConfig.from_mapping(
            {
                "nodes": [{"address": "10.0.0.1", "gpu_ids": [0]}],
                "allow_force_claim": "false",
                "log_root": str(tmp_path),
            }
        )


def test_bridge_client_never_force_claims_by_default() -> None:
    calls: list[list[str]] = []

    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs["env"] is None
        return subprocess.CompletedProcess(argv, 0, "", "")

    client = ClusterBridgeClient("/clusterbridge.sh", runner=runner)
    node = ClusterNode("10.0.0.1", (0,))

    client.claim([node], purpose="test", ttl_min=60)
    assert "--force" not in calls[0]
    with pytest.raises(UnsafeForceClaimError):
        client.claim([node], purpose="test", ttl_min=60, force=True)
    assert len(calls) == 1


def test_bridge_node_run_disables_auto_claim() -> None:
    captured: dict[str, Any] = {}

    def runner(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    client = ClusterBridgeClient("/clusterbridge.sh", runner=runner)
    client.run_node("10.0.0.1", "hostname", timeout_sec=12)

    assert captured["argv"] == [
        "bash",
        "/clusterbridge.sh",
        "10.0.0.1",
        "run",
        "hostname",
    ]
    assert captured["env"]["CB_NO_AUTOCLAIM"] == "1"
    assert "ssh" not in " ".join(captured["argv"]).lower()


def test_pool_requires_claim_before_node_operations(tmp_path: Path) -> None:
    pool = ClusterBridgePool(_config(tmp_path), client=FakeClient())
    with pytest.raises(PoolNotClaimedError):
        pool.cleanup_nodes()


def test_claim_renew_release_lifecycle(tmp_path: Path) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(_config(tmp_path), client=client)
    pool.claim(start_keepalive=False)
    pool.renew()
    pool.release(restore_spin=False, cleanup=False)

    assert [call[0] for call in client.calls] == [
        "claim",
        "claim_records",
        "renew",
        "claim_records",
        "claim_records",
        "release",
    ]
    assert client.calls[0][-1] is False
    state = json.loads((pool.state_dir / "state.json").read_text())
    assert state["claimed"] is False


def test_failed_batch_claim_attempts_owned_claim_rollback(
    tmp_path: Path,
) -> None:
    client = ClaimFailClient()
    pool = ClusterBridgePool(_config(tmp_path), client=client)

    with pytest.raises(RuntimeError, match="partial claim failure"):
        pool.claim(start_keepalive=False)

    assert [call[0] for call in client.calls] == ["claim", "release"]
    assert pool.claimed is False


def test_restore_state_fails_closed_when_live_claim_is_missing(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    original = ClusterBridgePool(config, client=FakeClient())
    original.claim(start_keepalive=False)
    client = FakeClient()
    client.claims_active = False
    restored = ClusterBridgePool(
        config,
        client=client,
        initialize_state=False,
    )

    with pytest.raises(Exception, match="ownership verification failed"):
        restored.restore_state()

    assert restored.claimed is False
    assert restored.prepared is False
    assert restored.ray_started is False


def test_release_refuses_to_touch_nodes_after_claim_owner_changes(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path)
    config = ClusterBridgePoolConfig(
        nodes=base.nodes,
        cb_command=base.cb_command,
        purpose=base.purpose,
        pool_id=base.pool_id,
        log_root=base.log_root,
        claim_ttl_min=base.claim_ttl_min,
        renew_interval_sec=base.renew_interval_sec,
        max_renew_failures=base.max_renew_failures,
        command_timeout_sec=base.command_timeout_sec,
        parallelism=base.parallelism,
        expected_total_gpus=base.expected_total_gpus,
        expected_claim_owner="test-owner",
        ray=base.ray,
    )
    client = FakeClient()
    pool = ClusterBridgePool(config, client=client)
    pool.claim(start_keepalive=False)
    client.claim_owner = "different-owner"

    with pytest.raises(Exception, match="owner"):
        pool.release(cleanup=False, restore_spin=False)

    assert not any(call[0] == "release" for call in client.calls)


def test_release_still_relinquishes_lease_when_spin_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(_config(tmp_path), client=client)
    pool.claim(start_keepalive=False)

    def fail_spin() -> None:
        raise RuntimeError("spin unavailable")

    monkeypatch.setattr(pool, "spin", fail_spin)
    with pytest.raises(Exception, match="spin unavailable"):
        pool.release(cleanup=False)

    assert any(call[0] == "release" for call in client.calls)
    assert pool.claimed is False


def test_prepare_orders_cleanup_pause_gpu_ray_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(_config(tmp_path), client=client)
    pool.claim(start_keepalive=False)
    order: list[str] = []
    resources = RayResources(16, 4, 16, 4, 2, {"GPU": 4}, {"GPU": 4})

    monkeypatch.setattr(pool, "cleanup_nodes", lambda: order.append("cleanup"))
    monkeypatch.setattr(pool, "pause_spin", lambda: order.append("pause"))
    monkeypatch.setattr(
        pool, "prepare_cache", lambda: order.append("prepare-cache")
    )
    monkeypatch.setattr(
        pool, "validate_node_gpus", lambda: order.append("validate-gpus")
    )
    monkeypatch.setattr(pool, "start_ray", lambda: order.append("start-ray"))
    monkeypatch.setattr(
        pool,
        "wait_for_ray_resources",
        lambda **kwargs: order.append("validate-ray") or resources,
    )

    observed = pool.prepare()

    assert observed == resources
    assert order == [
        "cleanup",
        "pause",
        "prepare-cache",
        "validate-gpus",
        "start-ray",
        "validate-ray",
    ]
    assert pool.prepared is True


def test_prepare_cache_is_idempotent_and_node_local(tmp_path: Path) -> None:
    base = _config(tmp_path)
    config = ClusterBridgePoolConfig(
        nodes=base.nodes,
        cb_command=base.cb_command,
        purpose=base.purpose,
        pool_id=base.pool_id,
        log_root=base.log_root,
        claim_ttl_min=base.claim_ttl_min,
        renew_interval_sec=base.renew_interval_sec,
        max_renew_failures=base.max_renew_failures,
        command_timeout_sec=base.command_timeout_sec,
        parallelism=base.parallelism,
        expected_total_gpus=base.expected_total_gpus,
        node_cleanup_script=base.node_cleanup_script,
        node_spin_script=base.node_spin_script,
        task_kill_grace_sec=base.task_kill_grace_sec,
        prepare_cache_dir="/data/cache/autoresearch-v2/huggingface",
        prepare_cache_archive="/root/sync/autoresearch-cache.tar",
        ray=base.ray,
    )
    client = FakeClient()
    pool = ClusterBridgePool(config, client=client)
    pool.claim(start_keepalive=False)

    results = pool.prepare_cache()

    assert sorted(results) == sorted(pool.node_addresses)
    commands = [call[2] for call in client.calls if call[0] == "run"]
    assert len(commands) == len(config.nodes)
    assert all("sha256sum \"$archive\"" in command for command in commands)
    assert all("cache-ready" in command for command in commands)
    assert all("tar -xf \"$archive\"" in command for command in commands)


def test_ray_start_commands_are_backgrounded_and_gpu_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()
    config = _config(tmp_path, gpu_ids=(2, 3))
    pool = ClusterBridgePool(config, client=client)
    pool.claim(start_keepalive=False)
    monkeypatch.setattr(pool, "_wait_for_ray_head", lambda **kwargs: None)

    pool.start_ray()

    run_commands = [
        call[2] for call in client.calls if call[0] == "run"
    ]
    head_commands = [command for command in run_commands if "--head" in command]
    worker_commands = [
        command
        for command in run_commands
        if "ray start --block" in command and "--head" not in command
    ]
    assert len(head_commands) == 1
    assert len(worker_commands) == 1
    assert "nohup setsid ray start --block" in head_commands[0]
    assert "CUDA_VISIBLE_DEVICES=2,3" in head_commands[0]
    assert "--num-gpus=2" in head_commands[0]
    assert "--address=10.0.0.1:6379" in worker_commands[0]
    assert pool.ray_started is True


def test_validate_node_gpu_ids(tmp_path: Path) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(_config(tmp_path), client=client)
    pool.claim(start_keepalive=False)

    def handler(node: ClusterNode, command: str) -> BridgeResult:
        assert "nvidia-smi" in command
        return _result(
            json.dumps(
                {
                    "visible_gpu_ids": [0, 1, 2, 3],
                    "configured_gpu_ids": list(node.gpu_ids),
                    "missing_gpu_ids": [],
                }
            )
            + "\n"
        )

    client.run_handler = handler
    result = pool.validate_node_gpus()

    assert set(result) == {"10.0.0.1", "10.0.0.2"}
    assert sum(len(item["configured_gpu_ids"]) for item in result.values()) == 4


def test_wait_for_ray_requires_exact_gpu_and_node_totals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = ClusterBridgePool(
        _config(tmp_path),
        client=FakeClient(),
        sleep=lambda _: None,
    )
    pool.claim(start_keepalive=False)
    values = iter(
        [
            RayResources(8, 2, 8, 2, 1, {"GPU": 2}, {"GPU": 2}),
            RayResources(16, 4, 16, 4, 2, {"GPU": 4}, {"GPU": 4}),
        ]
    )
    monkeypatch.setattr(pool, "query_ray_resources", lambda: next(values))
    ticks = iter([0.0, 0.0, 0.1, 0.2])
    pool._monotonic = lambda: next(ticks)

    result = pool.wait_for_ray_resources(timeout_sec=1)
    assert result.total_gpu == 4
    assert result.alive_nodes == 2


def test_wait_for_ray_raises_when_32_gpus_are_not_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, gpu_ids=tuple(range(8)), node_count=4)
    clock = iter([0.0, 0.0, 1.0, 2.0])
    pool = ClusterBridgePool(
        config,
        client=FakeClient(),
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )
    pool.claim(start_keepalive=False)
    monkeypatch.setattr(
        pool,
        "query_ray_resources",
        lambda: RayResources(128, 24, 128, 24, 4, {"GPU": 24}, {"GPU": 24}),
    )

    with pytest.raises(RayResourceError, match="expected 32 GPUs"):
        pool.wait_for_ray_resources(timeout_sec=1)


def test_wait_for_ray_rejects_wrong_node_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 0.0, 1.0, 2.0])
    pool = ClusterBridgePool(
        _config(tmp_path),
        client=FakeClient(),
        sleep=lambda _: None,
        monotonic=lambda: next(clock),
    )
    pool.claim(start_keepalive=False)
    monkeypatch.setattr(
        pool,
        "query_ray_resources",
        lambda: RayResources(
            16,
            4,
            16,
            4,
            2,
            {"GPU": 4},
            {"GPU": 4},
            nodes=(
                {"node_ip": "10.0.0.1", "gpu": 2},
                {"node_ip": "10.0.0.99", "gpu": 2},
            ),
        ),
    )

    with pytest.raises(RayResourceError):
        pool.wait_for_ray_resources(timeout_sec=1)


def test_background_task_collects_logs_and_result(tmp_path: Path) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
        sleep=lambda _: None,
        monotonic=iter([0.0, 0.1, 0.2]).__next__,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True

    def handler(node: ClusterNode, command: str) -> BridgeResult:
        del node
        if "__RESEARCHCLAW_POOL_RESULT__" in command:
            return _result(
                "__RESEARCHCLAW_POOL_RESULT__="
                '{"state":"finished","returncode":0,'
                '"stdout.log":"trained\\n","stderr.log":""}\n'
            )
        if "nohup setsid --wait bash" in command:
            return _result("4242\n")
        return _result()

    client.run_handler = handler
    result = pool.run_task(
        "python train.py",
        timeout_sec=30,
        task_id="unit-task",
        poll_interval_sec=0.01,
    )

    assert result.returncode == 0
    assert result.stdout == "trained\n"
    assert result.pid == 4242
    assert Path(result.stdout_path).name == "stdout.log"
    summary = json.loads(
        (pool.state_dir / "tasks" / "unit-task" / "summary.json").read_text()
    )
    assert summary["returncode"] == 0


def test_async_submit_probe_collect_and_idempotent_adoption(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True
    probes = {"finished": False}

    def handler(node: ClusterNode, command: str) -> BridgeResult:
        del node
        if "nohup setsid --wait bash" in command:
            return _result("4242\n")
        if "__RESEARCHCLAW_POOL_RESULT__" in command:
            state = "finished" if probes["finished"] else "running"
            return _result(
                "__RESEARCHCLAW_POOL_RESULT__="
                + json.dumps(
                    {
                        "state": state,
                        "pid": 4242,
                        "returncode": 0 if probes["finished"] else None,
                        "stdout.log": "done\n" if probes["finished"] else "",
                        "stderr.log": "",
                    }
                )
                + "\n"
            )
        return _result()

    client.run_handler = handler
    handle = pool.submit_task(
        "python train.py",
        timeout_sec=30,
        task_id="async-task",
    )
    adopted = pool.submit_task(
        "python train.py",
        timeout_sec=30,
        task_id="async-task",
    )
    assert adopted == handle
    assert sum(
        "nohup setsid --wait bash" in call[2]
        for call in client.calls
        if call[0] == "run"
    ) == 1
    assert pool.probe_task("async-task").state == "running"
    with pytest.raises(PoolTaskNotFinished):
        pool.collect_task("async-task")

    probes["finished"] = True
    assert pool.probe_task("async-task").state == "finished"
    result = pool.collect_task("async-task")
    assert result.returncode == 0
    assert result.stdout == "done\n"


def test_async_task_id_conflict_fails_closed(tmp_path: Path) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True
    client.run_handler = lambda node, command: (
        _result("55\n") if "nohup setsid --wait bash" in command else _result()
    )
    pool.submit_task("echo first", timeout_sec=10, task_id="same-id")
    with pytest.raises(PoolTaskConflict):
        pool.submit_task("echo different", timeout_sec=10, task_id="same-id")


def test_async_task_uses_ray_resource_reservation(tmp_path: Path) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True
    client.run_handler = lambda node, command: (
        _result("77\n") if "nohup setsid --wait bash" in command else _result()
    )

    pool.submit_task(
        "python train.py",
        timeout_sec=10,
        task_id="ray-reserved",
        num_gpus=2,
        num_cpus=4,
        env={"EXAMPLE": "value"},
    )

    task_dir = pool.state_dir / "tasks" / "ray-reserved"
    request = json.loads((task_dir / "request.json").read_text())
    payload = json.loads((task_dir / "ray_task.json").read_text())
    script = (task_dir / "ray_task.py").read_text()
    assert request["num_gpus"] == 2
    assert request["num_cpus"] == 4
    assert payload["env"]["EXAMPLE"] == "value"
    assert payload["task_id"] == "ray-reserved"
    assert payload["evidence_path"].endswith("trusted_gpu_evidence.json")
    assert "@ray.remote(num_gpus=payload['num_gpus']" in script
    assert "ray.get(run.remote" in script
    assert "nvidia-smi" in script
    assert "CUDA_VISIBLE_DEVICES" in script
    assert "ray_task_id" in script
    launch = next(
        call[2]
        for call in client.calls
        if call[0] == "run" and "nohup setsid --wait bash" in call[2]
    )
    assert "nohup setsid --wait bash" in launch
    assert "/tmp/researchclaw-autoresearch-v2/test-pool/tasks/ray-reserved" in launch
    assert "RESEARCHCLAW_RAY_TASK_PY" in launch
    assert "RESEARCHCLAW_RAY_TASK_JSON" in launch
    assert str(task_dir / "ray_task.py") not in launch
    command_group = launch.index("{")
    remote_task_dir = (
        "/tmp/researchclaw-autoresearch-v2/test-pool/tasks/ray-reserved"
    )
    assert f"mkdir -p {remote_task_dir}" in launch
    stdout_redirect = launch.index(f"{remote_task_dir}/stdout.log")
    assert command_group < stdout_redirect
    assert f"{remote_task_dir}/result.json" in launch
    assert f"{remote_task_dir}/pid" in launch
    assert f"{remote_task_dir}/launcher.log" in launch
    assert str(task_dir / "pid") not in launch
    assert str(task_dir / "launcher.log") not in launch


def test_async_gpu_task_with_live_remote_pid_does_not_finish_lost(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True
    task_id = f"live-remote-pid-{tmp_path.name}"
    remote_task_dir = (
        Path("/tmp/researchclaw-autoresearch-v2")
        / "test-pool"
        / "tasks"
        / task_id
    )

    def handler(node: ClusterNode, command: str) -> BridgeResult:
        del node
        if "nohup setsid --wait bash" in command:
            return _result("77\n")
        if "__RESEARCHCLAW_POOL_RESULT__" in command:
            completed = subprocess.run(
                command,
                shell=True,
                executable="/bin/bash",
                text=True,
                capture_output=True,
                check=False,
            )
            return BridgeResult(
                argv=("bash", "-lc", command),
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                elapsed_sec=0.01,
            )
        return _result()

    client.run_handler = handler
    try:
        pool.submit_task(
            "python train.py",
            timeout_sec=30,
            task_id=task_id,
            num_gpus=1,
            num_cpus=2,
        )
        launch = next(
            call[2]
            for call in client.calls
            if call[0] == "run" and "nohup setsid --wait bash" in call[2]
        )
        assert f"rm -f {remote_task_dir}/result.json" in launch
        assert f"{remote_task_dir}/pid" in launch
        assert f"{remote_task_dir}/launcher.log" in launch

        remote_task_dir.mkdir(parents=True, exist_ok=True)
        (remote_task_dir / "pid").write_text(
            f"{os.getpid()}\n",
            encoding="utf-8",
        )

        probe = pool.probe_task(task_id)

        assert probe.state == "running"
        assert probe.returncode is None
        assert not (
            pool.state_dir / "tasks" / task_id / "summary.json"
        ).exists()
    finally:
        subprocess.run(
            ["rm", "-rf", str(remote_task_dir)],
            check=False,
        )


def test_async_ray_task_returns_remote_trusted_gpu_evidence(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True
    evidence = {
        "schema": "autoresearch_v2.trusted_gpu_evidence",
        "version": 1,
        "task_id": "ray-evidence",
        "allocated_gpus": 1,
        "gpu_uuids": ["GPU-unit-test"],
    }

    def handler(node: ClusterNode, command: str) -> BridgeResult:
        del node
        if "nohup setsid --wait bash" in command:
            return _result("77\n")
        if "__RESEARCHCLAW_POOL_RESULT__" in command:
            assert (
                "/tmp/researchclaw-autoresearch-v2/test-pool/tasks/"
                "ray-evidence"
            ) in command
            assert "trusted_gpu_evidence.json" in command
            assert str(pool.state_dir) not in command
            return _result(
                "__RESEARCHCLAW_POOL_RESULT__="
                + json.dumps(
                    {
                        "state": "finished",
                        "pid": 77,
                        "returncode": 0,
                        "stdout.log": "trained\n",
                        "stderr.log": "",
                        "trusted_gpu_evidence": evidence,
                    }
                )
                + "\n"
            )
        return _result()

    client.run_handler = handler
    pool.submit_task(
        "python train.py",
        timeout_sec=10,
        task_id="ray-evidence",
        num_gpus=1,
        num_cpus=2,
    )

    assert pool.probe_task("ray-evidence").state == "finished"
    result = pool.collect_task("ray-evidence")
    assert result.trusted_gpu_evidence == evidence
    summary = json.loads(
        (
            pool.state_dir
            / "tasks"
            / "ray-evidence"
            / "summary.json"
        ).read_text()
    )
    assert summary["trusted_gpu_evidence"] == evidence


def test_async_task_resource_change_conflicts_with_existing_task(
    tmp_path: Path,
) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True
    client.run_handler = lambda node, command: (
        _result("77\n") if "nohup setsid --wait bash" in command else _result()
    )

    pool.submit_task(
        "python train.py",
        timeout_sec=10,
        task_id="resource-conflict",
        num_gpus=1,
        num_cpus=2,
    )

    with pytest.raises(PoolTaskConflict):
        pool.submit_task(
            "python train.py",
            timeout_sec=10,
            task_id="resource-conflict",
            num_gpus=2,
            num_cpus=2,
        )


def test_task_probe_uses_short_transport_timeout(tmp_path: Path) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True
    client.run_handler = lambda node, command: (
        _result("77\n") if "nohup setsid --wait bash" in command else _result(
            "__RESEARCHCLAW_POOL_RESULT__="
            '{"state":"running","pid":77,"stdout.log":"","stderr.log":""}\n'
        )
    )
    pool.submit_task("true", timeout_sec=30, task_id="probe-timeout")

    pool.probe_task("probe-timeout")

    assert client.calls[-1][3] <= 10.0


def test_background_task_timeout_sends_term_and_kill(tmp_path: Path) -> None:
    client = FakeClient()
    ticks = iter([0.0, 0.0, 2.0, 2.0])
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
        sleep=lambda _: None,
        monotonic=lambda: next(ticks),
    )
    pool.claim(start_keepalive=False)
    pool._prepared = True

    def handler(node: ClusterNode, command: str) -> BridgeResult:
        del node
        if "nohup setsid --wait bash" in command:
            return _result("55\n")
        if "__RESEARCHCLAW_POOL_RESULT__" in command:
            return _result(
                "__RESEARCHCLAW_POOL_RESULT__="
                '{"state":"running","pid":55,'
                '"stdout.log":"","stderr.log":""}\n'
            )
        return _result()

    client.run_handler = handler
    with pytest.raises(PoolTaskTimeout):
        pool.run_task(
            "sleep 100",
            timeout_sec=1,
            task_id="timeout-task",
            poll_interval_sec=0.01,
        )

    commands = [call[2] for call in client.calls if call[0] == "run"]
    assert any("kill -TERM" in command for command in commands)
    assert any(
        "/tmp/researchclaw-autoresearch-v2/test-pool/tasks/"
        "timeout-task/pid" in command
        for command in commands
        if "kill -TERM" in command
    )
    assert any("kill -KILL" in command for command in commands)


def test_cancel_task_terminates_running_detached_task(tmp_path: Path) -> None:
    client = FakeClient()
    pool = ClusterBridgePool(
        _config(tmp_path, node_count=1),
        client=client,
    )
    pool.claim(start_keepalive=False)
    task_dir = pool.state_dir / "tasks" / "cancel-me"
    task_dir.mkdir(parents=True)
    (task_dir / "pid").write_text("55\n", encoding="utf-8")
    probes = {"count": 0}

    def handler(node: ClusterNode, command: str) -> BridgeResult:
        del node
        if "__RESEARCHCLAW_POOL_RESULT__" in command:
            probes["count"] += 1
            state = "running" if probes["count"] == 1 else "lost"
            return _result(
                "__RESEARCHCLAW_POOL_RESULT__="
                + json.dumps(
                    {
                        "state": state,
                        "pid": 55,
                        "stdout.log": "",
                        "stderr.log": "",
                    }
                )
                + "\n"
            )
        return _result()

    client.run_handler = handler
    result = pool.cancel_task("cancel-me")

    assert result.returncode == 130
    assert pool.cancel_task("cancel-me") == result
    commands = [call[2] for call in client.calls if call[0] == "run"]
    assert any("kill -TERM" in command for command in commands)
    assert any(
        "/tmp/researchclaw-autoresearch-v2/test-pool/tasks/cancel-me/pid"
        in command
        for command in commands
        if "kill -TERM" in command
    )


def test_keepalive_renews_and_reports_terminal_failure() -> None:
    renewed = threading.Event()
    calls = 0

    def renew() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            renewed.set()
            return
        raise RuntimeError("renew failed")

    keepalive = LeaseKeepalive(renew, interval_sec=0.01, max_failures=2)
    keepalive.start()
    assert renewed.wait(timeout=1)
    for _ in range(100):
        try:
            keepalive.assert_healthy()
        except RuntimeError as exc:
            assert "2 consecutive" in str(exc)
            break
        threading.Event().wait(0.01)
    else:
        raise AssertionError("keepalive did not expose terminal failure")
    keepalive.stop()
    assert keepalive.snapshot().renew_count == 1


def test_keepalive_restart_clears_terminal_failure() -> None:
    calls = {"count": 0}
    renewed = threading.Event()

    def renew() -> None:
        calls["count"] += 1
        if calls["count"] <= 2:
            raise RuntimeError("temporary failure")
        renewed.set()

    keepalive = LeaseKeepalive(renew, interval_sec=0.01, max_failures=2)
    keepalive.start()
    for _ in range(100):
        try:
            keepalive.assert_healthy()
        except RuntimeError:
            break
        threading.Event().wait(0.01)
    else:
        raise AssertionError("keepalive did not reach terminal failure")

    keepalive.stop()
    keepalive.start()
    assert renewed.wait(timeout=1)
    keepalive.assert_healthy()
    keepalive.stop()


def test_restore_state_does_not_overwrite_prior_claim(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = ClusterBridgePool(config, client=FakeClient())
    original.claim(start_keepalive=False)

    restored = ClusterBridgePool(
        config,
        client=FakeClient(),
        initialize_state=False,
    )
    state = restored.restore_state()

    assert state["claimed"] is True
    assert restored.claimed is True
