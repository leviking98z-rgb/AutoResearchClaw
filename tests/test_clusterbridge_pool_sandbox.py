from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from researchclaw.config import (
    ClusterBridgePoolSandboxConfig,
    ExperimentConfig,
)
from researchclaw.experiment.clusterbridge_pool import PoolTaskResult
from researchclaw.experiment.clusterbridge_pool_sandbox import (
    ClusterBridgePoolSandbox,
)
from researchclaw.experiment.factory import create_sandbox


class FakePool:
    def __init__(self, root: Path) -> None:
        self.state_dir = root
        self.claimed = True
        self.prepared = True
        self.ray_started = True
        self.config = SimpleNamespace(
            pool_id="fake-32",
            ray=SimpleNamespace(resource_timeout_sec=30),
        )
        self.calls: list[dict[str, object]] = []

    def run_task(self, command: str, **kwargs) -> PoolTaskResult:
        self.calls.append({"command": command, **kwargs})
        run_dir = next((self.state_dir / "experiments").iterdir())
        (run_dir / "results.json").write_text(
            json.dumps({"metrics": {"secondary_metric": 2.5}}),
            encoding="utf-8",
        )
        return PoolTaskResult(
            task_id=str(kwargs["task_id"]),
            returncode=0,
            stdout="primary_metric: 1.25\n",
            stderr="",
            elapsed_sec=1.5,
            timed_out=False,
            remote_dir=str(self.state_dir),
            stdout_path=str(self.state_dir / "stdout.log"),
            stderr_path=str(self.state_dir / "stderr.log"),
            result_path=str(self.state_dir / "result.json"),
            pid=42,
        )


def test_pool_sandbox_submits_project_to_ready_pool(tmp_path: Path) -> None:
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text("clusterbridge_pool: {}\n", encoding="utf-8")
    fake_pool = FakePool(tmp_path / "state")
    fake_pool.state_dir.mkdir()

    def factory(path, *, restore_state):
        assert Path(path) == pool_config
        assert restore_state is True
        return fake_pool

    config = ClusterBridgePoolSandboxConfig(
        config_file=str(pool_config),
        cleanup_remote=False,
        network_isolation=False,
    )
    sandbox = ClusterBridgePoolSandbox(
        config,
        tmp_path / "work",
        pool_factory=factory,
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(
        "import ray\nray.init(address='auto')\nprint('primary_metric: 1.25')\n",
        encoding="utf-8",
    )

    result = sandbox.run_project(project, timeout_sec=20)

    assert result.returncode == 0
    assert result.metrics["primary_metric"] == 1.25
    assert result.metrics["secondary_metric"] == 2.5
    assert fake_pool.calls[0]["require_ready"] is True
    assert "main.py" in str(fake_pool.calls[0]["command"])
    metadata = json.loads(
        next((tmp_path / "work").glob(
            "_clusterbridge_pool_project_*/.clusterbridge_pool_task.json"
        )).read_text(encoding="utf-8")
    )
    assert metadata["state"] == "finished"
    assert metadata["task_id"] == fake_pool.calls[0]["task_id"]


def test_pool_factory_selects_clusterbridge_pool_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text("clusterbridge_pool: {}\n", encoding="utf-8")
    cfg = ClusterBridgePoolSandboxConfig(config_file=str(pool_config))
    monkeypatch.setattr(
        ClusterBridgePoolSandbox,
        "check_available",
        staticmethod(lambda _config: (True, "ready")),
    )

    sandbox = create_sandbox(
        ExperimentConfig(mode="clusterbridge_pool", clusterbridge_pool=cfg),
        tmp_path / "work",
    )

    assert isinstance(sandbox, ClusterBridgePoolSandbox)


def test_factory_work_item_env_makes_pool_task_id_deterministic(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text("clusterbridge_pool: {}\n", encoding="utf-8")
    fake_pool = FakePool(tmp_path / "state")
    fake_pool.state_dir.mkdir()
    monkeypatch.setenv("RESEARCHCLAW_FACTORY_ID", "factory-a")
    monkeypatch.setenv("RESEARCHCLAW_IDEA_ID", "idea-a")
    monkeypatch.setenv("RESEARCHCLAW_WORK_ITEM_ID", "idea-a-pilot")
    monkeypatch.setenv("RESEARCHCLAW_WORK_ITEM_ATTEMPT", "1")
    monkeypatch.setenv("RESEARCHCLAW_GPU_REQUEST", "2")

    config = ClusterBridgePoolSandboxConfig(
        config_file=str(pool_config),
        cleanup_remote=False,
        network_isolation=False,
    )
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(
        "print('primary_metric: 1.0')\n",
        encoding="utf-8",
    )

    first = ClusterBridgePoolSandbox(
        config,
        tmp_path / "work-one",
        pool_factory=lambda *_args, **_kwargs: fake_pool,
    )
    first.run_project(project, timeout_sec=20)
    first_task_id = str(fake_pool.calls[-1]["task_id"])
    first_env = fake_pool.calls[-1]["env"]

    # A restarted sandbox for the same Factory Work Item derives the same
    # durable idempotency key instead of a fresh random UUID.
    second_root = tmp_path / "state-two"
    fake_pool_two = FakePool(second_root)
    second_root.mkdir()
    second = ClusterBridgePoolSandbox(
        config,
        tmp_path / "work-two",
        pool_factory=lambda *_args, **_kwargs: fake_pool_two,
    )
    second.run_project(project, timeout_sec=20)

    assert str(fake_pool_two.calls[-1]["task_id"]) == first_task_id
    assert first_env["RESEARCHCLAW_FACTORY_ID"] == "factory-a"
    assert first_env["RESEARCHCLAW_IDEA_ID"] == "idea-a"
    assert first_env["RESEARCHCLAW_WORK_ITEM_ATTEMPT"] == "1"
    assert first_env["RESEARCHCLAW_GPU_REQUEST"] == "2"
    assert fake_pool.calls[-1]["num_gpus"] == 2
    assert fake_pool_two.calls[-1]["num_gpus"] == 2


def test_factory_retry_attempt_uses_distinct_pool_task_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text("clusterbridge_pool: {}\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text(
        "print('primary_metric: 1.0')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RESEARCHCLAW_WORK_ITEM_ID", "idea-a-pilot")
    monkeypatch.setenv("RESEARCHCLAW_GPU_REQUEST", "1")

    first_pool = FakePool(tmp_path / "state-one")
    first_pool.state_dir.mkdir()
    monkeypatch.setenv("RESEARCHCLAW_WORK_ITEM_ATTEMPT", "1")
    ClusterBridgePoolSandbox(
        ClusterBridgePoolSandboxConfig(config_file=str(pool_config)),
        tmp_path / "work-one",
        pool_factory=lambda *_args, **_kwargs: first_pool,
    ).run_project(project)

    second_pool = FakePool(tmp_path / "state-two")
    second_pool.state_dir.mkdir()
    monkeypatch.setenv("RESEARCHCLAW_WORK_ITEM_ATTEMPT", "2")
    ClusterBridgePoolSandbox(
        ClusterBridgePoolSandboxConfig(config_file=str(pool_config)),
        tmp_path / "work-two",
        pool_factory=lambda *_args, **_kwargs: second_pool,
    ).run_project(project)

    assert first_pool.calls[-1]["task_id"] != second_pool.calls[-1]["task_id"]


def test_network_isolation_does_not_sever_ray_control_plane(
    tmp_path: Path,
) -> None:
    config = ClusterBridgePoolSandboxConfig(
        config_file=str(tmp_path / "pool.yaml"),
        network_isolation=True,
    )
    sandbox = ClusterBridgePoolSandbox(config, tmp_path / "work")

    command = sandbox._build_task_command(
        Path("/root/shared/run"),
        entry_point="main.py",
        args=None,
    )

    assert "researchclaw-ray-netns" in command
    assert "unshare --net" not in command
