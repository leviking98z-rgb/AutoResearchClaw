from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from researchclaw.autoresearch_v2 import runtime
from researchclaw.autoresearch_v2.config import V2Config
from researchclaw.autoresearch_v2.ideas import StaticIdeaGenerator
from researchclaw.experiment.clusterbridge_pool import PoolLeaseOwnershipError


def _pool_config(path: Path) -> Path:
    path.write_text(
        """
clusterbridge_pool:
  expected_total_gpus: 8
  nodes:
    - address: 10.0.0.1
      ray_ip: 10.0.0.1
      gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_production_controller_starts_without_live_gpu_lease(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Router:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.decision = object()
            self.worker = object()
            self.utility = object()

    pool_config = _pool_config(tmp_path / "pool.yaml")
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path / "runs" / "canary"),
                "models": {
                    "researchclaw_config": str(tmp_path / "unused.yaml"),
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": str(pool_config),
                    "shared_workspace_root": str(tmp_path / "runs"),
                },
            }
        }
    )
    monkeypatch.setattr(runtime, "RoleRouter", _Router)
    monkeypatch.setattr(
        runtime,
        "InfoHubLiteratureProvider",
        lambda config: SimpleNamespace(config=config),
    )
    monkeypatch.setattr(
        runtime,
        "build_clusterbridge_broker",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PoolLeaseOwnershipError("inactive or missing lease")
        ),
    )

    controller = runtime.build_production_controller(config)

    assert controller.gpu_broker is None
    assert controller.configured_gpu_capacity == 8
    events = controller.store.list_events(limit=10)
    unavailable = [
        event
        for event in events
        if event["event_type"] == "gpu_broker_unavailable"
    ]
    assert len(unavailable) == 1
    assert unavailable[0]["configured_gpu_capacity"] == 8
    assert "inactive or missing lease" in unavailable[0]["error"]
    assert controller.snapshot()["gpu"]["state"] == "unavailable"
    controller.initialize()
    controller.close()


def test_invalid_pool_configuration_is_not_silently_degraded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Router:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.decision = object()
            self.worker = object()
            self.utility = object()

    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path / "runs" / "canary"),
                "gpu": {
                    "enabled": True,
                    "pool_config": str(tmp_path / "missing-pool.yaml"),
                    "shared_workspace_root": str(tmp_path / "runs"),
                },
            }
        }
    )
    monkeypatch.setattr(runtime, "RoleRouter", _Router)

    try:
        runtime.build_production_controller(config)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("invalid pool configuration must fail closed")


def test_production_gpu_tasks_do_not_inherit_forced_offline_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    class _Router:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.decision = object()
            self.worker = object()
            self.utility = object()

    captured: dict[str, object] = {}

    def build_broker(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace()

    pool_config = _pool_config(tmp_path / "pool.yaml")
    config = V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(tmp_path / "runs" / "canary"),
                "models": {
                    "researchclaw_config": str(tmp_path / "unused.yaml"),
                },
                "execution": {
                    "allowed_env_keys": [
                        "HF_TOKEN",
                        "HF_HUB_OFFLINE",
                        "TRANSFORMERS_OFFLINE",
                    ]
                },
                "gpu": {
                    "enabled": True,
                    "pool_config": str(pool_config),
                    "shared_workspace_root": str(tmp_path / "runs"),
                },
            }
        }
    )
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "true")
    monkeypatch.setattr(runtime, "RoleRouter", _Router)
    monkeypatch.setattr(runtime, "build_clusterbridge_broker", build_broker)

    controller = runtime.build_production_controller(
        config,
        generator=StaticIdeaGenerator([]),
    )

    assert captured["task_env"] == {"HF_TOKEN": "test-token"}
    controller._pool.shutdown(wait=True)
    controller._idea_pool.shutdown(wait=True)
