from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from researchclaw.autoresearch_v2 import runtime
from researchclaw.autoresearch_v2.config import V2Config
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
