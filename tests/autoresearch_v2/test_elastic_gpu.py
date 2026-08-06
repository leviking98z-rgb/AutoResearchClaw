from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.autoresearch_v2.config import GPUConfig, V2Config
from researchclaw.autoresearch_v2.controller import V2Controller
from researchclaw.autoresearch_v2.elastic_gpu import (
    ResourceManagedGPUManager,
)
from researchclaw.autoresearch_v2.ideas import StaticIdeaGenerator
from researchclaw.autoresearch_v2.store import V2Store

OWNER = "11111111-1111-4111-8111-111111111111"


def _config(
    root: Path,
    *,
    release_on_shutdown: bool = False,
    min_gpus: int = 8,
    desired_gpus: int = 16,
    max_gpus: int = 32,
) -> V2Config:
    return V2Config.from_mapping(
        {
            "autoresearch_v2": {
                "enabled": True,
                "state_dir": str(root / "runs" / "elastic-test"),
                "gpu": {
                    "enabled": True,
                    "mode": "resource_manager",
                    # Deliberately absent: elastic mode must not require a
                    # prewritten fixed-pool YAML.
                    "pool_config": "",
                    "shared_workspace_root": str(root / "runs"),
                    "resource_manager": {
                        "owner": OWNER,
                        "cb_command": "/fake/cb",
                        "project": "AutoResearch",
                        "purpose": "elastic unit test",
                        "min_gpus": min_gpus,
                        "desired_gpus": desired_gpus,
                        "max_gpus": max_gpus,
                        "duration_min": 120,
                        "renew_ttl_min": 120,
                        "renew_interval_sec": 30,
                        "reconcile_interval_sec": 1,
                        "allow_cross_cluster": True,
                        "gpu_type": "H20",
                        "priority": "high",
                        "release_on_shutdown": release_on_shutdown,
                        "log_root": str(root / "pool-logs"),
                    },
                },
            }
        }
    )


def _allocation(
    *,
    allocation_id: str = "alloc-1",
    gpus: int = 16,
) -> dict[str, Any]:
    nodes = []
    remaining = gpus
    index = 1
    while remaining:
        count = min(8, remaining)
        nodes.append(
            {
                "ip": f"10.0.0.{index}",
                "gpus": count,
            }
        )
        remaining -= count
        index += 1
    return {
        "id": allocation_id,
        "owner": OWNER,
        "project": "AutoResearch",
        "status": "active",
        "gpu_count": gpus,
        "nodes": [item["ip"] for item in nodes],
        "node_details": nodes,
    }


class _FakeResourceClient:
    def __init__(self) -> None:
        self.allocations: list[dict[str, Any]] = []
        self.queue: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.renewals: list[tuple[str, int]] = []
        self.releases: list[str] = []

    def snapshot(self) -> dict[str, Any]:
        return {
            "allocations": [dict(item) for item in self.allocations],
            "queue": [dict(item) for item in self.queue],
        }

    def request(self, **kwargs: Any) -> None:
        self.requests.append(dict(kwargs))
        self.queue = [
            {
                "id": "req-1",
                "owner": OWNER,
                "project": str(kwargs["project"]),
                "status": "queued",
            }
        ]

    def renew(self, allocation_id: str, *, ttl_min: int) -> None:
        self.renewals.append((allocation_id, ttl_min))

    def release(self, allocation_id: str) -> None:
        self.releases.append(allocation_id)


class _FakePool:
    def __init__(self, pool_config: Any, **kwargs: Any) -> None:
        self.config = pool_config
        self.kwargs = kwargs
        self.claimed = False
        self.prepared = False
        self.state_dir = Path(pool_config.log_root) / pool_config.pool_id
        self._state_lock = threading.RLock()
        self.adopted = 0
        self.prepared_calls = 0

    def adopt_claimed_lease(self) -> None:
        self.adopted += 1
        self.claimed = True

    def _write_state(self) -> None:
        return None

    def prepare(self) -> None:
        self.prepared_calls += 1
        self.claimed = True
        self.prepared = True


class _FakeBroker:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.closed = 0
        self.leases: dict[str, Any] = {}
        self.scheduler = SimpleNamespace(order=lambda jobs, **kwargs: jobs)

    def close(self) -> None:
        self.closed += 1

    def reconcile(self) -> list[tuple[str, dict[str, Any]]]:
        return []

    def snapshot(self, *, pending_jobs: int = 0) -> dict[str, Any]:
        return {
            "total_gpus": self.capacity,
            "allocated_gpus": 0,
            "available_gpus": self.capacity,
            "utilization": 0.0,
            "target_utilization": 0.9,
            "pending_jobs": pending_jobs,
            "leases": [],
        }


class _PoolFactory:
    def __init__(self) -> None:
        self.pools: list[_FakePool] = []

    def __call__(self, pool_config: Any, **kwargs: Any) -> _FakePool:
        pool = _FakePool(pool_config, **kwargs)
        self.pools.append(pool)
        return pool


class _BrokerFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.brokers: list[_FakeBroker] = []

    def __call__(self, pool: Any, **kwargs: Any) -> _FakeBroker:
        self.calls.append({"pool": pool, **kwargs})
        broker = _FakeBroker(int(kwargs["total_gpus"]))
        self.brokers.append(broker)
        return broker


def _manager(
    config: GPUConfig,
    *,
    client: _FakeResourceClient,
    clock: list[float],
    pool_factory: _PoolFactory | None = None,
    broker_factory: _BrokerFactory | None = None,
) -> tuple[
    ResourceManagedGPUManager,
    _PoolFactory,
    _BrokerFactory,
]:
    pools = pool_factory or _PoolFactory()
    brokers = broker_factory or _BrokerFactory()
    manager = ResourceManagedGPUManager(
        config,
        client=client,
        pool_factory=pools,
        broker_factory=brokers,
        monotonic=lambda: clock[0],
        prepare_async=False,
    )
    return manager, pools, brokers


def test_resource_manager_mode_parses_full_policy_without_pool_config(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    assert config.gpu.mode == "resource_manager"
    assert config.gpu.pool_config == ""
    assert config.gpu.resource_manager.owner == OWNER
    assert config.gpu.resource_manager.cb_command == "/fake/cb"
    assert config.gpu.resource_manager.project == "AutoResearch"
    assert config.gpu.resource_manager.purpose == "elastic unit test"
    assert (
        config.gpu.resource_manager.min_gpus,
        config.gpu.resource_manager.desired_gpus,
        config.gpu.resource_manager.max_gpus,
    ) == (8, 16, 32)
    assert config.gpu.resource_manager.duration_min == 120
    assert config.gpu.resource_manager.renew_ttl_min == 120
    assert config.gpu.resource_manager.renew_interval_sec == 30
    assert config.gpu.resource_manager.reconcile_interval_sec == 1
    assert config.gpu.resource_manager.allow_cross_cluster is True
    assert config.gpu.resource_manager.gpu_type == "H20"
    assert config.gpu.resource_manager.priority == "high"
    assert config.gpu.resource_manager.release_on_shutdown is False
    assert config.gpu.resource_manager.log_root == str(
        tmp_path / "pool-logs"
    )


def test_resource_manager_exposes_declared_controller_api() -> None:
    assert callable(getattr(ResourceManagedGPUManager, "bootstrap", None))
    assert callable(getattr(ResourceManagedGPUManager, "reconcile", None))
    assert callable(getattr(ResourceManagedGPUManager, "close", None))
    assert isinstance(ResourceManagedGPUManager.broker, property)
    assert isinstance(
        ResourceManagedGPUManager.configured_capacity,
        property,
    )
    assert callable(getattr(ResourceManagedGPUManager, "snapshot", None))


@pytest.mark.parametrize(
    ("minimum", "desired", "maximum"),
    [(0, 8, 16), (16, 8, 32), (8, 64, 32)],
)
def test_resource_manager_mode_rejects_invalid_capacity_bounds(
    tmp_path: Path,
    minimum: int,
    desired: int,
    maximum: int,
) -> None:
    with pytest.raises(ValueError, match="GPU bounds"):
        _config(
            tmp_path,
            min_gpus=minimum,
            desired_gpus=desired,
            max_gpus=maximum,
        )


def test_bootstrap_requests_desired_capacity_when_no_allocation_exists(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    manager, pools, brokers = _manager(
        config.gpu,
        client=client,
        clock=[0.0],
    )

    manager.bootstrap()

    assert client.requests == [
        {
            "project": "AutoResearch",
            "purpose": "elastic unit test",
            "gpus": 16,
            "duration_min": 120,
            "allow_cross_cluster": True,
            "gpu_type": "H20",
            "priority": "high",
        }
    ]
    assert manager.broker is None
    assert manager.configured_capacity == 16
    assert manager.snapshot()["state"] in {
        "requesting",
        "waiting_allocation",
    }
    assert manager.snapshot()["request_pending"] is True
    assert pools.pools == []
    assert brokers.calls == []


def test_later_reconcile_hot_attaches_granted_allocation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    clock = [0.0]
    manager, pools, brokers = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )
    manager.bootstrap()
    client.queue = []
    client.allocations = [_allocation(gpus=16)]
    clock[0] = 2.0

    changed = manager.reconcile()

    assert changed is True
    assert manager.broker is brokers.brokers[0]
    assert brokers.calls[0]["total_gpus"] == 16
    assert pools.pools[0].config.expected_total_gpus == 16
    assert pools.pools[0].config.expected_claim_owner == OWNER
    assert pools.pools[0].adopted == 1
    assert pools.pools[0].prepared_calls == 1
    assert client.renewals == [("alloc-1", 120)]
    assert manager.snapshot()["state"] == "ready"
    assert manager.snapshot()["allocated_gpus"] == 16


def test_renewal_does_not_block_reconcile(tmp_path: Path) -> None:
    config = _config(tmp_path)
    started = threading.Event()
    release = threading.Event()

    class _BlockingClient(_FakeResourceClient):
        def renew(self, allocation_id: str, *, ttl_min: int) -> None:
            started.set()
            release.wait(timeout=2)
            super().renew(allocation_id, ttl_min=ttl_min)

    client = _BlockingClient()
    client.allocations = [_allocation()]
    pools = _PoolFactory()
    brokers = _BrokerFactory()
    manager = ResourceManagedGPUManager(
        config.gpu,
        client=client,
        pool_factory=pools,
        broker_factory=brokers,
        monotonic=lambda: 0.0,
        prepare_async=False,
    )

    manager.bootstrap()

    assert started.wait(timeout=1)
    assert manager.broker is brokers.brokers[0]
    release.set()
    manager.close()


def test_async_resource_snapshot_never_blocks_status_reads(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    started = threading.Event()
    release = threading.Event()

    class _BlockingClient(_FakeResourceClient):
        def snapshot(self) -> dict[str, Any]:
            started.set()
            release.wait(timeout=2)
            return super().snapshot()

    client = _BlockingClient()
    manager = ResourceManagedGPUManager(
        config.gpu,
        client=client,
        pool_factory=_PoolFactory(),
        broker_factory=_BrokerFactory(),
        monotonic=lambda: 0.0,
        prepare_async=True,
    )

    manager.bootstrap()

    assert started.wait(timeout=1)
    assert manager.broker is None
    assert manager.configured_capacity == 16
    assert manager.snapshot()["state"] == "starting"
    release.set()
    manager.close()


def test_allocation_change_replaces_broker_and_updates_capacity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    client.allocations = [_allocation(allocation_id="alloc-1", gpus=16)]
    clock = [0.0]
    manager, _, brokers = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )
    manager.bootstrap()
    first = manager.broker
    client.allocations = [_allocation(allocation_id="alloc-2", gpus=24)]
    clock[0] = 2.0

    changed = manager.reconcile()

    assert changed is True
    assert first is not None
    assert first.closed == 1
    assert manager.broker is brokers.brokers[1]
    assert brokers.calls[1]["total_gpus"] == 24
    assert manager.configured_capacity == 24
    assert manager.snapshot()["allocation_id"] == "alloc-2"
    assert manager.snapshot()["allocated_gpus"] == 24


@pytest.mark.parametrize(
    ("release_on_shutdown", "expected_releases"),
    [(False, []), (True, ["alloc-1"])],
)
def test_close_obeys_release_policy(
    tmp_path: Path,
    release_on_shutdown: bool,
    expected_releases: list[str],
) -> None:
    config = _config(
        tmp_path,
        release_on_shutdown=release_on_shutdown,
    )
    client = _FakeResourceClient()
    client.allocations = [_allocation()]
    manager, _, _ = _manager(
        config.gpu,
        client=client,
        clock=[0.0],
    )
    manager.bootstrap()
    broker = manager.broker

    manager.close()
    manager.close()

    assert broker is not None
    assert broker.closed == 1
    assert client.releases == expected_releases
    assert manager.broker is None
    assert manager.snapshot()["state"] == "closed"


def test_controller_tick_hot_syncs_manager_broker_and_capacity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    broker = _FakeBroker(16)

    class _FakeManager:
        configured_capacity = 16

        def __init__(self) -> None:
            self.broker = None
            self.reconciles = 0
            self.closed = 0

        def reconcile(self) -> bool:
            self.reconciles += 1
            self.broker = broker
            self.configured_capacity = 24
            return True

        def snapshot(self) -> dict[str, Any]:
            return {
                "mode": "resource_manager",
                "state": "ready",
                "allocated_gpus": 16,
            }

        def close(self) -> None:
            self.closed += 1

    manager = _FakeManager()
    store = V2Store(config.root)
    controller = V2Controller(
        config=config,
        store=store,
        generator=StaticIdeaGenerator([]),
        gpu_manager=manager,
        configured_gpu_capacity=manager.configured_capacity,
        sleep=lambda _: None,
    )
    controller.initialize()
    store.set_control("pause", "isolate elastic reconciliation")

    snapshot = controller.tick()

    assert manager.reconciles == 1
    assert controller.gpu_broker is broker
    assert controller.configured_gpu_capacity == 24
    assert snapshot["gpu"]["total_gpus"] == 16
    assert snapshot["gpu"]["elastic"]["state"] == "ready"
    assert any(
        event["event_type"] == "gpu_broker_reconnected"
        for event in store.list_events(limit=20)
    )
    controller.close()
    assert manager.closed == 1
