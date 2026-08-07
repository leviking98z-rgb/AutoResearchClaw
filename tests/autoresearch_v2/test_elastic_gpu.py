from __future__ import annotations

import json
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
    min_gpus: int = 0,
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
        self.cancellations: list[str] = []

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

    def cancel(self, request_id: str) -> None:
        self.cancellations.append(request_id)
        self.queue = [
            item for item in self.queue if item.get("id") != request_id
        ]


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
        self.restored = 0
        self.pause_spin_calls = 0
        self.spin_calls = 0
        self.stop_keepalive_calls = 0

    def adopt_claimed_lease(self) -> None:
        self.adopted += 1
        self.claimed = True

    def _write_state(self) -> None:
        return None

    def prepare(self) -> None:
        self.prepared_calls += 1
        self.claimed = True
        self.prepared = True

    def restore_allocated_state(self, data: dict[str, Any]) -> None:
        self.restored += 1
        self.claimed = bool(data.get("claimed"))
        self.prepared = bool(data.get("prepared"))

    def pause_spin(self) -> None:
        self.pause_spin_calls += 1

    def spin(self) -> None:
        self.spin_calls += 1

    def stop_keepalive(self) -> None:
        self.stop_keepalive_calls += 1


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
    ) == (0, 16, 32)
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


def test_resource_manager_projects_cache_prepare_policy(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
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
        cache_dir="/data/cache/autoresearch-v2/huggingface",
        cache_archive="/root/sync/autoresearch-cache.tar",
    )

    manager.bootstrap(required_gpus=1)

    pool_config = pools.pools[0].config
    assert (
        pool_config.prepare_cache_dir
        == "/data/cache/autoresearch-v2/huggingface"
    )
    assert (
        pool_config.prepare_cache_archive
        == "/root/sync/autoresearch-cache.tar"
    )


def test_resource_manager_passes_task_namespace_to_broker(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    client.allocations = [_allocation()]
    brokers = _BrokerFactory()
    manager = ResourceManagedGPUManager(
        config.gpu,
        client=client,
        pool_factory=_PoolFactory(),
        broker_factory=brokers,
        monotonic=lambda: 0.0,
        prepare_async=False,
        task_namespace="rsi-canary-seven",
    )

    manager.bootstrap(required_gpus=1)

    assert brokers.calls[0]["task_namespace"] == "rsi-canary-seven"
    assert brokers.calls[0]["manage_pool_keepalive"] is False


def test_resource_manager_can_pin_exact_allocation(tmp_path: Path) -> None:
    base = _config(tmp_path)
    raw = {
        "autoresearch_v2": {
            "enabled": True,
            "state_dir": str(tmp_path / "runs" / "elastic-test"),
            "gpu": {
                "enabled": True,
                "mode": "resource_manager",
                "shared_workspace_root": str(tmp_path / "runs"),
                "resource_manager": {
                    **{
                        name: getattr(base.gpu.resource_manager, name)
                        for name in (
                            "owner",
                            "cb_command",
                            "project",
                            "purpose",
                            "min_gpus",
                            "desired_gpus",
                            "max_gpus",
                            "duration_min",
                            "renew_ttl_min",
                            "renew_interval_sec",
                            "reconcile_interval_sec",
                            "allow_cross_cluster",
                            "gpu_type",
                            "priority",
                            "release_on_shutdown",
                            "log_root",
                            "ray_command",
                            "ray_python",
                            "ray_port",
                            "command_timeout_sec",
                            "prepare_timeout_sec",
                        )
                    },
                    "preferred_allocation_id": "alloc-preferred",
                },
            },
        }
    }
    config = V2Config.from_mapping(raw)
    client = _FakeResourceClient()
    client.allocations = [
        _allocation(allocation_id="alloc-other", gpus=32),
        _allocation(allocation_id="alloc-preferred", gpus=16),
    ]
    manager, pools, _ = _manager(
        config.gpu,
        client=client,
        clock=[0.0],
    )

    manager.bootstrap(required_gpus=1)

    assert manager.snapshot()["allocation_id"] == "alloc-preferred"
    assert pools.pools[0].config.expected_total_gpus == 16


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


def test_resource_manager_mode_rejects_non_positive_maximum(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="max_gpus"):
        _config(
            tmp_path,
            max_gpus=0,
        )


def test_bootstrap_without_demand_does_not_request_capacity(
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

    assert client.requests == []
    assert manager.broker is None
    assert manager.configured_capacity == 0
    assert manager.snapshot()["state"] == "idle"
    assert manager.snapshot()["request_pending"] is False
    assert pools.pools == []
    assert brokers.calls == []


@pytest.mark.parametrize(
    ("required_gpus", "expected_gpus"),
    [(1, 1), (9, 9), (17, 17), (99, 32)],
)
def test_demand_requests_rounded_capacity_when_no_allocation_exists(
    tmp_path: Path,
    required_gpus: int,
    expected_gpus: int,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    manager, pools, brokers = _manager(
        config.gpu,
        client=client,
        clock=[0.0],
    )

    manager.reconcile(
        required_gpus=required_gpus,
        pending_gpu_jobs=5,
        force=True,
    )

    assert client.requests == [
        {
            "project": "AutoResearch",
            "purpose": "elastic unit test",
            "gpus": expected_gpus,
            "duration_min": 120,
            "allow_cross_cluster": True,
            "gpu_type": "H20",
            "priority": "high",
        }
    ]
    assert manager.broker is None
    assert manager.configured_capacity == 0
    assert manager.snapshot()["state"] in {
        "requesting",
        "waiting_allocation",
    }
    assert manager.snapshot()["request_pending"] is True
    assert pools.pools == []
    assert brokers.calls == []


def test_demand_drop_cancels_queued_request(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    manager, _, _ = _manager(
        config.gpu,
        client=client,
        clock=[0.0],
    )
    manager.reconcile(required_gpus=2, force=True)

    manager.reconcile(required_gpus=0, force=True)

    assert client.cancellations == ["req-1"]
    assert client.queue == []
    assert manager.snapshot()["state"] == "idle"
    assert manager.snapshot()["request_pending"] is False


def test_invisible_submitted_request_is_not_duplicated_within_grace(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class _LaggingClient(_FakeResourceClient):
        def request(self, **kwargs: Any) -> None:
            self.requests.append(dict(kwargs))

    client = _LaggingClient()
    clock = [0.0]
    manager, _, _ = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )

    manager.reconcile(required_gpus=6, force=True)
    for now in (2.0, 15.0, 59.9):
        clock[0] = now
        manager.reconcile(required_gpus=6, force=True)

    assert len(client.requests) == 1
    assert manager.snapshot()["request_pending"] is True
    assert manager.snapshot()["requested_gpus"] == 6


def test_invisible_submitted_request_retries_after_grace(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    class _LaggingClient(_FakeResourceClient):
        def request(self, **kwargs: Any) -> None:
            self.requests.append(dict(kwargs))

    client = _LaggingClient()
    clock = [0.0]
    manager, _, _ = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )

    manager.reconcile(required_gpus=6, force=True)
    clock[0] = 60.0
    manager.reconcile(required_gpus=6, force=True)

    assert len(client.requests) == 2
    assert manager.snapshot()["request_pending"] is True


def test_recently_visible_queue_is_protected_when_snapshot_turns_empty(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    client.queue = [
        {
            "id": "req-existing",
            "owner": OWNER,
            "project": "AutoResearch",
            "status": "queued",
        }
    ]
    clock = [10.0]
    manager, _, _ = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )

    manager.reconcile(required_gpus=6, force=True)
    client.queue = []
    clock[0] = 20.0
    manager.reconcile(required_gpus=6, force=True)

    assert client.requests == []
    assert manager.snapshot()["request_pending"] is True


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
    manager.bootstrap(required_gpus=1)
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
    assert manager.snapshot()["request_pending"] is False


def test_existing_allocated_pool_state_skips_per_node_claim_restore(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    allocation = _allocation()
    client.allocations = [allocation]
    pools = _PoolFactory()
    brokers = _BrokerFactory()
    pool_id = "autoresearch-v2-alloc-1"
    state_dir = Path(config.gpu.resource_manager.log_root) / pool_id
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text(
        json.dumps(
            {
                "pool_id": pool_id,
                "nodes": [
                    {"address": item["ip"]}
                    for item in allocation["node_details"]
                ],
                "claimed": True,
                "prepared": True,
                "ray_started": True,
            }
        ),
        encoding="utf-8",
    )
    manager = ResourceManagedGPUManager(
        config.gpu,
        client=client,
        pool_factory=pools,
        broker_factory=brokers,
        monotonic=lambda: 0.0,
        prepare_async=False,
    )

    manager.bootstrap(required_gpus=1)

    pool = pools.pools[0]
    assert pool.restored == 1
    assert pool.adopted == 0
    assert pool.prepared_calls == 0
    assert pool.stop_keepalive_calls == 1
    assert pool.pause_spin_calls == 0
    assert pool.spin_calls == 0
    assert manager.broker is brokers.brokers[0]


def test_attached_pool_does_not_manage_cluster_spin(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    client.allocations = [_allocation()]
    clock = [0.0]
    manager, pools, _ = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )

    manager.bootstrap(required_gpus=1)
    pool = pools.pools[0]
    assert pool.pause_spin_calls == 0
    assert pool.spin_calls == 0

    clock[0] = 2.0
    manager.reconcile(required_gpus=1, running_gpu_jobs=1)
    assert pool.pause_spin_calls == 0
    assert pool.spin_calls == 0


def test_idle_allocation_releases_immediately(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    client.allocations = [_allocation()]
    clock = [0.0]
    manager, pools, _ = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )
    manager.bootstrap(required_gpus=1)
    broker = manager.broker
    pool = pools.pools[0]

    clock[0] = 2.0
    manager.reconcile(required_gpus=0, pending_gpu_jobs=0)
    assert client.releases == ["alloc-1"]
    assert pool.spin_calls == 0
    assert broker is not None
    assert broker.closed == 1
    assert manager.broker is None
    assert manager.configured_capacity == 0
    assert manager.snapshot()["state"] == "idle"


def test_durable_running_gpu_job_prevents_idle_release(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    client.allocations = [_allocation()]
    clock = [0.0]
    manager, _, _ = _manager(
        config.gpu,
        client=client,
        clock=clock,
    )
    manager.bootstrap(required_gpus=1)
    broker = manager.broker

    clock[0] = 2.0
    manager.reconcile(
        required_gpus=0,
        pending_gpu_jobs=0,
        running_gpu_jobs=1,
    )

    assert broker is not None
    assert manager.broker is broker
    assert broker.closed == 0
    assert client.releases == []
    assert manager.snapshot()["running_gpu_jobs"] == 1


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

    manager.bootstrap(required_gpus=1)

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

    manager.bootstrap(required_gpus=1)

    assert started.wait(timeout=1)
    assert manager.broker is None
    assert manager.configured_capacity == 0
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
    manager.bootstrap(required_gpus=1)
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


def test_duplicate_owned_allocations_are_released(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = _FakeResourceClient()
    client.allocations = [
        _allocation(allocation_id="alloc-small", gpus=8),
        _allocation(allocation_id="alloc-large", gpus=16),
    ]
    manager, _, _ = _manager(
        config.gpu,
        client=client,
        clock=[0.0],
    )

    manager.bootstrap(required_gpus=1)

    assert manager.snapshot()["allocation_id"] == "alloc-large"
    assert client.releases == ["alloc-small"]


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
    manager.bootstrap(required_gpus=1)
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

        def reconcile(self, **kwargs: Any) -> bool:
            self.reconciles += 1
            self.demand = dict(kwargs)
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

    assert manager.reconciles == 2
    assert manager.demand == {
        "required_gpus": 0,
        "pending_gpu_jobs": 0,
        "running_gpu_jobs": 0,
    }
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
