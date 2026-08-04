"""Configuration models for a multi-node ClusterBridge pool.

This module deliberately does not depend on :mod:`researchclaw.config`.  The
pool is an optional execution layer that can be wired into the main
ResearchClaw configuration once its public API has settled.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_NODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}$")


@dataclass(frozen=True, slots=True)
class ClusterNode:
    """One ClusterBridge node and the GPUs assigned to this pool."""

    address: str
    gpu_ids: tuple[int, ...]
    ray_ip: str | None = None

    def __post_init__(self) -> None:
        address = self.address.strip()
        if not address or not _SAFE_NODE.fullmatch(address):
            raise ValueError(f"invalid ClusterBridge node address: {self.address!r}")
        object.__setattr__(self, "address", address)

        gpu_ids = tuple(int(value) for value in self.gpu_ids)
        if not gpu_ids:
            raise ValueError(f"node {address} must assign at least one GPU")
        if any(value < 0 for value in gpu_ids):
            raise ValueError(f"node {address} has a negative GPU id")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError(f"node {address} has duplicate GPU ids")
        object.__setattr__(self, "gpu_ids", gpu_ids)

        ray_ip = (self.ray_ip or address).strip()
        if not ray_ip or not _SAFE_NODE.fullmatch(ray_ip):
            raise ValueError(f"invalid Ray node address: {ray_ip!r}")
        object.__setattr__(self, "ray_ip", ray_ip)

    @property
    def gpu_count(self) -> int:
        return len(self.gpu_ids)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", self.address)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ClusterNode:
        address = value.get("address", value.get("node", value.get("ip", "")))
        gpu_ids = value.get("gpu_ids", value.get("gpus", ()))
        if isinstance(gpu_ids, int):
            gpu_ids = tuple(range(gpu_ids))
        if not isinstance(gpu_ids, Sequence) or isinstance(gpu_ids, (str, bytes)):
            raise TypeError("node gpu_ids must be a sequence of integers")
        return cls(
            address=str(address),
            gpu_ids=tuple(int(item) for item in gpu_ids),
            ray_ip=str(value["ray_ip"]) if value.get("ray_ip") else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "gpu_ids": list(self.gpu_ids),
            "ray_ip": self.ray_ip,
        }


@dataclass(frozen=True, slots=True)
class RayPoolConfig:
    """Ray process and readiness settings."""

    command: str = "ray"
    python: str = "python3"
    head_node: str | None = None
    port: int = 6379
    start_timeout_sec: float = 120.0
    resource_timeout_sec: float = 180.0
    poll_interval_sec: float = 2.0
    stop_timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        if not self.command.strip():
            raise ValueError("ray.command must not be empty")
        if not self.python.strip():
            raise ValueError("ray.python must not be empty")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("ray.port must be between 1 and 65535")
        for name in (
            "start_timeout_sec",
            "resource_timeout_sec",
            "poll_interval_sec",
            "stop_timeout_sec",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"ray.{name} must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RayPoolConfig:
        raw = dict(value or {})
        return cls(
            command=str(raw.get("command", "ray")),
            python=str(raw.get("python", "python3")),
            head_node=(
                str(raw["head_node"]).strip() if raw.get("head_node") else None
            ),
            port=int(raw.get("port", 6379)),
            start_timeout_sec=float(raw.get("start_timeout_sec", 120.0)),
            resource_timeout_sec=float(raw.get("resource_timeout_sec", 180.0)),
            poll_interval_sec=float(raw.get("poll_interval_sec", 2.0)),
            stop_timeout_sec=float(raw.get("stop_timeout_sec", 30.0)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "python": self.python,
            "head_node": self.head_node,
            "port": self.port,
            "start_timeout_sec": self.start_timeout_sec,
            "resource_timeout_sec": self.resource_timeout_sec,
            "poll_interval_sec": self.poll_interval_sec,
            "stop_timeout_sec": self.stop_timeout_sec,
        }


@dataclass(frozen=True, slots=True)
class ClusterBridgePoolConfig:
    """Complete configuration for a claimed multi-node execution pool."""

    nodes: tuple[ClusterNode, ...]
    cb_command: str = "/root/shared/.clusters/.tools/clusterbridge.sh"
    purpose: str = "AutoResearchClaw multi-node experiments"
    pool_id: str = "autoresearch"
    log_root: str = "/root/shared/.clusters/.tmp/autoresearch-cluster-pool"
    claim_ttl_min: int = 240
    renew_interval_sec: float = 3600.0
    max_renew_failures: int = 3
    command_timeout_sec: float = 120.0
    parallelism: int = 8
    expected_total_gpus: int | None = None
    allow_force_claim: bool = False
    expected_claim_owner: str | None = None
    node_cleanup_script: str = (
        "/root/shared/.clusters/.tools/node_cleanup.sh"
    )
    node_spin_script: str = "/root/shared/.clusters/.tools/node_spin.sh"
    task_kill_grace_sec: int = 30
    ray: RayPoolConfig = field(default_factory=RayPoolConfig)

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        if not nodes:
            raise ValueError("cluster pool must contain at least one node")
        addresses = [node.address for node in nodes]
        if len(set(addresses)) != len(addresses):
            raise ValueError("cluster pool contains duplicate node addresses")
        ray_ips = [str(node.ray_ip) for node in nodes]
        if len(set(ray_ips)) != len(ray_ips):
            raise ValueError("cluster pool contains duplicate Ray node addresses")
        object.__setattr__(self, "nodes", nodes)

        if not _SAFE_ID.fullmatch(self.pool_id):
            raise ValueError(
                "pool_id must contain only letters, digits, '.', '_' or '-'"
            )
        if not self.purpose.strip():
            raise ValueError("purpose must not be empty")
        expected_claim_owner = (
            str(self.expected_claim_owner).strip()
            if self.expected_claim_owner is not None
            else None
        )
        if expected_claim_owner == "${AUTORESEARCH_CLAIM_OWNER}":
            expected_claim_owner = (
                os.environ.get("AUTORESEARCH_CLAIM_OWNER")
                or os.environ.get("CODEX_THREAD_ID")
                or os.environ.get("CB_SID")
                or None
            )
        if expected_claim_owner == "":
            expected_claim_owner = None
        if expected_claim_owner is not None and not _SAFE_ID.fullmatch(
            expected_claim_owner
        ):
            raise ValueError(
                "expected_claim_owner must contain only letters, digits, "
                "'.', '_' or '-'"
            )
        object.__setattr__(self, "expected_claim_owner", expected_claim_owner)
        if int(self.claim_ttl_min) <= 0:
            raise ValueError("claim_ttl_min must be positive")
        if float(self.renew_interval_sec) <= 0:
            raise ValueError("renew_interval_sec must be positive")
        if self.renew_interval_sec >= self.claim_ttl_min * 60:
            raise ValueError("renew_interval_sec must be shorter than claim TTL")
        if int(self.max_renew_failures) <= 0:
            raise ValueError("max_renew_failures must be positive")
        if float(self.command_timeout_sec) <= 0:
            raise ValueError("command_timeout_sec must be positive")
        if int(self.parallelism) <= 0:
            raise ValueError("parallelism must be positive")
        if int(self.task_kill_grace_sec) <= 0:
            raise ValueError("task_kill_grace_sec must be positive")
        if not self.cb_command.strip():
            raise ValueError("cb_command must not be empty")
        if not self.node_cleanup_script.strip():
            raise ValueError("node_cleanup_script must not be empty")
        if not self.node_spin_script.strip():
            raise ValueError("node_spin_script must not be empty")

        expected = (
            self.configured_gpu_count
            if self.expected_total_gpus is None
            else int(self.expected_total_gpus)
        )
        if expected <= 0:
            raise ValueError("expected_total_gpus must be positive")
        if expected != self.configured_gpu_count:
            raise ValueError(
                "expected_total_gpus does not match the GPU IDs assigned to nodes "
                f"({expected} != {self.configured_gpu_count})"
            )
        object.__setattr__(self, "expected_total_gpus", expected)

        if self.ray.head_node and self.ray.head_node not in addresses:
            raise ValueError(
                f"ray.head_node {self.ray.head_node!r} is not in the node pool"
            )

    @property
    def configured_gpu_count(self) -> int:
        return sum(node.gpu_count for node in self.nodes)

    @property
    def head_node(self) -> ClusterNode:
        if self.ray.head_node:
            return next(
                node for node in self.nodes if node.address == self.ray.head_node
            )
        return self.nodes[0]

    @property
    def worker_nodes(self) -> tuple[ClusterNode, ...]:
        head = self.head_node
        return tuple(node for node in self.nodes if node.address != head.address)

    @property
    def state_dir(self) -> Path:
        return Path(self.log_root).expanduser().resolve() / self.pool_id

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> ClusterBridgePoolConfig:
        raw: Mapping[str, Any] = value
        if isinstance(value.get("cluster_pool"), Mapping):
            raw = value["cluster_pool"]  # type: ignore[assignment]
        elif isinstance(value.get("clusterbridge_pool"), Mapping):
            raw = value["clusterbridge_pool"]  # type: ignore[assignment]

        node_values = raw.get("nodes", ())
        if not isinstance(node_values, Sequence) or isinstance(
            node_values, (str, bytes)
        ):
            raise TypeError("nodes must be a sequence")
        nodes = tuple(
            ClusterNode.from_mapping(item)
            for item in node_values
            if isinstance(item, Mapping)
        )
        if len(nodes) != len(node_values):
            raise ValueError("each nodes entry must be a mapping")

        expected_raw = raw.get("expected_total_gpus")
        allow_force_claim = raw.get("allow_force_claim", False)
        if not isinstance(allow_force_claim, bool):
            raise TypeError("allow_force_claim must be a YAML boolean")
        return cls(
            nodes=nodes,
            cb_command=str(
                raw.get(
                    "cb_command",
                    "/root/shared/.clusters/.tools/clusterbridge.sh",
                )
            ),
            purpose=str(
                raw.get(
                    "purpose",
                    "AutoResearchClaw multi-node experiments",
                )
            ),
            pool_id=str(raw.get("pool_id", "autoresearch")),
            log_root=str(
                raw.get(
                    "log_root",
                    "/root/shared/.clusters/.tmp/autoresearch-cluster-pool",
                )
            ),
            claim_ttl_min=int(raw.get("claim_ttl_min", 240)),
            renew_interval_sec=float(raw.get("renew_interval_sec", 3600.0)),
            max_renew_failures=int(raw.get("max_renew_failures", 3)),
            command_timeout_sec=float(raw.get("command_timeout_sec", 120.0)),
            parallelism=int(raw.get("parallelism", 8)),
            expected_total_gpus=(
                int(expected_raw) if expected_raw is not None else None
            ),
            allow_force_claim=allow_force_claim,
            expected_claim_owner=(
                str(raw["expected_claim_owner"])
                if raw.get("expected_claim_owner") is not None
                else None
            ),
            node_cleanup_script=str(
                raw.get(
                    "node_cleanup_script",
                    "/root/shared/.clusters/.tools/node_cleanup.sh",
                )
            ),
            node_spin_script=str(
                raw.get(
                    "node_spin_script",
                    "/root/shared/.clusters/.tools/node_spin.sh",
                )
            ),
            task_kill_grace_sec=int(raw.get("task_kill_grace_sec", 30)),
            ray=RayPoolConfig.from_mapping(
                raw.get("ray") if isinstance(raw.get("ray"), Mapping) else None
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ClusterBridgePoolConfig:
        config_path = Path(path).expanduser().resolve()
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            value = json.loads(text)
        else:
            value = yaml.safe_load(text)
        if not isinstance(value, Mapping):
            raise TypeError("cluster pool config must contain a mapping")
        return cls.from_mapping(value)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_mapping() for node in self.nodes],
            "cb_command": self.cb_command,
            "purpose": self.purpose,
            "pool_id": self.pool_id,
            "log_root": self.log_root,
            "claim_ttl_min": self.claim_ttl_min,
            "renew_interval_sec": self.renew_interval_sec,
            "max_renew_failures": self.max_renew_failures,
            "command_timeout_sec": self.command_timeout_sec,
            "parallelism": self.parallelism,
            "expected_total_gpus": self.expected_total_gpus,
            "allow_force_claim": self.allow_force_claim,
            "expected_claim_owner": self.expected_claim_owner,
            "node_cleanup_script": self.node_cleanup_script,
            "node_spin_script": self.node_spin_script,
            "task_kill_grace_sec": self.task_kill_grace_sec,
            "ray": self.ray.to_mapping(),
        }
