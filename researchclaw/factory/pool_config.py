"""Minimal pool-config adapter used by the Factory GPU Broker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class PoolConfigSummary:
    config_path: Path
    expected_total_gpus: int
    max_gpus_per_node: int

    @classmethod
    def from_file(cls, path: str | Path) -> PoolConfigSummary:
        config_path = Path(path).expanduser().resolve()
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("pool config must contain a mapping")
        raw: Mapping[str, Any] = value
        if isinstance(value.get("clusterbridge_pool"), Mapping):
            raw = value["clusterbridge_pool"]
        elif isinstance(value.get("cluster_pool"), Mapping):
            raw = value["cluster_pool"]
        nodes = raw.get("nodes", ())
        if not isinstance(nodes, list):
            raise TypeError("pool nodes must be a list")
        node_gpu_counts = [
            len(node.get("gpu_ids", ()))
            for node in nodes
            if isinstance(node, Mapping)
        ]
        expected = raw.get("expected_total_gpus")
        if expected is None:
            expected = sum(node_gpu_counts)
        return cls(
            config_path=config_path,
            expected_total_gpus=int(expected),
            max_gpus_per_node=max(node_gpu_counts, default=int(expected)),
        )
