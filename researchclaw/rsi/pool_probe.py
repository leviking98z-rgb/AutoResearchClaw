"""Fail-closed readiness probe for an existing ClusterBridge/Ray pool."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from researchclaw.experiment.clusterbridge_pool import ClusterBridgePool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    pool = ClusterBridgePool.from_file(
        args.config,
        restore_state=True,
    )
    if not (pool.claimed and pool.prepared and pool.ray_started):
        raise RuntimeError(
            "pool state is not claimed, prepared, and Ray-ready"
        )
    resources = pool.wait_for_ray_resources(
        timeout_sec=min(30.0, pool.config.ray.resource_timeout_sec)
    )
    print(
        json.dumps(
            {
                "pool_id": pool.config.pool_id,
                "claimed": pool.claimed,
                "prepared": pool.prepared,
                "ray_started": pool.ray_started,
                "resources": asdict(resources),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
