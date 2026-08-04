"""Minimal real-execution smoke test for the configured Docker sandbox."""

from __future__ import annotations

import json
from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.experiment.docker_sandbox import DockerSandbox


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = RCConfig.load(root / "config.local-smoke.yaml", check_paths=False)
    sandbox = DockerSandbox(
        config.experiment.docker,
        root / "artifacts" / "local-docker-smoke",
    )
    code = """
import json
import numpy as np

rng = np.random.default_rng(20260804)
x = rng.normal(size=256)
baseline = float(np.mean((x - 0.0) ** 2))
proposed = float(np.mean((x - np.mean(x)) ** 2))
improvement = baseline - proposed

payload = {
    "primary_metric": improvement,
    "baseline_mse": baseline,
    "proposed_mse": proposed,
    "seed": 20260804,
}
with open("results.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)

print(f"primary_metric: {improvement}")
print(f"condition=baseline baseline_mse: {baseline}")
print(f"condition=proposed proposed_mse: {proposed}")
"""
    result = sandbox.run(code, timeout_sec=60)
    print(json.dumps({
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "elapsed_sec": result.elapsed_sec,
        "metrics": result.metrics,
        "stderr_tail": result.stderr[-500:],
    }, indent=2))
    if result.returncode != 0 or "primary_metric" not in result.metrics:
        raise SystemExit("Docker sandbox smoke test failed")


if __name__ == "__main__":
    main()

