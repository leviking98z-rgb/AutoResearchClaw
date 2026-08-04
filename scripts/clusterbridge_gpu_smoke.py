#!/usr/bin/env python3
"""Exercise a real CUDA kernel through ResearchClaw's ClusterBridge sandbox."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.experiment.factory import create_sandbox


def main() -> None:
    config = RCConfig.load(Path("config.gpu.yaml"), check_paths=False)
    code = r'''
import json
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, "GPU isolation must expose exactly one GPU"
device = torch.device("cuda:0")
a = torch.randn((2048, 2048), device=device)
b = torch.randn((2048, 2048), device=device)
c = a @ b
torch.cuda.synchronize()
metric = float(c.square().mean().sqrt().item())
print(f"primary_metric: {metric}")
print(f"gpu_count: {torch.cuda.device_count()}")
print(f"gpu_memory_mb: {torch.cuda.get_device_properties(0).total_memory / 1024**2}")
with open("results.json", "w", encoding="utf-8") as f:
    json.dump({
        "primary_metric": metric,
        "gpu_count": torch.cuda.device_count(),
        "gpu_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1024**2,
    }, f)
'''
    with tempfile.TemporaryDirectory(prefix="researchclaw-cb-smoke-") as td:
        sandbox = create_sandbox(config.experiment, Path(td))
        result = sandbox.run(code, timeout_sec=180)
    payload = {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "elapsed_sec": result.elapsed_sec,
        "metrics": result.metrics,
        "stderr_tail": result.stderr[-2000:],
    }
    print(json.dumps(payload, indent=2))
    assert result.returncode == 0, payload
    assert not result.timed_out, payload
    assert float(result.metrics.get("primary_metric", 0.0)) > 0.0, payload
    assert int(float(result.metrics.get("gpu_count", 0))) == 1, payload
    assert float(result.metrics.get("gpu_memory_mb", 0.0)) > 90000, payload


if __name__ == "__main__":
    main()
