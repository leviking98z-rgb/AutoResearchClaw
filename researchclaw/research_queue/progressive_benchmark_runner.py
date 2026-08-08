"""Run one immutable treatment on a frozen progressive evidence partition.

This module is framework-owned code.  The LLM generates only ``treatment.py``;
it does not generate a second synthetic experiment whose data or metric can
drift away from the confirmatory benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from researchclaw.benchmark_adapter.cifar10_calibration import (
    BenchmarkResult,
    run_from_file,
    run_from_logits_cache,
    sha256_path,
)

from .benchmark_runner import build_promoted_benchmark_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen B0/B1/B2 evidence partition",
    )
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--treatment-path", required=True, type=Path)
    parser.add_argument("--logits-cache", type=Path)
    return parser


def _budget_parameters() -> dict[str, Any]:
    raw = os.environ.get("RESEARCH_QUEUE_BUDGET_JSON", "")
    if not raw:
        raise ValueError("RESEARCH_QUEUE_BUDGET_JSON is required")
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise TypeError("RESEARCH_QUEUE_BUDGET_JSON must contain an object")
    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping):
        raise TypeError("budget parameters must be an object")
    return {str(key): item for key, item in parameters.items()}


def _selected_seeds(parameters: Mapping[str, Any]) -> tuple[int, ...]:
    raw = parameters.get("seeds")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("progressive budget parameters require a seeds list")
    seeds = tuple(int(item) for item in raw)
    if len(set(seeds)) != len(seeds):
        raise ValueError("progressive budget seeds must be unique")
    return seeds


def _render_partition_config(
    *,
    template: Path,
    treatment_path: Path,
    output_dir: Path,
    destination: Path,
    seeds: tuple[int, ...],
) -> Path:
    runtime = build_promoted_benchmark_config(
        template_path=template,
        treatment_path=treatment_path,
        output_dir=output_dir,
        destination=destination,
    )
    value = yaml.safe_load(runtime.read_text(encoding="utf-8")) or {}
    benchmark = value.get("benchmark", value)
    if not isinstance(benchmark, Mapping):
        raise TypeError("rendered benchmark config must contain one mapping")
    payload = dict(benchmark)
    payload["seeds"] = list(seeds)
    runtime.write_text(
        yaml.safe_dump(
            {"benchmark": payload},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return runtime


def _queue_metrics(result: BenchmarkResult) -> dict[str, Any]:
    metrics = dict(result.metrics)
    per_pair = [
        {
            "seed": row.get("seed"),
            "effect_ece": row.get("effect_ece"),
            "effect_nll": row.get("effect_nll"),
            "effect_accuracy": row.get("effect_accuracy"),
        }
        for row in result.per_seed
    ]
    return {
        "primary_metric": "ece",
        "primary_value": metrics.get("effect_ece"),
        "treatment_value": metrics.get("treatment_ece"),
        "control_value": metrics.get("baseline_ece"),
        "effect": metrics.get("effect_ece"),
        **metrics,
        "uncertainty": dict(result.uncertainty),
        "per_pair": per_pair,
        "independent_pairs": len(result.per_seed),
    }


def run_partition(
    *,
    benchmark_config: str | Path,
    treatment_path: str | Path,
    output_dir: str | Path,
    parameters: Mapping[str, Any],
    logits_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one declared partition and write the Queue result contract."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = _selected_seeds(parameters)
    partition = str(
        parameters.get(
            "evidence_partition",
            os.environ.get("RESEARCH_QUEUE_BUDGET", ""),
        )
        or ""
    )
    benchmark_output = root / "benchmark"
    runtime_config = _render_partition_config(
        template=Path(benchmark_config).expanduser().resolve(),
        treatment_path=Path(treatment_path).expanduser().resolve(),
        output_dir=benchmark_output,
        destination=root / "benchmark-config.yaml",
        seeds=seeds,
    )
    cache = (
        Path(logits_cache).expanduser().resolve()
        if logits_cache is not None
        else None
    )
    result = (
        run_from_logits_cache(runtime_config, cache_path=cache)
        if cache is not None and cache.is_file()
        else run_from_file(runtime_config)
    )
    raw_result = root / "benchmark-result.json"
    shutil.copy2(benchmark_output / "result.json", raw_result)
    payload = {
        "status": result.status,
        "metrics": _queue_metrics(result),
        "artifacts": [
            str(raw_result),
            str(runtime_config),
            *result.artifacts,
        ],
        "usage": {
            **dict(result.usage),
            "budget_parameters": dict(parameters),
            "evidence_partition": partition,
            "selected_seeds": list(seeds),
            "benchmark_config_sha256": sha256_path(runtime_config),
            "treatment_sha256": sha256_path(treatment_path),
            "logits_cache_sha256": (
                sha256_path(cache)
                if cache is not None and cache.is_file()
                else ""
            ),
        },
        "error": result.error,
    }
    (root / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = os.environ.get("RESEARCH_QUEUE_OUTPUT_DIR", "")
    if not output:
        raise SystemExit("RESEARCH_QUEUE_OUTPUT_DIR is required")
    parameters: dict[str, Any] = {}
    try:
        parameters = _budget_parameters()
        payload = run_partition(
            benchmark_config=args.benchmark_config,
            treatment_path=args.treatment_path,
            output_dir=output,
            parameters=parameters,
            logits_cache=args.logits_cache,
        )
    except Exception as exc:  # noqa: BLE001
        root = Path(output).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "error",
            "metrics": {},
            "artifacts": [],
            "usage": {
                "budget_parameters": parameters,
            },
            "error": f"{type(exc).__name__}: {exc}",
        }
        (root / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return 1
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
