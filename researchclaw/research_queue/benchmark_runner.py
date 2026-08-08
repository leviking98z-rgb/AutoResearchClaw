"""Execute a promoted treatment through a configured Benchmark Adapter."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from researchclaw.benchmark_adapter.cifar10_calibration import BenchmarkConfig

from .models import RunResult


def build_promoted_benchmark_config(
    *,
    template_path: str | Path,
    treatment_path: str | Path,
    output_dir: str | Path,
    destination: str | Path,
) -> Path:
    source = Path(template_path).expanduser().resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    benchmark = value.get("benchmark", value)
    if not isinstance(benchmark, Mapping):
        raise TypeError("benchmark template must contain one mapping")
    payload = dict(benchmark)
    payload["treatment_path"] = str(Path(treatment_path).expanduser().resolve())
    payload["output_dir"] = str(Path(output_dir).expanduser().resolve())
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {"benchmark": payload},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return target


def benchmark_command(
    config_path: str | Path,
    *,
    python_executable: str = "python",
    logits_cache: str | Path | None = None,
) -> tuple[str, ...]:
    command = [
        python_executable,
        "-m",
        "researchclaw.research_queue.benchmark_runner",
        str(Path(config_path).expanduser().resolve()),
    ]
    if logits_cache:
        command += [
            "--logits-cache",
            str(Path(logits_cache).expanduser().resolve()),
        ]
    return tuple(command)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise SystemExit(
            "usage: benchmark_runner <benchmark-config.yaml> "
            "[--logits-cache <cache.npz>]"
        )
    config_path = Path(arguments[0]).expanduser().resolve()
    logits_cache: Path | None = None
    if len(arguments) == 3 and arguments[1] == "--logits-cache":
        logits_cache = Path(arguments[2]).expanduser().resolve()
    elif len(arguments) != 1:
        raise SystemExit("invalid benchmark_runner arguments")
    config = BenchmarkConfig.from_file(config_path)
    result_path = config.output_dir / "result.json"
    final_path = config.output_dir / "final-result.json"
    try:
        from researchclaw.benchmark_adapter.cifar10_calibration import (
            run_from_file,
            run_from_logits_cache,
        )

        if logits_cache is not None and logits_cache.is_file():
            run_from_logits_cache(config_path, cache_path=logits_cache)
        else:
            run_from_file(config_path)
        shutil.copy2(result_path, final_path)
        return 0
    except Exception as exc:  # noqa: BLE001
        config.output_dir.mkdir(parents=True, exist_ok=True)
        failed = RunResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            returncode=1,
        ).to_dict()
        failed["status"] = "error"
        result_path.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        final_path.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
