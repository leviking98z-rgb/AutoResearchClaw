#!/usr/bin/env python3
"""Run reproducible PortfolioBench variants and compare their VCO reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SHARED_ROOT = Path("/root/shared/.clusters/.workdir/portfolio-bench")
DEFAULT_STATE_ROOT = Path("/root/.local/state/portfolio-bench")
CB = Path("/root/shared/.clusters/.tools/clusterbridge.sh")
AM = Path("/root/shared/.clusters/.tools/am.sh")

VARIANTS: dict[str, dict[str, Any]] = {
    "rq-sequential": {
        "max_active_ideas": 1,
        "max_llm_jobs": 1,
        "max_run_jobs": 1,
        "direct_all_admitted": False,
    },
    "rq-no-early-exit": {
        "max_active_ideas": 4,
        "max_llm_jobs": 2,
        "max_run_jobs": 8,
        "direct_all_admitted": True,
    },
    "rq-full": {
        "max_active_ideas": 4,
        "max_llm_jobs": 2,
        "max_run_jobs": 8,
        "direct_all_admitted": False,
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic uses deterministic workers and a CPU logits fixture",
    )
    parser.add_argument(
        "--variants",
        default="rq-sequential,rq-no-early-exit,rq-full",
    )
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--token-budget", type=int, default=1_200_000)
    parser.add_argument("--max-gpus", type=int, default=8)
    parser.add_argument("--ideas-limit", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument(
        "--idea-pool",
        type=Path,
        default=EXPERIMENT_ROOT / "ideas.v1.json",
    )
    parser.add_argument(
        "--benchmark-template",
        type=Path,
        default=EXPERIMENT_ROOT / "benchmark.cifar10.v1.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT / "config.rsi.yaml",
    )
    parser.add_argument(
        "--real-cache-root",
        type=Path,
        default=Path(
            "/root/shared/.clusters/.workdir/"
            "research-queue-real-benchmark/cache"
        ),
    )
    parser.add_argument(
        "--logits-cache",
        type=Path,
        help=(
            "Optional trusted partitioned cache; omit to exercise real GPU "
            "runs."
        ),
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a real development smoke from an uncommitted source tree.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    variants = [
        item.strip() for item in args.variants.split(",") if item.strip()
    ]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    ideas = json.loads(args.idea_pool.read_text(encoding="utf-8"))
    if not isinstance(ideas, list) or not ideas:
        raise SystemExit("idea pool must be a non-empty JSON array")
    ideas = ideas[: max(1, args.ideas_limit)]
    dirty = _git_dirty()
    if args.mode == "real" and dirty and not args.allow_dirty:
        raise SystemExit(
            "real PortfolioBench requires a clean committed source tree; "
            "use --allow-dirty only for a non-formal development smoke"
        )

    suite_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    suite_root = _lexical_absolute(args.shared_root) / suite_id
    suite_root.mkdir(parents=True, exist_ok=False)
    suite_manifest = {
        "schema_version": 1,
        "suite_id": suite_id,
        "mode": args.mode,
        "variants": variants,
        "repeats": args.repeat,
        "duration_sec": args.duration_sec,
        "token_budget": args.token_budget,
        "max_gpus": args.max_gpus,
        "idea_pool_sha256": _sha256(args.idea_pool),
        "benchmark_template_sha256": _sha256(args.benchmark_template),
        "model_config_sha256": _sha256(args.model_config),
        "framework_commit": _git_commit(),
        "framework_dirty": dirty,
        "framework_tree_sha256": _source_tree_sha256(),
    }
    _write_json(suite_root / "suite-manifest.json", suite_manifest)

    source_root = (
        _sync_source(
            suite_root,
            suite_manifest["framework_commit"],
            suite_manifest["framework_tree_sha256"],
        )
        if args.mode == "real"
        else None
    )
    owner = _agent_id() if args.mode == "real" else ""
    reports: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in variants
    }
    for repeat in range(1, args.repeat + 1):
        for variant in variants:
            report = _run_variant(
                args=args,
                suite_root=suite_root,
                suite_id=suite_id,
                variant=variant,
                repeat=repeat,
                ideas=ideas,
                source_root=source_root,
                owner=owner,
            )
            reports[variant].append(report)

    comparisons: list[dict[str, Any]] = []
    baseline_variant = variants[0]
    for repeat in range(args.repeat):
        baseline = reports[baseline_variant][repeat]
        for variant in variants[1:]:
            comparison = _compare(baseline, reports[variant][repeat])
            comparison["repeat"] = repeat + 1
            comparisons.append(comparison)
    _write_json(suite_root / "comparisons.json", comparisons)
    aggregate = _aggregate(reports)
    _write_json(suite_root / "aggregate.json", aggregate)
    print(
        json.dumps(
            {
                "suite_root": str(suite_root),
                "aggregate": aggregate,
                "comparisons": comparisons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_variant(
    *,
    args: argparse.Namespace,
    suite_root: Path,
    suite_id: str,
    variant: str,
    repeat: int,
    ideas: list[dict[str, Any]],
    source_root: Path | None,
    owner: str,
) -> dict[str, Any]:
    run_id = f"{variant}-r{repeat:02d}"
    run_root = suite_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    state_dir = _lexical_absolute(args.state_root) / suite_id / run_id
    artifact_dir = run_root / "artifacts"
    ideas_path = run_root / "ideas.json"
    _write_json(ideas_path, ideas)

    benchmark_path = run_root / "benchmark.yaml"
    logits_cache: Path | None
    if args.mode == "synthetic":
        logits_cache = run_root / "synthetic-logits.npz"
        _build_synthetic_logits_cache(logits_cache)
        benchmark = _synthetic_benchmark(
            run_root=run_root,
            logits_cache=logits_cache,
        )
    else:
        logits_cache = (
            _lexical_absolute(args.logits_cache)
            if args.logits_cache is not None
            else None
        )
        benchmark = _real_benchmark(
            template=args.benchmark_template,
            run_root=run_root,
            cache_root=_lexical_absolute(args.real_cache_root),
        )
    benchmark_path.write_text(
        yaml.safe_dump(
            {"benchmark": benchmark},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    config = _queue_config(
        args=args,
        suite_id=suite_id,
        run_id=run_id,
        variant=variant,
        idea_count=len(ideas),
        state_dir=state_dir,
        artifact_dir=artifact_dir,
        benchmark_path=benchmark_path,
        logits_cache=logits_cache,
        source_root=source_root,
        owner=owner,
        run_root=run_root,
    )
    config_path = run_root / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"research_queue": config},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    before = _resource_status() if args.mode == "real" else {}
    manifest = {
        "run_id": run_id,
        "variant": variant,
        "repeat": repeat,
        "mode": args.mode,
        "framework_commit": _git_commit(),
        "framework_dirty": _git_dirty(),
        "framework_tree_sha256": _source_tree_sha256(),
        "idea_pool_sha256": _sha256(ideas_path),
        "benchmark_config_sha256": _sha256(benchmark_path),
        "queue_config_sha256": _sha256(config_path),
        "model_config_sha256": _sha256(args.model_config),
        "resource_status_before": before,
    }
    _write_json(run_root / "manifest.json", manifest)

    command = [
        sys.executable,
        "-m",
        "researchclaw.research_queue.cli",
        "-c",
        str(config_path),
        "start",
        "--max-seconds",
        str(args.duration_sec),
        "--until-idle",
        "--ideas-json",
        str(ideas_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)
    started = datetime.now(UTC)
    with (run_root / "controller.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(60.0, args.duration_sec + 300.0),
            check=False,
        )
    ended = datetime.now(UTC)
    after = _resource_status() if args.mode == "real" else {}
    manifest.update(
        {
            "returncode": completed.returncode,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "wall_seconds": (ended - started).total_seconds(),
            "resource_status_after": after,
            "resource_released": (
                _resources_released(after) if args.mode == "real" else True
            ),
        }
    )
    _write_json(run_root / "manifest.json", manifest)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{run_id} failed with return code {completed.returncode}; "
            f"see {run_root / 'controller.log'}"
        )
    if completed.returncode == 0:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "researchclaw.research_queue.cli",
                "-c",
                str(config_path),
                "portfolio",
                "--output-dir",
                str(state_dir / "portfolio"),
                "--window-seconds",
                str(args.duration_sec),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    report_path = state_dir / "portfolio" / "portfolio-report.json"
    if not report_path.is_file():
        raise RuntimeError(
            f"{run_id} did not produce a Portfolio report; "
            f"see {run_root / 'controller.log'}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    shutil.copytree(
        state_dir / "portfolio",
        run_root / "portfolio",
        dirs_exist_ok=True,
    )
    manifest["portfolio_summary"] = report["summary"]
    _write_json(run_root / "manifest.json", manifest)
    if args.mode == "real" and not manifest["resource_released"]:
        raise RuntimeError(f"{run_id} left a GPU allocation or request behind")
    return report


def _queue_config(
    *,
    args: argparse.Namespace,
    suite_id: str,
    run_id: str,
    variant: str,
    idea_count: int,
    state_dir: Path,
    artifact_dir: Path,
    benchmark_path: Path,
    logits_cache: Path | None,
    source_root: Path | None,
    owner: str,
    run_root: Path,
) -> dict[str, Any]:
    settings = VARIANTS[variant]
    synthetic = args.mode == "synthetic"
    direct = bool(settings["direct_all_admitted"])
    max_gpus = 0 if synthetic else max(1, args.max_gpus)
    return {
        "enabled": True,
        "system_id": f"portfolio-bench-{suite_id}-{run_id}",
        "state_dir": str(state_dir),
        "artifact_dir": str(artifact_dir),
        "brief": (
            "Evaluate fixed post-hoc calibration Ideas against scalar "
            "temperature scaling under a frozen CIFAR-10 corruption protocol."
        ),
        "limits": {
            "candidate_target": idea_count,
            "generation_batch_size": idea_count,
            "generation_interval_sec": 0,
            "generation_max_batches": 1,
            "max_active_ideas": settings["max_active_ideas"],
            "max_total_ideas": idea_count,
            "max_total_tokens": args.token_budget,
            "max_revisions_per_idea": 1,
            "max_runs_per_budget": 1,
            "max_steps_per_idea": 10,
            "max_infra_retries": 0,
            "max_prepare_repairs": 1,
            "duplicate_threshold": 0.95,
            "required_paths": [] if direct else ["b2"],
        },
        "concurrency": {
            "max_llm_jobs": settings["max_llm_jobs"],
            "max_run_jobs": settings["max_run_jobs"],
            "poll_interval_sec": 0.05 if synthetic else 0.25,
            "llm_call_timeout_sec": 900,
        },
        "models": {
            "researchclaw_config": str(
                args.model_config.expanduser().resolve()
            ),
            "decision_role": "research_director",
            "worker_role": "coding_engineer",
            "utility_role": "literature_researcher",
        },
        "execution": {
            "backend": "local" if synthetic else "clusterbridge",
            "python_executable": sys.executable if synthetic else "python3",
            "simulation": synthetic,
            "allowed_python_imports": ["numpy"],
            "remote_pythonpath": str(source_root or ""),
        },
        "scientific_gate": {
            "enabled": True,
            "require_structured_spec": True,
            "max_repairs": 1,
        },
        "promotion": {
            "enabled": True,
            "benchmark_id": "cifar10_calibration",
            "benchmark_config": str(benchmark_path),
            "trigger_conclusions": (
                ["positive", "negative", "inconclusive"]
                if direct
                else ["positive", "inconclusive"]
            ),
            "minimum_priority": 0,
            "minimum_effect": 0.001,
            "max_promotions": idea_count,
            "max_treatment_repairs": 1,
            "preflight_examples": 48 if synthetic else 96,
            "preflight_classes": 10,
            "preflight_timeout_sec": 20,
            "runtime_python": (
                sys.executable
                if synthetic
                else "/opt/conda/envs/torch-base/bin/python"
            ),
            "logits_cache": str(logits_cache or ""),
            "prefer_logits_cache": logits_cache is not None,
            "direct_all_admitted": direct,
            "progressive_pilot": True,
        },
        "research_memory": {"enabled": False},
        "gpu": {
            "max_total_gpus": max_gpus,
            "max_gpus_per_run": 0 if synthetic else 1,
            "poll_interval_sec": 0.25,
            "pass_env": [],
            "resource_manager": {
                "owner": owner,
                "cb_command": str(CB),
                "project": "PortfolioBench",
                "purpose": f"PortfolioBench {suite_id} {run_id}",
                "max_gpus": max_gpus,
                "duration_min": max(30, int(args.duration_sec / 60) + 15),
                "renew_ttl_min": 60,
                "renew_interval_sec": 300,
                "reconcile_interval_sec": 1,
                "allow_cross_cluster": True,
                "gpu_type": "H20",
                "priority": "normal",
                "release_on_shutdown": True,
                "log_root": str(run_root / "cluster"),
            },
        },
        "budgets": {
            "B0": {
                "gpus": 0,
                "timeout_sec": 120,
                "parameters": {
                    "evidence_partition": "pilot-b0",
                    "seeds": [101, 103],
                },
            },
            "B1": {
                "gpus": 0,
                "timeout_sec": 240,
                "parameters": {
                    "evidence_partition": "pilot-b1",
                    "seeds": [107, 109, 113],
                },
            },
            "B2": {
                "gpus": 0,
                "timeout_sec": 480,
                "parameters": {
                    "evidence_partition": "confirmatory-b2",
                    "seeds": [17, 29, 43, 59, 71],
                },
            },
        },
    }


def _synthetic_benchmark(
    *,
    run_root: Path,
    logits_cache: Path,
) -> dict[str, Any]:
    return {
        "cache_dir": str(run_root / "cache"),
        "output_dir": str(run_root / "placeholder-output"),
        "treatment_path": str(run_root / "placeholder-treatment.py"),
        "examples": 64,
        "calibration_examples": 64,
        "pairing_seeds": [101, 103, 107, 109, 113, 17, 29, 43, 59, 71],
        "seeds": [17, 29, 43, 59, 71],
        "corruption": "gaussian_noise",
        "corruption_severity": 0.04,
        "ece_bins": 10,
        "require_cuda": False,
        "device": "cpu",
        "allow_downloads": False,
        "timeout_sec": 120,
        "fixture_logits_cache": str(logits_cache),
    }


def _real_benchmark(
    *,
    template: Path,
    run_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    raw = yaml.safe_load(template.read_text(encoding="utf-8")) or {}
    value = dict(raw.get("benchmark", raw))
    value.update(
        {
            "cache_dir": str(cache_root),
            "output_dir": str(run_root / "placeholder-output"),
            "treatment_path": str(run_root / "placeholder-treatment.py"),
            "model_source_archive": str(
                cache_root
                / (
                    "model-source-"
                    "786c16252c0fc58ee9adac063f8337cc4a7a497a.tar.gz"
                )
            ),
        }
    )
    return value


def _build_synthetic_logits_cache(path: Path) -> None:
    seeds = [101, 103, 107, 109, 113, 17, 29, 43, 59, 71]
    selected_seeds = [17, 29, 43, 59, 71]
    examples = 64
    classes = 10
    labels = np.arange(examples, dtype=np.int64) % classes
    arrays: dict[str, np.ndarray] = {}
    for seed in seeds:
        arrays[f"calibration_logits_{seed}"] = np.zeros(
            (examples, classes),
            dtype=np.float64,
        )
        arrays[f"calibration_labels_{seed}"] = labels
        arrays[f"evaluation_logits_{seed}"] = np.zeros(
            (examples, classes),
            dtype=np.float64,
        )
        arrays[f"evaluation_labels_{seed}"] = labels
    metadata = {
        "schema_version": 4,
        "seeds": seeds,
        "selected_seeds": selected_seeds,
        "ece_bins": 10,
        "assets": {
            "fixture": "uniform-logits-v1",
            "model_name": "synthetic",
        },
        "provenance": {
            "corruption": "gaussian_noise",
            "corruption_severity": 0.04,
            "examples": examples,
            "calibration_examples": examples,
            "calibration_split": "clean",
            "evaluation_split": "corrupted",
            "pairing_strategy": "disjoint_example_blocks",
            "pairing_seeds": seeds,
        },
    }
    arrays["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _sync_source(
    suite_root: Path,
    commit: str,
    tree_sha256: str,
) -> Path:
    destination = suite_root / f"source-{commit[:12]}"
    shutil.copytree(
        REPO_ROOT / "researchclaw",
        destination / "researchclaw",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (destination / "SOURCE_COMMIT").write_text(commit + "\n", encoding="utf-8")
    (destination / "SOURCE_TREE_SHA256").write_text(
        tree_sha256 + "\n",
        encoding="utf-8",
    )
    return destination


def _agent_id() -> str:
    completed = subprocess.run(
        ["bash", str(AM), "whoami"],
        cwd=Path("/root/shared/.clusters"),
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip().splitlines()[-1]


def _resource_status() -> dict[str, Any]:
    completed = subprocess.run(
        ["bash", str(CB), "resource-status", "--json"],
        cwd=Path("/root/shared/.clusters"),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "resource-status returned invalid JSON"}


def _resources_released(status: Mapping[str, Any]) -> bool:
    snapshot = status.get("snapshot", {})
    if not isinstance(snapshot, Mapping):
        return False
    return not snapshot.get("allocations") and not snapshot.get("queue")


def _compare(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    from researchclaw.research_queue.portfolio import compare_portfolio_reports

    return compare_portfolio_reports(baseline, candidate)


def _aggregate(
    reports: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant, values in reports.items():
        summaries = [dict(item["summary"]) for item in values]
        output[variant] = {
            "repeats": len(summaries),
            "vco_at_window": [
                item.get("vco_at_window") for item in summaries
            ],
            "ttfv_seconds": [
                item.get("ttfv_seconds") for item in summaries
            ],
            "total_tokens": [
                dict(value["usage"]).get("total_tokens") for value in values
            ],
            "total_gpu_seconds": [
                dict(value["usage"]).get("total_gpu_seconds")
                for value in values
            ],
            "false_accept_rate": [
                item.get("false_accept_rate") for item in summaries
            ],
        }
    return output


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    )


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    roots = (
        REPO_ROOT / "researchclaw",
        REPO_ROOT / "experiments" / "portfolio_bench",
        REPO_ROOT / "experiments" / "evidence_pack_bench",
    )
    files = [
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    ]
    files.extend((REPO_ROOT / "pyproject.toml", REPO_ROOT / "config.rsi.yaml"))
    for path in sorted(set(files)):
        relative = path.relative_to(REPO_ROOT).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without resolving the /root/shared symlink."""

    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
