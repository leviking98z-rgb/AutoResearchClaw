"""Deterministic candidate and runtime validation for v2 jobs."""

from __future__ import annotations

import json
import math
import py_compile
from pathlib import Path
from typing import Any


def validate_python_tree(root: Path) -> dict[str, Any]:
    from researchclaw.experiment.validator import validate_code

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    files = sorted(root.rglob("*.py"))
    if not files:
        errors.append(
            {
                "file": "",
                "category": "structure",
                "message": "candidate contains no Python files",
            }
        )
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(
                {
                    "file": str(path.relative_to(root)),
                    "category": "syntax",
                    "message": str(exc),
                }
            )
            continue
        result = validate_code(path.read_text(encoding="utf-8"))
        for issue in result.issues:
            row = {
                "file": str(path.relative_to(root)),
                "category": issue.category,
                "message": issue.message,
                "line": issue.line,
            }
            (errors if issue.severity == "error" else warnings).append(row)
    return {
        "ok": not errors,
        "files_checked": [str(path.relative_to(root)) for path in files],
        "errors": errors,
        "warnings": warnings,
    }


def validate_plan(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "research_question",
        "hypothesis",
        "primary_metric",
        "pilot",
        "promotion_rule",
        "early_stop_rule",
    ):
        if not value.get(field):
            errors.append(f"missing {field}")
    for field in ("datasets", "models", "baselines", "ablations"):
        if not isinstance(value.get(field), list) or not value[field]:
            errors.append(f"missing {field}")
    pilot = value.get("pilot")
    if isinstance(pilot, dict):
        for field in ("max_gpus", "max_examples", "max_seeds", "timeout_sec"):
            try:
                number = float(pilot[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid pilot.{field}")
                continue
            if number <= 0:
                errors.append(f"invalid pilot.{field}")
    return errors


def validate_build_output(value: dict[str, Any]) -> list[str]:
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        return ["files must be a non-empty object"]
    errors: list[str] = []
    for filename, content in files.items():
        if (
            not isinstance(filename, str)
            or filename.startswith("/")
            or ".." in Path(filename).parts
        ):
            errors.append(f"invalid filename {filename!r}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"empty file {filename!r}")
    commands = value.get("commands")
    if not isinstance(commands, dict):
        errors.append("commands must be an object")
    else:
        for field in ("smoke", "pilot", "scale"):
            if not str(commands.get(field, "") or "").strip():
                errors.append(f"missing commands.{field}")
    return errors


def validate_metrics_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "errors": [f"metrics unreadable: {exc}"]}
    errors: list[str] = []
    if not isinstance(value, dict):
        errors.append("metrics root must be an object")
        value = {}
    if value.get("result_valid") is not True:
        errors.append("result_valid must be true")
    if not isinstance(value.get("metrics"), dict) or not value.get("metrics"):
        errors.append("finite metrics object is required")
    else:
        for name, metric in value["metrics"].items():
            if isinstance(metric, bool):
                continue
            if isinstance(metric, (int, float)) and not math.isfinite(
                float(metric)
            ):
                errors.append(f"metric {name} is not finite")
    return {"ok": not errors, "errors": errors, "metrics": value}


def validate_runtime_evidence_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "errors": [f"runtime evidence unreadable: {exc}"],
            "evidence": {},
        }
    if not isinstance(value, dict):
        return {
            "ok": False,
            "errors": ["runtime evidence root must be an object"],
            "evidence": {},
        }
    errors: list[str] = []
    required = (
        "model_loaded",
        "datasets_loaded",
        "examples_processed",
        "seeds",
        "gpu_count",
        "gate_decision",
        "metrics",
    )
    for field in required:
        if field not in value:
            errors.append(f"missing runtime evidence {field}")
    if not str(value.get("model_loaded", "") or "").strip():
        errors.append("model_loaded must identify a real model")
    if not isinstance(value.get("datasets_loaded"), list) or not value.get(
        "datasets_loaded"
    ):
        errors.append("datasets_loaded must be a non-empty list")
    for field in ("examples_processed", "gpu_count"):
        try:
            number = int(value.get(field, -1))
        except (TypeError, ValueError):
            errors.append(f"invalid runtime evidence {field}")
        else:
            if number < 0:
                errors.append(f"invalid runtime evidence {field}")
    seeds = value.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("seeds must be a non-empty list")
    if not isinstance(value.get("metrics"), dict) or not value.get("metrics"):
        errors.append("runtime evidence metrics must be a non-empty object")
    decision = str(value.get("gate_decision", "") or "").casefold()
    if decision not in {
        "promote",
        "continue",
        "reject",
        "stop",
        "complete_negative",
    }:
        errors.append("invalid gate_decision")
    return {
        "ok": not errors,
        "errors": errors,
        "evidence": value,
    }


def validate_experiment_artifacts(output_dir: Path) -> dict[str, Any]:
    metrics = validate_metrics_file(output_dir / "metrics.json")
    runtime = validate_runtime_evidence_file(
        output_dir / "runtime_evidence.json"
    )
    errors = [
        *metrics.get("errors", []),
        *runtime.get("errors", []),
    ]
    return {
        "ok": not errors,
        "errors": errors,
        "metrics": metrics.get("metrics", {}),
        "runtime_evidence": runtime.get("evidence", {}),
        "files": [
            str(path.relative_to(output_dir))
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        ]
        if output_dir.is_dir()
        else [],
    }
