"""Deterministic candidate and runtime validation for v2 jobs."""

from __future__ import annotations

import ast
import hashlib
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


def validate_research_implementation(
    root: Path,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Verify the generated project contains a real model/benchmark path."""

    errors: list[str] = []
    evidence = {
        "model_loader_calls": [],
        "dataset_loader_calls": [],
        "artifact_writes": [],
        "source_sha256": {},
    }
    python_files = sorted(root.rglob("*.py"))
    for path in python_files:
        relative = str(path.relative_to(root))
        source = path.read_text(encoding="utf-8")
        evidence["source_sha256"][relative] = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if (
                name.endswith(".from_pretrained")
                or name in {
                    "transformers.pipeline",
                    "pipeline",
                    "vllm.LLM",
                    "LLM",
                }
            ):
                evidence["model_loader_calls"].append(
                    {"file": relative, "line": node.lineno, "call": name}
                )
            if name in {
                "datasets.load_dataset",
                "load_dataset",
                "evaluate.load",
                "load",
            }:
                evidence["dataset_loader_calls"].append(
                    {"file": relative, "line": node.lineno, "call": name}
                )
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.endswith(
                        ("metrics.json", "runtime_evidence.json")
                    )
                ):
                    evidence["artifact_writes"].append(
                        {
                            "file": relative,
                            "line": node.lineno,
                            "path": arg.value,
                        }
                    )
    if not evidence["model_loader_calls"]:
        errors.append("no real model loader call found")
    if not evidence["dataset_loader_calls"]:
        errors.append("no real dataset/benchmark loader call found")
    artifact_names = {
        Path(str(row["path"])).name
        for row in evidence["artifact_writes"]
    }
    if "metrics.json" not in artifact_names:
        errors.append("no metrics.json artifact write found")
    if "runtime_evidence.json" not in artifact_names:
        errors.append("no runtime_evidence.json artifact write found")
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").casefold()
        for path in python_files
    )
    for marker in (
        "fake_model",
        "mock_model",
        "synthetic_accuracy",
        "random.random()",
    ):
        if marker in combined:
            errors.append(f"forbidden synthetic implementation marker: {marker}")
    declared_models = [
        str(item.get("name", "") if isinstance(item, dict) else item)
        for item in plan.get("models", [])
    ]
    declared_datasets = [
        str(item.get("name", "") if isinstance(item, dict) else item)
        for item in plan.get("datasets", [])
    ]
    evidence["declared_models"] = declared_models
    evidence["declared_datasets"] = declared_datasets
    return {"ok": not errors, "errors": errors, "evidence": evidence}


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def validate_plan(value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "research_question",
        "hypothesis",
        "primary_metric",
        "pilot",
        "promotion_rule",
        "early_stop_rule",
        "estimand",
        "sample_size_rationale",
        "workload_budget",
        "decision_table",
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
    workload = value.get("workload_budget")
    if isinstance(workload, dict):
        for field in (
            "conditions",
            "models",
            "examples",
            "seeds",
            "max_new_tokens",
            "estimated_model_calls",
        ):
            try:
                number = int(workload[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"invalid workload_budget.{field}")
                continue
            if number <= 0:
                errors.append(f"invalid workload_budget.{field}")
    table = value.get("decision_table")
    if not isinstance(table, list) or not table:
        errors.append("decision_table must cover every outcome region")
    else:
        for index, row in enumerate(table):
            if not isinstance(row, dict):
                errors.append(f"invalid decision_table[{index}]")
                continue
            if not str(row.get("condition", "") or "").strip():
                errors.append(f"missing decision_table[{index}].condition")
            if str(row.get("decision", "") or "") not in {
                "promote",
                "retry",
                "reject",
            }:
                errors.append(f"invalid decision_table[{index}].decision")
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
            if isinstance(metric, bool) or not isinstance(
                metric,
                (int, float),
            ):
                errors.append(f"metric {name} must be numeric")
            elif not math.isfinite(float(metric)):
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
    else:
        for name, metric in value["metrics"].items():
            if isinstance(metric, bool) or not isinstance(
                metric,
                (int, float),
            ):
                errors.append(
                    f"runtime evidence metric {name} must be numeric"
                )
            elif not math.isfinite(float(metric)):
                errors.append(
                    f"runtime evidence metric {name} is not finite"
                )
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
    metrics_payload = metrics.get("metrics", {})
    measured = (
        metrics_payload.get("metrics", {})
        if isinstance(metrics_payload, dict)
        else {}
    )
    reported = runtime.get("evidence", {}).get("metrics", {})
    if (
        isinstance(measured, dict)
        and isinstance(reported, dict)
        and measured != reported
    ):
        errors.append(
            "metrics.json and runtime_evidence.json metrics disagree"
        )
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


def validate_runtime_against_contract(
    *,
    plan: dict[str, Any],
    runtime_evidence: dict[str, Any],
    allocated_gpus: int,
    mode: str,
    pilot_runtime: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        reported_gpus = int(runtime_evidence.get("gpu_count", -1))
    except (TypeError, ValueError):
        reported_gpus = -1
    if reported_gpus != int(allocated_gpus):
        errors.append(
            f"reported gpu_count={reported_gpus} does not match "
            f"allocation={allocated_gpus}"
        )
    pilot = plan.get("pilot", {})
    if mode == "pilot" and isinstance(pilot, dict):
        for evidence_field, plan_field in (
            ("examples_processed", "max_examples"),
        ):
            try:
                actual = int(runtime_evidence.get(evidence_field, -1))
                maximum = int(pilot.get(plan_field, -1))
            except (TypeError, ValueError):
                continue
            if maximum > 0 and actual > maximum:
                errors.append(
                    f"{evidence_field}={actual} exceeds pilot {plan_field}={maximum}"
                )
        seeds = runtime_evidence.get("seeds", [])
        try:
            max_seeds = int(pilot.get("max_seeds", -1))
        except (TypeError, ValueError):
            max_seeds = -1
        if (
            isinstance(seeds, list)
            and max_seeds > 0
            and len(seeds) > max_seeds
        ):
            errors.append(
                f"seed count={len(seeds)} exceeds pilot max_seeds={max_seeds}"
            )
    if mode == "scale" and pilot_runtime:
        try:
            scale_examples = int(
                runtime_evidence.get("examples_processed", -1)
            )
            pilot_examples = int(
                pilot_runtime.get("examples_processed", -1)
            )
        except (TypeError, ValueError):
            scale_examples = pilot_examples = -1
        scale_seeds = runtime_evidence.get("seeds", [])
        prior_seeds = pilot_runtime.get("seeds", [])
        if (
            scale_examples <= pilot_examples
            and len(scale_seeds if isinstance(scale_seeds, list) else [])
            <= len(prior_seeds if isinstance(prior_seeds, list) else [])
        ):
            errors.append(
                "scale run did not increase examples or seed coverage"
            )
    return errors
