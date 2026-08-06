"""Controller-owned runtime artifact compiler.

Generated experiment code owns scientific computation, but it does not own
the durable artifact schema or the compiled go/no-go decision.  Build injects
this file into every candidate and rewrites smoke/pilot/scale commands through
it.  The wrapper executes the generated core with ``shell=False``, preserves
its raw artifacts, and deterministically emits the canonical v2 artifacts.

The module intentionally uses only the Python standard library because the
same source is copied into candidate workspaces and executed on GPU nodes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

WRAPPER_FILENAME = "_autoresearch_runtime.py"
WRAPPER_SCHEMA = "autoresearch_v2.controller_runtime"
WRAPPER_VERSION = 5
RAW_DIRNAME = "_raw"


class RuntimeArtifactError(ValueError):
    """Raised when generated results cannot be compiled without guessing."""


def wrapper_source() -> str:
    """Return the exact self-contained source injected into candidates."""

    return Path(__file__).read_text(encoding="utf-8")


def compile_build_output(
    worker_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind Worker commands to the controller-owned runtime wrapper."""

    files = dict(worker_output.get("files", {}) or {})
    if WRAPPER_FILENAME in files:
        raise RuntimeArtifactError(
            f"{WRAPPER_FILENAME} is reserved for the Controller"
        )
    commands = worker_output.get("commands")
    if not isinstance(commands, Mapping):
        raise RuntimeArtifactError("commands must be an object")

    core_commands: dict[str, list[str]] = {}
    wrapped_commands: dict[str, list[str]] = {}
    for mode in ("smoke", "pilot", "scale"):
        core = _command_argv(commands.get(mode))
        _validate_core_argv(core, mode=mode)
        core = _bind_mode_output(core, mode=mode)
        output = f"artifacts/{mode}"
        core_commands[mode] = core
        wrapped_commands[mode] = [
            "python",
            WRAPPER_FILENAME,
            "--mode",
            mode,
            "--output",
            output,
            "--plan",
            "plan.json",
            "--",
            *core,
        ]

    files[WRAPPER_FILENAME] = wrapper_source()
    return {
        **dict(worker_output),
        "files": files,
        "commands": wrapped_commands,
        "controller_runtime": {
            "schema": WRAPPER_SCHEMA,
            "version": WRAPPER_VERSION,
            "wrapper": WRAPPER_FILENAME,
            "core_commands": core_commands,
        },
    }


def normalize_runtime_artifacts(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    mode: str,
    allocated_gpus: int,
    core_returncode: int = 0,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Compile raw generated artifacts into the canonical v2 contract.

    The compiler may normalize representation (for example ``seed`` to
    ``seeds`` or a list of criterion rows to an id-keyed object), and it may
    derive plan-owned metadata and decisions.  It never invents a missing
    measured metric, model identifier, dataset load, example count, seed, or
    call count.
    """

    mode = str(mode).casefold()
    if mode not in {"smoke", "pilot", "scale"}:
        raise RuntimeArtifactError(f"unsupported execution mode: {mode!r}")
    output_dir = output_dir.resolve()
    cwd = (cwd or Path.cwd()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = _locate_artifact(output_dir, cwd, "metrics.json")
    runtime_path = _locate_artifact(
        output_dir,
        cwd,
        "runtime_evidence.json",
    )
    raw_metrics = _read_object(metrics_path, label="metrics.json")
    raw_runtime = _read_object(
        runtime_path,
        label="runtime_evidence.json",
    )
    if _is_canonical_artifact_pair(raw_metrics, raw_runtime):
        return {
            "metrics": raw_metrics,
            "runtime_evidence": raw_runtime,
            "raw_dir": str(output_dir / RAW_DIRNAME),
            "already_compiled": True,
        }

    raw_dir = output_dir / RAW_DIRNAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_json(raw_dir / "metrics.json", raw_metrics)
    _write_json(raw_dir / "runtime_evidence.json", raw_runtime)

    metrics, metric_diagnostics = _numeric_metrics(
        raw_metrics=raw_metrics,
        raw_runtime=raw_runtime,
    )
    if mode == "smoke":
        # This is an operational integration metric owned by the wrapper, not
        # a scientific result.  Trusted GPU telemetry independently verifies
        # that physical CUDA work occurred.
        metrics["smoke_forward_pass"] = (
            1.0 if int(core_returncode) == 0 else 0.0
        )
    if not metrics:
        raise RuntimeArtifactError(
            "generated artifacts contain no finite scalar metrics"
        )
    uncertainty = _compile_uncertainty_evidence(
        plan=plan,
        raw_runtime=raw_runtime,
        metrics=metrics,
        mode=mode,
    )

    model_loaded, model_metadata = _normalize_model_loaded(
        raw_runtime.get("model_loaded")
    )
    if not model_loaded:
        raise RuntimeArtifactError(
            "runtime_evidence.model_loaded must be a non-empty model id"
        )
    datasets_loaded = _normalize_datasets_loaded(
        raw_runtime.get("datasets_loaded")
    )
    if not datasets_loaded:
        raise RuntimeArtifactError(
            "runtime_evidence.datasets_loaded must be a non-empty list"
        )
    reported_examples_processed = _nonnegative_int(
        raw_runtime.get("examples_processed"),
        "runtime_evidence.examples_processed",
    )
    seeds = _normalize_seeds(raw_runtime)
    examples_by_role = _normalize_examples_by_role(
        raw_runtime.get("examples_by_role")
    )
    examples_processed, example_diagnostics = _normalize_examples_processed(
        mode=mode,
        reported=reported_examples_processed,
        examples_by_role=examples_by_role,
    )
    call_counts = _normalize_call_counts(raw_runtime.get("call_counts"))
    _validate_call_counts(
        plan=plan,
        mode=mode,
        call_counts=call_counts,
    )
    dataset_roles, split_identifiers = _compile_dataset_contract(
        plan,
        mode=mode,
    )

    runtime: dict[str, Any] = {
        "wrapper_schema": WRAPPER_SCHEMA,
        "wrapper_version": WRAPPER_VERSION,
        "mode": mode,
        "model_loaded": model_loaded,
        "datasets_loaded": datasets_loaded,
        "examples_processed": examples_processed,
        "seeds": seeds,
        "gpu_count": max(0, int(allocated_gpus)),
        "metrics": metrics,
    }
    if examples_by_role:
        runtime["examples_by_role"] = examples_by_role
    if model_metadata:
        runtime["model_metadata"] = model_metadata
    if call_counts:
        runtime["call_counts"] = call_counts
    if dataset_roles:
        runtime["dataset_roles"] = dataset_roles
    if split_identifiers:
        runtime["split_identifiers"] = split_identifiers
    if metric_diagnostics:
        runtime["metric_diagnostics"] = metric_diagnostics
    if example_diagnostics:
        runtime["example_diagnostics"] = example_diagnostics
    if uncertainty:
        runtime["uncertainty"] = uncertainty

    if mode == "smoke":
        evidence_valid = (
            int(core_returncode) == 0 and examples_processed > 0
        )
        runtime.update(
            {
                "evidence_valid": evidence_valid,
                "gate_statistic_defined": False,
                "gate_decision": (
                    "promote" if evidence_valid else "retry"
                ),
            }
        )
    else:
        criterion_results = _compile_criterion_results(
            plan=plan,
            raw_runtime=raw_runtime,
            raw_metrics=raw_metrics,
            metrics=metrics,
        )
        validity_ids = _criterion_ids(plan, "validity_criteria")
        promotion_ids = _criterion_ids(plan, "promotion_criteria")
        raw_evidence_valid = raw_runtime.get("evidence_valid")
        computed_validity = all(
            criterion_results.get(identifier, {}).get("passed") is True
            for identifier in validity_ids
        )
        evidence_valid = (
            raw_evidence_valid is not False and computed_validity
        )
        gate_name = str(
            (
                plan.get("gate_statistic")
                if isinstance(plan.get("gate_statistic"), Mapping)
                else {}
            ).get("name", "")
            or ""
        )
        gate_defined = bool(
            gate_name and _finite_number(metrics.get(gate_name))
        )
        raw_gate_defined = raw_runtime.get("gate_statistic_defined")
        if raw_gate_defined is False:
            gate_defined = False
        promotion_pass = all(
            criterion_results.get(identifier, {}).get("passed") is True
            for identifier in promotion_ids
        )
        if not evidence_valid:
            decision = "retry"
        elif not gate_defined:
            decision = "reject"
        elif promotion_pass:
            decision = "promote"
        else:
            decision = "reject"
        runtime.update(
            {
                "evidence_valid": evidence_valid,
                "gate_statistic_defined": gate_defined,
                "criterion_results": criterion_results,
                "gate_decision": decision,
            }
        )

    metrics_out = {
        "result_valid": runtime["evidence_valid"] is True,
        "metrics": metrics,
        "decision": runtime["gate_decision"],
        "wrapper_schema": WRAPPER_SCHEMA,
        "wrapper_version": WRAPPER_VERSION,
    }
    _write_json(output_dir / "metrics.json", metrics_out)
    _write_json(output_dir / "runtime_evidence.json", runtime)
    return {
        "metrics": metrics_out,
        "runtime_evidence": runtime,
        "raw_dir": str(raw_dir),
    }


def _numeric_metrics(
    *,
    raw_metrics: Mapping[str, Any],
    raw_runtime: Mapping[str, Any],
) -> tuple[dict[str, float | int], dict[str, Any]]:
    candidates: dict[str, Any] = {}
    runtime_metrics = raw_runtime.get("metrics")
    if isinstance(runtime_metrics, Mapping):
        candidates.update(runtime_metrics)
    nested = raw_metrics.get("metrics")
    if isinstance(nested, Mapping):
        candidates.update(nested)
    else:
        candidates.update(raw_metrics)

    metrics: dict[str, float | int] = {}
    diagnostics: dict[str, Any] = {}
    reserved = {
        "result_valid",
        "decision",
        "criterion_results",
        "gate_decision",
        "gate_statistic_name",
        "wrapper_schema",
        "wrapper_version",
    }
    for raw_name, value in candidates.items():
        name = str(raw_name)
        if name in reserved:
            continue
        if _finite_number(value):
            metrics[name] = value
        else:
            diagnostics[name] = value
    return metrics, diagnostics


def _compile_criterion_results(
    *,
    plan: Mapping[str, Any],
    raw_runtime: Mapping[str, Any],
    raw_metrics: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_results = _criterion_rows(
        raw_runtime.get(
            "criterion_results",
            raw_metrics.get("criterion_results"),
        )
    )
    compiled: dict[str, dict[str, Any]] = {}
    for field in ("validity_criteria", "promotion_criteria"):
        criteria = plan.get(field)
        if not isinstance(criteria, list):
            continue
        for criterion in criteria:
            if not isinstance(criterion, Mapping):
                continue
            identifier = str(criterion.get("id", "") or "")
            metric_name = str(criterion.get("metric", "") or "")
            if not identifier:
                continue
            row = raw_results.get(identifier, {})
            value = row.get("value")
            if not _finite_number(value):
                value = metrics.get(metric_name)
            if not _finite_number(value):
                raise RuntimeArtifactError(
                    "missing measured criterion metric "
                    f"{metric_name!r} for {identifier!r}"
                )
            passed = _criterion_passes(criterion, float(value))
            compiled[identifier] = {
                "value": value,
                "passed": bool(passed),
            }
    if not compiled:
        raise RuntimeArtifactError(
            "compiled protocol contains no executable criteria"
        )
    return compiled


def _criterion_rows(value: Any) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(value, Mapping):
        for identifier, row in value.items():
            if isinstance(row, Mapping):
                rows[str(identifier)] = dict(row)
        return rows
    if not isinstance(value, list):
        return rows
    for row in value:
        if not isinstance(row, Mapping):
            continue
        identifier = str(row.get("id", "") or "")
        if identifier:
            rows[identifier] = dict(row)
    return rows


def _criterion_ids(
    plan: Mapping[str, Any],
    field: str,
) -> list[str]:
    criteria = plan.get(field)
    if not isinstance(criteria, list):
        return []
    return [
        str(item.get("id"))
        for item in criteria
        if isinstance(item, Mapping) and str(item.get("id", "") or "")
    ]


def _criterion_passes(
    criterion: Mapping[str, Any],
    measured: float,
) -> bool:
    try:
        threshold = float(criterion.get("value"))
    except (TypeError, ValueError) as exc:
        raise RuntimeArtifactError(
            f"criterion {criterion.get('id')!r} has invalid threshold"
        ) from exc
    operator = str(criterion.get("operator", "") or "")
    if operator == "<":
        return measured < threshold
    if operator == "<=":
        return measured <= threshold
    if operator == ">":
        return measured > threshold
    if operator == ">=":
        return measured >= threshold
    if operator in {"=", "=="}:
        return math.isclose(
            measured,
            threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    raise RuntimeArtifactError(
        f"criterion {criterion.get('id')!r} has unsupported operator "
        f"{operator!r}"
    )


def _compile_dataset_contract(
    plan: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    active_roles = (
        {"development"}
        if mode == "smoke"
        else {"development", "screening"}
        if mode == "pilot"
        else {"development", "confirmatory"}
    )
    datasets = plan.get("datasets")
    if not isinstance(datasets, list):
        return {}, {}
    roles: dict[str, dict[str, Any]] = {}
    split_ids: dict[str, str] = {}
    for item in datasets:
        if not isinstance(item, Mapping):
            continue
        role = _dataset_role(item.get("split_role"))
        if role not in active_roles:
            continue
        name = str(item.get("name", "") or "").strip()
        split_id = str(
            item.get("split_id", item.get("split_identifier", "")) or ""
        ).strip()
        if not name or not split_id:
            continue
        declaration: dict[str, Any] = {
            "role": role,
            "split_id": split_id,
        }
        resource_id = str(item.get("resource_id", "") or "").strip()
        if resource_id:
            declaration["resource_id"] = resource_id
        if role == "confirmatory":
            declaration["untouched"] = bool(
                item.get(
                    "untouched",
                    item.get("split_untouched", False),
                )
            )
        roles[name] = declaration
        split_ids[role] = split_id
    return roles, split_ids


def _dataset_role(value: Any) -> str:
    normalized = (
        str(value).casefold().replace("-", "_").replace(" ", "_")
    )
    if normalized in {"dev", "development"}:
        return "development"
    if normalized in {
        "screening",
        "screening_pilot",
        "screening_evaluation",
    }:
        return "screening"
    if normalized in {
        "confirmatory",
        "heldout",
        "heldout_confirmatory",
        "confirmatory_heldout",
    }:
        return "confirmatory"
    return ""


def _normalize_examples_by_role(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, int] = {}
    for raw_role, raw_count in value.items():
        role = _dataset_role(raw_role)
        if not role:
            continue
        normalized[role] = _nonnegative_int(
            raw_count,
            f"examples_by_role[{raw_role!r}]",
        )
    return normalized


def _normalize_examples_processed(
    *,
    mode: str,
    reported: int,
    examples_by_role: Mapping[str, int],
) -> tuple[int, dict[str, Any]]:
    """Canonicalize the evaluated-example count without guessing a split.

    ``pilot.max_examples`` and ``confirmatory_followup.examples`` describe
    endpoint evaluation units. Development examples have a separate budget in
    the compiled protocol, so they must not be added to
    ``examples_processed``. Older generated workers reported the sum of every
    role. We can migrate that representation only when the worker also
    provided an explicit role accounting whose exact sum matches the scalar.
    """

    endpoint_role = "screening" if mode in {"smoke", "pilot"} else "confirmatory"
    endpoint_count = examples_by_role.get(endpoint_role)
    if endpoint_count is None:
        return reported, {}
    if reported == endpoint_count:
        return reported, {}
    role_total = sum(examples_by_role.values())
    if reported == role_total:
        return endpoint_count, {
            "reported_examples_processed": reported,
            "canonical_examples_processed": endpoint_count,
            "normalization": "legacy_all_roles_total_to_endpoint_count",
        }
    raise RuntimeArtifactError(
        "runtime_evidence.examples_processed must equal "
        f"examples_by_role[{endpoint_role!r}]={endpoint_count}; "
        f"reported {reported}"
    )


def _compile_uncertainty_evidence(
    *,
    plan: Mapping[str, Any],
    raw_runtime: Mapping[str, Any],
    metrics: Mapping[str, float | int],
    mode: str,
) -> dict[str, Any]:
    """Expose auditable uncertainty without inventing unavailable samples.

    Generated workers may emit a complete uncertainty object. Otherwise a
    single-seed screening run cannot reconstruct a paired item bootstrap from
    aggregate counts, so the controller records that limitation explicitly.
    This does not invalidate a point-estimate decision unless a typed validity
    or promotion criterion references an uncertainty metric.
    """

    if mode == "smoke":
        return {}
    configured = plan.get("uncertainty")
    if not isinstance(configured, Mapping):
        return {}
    raw = raw_runtime.get("uncertainty")
    if isinstance(raw, Mapping):
        supplied = dict(raw)
        supplied.setdefault("method", configured.get("method"))
        supplied.setdefault(
            "confidence_level",
            configured.get("confidence_level"),
        )
        supplied.setdefault("decision_role", "descriptive")
        return supplied
    gate = plan.get("gate_statistic")
    gate_name = str(
        gate.get("name", "") if isinstance(gate, Mapping) else ""
    )
    return {
        "available": False,
        "method": configured.get("method"),
        "confidence_level": configured.get("confidence_level"),
        "resamples": configured.get("resamples"),
        "rng_seed": configured.get("rng_seed"),
        "undefined_resample_policy": configured.get(
            "undefined_resample_policy"
        ),
        "decision_role": configured.get(
            "decision_role",
            "descriptive",
        ),
        "metric": gate_name,
        "point_estimate": metrics.get(gate_name),
        "reason": "item_level_paired_observations_not_emitted",
    }


def _normalize_call_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    counts: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        name = str(raw_name).strip()
        if not name or name == "total":
            continue
        counts[name] = _nonnegative_int(
            raw_count,
            f"call_counts[{name!r}]",
        )
    return counts


def _validate_call_counts(
    *,
    plan: Mapping[str, Any],
    mode: str,
    call_counts: Mapping[str, int],
) -> None:
    if mode == "smoke":
        return
    ledger = plan.get("call_ledger")
    if not isinstance(ledger, Mapping):
        return
    components = ledger.get("components")
    if not isinstance(components, list):
        return
    required = {
        str(component.get("name", "") or "").strip()
        for component in components
        if isinstance(component, Mapping)
        and str(component.get("name", "") or "").strip()
    }
    missing = sorted(required - set(call_counts))
    if missing:
        raise RuntimeArtifactError(
            "runtime_evidence.call_counts missing compiled components: "
            + ", ".join(missing)
        )


def _normalize_seeds(runtime: Mapping[str, Any]) -> list[Any]:
    raw = runtime.get("seeds")
    if raw is None and "seed" in runtime:
        raw = [runtime["seed"]]
    if not isinstance(raw, list) or not raw:
        raise RuntimeArtifactError(
            "runtime_evidence must provide seeds or scalar seed"
        )
    if any(isinstance(item, (dict, list, set, tuple)) for item in raw):
        raise RuntimeArtifactError("runtime seeds must be scalar identifiers")
    identities = {f"{type(item).__name__}:{item!r}" for item in raw}
    if len(identities) != len(raw):
        raise RuntimeArtifactError("runtime seeds must be unique")
    return list(raw)


def _normalize_model_loaded(value: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(value, str):
        return value.strip(), {}
    if not isinstance(value, Mapping):
        return "", {}
    model_id = str(
        value.get("model_id")
        or value.get("id")
        or value.get("name")
        or value.get("resource_id")
        or ""
    ).strip()
    if not model_id:
        return "", {}
    metadata = {
        str(key): item
        for key, item in value.items()
        if str(key) not in {"model_id", "id", "name", "resource_id"}
    }
    return model_id, metadata


def _normalize_datasets_loaded(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    datasets: list[str] = []
    for item in value:
        if isinstance(item, str):
            identifier = item.strip()
        elif isinstance(item, Mapping):
            identifier = str(
                item.get("resource_id")
                or item.get("dataset_id")
                or item.get("id")
                or item.get("name")
                or ""
            ).strip()
        else:
            identifier = ""
        if identifier and identifier not in datasets:
            datasets.append(identifier)
    return datasets


def _locate_artifact(
    output_dir: Path,
    cwd: Path,
    name: str,
) -> Path:
    preferred = output_dir / name
    if preferred.is_file():
        return preferred
    fallback = cwd / name
    if fallback.is_file():
        return fallback
    raise RuntimeArtifactError(f"generated runtime did not write {name}")


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeArtifactError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeArtifactError(f"{label} root must be an object")
    return dict(value)


def _is_canonical_artifact_pair(
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> bool:
    metric_marker = metrics.get("wrapper_schema")
    runtime_marker = runtime.get("wrapper_schema")
    if metric_marker != WRAPPER_SCHEMA and runtime_marker != WRAPPER_SCHEMA:
        return False
    if (
        metric_marker != WRAPPER_SCHEMA
        or runtime_marker != WRAPPER_SCHEMA
    ):
        raise RuntimeArtifactError(
            "controller runtime artifacts have inconsistent wrapper markers"
        )
    if metrics.get("wrapper_version") != WRAPPER_VERSION or (
        runtime.get("wrapper_version") != WRAPPER_VERSION
    ):
        raise RuntimeArtifactError(
            "controller runtime artifact version is unsupported"
        )
    measured = metrics.get("metrics")
    reported = runtime.get("metrics")
    if not isinstance(measured, Mapping) or dict(measured) != dict(
        reported if isinstance(reported, Mapping) else {}
    ):
        raise RuntimeArtifactError(
            "canonical controller runtime metrics disagree"
        )
    return True


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _command_argv(value: Any) -> list[str]:
    if isinstance(value, list) and all(
        isinstance(item, str) and item for item in value
    ):
        return list(value)
    if isinstance(value, str):
        import shlex

        return shlex.split(value)
    return []


def _validate_core_argv(argv: list[str], *, mode: str) -> None:
    if len(argv) < 2:
        raise RuntimeArtifactError(
            f"commands.{mode} must be a direct Python entrypoint"
        )
    if Path(argv[0]).name.casefold() not in {
        "python",
        "python3",
        "python3.10",
        "python3.11",
        "python3.12",
    }:
        raise RuntimeArtifactError(
            f"commands.{mode} must use an approved Python interpreter"
        )
    entrypoint = Path(argv[1])
    if (
        entrypoint.is_absolute()
        or ".." in entrypoint.parts
        or entrypoint.suffix != ".py"
        or entrypoint.name == WRAPPER_FILENAME
    ):
        raise RuntimeArtifactError(
            f"commands.{mode} has an invalid generated entrypoint"
        )
    for argument in argv:
        if any(
            token in argument
            for token in (";", "&&", "||", "|", ">", "<", "`", "$(")
        ):
            raise RuntimeArtifactError(
                f"commands.{mode} contains shell syntax"
            )


def _bind_mode_output(argv: list[str], *, mode: str) -> list[str]:
    bound = list(argv)
    _replace_option(bound, "--mode", mode)
    _replace_option(bound, "--output", f"artifacts/{mode}")
    return bound


def _replace_option(argv: list[str], option: str, value: str) -> None:
    for index, item in enumerate(argv):
        if item == option and index + 1 < len(argv):
            argv[index + 1] = value
            return
        if item.startswith(option + "="):
            argv[index] = f"{option}={value}"
            return


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeArtifactError(f"{label} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeArtifactError(
            f"{label} must be a non-negative integer"
        ) from exc
    if parsed < 0:
        raise RuntimeArtifactError(f"{label} must be non-negative")
    return parsed


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _prepare_core_artifacts(output_dir: Path, cwd: Path) -> None:
    for root in {output_dir.resolve(), cwd.resolve()}:
        for name in (
            "metrics.json",
            "runtime_evidence.json",
            "runtime_wrapper_error.json",
        ):
            path = root / name
            if path.is_file():
                path.unlink()
    shutil.rmtree(output_dir / RAW_DIRNAME, ignore_errors=True)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "pilot", "scale"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--plan", default="plan.json")
    parser.add_argument("core_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    core_argv = list(args.core_argv)
    if core_argv and core_argv[0] == "--":
        core_argv.pop(0)
    _validate_core_argv(core_argv, mode=args.mode)
    core_argv[0] = sys.executable

    cwd = Path.cwd().resolve()
    output_dir = Path(
        os.environ.get("AUTORESEARCH_V2_OUTPUT_DIR", args.output)
    ).resolve()
    plan = _read_object((cwd / args.plan).resolve(), label="plan.json")
    _prepare_core_artifacts(output_dir, cwd)

    completed = subprocess.run(
        core_argv,
        cwd=cwd,
        env=os.environ.copy(),
        shell=False,
        check=False,
    )
    if completed.returncode != 0:
        return int(completed.returncode)
    try:
        normalize_runtime_artifacts(
            output_dir=output_dir,
            plan=plan,
            mode=args.mode,
            allocated_gpus=int(
                os.environ.get("AUTORESEARCH_V2_GPU_COUNT", "0") or 0
            ),
            core_returncode=completed.returncode,
            cwd=cwd,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed in copied script.
        _write_json(
            output_dir / "runtime_wrapper_error.json",
            {
                "schema": WRAPPER_SCHEMA,
                "version": WRAPPER_VERSION,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print(
            f"controller runtime artifact compilation failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 86
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
