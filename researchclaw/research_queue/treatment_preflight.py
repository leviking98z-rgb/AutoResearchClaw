"""CPU-only contract preflight for generated calibration treatments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from researchclaw.benchmark_adapter.cifar10_calibration import _load_treatment


def preflight_treatment(
    path: str | Path,
    *,
    examples: int = 96,
    classes: int = 10,
    timeout_sec: float = 20.0,
) -> dict[str, Any]:
    treatment_path = Path(path).expanduser().resolve()
    command = [
        sys.executable,
        "-m",
        "researchclaw.research_queue.treatment_preflight",
        str(treatment_path),
        str(max(8, int(examples))),
        str(max(2, int(classes))),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1.0, float(timeout_sec)),
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError:
        value = {
            "passed": False,
            "error": completed.stderr.strip()
            or completed.stdout.strip()
            or f"preflight exited with {completed.returncode}",
        }
    if completed.returncode != 0:
        value["passed"] = False
        value.setdefault(
            "error",
            completed.stderr.strip() or f"preflight exited with {completed.returncode}",
        )
    return value


def _run(path: Path, examples: int, classes: int) -> dict[str, Any]:
    treatment = _load_treatment(path)
    rng = np.random.default_rng(20260808)
    calibration_logits = rng.normal(size=(examples, classes))
    calibration_labels = rng.integers(0, classes, size=examples, dtype=np.int64)
    evaluation_logits = rng.normal(size=(examples, classes))
    calibration_before = calibration_logits.copy()
    labels_before = calibration_labels.copy()
    evaluation_before = evaluation_logits.copy()
    state = treatment.fit(calibration_logits, calibration_labels)
    first = np.asarray(
        treatment.transform(evaluation_logits, state),
        dtype=np.float64,
    )
    second = np.asarray(
        treatment.transform(evaluation_logits.copy(), state),
        dtype=np.float64,
    )
    checks = {
        "fit_does_not_mutate_logits": np.array_equal(
            calibration_logits,
            calibration_before,
        ),
        "fit_does_not_mutate_labels": np.array_equal(
            calibration_labels,
            labels_before,
        ),
        "transform_does_not_mutate_input": np.array_equal(
            evaluation_logits,
            evaluation_before,
        ),
        "shape_preserved": first.shape == evaluation_before.shape,
        "finite_output": bool(np.isfinite(first).all()),
        "deterministic": bool(np.array_equal(first, second)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "state_type": type(state).__name__,
        "output_shape": list(first.shape),
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 3:
        raise SystemExit(
            "usage: treatment_preflight <treatment.py> <examples> <classes>"
        )
    path = Path(arguments[0]).expanduser().resolve()
    examples = int(arguments[1])
    classes = int(arguments[2])
    try:
        value = _run(path, examples, classes)
    except Exception as exc:  # noqa: BLE001
        value = {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0 if value.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
