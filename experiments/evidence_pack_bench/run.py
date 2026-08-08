#!/usr/bin/env python3
"""Compare a legacy mean-only reviewer with the deterministic scientific gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from researchclaw.research_queue.benchmark_profile import TREATMENT_API
from researchclaw.research_queue.models import (
    MetricDirection,
    MetricGuardrail,
    MetricRelation,
    ResearchSpec,
)
from researchclaw.research_queue.promotion import review_benchmark_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/evidence-pack-bench"),
    )
    return parser


def _spec() -> ResearchSpec:
    return ResearchSpec(
        question="Does treatment improve calibration under corruption?",
        hypothesis="Treatment lowers ECE without worse NLL or changed argmax.",
        treatment="Candidate post-hoc calibration treatment.",
        control="Scalar temperature scaling.",
        primary_metric="ece",
        metric_direction=MetricDirection.MINIMIZE,
        guardrails=("accuracy unchanged", "NLL no worse"),
        validity_conditions=("frozen benchmark protocol",),
        compute_matching=("same examples, seeds, and model logits",),
        stopping_rules=("reject invalid or unsupported evidence",),
        benchmark_id="cifar10_calibration",
        treatment_api=TREATMENT_API,
        minimum_effect=0.0,
        primary_requires_effect_ci=True,
        guardrail_metrics=(
            MetricGuardrail(
                metric="accuracy",
                direction=MetricDirection.MAXIMIZE,
                relation=MetricRelation.EQUAL,
                tolerance=0.0,
                per_pair=True,
            ),
            MetricGuardrail(
                metric="nll",
                direction=MetricDirection.MINIMIZE,
                relation=MetricRelation.NO_WORSE,
                tolerance=0.0,
                require_effect_ci=True,
            ),
        ),
        minimum_pairs=5,
        confidence_level=0.95,
        calibration_split="clean",
        evaluation_split="corrupted",
        pairing_strategy="disjoint_example_blocks",
        require_per_example_argmax=True,
        required_compute_accounting=(
            "calibration_examples",
            "evaluation_examples",
            "model_forward_examples",
        ),
    )


def _base_result(
    *,
    baseline_ece: float = 0.08,
    treatment_ece: float = 0.04,
    baseline_nll: float = 1.0,
    treatment_nll: float = 0.9,
) -> dict[str, Any]:
    rows = []
    for index, seed in enumerate((17, 29, 43, 59, 71)):
        rows.append(
            {
                "seed": seed,
                "baseline": {
                    "ece": baseline_ece,
                    "accuracy": 0.7,
                    "nll": baseline_nll,
                },
                "treatment": {
                    "ece": treatment_ece,
                    "accuracy": 0.7,
                    "nll": treatment_nll,
                },
                "effect_ece": baseline_ece - treatment_ece,
                "effect_accuracy": 0.0,
                "effect_nll": baseline_nll - treatment_nll,
                "evidence": {
                    "argmax_preserved": True,
                    "argmax_changed_count": 0,
                    "baseline_predictions_sha256": f"baseline-{index}",
                    "treatment_predictions_sha256": f"treatment-{index}",
                },
            }
        )
    return {
        "status": "ok",
        "metrics": {
            "baseline_ece": baseline_ece,
            "treatment_ece": treatment_ece,
            "baseline_accuracy": 0.7,
            "treatment_accuracy": 0.7,
            "baseline_nll": baseline_nll,
            "treatment_nll": treatment_nll,
            "effect_ece": baseline_ece - treatment_ece,
            "effect_accuracy": 0.0,
            "effect_nll": baseline_nll - treatment_nll,
        },
        "uncertainty": {
            "effect_ece_ci": [
                baseline_ece - treatment_ece,
                baseline_ece - treatment_ece,
            ],
            "effect_accuracy_ci": [0.0, 0.0],
            "effect_nll_ci": [
                baseline_nll - treatment_nll,
                baseline_nll - treatment_nll,
            ],
        },
        "per_seed": rows,
        "evidence": {
            "protocol": {
                "calibration_split": "clean",
                "evaluation_split": "corrupted",
                "pairing_strategy": "disjoint_example_blocks",
                "independent_pairs": 5,
            },
            "argmax": {
                "argmax_preserved": True,
                "argmax_changed_count": 0,
                "per_example_prediction_hashes": True,
            },
            "compute_accounting": {
                "matched_dimensions": [
                    "calibration_examples",
                    "evaluation_examples",
                    "model_forward_examples",
                ],
                "all_declared_dimensions_matched": True,
            },
        },
    }


def _packs() -> list[dict[str, Any]]:
    packs: list[dict[str, Any]] = []

    def add(
        name: str,
        gold: str,
        mutate: Callable[[dict[str, Any]], None] | None = None,
        *,
        result: dict[str, Any] | None = None,
    ) -> None:
        value = copy.deepcopy(result or _base_result())
        if mutate is not None:
            mutate(value)
        packs.append({"id": name, "gold": gold, "result": value})

    add("valid-positive-clear", "positive")
    add(
        "valid-positive-small",
        "positive",
        result=_base_result(treatment_ece=0.07, treatment_nll=0.98),
    )
    add(
        "valid-positive-perfect-guardrails",
        "positive",
        result=_base_result(treatment_ece=0.05, treatment_nll=1.0),
    )
    add(
        "valid-negative-adverse-primary",
        "negative",
        result=_base_result(treatment_ece=0.10, treatment_nll=0.9),
    )
    add(
        "valid-negative-null-primary",
        "negative",
        result=_base_result(treatment_ece=0.08, treatment_nll=1.0),
    )

    add(
        "insufficient-two-pairs",
        "inconclusive",
        lambda value: value["per_seed"].__setitem__(
            slice(None), value["per_seed"][:2]
        ),
    )
    add(
        "insufficient-four-pairs",
        "inconclusive",
        lambda value: value["per_seed"].pop(),
    )
    add(
        "explicit-ci-omitted-but-reconstructable",
        "positive",
        lambda value: value["uncertainty"].pop("effect_ece_ci"),
    )
    add(
        "missing-nll-metric",
        "inconclusive",
        lambda value: value["metrics"].pop("treatment_nll"),
    )
    add(
        "nll-aggregate-regression",
        "inconclusive",
        result=_base_result(treatment_ece=0.04, treatment_nll=1.1),
    )

    def bad_nll_ci(value: dict[str, Any]) -> None:
        value["uncertainty"]["effect_nll_ci"] = [-0.02, 0.01]

    add("nll-ci-regression", "inconclusive", bad_nll_ci)

    def aggregate_accuracy_drop(value: dict[str, Any]) -> None:
        value["metrics"]["treatment_accuracy"] = 0.69

    add("accuracy-aggregate-drop", "inconclusive", aggregate_accuracy_drop)

    def per_pair_accuracy_swap(value: dict[str, Any]) -> None:
        value["per_seed"][0]["treatment"]["accuracy"] = 0.69
        value["per_seed"][1]["treatment"]["accuracy"] = 0.71

    add("accuracy-per-pair-failure", "inconclusive", per_pair_accuracy_swap)

    def changed_argmax(value: dict[str, Any]) -> None:
        value["per_seed"][0]["evidence"]["argmax_preserved"] = False
        value["per_seed"][0]["evidence"]["argmax_changed_count"] = 2
        value["evidence"]["argmax"]["argmax_preserved"] = False
        value["evidence"]["argmax"]["argmax_changed_count"] = 2

    add("argmax-changed-with-equal-accuracy", "inconclusive", changed_argmax)

    def missing_prediction_hash(value: dict[str, Any]) -> None:
        value["per_seed"][0]["evidence"].pop(
            "treatment_predictions_sha256"
        )

    add("missing-per-example-hash", "inconclusive", missing_prediction_hash)
    add(
        "wrong-calibration-split",
        "inconclusive",
        lambda value: value["evidence"]["protocol"].__setitem__(
            "calibration_split", "corrupted"
        ),
    )
    add(
        "wrong-evaluation-split",
        "inconclusive",
        lambda value: value["evidence"]["protocol"].__setitem__(
            "evaluation_split", "clean"
        ),
    )
    add(
        "overlapping-pairing-strategy",
        "inconclusive",
        lambda value: value["evidence"]["protocol"].__setitem__(
            "pairing_strategy", "overlapping_seed_resamples"
        ),
    )
    add(
        "missing-compute-dimension",
        "inconclusive",
        lambda value: value["evidence"]["compute_accounting"].__setitem__(
            "matched_dimensions",
            ["calibration_examples", "evaluation_examples"],
        ),
    )
    add(
        "compute-attestation-false",
        "inconclusive",
        lambda value: value["evidence"]["compute_accounting"].__setitem__(
            "all_declared_dimensions_matched", False
        ),
    )

    def nonfinite_metric(value: dict[str, Any]) -> None:
        value["metrics"]["treatment_nll"] = float("nan")

    add("nonfinite-guardrail", "inconclusive", nonfinite_metric)

    def execution_error(value: dict[str, Any]) -> None:
        value["status"] = "error"
        value["error"] = "worker failed"

    add("execution-error-with-residual-metrics", "inconclusive", execution_error)
    add(
        "missing-pair-rows",
        "inconclusive",
        lambda value: value.__setitem__("per_seed", []),
    )

    def crossing_primary_ci(value: dict[str, Any]) -> None:
        value["uncertainty"]["effect_ece_ci"] = [-0.01, 0.05]

    add("valid-negative-ci-crosses-zero", "negative", crossing_primary_ci)

    def malformed_status(value: dict[str, Any]) -> None:
        value["status"] = "unknown"

    add("unknown-status-with-positive-means", "inconclusive", malformed_status)
    return packs


def _legacy_mean_only(result: dict[str, Any]) -> str:
    """The pre-fix behavior: trust successful execution and aggregate means."""

    metrics = result.get("metrics", {})
    try:
        baseline = float(metrics["baseline_ece"])
        treatment = float(metrics["treatment_ece"])
    except (KeyError, TypeError, ValueError):
        return "inconclusive"
    if not math.isfinite(baseline) or not math.isfinite(treatment):
        return "inconclusive"
    return "positive" if treatment < baseline else "negative"


def _deterministic(result: dict[str, Any], spec: ResearchSpec) -> str:
    outcome = review_benchmark_result(
        spec=spec,
        benchmark_result=result,
        execution_passed=str(result.get("status", "")).casefold()
        in {"ok", "success", "succeeded", "passed"},
        execution_error=str(result.get("error", "") or ""),
    )
    if not outcome.scientific_valid:
        return "inconclusive"
    return "positive" if outcome.hypothesis_supported is True else "negative"


def _evaluate(
    name: str,
    packs: list[dict[str, Any]],
    reviewer: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for pack in packs:
        predicted = reviewer(copy.deepcopy(pack["result"]))
        rows.append(
            {
                "id": pack["id"],
                "gold": pack["gold"],
                "predicted": predicted,
                "correct": predicted == pack["gold"],
            }
        )
    elapsed = time.perf_counter() - started
    accepted = [row for row in rows if row["predicted"] == "positive"]
    false_accepts = [row for row in accepted if row["gold"] != "positive"]
    gold_positives = [row for row in rows if row["gold"] == "positive"]
    false_rejects = [
        row for row in gold_positives if row["predicted"] != "positive"
    ]
    return {
        "reviewer": name,
        "packs": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "verdict_accuracy": sum(row["correct"] for row in rows) / len(rows),
        "accepts": len(accepted),
        "false_accepts": len(false_accepts),
        "false_accept_rate": (
            len(false_accepts) / len(accepted) if accepted else None
        ),
        "gold_positives": len(gold_positives),
        "false_rejects": len(false_rejects),
        "false_reject_rate": (
            len(false_rejects) / len(gold_positives)
            if gold_positives
            else None
        ),
        "latency_seconds": elapsed,
        "token_cost": 0,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    packs = _packs()
    legacy = _evaluate("legacy_mean_only", packs, _legacy_mean_only)
    deterministic = _evaluate(
        "research_spec_deterministic_gate",
        packs,
        lambda result: _deterministic(result, spec),
    )
    pack_payload = {"research_spec": spec.to_dict(), "packs": packs}
    encoded_packs = json.dumps(
        pack_payload,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=True,
    ).encode()
    report = {
        "schema_version": 1,
        "pack_count": len(packs),
        "packs_sha256": hashlib.sha256(encoded_packs).hexdigest(),
        "legacy": legacy,
        "deterministic": deterministic,
        "improvement": {
            "verdict_accuracy_absolute": (
                deterministic["verdict_accuracy"] - legacy["verdict_accuracy"]
            ),
            "false_accept_rate_absolute": (
                deterministic["false_accept_rate"] - legacy["false_accept_rate"]
                if deterministic["false_accept_rate"] is not None
                and legacy["false_accept_rate"] is not None
                else None
            ),
            "false_accepts_avoided": (
                legacy["false_accepts"] - deterministic["false_accepts"]
            ),
        },
    }
    (output / "packs.json").write_text(
        json.dumps(
            pack_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "report.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
