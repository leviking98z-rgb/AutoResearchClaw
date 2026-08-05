from __future__ import annotations

import json
from pathlib import Path

from researchclaw.autoresearch_v2.validation import (
    validate_experiment_artifacts,
    validate_research_implementation,
    validate_runtime_against_contract,
)


def test_artifact_metrics_must_be_numeric_and_consistent(
    tmp_path: Path,
) -> None:
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "result_valid": True,
                "metrics": {"accuracy": "0.9"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runtime_evidence.json").write_text(
        json.dumps(
            {
                "model_loaded": "Qwen",
                "datasets_loaded": ["GSM8K"],
                "examples_processed": 10,
                "seeds": [0],
                "gpu_count": 1,
                "gate_decision": "promote",
                "metrics": {"accuracy": 0.8},
            }
        ),
        encoding="utf-8",
    )
    value = validate_experiment_artifacts(tmp_path)
    assert not value["ok"]
    assert any("numeric" in error for error in value["errors"])
    assert any("disagree" in error for error in value["errors"])


def test_research_implementation_requires_real_loaders(
    tmp_path: Path,
) -> None:
    (tmp_path / "main.py").write_text(
        "from pathlib import Path\n"
        "Path('metrics.json').write_text('{}')\n"
        "Path('runtime_evidence.json').write_text('{}')\n",
        encoding="utf-8",
    )
    value = validate_research_implementation(
        tmp_path,
        plan={"models": ["Qwen"], "datasets": ["GSM8K"]},
    )
    assert not value["ok"]
    assert "no real model loader call found" in value["errors"]
    assert "no real dataset/benchmark loader call found" in value["errors"]


def test_scale_must_expand_pilot_coverage() -> None:
    errors = validate_runtime_against_contract(
        plan={},
        runtime_evidence={
            "gpu_count": 1,
            "examples_processed": 10,
            "seeds": [0],
        },
        allocated_gpus=1,
        mode="scale",
        pilot_runtime={
            "examples_processed": 10,
            "seeds": [0],
        },
    )
    assert "scale run did not increase examples or seed coverage" in errors
