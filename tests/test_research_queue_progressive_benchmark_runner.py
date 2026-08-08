from __future__ import annotations

import json

import numpy as np

from researchclaw.benchmark_adapter.cifar10_calibration import sha256_path
from researchclaw.research_queue.progressive_benchmark_runner import (
    run_partition,
)


def test_progressive_partitions_reuse_one_treatment_and_disjoint_cache(
    tmp_path,
) -> None:
    treatment = tmp_path / "treatment.py"
    treatment.write_text(
        """
class Identity:
    def fit(self, calibration_logits, calibration_labels):
        return {}

    def transform(self, logits, state):
        return logits


def build_treatment():
    return Identity()
""".lstrip()
    )
    pairing_seeds = [101, 103, 17, 29]
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        f"""
benchmark:
  cache_dir: {tmp_path / "cache"}
  output_dir: placeholder
  treatment_path: placeholder.py
  examples: 4
  calibration_examples: 4
  pairing_seeds: {pairing_seeds}
  seeds: [17, 29]
  corruption: gaussian_noise
  corruption_severity: 0.04
  ece_bins: 3
  require_cuda: true
  device: cuda
""".lstrip()
    )
    rng = np.random.default_rng(13)
    arrays = {}
    for seed in pairing_seeds:
        arrays[f"calibration_logits_{seed}"] = rng.normal(size=(4, 3))
        arrays[f"calibration_labels_{seed}"] = np.asarray([0, 1, 2, 0])
        arrays[f"evaluation_logits_{seed}"] = rng.normal(size=(4, 3))
        arrays[f"evaluation_labels_{seed}"] = np.asarray([0, 1, 2, 0])
    metadata = {
        "schema_version": 4,
        "seeds": pairing_seeds,
        "selected_seeds": [17, 29],
        "ece_bins": 3,
        "assets": {"model_name": "fake"},
        "provenance": {
            "corruption": "gaussian_noise",
            "corruption_severity": 0.04,
            "examples": 4,
            "calibration_examples": 4,
            "calibration_split": "clean",
            "evaluation_split": "corrupted",
            "pairing_strategy": "disjoint_example_blocks",
            "pairing_seeds": pairing_seeds,
        },
    }
    cache = tmp_path / "partitioned.npz"
    np.savez_compressed(
        cache,
        metadata_json=np.asarray(json.dumps(metadata)),
        **arrays,
    )

    pilot = run_partition(
        benchmark_config=benchmark,
        treatment_path=treatment,
        output_dir=tmp_path / "pilot",
        parameters={
            "evidence_partition": "pilot-b0",
            "seeds": [101, 103],
        },
        logits_cache=cache,
    )
    final = run_partition(
        benchmark_config=benchmark,
        treatment_path=treatment,
        output_dir=tmp_path / "final",
        parameters={
            "evidence_partition": "confirmatory-b2",
            "seeds": [17, 29],
        },
        logits_cache=cache,
    )

    assert pilot["usage"]["selected_seeds"] == [101, 103]
    assert final["usage"]["selected_seeds"] == [17, 29]
    assert set(pilot["usage"]["selected_seeds"]).isdisjoint(
        final["usage"]["selected_seeds"]
    )
    assert pilot["usage"]["treatment_sha256"] == sha256_path(treatment)
    assert final["usage"]["treatment_sha256"] == sha256_path(treatment)
    assert pilot["metrics"]["independent_pairs"] == 2
    assert final["metrics"]["independent_pairs"] == 2
    assert (tmp_path / "pilot" / "benchmark-result.json").is_file()
    assert (tmp_path / "final" / "benchmark-result.json").is_file()
