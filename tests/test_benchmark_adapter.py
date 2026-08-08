from __future__ import annotations

import json

import numpy as np
import pytest

from researchclaw.benchmark_adapter.cifar10_calibration import (
    BenchmarkConfig,
    ContractError,
    _load_cifar10_test,
    _load_treatment,
    evaluate_probabilities,
    paired_bootstrap_mean_interval,
    run_from_logits_cache,
    sha256_path,
)


def test_evaluate_probabilities_reports_expected_metrics() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)
    metrics = evaluate_probabilities(probabilities, np.asarray([0, 1]), bins=2)
    assert metrics["accuracy"] == 1.0
    assert metrics["nll"] == pytest.approx(-(np.log(0.8) + np.log(0.6)) / 2)
    assert 0.0 <= metrics["ece"] <= 1.0


def test_evaluate_probabilities_rejects_invalid_contract() -> None:
    with pytest.raises(ContractError, match="probabilities"):
        evaluate_probabilities(np.asarray([[0.8, 0.8]]), np.asarray([0]))


def test_paired_bootstrap_mean_interval_is_deterministic() -> None:
    interval = paired_bootstrap_mean_interval([0.1, 0.2, 0.3, 0.4, 0.5])

    assert interval == paired_bootstrap_mean_interval([0.1, 0.2, 0.3, 0.4, 0.5])
    assert interval[0] > 0
    assert interval[0] <= 0.3 <= interval[1]


def test_treatment_plugin_contract_and_hash(tmp_path) -> None:
    path = tmp_path / "treatment.py"
    path.write_text(
        """
class IdentityTreatment:
    def fit(self, calibration_logits, calibration_labels):
        return {}

    def transform(self, logits, state):
        return logits


def build_treatment():
    return IdentityTreatment()
""".lstrip(),
        encoding="utf-8",
    )
    treatment = _load_treatment(path)
    logits = np.asarray([[1.0, 2.0]])
    assert np.array_equal(
        treatment.transform(logits, treatment.fit(logits, [1])), logits
    )
    assert len(sha256_path(path)) == 64


def test_treatment_plugin_requires_factory(tmp_path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("VALUE = {'bad': True}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="build_treatment"):
        _load_treatment(path)


def test_config_resolves_relative_paths_once(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
benchmark:
  cache_dir: cache
  output_dir: output
  treatment_path: treatment.py
""".lstrip(),
        encoding="utf-8",
    )
    config = BenchmarkConfig.from_file(config_path)
    assert config.cache_dir == (tmp_path / "cache").resolve()
    assert config.output_dir == (tmp_path / "output").resolve()


def test_load_cifar10_parquet_contract(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    from io import BytesIO

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(
        buffer,
        format="PNG",
    )
    path = tmp_path / "test.parquet"
    table = pa.table(
        {
            "img": [{"bytes": buffer.getvalue(), "path": None}],
            "label": [3],
        }
    )
    pq.write_table(table, path)
    images, labels = _load_cifar10_test(path, dataset_format="parquet")
    assert images.shape == (1, 3, 32, 32)
    assert labels.tolist() == [3]


def test_run_from_logits_cache_avoids_model_inference(tmp_path) -> None:
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
    output = tmp_path / "output"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
benchmark:
  cache_dir: {tmp_path / "cache"}
  output_dir: {output}
  treatment_path: {treatment}
  examples: 4
  calibration_examples: 4
  seeds: [17]
  corruption: gaussian_noise
  corruption_severity: 0.04
  ece_bins: 3
  require_cuda: true
  device: cuda
""".lstrip()
    )
    rng = np.random.default_rng(3)
    cache = tmp_path / "logits.npz"
    metadata = {
        "schema_version": 1,
        "seeds": [17],
        "ece_bins": 3,
        "assets": {"model_name": "fake"},
        "provenance": {
            "corruption": "gaussian_noise",
            "corruption_severity": 0.04,
            "examples": 4,
            "calibration_examples": 4,
        },
    }
    np.savez_compressed(
        cache,
        metadata_json=np.asarray(json.dumps(metadata)),
        calibration_logits_17=rng.normal(size=(4, 3)),
        calibration_labels_17=np.asarray([0, 1, 2, 0]),
        evaluation_logits_17=rng.normal(size=(4, 3)),
        evaluation_labels_17=np.asarray([0, 1, 2, 0]),
    )

    result = run_from_logits_cache(config, cache_path=cache)

    assert result.status == "ok"
    assert result.usage["device"] == "cpu-logits-cache"
    assert result.usage["gpu_count"] == 0
    assert np.isfinite(result.metrics["effect_nll"])
    assert result.metrics["effect_accuracy"] == pytest.approx(0.0)
