from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "evidence_pack_bench"
        / "run.py"
    )
    spec = importlib.util.spec_from_file_location("evidence_pack_bench", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deterministic_gate_beats_legacy_mean_only_reviewer() -> None:
    module = _module()
    packs = module._packs()
    spec = module._spec()
    legacy = module._evaluate(
        "legacy",
        packs,
        module._legacy_mean_only,
    )
    deterministic = module._evaluate(
        "deterministic",
        packs,
        lambda result: module._deterministic(result, spec),
    )

    assert len(packs) == 25
    assert legacy["false_accept_rate"] > 0.8
    assert deterministic["verdict_accuracy"] == 1.0
    assert deterministic["false_accept_rate"] == 0.0
