from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_factory_service_is_separate_and_restartable() -> None:
    unit = (
        ROOT / "deploy" / "systemd" / "researchclaw-factory@.service"
    ).read_text(encoding="utf-8")
    assert "researchclaw-factory" in unit
    assert "Restart=on-failure" in unit
    assert "autoresearch-rsi-supervisor" not in unit
    assert "ExecStart=" in unit
