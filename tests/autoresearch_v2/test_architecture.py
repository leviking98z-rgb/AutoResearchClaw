from __future__ import annotations

from pathlib import Path


def test_v2_never_imports_legacy_control_plane() -> None:
    root = Path("researchclaw/autoresearch_v2")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.glob("*.py")
    )
    assert "from researchclaw.pipeline" not in source
    assert "import researchclaw.pipeline" not in source
    assert "from researchclaw.rsi" not in source
    assert "import researchclaw.rsi" not in source
    assert "PipelineIdeaWorker" not in source
    assert "CodeAgent" not in source
