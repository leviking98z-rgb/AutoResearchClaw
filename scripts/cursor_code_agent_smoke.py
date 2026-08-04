"""Smoke test for the Cursor-backed multi-file code-generation provider."""

from __future__ import annotations

import tempfile
from pathlib import Path

from researchclaw.config import RCConfig
from researchclaw.experiment.code_agent import create_code_agent


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = RCConfig.load(root / "config.arc.yaml", check_paths=False)
    agent = create_code_agent(config)
    with tempfile.TemporaryDirectory(prefix="rc-cursor-smoke-", dir="/data/tmp") as tmp:
        result = agent.generate(
            exp_plan=(
                "Compare the sample mean against a 10% trimmed mean on "
                "deterministic synthetic Gaussian data. Keep the program tiny."
            ),
            topic="Cursor code-agent installation smoke test",
            metric_key="primary_metric",
            pkg_hint="Use only numpy and json.",
            compute_budget="Complete in under 10 seconds.",
            extra_guidance=(
                "Create main.py and optionally one helper file. "
                "Do not download anything."
            ),
            workdir=Path(tmp),
            timeout_sec=180,
        )
        print(
            f"provider={result.provider_name} ok={result.ok} "
            f"elapsed={result.elapsed_sec:.1f}s files={sorted(result.files)}"
        )
        if not result.ok or "main.py" not in result.files:
            raise SystemExit(result.error or "Cursor code-agent smoke test failed")


if __name__ == "__main__":
    main()

