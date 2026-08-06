from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "bin"
    / "autoresearch-v2-allocation-cleanup"
)
OWNER = "019fc877-7045-7a40-935d-d2bef7883945"
ALLOCATION = "alloc_unit123"


def _write_fake_cb(path: Path, *, owner: str = OWNER) -> None:
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "${{CALLS_FILE}}"
if [[ "$1" == "resource-status" ]]; then
  printf '%s\\n' '[AgentMail hook] unrelated prefix'
  printf '%s\\n' '{{"allocation":{{"owner":"{owner}","status":"active"}}}}'
  exit 0
fi
if [[ "$1" == "alloc" ]]; then
  exit 0
fi
if [[ "$1" == "alloc-release" ]]; then
  exit 0
fi
exit 9
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _run(tmp_path: Path, *, owner: str = OWNER) -> subprocess.CompletedProcess:
    cb = tmp_path / "cb"
    calls = tmp_path / "calls"
    log = tmp_path / "cleanup.log"
    _write_fake_cb(cb, owner=owner)
    env = {
        **os.environ,
        "AUTORESEARCH_V2_CLEANUP_ALLOCATION": ALLOCATION,
        "AUTORESEARCH_V2_CLEANUP_OWNER": OWNER,
        "AUTORESEARCH_V2_CLEANUP_CB": str(cb),
        "AUTORESEARCH_V2_CLEANUP_LOG": str(log),
        "CALLS_FILE": str(calls),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_cleanup_parses_json_after_hook_prefix(tmp_path: Path) -> None:
    completed = _run(tmp_path)

    assert completed.returncode == 0
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert f"resource-status {ALLOCATION} --json" in calls
    assert f"alloc {ALLOCATION} run-all" in calls
    assert f"alloc-release {ALLOCATION}" in calls
    log = (tmp_path / "cleanup.log").read_text(encoding="utf-8")
    assert "cleanup-release-succeeded" in log


def test_cleanup_refuses_owner_mismatch_with_prefixed_json(
    tmp_path: Path,
) -> None:
    completed = _run(
        tmp_path,
        owner="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    )

    assert completed.returncode == 3
    calls = (tmp_path / "calls").read_text(encoding="utf-8")
    assert "alloc-release" not in calls
    log = (tmp_path / "cleanup.log").read_text(encoding="utf-8")
    assert "cleanup-refused owner-mismatch" in log
