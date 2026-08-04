"""Tests for automatic ResearchClaw runtime dependency installation."""

from __future__ import annotations

import subprocess
import sys

from researchclaw.runtime_dependencies import (
    dependencies_satisfied,
    ensure_runtime_dependencies,
)


def test_present_dependency_does_not_run_pip() -> None:
    calls: list[list[str]] = []

    def forbidden(args, **_kwargs):
        calls.append(list(args))
        raise AssertionError("pip must not run for an importable module")

    results = ensure_runtime_dependencies(
        packages={"json": "json"},
        runner=forbidden,
    )

    assert dependencies_satisfied(results)
    assert results[0].status == "present"
    assert calls == []


def test_missing_dependency_installs_with_selected_interpreter() -> None:
    calls: list[list[str]] = []
    probes = iter(
        [
            subprocess.CompletedProcess(
                [],
                0,
                stdout='{"available": false, "version": ""}',
                stderr="",
            ),
            subprocess.CompletedProcess([], 0, stdout="installed", stderr=""),
            subprocess.CompletedProcess(
                [],
                0,
                stdout='{"available": true, "version": "4.0.1"}',
                stderr="",
            ),
        ]
    )

    def fake_runner(args, **_kwargs):
        calls.append([str(value) for value in args])
        return next(probes)

    results = ensure_runtime_dependencies(
        python_executable="/opt/test/bin/python3",
        packages={"arxiv": "arxiv>=2.1,<5"},
        runner=fake_runner,
    )

    assert dependencies_satisfied(results)
    assert results[0].status == "installed"
    assert calls[1] == [
        "/opt/test/bin/python3",
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "arxiv>=2.1,<5",
    ]


def test_check_only_reports_actionable_missing_dependency() -> None:
    def fake_runner(_args, **_kwargs):
        return subprocess.CompletedProcess(
            [],
            0,
            stdout='{"available": false, "version": ""}',
            stderr="",
        )

    results = ensure_runtime_dependencies(
        python_executable="/opt/test/bin/python3",
        packages={"arxiv": "arxiv>=2.1,<5"},
        auto_install=False,
        runner=fake_runner,
    )

    assert not dependencies_satisfied(results)
    assert results[0].status == "missing"
    assert "/opt/test/bin/python3 -m pip install" in results[0].detail


def test_current_interpreter_probe_uses_import_system() -> None:
    results = ensure_runtime_dependencies(
        python_executable=sys.executable,
        packages={"definitely_missing_researchclaw_test_module": "fake-package"},
        auto_install=False,
    )

    assert results[0].status == "missing"
