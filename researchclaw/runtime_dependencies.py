"""Install small, required runtime dependencies for autonomous operation.

ResearchClaw is sometimes executed from a source checkout with a different
Python interpreter than the repository virtual environment.  Long-running RSI
services must therefore verify the interpreter that will actually launch the
pipeline, rather than assuming dependencies were installed elsewhere.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REQUIRED_RUNTIME_PACKAGES: Mapping[str, str] = {
    "arxiv": "arxiv>=2.1,<5",
}


@dataclass(frozen=True)
class DependencyResult:
    """Result of checking or installing one importable dependency."""

    module: str
    requirement: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "module": self.module,
            "requirement": self.requirement,
            "status": self.status,
            "detail": self.detail,
        }


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _import_version(module: str) -> str:
    try:
        imported = importlib.import_module(module)
    except (ImportError, ModuleNotFoundError):
        return ""
    return str(getattr(imported, "__version__", "") or "")


def ensure_runtime_dependencies(
    *,
    python_executable: str | Path = sys.executable,
    packages: Mapping[str, str] = REQUIRED_RUNTIME_PACKAGES,
    auto_install: bool = True,
    timeout_sec: float = 300.0,
    env: Mapping[str, str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[DependencyResult]:
    """Ensure required packages are importable in *python_executable*.

    The common in-process path avoids a subprocess when the current interpreter
    already has everything.  When another interpreter is selected, a small
    JSON probe is used so the check matches the process that will run the
    pipeline.
    """

    executable = str(Path(python_executable).expanduser())
    current = Path(sys.executable).resolve()
    selected = Path(executable).resolve()

    def available(module: str) -> tuple[bool, str]:
        if selected == current:
            return _module_available(module), _import_version(module)
        probe = (
            "import importlib.util,json,sys;"
            "name=sys.argv[1];"
            "spec=importlib.util.find_spec(name);"
            "version='';"
            "\nif spec is not None:\n"
            " import importlib\n"
            " value=importlib.import_module(name)\n"
            " version=str(getattr(value,'__version__','') or '')\n"
            "print(json.dumps({'available':spec is not None,'version':version}))"
        )
        try:
            completed = runner(
                [executable, "-c", probe, module],
                text=True,
                capture_output=True,
                check=False,
                timeout=max(1.0, timeout_sec),
                env=dict(env) if env is not None else None,
            )
            payload = json.loads(completed.stdout or "{}")
        except (
            json.JSONDecodeError,
            OSError,
            subprocess.SubprocessError,
            TypeError,
        ):
            return False, ""
        return bool(payload.get("available")), str(payload.get("version") or "")

    results: list[DependencyResult] = []
    for module, requirement in packages.items():
        found, version = available(module)
        if found:
            results.append(
                DependencyResult(
                    module,
                    requirement,
                    "present",
                    f"version {version}" if version else "importable",
                )
            )
            continue
        if not auto_install:
            results.append(
                DependencyResult(
                    module,
                    requirement,
                    "missing",
                    f"install with: {executable} -m pip install {requirement}",
                )
            )
            continue

        command = [
            executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            requirement,
        ]
        install_env = os.environ.copy()
        if env is not None:
            install_env.update({str(key): str(value) for key, value in env.items()})
        try:
            completed = runner(
                command,
                text=True,
                capture_output=True,
                check=False,
                timeout=max(1.0, timeout_sec),
                env=install_env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            results.append(
                DependencyResult(module, requirement, "failed", str(exc))
            )
            continue
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            results.append(
                DependencyResult(
                    module,
                    requirement,
                    "failed",
                    detail[-4000:],
                )
            )
            continue

        if selected == current:
            importlib.invalidate_caches()
        found, version = available(module)
        if found:
            results.append(
                DependencyResult(
                    module,
                    requirement,
                    "installed",
                    f"version {version}" if version else "importable",
                )
            )
        else:
            results.append(
                DependencyResult(
                    module,
                    requirement,
                    "failed",
                    "pip returned success but the module is still not importable",
                )
            )
    return results


def dependencies_satisfied(results: Sequence[DependencyResult]) -> bool:
    return all(result.status in {"present", "installed"} for result in results)
