"""ClusterBridge remote sandbox for generated GPU experiment code.

The control machine and GPU workers share CephFS, so experiment projects are
staged directly under ``/root/shared`` and only process execution is routed
through ``clusterbridge`` (``cb``).  This intentionally avoids SSH, matching
the cluster's operational contract.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from researchclaw.config import ClusterBridgeConfig
from researchclaw.experiment.sandbox import (
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)

logger = logging.getLogger(__name__)

_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ClusterBridgeSandbox:
    """Execute experiment projects on a claimed ClusterBridge GPU node."""

    def __init__(self, config: ClusterBridgeConfig, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.shared_root = Path(config.shared_root).expanduser().resolve()
        self.shared_root.mkdir(parents=True, exist_ok=True)
        self._run_counter = 0

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        self._run_counter += 1
        staging = self.workdir / f"_clusterbridge_run_{self._run_counter}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "main.py").write_text(code, encoding="utf-8")
        self._inject_harness(staging)
        return self._execute(staging, entry_point="main.py", timeout_sec=timeout_sec)

    def run_project(
        self,
        project_dir: Path,
        *,
        entry_point: str = "main.py",
        timeout_sec: int = 300,
        args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        self._run_counter += 1
        staging = self.workdir / f"_clusterbridge_project_{self._run_counter}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        err = validate_entry_point(entry_point)
        if err:
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=err,
                elapsed_sec=0.0,
                metrics={},
            )

        self._inject_harness(staging)
        for src_item in project_dir.iterdir():
            dest = staging / src_item.name
            if dest.name == "experiment_harness.py":
                logger.warning(
                    "Project contains experiment_harness.py — skipping (immutable)"
                )
                continue
            if src_item.is_dir() and not src_item.name.startswith((".", "__")):
                shutil.copytree(src_item, dest, dirs_exist_ok=True)
            elif src_item.is_file():
                dest.write_bytes(src_item.read_bytes())

        err = validate_entry_point_resolved(staging, entry_point)
        if err:
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=err,
                elapsed_sec=0.0,
                metrics={},
            )
        if not (staging / entry_point).exists():
            return SandboxResult(
                returncode=-1,
                stdout="",
                stderr=f"Entry point {entry_point} not found in project",
                elapsed_sec=0.0,
                metrics={},
            )

        return self._execute(
            staging,
            entry_point=entry_point,
            timeout_sec=timeout_sec,
            entry_args=args,
            env_overrides=env_overrides,
        )

    @staticmethod
    def check_available(config: ClusterBridgeConfig) -> tuple[bool, str]:
        """Check the transport and selected node without claiming extra nodes."""
        if not config.node:
            return False, "clusterbridge.node is empty"
        cb = Path(config.cb_command).expanduser()
        if not cb.is_file():
            return False, f"clusterbridge command not found: {cb}"
        try:
            cp = subprocess.run(
                ["bash", str(cb), config.node, "run", "printf researchclaw-cb-ok"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            return False, f"ClusterBridge check failed: {exc}"
        if cp.returncode == 0 and "researchclaw-cb-ok" in cp.stdout:
            return True, f"ClusterBridge node {config.node} OK"
        detail = cp.stderr.strip() or cp.stdout.strip()
        return False, f"ClusterBridge node check failed (exit {cp.returncode}): {detail}"

    @staticmethod
    def _inject_harness(target_dir: Path) -> None:
        harness_src = Path(__file__).parent / "harness_template.py"
        if harness_src.exists():
            (target_dir / "experiment_harness.py").write_text(
                harness_src.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    def _execute(
        self,
        staging_dir: Path,
        *,
        entry_point: str,
        timeout_sec: int,
        entry_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        run_id = f"rc-{uuid.uuid4().hex[:12]}"
        shared_run = self.shared_root / run_id
        shutil.copytree(staging_dir, shared_run)

        # Avoid accepting stale artifacts copied from an earlier local run.
        for artifact in ("results.json", ".clusterbridge_result.json"):
            try:
                (shared_run / artifact).unlink()
            except FileNotFoundError:
                pass

        effective_timeout = max(
            1,
            min(
                int(timeout_sec),
                int(self.config.timeout_sec)
                if self.config.timeout_sec > 0
                else int(timeout_sec),
            ),
        )
        command = self._build_remote_command(
            shared_run,
            entry_point=entry_point,
            args=entry_args,
            env_overrides=env_overrides,
            timeout_sec=effective_timeout,
        )
        cb_cmd = [
            "bash",
            str(Path(self.config.cb_command).expanduser()),
            self.config.node,
            "run",
            command,
        ]

        start = time.monotonic()
        timed_out = False
        stdout = ""
        stderr = ""
        returncode = -1
        try:
            cp = subprocess.run(
                cb_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=effective_timeout + 90,
                check=False,
                env={
                    **os.environ,
                    "CB_AUTO_TTL": str(max(15, self.config.claim_ttl_min)),
                    "BASHBRIDGE_TIMEOUT": str(effective_timeout + 60),
                },
            )
            stdout = cp.stdout
            stderr = cp.stderr
            returncode = cp.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _decode_timeout_stream(exc.stdout)
            stderr = _decode_timeout_stream(exc.stderr)
            returncode = -1
        except Exception as exc:  # noqa: BLE001
            stderr = f"ClusterBridge execution error: {exc}"

        elapsed = time.monotonic() - start
        metrics: dict[str, object] = dict(parse_metrics(stdout))

        result_path = shared_run / "results.json"
        if result_path.is_file():
            try:
                structured = json.loads(result_path.read_text(encoding="utf-8"))
                _merge_numeric_metrics(metrics, structured)
                # Preserve the remote artifact in the local sandbox workspace.
                shutil.copy2(result_path, staging_dir / "results.json")
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read remote results.json at %s", result_path)

        if self.config.cleanup_remote:
            shutil.rmtree(shared_run, ignore_errors=True)

        return SandboxResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_sec=elapsed,
            metrics=metrics,
            timed_out=timed_out or returncode == 124,
        )

    def _build_remote_command(
        self,
        shared_run: Path,
        *,
        entry_point: str,
        args: list[str] | None,
        env_overrides: dict[str, str] | None,
        timeout_sec: int | None = None,
    ) -> str:
        cfg = self.config
        rd = shlex.quote(str(shared_run))
        py = shlex.quote(cfg.remote_python)
        ep = shlex.quote(entry_point)
        arg_text = " ".join(shlex.quote(arg) for arg in (args or []))
        arg_suffix = f" {arg_text}" if arg_text else ""

        env_parts = [
            "PYTHONUNBUFFERED=1",
            "TOKENIZERS_PARALLELISM=false",
            f"HOME={shlex.quote(str(shared_run / '.home'))}",
            f"TORCH_HOME={shlex.quote(str(shared_run / '.cache' / 'torch'))}",
            f"MPLCONFIGDIR={shlex.quote(str(shared_run / '.cache' / 'matplotlib'))}",
        ]
        if cfg.gpu_ids:
            env_parts.append(
                "CUDA_VISIBLE_DEVICES="
                + shlex.quote(",".join(str(gpu) for gpu in cfg.gpu_ids))
            )
        for name, value in sorted((env_overrides or {}).items()):
            if value and _SAFE_ENV_NAME.match(name):
                env_parts.append(f"{name}={shlex.quote(str(value))}")
        env_prefix = " ".join(env_parts)

        setup_parts = [
            (
                f"mkdir -p {shlex.quote(str(shared_run / '.home'))} "
                f"{shlex.quote(str(shared_run / '.cache' / 'torch'))} "
                f"{shlex.quote(str(shared_run / '.cache' / 'matplotlib'))}"
            )
        ]
        setup_parts.extend(cfg.setup_commands)
        setup = " && ".join(f"({cmd})" for cmd in setup_parts)

        python_cmd = f"env {env_prefix} {py} -u {ep}{arg_suffix}"
        if cfg.network_isolation:
            python_cmd = (
                "if command -v unshare >/dev/null 2>&1; then "
                f"unshare --net {python_cmd}; "
                "else echo 'ERROR: unshare is required for network isolation' >&2; "
                "exit 126; fi"
            )

        # ClusterBridge's daemon has its own wall-clock cap; this inner timeout
        # ensures the experiment is terminated before the transport cap.
        remote_timeout = max(
            1,
            int(timeout_sec)
            if timeout_sec is not None
            else int(cfg.timeout_sec),
        )
        return (
            "set -euo pipefail; "
            f"cd {rd}; "
            f"{setup}; "
            f"timeout -k 30 {remote_timeout}s "
            f"bash -lc {shlex.quote(python_cmd)}"
        )


def _decode_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _merge_numeric_metrics(target: dict[str, object], value: object) -> None:
    """Merge numeric values from common results.json layouts."""
    if not isinstance(value, dict):
        return
    metrics = value.get("metrics")
    if isinstance(metrics, dict):
        for key, metric in metrics.items():
            if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                target.setdefault(str(key), float(metric))
    for key, metric in value.items():
        if isinstance(metric, (int, float)) and not isinstance(metric, bool):
            target.setdefault(str(key), float(metric))
