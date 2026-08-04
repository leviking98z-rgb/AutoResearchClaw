"""Sandbox adapter for an already prepared multi-node ClusterBridge/Ray pool."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import uuid
from pathlib import Path

from researchclaw.config import ClusterBridgePoolSandboxConfig
from researchclaw.experiment.clusterbridge_pool import (
    ClusterBridgePool,
    ClusterPoolError,
    PoolNotReadyError,
    PoolTaskTimeout,
)
from researchclaw.experiment.sandbox import (
    SandboxResult,
    parse_metrics,
    validate_entry_point,
    validate_entry_point_resolved,
)

logger = logging.getLogger(__name__)

_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ClusterBridgePoolSandbox:
    """Submit generated experiment projects through ``ClusterBridgePool``.

    The adapter does not claim, release, clean, or restart the shared pool.
    Those lifecycle operations remain with the external pool controller.  Each
    sandbox run restores the durable pool state, verifies readiness, and
    submits a detached task to the head node with ``RAY_ADDRESS`` exported.
    Generated code can therefore use the full 32-GPU Ray allocation instead
    of being pinned to a single ClusterBridge node.
    """

    def __init__(
        self,
        config: ClusterBridgePoolSandboxConfig,
        workdir: Path,
        *,
        pool_factory=ClusterBridgePool.from_file,
    ) -> None:
        self.config = config
        self.workdir = workdir.resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.pool_config_path = Path(config.config_file).expanduser().resolve()
        self._pool_factory = pool_factory
        self._run_counter = 0

    @staticmethod
    def check_available(
        config: ClusterBridgePoolSandboxConfig,
    ) -> tuple[bool, str]:
        path = Path(config.config_file).expanduser().resolve()
        if not path.is_file():
            return False, f"cluster pool config not found: {path}"
        try:
            pool = ClusterBridgePool.from_file(
                path,
                restore_state=config.restore_state,
            )
            if config.require_prepared and not (
                pool.claimed and pool.prepared and pool.ray_started
            ):
                return (
                    False,
                    (
                        "cluster pool state is not claimed/prepared/Ray-ready; "
                        "run bin/cluster-pool prepare first"
                    ),
                )
            resources = pool.wait_for_ray_resources(
                timeout_sec=min(
                    30.0,
                    pool.config.ray.resource_timeout_sec,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"cluster pool readiness check failed: {exc}"
        return (
            True,
            (
                f"ClusterBridge/Ray pool {pool.config.pool_id} ready "
                f"({int(resources.total_gpu)} GPUs, "
                f"{resources.alive_nodes} nodes)"
            ),
        )

    def run(self, code: str, *, timeout_sec: int = 300) -> SandboxResult:
        self._run_counter += 1
        staging = self.workdir / f"_clusterbridge_pool_run_{self._run_counter}"
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
        staging = self.workdir / f"_clusterbridge_pool_project_{self._run_counter}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        error = validate_entry_point(entry_point)
        if error:
            return self._error_result(error)

        self._inject_harness(staging)
        for source in project_dir.iterdir():
            destination = staging / source.name
            if destination.name == "experiment_harness.py":
                continue
            if source.is_dir() and not source.name.startswith((".", "__")):
                shutil.copytree(source, destination, dirs_exist_ok=True)
            elif source.is_file():
                destination.write_bytes(source.read_bytes())

        error = validate_entry_point_resolved(staging, entry_point)
        if error:
            return self._error_result(error)
        if not (staging / entry_point).is_file():
            return self._error_result(
                f"Entry point {entry_point} not found in project"
            )
        return self._execute(
            staging,
            entry_point=entry_point,
            timeout_sec=timeout_sec,
            entry_args=args,
            env_overrides=env_overrides,
        )

    @staticmethod
    def _inject_harness(target_dir: Path) -> None:
        source = Path(__file__).parent / "harness_template.py"
        if source.is_file():
            (target_dir / "experiment_harness.py").write_text(
                source.read_text(encoding="utf-8"),
                encoding="utf-8",
            )

    @staticmethod
    def _error_result(message: str) -> SandboxResult:
        return SandboxResult(
            returncode=-1,
            stdout="",
            stderr=message,
            elapsed_sec=0.0,
            metrics={},
        )

    def _load_pool(self) -> ClusterBridgePool:
        pool = self._pool_factory(
            self.pool_config_path,
            restore_state=self.config.restore_state,
        )
        if self.config.require_prepared and not (
            pool.claimed and pool.prepared and pool.ray_started
        ):
            raise PoolNotReadyError(
                "ClusterBridge pool is not claimed, prepared, and Ray-ready"
            )
        return pool

    def _execute(
        self,
        staging_dir: Path,
        *,
        entry_point: str,
        timeout_sec: int,
        entry_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> SandboxResult:
        try:
            pool = self._load_pool()
        except Exception as exc:  # noqa: BLE001
            return self._error_result(f"ClusterBridge pool unavailable: {exc}")

        namespace = (
            self.config.deterministic_task_namespace.strip()
            or os.environ.get("RESEARCHCLAW_WORK_ITEM_ID", "").strip()
        )
        attempt = os.environ.get(
            "RESEARCHCLAW_WORK_ITEM_ATTEMPT",
            "",
        ).strip()
        if namespace:
            digest = hashlib.sha256(
                "|".join(
                    (
                        namespace,
                        attempt,
                        entry_point,
                        json.dumps(entry_args or [], sort_keys=True),
                        str(self._run_counter),
                    )
                ).encode("utf-8")
            ).hexdigest()[:12]
            run_id = f"rc-pool-{digest}"
        else:
            run_id = f"rc-pool-{uuid.uuid4().hex[:12]}"
        shared_run = pool.state_dir / "experiments" / run_id
        shared_run.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staging_dir, shared_run)
        task_metadata_paths = (
            staging_dir / ".clusterbridge_pool_task.json",
            shared_run / ".clusterbridge_pool_task.json",
        )
        for artifact in ("results.json", ".clusterbridge_pool_result.json"):
            (shared_run / artifact).unlink(missing_ok=True)

        effective_timeout = max(
            1,
            min(
                int(timeout_sec),
                int(self.config.timeout_sec)
                if self.config.timeout_sec > 0
                else int(timeout_sec),
            ),
        )
        env = {
            str(name): str(value)
            for name, value in (env_overrides or {}).items()
            if value and _SAFE_ENV_NAME.fullmatch(str(name))
        }
        for name in (
            "RESEARCHCLAW_FACTORY_ID",
            "RESEARCHCLAW_IDEA_ID",
            "RESEARCHCLAW_WORK_ITEM_ID",
            "RESEARCHCLAW_WORK_ITEM_ATTEMPT",
            "RESEARCHCLAW_GPU_REQUEST",
        ):
            if os.environ.get(name):
                env.setdefault(name, os.environ[name])
        command = self._build_task_command(
            shared_run,
            entry_point=entry_point,
            args=entry_args,
        )
        try:
            requested_gpus = max(
                0,
                int(env.get("RESEARCHCLAW_GPU_REQUEST", "0") or 0),
            )
        except ValueError:
            requested_gpus = 0
        timed_out = False
        self._write_task_metadata(
            task_metadata_paths,
            {
                "task_id": run_id,
                "pool_config": str(self.pool_config_path),
                "state": "starting",
            },
        )
        try:
            task = pool.run_task(
                command,
                timeout_sec=effective_timeout,
                env=env,
                task_id=run_id,
                require_ready=self.config.require_prepared,
                num_gpus=requested_gpus,
            )
            returncode = task.returncode
            stdout = task.stdout
            stderr = task.stderr
            elapsed = task.elapsed_sec
            timed_out = task.timed_out
            self._write_task_metadata(
                task_metadata_paths,
                {
                    "task_id": run_id,
                    "pool_config": str(self.pool_config_path),
                    "state": "finished",
                    "returncode": returncode,
                },
            )
        except PoolTaskTimeout as exc:
            task_summary = pool.state_dir / "tasks" / run_id / "summary.json"
            payload: dict[str, object] = {}
            if task_summary.is_file():
                try:
                    payload = json.loads(
                        task_summary.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    payload = {}
            returncode = int(payload.get("returncode", 124))
            stdout = str(payload.get("stdout", ""))
            stderr = str(payload.get("stderr", "")) or str(exc)
            elapsed = float(payload.get("elapsed_sec", effective_timeout))
            timed_out = True
            self._write_task_metadata(
                task_metadata_paths,
                {
                    "task_id": run_id,
                    "pool_config": str(self.pool_config_path),
                    "state": "timed_out",
                    "returncode": returncode,
                },
            )
        except (ClusterPoolError, OSError, ValueError) as exc:
            self._write_task_metadata(
                task_metadata_paths,
                {
                    "task_id": run_id,
                    "pool_config": str(self.pool_config_path),
                    "state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return self._error_result(
                f"ClusterBridge pool task failed: {type(exc).__name__}: {exc}"
            )

        metrics: dict[str, object] = dict(parse_metrics(stdout))
        result_path = shared_run / "results.json"
        if result_path.is_file():
            try:
                structured = json.loads(result_path.read_text(encoding="utf-8"))
                self._merge_numeric_metrics(metrics, structured)
                shutil.copy2(result_path, staging_dir / "results.json")
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read pool results.json at %s", result_path)

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

    @staticmethod
    def _write_task_metadata(
        paths: tuple[Path, ...],
        payload: dict[str, object],
    ) -> None:
        text = json.dumps(payload, sort_keys=True) + "\n"
        for path in paths:
            path.write_text(text, encoding="utf-8")

    def _build_task_command(
        self,
        shared_run: Path,
        *,
        entry_point: str,
        args: list[str] | None,
    ) -> str:
        run_dir = shlex.quote(str(shared_run))
        python = shlex.quote(self.config.remote_python)
        entry = shlex.quote(entry_point)
        arg_text = " ".join(shlex.quote(arg) for arg in (args or ()))
        command = f"{python} -u {entry}"
        if arg_text:
            command += f" {arg_text}"

        setup = [
            (
                f"mkdir -p {shlex.quote(str(shared_run / '.home'))} "
                f"{shlex.quote(str(shared_run / '.cache' / 'torch'))} "
                f"{shlex.quote(str(shared_run / '.cache' / 'matplotlib'))}"
            )
        ]
        setup.extend(self.config.setup_commands)
        setup_text = " && ".join(f"({item})" for item in setup)
        env_text = " ".join(
            (
                "PYTHONUNBUFFERED=1",
                "TOKENIZERS_PARALLELISM=false",
                f"HOME={shlex.quote(str(shared_run / '.home'))}",
                f"TORCH_HOME={shlex.quote(str(shared_run / '.cache' / 'torch'))}",
                f"MPLCONFIGDIR={shlex.quote(str(shared_run / '.cache' / 'matplotlib'))}",
            )
        )
        python_command = f"env {env_text} {command}"
        if self.config.network_isolation:
            python_command = (
                # A fresh network namespace would also sever the task from the
                # Ray GCS on the head node.  Prefer a network namespace with
                # only loopback and the Ray control-plane routes when the host
                # provides a policy helper; otherwise keep host networking so
                # the multi-node pool remains usable.
                "if command -v researchclaw-ray-netns >/dev/null 2>&1; then "
                f"researchclaw-ray-netns {python_command}; "
                f"else {python_command}; fi"
            )
        return (
            "set -euo pipefail; "
            f"cd {run_dir}; "
            f"{setup_text}; "
            f"bash -lc {shlex.quote(python_command)}"
        )

    @staticmethod
    def _merge_numeric_metrics(
        target: dict[str, object],
        value: object,
    ) -> None:
        if not isinstance(value, dict):
            return
        nested = value.get("metrics")
        if isinstance(nested, dict):
            for key, metric in nested.items():
                if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                    target.setdefault(str(key), float(metric))
        for key, metric in value.items():
            if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                target.setdefault(str(key), float(metric))
