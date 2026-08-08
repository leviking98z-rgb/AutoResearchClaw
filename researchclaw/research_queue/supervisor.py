"""Durable gate supervisor for canary-to-benchmark progression."""

from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CB = Path("/root/shared/.clusters/.tools/clusterbridge.sh")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_command(
    command: list[str],
    *,
    check: bool = False,
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _systemctl_properties(unit: str) -> dict[str, str]:
    result = run_command(
        [
            "systemctl",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "Result",
            "-p",
            "NRestarts",
            "-p",
            "ExecMainStatus",
            "-p",
            "ExecMainStartTimestamp",
            "-p",
            "ExecMainExitTimestamp",
        ]
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key:
            values[key] = value
    return values


def _resource_status(cb_path: Path) -> dict[str, Any]:
    result = run_command(["bash", str(cb_path), "resource-status", "--json"])
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


def collect_canary_report(
    *,
    state_dir: Path,
    config_path: Path,
    unit: str,
    cb_path: Path = DEFAULT_CB,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    queue = config["research_queue"]
    artifact_root = Path(queue["artifact_dir"]).expanduser().resolve()
    token_cap = int(queue.get("limits", {}).get("max_total_tokens", 0) or 0)
    expected_waves = int(queue.get("limits", {}).get("generation_max_batches", 0) or 0)
    conn = sqlite3.connect(state_dir / "research_queue.db")
    conn.row_factory = sqlite3.Row
    events = []
    for row in conn.execute(
        """
        SELECT seq,timestamp,event_type,idea_id,run_id,payload_json
        FROM events ORDER BY seq
        """
    ):
        events.append(
            {
                "seq": row["seq"],
                "timestamp": row["timestamp"],
                "event": row["event_type"],
                "idea_id": row["idea_id"] or "",
                "run_id": row["run_id"] or "",
                **json.loads(row["payload_json"]),
            }
        )
    ideas = [
        json.loads(row["data_json"])
        for row in conn.execute("SELECT data_json FROM ideas ORDER BY updated_at")
    ]
    runs = [
        json.loads(row["data_json"])
        for row in conn.execute("SELECT data_json FROM runs ORDER BY updated_at")
    ]
    conn.close()

    audit = []
    for path in sorted((state_dir / "llm-audit").glob("*/calls.jsonl*")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                audit.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    by_model: dict[str, int] = defaultdict(int)
    for value in audit:
        by_model[str(value.get("model") or "unknown")] += int(
            value.get("total_tokens", 0) or 0
        )
    start = next(
        (
            event["timestamp"]
            for event in events
            if event["event"] == "controller_started"
        ),
        "",
    )
    stop = next(
        (
            event["timestamp"]
            for event in reversed(events)
            if event["event"] == "controller_stopped"
        ),
        "",
    )
    duration = (
        (datetime.fromisoformat(stop) - datetime.fromisoformat(start)).total_seconds()
        if start and stop
        else None
    )
    imports = []
    for path in artifact_root.glob("ideas/*/revisions/revision-*/*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            imports.append({"path": str(path), "imports": [], "syntax_error": str(exc)})
            continue
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
        imports.append({"path": str(path), "imports": sorted(modules)})
    stdlib = set(sys.stdlib_module_names)
    allowed = set(queue.get("execution", {}).get("allowed_python_imports", []))
    disallowed = []
    syntax_errors = []
    for item in imports:
        if item.get("syntax_error"):
            syntax_errors.append(item)
        illegal = [
            name
            for name in item["imports"]
            if name not in stdlib and name not in allowed
        ]
        if illegal:
            disallowed.append({"path": item["path"], "imports": illegal})
    applied = [
        {
            "run_id": run["run_id"],
            "idea_id": run["idea_id"],
            "budget": run["budget"],
            "status": run["status"],
            "budget_parameters": (
                run.get("result", {}).get("usage", {}).get("budget_parameters")
            ),
            "error": run.get("error", ""),
        }
        for run in runs
    ]
    resource = _resource_status(cb_path)
    owner_snapshot = resource.get("snapshot", {})
    total_tokens = sum(int(item.get("total_tokens", 0) or 0) for item in audit)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "unit": unit,
        "unit_properties": _systemctl_properties(unit),
        "controller_started_at": start,
        "controller_stopped_at": stop,
        "duration_seconds": duration,
        "ideas": dict(Counter(item["status"] for item in ideas)),
        "runs": dict(Counter(item["status"] for item in runs)),
        "conclusions": dict(
            Counter(item.get("conclusion") or "none" for item in ideas)
        ),
        "idea_count": len(ideas),
        "run_count": len(runs),
        "research_notes": len(list(artifact_root.glob("ideas/*/research_note.md"))),
        "generation_batches_started": sum(
            event["event"] == "idea_generation_started" for event in events
        ),
        "prepare_validation_failures": [
            event for event in events if event["event"] == "prepare_validation_failed"
        ],
        "idea_loop_failures": [
            event for event in events if event["event"] == "idea_loop_failed"
        ],
        "generation_failures": [
            event for event in events if event["event"] == "idea_generation_failed"
        ],
        "llm_calls": len(audit),
        "llm_failed_calls": sum(item.get("outcome") != "success" for item in audit),
        "audit_total_tokens": total_tokens,
        "tokens_by_model": dict(by_model),
        "applied_budget_parameters": applied,
        "generated_imports": imports,
        "resource_status": resource,
        "latest_events": events[-40:],
        "disallowed_imports": disallowed,
        "syntax_errors": syntax_errors,
    }
    report["checks"] = {
        "duration_at_least_2h": duration is not None and duration >= 7200,
        "expected_generation_waves": (
            expected_waves == 0
            or report["generation_batches_started"] == expected_waves
        ),
        "no_prepare_validation_failure": not report["prepare_validation_failures"],
        "no_idea_loop_failure": not report["idea_loop_failures"],
        "no_generation_failure": not report["generation_failures"],
        "no_llm_call_failure": report["llm_failed_calls"] == 0,
        "tokens_within_cap": token_cap <= 0 or total_tokens <= token_cap,
        "no_disallowed_imports_materialized": not disallowed,
        "no_syntax_errors_materialized": not syntax_errors,
        "all_succeeded_runs_attest_budget": all(
            item["budget_parameters"] is not None
            for item in applied
            if item["status"] == "succeeded"
        ),
        "no_failed_runs": all(item["status"] != "failed" for item in applied),
        "no_owner_allocations": not owner_snapshot.get("allocations"),
        "no_owner_queue": not owner_snapshot.get("queue"),
        "controller_stopped": bool(stop),
        "unit_not_failed": (
            report["unit_properties"].get("Result", "") != "failed"
            and report["unit_properties"].get("ActiveState", "") != "failed"
        ),
    }
    report["passed"] = all(report["checks"].values())
    report["failure_reasons"] = [
        name for name, passed in report["checks"].items() if not passed
    ]
    return report


@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    state_dir: Path
    canary_state_dir: Path
    canary_config: Path
    canary_unit: str
    canary_report: Path
    benchmark_config: Path
    benchmark_output_dir: Path
    benchmark_remote_script: Path
    benchmark_result_path: Path | None = None
    benchmark_project: str = "autoresearch"
    benchmark_purpose: str = "minimal real benchmark gate"
    benchmark_nodes: int = 1
    benchmark_gpus: int = 1
    benchmark_duration_minutes: int = 45
    poll_interval_sec: float = 60.0
    canary_grace_sec: float = 900.0
    cb_path: Path = DEFAULT_CB
    owner: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> SupervisorConfig:
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        value = raw.get("supervisor", raw)
        if not isinstance(value, dict):
            raise TypeError("supervisor configuration must be a mapping")

        def resolved(name: str) -> Path:
            candidate = Path(str(value[name])).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            return candidate.resolve()

        return cls(
            state_dir=resolved("state_dir"),
            canary_state_dir=resolved("canary_state_dir"),
            canary_config=resolved("canary_config"),
            canary_unit=str(value["canary_unit"]),
            canary_report=resolved("canary_report"),
            benchmark_config=resolved("benchmark_config"),
            benchmark_output_dir=resolved("benchmark_output_dir"),
            benchmark_remote_script=resolved("benchmark_remote_script"),
            benchmark_result_path=(
                resolved("benchmark_result_path")
                if value.get("benchmark_result_path")
                else None
            ),
            benchmark_project=str(value.get("benchmark_project", "autoresearch")),
            benchmark_purpose=str(
                value.get("benchmark_purpose", "minimal real benchmark gate")
            ),
            benchmark_nodes=max(1, int(value.get("benchmark_nodes", 1))),
            benchmark_gpus=max(1, int(value.get("benchmark_gpus", 1))),
            benchmark_duration_minutes=max(
                5, int(value.get("benchmark_duration_minutes", 45))
            ),
            poll_interval_sec=max(5.0, float(value.get("poll_interval_sec", 60))),
            canary_grace_sec=max(0.0, float(value.get("canary_grace_sec", 900))),
            cb_path=Path(value.get("cb_path", DEFAULT_CB)).expanduser().resolve(),
            owner=str(value.get("owner", "") or ""),
        )


class AcceptanceSupervisor:
    def __init__(self, config: SupervisorConfig) -> None:
        self.config = config
        self.stop_requested = False
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir = self.config.state_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.config.state_dir / "state.json"
        self.events_path = self.config.state_dir / "events.jsonl"
        self.lock_path = self.config.state_dir / "supervisor.lock"
        self.allocation_path = self.config.state_dir / "allocation.json"

    def request_stop(self, *_: object) -> None:
        self.stop_requested = True

    def event(self, event_type: str, **payload: Any) -> None:
        row = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "event": event_type,
            **payload,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def write_state(self, phase: str, **payload: Any) -> None:
        atomic_json(
            self.state_path,
            {
                "updated_at": datetime.now().astimezone().isoformat(),
                "pid": os.getpid(),
                "phase": phase,
                **payload,
            },
        )

    def run(self) -> int:
        with self.lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another acceptance supervisor is active")
            signal.signal(signal.SIGTERM, self.request_stop)
            signal.signal(signal.SIGINT, self.request_stop)
            self.event("supervisor_started")
            try:
                passed = self._wait_for_canary()
                if not passed:
                    return 2
                return self._run_benchmark()
            finally:
                self._release_recorded_allocation()
                self.event("supervisor_stopped")

    def _wait_for_canary(self) -> bool:
        started = time.monotonic()
        while not self.stop_requested:
            report = collect_canary_report(
                state_dir=self.config.canary_state_dir,
                config_path=self.config.canary_config,
                unit=self.config.canary_unit,
                cb_path=self.config.cb_path,
            )
            stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
            atomic_json(self.snapshots_dir / f"canary-{stamp}.json", report)
            properties = report["unit_properties"]
            terminal = properties.get("ActiveState") in {"inactive", "failed"}
            if report["controller_stopped_at"]:
                terminal = True
            self.write_state(
                "waiting_canary" if not terminal else "evaluating_canary",
                canary={
                    "idea_count": report["idea_count"],
                    "run_count": report["run_count"],
                    "generation_batches_started": report["generation_batches_started"],
                    "audit_total_tokens": report["audit_total_tokens"],
                    "unit_properties": properties,
                },
            )
            if terminal:
                atomic_json(self.config.canary_report, report)
                self.event(
                    "canary_evaluated",
                    passed=report["passed"],
                    failure_reasons=report["failure_reasons"],
                )
                if report["passed"]:
                    return True
                self.write_state(
                    "stopped_by_gate",
                    canary_passed=False,
                    failure_reasons=report["failure_reasons"],
                )
                return False
            if time.monotonic() - started > 7200 + self.config.canary_grace_sec:
                self.event("canary_timeout")
                self.write_state("stopped_by_gate", failure_reasons=["canary_timeout"])
                return False
            time.sleep(self.config.poll_interval_sec)
        self.write_state("stopped", reason="signal")
        return False

    def _owner(self) -> str:
        if self.config.owner:
            return self.config.owner
        result = run_command(
            ["bash", "/root/shared/.clusters/.tools/am.sh", "whoami"],
            check=True,
        )
        owner = result.stdout.strip()
        if not owner:
            raise RuntimeError("cannot determine ClusterBridge owner")
        return owner

    def _run_benchmark(self) -> int:
        result_path = (
            self.config.benchmark_result_path
            or self.config.benchmark_output_dir / "result.json"
        )
        if result_path.exists():
            archived = result_path.with_name(
                f"{result_path.name}.previous-{int(time.time())}"
            )
            result_path.replace(archived)
        self.write_state("requesting_benchmark_resource", canary_passed=True)
        owner = self._owner()
        command = [
            "bash",
            str(self.config.cb_path),
            "request",
            "--nodes",
            str(self.config.benchmark_nodes),
            "--gpus",
            str(self.config.benchmark_gpus),
            "--project",
            self.config.benchmark_project,
            "--purpose",
            self.config.benchmark_purpose,
            "--duration",
            str(self.config.benchmark_duration_minutes),
        ]
        requested = run_command(command)
        payload = self._resource_ids_from_output(requested.stdout)
        if requested.returncode != 0 or not payload:
            self.event(
                "benchmark_resource_request_failed",
                returncode=requested.returncode,
                stdout=requested.stdout,
                stderr=requested.stderr,
            )
            self.write_state(
                "benchmark_failed",
                error="resource request failed",
                stdout=requested.stdout,
                stderr=requested.stderr,
            )
            return 3
        request_id = str(payload.get("request_id") or "")
        allocation_id = str(payload.get("allocation_id") or "")
        if not allocation_id and request_id:
            waited = run_command(
                [
                    "bash",
                    str(self.config.cb_path),
                    "wait",
                    request_id,
                    "--timeout",
                    str(min(1800, self.config.benchmark_duration_minutes * 60)),
                ],
                timeout=min(1860, self.config.benchmark_duration_minutes * 60 + 60),
            )
            wait_payload = self._resource_ids_from_output(waited.stdout)
            allocation_id = str(wait_payload.get("allocation_id") or "")
        if not allocation_id:
            self.event("benchmark_allocation_missing", payload=payload)
            self.write_state(
                "benchmark_failed",
                error="resource manager returned no allocation id",
                request=payload,
            )
            return 3
        atomic_json(
            self.allocation_path,
            {"allocation_id": allocation_id, "owner": owner, "request_id": request_id},
        )
        self.event("benchmark_allocation_acquired", allocation_id=allocation_id)
        self.write_state(
            "running_benchmark",
            allocation_id=allocation_id,
            benchmark_config=str(self.config.benchmark_config),
        )
        remote = run_command(
            [
                "bash",
                str(self.config.cb_path),
                "alloc",
                allocation_id,
                "run",
                "bash",
                str(self.config.benchmark_remote_script),
                str(self.config.benchmark_config),
            ],
            timeout=self.config.benchmark_duration_minutes * 60,
        )
        self.event(
            "benchmark_remote_finished",
            allocation_id=allocation_id,
            returncode=remote.returncode,
        )
        result: dict[str, Any] = {}
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                result = {"status": "error", "error": f"invalid result.json: {exc}"}
        success = (
            remote.returncode == 0
            and result.get("status") == "ok"
            and result_path.is_file()
        )
        self._release_recorded_allocation()
        resource = _resource_status(self.config.cb_path)
        no_owner_resources = not resource.get("snapshot", {}).get(
            "allocations"
        ) and not resource.get("snapshot", {}).get("queue")
        report = {
            "generated_at": datetime.now().astimezone().isoformat(),
            "passed": success and no_owner_resources,
            "remote_returncode": remote.returncode,
            "remote_stdout": remote.stdout[-20000:],
            "remote_stderr": remote.stderr[-20000:],
            "result": result,
            "resource_status_after_release": resource,
            "no_owner_resources_after_release": no_owner_resources,
        }
        atomic_json(self.config.state_dir / "benchmark-report.json", report)
        self.write_state(
            "completed" if report["passed"] else "benchmark_failed",
            benchmark_passed=report["passed"],
            benchmark_report=str(self.config.state_dir / "benchmark-report.json"),
        )
        return 0 if report["passed"] else 4

    def _release_recorded_allocation(self) -> None:
        if not self.allocation_path.is_file():
            return
        try:
            value = json.loads(self.allocation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        allocation_id = str(value.get("allocation_id", "") or "")
        owner = str(value.get("owner", "") or "")
        if not allocation_id or not owner:
            return
        released = run_command(
            [
                "bash",
                str(self.config.cb_path),
                "alloc-release",
                allocation_id,
            ]
        )
        self.event(
            "benchmark_allocation_released",
            allocation_id=allocation_id,
            returncode=released.returncode,
        )
        if released.returncode == 0:
            self.allocation_path.unlink(missing_ok=True)

    @staticmethod
    def _resource_ids_from_output(output: str) -> dict[str, Any]:
        output = output.strip()
        if not output:
            return {}
        try:
            value = json.loads(output)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            pass
        for line in reversed(output.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        result: dict[str, str] = {}
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith("request "):
                result["request_id"] = stripped.split(maxsplit=1)[1]
            elif stripped.startswith("allocation "):
                result["allocation_id"] = stripped.split(maxsplit=1)[1]
            elif stripped.startswith("allocation:"):
                result["allocation_id"] = stripped.partition(":")[2].strip()
        return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-queue-supervisor",
        description="Gate a clean Research Queue canary before a real benchmark",
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--snapshot-once",
        action="store_true",
        help="Collect one canary report without waiting or requesting resources.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = SupervisorConfig.from_file(args.config)
    if args.snapshot_once:
        report = collect_canary_report(
            state_dir=config.canary_state_dir,
            config_path=config.canary_config,
            unit=config.canary_unit,
            cb_path=config.cb_path,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    return AcceptanceSupervisor(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
