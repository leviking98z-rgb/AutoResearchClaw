"""CLI implementations used by the standalone ``bin/rsi-*`` entry points."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .monitor import _process_start_ticks as _monitor_process_start_ticks
from .monitor import _read_pid_identity
from .storage import CampaignStore, atomic_write_json
from .supervisor import (
    DEFAULT_CAMPAIGN_ROOT,
    CampaignSupervisor,
    SupervisorOptions,
    _process_start_ticks,
    campaign_status,
    new_campaign_id,
    resolve_campaign,
    run_policy_from_options,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _campaign_root(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else DEFAULT_CAMPAIGN_ROOT


def _recorded_process_is_alive(state: dict[str, object]) -> bool:
    try:
        pid = int(state.get("pid") or 0)
        expected_ticks = int(state.get("supervisor_start_ticks"))
    except (TypeError, ValueError):
        return False
    return pid > 0 and _process_start_ticks(pid) == expected_ticks


_POLICY_DEFAULTS: dict[str, object] = {
    "continuous": False,
    "max_cycles": 20,
    "single_cycle": False,
    "dry_run": False,
    "max_consecutive_failures": 3,
    "max_no_improvement_cycles": 5,
    "backoff_initial_sec": 30.0,
    "backoff_max_sec": 900.0,
    "heartbeat_interval_sec": 15.0,
    "control_poll_sec": 1.0,
    "model": "codebuddy/deepseek-v4-pro-ioa",
    "bridge_url": "http://127.0.0.1:8787/v1",
    "api_key_env": "BRIDGE_LOCAL_API_KEY",
    "llm_timeout_sec": 1800,
    "skip_preflight": False,
    "no_aevolve": False,
    "pipeline_extra_args": [],
}


def _load_run_policy(
    store: CampaignStore,
    manifest: dict[str, object],
) -> dict[str, object]:
    """Load a campaign policy, migrating pre-policy campaign manifests."""

    policy = dict(_POLICY_DEFAULTS)
    embedded = manifest.get("run_policy")
    if isinstance(embedded, dict):
        policy.update(embedded)
    try:
        persisted = json.loads(store.policy_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        persisted = {}
    if isinstance(persisted, dict):
        policy.update(persisted)
    for key in ("model", "bridge_url"):
        if manifest.get(key):
            policy[key] = manifest[key]
    return policy


def _resume_value(
    override: object,
    policy: dict[str, object],
    key: str,
) -> object:
    return policy.get(key, _POLICY_DEFAULTS[key]) if override is None else override


def _submit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsi-submit",
        description="Create or run a persistent AutoResearchClaw RSI campaign.",
    )
    parser.add_argument("topic", nargs="?", help="research brief or topic")
    parser.add_argument("--brief-file", help="read the research brief from a file")
    parser.add_argument("--campaign-id", help="explicit campaign identifier")
    parser.add_argument("--campaign-root", help="campaign storage root")
    parser.add_argument(
        "--config",
        default=str(_repo_root() / "config.rsi.yaml"),
        help="base ResearchClaw YAML config",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=20,
        help="campaign cycle budget; 0 explicitly means unbounded",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "keep iterating until an explicit pause/stop; ignores cycle, "
            "failure, plateau, and LLM stop recommendations"
        ),
    )
    parser.add_argument("--single-cycle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-aevolve", action="store_true")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--max-no-improvement-cycles", type=int, default=5)
    parser.add_argument("--backoff-initial-sec", type=float, default=30.0)
    parser.add_argument("--backoff-max-sec", type=float, default=900.0)
    parser.add_argument("--heartbeat-interval-sec", type=float, default=15.0)
    parser.add_argument("--control-poll-sec", type=float, default=1.0)
    parser.add_argument(
        "--model",
        default="codebuddy/deepseek-v4-pro-ioa",
    )
    parser.add_argument(
        "--bridge-url",
        default="http://127.0.0.1:8787/v1",
    )
    parser.add_argument("--api-key-env", default="BRIDGE_LOCAL_API_KEY")
    parser.add_argument("--llm-timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--pipeline-arg",
        action="append",
        default=[],
        help="extra argument appended to `researchclaw run` (repeatable)",
    )
    parser.add_argument(
        "--_run-supervisor",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_resume-existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_startup-handshake",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_supervisor-log",
        help=argparse.SUPPRESS,
    )
    return parser


def _read_topic(args: argparse.Namespace) -> str:
    if args.brief_file:
        return Path(args.brief_file).expanduser().read_text(encoding="utf-8").strip()
    return str(args.topic or "").strip()


def _options_from_submit(
    args: argparse.Namespace,
    *,
    campaign_dir: Path,
    topic: str,
) -> SupervisorOptions:
    return SupervisorOptions(
        campaign_dir=campaign_dir,
        repo_root=_repo_root(),
        base_config=Path(args.config).expanduser().resolve(),
        topic=topic,
        max_cycles=max(0, args.max_cycles),
        continuous=bool(args.continuous),
        single_cycle=bool(args.single_cycle),
        dry_run=bool(args.dry_run),
        max_consecutive_failures=max(1, args.max_consecutive_failures),
        max_no_improvement_cycles=max(1, args.max_no_improvement_cycles),
        backoff_initial_sec=max(0.0, args.backoff_initial_sec),
        backoff_max_sec=max(0.0, args.backoff_max_sec),
        heartbeat_interval_sec=max(0.1, args.heartbeat_interval_sec),
        control_poll_sec=max(0.05, args.control_poll_sec),
        model=args.model,
        bridge_url=args.bridge_url,
        api_key_env=args.api_key_env,
        llm_timeout_sec=max(1, args.llm_timeout_sec),
        skip_preflight=bool(args.skip_preflight),
        no_aevolve=bool(args.no_aevolve),
        resume_existing=bool(args._resume_existing),
        pipeline_extra_args=tuple(args.pipeline_arg),
    )


def _wait_for_supervisor_start(
    *,
    campaign_dir: Path,
    child: subprocess.Popen[str],
    timeout_sec: float = 30.0,
) -> tuple[bool, str]:
    """Wait until the child proves it owns and is running the campaign."""

    store = CampaignStore(campaign_dir)
    launch_started_at = time.time()
    deadline = time.monotonic() + max(0.1, timeout_sec)
    last = ""
    while time.monotonic() < deadline:
        returncode = child.poll()
        if returncode is not None:
            return False, f"supervisor exited during startup with {returncode}"
        try:
            state = store.load_state()
        except FileNotFoundError:
            state = {}
        try:
            heartbeat = json.loads(
                store.supervisor_heartbeat_path.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            heartbeat = {}
        state_pid = int(state.get("pid") or 0)
        heartbeat_pid = int(heartbeat.get("supervisor_pid") or 0)
        heartbeat_timestamp = str(heartbeat.get("timestamp") or "")
        try:
            from datetime import datetime

            heartbeat_epoch = datetime.fromisoformat(heartbeat_timestamp).timestamp()
        except (TypeError, ValueError):
            heartbeat_epoch = 0.0
        status = str(state.get("status") or "")
        try:
            recorded_start_ticks = int(state.get("supervisor_start_ticks"))
        except (TypeError, ValueError):
            recorded_start_ticks = None
        child_start_ticks = _process_start_ticks(child.pid)
        last = (
            f"status={status!r} state_pid={state_pid} "
            f"heartbeat_pid={heartbeat_pid} "
            f"start_ticks={recorded_start_ticks!r}/{child_start_ticks!r}"
        )
        # The state transition is the authoritative ownership handshake.  A
        # heartbeat from the new child normally follows immediately, but an
        # older heartbeat must not cause a false reload failure.
        if (
            state_pid == child.pid
            and recorded_start_ticks is not None
            and recorded_start_ticks == child_start_ticks
            and status == "running"
            and (
                heartbeat_pid == child.pid
                or heartbeat_epoch < launch_started_at
            )
        ):
            return True, last
        time.sleep(0.1)
    return False, f"startup handshake timed out ({last or 'no state'})"


def submit_main(argv: Sequence[str] | None = None) -> int:
    parser = _submit_parser()
    args = parser.parse_args(argv)
    topic = _read_topic(args)
    if not topic:
        parser.error("provide a topic or --brief-file")
    base_config = Path(args.config).expanduser().resolve()
    if not base_config.is_file():
        parser.error(f"config file not found: {base_config}")

    root = _campaign_root(args.campaign_root)
    campaign_id = args.campaign_id or new_campaign_id(topic)
    campaign_dir = resolve_campaign(campaign_id, campaign_root=root)
    options = _options_from_submit(args, campaign_dir=campaign_dir, topic=topic)

    if args._run_supervisor or args.foreground:
        if args._supervisor_log:
            log_path = Path(args._supervisor_log).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            sys.stdout = log_handle
            sys.stderr = log_handle
        if args._startup_handshake:
            store = CampaignStore(campaign_dir)
            # The parent already initialized the campaign before spawning this
            # child. Avoid re-statting every CephFS campaign subdirectory on
            # the startup-critical path; the durable root/control directories
            # are sufficient for the ownership handshake.
            store.root.mkdir(parents=True, exist_ok=True)
            store.control_dir.mkdir(parents=True, exist_ok=True)
            store.clear_control("pause")
            store.clear_control("stop")
            state = store.load_state()
            state.update(
                {
                    "status": "running",
                    "phase": "starting",
                    "pid": os.getpid(),
                    "supervisor_start_ticks": _process_start_ticks(os.getpid()),
                    "active_child_pid": None,
                    "active_child_start_ticks": None,
                    "last_error": None,
                }
            )
            store.save_state(state)
        if args._resume_existing:
            resume_store = CampaignStore(campaign_dir)
            policy = run_policy_from_options(options)
            atomic_write_json(resume_store.policy_path, policy)
            manifest = json.loads(
                resume_store.manifest_path.read_text(encoding="utf-8")
            )
            manifest["run_policy"] = policy
            manifest["model"] = options.model
            manifest["bridge_url"] = options.bridge_url
            atomic_write_json(resume_store.manifest_path, manifest)
        return CampaignSupervisor(options).run()

    # Initialize synchronously so status/control commands are immediately safe.
    CampaignSupervisor(options).initialize()
    log_path = campaign_dir / "supervisor.log"
    child_args = [
        str(_repo_root() / "bin" / "rsi-submit"),
        topic,
        "--campaign-id",
        campaign_id,
        "--campaign-root",
        str(root),
        "--config",
        str(base_config),
        "--max-cycles",
        str(args.max_cycles),
        "--max-consecutive-failures",
        str(args.max_consecutive_failures),
        "--max-no-improvement-cycles",
        str(args.max_no_improvement_cycles),
        "--backoff-initial-sec",
        str(args.backoff_initial_sec),
        "--backoff-max-sec",
        str(args.backoff_max_sec),
        "--heartbeat-interval-sec",
        str(args.heartbeat_interval_sec),
        "--control-poll-sec",
        str(args.control_poll_sec),
        "--model",
        args.model,
        "--bridge-url",
        args.bridge_url,
        "--api-key-env",
        args.api_key_env,
        "--llm-timeout-sec",
        str(args.llm_timeout_sec),
        "--_run-supervisor",
        "--_startup-handshake",
        "--_supervisor-log",
        str(log_path),
    ]
    for flag in (
        "continuous",
        "single_cycle",
        "dry_run",
        "skip_preflight",
        "no_aevolve",
    ):
        if getattr(args, flag):
            child_args.append("--" + flag.replace("_", "-"))
    if args._resume_existing:
        child_args.append("--_resume-existing")
    for value in args.pipeline_arg:
        child_args.append(f"--pipeline-arg={value}")

    env = os.environ.copy()
    if not env.get(args.api_key_env):
        env[args.api_key_env] = "local-bridge"
    child = subprocess.Popen(
        child_args,
        cwd=_repo_root(),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        text=True,
    )
    store = CampaignStore(campaign_dir)
    store.log.append(
        "supervisor_spawned",
        campaign_id=campaign_id,
        pid=child.pid,
        log_path=str(log_path),
    )
    started, startup_detail = _wait_for_supervisor_start(
        campaign_dir=campaign_dir,
        child=child,
    )
    store.log.append(
        "supervisor_startup_handshake",
        campaign_id=campaign_id,
        pid=child.pid,
        success=started,
        detail=startup_detail,
    )
    if not started:
        try:
            child.terminate()
            child.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            child.kill()
        state = store.load_state()
        state.update(
            {
                "status": "crashed",
                "phase": "launch",
                "pid": None,
                "last_error": startup_detail,
            }
        )
        store.save_state(state)
        if args._resume_existing:
            store.set_control("pause", f"resume startup failed: {startup_detail}")
        print(
            json.dumps(
                {
                    "campaign_id": campaign_id,
                    "campaign_dir": str(campaign_dir),
                    "pid": child.pid,
                    "status": "startup_failed",
                    "error": startup_detail,
                    "log": str(log_path),
                },
                indent=2,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    monitor: dict[str, object] = {"started": False}
    daemon = _repo_root() / "bin" / "rsi-daemon"
    resume = _repo_root() / "bin" / "rsi-resume"
    if daemon.is_file() and os.access(daemon, os.X_OK):
        restart_command = json.dumps(
            [str(resume), str(campaign_dir)],
            ensure_ascii=False,
        )
        try:
            daemon_result = subprocess.run(
                [
                    str(daemon),
                    str(campaign_dir),
                    "--restart-command-json",
                    restart_command,
                ],
                cwd=_repo_root(),
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            daemon_output = f"{type(exc).__name__}: {exc}"
            monitor = {
                "started": False,
                "returncode": None,
                "detail": daemon_output,
                "pid_file": str(campaign_dir / "rsi-monitor.pid"),
                "log": str(campaign_dir / "rsi-monitor.log"),
            }
            store.log.append(
                "monitor_start_attempted",
                campaign_id=campaign_id,
                success=False,
                returncode=None,
                detail=daemon_output,
            )
        else:
            daemon_output = (
                daemon_result.stdout.strip() or daemon_result.stderr.strip()
            )
            monitor = {
                "started": daemon_result.returncode == 0,
                "returncode": daemon_result.returncode,
                "detail": daemon_output,
                "pid_file": str(campaign_dir / "rsi-monitor.pid"),
                "log": str(campaign_dir / "rsi-monitor.log"),
            }
            store.log.append(
                "monitor_start_attempted",
                campaign_id=campaign_id,
                success=daemon_result.returncode == 0,
                returncode=daemon_result.returncode,
                detail=daemon_output,
            )
    print(
        json.dumps(
            {
                "campaign_id": campaign_id,
                "campaign_dir": str(campaign_dir),
                "pid": child.pid,
                "status": "running",
                "log": str(log_path),
                "monitor": monitor,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _control_parser(prog: str, verb: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("campaign", help="campaign ID or directory")
    parser.add_argument("--campaign-root", help="campaign storage root")
    parser.add_argument("--reason", default=f"requested by {prog}")
    if verb == "resume":
        parser.add_argument("--foreground", action="store_true")
        parser.add_argument(
            "--continuous",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            "--single-cycle",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            "--dry-run",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument("--max-cycles", type=int)
        parser.add_argument("--max-consecutive-failures", type=int)
        parser.add_argument("--max-no-improvement-cycles", type=int)
        parser.add_argument("--backoff-initial-sec", type=float)
        parser.add_argument("--backoff-max-sec", type=float)
        parser.add_argument("--heartbeat-interval-sec", type=float)
        parser.add_argument("--control-poll-sec", type=float)
        parser.add_argument("--llm-timeout-sec", type=int)
        parser.add_argument("--api-key-env")
        parser.add_argument(
            "--skip-preflight",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            "--no-aevolve",
            action=argparse.BooleanOptionalAction,
            default=None,
        )
        parser.add_argument(
            "--pipeline-arg",
            action="append",
            default=None,
            help="replace persisted extra pipeline arguments (repeatable)",
        )
    return parser


def status_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rsi-status")
    parser.add_argument("campaign", help="campaign ID or directory")
    parser.add_argument("--campaign-root", help="campaign storage root")
    parser.add_argument("--events", type=int, default=0, help="include last N events")
    args = parser.parse_args(argv)
    campaign_dir = resolve_campaign(
        args.campaign,
        campaign_root=_campaign_root(args.campaign_root),
    )
    try:
        status = campaign_status(campaign_dir)
    except FileNotFoundError as exc:
        print(f"rsi-status: {exc}", file=sys.stderr)
        return 1
    if args.events > 0:
        events = CampaignStore(campaign_dir).log.read_all()
        status["recent_events"] = events[-args.events :]
    print(json.dumps(status, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def _signal_pid(state: dict[str, object], signum: int) -> None:
    try:
        pid = int(state.get("pid") or 0)
        expected_ticks = int(state.get("supervisor_start_ticks"))
    except (TypeError, ValueError):
        return
    if pid <= 0:
        return
    if _process_start_ticks(pid) != expected_ticks:
        return
    try:
        os.kill(pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def _stop_monitor(
    campaign_dir: Path,
    *,
    wait_timeout_sec: float = 75.0,
    kill_timeout_sec: float = 2.0,
) -> dict[str, object]:
    """Best-effort stop of the campaign-owned monitor daemon.

    A permanent campaign stop should not leave a detached watchdog polling the
    Bridge and GPU pool forever.  Pause intentionally does not call this helper
    because its monitor is responsible for restart-on-crash while resumable.
    """

    pid_path = campaign_dir / "rsi-monitor.pid"
    watchdog_pid_path = campaign_dir / "rsi-monitor.pid.watchdog.pid"
    stop_path = campaign_dir / "rsi-monitor.pid.watchdog.stop"
    try:
        watchdog_pid = int(
            watchdog_pid_path.read_text(encoding="utf-8").strip()
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        watchdog_pid = None
    if watchdog_pid is not None:
        stop_path.touch()

    def stop_watchdog() -> None:
        if watchdog_pid is None:
            return
        try:
            os.kill(watchdog_pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    pid, expected_ticks = _read_pid_identity(pid_path)
    if pid is None:
        stop_watchdog()
        return {"requested": False, "pid": None, "reason": "pid_unavailable"}
    if (
        expected_ticks is not None
        and _monitor_process_start_ticks(pid) != expected_ticks
    ):
        pid_path.unlink(missing_ok=True)
        stop_watchdog()
        return {"requested": False, "pid": pid, "reason": "identity_mismatch"}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        stop_watchdog()
        return {"requested": False, "pid": pid, "reason": "not_running"}
    except PermissionError:
        stop_watchdog()
        return {"requested": False, "pid": pid, "reason": "permission_denied"}
    stop_watchdog()
    # A poll may be inside several ClusterBridge subprocesses. Give the signal
    # handler one normal command timeout window before escalating.
    deadline = time.monotonic() + max(0.0, wait_timeout_sec)
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            if watchdog_pid is None:
                stop_path.unlink(missing_ok=True)
            return {"requested": True, "pid": pid, "reason": "exited"}
        except PermissionError:
            return {
                "requested": True,
                "pid": pid,
                "reason": "sigterm_permission_unknown",
            }
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        if watchdog_pid is None:
            stop_path.unlink(missing_ok=True)
        return {"requested": True, "pid": pid, "reason": "exited"}
    except PermissionError:
        return {
            "requested": True,
            "pid": pid,
            "reason": "sigkill_permission_denied",
        }
    kill_deadline = time.monotonic() + max(0.0, kill_timeout_sec)
    while time.monotonic() < kill_deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            if watchdog_pid is None:
                stop_path.unlink(missing_ok=True)
            return {"requested": True, "pid": pid, "reason": "sigkill"}
        except PermissionError:
            return {
                "requested": True,
                "pid": pid,
                "reason": "sigkill_permission_unknown",
            }
        time.sleep(0.05)
    return {"requested": True, "pid": pid, "reason": "sigkill_pending"}


def stop_main(argv: Sequence[str] | None = None) -> int:
    args = _control_parser("rsi-stop", "stop").parse_args(argv)
    campaign_dir = resolve_campaign(
        args.campaign,
        campaign_root=_campaign_root(args.campaign_root),
    )
    store = CampaignStore(campaign_dir)
    try:
        state = store.load_state()
    except FileNotFoundError as exc:
        print(f"rsi-stop: {exc}", file=sys.stderr)
        return 1
    store.set_control("stop", args.reason)
    store.log.append(
        "stop_requested",
        campaign_id=str(state.get("campaign_id", campaign_dir.name)),
        reason=args.reason,
    )
    _signal_pid(state, signal.SIGTERM)
    monitor = _stop_monitor(campaign_dir)
    store.log.append(
        "monitor_stop_attempted",
        campaign_id=str(state.get("campaign_id", campaign_dir.name)),
        **monitor,
    )
    print(
        json.dumps(
            {
                "campaign_dir": str(campaign_dir),
                "stop_requested": True,
                "monitor": monitor,
            }
        )
    )
    return 0


def resume_main(argv: Sequence[str] | None = None) -> int:
    args = _control_parser("rsi-resume", "resume").parse_args(argv)
    campaign_dir = resolve_campaign(
        args.campaign,
        campaign_root=_campaign_root(args.campaign_root),
    )
    store = CampaignStore(campaign_dir)
    try:
        state = store.load_state()
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"rsi-resume: invalid campaign: {exc}", file=sys.stderr)
        return 1

    if _recorded_process_is_alive(state):
        pid = int(state["pid"])
        print(
            f"rsi-resume: supervisor already running with pid {pid}",
            file=sys.stderr,
        )
        return 1

    # The original manifest topic is the immutable campaign meta-brief. Do not
    # substitute model-generated topic refinements when rebuilding options.
    topic = str(
        manifest.get("topic")
        or state.get("topic")
        or (
            store.shared_brief_path.read_text(encoding="utf-8").strip()
            if store.shared_brief_path.exists()
            else ""
        )
    )
    policy = _load_run_policy(store, manifest)
    max_cycles = max(
        0,
        int(_resume_value(args.max_cycles, policy, "max_cycles")),
    )
    max_failures = max(
        1,
        int(
            _resume_value(
                args.max_consecutive_failures,
                policy,
                "max_consecutive_failures",
            )
        ),
    )
    max_no_improvement = max(
        1,
        int(
            _resume_value(
                args.max_no_improvement_cycles,
                policy,
                "max_no_improvement_cycles",
            )
        ),
    )
    backoff_initial = max(
        0.0,
        float(
            _resume_value(
                args.backoff_initial_sec,
                policy,
                "backoff_initial_sec",
            )
        ),
    )
    backoff_max = max(
        0.0,
        float(
            _resume_value(
                args.backoff_max_sec,
                policy,
                "backoff_max_sec",
            )
        ),
    )
    heartbeat_interval = max(
        0.1,
        float(
            _resume_value(
                args.heartbeat_interval_sec,
                policy,
                "heartbeat_interval_sec",
            )
        ),
    )
    control_poll = max(
        0.05,
        float(
            _resume_value(
                args.control_poll_sec,
                policy,
                "control_poll_sec",
            )
        ),
    )
    llm_timeout = max(
        1,
        int(
            _resume_value(
                args.llm_timeout_sec,
                policy,
                "llm_timeout_sec",
            )
        ),
    )
    api_key_env = str(
        _resume_value(args.api_key_env, policy, "api_key_env")
    ).strip() or "BRIDGE_LOCAL_API_KEY"
    continuous = bool(
        _resume_value(args.continuous, policy, "continuous")
    )
    single_cycle = bool(
        _resume_value(args.single_cycle, policy, "single_cycle")
    )
    dry_run = bool(_resume_value(args.dry_run, policy, "dry_run"))
    skip_preflight = bool(
        _resume_value(args.skip_preflight, policy, "skip_preflight")
    )
    no_aevolve = bool(
        _resume_value(args.no_aevolve, policy, "no_aevolve")
    )
    pipeline_args = (
        [str(value) for value in args.pipeline_arg]
        if args.pipeline_arg is not None
        else [
            str(value)
            for value in policy.get("pipeline_extra_args", [])
            if str(value)
        ]
    )
    submit_args = [
        topic,
        "--campaign-id",
        str(state.get("campaign_id", campaign_dir.name)),
        "--campaign-root",
        str(campaign_dir.parent),
        "--config",
        str(manifest["base_config"]),
        "--max-cycles",
        str(max_cycles),
        "--max-consecutive-failures",
        str(max_failures),
        "--max-no-improvement-cycles",
        str(max_no_improvement),
        "--backoff-initial-sec",
        str(backoff_initial),
        "--backoff-max-sec",
        str(backoff_max),
        "--heartbeat-interval-sec",
        str(heartbeat_interval),
        "--control-poll-sec",
        str(control_poll),
        "--model",
        str(policy["model"]),
        "--bridge-url",
        str(policy["bridge_url"]),
        "--api-key-env",
        api_key_env,
        "--llm-timeout-sec",
        str(llm_timeout),
        "--_resume-existing",
    ]
    if args.foreground:
        submit_args.append("--foreground")
    if continuous:
        submit_args.append("--continuous")
    if single_cycle:
        submit_args.append("--single-cycle")
    if dry_run:
        submit_args.append("--dry-run")
    if skip_preflight:
        submit_args.append("--skip-preflight")
    if no_aevolve:
        submit_args.append("--no-aevolve")
    for value in pipeline_args:
        submit_args.append(f"--pipeline-arg={value}")
    store.log.append(
        "resume_requested",
        campaign_id=str(state.get("campaign_id", campaign_dir.name)),
        next_cycle=state.get("next_cycle"),
        policy={
            "continuous": continuous,
            "max_cycles": max_cycles,
            "single_cycle": single_cycle,
            "dry_run": dry_run,
            "max_consecutive_failures": max_failures,
            "max_no_improvement_cycles": max_no_improvement,
            "model": str(policy["model"]),
            "bridge_url": str(policy["bridge_url"]),
            "api_key_env": api_key_env,
            "llm_timeout_sec": llm_timeout,
            "skip_preflight": skip_preflight,
            "no_aevolve": no_aevolve,
            "pipeline_extra_args": pipeline_args,
        },
    )
    return submit_main(submit_args)
