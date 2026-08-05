"""Persistent watchdog and health monitor for RSI research campaigns.

The monitor is deliberately independent of the RSI supervisor implementation:
it consumes the supervisor's files, probes external dependencies, and can
restart the supervisor through an injected command runner.  This makes it
usable while the rest of :mod:`researchclaw.rsi` evolves and keeps unit tests
free of real ClusterBridge, GPU, HTTP, or process side effects.

Expected campaign files (all optional while a campaign is bootstrapping):

``supervisor_heartbeat.json``
    Supervisor liveness record.  ``heartbeat.json`` and
    ``state/supervisor_heartbeat.json`` are accepted compatibility names.
``campaign_state.json``
    Campaign state.  ``state.json`` and ``state/campaign_state.json`` are also
    accepted.
``pipeline_checkpoint.json``
    A campaign-level pipeline checkpoint.  If the state identifies a run
    directory, that run's standard ResearchClaw ``checkpoint.json`` is used.
``pause.request.json``
    Cooperative pause request.  A pause suppresses automatic restarts but does
    not terminate either the supervisor or this monitor.

Every poll writes ``monitor_snapshot.json`` atomically.  Restart bookkeeping is
stored separately in ``monitor_state.json`` so exponential backoff survives a
monitor process restart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

DEFAULT_BRIDGE_HEALTH_URL = "http://127.0.0.1:8787/health"
DEFAULT_CB_COMMAND = "/root/shared/.clusters/.tools/clusterbridge.sh"
DEFAULT_POOL_CONFIG = ""
DEFAULT_LEASE_HEARTBEAT = (
    "/root/shared/.clusters/.tmp/autoresearch-rsi/"
    "lease-keepalive-heartbeat.json"
)
DEFAULT_SNAPSHOT_NAME = "monitor_snapshot.json"
DEFAULT_MONITOR_STATE_NAME = "monitor_state.json"
DEFAULT_MONITOR_HEARTBEAT_NAME = "monitor_heartbeat.json"
DEFAULT_PAUSE_REQUEST_NAME = "pause.request.json"
DEFAULT_DAEMON_PID_NAME = "rsi-monitor.pid"
DEFAULT_RESOURCE_SNAPSHOT_NAME = ".resource_manager/snapshot.json"

_HEALTH_RANK = {"ok": 0, "degraded": 1, "fail": 2}
_HEALTH_VALUES = frozenset(_HEALTH_RANK)
_TIMESTAMP_KEYS = (
    "timestamp",
    "updated_at",
    "last_heartbeat",
    "heartbeat_at",
    "generated",
    "created_at",
)
_SUPERVISOR_HEARTBEAT_CANDIDATES = (
    "heartbeat.json",
    "supervisor_heartbeat.json",
    "state/supervisor_heartbeat.json",
    "state/heartbeat.json",
)
_CAMPAIGN_STATE_CANDIDATES = (
    "state.json",
    "campaign_state.json",
    "state/campaign_state.json",
)
_PIPELINE_CHECKPOINT_CANDIDATES = (
    "pipeline_checkpoint.json",
    "checkpoint.json",
    "state/pipeline_checkpoint.json",
)
_RUN_DIR_KEYS = (
    "run_dir",
    "current_run_dir",
    "pipeline_run_dir",
    "active_run_dir",
    "artifact_dir",
    "output_dir",
)
_SUPERVISOR_PID_KEYS = ("supervisor_pid", "pid")
_TERMINAL_CAMPAIGN_STATES = frozenset(
    {
        "complete",
        "completed",
        "done",
        "stopped",
        "paused",
        "paused_single_cycle",
        "paused_failure",
        "paused_failure_threshold",
        "paused_no_improvement",
    }
)
_NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,254}$")
_CB_LIST_ROW_RE = re.compile(
    r"^(?P<node>\S+)\s+"
    r"(?P<alive>yes|no-hb|STALE\([^)]*\)|\?)\s+"
    r"(?P<gpus>\S+)\s+"
    r"(?P<cluster>\S+)\s+"
    r"(?P<claim>.+?)\s+"
    r"(?P<expires>-|\S+\(\d+m\))\s*$"
)
_NVIDIA_ROW_RE = re.compile(
    r"^\s*(?P<index>\d+)\s*,\s*"
    r"(?P<name>.+?)\s*,\s*"
    r"(?P<memory_total>\d+)\s*,\s*"
    r"(?P<memory_used>\d+)\s*,\s*"
    r"(?P<utilization>\d+)\s*$"
)


class CommandRunner(Protocol):
    """Callable compatible with :func:`subprocess.run`."""

    def __call__(self, args: Sequence[str], **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class Check:
    """One monitor check suitable for JSON serialization."""

    name: str
    status: str
    detail: str
    observed_at: str
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _HEALTH_VALUES:
            raise ValueError(f"unsupported health status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    """Exponential-backoff policy for supervisor restarts."""

    initial_delay_sec: float = 30.0
    multiplier: float = 2.0
    max_delay_sec: float = 1800.0
    max_attempts: int = 8
    reset_after_healthy_sec: float = 900.0

    def __post_init__(self) -> None:
        if self.initial_delay_sec < 0:
            raise ValueError("initial_delay_sec must be non-negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least 1")
        if self.max_delay_sec < self.initial_delay_sec:
            raise ValueError("max_delay_sec must be >= initial_delay_sec")
        if self.max_attempts < 0:
            raise ValueError("max_attempts must be non-negative")
        if self.reset_after_healthy_sec < 0:
            raise ValueError("reset_after_healthy_sec must be non-negative")

    def delay_for_attempt(self, attempt: int) -> float:
        """Return delay before ``attempt`` where the first attempt is 1."""

        if attempt <= 1:
            return self.initial_delay_sec
        raw = self.initial_delay_sec * (self.multiplier ** (attempt - 1))
        return min(self.max_delay_sec, raw)


@dataclass(slots=True)
class RestartState:
    """Persisted restart/backoff state."""

    consecutive_failures: int = 0
    total_attempts: int = 0
    last_attempt_at: str | None = None
    last_success_at: str | None = None
    next_attempt_at: str | None = None
    last_error: str | None = None
    last_pid: int | None = None
    healthy_since: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RestartState:
        return cls(
            consecutive_failures=_nonnegative_int(
                value.get("consecutive_failures"), default=0
            ),
            total_attempts=_nonnegative_int(value.get("total_attempts"), default=0),
            last_attempt_at=_optional_string(value.get("last_attempt_at")),
            last_success_at=_optional_string(value.get("last_success_at")),
            next_attempt_at=_optional_string(value.get("next_attempt_at")),
            last_error=_optional_string(value.get("last_error")),
            last_pid=_optional_positive_int(value.get("last_pid")),
            healthy_since=_optional_string(value.get("healthy_since")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MonitorConfig:
    """Runtime configuration for :class:`CampaignMonitor`."""

    campaign_dir: Path
    bridge_health_url: str = DEFAULT_BRIDGE_HEALTH_URL
    cb_command: str = DEFAULT_CB_COMMAND
    nodes: tuple[str, ...] = ()
    expected_gpus_per_node: int | None = None
    expected_claim_owner: str | None = None
    expected_claim_purpose: str | None = None
    pool_config: Path | None = None
    lease_heartbeat_path: Path | None = None
    lease_heartbeat_stale_sec: float = 2400.0
    heartbeat_stale_sec: float = 300.0
    checkpoint_stale_sec: float = 1800.0
    campaign_state_stale_sec: float = 900.0
    pipeline_progress_stale_sec: float = 14400.0
    bridge_timeout_sec: float = 5.0
    command_timeout_sec: float = 60.0
    cluster_probe_timeout_sec: float | None = None
    gpu_probe_timeout_sec: float | None = None
    pool_probe_timeout_sec: float | None = None
    external_probe_deadline_sec: float | None = None
    snapshot_stale_sec: float = 120.0
    resource_snapshot_path: Path | None = None
    poll_interval_sec: float = 30.0
    snapshot_path: Path | None = None
    monitor_state_path: Path | None = None
    pause_request_path: Path | None = None
    monitor_heartbeat_path: Path | None = None
    daemon_pid_path: Path | None = None
    restart_command: tuple[str, ...] = ()
    restart_policy: RestartPolicy = field(default_factory=RestartPolicy)

    def __post_init__(self) -> None:
        campaign_dir = Path(self.campaign_dir).expanduser().resolve()
        object.__setattr__(self, "campaign_dir", campaign_dir)
        if not self.bridge_health_url.strip():
            raise ValueError("bridge_health_url must not be empty")
        if not self.cb_command.strip():
            raise ValueError("cb_command must not be empty")
        for name in (
            "heartbeat_stale_sec",
            "checkpoint_stale_sec",
            "campaign_state_stale_sec",
            "pipeline_progress_stale_sec",
            "bridge_timeout_sec",
            "command_timeout_sec",
            "snapshot_stale_sec",
            "poll_interval_sec",
            "lease_heartbeat_stale_sec",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name, default in (
            ("cluster_probe_timeout_sec", min(self.command_timeout_sec, 15.0)),
            ("gpu_probe_timeout_sec", min(self.command_timeout_sec, 20.0)),
            ("pool_probe_timeout_sec", min(self.command_timeout_sec, 45.0)),
        ):
            value = getattr(self, name)
            value = default if value is None else float(value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        external_deadline = self.external_probe_deadline_sec
        if external_deadline is None:
            external_deadline = max(
                self.bridge_timeout_sec,
                self.cluster_probe_timeout_sec,
                self.gpu_probe_timeout_sec,
                self.pool_probe_timeout_sec,
            ) + 5.0
        external_deadline = float(external_deadline)
        if external_deadline <= 0:
            raise ValueError("external_probe_deadline_sec must be positive")
        object.__setattr__(
            self,
            "external_probe_deadline_sec",
            external_deadline,
        )
        if (
            self.expected_gpus_per_node is not None
            and int(self.expected_gpus_per_node) <= 0
        ):
            raise ValueError("expected_gpus_per_node must be positive")
        nodes = tuple(_validate_node(node) for node in self.nodes)
        object.__setattr__(self, "nodes", nodes)
        owner = _optional_string(self.expected_claim_owner)
        purpose = _optional_string(self.expected_claim_purpose)
        object.__setattr__(self, "expected_claim_owner", owner)
        object.__setattr__(self, "expected_claim_purpose", purpose)
        if self.pool_config is not None:
            object.__setattr__(
                self,
                "pool_config",
                Path(self.pool_config).expanduser().resolve(),
            )
        if self.lease_heartbeat_path is not None:
            object.__setattr__(
                self,
                "lease_heartbeat_path",
                Path(self.lease_heartbeat_path).expanduser().resolve(),
            )
        resource_snapshot_path = self.resource_snapshot_path
        if resource_snapshot_path is None:
            resource_snapshot_path = (
                Path(self.cb_command).expanduser().resolve().parent.parent
                / DEFAULT_RESOURCE_SNAPSHOT_NAME
            )
        else:
            resource_snapshot_path = (
                Path(resource_snapshot_path).expanduser().resolve()
            )
        object.__setattr__(
            self,
            "resource_snapshot_path",
            resource_snapshot_path,
        )
        object.__setattr__(
            self,
            "restart_command",
            tuple(str(part) for part in self.restart_command if str(part)),
        )
        for field_name, default_name in (
            ("snapshot_path", DEFAULT_SNAPSHOT_NAME),
            ("monitor_state_path", DEFAULT_MONITOR_STATE_NAME),
            ("pause_request_path", DEFAULT_PAUSE_REQUEST_NAME),
            ("monitor_heartbeat_path", DEFAULT_MONITOR_HEARTBEAT_NAME),
            ("daemon_pid_path", DEFAULT_DAEMON_PID_NAME),
        ):
            raw = getattr(self, field_name)
            path = (
                campaign_dir / default_name
                if raw is None
                else Path(raw).expanduser().resolve()
            )
            object.__setattr__(self, field_name, path)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(UTC)


def utc_iso(value: datetime | None = None) -> str:
    """Serialize a datetime in a stable UTC representation."""

    current = value or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).isoformat(timespec="seconds")


def parse_timestamp(value: object) -> datetime | None:
    """Best-effort parser for ISO timestamps and Unix epoch values."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        try:
            return datetime.fromtimestamp(float(text), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` with a pretty-printed JSON object."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: FileNotFoundError | None = None
    # CephFS/FUSE can briefly lose visibility of a just-created temp dentry.
    # Retry with a fresh temp name instead of crashing the 24/7 monitor.
    for attempt in range(3):
        fd, temporary = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
            return
        except FileNotFoundError as exc:
            last_error = exc
            temp_path.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            temp_path.unlink(missing_ok=True)
            raise
    if last_error is not None:
        raise last_error


def create_pause_request(
    campaign_dir: str | Path,
    *,
    reason: str = "operator requested cooperative pause",
    requested_by: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Create/update a cooperative pause request and return its path.

    A pause request is intentionally only a state file.  It does not send a
    signal, kill the supervisor, stop GPU tasks, or release ClusterBridge
    claims.  The supervisor can finish its current atomic operation and enter a
    resumable paused state.
    """

    root = Path(campaign_dir).expanduser().resolve()
    target = root / DEFAULT_PAUSE_REQUEST_NAME
    payload = {
        "action": "pause",
        "requested_at": utc_iso(now),
        "requested_by": requested_by or _default_requester(),
        "reason": reason,
        "semantics": "cooperative_pause_not_stop",
    }
    atomic_write_json(target, payload)
    return target


class CampaignMonitor:
    """Poll campaign, Bridge, process, ClusterBridge, and GPU health."""

    def __init__(
        self,
        config: MonitorConfig,
        *,
        runner: CommandRunner = subprocess.run,
        urlopen: Callable[..., Any] = urllib.request.urlopen,
        clock: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        pid_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self.config = config
        self._runner = runner
        self._uses_default_runner = runner is subprocess.run
        self._urlopen = urlopen
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._pid_alive = pid_alive or _pid_alive
        self._restart_state = self._load_restart_state()
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request that :meth:`run_forever` exit after the current poll."""

        self._stop_requested = True

    def poll_once(self) -> dict[str, Any]:
        """Collect one complete health snapshot and persist it atomically."""

        now = self._clock()
        observed_at = utc_iso(now)
        campaign_state, campaign_state_path, campaign_state_error = (
            self._read_first_json(_CAMPAIGN_STATE_CANDIDATES)
        )
        pause = self._pause_info(now)
        supervisor = self._check_supervisor(
            now,
            campaign_state=campaign_state,
        )
        campaign = self._check_campaign_state(
            now,
            campaign_state,
            campaign_state_path,
            campaign_state_error,
        )
        checkpoint = self._check_pipeline_checkpoint(now, campaign_state)
        progress = self._check_pipeline_progress(
            now,
            campaign_state=campaign_state,
            supervisor=supervisor,
        )
        lease = self._check_lease_keepalive(now)
        external_checks = self._run_external_checks(now)
        bridge = external_checks["bridge"]
        cluster = external_checks["cluster"]
        gpu = external_checks["gpu"]
        pool = external_checks["pool"]

        checks = {
            "supervisor": supervisor,
            "bridge": bridge,
            "campaign": campaign,
            "checkpoint": checkpoint,
            "progress": progress,
            "cluster": cluster,
            "gpu": gpu,
            "pool": pool,
            "lease": lease,
        }
        restart = self._maybe_restart(
            now,
            supervisor=supervisor,
            campaign=campaign,
            progress=progress,
            pause_requested=bool(pause["requested"]),
            dependencies=(bridge, cluster, gpu, pool, lease),
        )
        overall = _worst_status(check.status for check in checks.values())
        if pause["requested"] and overall == "ok":
            overall = "degraded"

        snapshot: dict[str, Any] = {
            "schema_version": 1,
            "campaign_dir": str(self.config.campaign_dir),
            "generated_at": observed_at,
            "overall": overall,
            "paused": bool(pause["requested"]),
            "pause": pause,
            "checks": {name: check.to_dict() for name, check in checks.items()},
            "restart": restart,
            "monitor": {
                "pid": os.getpid(),
                "poll_interval_sec": self.config.poll_interval_sec,
                "snapshot_path": str(self.config.snapshot_path),
            },
        }
        atomic_write_json(self.config.snapshot_path, snapshot)
        atomic_write_json(
            self.config.monitor_heartbeat_path,
            {
                "pid": os.getpid(),
                "timestamp": observed_at,
                "overall": overall,
                "campaign_dir": str(self.config.campaign_dir),
            },
        )
        return snapshot

    def run_forever(
        self,
        *,
        max_iterations: int | None = None,
        stop_when: Callable[[], bool] | None = None,
    ) -> int:
        """Poll until signalled, ``stop_when`` is true, or a test limit is hit."""

        if max_iterations is not None and max_iterations <= 0:
            return 0
        iterations = 0
        next_poll = self._monotonic()
        while not self._stop_requested:
            if stop_when is not None and stop_when():
                break
            self.poll_once()
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break
            next_poll += self.config.poll_interval_sec
            remaining = next_poll - self._monotonic()
            if remaining > 0:
                self._sleep(remaining)
            else:
                next_poll = self._monotonic()
        return iterations

    def _run_external_checks(self, now: datetime) -> dict[str, Check]:
        """Run independent network and cluster probes under one poll deadline."""

        probes: dict[str, Callable[[datetime], Check]] = {
            "bridge": self._check_bridge,
            "cluster": self._check_cluster,
            "gpu": self._check_gpus,
            "pool": self._check_pool,
        }
        executor = ThreadPoolExecutor(
            max_workers=len(probes),
            thread_name_prefix="rsi-monitor-external",
        )
        futures: dict[Future[Check], str] = {
            executor.submit(probe, now): name for name, probe in probes.items()
        }
        completed, pending = wait(
            futures,
            timeout=self.config.external_probe_deadline_sec,
        )
        results: dict[str, Check] = {}
        for future in completed:
            name = futures[future]
            try:
                results[name] = future.result()
            except BaseException as exc:  # noqa: BLE001
                results[name] = Check(
                    name,
                    "fail",
                    f"{name} probe raised {type(exc).__name__}: {exc}",
                    utc_iso(now),
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
        for future in pending:
            name = futures[future]
            future.cancel()
            results[name] = Check(
                name,
                "fail",
                f"{name} probe exceeded aggregate deadline "
                f"{self.config.external_probe_deadline_sec:.1f}s",
                utc_iso(now),
                {
                    "deadline_exceeded": True,
                    "aggregate_deadline_sec": (
                        self.config.external_probe_deadline_sec
                    ),
                },
            )
        executor.shutdown(wait=False, cancel_futures=True)
        return results

    def _check_supervisor(
        self,
        now: datetime,
        *,
        campaign_state: Mapping[str, Any] | None,
    ) -> Check:
        heartbeat, path, error = self._read_first_json(
            _SUPERVISOR_HEARTBEAT_CANDIDATES
        )
        observed_at = utc_iso(now)
        if heartbeat is None:
            detail = (
                f"supervisor heartbeat is unreadable: {error}"
                if error
                else "supervisor heartbeat is missing"
            )
            return Check(
                "supervisor",
                "fail",
                detail,
                observed_at,
                {"path": str(path) if path else None},
            )

        timestamp = _mapping_timestamp(heartbeat)
        age_sec = _age_seconds(now, timestamp)
        pid = _first_positive_int(heartbeat, _SUPERVISOR_PID_KEYS)
        if pid is None and campaign_state is not None:
            pid = _first_positive_int(campaign_state, _SUPERVISOR_PID_KEYS)
        alive = self._pid_alive(pid) if pid is not None else None
        stale = age_sec is None or age_sec > self.config.heartbeat_stale_sec
        campaign_phase = (
            _first_string(
                campaign_state or {},
                ("status", "phase", "campaign_status", "state"),
            )
            or ""
        ).strip().lower()
        data = {
            "path": str(path),
            "pid": pid,
            "pid_alive": alive,
            "timestamp": utc_iso(timestamp) if timestamp else None,
            "age_sec": _round_or_none(age_sec),
            "stale_after_sec": self.config.heartbeat_stale_sec,
            "heartbeat": dict(heartbeat),
            "campaign_phase": campaign_phase,
        }
        if campaign_phase in _TERMINAL_CAMPAIGN_STATES:
            return Check(
                "supervisor",
                "ok",
                f"supervisor is not required for terminal campaign state "
                f"{campaign_phase!r}",
                observed_at,
                data,
            )
        if alive is False:
            return Check(
                "supervisor",
                "fail",
                f"supervisor PID {pid} is not alive",
                observed_at,
                data,
            )
        if stale:
            return Check(
                "supervisor",
                "fail",
                "supervisor heartbeat is stale or has no valid timestamp",
                observed_at,
                data,
            )
        if pid is None:
            return Check(
                "supervisor",
                "degraded",
                "supervisor heartbeat is fresh but has no PID",
                observed_at,
                data,
            )
        return Check(
            "supervisor",
            "ok",
            f"supervisor heartbeat is fresh; PID {pid} is alive",
            observed_at,
            data,
        )

    def _check_campaign_state(
        self,
        now: datetime,
        state: Mapping[str, Any] | None,
        path: Path | None,
        error: str | None,
    ) -> Check:
        observed_at = utc_iso(now)
        if state is None:
            detail = (
                f"campaign state is unreadable: {error}"
                if error
                else "campaign state is missing"
            )
            return Check(
                "campaign",
                "fail",
                detail,
                observed_at,
                {"path": str(path) if path else None},
            )
        timestamp = _mapping_timestamp(state)
        if timestamp is None and path is not None:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        age_sec = _age_seconds(now, timestamp)
        phase = _first_string(
            state,
            ("status", "phase", "campaign_status", "state"),
        )
        normalized = (phase or "").strip().lower()
        data = {
            "path": str(path),
            "timestamp": utc_iso(timestamp) if timestamp else None,
            "age_sec": _round_or_none(age_sec),
            "stale_after_sec": self.config.campaign_state_stale_sec,
            "phase": phase,
            "cycle": state.get("cycle", state.get("current_cycle")),
            "state": dict(state),
        }
        if normalized in {"failed", "error", "crashed", "aborted"}:
            return Check(
                "campaign",
                "fail",
                f"campaign reports terminal failure state {phase!r}",
                observed_at,
                data,
            )
        if (
            age_sec is not None
            and age_sec > self.config.campaign_state_stale_sec
            and normalized not in {"complete", "completed", "done", "stopped"}
        ):
            return Check(
                "campaign",
                "degraded",
                "campaign state has not been updated recently",
                observed_at,
                data,
            )
        return Check(
            "campaign",
            "ok",
            f"campaign state is readable{f' ({phase})' if phase else ''}",
            observed_at,
            data,
        )

    def _check_pipeline_checkpoint(
        self,
        now: datetime,
        campaign_state: Mapping[str, Any] | None,
    ) -> Check:
        observed_at = utc_iso(now)
        candidates = list(_PIPELINE_CHECKPOINT_CANDIDATES)
        run_dir = self._extract_run_dir(campaign_state)
        if run_dir is not None:
            candidates[:0] = [
                str(run_dir / "checkpoint.json"),
                str(run_dir / "pipeline_checkpoint.json"),
            ]
        checkpoint, path, error = self._read_first_json(candidates)
        if checkpoint is None:
            campaign_phase = (
                _first_string(
                    campaign_state or {},
                    ("status", "phase", "campaign_status", "state"),
                )
                or ""
            ).lower()
            terminal = campaign_phase in _TERMINAL_CAMPAIGN_STATES
            status = (
                "ok"
                if terminal
                else (
                    "degraded"
                    if campaign_phase
                    in {
                        "created",
                        "queued",
                        "initializing",
                        "idle",
                    }
                    else "fail"
                )
            )
            detail = (
                f"pipeline checkpoint is unreadable: {error}"
                if error
                else (
                    "pipeline checkpoint is not required for terminal campaign "
                    f"state {campaign_phase!r}"
                    if terminal
                    else "pipeline checkpoint is missing"
                )
            )
            return Check(
                "checkpoint",
                status,
                detail,
                observed_at,
                {
                    "path": str(path) if path else None,
                    "run_dir": str(run_dir) if run_dir else None,
                },
            )
        timestamp = _mapping_timestamp(checkpoint)
        if timestamp is None and path is not None:
            timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        age_sec = _age_seconds(now, timestamp)
        stage = checkpoint.get(
            "last_completed_stage",
            checkpoint.get("stage", checkpoint.get("current_stage")),
        )
        data = {
            "path": str(path),
            "run_dir": str(run_dir) if run_dir else None,
            "timestamp": utc_iso(timestamp) if timestamp else None,
            "age_sec": _round_or_none(age_sec),
            "stale_after_sec": self.config.checkpoint_stale_sec,
            "last_completed_stage": stage,
            "checkpoint": dict(checkpoint),
        }
        if age_sec is not None and age_sec > self.config.checkpoint_stale_sec:
            return Check(
                "checkpoint",
                "degraded",
                "pipeline checkpoint is stale",
                observed_at,
                data,
            )
        return Check(
            "checkpoint",
            "ok",
            "pipeline checkpoint is readable",
            observed_at,
            data,
        )

    def _check_pipeline_progress(
        self,
        now: datetime,
        *,
        campaign_state: Mapping[str, Any] | None,
        supervisor: Check,
    ) -> Check:
        """Detect a live supervisor whose active pipeline stopped progressing.

        Supervisor heartbeats prove process liveness, not useful work.  Track the
        newest durable artifact in the active run so a wedged child eventually
        becomes restartable instead of looking healthy forever.
        """

        observed_at = utc_iso(now)
        if not campaign_state:
            return Check(
                "progress",
                "degraded",
                "pipeline progress cannot be checked without campaign state",
                observed_at,
                {},
            )
        phase = str(
            _first_string(
                campaign_state,
                ("phase", "status", "campaign_status", "state"),
            )
            or ""
        ).strip().lower()
        run_dir = self._extract_run_dir(campaign_state)
        child_pid = _first_positive_int(
            campaign_state,
            ("active_child_pid", "child_pid"),
        )
        if phase != "pipeline" or run_dir is None:
            return Check(
                "progress",
                "ok",
                f"pipeline progress watchdog is idle (phase={phase or 'unknown'})",
                observed_at,
                {
                    "phase": phase,
                    "run_dir": str(run_dir) if run_dir else None,
                    "child_pid": child_pid,
                },
            )

        candidates = [
            run_dir / "checkpoint.json",
            run_dir / "heartbeat.json",
            run_dir / "pipeline_summary.json",
            run_dir / "pipeline.log",
        ]
        latest_path: Path | None = None
        latest_timestamp: datetime | None = None
        errors: list[str] = []
        for path in candidates:
            try:
                timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except FileNotFoundError:
                continue
            except OSError as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
                continue
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_path = path
                latest_timestamp = timestamp

        age_sec = _age_seconds(now, latest_timestamp)
        child_alive = self._pid_alive(child_pid) if child_pid is not None else None
        data = {
            "phase": phase,
            "run_dir": str(run_dir),
            "child_pid": child_pid,
            "child_pid_alive": child_alive,
            "latest_progress_path": (
                str(latest_path) if latest_path is not None else None
            ),
            "latest_progress_at": (
                utc_iso(latest_timestamp) if latest_timestamp else None
            ),
            "age_sec": _round_or_none(age_sec),
            "stale_after_sec": self.config.pipeline_progress_stale_sec,
            "errors": errors,
        }
        if child_alive is False:
            return Check(
                "progress",
                "fail",
                f"active pipeline child PID {child_pid} is not alive",
                observed_at,
                data,
            )
        if latest_timestamp is None:
            return Check(
                "progress",
                "degraded",
                "active pipeline has no durable progress artifact yet",
                observed_at,
                data,
            )
        if (
            age_sec is not None
            and age_sec > self.config.pipeline_progress_stale_sec
        ):
            return Check(
                "progress",
                "fail",
                "active pipeline has made no durable progress within the "
                "configured deadline",
                observed_at,
                data,
            )
        if supervisor.status == "fail":
            return Check(
                "progress",
                "degraded",
                "pipeline artifacts are fresh but supervisor health failed",
                observed_at,
                data,
            )
        return Check(
            "progress",
            "ok",
            "active pipeline has recent durable progress",
            observed_at,
            data,
        )

    def _check_bridge(self, now: datetime) -> Check:
        observed_at = utc_iso(now)
        started = self._monotonic()
        request = urllib.request.Request(
            self.config.bridge_health_url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            response = self._urlopen(request, timeout=self.config.bridge_timeout_sec)
            try:
                raw = response.read()
                status_code = int(getattr(response, "status", 200))
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return Check(
                "bridge",
                "fail",
                f"Bridge health probe failed: {type(exc).__name__}: {exc}",
                observed_at,
                {
                    "url": self.config.bridge_health_url,
                    "latency_sec": round(self._monotonic() - started, 3),
                },
            )
        try:
            payload: Any = json.loads(_decode_output(raw))
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            return Check(
                "bridge",
                "fail",
                f"Bridge health returned invalid JSON: {exc}",
                observed_at,
                {
                    "url": self.config.bridge_health_url,
                    "http_status": status_code,
                    "latency_sec": round(self._monotonic() - started, 3),
                },
            )
        health_value = ""
        if isinstance(payload, Mapping):
            health_value = str(
                payload.get("status", payload.get("health", payload.get("ok", "")))
            ).lower()
        healthy = status_code < 400 and health_value in {
            "ok",
            "healthy",
            "ready",
            "true",
            "1",
        }
        data = {
            "url": self.config.bridge_health_url,
            "http_status": status_code,
            "latency_sec": round(self._monotonic() - started, 3),
            "payload": _json_safe(payload),
        }
        if not healthy:
            return Check(
                "bridge",
                "fail",
                f"Bridge is not healthy (HTTP {status_code}, status={health_value!r})",
                observed_at,
                data,
            )
        return Check(
            "bridge",
            "ok",
            f"Bridge health is {health_value!r}",
            observed_at,
            data,
        )

    def _check_cluster(self, now: datetime) -> Check:
        observed_at = utc_iso(now)
        env = {**os.environ, "CB_NO_AUTOCLAIM": "1"}
        commands = {
            "list": ["bash", self.config.cb_command, "list"],
            "claims": ["bash", self.config.cb_command, "claims"],
        }
        command_results = self._run_commands_parallel(
            commands,
            command_timeout_sec=self.config.cluster_probe_timeout_sec,
            deadline_sec=self.config.cluster_probe_timeout_sec,
            env=env,
        )
        list_result = command_results["list"]
        claims_result = command_results["claims"]
        resource_snapshot = self._read_resource_snapshot(now)
        configured = set(self.config.nodes)

        if list_result["error"]:
            fallback_nodes = self._resource_snapshot_nodes(resource_snapshot)
            claim_records = self._read_claim_records(now)
            active_claim_nodes = {
                node
                for node, record in claim_records["nodes"].items()
                if record.get("active")
                and self._claim_record_matches_expectation(record)
            }
            usable_fallback = (
                configured
                and configured.issubset(fallback_nodes)
                and configured.issubset(active_claim_nodes)
                and bool(resource_snapshot.get("fresh"))
                and not resource_snapshot.get("error")
                and not claim_records.get("error")
            )
            if usable_fallback:
                return Check(
                    "cluster",
                    "degraded",
                    "cb list failed; fresh resource-manager allocation and "
                    "claim records still confirm configured nodes",
                    observed_at,
                    {
                        "configured_nodes": list(self.config.nodes),
                        "list": list_result,
                        "claims": claims_result,
                        "resource_snapshot": resource_snapshot,
                        "claim_records": claim_records,
                        "fallback_used": True,
                    },
                )
            return Check(
                "cluster",
                "fail",
                f"ClusterBridge list probe failed: {list_result['error']}",
                observed_at,
                {
                    "configured_nodes": list(self.config.nodes),
                    "list": list_result,
                    "claims": claims_result,
                    "resource_snapshot": resource_snapshot,
                    "claim_records": claim_records,
                    "fallback_used": False,
                },
            )

        if claims_result["error"]:
            details = "; ".join(
                str(item)
                for item in (list_result["error"], claims_result["error"])
                if item
            )
            return Check(
                "cluster",
                "fail",
                f"ClusterBridge health probe failed: {details}",
                observed_at,
                {
                    "list": list_result,
                    "claims": claims_result,
                    "resource_snapshot": resource_snapshot,
                },
            )
        list_text = _combined_output(list_result)
        claims_text = _combined_output(claims_result)
        parsed_nodes = parse_cb_list(list_text)
        missing = sorted(configured - set(parsed_nodes))
        dead = sorted(
            node
            for node in configured
            if node in parsed_nodes and parsed_nodes[node]["alive"] != "yes"
        )
        unclaimed = sorted(
            node
            for node in configured
            if node in parsed_nodes
            and str(parsed_nodes[node]["claim"]).strip() == "unclaimed"
        )
        claim_mismatches: dict[str, str] = {}
        for node in configured:
            if node not in parsed_nodes:
                continue
            claim = str(parsed_nodes[node]["claim"]).strip()
            if claim == "unclaimed":
                continue
            owner, _, purpose = claim.partition("/")
            if (
                self.config.expected_claim_owner
                and owner != self.config.expected_claim_owner
            ):
                claim_mismatches[node] = (
                    f"owner {owner!r} != "
                    f"{self.config.expected_claim_owner!r}"
                )
                continue
            if (
                self.config.expected_claim_purpose
                and purpose != self.config.expected_claim_purpose
            ):
                claim_mismatches[node] = (
                    f"purpose {purpose!r} != "
                    f"{self.config.expected_claim_purpose!r}"
                )
        data = {
            "configured_nodes": list(self.config.nodes),
            "nodes": parsed_nodes,
            "missing_nodes": missing,
            "unhealthy_nodes": dead,
            "unclaimed_nodes": unclaimed,
            "claim_mismatches": claim_mismatches,
            "list": list_result,
            "claims": claims_result,
            "claims_text": claims_text,
            "resource_snapshot": resource_snapshot,
        }
        if self.config.nodes and not parsed_nodes:
            return Check(
                "cluster",
                "fail",
                "cb list output could not be parsed for configured nodes",
                observed_at,
                data,
            )
        if missing or dead or unclaimed or claim_mismatches:
            pieces: list[str] = []
            if missing:
                pieces.append(f"missing={','.join(missing)}")
            if dead:
                pieces.append(f"unhealthy={','.join(dead)}")
            if unclaimed:
                pieces.append(f"unclaimed={','.join(unclaimed)}")
            if claim_mismatches:
                pieces.append(
                    "claim_mismatch="
                    + ",".join(
                        f"{node}({detail})"
                        for node, detail in sorted(claim_mismatches.items())
                    )
                )
            return Check(
                "cluster",
                "fail",
                "ClusterBridge node/claim health failed: " + "; ".join(pieces),
                observed_at,
                data,
            )
        if not self.config.nodes:
            return Check(
                "cluster",
                "degraded",
                "cb list/claims succeeded; no campaign nodes were configured",
                observed_at,
                data,
            )
        return Check(
            "cluster",
            "ok",
            f"{len(self.config.nodes)} configured nodes are alive and claimed",
            observed_at,
            data,
        )

    def _check_pool(self, now: datetime) -> Check:
        """Verify durable pool state plus live Ray node/GPU identity."""

        observed_at = utc_iso(now)
        if self.config.pool_config is None:
            return Check(
                "pool",
                "degraded",
                "Ray pool probe skipped because no pool config was provided",
                observed_at,
                {},
            )
        if not self.config.pool_config.is_file():
            return Check(
                "pool",
                "fail",
                f"pool config is missing: {self.config.pool_config}",
                observed_at,
                {"config": str(self.config.pool_config)},
            )
        command = [
            sys.executable,
            "-m",
            "researchclaw.rsi.pool_probe",
            "--config",
            str(self.config.pool_config),
        ]
        result = self._run_command(
            command,
            timeout=self.config.pool_probe_timeout_sec,
            env=os.environ.copy(),
        )
        data = {"config": str(self.config.pool_config), "command": result}
        if result["error"]:
            return Check(
                "pool",
                "fail",
                f"Ray pool readiness probe failed: {result['error']}",
                observed_at,
                data,
            )
        try:
            payload = json.loads(result["stdout"])
        except json.JSONDecodeError as exc:
            return Check(
                "pool",
                "fail",
                f"Ray pool readiness probe returned invalid JSON: {exc}",
                observed_at,
                data,
            )
        if not isinstance(payload, dict):
            return Check(
                "pool",
                "fail",
                "Ray pool readiness probe did not return an object",
                observed_at,
                data,
            )
        data["pool"] = payload
        resources = payload.get("resources")
        total_gpu = (
            int(float(resources.get("total_gpu", 0)))
            if isinstance(resources, Mapping)
            else 0
        )
        alive_nodes = (
            int(resources.get("alive_nodes", 0))
            if isinstance(resources, Mapping)
            else 0
        )
        return Check(
            "pool",
            "ok",
            f"Ray pool is lease-verified and ready "
            f"({total_gpu} GPUs, {alive_nodes} nodes)",
            observed_at,
            data,
        )

    def _check_lease_keepalive(self, now: datetime) -> Check:
        observed_at = utc_iso(now)
        path = self.config.lease_heartbeat_path
        if path is None:
            return Check(
                "lease",
                "degraded",
                "lease keepalive heartbeat probe is not configured",
                observed_at,
                {},
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Check(
                "lease",
                "fail",
                f"lease keepalive heartbeat is missing: {path}",
                observed_at,
                {"path": str(path)},
            )
        except (OSError, json.JSONDecodeError) as exc:
            return Check(
                "lease",
                "fail",
                f"lease keepalive heartbeat is unreadable: {exc}",
                observed_at,
                {"path": str(path)},
            )
        if not isinstance(payload, dict):
            return Check(
                "lease",
                "fail",
                "lease keepalive heartbeat is not a JSON object",
                observed_at,
                {"path": str(path)},
            )
        timestamp = _mapping_timestamp(payload)
        age_sec = _age_seconds(now, timestamp)
        pid = _optional_positive_int(payload.get("pid"))
        alive = self._pid_alive(pid) if pid is not None else False
        status = str(payload.get("status", "")).lower()
        data = {
            "path": str(path),
            "pid": pid,
            "pid_alive": alive,
            "status": status,
            "timestamp": utc_iso(timestamp) if timestamp else None,
            "age_sec": _round_or_none(age_sec),
            "stale_after_sec": self.config.lease_heartbeat_stale_sec,
            "heartbeat": payload,
        }
        if (
            status != "ok"
            or not alive
            or age_sec is None
            or age_sec > self.config.lease_heartbeat_stale_sec
        ):
            return Check(
                "lease",
                "fail",
                "lease keepalive is failed, dead, or stale",
                observed_at,
                data,
            )
        return Check(
            "lease",
            "ok",
            f"lease keepalive PID {pid} is healthy",
            observed_at,
            data,
        )

    def _check_gpus(self, now: datetime) -> Check:
        observed_at = utc_iso(now)
        if not self.config.nodes:
            return Check(
                "gpu",
                "degraded",
                "GPU checks skipped because no campaign nodes were configured",
                observed_at,
                {"nodes": {}},
            )
        remote_command = (
            "nvidia-smi "
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu "
            "--format=csv,noheader,nounits"
        )
        results: dict[str, Any] = {}
        failures: list[str] = []
        degraded: list[str] = []
        commands = {
            node: [
                "bash",
                self.config.cb_command,
                node,
                "run",
                remote_command,
            ]
            for node in self.config.nodes
        }
        probe_results = self._run_commands_parallel(
            commands,
            command_timeout_sec=self.config.gpu_probe_timeout_sec,
            deadline_sec=self.config.gpu_probe_timeout_sec,
            env={
                **os.environ,
                "CB_NO_AUTOCLAIM": "1",
                "BASHBRIDGE_TIMEOUT": str(
                    max(1, int(self.config.gpu_probe_timeout_sec))
                ),
            },
        )
        for node in self.config.nodes:
            result = probe_results[node]
            if result["error"]:
                failures.append(node)
                results[node] = result
                continue
            gpus = parse_nvidia_smi(result["stdout"])
            result["gpus"] = gpus
            result["gpu_count"] = len(gpus)
            results[node] = result
            expected = self.config.expected_gpus_per_node
            if not gpus:
                failures.append(node)
            elif expected is not None and len(gpus) != expected:
                degraded.append(node)
        data = {
            "expected_gpus_per_node": self.config.expected_gpus_per_node,
            "nodes": results,
            "failed_nodes": failures,
            "count_mismatch_nodes": degraded,
            "total_gpus": sum(
                int(value.get("gpu_count", 0)) for value in results.values()
            ),
        }
        if failures:
            pool_ready = self._latest_snapshot_check_is_ok("pool", now)
            if pool_ready:
                resource_snapshot = self._read_resource_snapshot(now)
                claim_records = self._read_claim_records(now)
                configured = set(self.config.nodes)
                fallback_nodes = self._resource_snapshot_nodes(resource_snapshot)
                active_claim_nodes = {
                    node
                    for node, record in claim_records["nodes"].items()
                    if record.get("active")
                    and self._claim_record_matches_expectation(record)
                }
                if (
                    configured
                    and configured.issubset(fallback_nodes)
                    and configured.issubset(active_claim_nodes)
                    and bool(resource_snapshot.get("fresh"))
                    and not resource_snapshot.get("error")
                    and not claim_records.get("error")
                ):
                    data.update(
                        {
                            "fallback_used": True,
                            "resource_snapshot": resource_snapshot,
                            "claim_records": claim_records,
                            "latest_pool_check_ok": True,
                        }
                    )
                    return Check(
                        "gpu",
                        "degraded",
                        "nvidia-smi probes timed out; fresh allocation/claim "
                        "state and the latest Ray pool probe still confirm "
                        "32-GPU readiness",
                        observed_at,
                        data,
                    )
            return Check(
                "gpu",
                "fail",
                f"nvidia-smi failed or returned no GPUs on: {', '.join(failures)}",
                observed_at,
                data,
            )
        if degraded:
            return Check(
                "gpu",
                "degraded",
                "GPU count differs from expectation on: " + ", ".join(degraded),
                observed_at,
                data,
            )
        return Check(
            "gpu",
            "ok",
            f"nvidia-smi reports {data['total_gpus']} GPUs across "
            f"{len(self.config.nodes)} nodes",
            observed_at,
            data,
        )

    def _latest_snapshot_check_is_ok(
        self,
        check_name: str,
        now: datetime,
    ) -> bool:
        """Return whether the prior monitor snapshot has a fresh OK check."""

        try:
            payload = json.loads(
                self.config.snapshot_path.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, Mapping):
            return False
        generated = parse_timestamp(payload.get("generated_at"))
        age_sec = _age_seconds(now, generated)
        if (
            age_sec is None
            or age_sec < 0
            or age_sec > max(
                self.config.snapshot_stale_sec,
                self.config.poll_interval_sec * 5,
            )
        ):
            return False
        checks = payload.get("checks")
        if not isinstance(checks, Mapping):
            return False
        check = checks.get(check_name)
        return isinstance(check, Mapping) and check.get("status") == "ok"

    def _maybe_restart(
        self,
        now: datetime,
        *,
        supervisor: Check,
        campaign: Check,
        progress: Check | None = None,
        pause_requested: bool,
        dependencies: Sequence[Check] = (),
    ) -> dict[str, Any]:
        state = self._restart_state
        policy = self.config.restart_policy
        unhealthy_progress = progress is not None and progress.status == "fail"
        healthy = supervisor.status != "fail" and not unhealthy_progress
        if healthy:
            if state.healthy_since is None:
                state.healthy_since = utc_iso(now)
            healthy_since = parse_timestamp(state.healthy_since)
            if (
                healthy_since is not None
                and _age_seconds(now, healthy_since) is not None
                and _age_seconds(now, healthy_since) >= policy.reset_after_healthy_sec
            ):
                state.consecutive_failures = 0
                state.next_attempt_at = None
                state.last_error = None
            self._persist_restart_state()
            return self._restart_view("not_needed", now)

        state.healthy_since = None
        if pause_requested:
            self._persist_restart_state()
            return self._restart_view("suppressed_paused", now)
        campaign_phase = str(campaign.data.get("phase") or "").lower()
        if campaign_phase in _TERMINAL_CAMPAIGN_STATES:
            self._persist_restart_state()
            return self._restart_view("suppressed_terminal", now)
        if not self.config.restart_command:
            self._persist_restart_state()
            return self._restart_view("disabled", now)
        # A stale pipeline must be restartable even when the same wedged Ray
        # task makes pool/cluster probes time out.  Bridge, GPU visibility and
        # the allocation lease are the hard prerequisites for recovery; the
        # supervisor restart itself is responsible for reconciling stale Ray
        # work.
        hard_dependency_names = {"bridge", "gpu", "lease"}
        if not unhealthy_progress:
            hard_dependency_names.update({"cluster", "pool"})
        failed_dependencies = [
            check.name
            for check in dependencies
            if check.status == "fail" and check.name in hard_dependency_names
        ]
        if failed_dependencies:
            self._persist_restart_state()
            view = self._restart_view("suppressed_dependency_failure", now)
            view["failed_dependencies"] = failed_dependencies
            return view
        if state.consecutive_failures >= policy.max_attempts:
            self._persist_restart_state()
            return self._restart_view("max_attempts_reached", now)

        next_attempt = parse_timestamp(state.next_attempt_at)
        if next_attempt is not None and now < next_attempt:
            self._persist_restart_state()
            view = self._restart_view("backoff", now)
            view["remaining_sec"] = round(
                max(0.0, (next_attempt - now).total_seconds()), 3
            )
            return view

        attempt = state.consecutive_failures + 1
        state.total_attempts += 1
        state.last_attempt_at = utc_iso(now)
        result = self._run_command(
            list(self.config.restart_command),
            timeout=self.config.command_timeout_sec,
            env=os.environ.copy(),
        )
        if result["error"]:
            state.consecutive_failures = attempt
            state.last_error = str(result["error"])
            delay = policy.delay_for_attempt(attempt)
            state.next_attempt_at = utc_iso(
                datetime.fromtimestamp(now.timestamp() + delay, tz=UTC)
            )
            action = "failed"
        else:
            state.consecutive_failures = attempt
            state.last_success_at = utc_iso(now)
            state.last_error = None
            state.last_pid = _extract_pid(result["stdout"] + "\n" + result["stderr"])
            delay = policy.delay_for_attempt(attempt)
            state.next_attempt_at = utc_iso(
                datetime.fromtimestamp(now.timestamp() + delay, tz=UTC)
            )
            action = "started"
        self._persist_restart_state()
        view = self._restart_view(action, now)
        view["command_result"] = result
        return view

    def _restart_view(self, action: str, now: datetime) -> dict[str, Any]:
        data = self._restart_state.to_dict()
        data.update(
            {
                "action": action,
                "enabled": bool(self.config.restart_command),
                "command": list(self.config.restart_command),
                "policy": asdict(self.config.restart_policy),
                "observed_at": utc_iso(now),
            }
        )
        return data

    def _load_restart_state(self) -> RestartState:
        path = self.config.monitor_state_path
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return RestartState()
        if not isinstance(raw, Mapping):
            return RestartState()
        restart = raw.get("restart", raw)
        if not isinstance(restart, Mapping):
            return RestartState()
        return RestartState.from_mapping(restart)

    def _persist_restart_state(self) -> None:
        atomic_write_json(
            self.config.monitor_state_path,
            {
                "schema_version": 1,
                "updated_at": utc_iso(self._clock()),
                "restart": self._restart_state.to_dict(),
            },
        )

    def _pause_info(self, now: datetime) -> dict[str, Any]:
        paths = tuple(
            dict.fromkeys(
                (
                    self.config.campaign_dir / "control" / "pause",
                    self.config.pause_request_path,
                )
            )
        )
        for path in paths:
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return {
                    "requested": True,
                    "path": str(path),
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return {
                "requested": True,
                "path": str(path),
                "valid": isinstance(value, Mapping),
                "request": _json_safe(value),
                "age_sec": _round_or_none(
                    _age_seconds(
                        now,
                        _mapping_timestamp(value)
                        if isinstance(value, Mapping)
                        else None,
                    )
                ),
            }
        return {
            "requested": False,
            "path": str(self.config.pause_request_path),
            "paths_checked": [str(path) for path in paths],
        }

    def _read_first_json(
        self,
        candidates: Sequence[str],
    ) -> tuple[Mapping[str, Any] | None, Path | None, str | None]:
        first_existing: Path | None = None
        first_error: str | None = None
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                path = self.config.campaign_dir / path
            if not path.exists():
                continue
            if first_existing is None:
                first_existing = path
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                if first_error is None:
                    first_error = f"{path}: {type(exc).__name__}: {exc}"
                continue
            if not isinstance(value, Mapping):
                if first_error is None:
                    first_error = f"{path}: JSON root must be an object"
                continue
            return value, path, None
        return None, first_existing, first_error

    def _extract_run_dir(
        self,
        campaign_state: Mapping[str, Any] | None,
    ) -> Path | None:
        if campaign_state is None:
            return None
        raw: object | None = None
        for key in _RUN_DIR_KEYS:
            if campaign_state.get(key):
                raw = campaign_state[key]
                break
        if raw is None and isinstance(campaign_state.get("run"), Mapping):
            nested = campaign_state["run"]
            for key in ("dir", "path", *_RUN_DIR_KEYS):
                if nested.get(key):
                    raw = nested[key]
                    break
        if raw is None and isinstance(campaign_state.get("pipeline"), Mapping):
            nested = campaign_state["pipeline"]
            for key in ("run_dir", "dir", "path"):
                if nested.get(key):
                    raw = nested[key]
                    break
        if raw is None:
            return None
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = self.config.campaign_dir / path
        return path.resolve()

    def _run_commands_parallel(
        self,
        commands: Mapping[str, Sequence[str]],
        *,
        command_timeout_sec: float,
        deadline_sec: float,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run independent commands concurrently with a bounded group deadline."""

        if not commands:
            return {}
        command_runner = (
            self._run_command
            if self._uses_default_runner
            else self._run_command_with_deadline
        )
        executor = ThreadPoolExecutor(
            max_workers=len(commands),
            thread_name_prefix="rsi-monitor-command",
        )
        futures: dict[Future[dict[str, Any]], str] = {
            executor.submit(
                command_runner,
                args,
                timeout=command_timeout_sec,
                env=env,
            ): name
            for name, args in commands.items()
        }
        completed, pending = wait(futures, timeout=deadline_sec)
        results: dict[str, dict[str, Any]] = {}
        for future in completed:
            name = futures[future]
            try:
                results[name] = future.result()
            except BaseException as exc:  # noqa: BLE001
                results[name] = self._command_error_result(
                    commands[name],
                    f"{type(exc).__name__}: {exc}",
                )
        for future in pending:
            name = futures[future]
            future.cancel()
            results[name] = self._command_error_result(
                commands[name],
                f"ProbeDeadlineExceeded: command group exceeded "
                f"{deadline_sec:.1f}s",
                deadline_exceeded=True,
            )
        executor.shutdown(wait=False, cancel_futures=True)
        return results

    def _run_command_with_deadline(
        self,
        args: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Bound an injected runner that may not enforce subprocess timeouts."""

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rsi-monitor-runner",
        )
        future = executor.submit(
            self._run_command,
            args,
            timeout=timeout,
            env=env,
        )
        completed, _ = wait((future,), timeout=timeout)
        if completed:
            try:
                result = future.result()
            except BaseException as exc:  # noqa: BLE001
                result = self._command_error_result(
                    args,
                    f"{type(exc).__name__}: {exc}",
                )
        else:
            future.cancel()
            result = self._command_error_result(
                args,
                f"TimeoutExpired: command exceeded {timeout:.1f}s",
                deadline_exceeded=True,
            )
        executor.shutdown(wait=False, cancel_futures=True)
        return result

    def _command_error_result(
        self,
        args: Sequence[str],
        error: str,
        *,
        deadline_exceeded: bool = False,
    ) -> dict[str, Any]:
        return {
            "argv": [str(part) for part in args],
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "elapsed_sec": 0.0,
            "error": error,
            "deadline_exceeded": deadline_exceeded,
        }

    def _read_resource_snapshot(self, now: datetime) -> dict[str, Any]:
        """Read the resource-manager snapshot as a bounded local fallback."""

        path = self.config.resource_snapshot_path
        data: dict[str, Any] = {
            "path": str(path),
            "fresh": False,
            "age_sec": None,
            "stale_after_sec": self.config.snapshot_stale_sec,
        }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            data["error"] = "resource-manager snapshot is missing"
            return data
        except (OSError, json.JSONDecodeError) as exc:
            data["error"] = f"{type(exc).__name__}: {exc}"
            return data
        if not isinstance(payload, Mapping):
            data["error"] = "resource-manager snapshot is not a JSON object"
            return data
        timestamp = parse_timestamp(payload.get("updated_at"))
        age_sec = _age_seconds(now, timestamp)
        data.update(
            {
                "timestamp": utc_iso(timestamp) if timestamp else None,
                "age_sec": _round_or_none(age_sec),
                "fresh": (
                    age_sec is not None
                    and 0 <= age_sec <= self.config.snapshot_stale_sec
                ),
                "summary": _json_safe(payload.get("summary")),
                "allocations": _json_safe(payload.get("allocations")),
            }
        )
        return data

    def _read_claim_records(self, now: datetime) -> dict[str, Any]:
        """Read configured nodes' authoritative local claim records."""

        bridge_root = (
            Path(self.config.cb_command).expanduser().resolve().parent.parent
            / ".bridge"
        )
        records: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for node in self.config.nodes:
            path = bridge_root / node / "claim.json"
            record: dict[str, Any] = {"path": str(path), "active": False}
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                record["error"] = "claim record is missing"
            except (OSError, json.JSONDecodeError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            else:
                if not isinstance(payload, Mapping):
                    record["error"] = "claim record is not a JSON object"
                else:
                    expires_at = parse_timestamp(payload.get("expires_at"))
                    record.update(_json_safe(payload))
                    record["active"] = bool(
                        expires_at is not None and expires_at > now
                    )
                    record["expires_at"] = (
                        utc_iso(expires_at) if expires_at else None
                    )
                    if not record["active"]:
                        record["error"] = "claim record is expired or invalid"
            if record.get("error"):
                errors.append(f"{node}: {record['error']}")
            records[node] = record
        return {
            "root": str(bridge_root),
            "nodes": records,
            "error": "; ".join(errors) if errors else None,
        }

    def _claim_record_matches_expectation(
        self,
        record: Mapping[str, Any],
    ) -> bool:
        if (
            self.config.expected_claim_owner
            and str(record.get("owner", "")).strip()
            != self.config.expected_claim_owner
        ):
            return False
        return not (
            self.config.expected_claim_purpose
            and str(record.get("purpose", "")).strip()
            != self.config.expected_claim_purpose
        )

    @staticmethod
    def _resource_snapshot_nodes(snapshot: Mapping[str, Any]) -> set[str]:
        nodes: set[str] = set()
        allocations = snapshot.get("allocations")
        if not isinstance(allocations, Sequence) or isinstance(
            allocations,
            (str, bytes),
        ):
            return nodes
        for allocation in allocations:
            if not isinstance(allocation, Mapping):
                continue
            if str(allocation.get("status", "")).lower() != "active":
                continue
            raw_nodes = allocation.get("nodes")
            if not isinstance(raw_nodes, Sequence) or isinstance(
                raw_nodes,
                (str, bytes),
            ):
                continue
            nodes.update(str(node) for node in raw_nodes if str(node).strip())
        return nodes

    def _run_command(
        self,
        args: Sequence[str],
        *,
        timeout: float,
        env: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        started = self._monotonic()
        argv = [str(part) for part in args]
        try:
            completed = self._runner(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=dict(env) if env is not None else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "argv": argv,
                "returncode": None,
                "stdout": _decode_output(exc.stdout),
                "stderr": _decode_output(exc.stderr),
                "elapsed_sec": round(self._monotonic() - started, 3),
                "error": f"TimeoutExpired: command exceeded {timeout:.1f}s",
            }
        except (OSError, ValueError) as exc:
            return {
                "argv": argv,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "elapsed_sec": round(self._monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        stdout = _decode_output(getattr(completed, "stdout", ""))
        stderr = _decode_output(getattr(completed, "stderr", ""))
        returncode = int(getattr(completed, "returncode", 1))
        error = None
        if returncode != 0:
            detail = stderr.strip() or stdout.strip() or "no output"
            error = f"exit {returncode}: {detail}"
        return {
            "argv": argv,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_sec": round(self._monotonic() - started, 3),
            "error": error,
        }


def parse_cb_list(text: str) -> dict[str, dict[str, Any]]:
    """Parse the stable columns emitted by ``clusterbridge.sh list``."""

    nodes: dict[str, dict[str, Any]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("NODE ", "-- alive", "[")):
            continue
        match = _CB_LIST_ROW_RE.match(line)
        if match is None:
            continue
        values = match.groupdict()
        gpu_text = values["gpus"]
        nodes[values["node"]] = {
            "alive": values["alive"],
            "gpus": int(gpu_text) if gpu_text.isdigit() else None,
            "cluster": values["cluster"],
            "claim": values["claim"].strip(),
            "expires": values["expires"],
        }
    return nodes


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    """Parse nounits CSV produced by the monitor's ``nvidia-smi`` query."""

    gpus: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _NVIDIA_ROW_RE.match(line)
        if match is None:
            continue
        values = match.groupdict()
        gpus.append(
            {
                "index": int(values["index"]),
                "name": values["name"].strip(),
                "memory_total_mib": int(values["memory_total"]),
                "memory_used_mib": int(values["memory_used"]),
                "utilization_gpu_percent": int(values["utilization"]),
            }
        )
    return gpus


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by ``bin/rsi-monitor`` and the daemon."""

    parser = argparse.ArgumentParser(
        prog="rsi-monitor",
        description="Monitor one persistent AutoResearch RSI campaign.",
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument(
        "--bridge-health-url",
        default=os.environ.get("RSI_BRIDGE_HEALTH_URL", DEFAULT_BRIDGE_HEALTH_URL),
    )
    parser.add_argument(
        "--cb-command",
        default=os.environ.get("RSI_CB_COMMAND", DEFAULT_CB_COMMAND),
    )
    parser.add_argument(
        "--node",
        dest="nodes",
        action="append",
        default=[],
        help="Expected ClusterBridge node; repeat for every campaign node.",
    )
    parser.add_argument(
        "--nodes",
        dest="nodes_csv",
        default=os.environ.get("RSI_NODES", ""),
        help="Comma-separated expected ClusterBridge nodes.",
    )
    parser.add_argument("--expected-gpus-per-node", type=int)
    parser.add_argument(
        "--expected-claim-owner",
        default=os.environ.get("RSI_EXPECTED_CLAIM_OWNER", ""),
    )
    parser.add_argument(
        "--expected-claim-purpose",
        default=os.environ.get("RSI_EXPECTED_CLAIM_PURPOSE", ""),
    )
    parser.add_argument(
        "--pool-config",
        type=Path,
        default=(
            Path(os.environ["RSI_POOL_CONFIG"])
            if os.environ.get("RSI_POOL_CONFIG")
            else None
        ),
        help="ClusterBridge pool YAML to lease-verify and probe via Ray.",
    )
    parser.add_argument(
        "--lease-heartbeat",
        type=Path,
        default=(
            Path(os.environ["RSI_LEASE_HEARTBEAT"])
            if os.environ.get("RSI_LEASE_HEARTBEAT")
            else None
        ),
    )
    parser.add_argument(
        "--lease-heartbeat-stale-sec",
        type=float,
        default=2400.0,
    )
    parser.add_argument("--heartbeat-stale-sec", type=float, default=300.0)
    parser.add_argument("--checkpoint-stale-sec", type=float, default=1800.0)
    parser.add_argument("--campaign-state-stale-sec", type=float, default=900.0)
    parser.add_argument(
        "--pipeline-progress-stale-sec",
        type=float,
        default=14400.0,
        help=(
            "restart a live-but-wedged active pipeline after this many seconds "
            "without a checkpoint, heartbeat, summary, or log update"
        ),
    )
    parser.add_argument("--bridge-timeout-sec", type=float, default=5.0)
    parser.add_argument("--command-timeout-sec", type=float, default=60.0)
    parser.add_argument("--cluster-probe-timeout-sec", type=float)
    parser.add_argument("--gpu-probe-timeout-sec", type=float)
    parser.add_argument("--pool-probe-timeout-sec", type=float)
    parser.add_argument(
        "--external-probe-deadline-sec",
        type=float,
    )
    parser.add_argument("--snapshot-stale-sec", type=float, default=120.0)
    parser.add_argument("--resource-snapshot", type=Path)
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--monitor-state", type=Path)
    parser.add_argument("--pause-request", type=Path)
    parser.add_argument("--monitor-heartbeat", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument(
        "--restart-command-json",
        default=os.environ.get("RSI_RESTART_COMMAND_JSON", ""),
        help="JSON argv array used to restart an unhealthy supervisor.",
    )
    parser.add_argument("--restart-initial-delay-sec", type=float, default=30.0)
    parser.add_argument("--restart-multiplier", type=float, default=2.0)
    parser.add_argument("--restart-max-delay-sec", type=float, default=1800.0)
    parser.add_argument("--restart-max-attempts", type=int, default=8)
    parser.add_argument("--restart-reset-after-sec", type=float, default=900.0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Collect one snapshot and exit (default).",
    )
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Poll continuously until SIGINT or SIGTERM.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Test-safe upper bound for loop iterations.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print each collected snapshot to stdout.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> MonitorConfig:
    """Convert parsed CLI arguments to :class:`MonitorConfig`."""

    restart_command: tuple[str, ...] = ()
    if args.restart_command_json:
        try:
            raw = json.loads(args.restart_command_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid --restart-command-json: {exc}") from exc
        if not isinstance(raw, list) or not raw or not all(
            isinstance(part, str) and part for part in raw
        ):
            raise ValueError(
                "--restart-command-json must be a non-empty JSON string array"
            )
        restart_command = tuple(raw)
    nodes = list(args.nodes)
    nodes.extend(
        item.strip() for item in str(args.nodes_csv).split(",") if item.strip()
    )
    return MonitorConfig(
        campaign_dir=args.campaign_dir,
        bridge_health_url=args.bridge_health_url,
        cb_command=args.cb_command,
        nodes=tuple(dict.fromkeys(nodes)),
        expected_gpus_per_node=args.expected_gpus_per_node,
        expected_claim_owner=args.expected_claim_owner,
        expected_claim_purpose=args.expected_claim_purpose,
        pool_config=args.pool_config,
        lease_heartbeat_path=args.lease_heartbeat,
        lease_heartbeat_stale_sec=args.lease_heartbeat_stale_sec,
        heartbeat_stale_sec=args.heartbeat_stale_sec,
        checkpoint_stale_sec=args.checkpoint_stale_sec,
        campaign_state_stale_sec=args.campaign_state_stale_sec,
        pipeline_progress_stale_sec=args.pipeline_progress_stale_sec,
        bridge_timeout_sec=args.bridge_timeout_sec,
        command_timeout_sec=args.command_timeout_sec,
        cluster_probe_timeout_sec=args.cluster_probe_timeout_sec,
        gpu_probe_timeout_sec=args.gpu_probe_timeout_sec,
        pool_probe_timeout_sec=args.pool_probe_timeout_sec,
        external_probe_deadline_sec=args.external_probe_deadline_sec,
        snapshot_stale_sec=args.snapshot_stale_sec,
        resource_snapshot_path=args.resource_snapshot,
        poll_interval_sec=args.interval_sec,
        snapshot_path=args.snapshot,
        monitor_state_path=args.monitor_state,
        pause_request_path=args.pause_request,
        monitor_heartbeat_path=args.monitor_heartbeat,
        daemon_pid_path=args.pid_file,
        restart_command=restart_command,
        restart_policy=RestartPolicy(
            initial_delay_sec=args.restart_initial_delay_sec,
            multiplier=args.restart_multiplier,
            max_delay_sec=args.restart_max_delay_sec,
            max_attempts=args.restart_max_attempts,
            reset_after_healthy_sec=args.restart_reset_after_sec,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for foreground and detached monitor processes."""

    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    config.campaign_dir.mkdir(parents=True, exist_ok=True)
    monitor = CampaignMonitor(config)

    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        monitor.request_stop()
        if signum == signal.SIGTERM:
            return

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(signum, handle_signal)
        except (OSError, ValueError):
            pass

    try:
        if args.loop:
            try:
                _write_pid_file(config.daemon_pid_path)
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if args.print_json:
                iterations = 0
                while not monitor._stop_requested:
                    snapshot = monitor.poll_once()
                    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
                    sys.stdout.flush()
                    iterations += 1
                    if (
                        args.max_iterations is not None
                        and iterations >= args.max_iterations
                    ):
                        break
                    monitor._sleep(config.poll_interval_sec)
            else:
                monitor.run_forever(max_iterations=args.max_iterations)
        else:
            snapshot = monitor.poll_once()
            if args.print_json:
                print(json.dumps(snapshot, ensure_ascii=False, indent=2))
            else:
                print(
                    f"{snapshot['overall']}: {config.snapshot_path}",
                    flush=True,
                )
        return 0
    finally:
        if args.loop:
            _remove_own_pid_file(config.daemon_pid_path)
        for signum, previous in previous_handlers.items():
            try:
                signal.signal(signum, previous)
            except (OSError, ValueError):
                pass


def _write_pid_file(path: Path) -> None:
    if path.exists():
        existing, expected_ticks = _read_pid_identity(path)
        identity_matches = (
            expected_ticks is None
            or _process_start_ticks(existing or 0) == expected_ticks
        )
        if (
            existing is not None
            and existing != os.getpid()
            and _pid_alive(existing)
            and identity_matches
        ):
            raise RuntimeError(f"monitor already running with PID {existing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        path,
        {
            "pid": os.getpid(),
            "start_ticks": _process_start_ticks(os.getpid()),
        },
    )


def _remove_own_pid_file(path: Path) -> None:
    existing, expected_ticks = _read_pid_identity(path)
    if (
        existing == os.getpid()
        and (
            expected_ticks is None
            or expected_ticks == _process_start_ticks(os.getpid())
        )
    ):
        path.unlink(missing_ok=True)


def _read_pid_file(path: Path) -> int | None:
    return _read_pid_identity(path)[0]


def _read_pid_identity(path: Path) -> tuple[int | None, int | None]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None, None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _optional_positive_int(text), None
    if isinstance(payload, Mapping):
        return (
            _optional_positive_int(payload.get("pid")),
            _optional_positive_int(payload.get("start_ticks")),
        )
    return _optional_positive_int(payload), None


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _mapping_timestamp(value: Mapping[str, Any]) -> datetime | None:
    for key in _TIMESTAMP_KEYS:
        parsed = parse_timestamp(value.get(key))
        if parsed is not None:
            return parsed
    return None


def _age_seconds(now: datetime, timestamp: datetime | None) -> float | None:
    if timestamp is None:
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return max(0.0, (now.astimezone(UTC) - timestamp).total_seconds())


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _first_positive_int(
    value: Mapping[str, Any],
    keys: Sequence[str],
) -> int | None:
    for key in keys:
        result = _optional_positive_int(value.get(key))
        if result is not None:
            return result
    return None


def _first_string(
    value: Mapping[str, Any],
    keys: Sequence[str],
) -> str | None:
    for key in keys:
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_positive_int(value: object) -> int | None:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _nonnegative_int(value: object, *, default: int) -> int:
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(0, result)


def _validate_node(value: object) -> str:
    node = str(value).strip()
    if not node or _NODE_RE.fullmatch(node) is None:
        raise ValueError(f"invalid ClusterBridge node: {value!r}")
    return node


def _worst_status(statuses: Sequence[str] | Any) -> str:
    worst = "ok"
    for status in statuses:
        if _HEALTH_RANK.get(status, 2) > _HEALTH_RANK[worst]:
            worst = status
    return worst


def _decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _combined_output(result: Mapping[str, Any]) -> str:
    return f"{result.get('stdout', '')}\n{result.get('stderr', '')}".strip()


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def _extract_pid(text: str) -> int | None:
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        value = None
    if isinstance(value, Mapping):
        pid = _optional_positive_int(value.get("pid"))
        if pid is not None:
            return pid
    for pattern in (
        r"\bpid\s*=\s*(\d+)\b",
        r"\bPID\s+(\d+)\b",
        r"^\s*(\d+)\s*$",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match is not None:
            return _optional_positive_int(match.group(1))
    return None


def _default_requester() -> str:
    return (
        os.environ.get("USER")
        or os.environ.get("LOGNAME")
        or f"pid:{os.getpid()}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
