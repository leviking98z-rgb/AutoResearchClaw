"""Safe subprocess wrapper around the authoritative ClusterBridge script."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from researchclaw.cluster.models import ClusterNode

SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """Captured result of one ClusterBridge invocation."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    elapsed_sec: float


class ClusterBridgeError(RuntimeError):
    """Raised when the ClusterBridge transport or a remote command fails."""

    def __init__(self, message: str, result: BridgeResult | None = None) -> None:
        super().__init__(message)
        self.result = result


class UnsafeForceClaimError(ClusterBridgeError):
    """Raised when a caller requests force-claim without an explicit opt-in."""


class ClusterBridgeClient:
    """Invoke ``clusterbridge.sh`` without SSH or implicit node claims.

    Node execution sets ``CB_NO_AUTOCLAIM=1``.  The higher-level pool must
    therefore claim all nodes explicitly before running any command.  This
    prevents a harmless health probe from silently taking a lease.
    """

    def __init__(
        self,
        cb_command: str | Path,
        *,
        command_timeout_sec: float = 120.0,
        allow_force_claim: bool = False,
        runner: SubprocessRunner = subprocess.run,
    ) -> None:
        self.cb_command = str(Path(cb_command).expanduser())
        self.command_timeout_sec = float(command_timeout_sec)
        self.allow_force_claim = bool(allow_force_claim)
        self._runner = runner

    def list_nodes(self) -> BridgeResult:
        return self._invoke(["list"], timeout_sec=self.command_timeout_sec)

    def claims(self) -> BridgeResult:
        return self._invoke(["claims"], timeout_sec=self.command_timeout_sec)

    def claim_records(
        self,
        nodes: Iterable[ClusterNode | str],
    ) -> dict[str, dict[str, object]]:
        """Read the authoritative active claim records for *nodes*.

        ClusterBridge currently exposes human-oriented ``list``/``claims``
        output but stores the actual lease records as JSON beside each node
        queue.  Reading those records gives pool restore/renew/release a
        fail-closed ownership check without changing the shared ClusterBridge
        script or relying on fragile column parsing.
        """

        addresses = _addresses(nodes)
        if not addresses:
            raise ValueError("claim_records requires at least one node")
        bridge_root = Path(self.cb_command).expanduser().resolve().parent.parent
        records: dict[str, dict[str, object]] = {}
        now = time.time()
        for address in addresses:
            path = bridge_root / ".bridge" / address / "claim.json"
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                records[address] = {
                    "active": False,
                    "path": str(path),
                    "error": "claim file is missing",
                }
                continue
            except (OSError, json.JSONDecodeError) as exc:
                records[address] = {
                    "active": False,
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            if not isinstance(raw, dict):
                records[address] = {
                    "active": False,
                    "path": str(path),
                    "error": "claim record is not a JSON object",
                }
                continue
            expires_at = str(raw.get("expires_at", "")).strip()
            try:
                expires_epoch = datetime.strptime(
                    expires_at,
                    "%Y-%m-%dT%H:%M:%SZ",
                ).replace(tzinfo=UTC).timestamp()
            except (OverflowError, ValueError):
                expires_epoch = 0.0
            records[address] = {
                **raw,
                "active": expires_epoch > now,
                "path": str(path),
                "expires_epoch": expires_epoch,
            }
        return records

    def claim(
        self,
        nodes: Iterable[ClusterNode | str],
        *,
        purpose: str,
        ttl_min: int,
        force: bool = False,
    ) -> BridgeResult:
        addresses = _addresses(nodes)
        if not addresses:
            raise ValueError("claim requires at least one node")
        if force and not self.allow_force_claim:
            raise UnsafeForceClaimError(
                "force claim is disabled; set allow_force_claim explicitly"
            )
        args = ["claim", *addresses, "--purpose", purpose, "--ttl", str(ttl_min)]
        if force:
            args.append("--force")
        return self._invoke(args, timeout_sec=self.command_timeout_sec)

    def renew(
        self,
        nodes: Iterable[ClusterNode | str],
        *,
        ttl_min: int,
    ) -> BridgeResult:
        addresses = _addresses(nodes)
        if not addresses:
            raise ValueError("renew requires at least one node")
        return self._invoke(
            ["renew", *addresses, "--ttl", str(ttl_min)],
            timeout_sec=self.command_timeout_sec,
        )

    def release(
        self,
        nodes: Iterable[ClusterNode | str],
        *,
        force: bool = False,
    ) -> BridgeResult:
        addresses = _addresses(nodes)
        if not addresses:
            raise ValueError("release requires at least one node")
        if force and not self.allow_force_claim:
            raise UnsafeForceClaimError(
                "force release is disabled; set allow_force_claim explicitly"
            )
        args = ["release", *addresses]
        if force:
            args.append("--force")
        return self._invoke(args, timeout_sec=self.command_timeout_sec)

    def run_node(
        self,
        node: ClusterNode | str,
        command: str,
        *,
        timeout_sec: float | None = None,
    ) -> BridgeResult:
        address = node.address if isinstance(node, ClusterNode) else str(node)
        timeout = float(timeout_sec or self.command_timeout_sec)
        env = {
            **os.environ,
            "CB_NO_AUTOCLAIM": "1",
            "BASHBRIDGE_TIMEOUT": str(max(1, int(timeout))),
        }
        return self._invoke(
            [address, "run", command],
            timeout_sec=timeout + 30,
            env=env,
        )

    def detach(
        self,
        command: Sequence[str],
        *,
        timeout_sec: float | None = None,
    ) -> BridgeResult:
        if not command:
            raise ValueError("detach requires a command")
        return self._invoke(
            ["detach", *[str(part) for part in command]],
            timeout_sec=float(timeout_sec or self.command_timeout_sec),
        )

    def _invoke(
        self,
        args: Sequence[str],
        *,
        timeout_sec: float,
        env: dict[str, str] | None = None,
    ) -> BridgeResult:
        argv = ("bash", self.cb_command, *[str(arg) for arg in args])
        started = time.monotonic()
        try:
            completed = self._runner(
                list(argv),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClusterBridgeError(
                f"ClusterBridge command timed out after {timeout_sec:.1f}s: "
                f"{' '.join(argv[:4])}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ClusterBridgeError(
                f"could not execute ClusterBridge command {self.cb_command}: {exc}"
            ) from exc

        result = BridgeResult(
            argv=argv,
            returncode=int(completed.returncode),
            stdout=_decode(completed.stdout),
            stderr=_decode(completed.stderr),
            elapsed_sec=time.monotonic() - started,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no output"
            raise ClusterBridgeError(
                f"ClusterBridge command failed with exit {result.returncode}: "
                f"{detail}",
                result,
            )
        return result


def _addresses(nodes: Iterable[ClusterNode | str]) -> list[str]:
    return [
        node.address if isinstance(node, ClusterNode) else str(node)
        for node in nodes
    ]


def _decode(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
