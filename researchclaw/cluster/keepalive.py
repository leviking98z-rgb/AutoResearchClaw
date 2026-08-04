"""Lease-renewal worker used by long-running pool owners."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KeepaliveSnapshot:
    running: bool
    renew_count: int
    consecutive_failures: int
    last_success_monotonic: float | None
    last_error: str | None


class LeaseKeepalive:
    """Renew a lease periodically and expose failures to the task runner."""

    def __init__(
        self,
        renew: Callable[[], None],
        *,
        interval_sec: float,
        max_failures: int,
        on_update: Callable[[KeepaliveSnapshot], None] | None = None,
    ) -> None:
        if interval_sec <= 0:
            raise ValueError("interval_sec must be positive")
        if max_failures <= 0:
            raise ValueError("max_failures must be positive")
        self._renew = renew
        self._interval_sec = float(interval_sec)
        self._max_failures = int(max_failures)
        self._on_update = on_update
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._renew_count = 0
        self._consecutive_failures = 0
        self._last_success_monotonic: float | None = None
        self._last_error: str | None = None
        self._terminal_error: RuntimeError | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            # A restarted worker gets a fresh health budget.  Without clearing
            # this, a previous terminal failure poisons all future starts even
            # after a successful lease renewal.
            self._terminal_error = None
            self._consecutive_failures = 0
            self._last_error = None
            self._thread = threading.Thread(
                target=self._run,
                name="clusterbridge-lease-keepalive",
                daemon=True,
            )
            self._thread.start()
        self._notify()

    def stop(self, timeout_sec: float = 10.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout_sec))
        self._notify()

    def assert_healthy(self) -> None:
        with self._lock:
            error = self._terminal_error
        if error is not None:
            raise error

    def snapshot(self) -> KeepaliveSnapshot:
        with self._lock:
            return KeepaliveSnapshot(
                running=bool(self._thread and self._thread.is_alive()),
                renew_count=self._renew_count,
                consecutive_failures=self._consecutive_failures,
                last_success_monotonic=self._last_success_monotonic,
                last_error=self._last_error,
            )

    def _run(self) -> None:
        while not self._stop.wait(self._interval_sec):
            try:
                self._renew()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._consecutive_failures += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    if self._consecutive_failures >= self._max_failures:
                        self._terminal_error = RuntimeError(
                            "ClusterBridge lease keepalive failed "
                            f"{self._consecutive_failures} consecutive times: "
                            f"{self._last_error}"
                        )
                self._notify()
                if self._terminal_error is not None:
                    return
            else:
                with self._lock:
                    self._renew_count += 1
                    self._consecutive_failures = 0
                    self._last_success_monotonic = time.monotonic()
                    self._last_error = None
                self._notify()
        self._notify()

    def _notify(self) -> None:
        if self._on_update is not None:
            self._on_update(self.snapshot())
