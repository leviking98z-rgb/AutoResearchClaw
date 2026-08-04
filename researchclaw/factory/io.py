"""Small durable filesystem primitives owned by Factory mode."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "access_token",
    "auth_token",
    "bearer_token",
    "refresh_token",
)
_JSONL_LOCKS_GUARD = threading.Lock()
_JSONL_LOCKS: dict[str, threading.Lock] = {}


def utc_now_ms() -> str:
    """Return a millisecond-resolution UTC timestamp for event journals."""

    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _jsonl_lock(path: Path) -> threading.Lock:
    key = str(path.expanduser().absolute())
    with _JSONL_LOCKS_GUARD:
        return _JSONL_LOCKS.setdefault(key, threading.Lock())


def redact_event_value(value: Any, *, key: str = "") -> Any:
    """Remove obvious credentials from structured operational events.

    Event payloads are intentionally metadata-only, but failures and provider
    responses sometimes contain nested request dictionaries.  Redact by key at
    every depth before serialization so a retrospective journal is safe to
    retain and share with maintainers.
    """

    normalized = str(key).casefold()
    if any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
        return _REDACTED
    if isinstance(value, Mapping):
        return {
            str(child_key): redact_event_value(
                child_value,
                key=str(child_key),
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact_event_value(item) for item in value]
    return value


def append_jsonl(
    path: Path,
    value: Mapping[str, Any],
    *,
    durable: bool = False,
    redact: bool = True,
) -> dict[str, Any]:
    """Append one mapping as JSONL with a process-wide per-path lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(
        redact_event_value(value) if redact else value
    )
    line = (
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    with _jsonl_lock(path), path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        if durable:
            os.fsync(stream.fileno())
    return record


def atomic_write_json(
    path: Path,
    value: Any,
    *,
    durable: bool = False,
) -> None:
    """Atomically replace JSON, with optional crash-durable fsync.

    Factory state is rewritten frequently and commonly lives on shared CephFS,
    where an fsync for every derived snapshot can dominate the scheduler tick.
    Atomic rename is the default. Callers that guard correctness boundaries
    (writer ownership, external publication, etc.) can request full durability.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(temp, path)
        if not durable:
            return
        try:
            directory_fd = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def tail_text_lines(
    path: Path,
    *,
    limit: int,
    max_bytes: int = 4 * 1024 * 1024,
) -> list[str]:
    """Read a bounded tail without loading an unbounded journal into memory."""

    count = max(1, int(limit))
    byte_limit = max(4096, int(max_bytes))
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            start = max(0, size - byte_limit)
            stream.seek(start)
            data = stream.read()
    except (FileNotFoundError, OSError):
        return []
    if start:
        separator = data.find(b"\n")
        if separator < 0:
            return []
        data = data[separator + 1 :]
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[-count:]


def tail_jsonl(
    path: Path,
    *,
    limit: int = 200,
    max_bytes: int = 4 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Parse a bounded JSONL tail and ignore malformed/partial rows."""

    rows: list[dict[str, Any]] = []
    requested = max(1, int(limit))
    # Read a small surplus because the physical tail may end with a partial or
    # malformed row from a concurrently appending writer.
    for line in tail_text_lines(
        path,
        limit=requested + 16,
        max_bytes=max_bytes,
    ):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows[-requested:]


def iter_jsonl(
    path: Path,
    *,
    max_lines: int | None = None,
) -> list[dict[str, Any]]:
    """Read JSONL fail-soft, optionally retaining only the latest N rows."""

    retained: deque[dict[str, Any]] | list[dict[str, Any]]
    retained = deque(maxlen=max_lines) if max_lines else []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return []
    with handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, Mapping):
                continue
            retained.append(dict(value))
    return list(retained)
