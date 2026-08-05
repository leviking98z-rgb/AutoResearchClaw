from __future__ import annotations

import signal
from types import SimpleNamespace

from researchclaw.autoresearch_v2 import cli


def test_signal_handlers_request_cooperative_stop(monkeypatch) -> None:
    installed = {}
    restored = {}
    stops: list[str] = []
    controller = SimpleNamespace(
        request_stop=lambda *, reason: stops.append(reason)
    )

    monkeypatch.setattr(
        cli.signal,
        "getsignal",
        lambda signum: f"old-{signum.name}",
    )
    monkeypatch.setattr(
        cli.signal,
        "signal",
        lambda signum, handler: installed.setdefault(signum, handler)
        if callable(handler)
        else restored.setdefault(signum, handler),
    )

    previous = cli._install_stop_handlers(controller)
    installed[signal.SIGTERM](signal.SIGTERM, None)
    installed[signal.SIGINT](signal.SIGINT, None)

    assert stops == ["SIGTERM", "SIGINT"]
    assert previous == {
        signal.SIGTERM: "old-SIGTERM",
        signal.SIGINT: "old-SIGINT",
    }
