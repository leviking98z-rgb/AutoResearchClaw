from __future__ import annotations

import signal
from types import SimpleNamespace

from researchclaw.autoresearch_v2 import cli


def test_signal_handlers_request_cooperative_stop(monkeypatch) -> None:
    calls = []
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
        lambda signum, handler: calls.append((signum, handler)),
    )

    previous = cli._install_stop_handlers(controller)
    installed = dict(calls)
    installed[signal.SIGTERM](signal.SIGTERM, None)
    installed[signal.SIGINT](signal.SIGINT, None)
    cli._restore_signal_handlers(previous)

    assert stops == ["SIGTERM", "SIGINT"]
    assert previous == {
        signal.SIGTERM: "old-SIGTERM",
        signal.SIGINT: "old-SIGINT",
    }
    assert calls[-2:] == [
        (signal.SIGTERM, "old-SIGTERM"),
        (signal.SIGINT, "old-SIGINT"),
    ]


def test_service_stop_uses_hard_exit_after_durable_cleanup(
    monkeypatch,
    tmp_path,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
autoresearch_v2:
  enabled: true
  state_dir: {tmp_path / "state"}
  population:
    reservoir_low_watermark: 1
    reservoir_target: 1
    generation_batch_size: 1
    active_idea_target: 1
    max_active_ideas: 1
    max_same_family: 1
""",
        encoding="utf-8",
    )

    class _Controller:
        stop_reason = "SIGTERM"

        def run(self, **kwargs):
            del kwargs
            return 1

        def request_stop(self, *, reason):
            del reason

    monkeypatch.setattr(
        cli,
        "build_production_controller",
        lambda config: _Controller(),
    )
    monkeypatch.setattr(
        cli,
        "_install_stop_handlers",
        lambda controller: {signal.SIGTERM: signal.SIG_DFL},
    )
    monkeypatch.setattr(cli, "_restore_signal_handlers", lambda value: None)
    exits = []
    monkeypatch.setattr(cli.os, "_exit", lambda code: exits.append(code))

    assert cli.main(["--config", str(config), "start"]) == 0
    assert exits == [0]
