"""CLI for the independent AutoResearch v2 controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import V2Config
from .controller import V2Controller
from .ideas import StaticIdeaGenerator
from .jobs import SimulatedJobExecutor
from .models import JobKind
from .runtime import build_production_controller
from .store import V2Store


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoresearch-v2",
        description="Transactional continuous multi-Idea AutoResearch",
    )
    parser.add_argument("-c", "--config", default="config.autoresearch-v2.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--max-ticks", type=int)
    start.add_argument("--simulation-candidates", type=Path)
    commands.add_parser("status")
    commands.add_parser("ideas")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = V2Config.from_file(args.config)
    store = V2Store(config.root)
    store.initialize()
    if args.command == "status":
        controller = V2Controller(
            config=config,
            store=store,
            generator=StaticIdeaGenerator([]),
        )
        snapshot = controller.snapshot()
        writer = store.writer_status()
        snapshot["controller_process"] = writer
        if writer["state"] != "live":
            snapshot["status"] = "stopped"
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))
        controller._pool.shutdown(wait=True)
        return 0
    if args.command == "ideas":
        print(
            json.dumps(
                [idea.to_dict() for idea in store.list_ideas()],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not config.enabled:
        raise SystemExit("autoresearch_v2.enabled must be true")
    if args.simulation_candidates:
        value = json.loads(
            args.simulation_candidates.read_text(encoding="utf-8")
        )
        if not isinstance(value, list):
            raise TypeError("simulation candidates must be a JSON array")
        controller = V2Controller(
            config=config,
            store=store,
            generator=StaticIdeaGenerator(value),
            executors={
                kind: SimulatedJobExecutor()
                for kind in JobKind
            },
        )
    else:
        controller = build_production_controller(config)
    controller.run(max_ticks=args.max_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
