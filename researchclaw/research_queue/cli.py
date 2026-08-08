"""CLI for the Continuous Research Queue prototype."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import ResearchQueueConfig
from .controller import ResearchQueueController
from .execution import build_run_backend
from .store import ResearchQueueStore
from .workers import StaticIdeaProducer, build_workers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-queue",
        description="Lightweight continuous multi-Idea research prototype",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config.research-queue.example.yaml",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start")
    start.add_argument("--max-seconds", type=float)
    start.add_argument(
        "--until-idle",
        action="store_true",
        help="Stop after a finite static/simulation workload drains.",
    )
    start.add_argument(
        "--ideas-json",
        type=Path,
        help="Optional finite JSON array of Idea proposals.",
    )
    commands.add_parser("status")
    commands.add_parser("ideas")
    commands.add_parser("runs")
    return parser


def build_controller(
    config: ResearchQueueConfig,
    *,
    ideas_json: Path | None = None,
) -> ResearchQueueController:
    store = ResearchQueueStore(
        config.root,
        artifact_root=config.artifact_root,
    )
    producer, preparer, reviewer = build_workers(config)
    if ideas_json is not None:
        value = json.loads(ideas_json.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise TypeError("--ideas-json must contain one JSON array")
        producer = StaticIdeaProducer(value, cycle=False)
    return ResearchQueueController(
        config=config,
        store=store,
        producer=producer,
        preparer=preparer,
        reviewer=reviewer,
        run_backend=build_run_backend(config),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ResearchQueueConfig.from_file(args.config)
    store = ResearchQueueStore(
        config.root,
        artifact_root=config.artifact_root,
    )
    store.initialize()
    if args.command == "status":
        print(json.dumps(store.snapshot(), ensure_ascii=False, indent=2))
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
    if args.command == "runs":
        print(
            json.dumps(
                [run.to_dict() for run in store.list_runs()],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not config.enabled:
        raise SystemExit("research_queue.enabled must be true")
    controller = build_controller(
        config,
        ideas_json=args.ideas_json,
    )
    snapshot = asyncio.run(
        controller.run(
            max_seconds=args.max_seconds,
            until_idle=args.until_idle,
        )
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
