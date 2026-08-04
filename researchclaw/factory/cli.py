"""Command-line control plane for the continuous Research Factory."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from .config import FactoryConfig
from .generator import LLMCandidateGenerator, StaticCandidateGenerator
from .gpu_broker import GPUBroker
from .orchestrator import FactoryOrchestrator
from .scheduler import FactoryScheduler
from .store import FactoryStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchclaw-factory",
        description="Continuous multi-Idea autonomous research factory",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.factory.yaml",
        help="Factory YAML config (default: config.factory.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="Run the persistent Factory loop")
    start.add_argument("--once", action="store_true")
    start.add_argument("--max-ticks", type=int)
    start.add_argument(
        "--simulation-candidates",
        type=Path,
        help="JSON array of deterministic candidates; avoids all LLM/GPU calls",
    )

    sub.add_parser("status", help="Print durable Factory status")
    listing = sub.add_parser("list", help="List Ideas")
    listing.add_argument("--status", action="append", default=[])
    pause = sub.add_parser("pause", help="Cooperatively pause new admissions")
    pause.add_argument("reason", nargs="*", default=[])
    sub.add_parser("resume", help="Clear a cooperative pause")
    stop = sub.add_parser("stop", help="Request graceful Factory shutdown")
    stop.add_argument("reason", nargs="*", default=[])
    return parser


def _load(config_path: Path) -> tuple[FactoryConfig, FactoryStore]:
    config = FactoryConfig.from_file(config_path)
    store = FactoryStore(config.root, factory_id=config.factory_id)
    store.initialize()
    return config, store


def _generator(
    config: FactoryConfig,
    *,
    config_path: Path,
    candidates_path: Path | None,
) -> Any:
    if candidates_path:
        value = json.loads(candidates_path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise TypeError("simulation candidate file must contain a JSON array")
        return StaticCandidateGenerator(value)
    from researchclaw.config import RCConfig

    try:
        from researchclaw.llm.roles import create_role_llm_client
    except ImportError as exc:
        raise RuntimeError(
            "LLM candidate generation requires role-aware LLM routing; "
            "use --simulation-candidates or install production role modules"
        ) from exc
    rc_config_path = Path(config.worker.pipeline_config or config_path)
    rc_config = RCConfig.load(rc_config_path, check_paths=False)
    llm = create_role_llm_client(
        rc_config,
        "topic_selector",
        run_dir=config.root / "generator",
    )
    brief = config.topic_brief or rc_config.research.campaign_brief
    return LLMCandidateGenerator(llm=llm, brief=brief)


def _broker(
    config: FactoryConfig,
    store: FactoryStore,
) -> tuple[GPUBroker | None, Any | None]:
    if not config.gpu.enabled:
        return None, None
    from researchclaw.experiment.clusterbridge_pool import ClusterBridgePool

    from .pool_config import PoolConfigSummary

    pool_summary = PoolConfigSummary.from_file(config.gpu.pool_config)
    pool = ClusterBridgePool.from_file(
        pool_summary.config_path,
        restore_state=config.gpu.restore_state,
    )
    if config.gpu.claim_on_start and not pool.claimed:
        pool.claim(start_keepalive=True)
    if config.gpu.prepare_on_start and not pool.prepared:
        pool.prepare()
    scheduler = FactoryScheduler(
        config,
        total_gpus=pool_summary.expected_total_gpus,
    )
    scheduler.max_gpus_per_node = pool_summary.max_gpus_per_node
    return GPUBroker(pool=pool, store=store, scheduler=scheduler), pool


def _signal_existing(store: FactoryStore) -> None:
    try:
        pid = int(store.load_state().get("pid") or 0)
    except (TypeError, ValueError):
        return
    if pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config, store = _load(config_path)

    if args.command == "status":
        print(json.dumps(store.snapshot(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "list":
        requested = {str(item).casefold() for item in args.status}
        ideas = [
            idea.to_dict()
            for idea in store.list_ideas()
            if not requested or idea.status.value in requested
        ]
        print(json.dumps(ideas, ensure_ascii=False, indent=2))
        return 0
    if args.command == "pause":
        reason = " ".join(args.reason) or "operator requested pause"
        print(store.set_control("pause", reason))
        return 0
    if args.command == "resume":
        store.clear_control("pause")
        print(f"resumed: {store.root}")
        return 0
    if args.command == "stop":
        reason = " ".join(args.reason) or "operator requested stop"
        print(store.set_control("stop", reason))
        _signal_existing(store)
        return 0

    if not config.enabled:
        print(
            "Factory mode is disabled. Set factory.enabled: true before start.",
            file=sys.stderr,
        )
        return 2
    candidates_path = getattr(args, "simulation_candidates", None)
    generator = _generator(
        config,
        config_path=config_path,
        candidates_path=candidates_path,
    )
    broker, pool = _broker(config, store)
    scheduler = (
        broker.scheduler if broker is not None else FactoryScheduler(config)
    )
    orchestrator = FactoryOrchestrator(
        config=config,
        store=store,
        generator=generator,
        scheduler=scheduler,
        gpu_broker=broker,
    )
    try:
        orchestrator.run(
            once=bool(args.once),
            max_ticks=args.max_ticks,
        )
    finally:
        if (
            pool is not None
            and config.gpu.release_on_shutdown
            and pool.claimed
        ):
            pool.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
