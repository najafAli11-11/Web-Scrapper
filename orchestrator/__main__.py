"""CLI for the orchestrator (M7).

  python -m orchestrator run <urls.txt> [--db queue.db] [--log events.db]
      [--config orchestrator.json] [--reset] [--mode batch|single]

Single-pass batch run over the URL frontier:
- default (no --reset): resume. pending rows are processed; done/blocked/
  flagged/failed rows are never reprocessed; retrying rows are processed once
  their backoff deadline has passed.
- --reset: clear the queue state table first (fresh run / manual re-ingestion
  of every URL).
- --db: queue state table (default data/queue.db). --log: shared events DB
  (default data/logs.db) — the same table M6's fetch/validation events use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from agents.config_loader import load_agent_config
from agents.llm.client import LiteLLMClient
from fetchers.config_loader import load_fetch_config, load_obstacle_config
from fetchers.logger import FetchLogger
from orchestrator.config_loader import load_orchestrator_config
from orchestrator.queue import UrlQueue
from orchestrator.run_batch import run_batch
from pipeline.config_loader import load_pipeline_config
from pipeline.embed import load_embedder
from pipeline.store import VectorStore


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _read_urls(path: Path) -> list[str]:
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def _run_import(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m orchestrator run",
        description="Run one single-pass batch over the URL frontier (Spec req. 6).",
    )
    parser.add_argument("urls_file", help="Text file with one URL per line; '#' comments are skipped")
    parser.add_argument("--db", default=None, help="Queue state DB (default: data/queue.db)")
    parser.add_argument("--log", default=None, help="Shared events DB (default: data/logs.db)")
    parser.add_argument("--config", default=None, help="Path to config/orchestrator.json")
    parser.add_argument("--reset", action="store_true",
                        help="Clear the queue state first (fresh run / manual re-ingestion)")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    args = parser.parse_args(argv)

    urls = _read_urls(Path(args.urls_file))
    if not urls:
        print("no URLs found in file")
        sys.exit(1)

    retry_cfg = load_orchestrator_config(Path(args.config) if args.config else None)
    agent_cfg = load_agent_config()
    client = LiteLLMClient(agent_cfg)
    pipeline_cfg = load_pipeline_config()
    obstacle_cfg = load_obstacle_config()
    fetch_cfg = load_fetch_config()

    with UrlQueue(args.db) as queue, FetchLogger(args.log) as logger:
        if args.reset:
            queue.reset()
        embedder = load_embedder(pipeline_cfg, logger=logger)
        store = VectorStore(
            pipeline_cfg["store"]["chroma_path"],
            collection_prefix=pipeline_cfg["store"]["collection_prefix"],
        )
        summary = run_batch(
            urls,
            queue=queue,
            logger=logger,
            agent_cfg=agent_cfg,
            client=client,
            embedder=embedder,
            store=store,
            pipeline_cfg=pipeline_cfg,
            retry_cfg=retry_cfg,
            mode=args.mode,
            obstacle_cfg=obstacle_cfg,
            fetch_cfg=fetch_cfg,
        )

    print(f"processed={summary['total']}")
    for state, n in sorted(summary["by_state"].items()):
        print(f"  {state}: {n}")
    print("--- queue states ---")
    with UrlQueue(args.db) as q:
        for row in q.states():
            reason = f" | {row['reason']}" if row["reason"] else ""
            print(f"  {row['state']:9s} attempts={row['attempts']} {row['url']}{reason}")
    print("--- recent events ---")
    with FetchLogger(args.log) as logger:
        for e in logger.recent_events(limit=15):
            print(json.dumps(e, default=str))


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_stdout()
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "run":
        _run_import(args[1:])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
