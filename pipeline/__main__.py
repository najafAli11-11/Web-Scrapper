"""CLI for the pipeline (strip / chunk / ingest).

  python -m pipeline <file> [--url URL] [--db PATH]       strip (default)
  python -m pipeline chunk --result <extraction.json> [--config PATH] [--db PATH]
      load a saved ExtractionResult -> chunk -> embed -> store in Chroma
  python -m pipeline ingest --url <url> [--mode batch|single] [--config PATH] [--db PATH]
      fetch -> strip -> extract -> validate -> chunk -> embed -> store

Per-URL failure handling follows the validator's flag-and-continue pattern
(Rule 6): a chunking failure on one URL logs a `chunk_failed` event and does
not crash the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from agents.config_loader import load_agent_config
from agents.extractor import extract_content, mime_to_content_type
from agents.llm.client import LiteLLMClient
from agents.validator import validate_result
from fetchers.fetch import fetch_page
from fetchers.logger import FetchLogger
from fetchers.types import FetchOutcome
from pipeline.chunk import chunk_result
from pipeline.config_loader import load_pipeline_config
from pipeline.embed import load_embedder
from pipeline.store import VectorStore, collection_name_for
from pipeline.strip import strip_html
from schemas.extraction import ContentType, ExtractionResult


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _ingest_one(result: ExtractionResult, cfg: dict, embedder, store: VectorStore, logger: FetchLogger):
    """Chunk, embed, and store one validated result. Returns (stored, error)."""
    try:
        chunks = chunk_result(result, max_chunk_chars=cfg["chunk"]["max_chunk_chars"])
    except ValueError as exc:
        logger.log_event(
            event_type="chunk_failed",
            url=result.source_url,
            outcome="failed",
            reason=str(exc),
            details={"stage": "chunk"},
        )
        return 0, str(exc)
    embeddings = embedder.embed([c.content for c in chunks])
    collection = collection_name_for(
        embedder.model_name, embedder.dimension, cfg["store"]["collection_prefix"]
    )
    stored = store.store_chunks(chunks, embeddings, collection_name=collection, logger=logger)
    return stored, None


def _run_strip(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline strip", description="Boilerplate-strip HTML (Spec req. 5)."
    )
    parser.add_argument("file", nargs="?", help="Local HTML file to strip")
    parser.add_argument("--url", help="Fetch a URL first, then strip its HTML")
    parser.add_argument("--db", default=None, help="SQLite log DB path (default: data/logs.db)")
    args = parser.parse_args(argv)
    if not args.file and not args.url:
        parser.error("provide a file or --url")

    with FetchLogger(args.db) as logger:
        if args.url:
            fetched = fetch_page(args.url, logger=logger)
            if fetched.outcome != FetchOutcome.SUCCESS or not fetched.html:
                print(f"fetch outcome={fetched.outcome.value} reason={fetched.reason} -> nothing to strip")
                for row in logger.rows_for_url(args.url):
                    print(json.dumps(row, default=str))
                return
            html, log_url = fetched.html, args.url
        else:
            html = Path(args.file).read_text(encoding="utf-8", errors="replace")
            log_url = str(Path(args.file).resolve())

        result = strip_html(html, url=log_url, logger=logger)

    print(
        f"url={result.url}\n"
        f"outcome={result.outcome.value}\n"
        f"method={result.method}\n"
        f"title={result.title}\n"
        f"chars_before={result.chars_before}\n"
        f"chars_after={result.chars_after}\n"
        f"num_blocks={result.num_blocks}\n"
        f"num_tables={result.num_tables}"
    )
    print("--- blocks ---")
    for i, block in enumerate(result.blocks, 1):
        head = f"[{block.heading}] " if block.heading else ""
        preview = block.text[:200] + ("..." if len(block.text) > 200 else "")
        print(f"{i}. {head}{preview}")
    print("--- log rows (chronological) ---")
    with FetchLogger(args.db) as logger:
        for row in logger.rows_for_url(log_url):
            print(json.dumps(row, default=str))


def _run_chunk(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline chunk",
        description="Chunk + embed + store a saved ExtractionResult (Spec req. 10-13).",
    )
    parser.add_argument("--result", required=True, help="Path to an ExtractionResult JSON file")
    parser.add_argument("--config", default=None, help="Path to config/embeddings.json")
    parser.add_argument("--db", default=None, help="SQLite log DB path (default: data/logs.db)")
    args = parser.parse_args(argv)

    with FetchLogger(args.db) as logger:
        result = ExtractionResult.model_validate(
            json.loads(Path(args.result).read_text(encoding="utf-8"))
        )
        cfg = load_pipeline_config(Path(args.config) if args.config else None)
        embedder = load_embedder(cfg, logger=logger)
        store = VectorStore(cfg["store"]["chroma_path"], collection_prefix=cfg["store"]["collection_prefix"])
        stored, err = _ingest_one(result, cfg, embedder, store, logger)

    print(f"stored_chunks={stored} collection={collection_name_for(embedder.model_name, embedder.dimension, cfg['store']['collection_prefix'])}")
    if err:
        print(f"chunk_failed reason={err}")
    print("--- log rows ---")
    with FetchLogger(args.db) as logger:
        for row in logger.rows_for_url(result.source_url):
            print(json.dumps(row, default=str))
    if err:
        sys.exit(1)


def _run_ingest(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline ingest",
        description="Fetch -> strip -> extract -> validate -> chunk -> store (Spec req. 10-13).",
    )
    parser.add_argument("--url", required=True, help="Source URL")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--config", default=None, help="Path to config/embeddings.json")
    parser.add_argument("--db", default=None, help="SQLite log DB path (default: data/logs.db)")
    args = parser.parse_args(argv)

    with FetchLogger(args.db) as logger:
        fetched = fetch_page(args.url, logger=logger)
        if fetched.outcome != FetchOutcome.SUCCESS:
            print(f"fetch outcome={fetched.outcome.value} reason={fetched.reason} -> no ingestion")
            for row in logger.rows_for_url(args.url):
                print(json.dumps(row, default=str))
            return

        ctype = mime_to_content_type(fetched.content_type)
        if ctype == ContentType.HTML and fetched.html:
            stripped = strip_html(fetched.html, url=args.url, logger=logger)
            content: object = stripped.text
            page_title = stripped.title
        elif fetched.raw is not None:
            content = fetched.raw
            page_title = None
        else:
            print("fetch succeeded but no content to ingest")
            return

        agent_cfg = load_agent_config()
        client = LiteLLMClient(agent_cfg)
        result = extract_content(
            content,
            content_type=ctype,
            source_url=args.url,
            page_title=page_title,
            mode=args.mode,
            client=client,
            agent_cfg=agent_cfg,
            logger=logger,
        )
        validation, final_result = validate_result(
            result,
            content=content,
            mode=args.mode,
            client=client,
            agent_cfg=agent_cfg,
            logger=logger,
        )
        if not validation.is_valid:
            print(f"validation failed -> not ingested (flagged, see validation_flagged event)")
            print(json.dumps(validation.model_dump(mode="json"), indent=2, ensure_ascii=False))
            for row in logger.rows_for_url(args.url):
                print(json.dumps(row, default=str))
            return

        cfg = load_pipeline_config(Path(args.config) if args.config else None)
        embedder = load_embedder(cfg, logger=logger)
        store = VectorStore(cfg["store"]["chroma_path"], collection_prefix=cfg["store"]["collection_prefix"])
        stored, err = _ingest_one(final_result, cfg, embedder, store, logger)

    print(
        f"url={args.url} is_valid=True stored_chunks={stored} "
        f"sections={len(final_result.sections)} confidence={final_result.confidence}"
    )
    if err:
        print(f"chunk_failed reason={err}")
    print("--- log rows ---")
    with FetchLogger(args.db) as logger:
        for row in logger.rows_for_url(args.url):
            print(json.dumps(row, default=str))
    if err:
        sys.exit(1)


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_stdout()
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("chunk", "ingest"):
        cmd = args.pop(0)
        if cmd == "chunk":
            _run_chunk(args)
        else:
            _run_ingest(args)
    else:
        _run_strip(args)


if __name__ == "__main__":
    main()
