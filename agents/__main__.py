"""CLI for Milestone 4/5 extractor + validator verification.

  python -m agents --url <url> [--db PATH] [--mode batch|single] [--validate]
      fetch -> (strip if HTML) -> extract, and optionally validate
      (with the one-repair-then-flag policy); print result(s) + log rows

  python -m agents --file <path> --content-type html|text|pdf|unknown --url <source> [--db PATH]
      extract from a local file of the declared content type (--url declares
      the provenance source, which must be an http(s) URL)
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
from pipeline.strip import strip_html
from schemas.extraction import ContentType


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description="Schema-constrained extraction (Spec req. 6-8).")
    parser.add_argument("--url", help="Source URL; fetched and extracted when --file is absent, otherwise declares the provenance source for --file")
    parser.add_argument("--file", help="Local file to extract from (requires --url to declare the http(s) source)")
    parser.add_argument("--content-type", choices=["html", "text", "pdf", "unknown"],
                        help="Declared content type for --file (default: inferred as text)")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch")
    parser.add_argument("--validate", action="store_true",
                        help="Run the validator (one repair, then flag) on the extraction result")
    parser.add_argument("--db", default=None, help="SQLite log DB path (default: data/logs.db)")
    args = parser.parse_args(argv)
    if not args.url and not args.file:
        parser.error("provide --url, or --file --url <source>")

    with FetchLogger(args.db) as logger:
        page_title = None
        if args.url and not args.file:
            fetched = fetch_page(args.url, logger=logger)
            if fetched.outcome != FetchOutcome.SUCCESS:
                print(f"fetch outcome={fetched.outcome.value} reason={fetched.reason} -> no extraction")
                for row in logger.rows_for_url(args.url):
                    print(json.dumps(row, default=str))
                return
            ctype = mime_to_content_type(fetched.content_type)
            if ctype == ContentType.HTML and fetched.html:
                stripped = strip_html(fetched.html, url=args.url, logger=logger)
                content: object = stripped.text
                page_title = stripped.title
                log_url = args.url
            elif fetched.raw is not None:
                content = fetched.raw
                log_url = args.url
            else:
                print("fetch succeeded but no content to extract")
                return
        else:
            if not args.url:
                parser.error("--file requires --url to declare the source URL (provenance requires an http(s) URL)")
            path = Path(args.file)
            ctype = ContentType(args.content_type) if args.content_type else ContentType.TEXT
            if ctype == ContentType.PDF:
                content = path.read_bytes()
            else:
                content = path.read_text(encoding="utf-8", errors="replace")
            log_url = args.url

        agent_cfg = load_agent_config()
        client = LiteLLMClient(agent_cfg)
        result = extract_content(
            content,
            content_type=ctype,
            source_url=log_url,
            page_title=page_title,
            mode=args.mode,
            client=client,
            agent_cfg=agent_cfg,
            logger=logger,
        )

        if args.validate:
            validation = validate_result(
                result,
                content=content,
                mode=args.mode,
                client=client,
                agent_cfg=agent_cfg,
                logger=logger,
            )
    print("--- extraction result ---")
    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))
    if args.validate:
        print("--- validation result ---")
        print(json.dumps(validation[0].model_dump(mode="json"), indent=2, ensure_ascii=False))
    print("--- log rows (chronological) ---")
    with FetchLogger(args.db) as logger:
        for row in logger.rows_for_url(log_url):
            print(json.dumps(row, default=str))


if __name__ == "__main__":
    main()
