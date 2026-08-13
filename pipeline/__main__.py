"""CLI for Milestone 3 stripping verification.

  python -m pipeline <html_file> [--db PATH]     strip a local HTML file
  python -m pipeline --url <url> [--db PATH]     fetch then strip a URL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from fetchers.fetch import fetch_page
from fetchers.logger import FetchLogger
from fetchers.types import FetchOutcome
from pipeline.strip import StripOutcome, strip_html


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description="Boilerplate-strip HTML (Spec req. 5).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("file", nargs="?", help="Local HTML file to strip")
    src.add_argument("--url", help="Fetch a URL first, then strip its HTML")
    parser.add_argument("--db", default=None, help="SQLite log DB path (default: data/logs.db)")
    args = parser.parse_args(argv)

    with FetchLogger(args.db) as logger:
        if args.url:
            fetched = fetch_page(args.url, logger=logger)
            if fetched.outcome != FetchOutcome.SUCCESS or not fetched.html:
                print(
                    f"fetch outcome={fetched.outcome.value} reason={fetched.reason} "
                    "-> nothing to strip"
                )
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


if __name__ == "__main__":
    main()
