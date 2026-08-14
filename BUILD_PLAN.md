# Phase 4 — Build Plan

Per AGENTS.md: this is a multi-file, architectural change, so it gets the
full loop — plan → build in small checkable steps → verify, with a commit
after each step. Each milestone below is scoped to be independently
testable before moving to the next. Feed these to opencode one at a time,
in order — don't ask it to build the whole system in one shot.

Each milestone lists: what to build, what "done" looks like (tie back to
SPEC.md acceptance criteria where relevant), and a suggested commit message.

## Milestone 0 — Scaffold (done)
Repo structure, AGENTS.md, SPEC.md, config/obstacles.json,
schemas/extraction.py, requirements.txt already in place.
`git init`, commit as the baseline.

## Milestone 1 — Schemas & config only
- Review/finalize `schemas/extraction.py` (already drafted — check it
  matches how you want to represent sections/content).
- Review `config/obstacles.json`.
- No logic yet. This milestone exists so every later step has a stable
  contract to build against.
- **Done when**: schemas import cleanly, no logic depends on them yet.
- Commit: `schemas: define extraction contract and obstacle config`

## Milestone 2 — Fetcher (static + headless, no LLM involved)
- `fetchers/static_fetch.py` — plain HTTP fetch, returns raw HTML or None.
- `fetchers/browser_fetch.py` — Playwright-based fetch with:
  - obstacle detection reading `config/obstacles.json` (popup/cookie
    dismiss, redirect-follow, rate-limit backoff)
  - CAPTCHA detection → returns a `blocked` status, never attempts to solve
  - logging of every attempt (url, timestamp, outcome, reason)
- Decision logic: try static first, fall back to browser fetch if content
  looks empty/JS-shell (per Spec req. 1).
- **Done when**: given a small manual test list (one static site, one
  JS-heavy site, one site you expect to block), each returns the correct
  outcome and every attempt is logged. This directly tests Spec acceptance
  criterion 1 (partially) and criterion 2 (CAPTCHA → blocked, no crash).
- Commit: `fetchers: static + headless fetch with obstacle handling`

## Milestone 3 — Boilerplate stripping
- `pipeline/strip.py` — wraps Trafilatura, takes raw HTML in, returns
  cleaned text/structure out.
- Handle the "malformed HTML" edge case from SPEC.md — should not crash on
  broken markup.
- **Done when**: raw HTML from Milestone 2's test sites produces
  meaningfully cleaner output (no nav/ads/scripts) than the raw source.
- Commit: `pipeline: boilerplate stripping via trafilatura`

## Milestone 4 — Extractor agent (LLM, schema-constrained)
- `agents/extractor.py` — takes stripped content, calls the LLM with the
  `ExtractionResult` schema (tool-use/function-calling), returns structured
  output. Support both batch and single-shot call signatures (Spec req. 7).
- Route non-HTML content types (e.g. PDF) here too, or flag if unsupported
  (Spec req. 8).
- **Done when**: given stripped content from Milestone 3, output validates
  against `ExtractionResult` and looks sensible for at least 3 very
  different page types (article, listing, PDF).
- Commit: `agents: schema-constrained extractor`

## Milestone 5 — Validator agent
- `agents/validator.py` — validates `ExtractionResult` against the schema
  and basic sanity rules (non-empty content, reasonable confidence).
  Implements the retry-once-then-flag policy (Spec req. 9).
- **Done when**: a deliberately broken/empty extraction gets one repair
  attempt, then is flagged (not silently dropped, not silently stored).
  Tests SPEC.md acceptance criterion 6.
- Commit: `agents: validator with retry-then-flag policy`

## Milestone 6 — Chunker + embedder + vector store
- `pipeline/chunk.py` — semantic chunking from `ExtractionResult.sections`
  (Spec req. 10).
- `pipeline/embed.py` — BGE-M3 (fallback MiniLM), wraps embedding calls.
- `pipeline/store.py` — Chroma write/query, with dedup-by-source-url on
  write (Spec req. 16 — replace, don't duplicate).
- **Done when**: a full extraction result becomes chunks-with-metadata in
  Chroma, re-ingesting the same URL replaces rather than duplicates.
  Tests SPEC.md acceptance criterion 5.
- Commit: `pipeline: semantic chunking, local embeddings, chroma storage`

## Milestone 7 — Orchestrator (deterministic queue/state machine)
- `orchestrator/queue.py` — URL frontier: seen/pending/failed/retrying
  state, resumable (Rule 6 — replayability).
- `orchestrator/run_batch.py` — wires fetcher → strip → extract → validate
  → chunk → store for a list of URLs. Plain control flow, no LLM here.
- **Done when**: given a batch of URLs including one that will fail, a run
  can be killed mid-way and resumed without re-processing completed URLs.
- Commit: `orchestrator: resumable batch pipeline`
- **Re-verification note**: a `Stop-Process -Force` on the launcher (`cmd` /
  `Start-Process` shim) does NOT kill the batch — the `python` worker keeps
  ingesting as an orphan, so a "kill" may look like it didn't work while the
  queue keeps moving. Kill the actual worker process (match `CommandLine`
  via `Get-CimInstance Win32_Process`) to leave a genuine `processing` row
  behind. This was the cause of the first kill attempt's false negative.

## Milestone 8 — Single-shot / live-query path
- `orchestrator/live_query.py` — implements the hybrid behavior (Spec req.
  15): check corpus first, fall back to single-shot scrape if absent.
- **Done when**: querying a URL already in the corpus returns instantly
  from storage; querying a new URL triggers the single-shot pipeline and
  returns an answer. Tests SPEC.md acceptance criterion 4.
- Commit: `orchestrator: hybrid live-query path`
- **Re-verification note**: any CLI parity/live check that writes to the
  store (e.g. `pipeline chunk --result <json>`) must target a SCRATCH
  chroma/events path (a temp dir), never `data/chroma` — a stray test URL
  pollutes the real corpus (this bit the M7 parity check).

## Milestone 9 — Minimal local web UI
- `ui/app.py` — a single lightweight app (Streamlit recommended for
  speed/simplicity at this scale; FastAPI+HTML if you prefer more control)
  with three views:
  - **Chat**: text box → queries corpus (Milestone 6 storage + LLM answer
    generation) → displays answer with source URL + scrape timestamp per
    citation.
  - **Ingestion**: paste/submit URLs → triggers Milestone 7's batch
    orchestrator → shows live per-URL status (pending/done/blocked/flagged),
    reading from the orchestrator's state table.
  - **Logs**: chronological feed of pipeline events — fetch attempts (from
    Milestone 2's logging), obstacle detections, validation failures/flags
    (from Milestone 5). Filterable by URL/status if useful, but a simple
    reverse-chronological list is enough for v1.
- Before this milestone, make sure Milestones 2 and 5 are actually writing
  structured logs somewhere queryable (a simple SQLite table or JSONL file
  is enough — don't build a logging service). The UI just reads it.
- No auth, no multi-user support, no separate frontend/backend split —
  single local process, per confirmed small-scale scope.
- **Done when**: you can submit a URL and watch it move through statuses,
  ask a question in chat and get a sourced answer, and see both a blocked
  fetch and a flagged validation event appear in the logs view — entirely
  from the UI, no manual script invocation or log-file inspection needed.
  Tests SPEC.md acceptance criteria on chatbot sourcing, ingestion status,
  and logs visibility.
- Commit: `ui: minimal local web app for chat, ingestion, and logs`

## Milestone 10 — End-to-end verification pass
- Run through every SPEC.md acceptance criterion explicitly, one at a time.
- Write/finalize `tests/` covering each.
- Commit: `tests: acceptance criteria coverage`

---

### How to drive this with opencode
- Point opencode at this repo — `AGENTS.md` is picked up automatically as
  persistent context.
- Work one milestone at a time: paste (or reference) the relevant
  milestone section as the task, let opencode plan before it writes code
  for anything beyond a one-line change (per AGENTS.md Phase 4 guidance),
  review the plan, then build.
- Commit after each milestone passes its "done when" check — don't batch
  multiple milestones into one commit, since that defeats the point of
  small checkable steps.
