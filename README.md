# Web-Scrapper

A multi-agent **web scraper → RAG pipeline** that extracts meaningful content
from *any* website — regardless of HTML complexity, JS rendering, or DOM
structure — and feeds it into a self-hosted, local-embedding vector store that
powers a chatbot, a search/retrieval interface, and a live-query agent tool.
No per-site selectors, no per-domain schemas, no paid embedding APIs.

Built for small/personal scale (dozens of pages). Scoped and verified against
[`SPEC.md`](SPEC.md); governed by the project constitution in
[`AGENTS.md`](AGENTS.md).

---

## Highlights

- **Generic extraction by design** — no hardcoded selectors or per-domain
  schemas. Page classification, structured extraction, and validation are done
  by LLM agents constrained by explicit Pydantic schemas (tool-use /
  function-calling), never free-form text parsing.
- **Headless browser as a baseline** — static fetch is attempted first;
  JS-shell or near-empty responses automatically fall back to Playwright
  rendering, so SPAs and dynamic pages work without custom logic.
- **Schema-constrained, validated output** — every extraction is validated
  before storage. Failures get one bounded repair attempt, then are **flagged
  for review — never silently dropped, never silently stored** (Rule 5).
- **CAPTCHA and access-control gates are terminal "blocked" states** —
  detected, logged with a reason, and skipped. The project adapts to site
  messiness (popups, banners, redirects, rate limits) but never tries to
  defeat a site's deliberate access control.
- **Full provenance, end to end** — every chunk and every answer carries
  `source_url`, scrape timestamp, page title, and section heading, so RAG
  answers are verifiable.
- **Semantic chunking** — chunks are split along the page's real structure
  (headings/sections), not fixed token windows, and not by a separate LLM pass.
- **Free, local embeddings** — BGE-M3 by default (multilingual, dense+sparse),
  with automatic fallback to `all-MiniLM-L6-v2` on constrained hardware.
- **Resumable by construction** — the orchestrator is a deterministic URL
  frontier (seen / pending / retrying / done / blocked / flagged / failed)
  persisted in SQLite. Kill a run at any point; restarting resumes, never
  re-scrapes from zero.
- **Hybrid live query** — checks the corpus first; on a miss it runs a
  single-shot scrape (fetch → strip → extract → validate → answer) without
  requiring a batch re-ingestion run, and never writes back to the corpus.
- **Provider-agnostic LLM access** — every agent calls a thin client
  abstraction (backed by [litellm](https://github.com/BerriAI/litellm)); any
  supported provider (NVIDIA NIM, OpenAI, Anthropic, Groq, Ollama, ...) is a
  config change, not a code change.

---

## Architecture

Deterministic control flow, non-deterministic extraction:

```
                     +------------------- ORCHESTRATOR (deterministic) -------------------+
                     |  queue / frontier   retry/backoff   rate limit   batch runner      |
                     +-------------------------------------------------------------------+
                                        |
                                        v
   urls ──► fetch ──► strip ──► extract ──► validate ──► chunk ──► embed ──► chroma ──► UI / query
             │         │          │              │
         static/    trafilatura   LLM agent    LLM agent
         browser    (fallback:    (schema-     (repair-once-
         (Playwright) text-content) constrained)  then-flag)
```

- **Fetchers** (`fetchers/`) — static HTTP + Playwright headless browser;
  obstacle detection (popups, cookie banners, redirects, rate limits, CAPTCHA)
  is driven entirely by the versioned [`config/obstacles.json`](config/obstacles.json),
  validated against its formal JSON Schema.
- **Strip** (`pipeline/strip.py`) — boilerplate removal via Trafilatura before
  anything reaches an LLM (Rule 3: strip before you prompt).
- **Agents** (`agents/`) — classifier/extractor/validator/answer agents. Each
  prompt + schema lives in its own file. All output validates against
  `schemas/` before it can proceed.
- **Pipeline** (`pipeline/`) — semantic chunking, local embedding (BGE-M3 /
  MiniLM fallback), Chroma storage with dedup-by-source-URL (re-ingesting a
  URL *replaces* its chunks).
- **Orchestrator** (`orchestrator/`) — resumable queue/state machine, batch
  runner, and the hybrid live-query path.
- **UI** (`ui/`) — Streamlit app with three views: Chat, Ingestion, Logs.

### Repository layout

```
agents/            LLM agents (extractor, validator, answer) + prompts + thin LLM client
config/            versioned JSON config (agents, embeddings, fetch, obstacles, orchestrator)
                   — each validated against a matching *.schema.json
fetchers/          static + headless-browser fetching, obstacle handling, logging, rate limit
orchestrator/      URL frontier, resumable batch runner, live-query path, CLI
pipeline/          stripping, semantic chunking, embedding, vector store, single-page ingestion
schemas/           Pydantic contracts (extraction, chunk, answer)
tests/             174 tests across every layer + full acceptance-criteria suite
ui/                Streamlit app (Chat / Ingestion / Logs)
```

---

## Requirements

- **Python 3.10+** (developed and tested on 3.14)
- `pip install -r requirements.txt`
- Playwright browser: `playwright install chromium`
- An LLM API key for the provider configured in `config/agents.json`
  (default: NVIDIA NIM, key in `NVIDIA_NIM_API_KEY`)

## Installation

```bash
git clone https://github.com/najafAli11-11/Web-Scrapper.git
cd Web-Scrapper

pip install -r requirements.txt
playwright install chromium

# Create .env from the template and set the API key(s) you need
copy .env.example .env     # Windows
# cp .env.example .env     # macOS / Linux
```

---

## Configuration

All behavior is config-driven; no values are hardcoded in agent logic.

| File | Purpose |
|------|---------|
| `config/agents.json` | LLM provider, model, temperature, tool-choice, parse-retry budget |
| `config/embeddings.json` | Embedding model (+ fallback), chunk size cap, Chroma path, collection prefix |
| `config/obstacles.json` | Obstacle → detection method → resolution policy (popup, banner, redirect, rate limit, CAPTCHA, ...) |
| `config/fetch.json` | Timeouts, retries, per-domain rate limits |
| `config/orchestrator.json` | Batch retry budget / backoff |

Every file is validated against its `*.schema.json` at load time.

**API keys** live in `.env` (gitignored). Each key is optional — only the
provider in `config/agents.json` needs one. See `.env.example`.

**Environment overrides** (for scratch/testing runs — keep the real corpus
clean):

| Variable | Effect |
|----------|--------|
| `SCRAPER_CHROMA_PATH` | Chroma store location for the CLI / pipeline |
| `UI_QUEUE_DB` / `UI_LOGS_DB` / `UI_CHROMA_PATH` | Queue, events, and corpus locations for the UI |

---

## Usage

### 1. Batch ingestion (CLI)

Put one URL per line in a file (`#` lines are comments):

```bash
python -m orchestrator run urls.txt
```

Resumable by default: `done`/`blocked`/`flagged`/`failed` rows are never
reprocessed; `pending`/`retrying` rows are. For a fresh run of every URL:

```bash
python -m orchestrator run urls.txt --reset
```

Optional flags: `--db <queue.db>` `--log <events.db>` `--mode batch|single`.

### 2. Live query (CLI)

```bash
python -m orchestrator query "https://example.com/article" --query "what does it say about X?"
```

Corpus hit → answered instantly from storage. Miss → single-shot scrape
(fetch → strip → extract → validate) and an answer from the fresh content,
**without writing to the corpus**.

### 3. Web UI

```bash
streamlit run ui/app.py
```

- **Chat** — ask a question; answers are grounded in the ingested corpus and
  every citation carries its source URL + scrape timestamp.
- **Ingestion** — paste URLs, submit, and watch each URL move through its
  status (`pending` → `done` / `blocked` / `flagged` / `failed`) live.
- **Logs** — reverse-chronological feed of pipeline events: fetch attempts,
  obstacle detections, validation failures/flags — each with a timestamp and
  reason.

---

## Testing

```bash
python -m pytest -q
```

**174 tests** cover every layer (fetching, stripping, extraction, validation,
chunking, embedding, storage, queue, batch runner, live query, UI, logger) —
including `tests/test_acceptance.py`, a hermetic suite shaped directly around
each of the 8 SPEC.md acceptance criteria.

All 8 acceptance criteria are additionally **verified live, end to end**
(real HTTP server, real Playwright browser, real BGE-M3 embeddings, real LLM):
static/SPA/PDF ingestion with provenance, CAPTCHA → blocked, chatbot answers
with traceable citations, per-URL ingestion statuses, blocked+flagged events
in logs, live-query fallback, dedup on re-ingest, and repair-once-then-flag.

---

## Scope & non-goals

Deliberately out of scope for v1 (see `SPEC.md` for the full list):

- Auth-gated / login-required scraping
- CAPTCHA-solving, stealth/evasion, proxy rotation
- Scheduled / automatic re-scraping (the corpus is static until manual re-ingestion)
- Cross-URL near-duplicate detection
- Distributed worker infrastructure
- Multi-modal (image/video) extraction
- Per-domain custom extraction schemas

---

## Documentation

- [`AGENTS.md`](AGENTS.md) — project constitution: architectural rules,
  definitions of done, and non-goals
- [`SPEC.md`](SPEC.md) — the v1 spec, functional requirements, and the 8
  acceptance criteria
- [`BUILD_PLAN.md`](BUILD_PLAN.md) — the milestone-by-milestone build record
