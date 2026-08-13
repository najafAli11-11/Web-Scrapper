# AGENTS.md — Project Constitution

This file defines persistent, project-wide rules for the multi-agent web
scraper → RAG pipeline. Every spec, plan, and build in this repo must follow
these rules. This is Phase 0 ("Constitution") — it sets intent; tests, hooks,
and review enforce it.

## What this project is

A robust web scraper built as a multi-agent pipeline, designed to scrape
**any website regardless of HTML complexity** — no assumptions about DOM
structure, layout, or rendering method may be hardcoded anywhere in the
pipeline. It must handle structural, content-type, availability, and
semantic ambiguity across arbitrary sites, feeding extracted data into a
RAG ingestion pipeline (chunking → embedding → vector store).

Confirmed scope: target sites are arbitrary / not a fixed known list, and
HTML structure cannot be assumed. This means:
- Headless-browser rendering (not just static fetch) is a baseline
  requirement, not a fallback for edge cases.
- No selector-based extraction anywhere — extraction must rely on
  agents/schemas that generalize across unknown layouts.
- Scraping proceeds regardless of robots.txt/ToS preference — no
  site-permission check gates a fetch. This does **not** extend to
  circumventing active access-control mechanisms (see Rule 7).
- Extraction is generic ("extract the meaningful content of this page"),
  not per-domain schemas — the project does not maintain a schema per
  site/vertical.

Confirmed consumption: small/personal scale (dozens of pages, not
distributed-scale). Output feeds a chatbot, a search/retrieval system, and
a live-query agent tool — so freshness and low-latency re-query matter more
than raw throughput.

Confirmed pipeline behavior:
- **Query mode is hybrid.** The live-query agent tool checks the ingested
  corpus first; if the URL/content isn't present, it falls back to a
  single-shot scrape (fetch → strip → extract → validate) and answers from
  that, without requiring a full batch re-ingestion run. This means the
  extractor/validator agents must work in two modes — batch (ingestion-time)
  and single-shot (query-time) — sharing the same schema and logic, differing
  only in latency/throughput expectations.
- **Freshness is static by default.** The corpus does not auto re-scrape or
  expire content on a timer. Content is only refreshed when ingestion is
  manually re-run. No background scheduling/expiry infrastructure is in
  scope. (A hybrid-mode live scrape, per above, may add fresh content but
  does not retroactively refresh existing corpus entries.)
- **Chunking is semantic, not fixed-size or LLM-driven.** Chunks are split
  along the page's actual structure (headings/sections/paragraphs) as
  surfaced by the extraction step, not by fixed token windows and not by a
  dedicated LLM chunking pass. This follows directly from generic,
  structure-aware extraction already being a project principle.
- **Embeddings are free and local, not a paid API.** Default embedding
  model is BGE-M3 (MIT license, self-hosted, dense+sparse+multi-vector) —
  no per-token cost, no data leaving the machine at embedding time. Fallback
  to all-MiniLM-L6-v2 if compute constraints make BGE-M3 impractical on the
  target hardware. Do not default to a paid embedding API (e.g. OpenAI)
  without an explicit spec-level reason to override this.

## Core architectural principles

1. **Deterministic control flow, non-deterministic extraction.**
   The orchestrator, URL frontier, queue, retry/backoff, and rate-limiting
   logic must be plain deterministic code — no LLM calls in these layers.
   LLM agents are reserved for: page classification, structured extraction,
   and validation/repair. This split must not be blurred as the system grows.

2. **Schema-constrained extraction only.**
   Extraction agents must return structured output validated against an
   explicit schema (tool-use / function-calling / Pydantic-style), never
   free-form text parsing. If a schema doesn't exist yet for a data type,
   define it before writing extraction logic for it.

3. **Strip before you prompt.**
   Raw HTML must go through boilerplate removal (readability-style
   extraction) before reaching any LLM call. Never send raw HTML with
   nav/ads/scripts to an extraction agent.

4. **Provenance is mandatory, end to end.**
   Every extracted record and every RAG chunk must carry source URL, scrape
   timestamp, and page title/section metadata all the way to the vector
   store. No exceptions — this is what makes RAG answers verifiable.

5. **Never silently drop or silently accept low-confidence data.**
   Failed validation or low-confidence extraction must be flagged for review
   or retried with a bounded retry budget (default: 1 repair attempt, then
   flag). Silent drops and silent pass-throughs are both bugs.

6. **Replayability over cleverness.**
   Every pipeline run must be resumable from its queue/state table (seen,
   pending, failed, retrying). If a run dies at page 40,000, restarting must
   not require re-scraping from zero.

7. **Obstacles are detected and handled per a defined policy — never
   circumvented.**
   The fetcher/navigator layer must treat popups, cookie banners, redirects,
   DOM drift, stale elements, slow responses, rate limits, session expiry,
   and similar friction as first-class, config-driven cases, each with an
   explicit resolution policy (dismiss, retry-with-backoff, re-authenticate,
   flag-and-skip, etc.) — not ad hoc handling bolted on per site.
   **CAPTCHA gates and comparable active access-control challenges are
   always a terminal "blocked" state**: detect, log with reason, skip, move
   on. This project does not build or call CAPTCHA-solving, stealth/evasion,
   or proxy-rotation infrastructure to defeat these mechanisms — the
   distinction is "adapt to site messiness" vs. "defeat a site's deliberate
   access control," and this project only does the former.

## Definition of "done" (applies to every spec unless overridden)

A feature/spec is done when:
- It has an explicit schema (if it touches extraction or storage)
- Failure paths are handled, not just the happy path (blocked, timeout,
  malformed HTML, empty page, non-HTML content-type)
- Provenance metadata is present on every output record
- It's covered by at least one test that would fail if the feature were
  removed or broken
- It does not silently expand scope beyond what the spec's "Out of scope"
  section allows

## Explicit non-goals (keep this project from over-growing)

- This is not a general-purpose crawler framework — build for the actual
  use case (RAG ingestion), not a hypothetical future one.
- Don't let the orchestrator become an LLM agent "for flexibility." If it
  needs more flexibility, extend the state machine, not the model.
- Don't build distributed-worker/queue infrastructure (Kubernetes, Celery
  clusters, etc.) for the current confirmed scale (dozens of pages,
  personal use). Sequential or simple concurrent processing is sufficient
  until a spec demonstrates otherwise.

## Tech/structure conventions

- Repo layout should separate: `orchestrator/`, `fetchers/`, `agents/`
  (classifier, extractor, validator), `pipeline/` (chunking, embedding,
  storage), `schemas/`, `tests/`.
- Config (target sites, schemas, rate limits) lives in versioned config
  files, not hardcoded in agent logic.
- Every LLM agent's prompt + schema lives in its own file, not inline in
  orchestration code — makes iteration and testing tractable.
- LLM access is provider-agnostic and config-driven, never hardcoded to a
  vendor or a raw HTTP client: agents call a thin client abstraction (backed
  by litellm), and provider/model/API-key-env-var live in
  `config/agents.json`. Any litellm-supported provider (NVIDIA NIM, OpenAI,
  Anthropic, Groq, Ollama, OpenAI-compatible `api_base` endpoints, ...) is a
  config change, not a code change.
- Obstacle handling policy (per Rule 7) lives in a single versioned config
  (obstacle type → enabled, detection method, resolution policy), not
  scattered per-site conditionals.
- `config/obstacles.json` is validated against `config/obstacles.schema.json`
  (formal JSON Schema), which is the source of truth for the allowed obstacle
  keys, detection methods, and resolution policies.

## How this file should be used

- Loaded at the start of every session/build in this repo.
- If a rule here would block a reasonable one-off task, that's a signal to
  either update this file deliberately (not silently work around it) or
  scope the task as an explicit exception in its own spec.
- This file should stay short. If a rule doesn't pass the test — "would
  removing this line let the agent make a mistake?" — it doesn't belong here.
