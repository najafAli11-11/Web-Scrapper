# Spec — Web-Scrapper (Multi-Agent Web Scraper → RAG Pipeline)

Grounded in `AGENTS.md` (constitution) and Phase 1 research findings.
This is a v1 spec for a personal-scale system; scope is deliberately bounded
per the constitution's non-goals.

## Goal

Build a scraper that can extract meaningful content from any website —
regardless of HTML complexity, JS-rendering, or structure — and feed it into
a RAG pipeline that powers a chatbot, a search/retrieval interface, and a
live-query agent tool, without requiring per-site logic or maintenance.

## User scenarios

- When a user provides a list of URLs, the system fetches, extracts, chunks,
  and stores each page's content in the vector store, regardless of the
  site's layout or rendering method.
- When a user queries the chatbot or search interface, they get answers
  grounded in the ingested corpus, with source attribution (URL + scrape
  date) traceable per result.
- When the live-query agent tool is asked about a URL not already in the
  corpus, it scrapes that URL on the spot (fetch → strip → extract →
  validate → answer), without needing a full batch re-ingestion run.
- When a page can't be scraped (CAPTCHA-gated, hard 403, non-HTML content
  the extractor can't handle), the user sees that it was skipped and why —
  never a silent gap in the corpal with no explanation.
- When a user re-runs ingestion manually, previously-scraped URLs are
  re-fetched and their corpus entries updated, without duplicating content.

## Functional requirements

**Fetching**
1. The system MUST attempt static fetch first, falling back to headless
   browser (Playwright) rendering when static fetch yields insufficient
   content (e.g. empty body, JS-shell detected).
2. The system MUST detect and classify obstacles per the config-driven
   Obstacle Handling policy (Rule 7): popup/cookie banners dismissed,
   redirects followed and logged, rate limits backed off, CAPTCHA gates
   marked as terminal-blocked (never solved/circumvented).
3. The system MUST respect a per-domain rate limit (configurable, default
   conservative) to avoid overwhelming target servers, independent of the
   robots.txt-permission decision already settled in the constitution.
4. Every fetch attempt (success or failure) MUST be logged with URL,
   timestamp, outcome, and reason if failed.

**Extraction**
5. Raw HTML MUST be passed through boilerplate stripping (Trafilatura)
   before reaching any LLM extraction call.
6. Extraction MUST produce output conforming to a fixed generic schema
   (see below) — not a free-form summary, and not a per-domain schema.
7. Extraction MUST run in two supported modes: batch (ingestion-time, one
   or many URLs from a queue) and single-shot (query-time, one URL,
   optimized for low latency) — sharing the same schema/logic.
8. Non-HTML content the system encounters (PDF, plain text, etc.) MUST be
   routed to an appropriate extraction path rather than failing outright;
   if no path exists for a content type, it's flagged, not silently dropped.

**Validation**
9. Extracted output MUST be validated against the schema before storage.
   On failure, one repair attempt is made (re-prompt with validation
   errors); on repeated failure, the record is flagged for review and
   excluded from the corpus (not silently included, not silently dropped
   without a trace).

**Chunking & storage**
10. Validated content MUST be chunked semantically (by heading/section
    structure surfaced during extraction), not by fixed token windows.
11. Each chunk MUST carry metadata: source URL, scrape timestamp, page
    title, and section heading (if available).
12. Embeddings MUST be generated using a free, local model (default:
    BGE-M3; fallback: all-MiniLM-L6-v2 if compute-constrained).
13. Chunks MUST be stored in Chroma (embedded/local mode, per confirmed
    scale).

**Query interfaces**
14. The chatbot and search interface query the corpus only (no live
    scraping triggered by these paths).
15. The live-query agent tool checks the corpus first; if the requested
    URL/content is absent, it performs a single-shot scrape and answers
    from the freshly extracted content, without writing back to the
    persistent corpus unless the user explicitly requests ingestion.
16. A minimal local web UI MUST expose three views: a chat box (queries
    the corpus, shows answers with source URL + scrape date), an ingestion
    view (submit URLs, see status per URL — pending/done/blocked/flagged),
    and a logs view (chronological feed of pipeline events — fetch
    attempts, obstacle detections, validation failures/flags — reading
    from the same log records already required by Spec req. 4). This is a
    thin layer over the query/orchestrator logic — no separate backend
    architecture, no auth/multi-user support, consistent with small/personal
    scale.

**Dedup**
17. Re-ingesting a URL already in the corpus MUST replace its prior chunks
    (by source URL identity), not create duplicates.
18. Near-duplicate content across *different* URLs is out of scope for v1
    (no cross-URL semantic dedup) — see Out of scope.

## Edge cases & rules

- **Auth-gated content**: out of scope for v1. The scraper does not manage
  logins, sessions, or credentials on the user's behalf. If a page requires
  auth, it's treated the same as any other hard block — detected, logged,
  skipped. (A user manually providing session cookies/headers via config is
  a possible future extension, not v1.)
- **Empty/near-empty extraction result**: flagged as a low-confidence
  extraction, not silently stored as a near-empty chunk.
- **Duplicate URLs in an ingestion batch**: deduped before fetching (fetch
  each unique URL once per run).
- **Malformed/broken HTML**: still passed through the fetch → strip →
  extract pipeline; extraction failure here is handled like any other
  validation failure (flag, don't crash the batch).
- **Non-English content**: extraction/embedding should not assume English;
  BGE-M3 is multilingual by design, which covers this without extra work.
- **Very large pages**: chunker must handle pages that produce many chunks
  without a hard cap that silently truncates content.

## Out of scope (v1)

- Auth-gated / login-required scraping
- CAPTCHA-solving, stealth/evasion, proxy rotation (per constitution, Rule 7)
- Scheduled/automatic re-scraping or freshness expiry (corpus is static
  until manual re-ingestion, per constitution)
- Cross-URL near-duplicate detection
- Distributed/parallel-worker infrastructure (Kubernetes, task queues at
  scale) — sequential or simple local concurrency only
- Multi-modal extraction (images, video) — text content only for v1
- Per-domain custom extraction schemas

## Acceptance criteria

- [ ] Given a list of URLs spanning at least: a static HTML site, a
      JS-rendered SPA, and a PDF link, the system successfully ingests all
      three content types into the corpus with correct provenance metadata.
- [ ] Given a URL that returns a CAPTCHA challenge, the system logs it as
      blocked and continues processing the rest of the batch without
      crashing or hanging.
- [ ] Given a chatbot query about ingested content in the web UI, the
      answer includes traceable source URL(s), displayed to the user.
- [ ] Given a batch of submitted URLs in the ingestion view, each shows a
      correct status (pending/done/blocked/flagged) without requiring the
      user to check logs manually.
- [ ] Given a pipeline run that includes at least one blocked fetch and one
      flagged validation failure, the logs view shows both events with
      timestamp and reason, without needing to inspect raw log files.
- [ ] Given a live-query agent tool request for a URL not in the corpus,
      the system performs a single-shot scrape and returns an answer
      without requiring a manual batch re-ingestion run first.
- [ ] Given the same URL ingested twice, the corpus contains one set of
      chunks for it, not duplicates.
- [ ] Given a validation failure on extracted content, the system retries
      once, then flags the record — it does not silently drop or silently
      store invalid data.
