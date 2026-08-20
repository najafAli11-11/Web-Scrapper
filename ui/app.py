"""M9 Streamlit UI: Chat / Ingestion / Logs (Spec req. 14/15/16).

AUTO-REFRESH IS BOUNDED BY THE BATCH CHILD'S EXIT, NOT AN ARBITRARY CAP.
At the end of each script run, if auto-refresh is armed, the script sleeps
2s and reruns while `BatchRunner.is_running()` is True; the moment the child
exits — including a crash or a Stop — is_running() is False, refresh ends,
and the UI shows the terminal state. is_running() is marker-based, so a UI
restart mid-batch resumes auto-refresh for the remainder of the run.

Constitutional boundaries kept here:
- The UI never writes queue or chroma state. Queue state is read-only via
  QueueView; the corpus is queried through VectorStore.query. The ONLY
  writer the UI opens is the shared events FetchLogger — chat answers
  PRODUCE events (answer_generation_failed, live_query), which is
  legitimate; it is cached like every other resource and WAL-safe against
  the read-only Logs view.
- Ingestion spawns the M7 orchestrator CLI as a detached child using
  sys.executable (never a bare "python"). A second submit against the same
  queue.db is blocked both by the disabled Submit button and, as the hard
  backstop, by BatchAlreadyRunning.
- The chat's corpus collection name comes from the SAME corpus_collection
  helper as the M8 live-query path — never hardcoded or re-derived here.
  Corpus row provenance likewise via the shared pipeline.store.row_provenance.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import streamlit as st

from agents.answer import generate_answer
from agents.config_loader import load_agent_config
from agents.llm.client import LiteLLMClient
from fetchers.logger import FetchLogger
from orchestrator.live_query import live_query
from pipeline.config_loader import load_pipeline_config
from pipeline.embed import load_embedder
from pipeline.store import VectorStore, corpus_collection, row_provenance
from ui.db_view import (
    BatchAlreadyRunning,
    BatchRunner,
    EventLogView,
    QueueView,
    clear_all_logs,
    delete_url_everywhere,
    spawn_ingestion,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
K = 5


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    p = Path(value) if value else default
    return p if p.is_absolute() else REPO_ROOT / p


QUEUE_DB = _env_path("UI_QUEUE_DB", REPO_ROOT / "data" / "queue.db")
LOGS_DB = _env_path("UI_LOGS_DB", REPO_ROOT / "data" / "logs.db")
CHROMA_PATH = _env_path("UI_CHROMA_PATH", REPO_ROOT / "data" / "chroma")


@st.cache_resource(show_spinner=False)
def _pipeline_config() -> dict:
    return load_pipeline_config()


@st.cache_resource(show_spinner=False)
def _agent_config() -> dict:
    return load_agent_config()


@st.cache_resource(show_spinner=False)
def _llm_client():
    client = LiteLLMClient(_agent_config())
    client.set_logger(_logger())
    return client


@st.cache_resource(show_spinner=False)
def _embedder():
    return load_embedder(_pipeline_config(), logger=None)


@st.cache_resource(show_spinner=False)
def _store():
    return VectorStore(
        str(CHROMA_PATH),
        collection_prefix=_pipeline_config()["store"]["collection_prefix"],
    )


@st.cache_resource(show_spinner=False)
def _logger() -> FetchLogger:
    # check_same_thread=False: the cached logger outlives one script run, and
    # Streamlit runs each rerun on a different thread — SQLite connections are
    # thread-bound by default and would drop events with a silent best-effort
    # failure (the M9 live verification caught exactly this).
    return FetchLogger(LOGS_DB, check_same_thread=False)


def _batch_runner() -> BatchRunner:
    if "batch_runner" not in st.session_state:
        st.session_state.batch_runner = BatchRunner(queue_db=QUEUE_DB, logs_db=LOGS_DB)
    return st.session_state.batch_runner


def _render_sidebar() -> None:
    runner = _batch_runner()
    with st.sidebar:
        st.subheader("Pipeline state")
        st.write(f"Queue: `{QUEUE_DB.name}`")
        st.write(f"Logs: `{LOGS_DB.name}`")
        st.write(f"Chroma: `{CHROMA_PATH.name}`")
        running = runner.is_running()
        st.metric("Batch running", "yes" if running else "no")
        st.caption("Auto-refresh polls every 2s while a batch is running and stops when it exits.")
        if running and st.button("Stop batch"):
            runner.stop()
            st.rerun()
        if st.button("Refresh now"):
            st.rerun()


def _render_chat() -> None:
    st.header("Chat")
    st.caption("Corpus-only Q&A (req 14): no live scraping on this path.")
    question = st.text_input("Ask about the corpus", key="chat_question")
    if st.button("Ask", key="ask_corpus") and question.strip():
        _corpus_qa(question.strip())

    st.divider()
    st.caption("URL live query (req 15): corpus first, single-shot scrape on miss.")
    col1, col2 = st.columns([3, 2])
    url = col1.text_input("URL", key="chat_url")
    url_query = col2.text_input("Optional query text", key="chat_url_query")
    if st.button("Ask about URL", key="ask_url") and url.strip():
        _url_query_start(url.strip(), url_query.strip() or None)
        st.rerun()

    _url_query_poll()


def _corpus_qa(question: str) -> None:
    embedder = _embedder()
    collection = corpus_collection(embedder, _pipeline_config())
    with st.spinner("Searching the corpus..."):
        query_embedding = embedder.embed([question])[0]
        rows = _store().query(query_embedding, k=K, collection_name=collection)
    evidence = [{"text": r["document"], "provenance": row_provenance(r["metadata"])} for r in rows]
    if not evidence:
        st.warning("No evidence in the corpus for that question.")
        return
    with st.spinner("Synthesizing the answer..."):
        answer = generate_answer(
            question, evidence, client=_llm_client(), agent_cfg=_agent_config(), logger=_logger(),
            url=None,
        )
    if answer is None:
        st.error("Answer synthesis failed — showing raw evidence below (see answer_generation_failed in Logs).")
    else:
        st.markdown(answer.answer)
        st.markdown("**Citations**")
        for i, citation in enumerate(answer.citations, 1):
            heading = citation.section_heading or "whole page"
            with st.expander(f"[{i}] {citation.source_url} — {heading}"):
                st.write(f"**Source:** {citation.source_url}")
                st.write(f"**Scraped:** {citation.scrape_timestamp}")
                if citation.page_title:
                    st.write(f"**Page title:** {citation.page_title}")
                if citation.section_heading:
                    st.write(f"**Section:** {citation.section_heading}")
                st.markdown(f"> {citation.quote}")
    st.markdown("**Raw evidence**")
    for i, ev in enumerate(evidence, 1):
        prov = ev["provenance"]
        heading = prov.get("section_heading") or "whole page"
        with st.expander(f"[{i}] {prov.get('source_url') or '?'} — {heading}"):
            st.write(prov)
            st.write(ev["text"])


def _render_live_event(event: dict) -> None:
    """Render one pipeline event as a compact one-line indicator."""
    outcome = event.get("outcome") or ""
    reason = event.get("reason") or ""
    ev_type = event.get("event_type") or ""

    if outcome in ("success", "ok", "corpus_hit", "kept_static", "completed",
                    "single_shot_ok", "corpus_miss_single_shot_ok", "dismissed",
                    "rescued", "stale_element_recovered", "title_ok"):
        icon = "\u2705"
    elif outcome in ("error", "failed", "fetch_failed", "blocked", "flagged",
                      "no_content", "parse_exhausted", "corpus_error",
                      "answer_generation_failed", "navigation_error",
                      "browser_error", "chunk_ingest_failed", "chunk_failed",
                      "corpus_error"):
        icon = "\u274c"
    elif outcome in ("obstacle_detected", "encoding_fallback", "rate_limit_wait",
                      "strip_parse_warning", "detection_failure", "timeout",
                      "answer_fallback_rescued", "answer_unwrap_rescue",
                      "networkidle_timeout", "load_more_stall"):
        icon = "\u26a0\ufe0f"
    elif outcome in ("tool_use", "json_from_content", "json_mode",
                      "retrying", "store_start", "fetch_done",
                      "validate_done", "stripping", "extracting"):
        icon = "\u23f3"
    else:
        icon = "\u25cf"

    parts = [f"{icon} **{ev_type}**"]
    if outcome:
        parts.append(outcome)
    if reason:
        parts.append(reason)
    st.markdown(" | ".join(parts))
    if event.get("details_json"):
        try:
            details = json.loads(event["details_json"])
            with st.expander("details", expanded=False):
                st.code(json.dumps(details, indent=2))
        except (json.JSONDecodeError, TypeError):
            pass


def _url_query_start(url: str, query: Optional[str]) -> None:
    """Start a live query in a background thread."""
    state: dict = {"result": None, "error": None, "done": False}

    def _run():
        try:
            state["result"] = live_query(
                url,
                query=query,
                logger=_logger(),
                agent_cfg=_agent_config(),
                client=_llm_client(),
                embedder=_embedder(),
                store=_store(),
                pipeline_cfg=_pipeline_config(),
                mode="single",
                k=K,
            )
        except Exception as exc:
            state["error"] = exc
        finally:
            state["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    st.session_state["lq_thread"] = thread
    st.session_state["lq_state"] = state
    st.session_state["lq_url"] = url
    st.session_state["lq_query"] = query
    st.session_state["lq_active"] = True


def _url_query_poll() -> None:
    """Poll the events DB and render live logs while the pipeline runs."""
    if not st.session_state.get("lq_active"):
        return

    thread = st.session_state["lq_thread"]
    state = st.session_state["lq_state"]
    url = st.session_state["lq_url"]

    st.markdown(f"**Live query:** `{url}`")

    view = EventLogView(LOGS_DB)
    events = view.recent(limit=50, url_filter=url)
    if events:
        for event in reversed(events):
            _render_live_event(event)

    if state["done"]:
        st.session_state["lq_active"] = False
        thread.join(timeout=5)

        result = state["result"]
        error = state["error"]

        st.divider()
        if error:
            st.error(f"Pipeline failed: {error}")
        elif result is None:
            st.error("Pipeline returned no result.")
        else:
            _render_url_result(result, url)
    else:
        st.info("\u23f3 Pipeline running...")
        time.sleep(1.5)
        st.rerun()


def _render_url_result(result, url: str) -> None:
    """Render the final LiveQueryResult in the Chat tab."""
    if result.status in ("ok", "single_shot_ok"):
        st.success(f"**{result.status}** \u2014 {result.source_used} \u00b7 evidence: {len(result.evidence)}")
    else:
        st.error(f"**{result.status}**" + (f" \u2014 {result.reason}" if result.reason else ""))

    prov = result.provenance
    st.write(
        f"**source_url:** {prov.get('source_url')} \u00b7 "
        f"**scrape_timestamp:** {prov.get('scrape_timestamp')} \u00b7 "
        f"**page_title:** {prov.get('page_title')}"
    )

    query = st.session_state.get("lq_query")
    if result.evidence and query:
        evidence = [{"text": ev.text, "provenance": ev.provenance} for ev in result.evidence]
        with st.spinner("Synthesizing the answer..."):
            answer = generate_answer(
                query, evidence, client=_llm_client(), agent_cfg=_agent_config(),
                logger=_logger(), url=url,
            )
        if answer is None:
            st.error("Answer synthesis failed \u2014 showing raw evidence below.")
        else:
            st.markdown(answer.answer)
            for citation in answer.citations:
                st.markdown(f"- [{citation.source_url}] {citation.quote}")
    elif result.evidence:
        st.markdown("**Evidence**")
        for i, ev in enumerate(result.evidence, 1):
            heading = ev.provenance.get("section_heading") or "whole page"
            with st.expander(f"[{i}] {ev.provenance.get('source_url')} \u2014 {heading}"):
                st.write(ev.provenance)
                st.write(ev.text)


def _render_ingestion() -> None:
    st.header("Ingestion")
    runner = _batch_runner()
    running = runner.is_running()

    with st.form("ingest_form"):
        urls_text = st.text_area(
            "URLs (one per line)",
            height=120,
            help="Spawned as a detached batch run via the M7 orchestrator CLI.",
        )
        submit = st.form_submit_button(
            "Submit ingestion",
            disabled=running,
            help=(
                "A batch is already running against the queue — wait for it to exit."
                if running
                else "Spawn a detached batch run."
            ),
        )
    if submit:
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.error("Enter at least one URL.")
        else:
            try:
                spawn_ingestion(urls, queue_db=QUEUE_DB, logs_db=LOGS_DB)
            except BatchAlreadyRunning:
                st.error(f"A batch is already running against {QUEUE_DB.name}.")
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.session_state["auto_refresh"] = True
                st.success(f"Batch started for {len(urls)} URL(s).")
                st.rerun()

    queue = QueueView(QUEUE_DB)
    counts = queue.counts()
    if counts:
        st.subheader("Queue status")
        for state, n in sorted(counts.items()):
            st.markdown(f"- **{state}**: {n}")
        st.dataframe(queue.states(), use_container_width=True)

        st.divider()
        st.subheader("Delete URL")
        st.caption("Remove a URL from the queue, vector store, and logs.")
        all_urls = [row["url"] for row in queue.states()]
        delete_url = st.selectbox("Select URL to delete", all_urls, key="delete_url_select")
        confirm_delete = st.checkbox("I confirm", key="delete_confirm")
        if st.button("Delete URL", disabled=not confirm_delete, type="primary"):
            if delete_url:
                with st.spinner(f"Deleting {delete_url}..."):
                    result = delete_url_everywhere(
                        delete_url,
                        queue_db=QUEUE_DB,
                        logs_db=LOGS_DB,
                        collection_name=corpus_collection(_embedder(), _pipeline_config()),
                        store=_store(),
                    )
                st.success(
                    f"Deleted: {result['queue_deleted']} queue row(s), "
                    f"{result['logs_deleted']} log event(s), and all vector store chunks."
                )
                st.rerun()
    else:
        st.info("Queue is empty — submit URLs above to start ingestion.")


def _render_logs() -> None:
    st.header("Logs")

    col1, col2 = st.columns([3, 1])
    with col2:
        confirm_clear = st.checkbox("I confirm", key="clear_logs_confirm")
        if st.button("Clear All Logs", disabled=not confirm_clear, type="primary"):
            deleted = clear_all_logs(db_path=LOGS_DB)
            st.success(f"Cleared {deleted} log event(s).")
            st.rerun()

    view = EventLogView(LOGS_DB)
    selected = st.multiselect("Event types", view.event_types(), key="log_types")
    url_filter = st.text_input("Filter by URL fragment", key="log_url_filter")
    events = view.recent(
        limit=200,
        event_types=selected or None,
        url_filter=url_filter.strip() or None,
    )
    if not events:
        st.info("No events yet.")
        return
    for event in events:
        label = f"{event['event_type']} | {event['url'] or ''} | {event['outcome'] or ''}"
        with st.expander(label):
            st.write(f"**ts:** {event['ts']}")
            if event["reason"]:
                st.write(f"**reason:** {event['reason']}")
            if event["details_json"]:
                st.code(json.dumps(json.loads(event["details_json"]), indent=2))


def main() -> None:
    st.set_page_config(page_title="Scraper RAG UI", layout="wide", page_icon="\U0001f50d")
    st.title("Web Scraper · RAG")
    _render_sidebar()
    tab_chat, tab_ingest, tab_logs = st.tabs(["Chat", "Ingestion", "Logs"])
    with tab_chat:
        _render_chat()
    with tab_ingest:
        _render_ingestion()
    with tab_logs:
        _render_logs()

    # Auto-refresh: bounded by the batch child's exit, not an arbitrary cap.
    if st.session_state.get("auto_refresh"):
        time.sleep(2)
        if _batch_runner().is_running():
            st.rerun()
        st.session_state["auto_refresh"] = False


if __name__ == "__main__":
    main()
