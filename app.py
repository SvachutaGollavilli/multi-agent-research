# app.py
# -----------------------------------------------------------------
# Multi-Agent Research Assistant -- Streamlit UI
#
# Uses stream_pipeline_async() which runs graph.astream() in a
# background daemon thread and bridges events back via a queue,
# giving Streamlit's synchronous main thread real-time node updates.
#
# Run locally:   uv run streamlit run app.py
# Deploy:        push to GitHub, connect at share.streamlit.io
#                Set ANTHROPIC_API_KEY and TAVILY_API_KEY in Secrets.
# -----------------------------------------------------------------

from __future__ import annotations

import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# -----------------------------------------------------------------
# Secrets injection -- before any Anthropic/Tavily/aiohttp import
# -----------------------------------------------------------------

def _inject_secrets() -> None:
    try:
        for key in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY", "DATABASE_URL"):
            val = st.secrets.get(key)
            if val and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


_inject_secrets()

from src.agents.graph         import stream_pipeline_async
from src.cache.research_cache import stats as cache_stats
from src.output.report_writer import write_report


# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="=",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: #1e293b;
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
}
</style>
""", unsafe_allow_html=True)

AGENT_LABELS: dict[str, str] = {
    "planner":        "Planner",
    "cache_loader":   "Cache (HIT)",
    "researcher":     "Researcher",
    "merge_research": "Merge & Dedup",
    "quality_gate":   "Quality Gate",
    "retry_counter":  "Retry Counter",
    "analyst":        "Analyst",
    "synthesizer":    "Synthesizer",
    "writer":         "Writer",
    "reviewer":       "Reviewer",
}


# -----------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------

def render_sidebar(metrics: dict) -> None:
    with st.sidebar:
        st.title("Multi-Agent Research")
        st.caption("Claude + LangGraph + async pipeline")
        st.divider()

        if metrics.get("started"):
            st.subheader("Run Metrics")
            c1, c2 = st.columns(2)
            c1.metric("Cost",    f"${metrics.get('cost', 0):.5f}")
            c2.metric("Tokens",  f"{metrics.get('tokens', 0):,}")
            c1.metric("Sources", metrics.get("sources", 0))
            c2.metric("Claims",  metrics.get("claims", 0))

            rev   = metrics.get("revisions")
            score = metrics.get("score")
            if rev is not None:
                c1.metric("Revisions", rev)
            if score:
                c2.metric("Review", f"{score}/10")

            qs = metrics.get("quality_score")
            if qs is not None:
                label = "PASS" if metrics.get("quality_passed") else "FAIL"
                st.caption(f"Quality gate: {qs:.2f} ({label})")

            dur = metrics.get("duration")
            if dur:
                st.caption(f"Elapsed: {dur:.1f}s")

            st.divider()

        try:
            cs = cache_stats()
            if cs.get("total_entries", 0) > 0:
                st.subheader("Cache")
                st.caption(
                    f"{cs.get('fresh_entries', 0)} fresh  |  "
                    f"{cs.get('total_hits', 0)} hits  |  "
                    f"threshold {cs.get('similarity_threshold', 0.60)}"
                )
                st.divider()
        except Exception:
            pass

        st.caption("Step 16 -- async pipeline + Supabase")


# -----------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------

def render_step_log(steps: list[dict]) -> None:
    for step in steps:
        icon    = "OK" if step["status"] == "done" else ">>"
        dur     = f"  [{step['duration']}ms]" if step.get("duration") else ""
        summary = f"  {step['summary']}" if step.get("summary") else ""
        st.markdown(f"`{icon}` **{step['label']}**{dur}{summary}")


def render_live_state(state: dict) -> None:
    parts = []

    if state.get("sub_topics"):
        parts.append(("Sub-topics", ", ".join(state["sub_topics"][:3])))

    src_n = len(state.get("sources", []))
    if src_n:
        parts.append(("Sources", str(src_n)))

    qs = state.get("quality_score")
    if qs is not None:
        label = "PASS" if state.get("quality_passed") else "FAIL"
        parts.append(("Quality", f"{qs:.2f} ({label})"))

    claim_n = len(state.get("key_claims", []))
    if claim_n:
        parts.append(("Claims", str(claim_n)))

    syn = state.get("synthesis")
    if syn:
        parts.append(("Synthesis", f"{len(syn)} chars"))

    rev = state.get("revision_count")
    if rev:
        parts.append(("Draft", f"v{rev}"))

    review = state.get("review") or {}
    if review.get("score"):
        label = "PASS" if review.get("passed") else "FAIL"
        parts.append(("Review", f"{review['score']}/10 ({label})"))

    cost = state.get("cost_usd", 0)
    if cost:
        parts.append(("Cost", f"${cost:.5f}"))

    if parts:
        for lbl, val in parts:
            st.markdown(f"**{lbl}:** {val}")
    else:
        st.caption("Waiting for first node...")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main() -> None:
    metrics: dict = {}
    render_sidebar(metrics)

    st.title("Multi-Agent Research Assistant")
    st.caption(
        "Enter a research question. The async pipeline will decompose, "
        "search (Tavily + Wikipedia in parallel), analyse, synthesise, "
        "write, and review -- all streamed live."
    )

    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "question",
            placeholder="e.g. What is FAISS and how does it work?",
            label_visibility="collapsed",
        )
    with col_btn:
        run_btn = st.button("Run", type="primary", use_container_width=True)

    if not run_btn or not query.strip():
        st.info("Enter a research question and click **Run**.")
        return

    query = query.strip()
    st.divider()

    col_steps, col_state = st.columns([3, 2])
    with col_steps:
        st.subheader("Pipeline")
        steps_ph = st.empty()
    with col_state:
        st.subheader("Live State")
        state_ph = st.empty()

    steps:       list[dict] = []
    final_state: dict       = {}
    run_t0     = time.time()
    run_error: str | None   = None

    status_box = st.status("Running async pipeline...", expanded=True)

    try:
        # stream_pipeline_async is a SYNC generator that internally uses
        # graph.astream() running in a background daemon thread.
        for node_name, update, accumulated in stream_pipeline_async(query):
            final_state = accumulated

            trace_entries = [
                t for t in update.get("pipeline_trace", [])
                if t.get("agent") == node_name
            ]
            trace = trace_entries[0] if trace_entries else {}

            steps.append({
                "label":    AGENT_LABELS.get(node_name, node_name),
                "duration": trace.get("duration_ms", 0),
                "summary":  trace.get("summary", ""),
                "status":   "done",
            })

            with steps_ph.container():
                render_step_log(steps)

            with state_ph.container():
                render_live_state(accumulated)

            status_box.write(f"Completed: {AGENT_LABELS.get(node_name, node_name)}")

            metrics.update({
                "started":        True,
                "cost":           accumulated.get("cost_usd", 0),
                "tokens":         accumulated.get("token_count", 0),
                "sources":        len(accumulated.get("sources", [])),
                "claims":         len(accumulated.get("key_claims", [])),
                "revisions":      accumulated.get("revision_count"),
                "score":          (accumulated.get("review") or {}).get("score"),
                "quality_score":  accumulated.get("quality_score"),
                "quality_passed": accumulated.get("quality_passed"),
                "duration":       time.time() - run_t0,
            })
            render_sidebar(metrics)

        status_box.update(label="Pipeline complete", state="complete")

    except Exception as e:
        run_error = str(e)
        status_box.update(label=f"Error: {e}", state="error")
        st.error(f"Pipeline error: {e}")

    metrics["duration"] = time.time() - run_t0
    render_sidebar(metrics)

    # Final report
    st.divider()
    final_report = (
        final_state.get("final_report")
        or final_state.get("current_draft", "")
    )

    if final_report:
        st.subheader("Final Report")
        st.markdown(final_report)

        st.divider()
        try:
            docx_path = write_report(final_state, run_id=final_state.get("run_id", ""))
            with open(docx_path, "rb") as fh:
                docx_bytes = fh.read()

            safe = "".join(
                c if (c.isalnum() or c == "_") else "_"
                for c in query.lower()[:40].replace(" ", "_")
            ).strip("_") or "report"

            st.download_button(
                label="Download Report (.docx)",
                data=docx_bytes,
                file_name=f"{safe}.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            )
        except Exception as e:
            st.caption(f"Could not generate .docx: {e}")

    elif not run_error:
        st.warning("Pipeline completed but produced no report.")

    trace = final_state.get("pipeline_trace", [])
    if trace:
        with st.expander("Full Pipeline Trace", expanded=False):
            try:
                import pandas as pd
                rows = [
                    {
                        "Agent":   t.get("agent", ""),
                        "ms":      t.get("duration_ms", 0),
                        "Tokens":  t.get("tokens", 0),
                        "Summary": t.get("summary", ""),
                    }
                    for t in trace
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except ImportError:
                for t in trace:
                    st.text(
                        f"{t.get('agent',''):14s} | "
                        f"{t.get('duration_ms',0):5d}ms | "
                        f"{t.get('summary','')}"
                    )


if __name__ == "__main__":
    main()
