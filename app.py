# app.py  —  Multi-Agent Research Assistant  (v5)
from __future__ import annotations

import os
import time
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Secrets injection ────────────────────────────────────────────
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
from src.output.report_writer import write_report


# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 16px; padding: 1.6rem 2rem 1.4rem; margin-bottom: 1rem;
    border: 1px solid #1e40af33;
}
.hero-row { display:flex; justify-content:space-between; align-items:flex-start; }
.hero-title { font-size:1.75rem; font-weight:700; color:#f1f5f9; margin:0 0 .3rem; letter-spacing:-.5px; }
.hero-sub   { font-size:.88rem; color:#94a3b8; margin:0; }
.session-pill {
    font-size:.7rem; font-family:monospace;
    background:#0f172a; border:1px solid #1e293b;
    color:#94a3b8; border-radius:6px;
    padding:4px 10px; white-space:nowrap; margin-top:4px;
}
.session-pill span { color:#60a5fa; font-weight:600; }

/* ── Pipeline tracker ── */
.pipeline-track { display:flex; align-items:center; flex-wrap:wrap; gap:0; margin:.8rem 0 .4rem; }
.pipe-node {
    display:flex; align-items:center; gap:5px; padding:5px 12px; border-radius:20px;
    font-size:.75rem; font-weight:500; white-space:nowrap; transition:all .3s ease;
}
.pipe-node.done    { background:#052e16; color:#4ade80; border:1px solid #166534; }
.pipe-node.active  { background:#1e3a5f; color:#60a5fa; border:1px solid #2563eb;
    animation:pn 1.2s ease-in-out infinite; }
.pipe-node.pending { background:#1e293b; color:#475569; border:1px solid #334155; }
.pipe-node.revisit { background:#1c1435; color:#c084fc; border:1px solid #7c3aed;
    animation:pn 1.2s ease-in-out infinite; }
@keyframes pn { 0%,100%{box-shadow:0 0 6px #2563eb44} 50%{box-shadow:0 0 14px #2563eb88} }
.pipe-arrow { color:#334155; font-size:.75rem; padding:0 1px; flex-shrink:0; }
.revision-badge {
    display:inline-flex; align-items:center; gap:6px; background:#1c1435;
    border:1px solid #7c3aed33; border-radius:8px; padding:5px 12px;
    font-size:.75rem; color:#c084fc; margin-top:.4rem;
}

/* ── Active agent banner ── */
.active-agent {
    background:linear-gradient(90deg,#0f2a4a,#0f172a); border:1px solid #2563eb44;
    border-radius:10px; padding:.6rem 1rem; margin:.4rem 0;
    display:flex; align-items:center; gap:10px; font-size:.88rem; color:#93c5fd;
}
.dot-pulse {
    width:8px; height:8px; border-radius:50%; background:#3b82f6;
    display:inline-block; flex-shrink:0; animation:dp 1s ease-in-out infinite;
}
.dot-pulse.purple { background:#a78bfa; }
@keyframes dp { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }

/* ── Result metric cards ── */
.result-bar { display:flex; gap:8px; margin:.8rem 0; }
.metric-card { flex:1; background:#0f172a; border:1px solid #1e293b; border-radius:10px; padding:.6rem .8rem; text-align:center; }
.metric-card .val { font-size:1.25rem; font-weight:700; color:#f1f5f9; display:block; }
.metric-card .lbl { font-size:.68rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.5px; display:block; margin-top:2px; }

/* ── Quality badge ── */
.quality-badge { display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:20px; font-size:.78rem; font-weight:600; }
.quality-badge.pass { background:#052e16; color:#4ade80; border:1px solid #166534; }
.quality-badge.fail { background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d; }

/* ── Download links row ── */
.dl-row { display:flex; flex-wrap:wrap; gap:10px; margin-top:1rem; }
.dl-chip {
    font-size:.78rem; color:#93c5fd; background:#0f172a;
    border:1px solid #1e293b; border-radius:6px; padding:4px 12px;
    text-decoration:none;
}

/* ── Report card ── */
.report-card {
    background:#0c1525; border:1px solid #1e293b;
    border-radius:12px; padding:1.2rem 1.5rem; margin-top:.5rem;
}
.report-label { font-size:.68rem; color:#475569; text-transform:uppercase; letter-spacing:.7px; margin-bottom:.5rem; }

/* ── Sidebar (fully HTML-rendered into one empty slot) ── */
section[data-testid="stSidebar"] { background:#080e1a !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:0 !important; }

.sb-wrap { padding: 0; }

/* Fixed header */
.sb-head {
    padding: 1rem 1rem .7rem;
    border-bottom: 1px solid #161f2e;
}
.sb-title   { font-size:.95rem; font-weight:700; color:#e2e8f0; margin:0 0 1px; }
.sb-sub     { font-size:.65rem; color:#64748b; margin:0 0 5px; }
.sb-badge   {
    display:inline-block; font-size:.6rem; color:#475569;
    background:#0d1525; border:1px solid #1e293b;
    border-radius:4px; padding:2px 7px; margin-bottom:4px;
}
.sb-sid {
    font-family:monospace; font-size:.62rem; color:#334155; margin-top:3px;
}
.sb-sid b { color:#60a5fa; }

/* Section label */
.sb-sec { font-size:.6rem; color:#475569; text-transform:uppercase;
    letter-spacing:.7px; padding:.6rem 1rem .25rem; display:block; }

/* Compact row */
.sb-r {
    display:flex; justify-content:space-between;
    padding:.22rem 1rem; border-bottom:1px solid #0d1525;
}
.sb-r .l { font-size:.72rem; color:#64748b; }
.sb-r .v { font-size:.75rem; font-weight:600; color:#cbd5e1; }

/* Run card */
.rc {
    background:#0d1525; border:1px solid #161f2e; border-radius:7px;
    margin:.3rem .75rem; padding:.45rem .65rem;
}
.rc-hd { font-size:.63rem; color:#334155; margin:0 0 5px; }
.rc-hd b { color:#475569; }
.rc-g { display:grid; grid-template-columns:1fr 1fr; gap:2px 8px; }
.rc-i { font-size:.69rem; }
.rc-i .il { color:#475569; }
.rc-i .iv { color:#94a3b8; font-weight:500; }

/* Form */
div[data-testid="stTextInput"] input {
    background:#0f172a !important; border:1px solid #1e293b !important;
    border-radius:10px !important; color:#f1f5f9 !important;
    font-size:.95rem !important; padding:.6rem 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color:#2563eb !important; box-shadow:0 0 0 2px #2563eb33 !important;
}
div[data-testid="stFormSubmitButton"] button {
    background:linear-gradient(135deg,#1d4ed8,#2563eb) !important;
    color:white !important; border-radius:10px !important; font-weight:600 !important;
    border:none !important; height:42px !important; font-size:.9rem !important;
}
div[data-testid="stFormSubmitButton"] button:hover { opacity:.85 !important; }

.query-hint { font-size:.78rem; color:#334155; margin-top:.3rem; }
.query-hint code {
    background:#0f172a; color:#475569; border:1px solid #1e293b;
    border-radius:4px; padding:1px 7px; font-size:.75rem;
}

footer, #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Node definitions ─────────────────────────────────────────────
PIPELINE_NODES = [
    ("planner",        "Planner",   "📋"),
    ("researcher",     "Research",  "🔍"),
    ("merge_research", "Merge",     "🔗"),
    ("quality_gate",   "Quality",   "✅"),
    ("analyst",        "Analysis",  "🧠"),
    ("synthesizer",    "Synthesis", "⚗️"),
    ("writer",         "Writer",    "✍️"),
    ("reviewer",       "Review",    "📝"),
]
NODE_MAP = {k: (lbl, ico) for k, lbl, ico in PIPELINE_NODES}
NODE_MAP.update({
    "cache_loader":     ("Cache Hit", "⚡"),
    "retry_counter":    ("Retry",     "🔄"),
    "fan_out_or_cache": ("Routing",   "↗️"),
})


# ── Session state ────────────────────────────────────────────────
def _init():
    if "sid" not in st.session_state:
        st.session_state.sid = uuid.uuid4().hex[:8].upper()
    for k, v in {
        "total_runs":   0,
        "total_cost":   0.0,
        "total_tokens": 0,
        "run_history":  [],   # newest first; each entry = dict
        "downloads":    [],   # list of {label, bytes, filename}
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Sidebar (single-slot, pure HTML replace) ─────────────────────
def _sb_slot():
    if "_sb" not in st.session_state:
        st.session_state._sb = st.sidebar.empty()
    return st.session_state._sb


def _row(l, v):
    return f"<div class='sb-r'><span class='l'>{l}</span><span class='v'>{v}</span></div>"


def render_sidebar():
    sid  = st.session_state.sid
    n    = st.session_state.total_runs
    cost = st.session_state.total_cost
    toks = st.session_state.total_tokens
    hist = st.session_state.run_history

    head = f"""
    <div class="sb-wrap">
    <div class="sb-head">
        <div class="sb-title">🔬 Research Assistant</div>
        <div class="sb-sub">Async multi-agent AI pipeline</div>
        <div class="sb-badge">LangGraph · Claude Haiku · 10-node</div><br>
        <div class="sb-sid">Session <b>#{sid}</b></div>
    </div>
    """

    totals = (
        "<span class='sb-sec'>Session</span>"
        + _row("Queries run",    str(n))
        + _row("Total API cost", f"${cost:.4f}")
        + _row("Total tokens",   f"{toks:,}")
    )

    history = ""
    if hist:
        history += "<span class='sb-sec'>Run History</span>"
        for i, r in enumerate(hist):
            run_num = n - i
            qs      = r.get("quality_score")
            q_label = ("PASS" if r.get("quality_passed") else "FAIL") if qs else "—"
            q_str   = f"{qs:.2f} {q_label}" if qs is not None else "—"
            score   = r.get("review_score")
            dur     = r.get("duration")
            history += f"""
            <div class="rc">
                <div class="rc-hd"><b>Run #{run_num}</b> · {r.get('qp','')}</div>
                <div class="rc-g">
                    <div class="rc-i"><span class="il">Sources </span><span class="iv">{r.get('sources','—')}</span></div>
                    <div class="rc-i"><span class="il">Claims </span><span class="iv">{r.get('claims','—')}</span></div>
                    <div class="rc-i"><span class="il">Time </span><span class="iv">{f"{dur:.0f}s" if dur else "—"}</span></div>
                    <div class="rc-i"><span class="il">Review </span><span class="iv">{f"{score}/10" if score else "—"}</span></div>
                    <div class="rc-i"><span class="il">Quality </span><span class="iv">{q_str}</span></div>
                    <div class="rc-i"><span class="il">Cost </span><span class="iv">${r.get('cost',0):.4f}</span></div>
                </div>
            </div>"""

    html = head + totals + history + "</div>"
    _sb_slot().markdown(html, unsafe_allow_html=True)


# ── Pipeline tracker ─────────────────────────────────────────────
def render_tracker(visit_log, active, revision_count):
    visited = set(visit_log)
    w_visits = visit_log.count("writer") + (1 if active == "writer" else 0)
    on_loop  = w_visits > 1

    nodes = ""
    for nid, lbl, ico in PIPELINE_NODES:
        is_act = nid == active
        done   = nid in visited
        if is_act and on_loop and nid == "writer":
            cls, ind = "revisit", '<span class="dot-pulse purple"></span>'
        elif is_act:
            cls, ind = "active", '<span class="dot-pulse"></span>'
        elif done:
            cls, ind = "done", "✓"
        else:
            cls, ind = "pending", ""
        nodes += f'<div class="pipe-node {cls}">{ind} {ico} {lbl}</div>'
        if nid != PIPELINE_NODES[-1][0]:
            nodes += '<span class="pipe-arrow">›</span>'

    html = f'<div class="pipeline-track">{nodes}</div>'
    if on_loop:
        html += f'<div class="revision-badge">🔄 Revision loop — pass {w_visits-1}</div>'
    st.markdown(html, unsafe_allow_html=True)


# ── Main ─────────────────────────────────────────────────────────
def main():
    _init()

    # Ensure sidebar slot is reserved before any other sidebar writes
    _sb_slot()
    render_sidebar()

    # ── Hero ─────────────────────────────────────────────────────
    sid = st.session_state.sid
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-row">
            <div>
                <div class="hero-title">🔬 Multi-Agent Research Assistant</div>
                <p class="hero-sub">Decompose · Search · Analyse · Synthesise · Report — streamed live</p>
            </div>
            <div class="session-pill">Session <span>#{sid}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search form ──────────────────────────────────────────────
    with st.form("sf", clear_on_submit=True):
        col_q, col_btn = st.columns([6, 1])
        with col_q:
            query = st.text_input(
                "q", value="",
                placeholder="Enter your research question…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Research →", use_container_width=True)

    st.markdown(
        "<div class='query-hint'>e.g. <code>What are vector databases and how do they work?</code></div>",
        unsafe_allow_html=True,
    )

    if not submitted or not query.strip():
        # ── Show previous downloads if any ───────────────────────
        if st.session_state.downloads:
            _render_downloads()
        return

    query = query.strip()

    # ── Clear previous report area immediately ───────────────────
    # Use a placeholder for the entire results region so it gets replaced
    results_area = st.empty()

    with results_area.container():
        st.divider()

        # ── Two-column layout: tracker left, live stats right ─────
        col_track, col_live = st.columns([3, 1])

        with col_track:
            tracker_ph = st.empty()
            active_ph  = st.empty()

        with col_live:
            live_ph = st.empty()

        visit_log:    list[str]  = []
        current_node: str | None = None
        revision_count           = 0

        def _refresh_tracker():
            with tracker_ph.container():
                render_tracker(visit_log, current_node, revision_count)

        def _refresh_live(acc: dict):
            srcs   = len(acc.get("sources", []))
            claims = len(acc.get("key_claims", []))
            cost   = acc.get("cost_usd", 0)
            qs     = acc.get("quality_score")
            live_ph.markdown(
                f"<div style='background:#0f172a;border:1px solid #1e293b;"
                f"border-radius:10px;padding:.8rem;font-size:.78rem;'>"
                f"<div style='color:#475569;font-size:.62rem;text-transform:uppercase;"
                f"letter-spacing:.6px;margin-bottom:.5rem'>Live</div>"
                f"<div style='color:#94a3b8;margin-bottom:3px'>Sources <b style='color:#e2e8f0'>{srcs}</b></div>"
                f"<div style='color:#94a3b8;margin-bottom:3px'>Claims <b style='color:#e2e8f0'>{claims}</b></div>"
                f"<div style='color:#94a3b8;margin-bottom:3px'>Cost <b style='color:#e2e8f0'>${cost:.4f}</b></div>"
                + (f"<div style='color:#94a3b8'>Quality <b style='color:#e2e8f0'>{qs:.2f}</b></div>" if qs else "")
                + "</div>",
                unsafe_allow_html=True,
            )

        _refresh_tracker()

        status_box = st.status("Starting pipeline…", expanded=False)

        # ── Run ───────────────────────────────────────────────────
        final_state: dict     = {}
        run_t0                = time.time()
        run_error: str | None = None

        try:
            for node_name, update, accumulated in stream_pipeline_async(query):
                final_state = accumulated

                if current_node and current_node not in ("retry_counter", "fan_out_or_cache"):
                    visit_log.append(current_node)

                current_node   = node_name
                revision_count = accumulated.get("revision_count", 0)

                label, icon = NODE_MAP.get(node_name, (node_name, "⚙️"))
                on_loop     = visit_log.count("writer") >= 1 and node_name == "writer"
                dot_cls     = "dot-pulse purple" if on_loop else "dot-pulse"
                txt_color   = "#c084fc"          if on_loop else "#93c5fd"
                extra       = " (revision)" if on_loop else ""

                active_ph.markdown(
                    f'<div class="active-agent">'
                    f'<span class="{dot_cls}"></span>'
                    f'<span style="color:{txt_color}">{icon} <strong>{label}</strong> — processing{extra}…</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                _refresh_tracker()
                _refresh_live(accumulated)
                status_box.update(label=f"Running: {label}…")

            if current_node and current_node not in ("retry_counter", "fan_out_or_cache"):
                visit_log.append(current_node)
            current_node = None
            active_ph.empty()
            _refresh_tracker()
            status_box.update(label="Pipeline complete ✅", state="complete")

        except Exception as e:
            run_error = str(e)
            active_ph.empty()
            status_box.update(label=f"Error: {e}", state="error")
            st.error(f"Pipeline error: {e}")

        duration = time.time() - run_t0

        # ── Update session state ──────────────────────────────────
        run_cost   = final_state.get("cost_usd", 0)
        run_tokens = final_state.get("token_count", 0)
        qs         = final_state.get("quality_score")

        st.session_state.total_runs   += 1
        st.session_state.total_cost   += run_cost
        st.session_state.total_tokens += run_tokens

        n = st.session_state.total_runs
        st.session_state.run_history.insert(0, {
            "qp":            query[:28] + ("…" if len(query) > 28 else ""),
            "sources":       len(final_state.get("sources", [])),
            "claims":        len(final_state.get("key_claims", [])),
            "duration":      duration,
            "cost":          run_cost,
            "quality_score": qs,
            "quality_passed": final_state.get("quality_passed"),
            "review_score":  (final_state.get("review") or {}).get("score"),
            "revisions":     final_state.get("revision_count"),
        })

        render_sidebar()

        # ── Result summary bar ────────────────────────────────────
        if not run_error:
            srcs   = len(final_state.get("sources", []))
            claims = len(final_state.get("key_claims", []))
            score  = (final_state.get("review") or {}).get("score")

            st.markdown(
                f"<div class='result-bar'>"
                + _mcard(srcs,                             "Sources")
                + _mcard(claims,                           "Claims")
                + _mcard(f"{duration:.0f}s",               "Run time")
                + _mcard(f"${run_cost:.4f}",               "API cost")
                + _mcard(f"{qs:.2f}" if qs else "—",       "Quality")
                + _mcard(f"{score}/10" if score else "—",  "Review")
                + "</div>",
                unsafe_allow_html=True,
            )

        # ── Report ────────────────────────────────────────────────
        final_report = (
            final_state.get("final_report")
            or final_state.get("current_draft", "")
        )

        if final_report:
            st.markdown(
                f"<div class='report-card'>"
                f"<div class='report-label'>Query {n} — {query[:60]}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(final_report)
            st.markdown("</div>", unsafe_allow_html=True)

            # Save download for this run
            try:
                docx_path = write_report(final_state, run_id=final_state.get("run_id", ""))
                with open(docx_path, "rb") as fh:
                    docx_bytes = fh.read()
                safe = "".join(
                    c if (c.isalnum() or c == "_") else "_"
                    for c in query.lower()[:35].replace(" ", "_")
                ).strip("_") or "report"
                st.session_state.downloads.insert(0, {
                    "label":    f"Query {n}: {query[:40]}",
                    "bytes":    docx_bytes,
                    "filename": f"q{n}_{safe}.docx",
                })
            except Exception as e:
                st.caption(f"Could not generate .docx: {e}")

        elif not run_error:
            st.warning("Pipeline completed but produced no report.")

        # ── Pipeline trace ────────────────────────────────────────
        trace = final_state.get("pipeline_trace", [])
        if trace:
            with st.expander("🔎 Pipeline trace", expanded=False):
                try:
                    import pandas as pd
                    rows = [
                        {
                            "Agent":     NODE_MAP.get(t.get("agent",""), (t.get("agent",""),""))[0],
                            "Time (ms)": t.get("duration_ms", 0),
                            "Tokens":    t.get("tokens", 0),
                            "Summary":   t.get("summary", ""),
                        }
                        for t in trace
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                except ImportError:
                    for t in trace:
                        st.text(f"{t.get('agent',''):14s}  {t.get('duration_ms',0):5d}ms  {t.get('summary','')}")

    # ── Downloads section (outside results_area, persists across runs) ──
    if st.session_state.downloads:
        _render_downloads()


def _mcard(val, lbl):
    return (
        f"<div class='metric-card'>"
        f"<span class='val'>{val}</span>"
        f"<span class='lbl'>{lbl}</span>"
        f"</div>"
    )


def _render_downloads():
    st.divider()
    st.markdown(
        "<p style='font-size:.78rem;color:#475569;margin-bottom:.5rem;'>"
        "📄 Session Reports</p>",
        unsafe_allow_html=True,
    )
    for dl in st.session_state.downloads:
        st.download_button(
            label=f"⬇  {dl['label']}",
            data=dl["bytes"],
            file_name=dl["filename"],
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=dl["filename"],
        )


if __name__ == "__main__":
    main()
