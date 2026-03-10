# Multi-Agent Research Assistant

A production-grade research pipeline built with LangGraph and Claude.
Given a question, it decomposes, searches, analyses, synthesises, writes, and reviews — all autonomously.

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Planner          Decomposes query into 1-3 focused sub-topics  │
│      │                                                          │
│  fan_out_or_cache ──── Cache HIT ──► cache_loader              │
│      │                                      │                  │
│      └── Cache MISS ──► researcher (×3, parallel Send())       │
│                               │                                 │
│                         merge_research  (dedup + cache write)  │
│                               │                                 │
│                          quality_gate ── FAIL ──► retry        │
│                               │ PASS                            │
│                            analyst     Extract claims           │
│                               │                                 │
│                          synthesizer  Narrative synthesis       │
│                               │                                 │
│                            writer ◄────────────────────┐       │
│                               │                        │       │
│                           reviewer ── FAIL (≤2×) ──────┘       │
│                               │ PASS                            │
│                             END → .docx saved                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **LangGraph Send() fan-out** — three researchers run truly in parallel at the LangGraph level, not via Python threads
- **Two-level semantic cache** — exact SHA-256 match first, then cosine similarity (threshold 0.60) with `all-MiniLM-L6-v2` embeddings
- **Background cache writes** — embedding compute + SQLite write are fire-and-forget on a daemon thread so the pipeline never blocks
- **HF_HUB_OFFLINE=1** — set before model load to suppress all huggingface_hub network calls after first download
- **LLM-as-judge evaluation** — 4-dimension scoring (accuracy, completeness, citations, coherence) with configurable weights

---

## Quick Start

### Requirements
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### 1. Clone and install

```bash
git clone https://github.com/yourusername/multi-agent-research
cd multi-agent-research
uv sync
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   TAVILY_API_KEY=tvly-...
```

### 3. Run

```bash
# Streamlit UI (recommended)
uv run streamlit run app.py

# CLI
uv run python main.py "What is FAISS and how does it work?"

# CLI (interactive prompt)
uv run python main.py
```

---

## Streamlit UI

```bash
uv run streamlit run app.py
```

Opens at **http://localhost:8501**.

Features:
- Live step-by-step pipeline progress as each agent completes
- Real-time cost, token, and source counters in the sidebar
- Rendered final report with one-click `.docx` download
- Collapsible full pipeline trace table

---

## Evaluation

Compare multi-agent pipeline vs single-agent baseline on 10 test questions:

```bash
# Single question -- fast sanity check (~30s)
uv run python evaluation/run_eval.py --id faiss_overview --no-baseline

# Single question with baseline comparison
uv run python evaluation/run_eval.py --id faiss_overview

# All questions in one category
uv run python evaluation/run_eval.py --category ml_tools

# Full eval -- all 10 questions, both pipelines (~15-20 min, ~$0.10)
uv run python evaluation/run_eval.py

# List saved results
uv run python evaluation/run_eval.py --list

# Reload a previous run without re-running the pipeline
uv run python evaluation/run_eval.py --load 2026-03-10T20-30-34.json
```

Results are saved to `evaluation/results/{timestamp}.json`.

---

## Configuration

All behaviour is controlled by `configs/base.yaml` — no hardcoded values in Python files.

Key settings:

| Section | Key | Default | Description |
|---|---|---|---|
| `pipeline` | `max_sub_topics` | 3 | Parallel researcher branches |
| `pipeline` | `max_revisions` | 2 | Writer/reviewer loops |
| `pipeline` | `quality_threshold` | 0.4 | Quality gate pass score |
| `pipeline` | `review_pass_score` | 7 | Reviewer score to accept report |
| `cache` | `similarity_threshold` | 0.60 | Semantic cache match threshold |
| `cache` | `ttl_seconds` | 86400 | Cache TTL (1 day) |
| `budget` | `hard_limit` | 0.10 | Max USD per pipeline run |
| `evaluation` | `weights.accuracy` | 0.35 | Judge dimension weight |

---

## Project Structure

```
multi-agent-research/
├── app.py                        # Streamlit UI
├── main.py                       # CLI entry point
├── configs/
│   └── base.yaml                 # All pipeline settings
├── src/
│   ├── agents/
│   │   ├── graph.py              # LangGraph graph + routing
│   │   ├── planner.py
│   │   ├── researcher.py         # Send()-aware, single-topic
│   │   ├── quality_gate.py       # Pure Python heuristic scorer
│   │   ├── analyst.py
│   │   ├── synthesizer.py
│   │   ├── writer.py
│   │   └── reviewer.py
│   ├── cache/
│   │   └── research_cache.py     # Two-level semantic cache (SQLite)
│   ├── models/
│   │   └── state.py              # ResearchState TypedDict + Pydantic schemas
│   ├── observability/
│   │   ├── logger.py             # Background queue DB logger
│   │   ├── cost.py               # Token pricing + budget enforcement
│   │   └── db.py                 # SQLite schema
│   ├── output/
│   │   └── report_writer.py      # .docx generation
│   ├── tools/
│   │   ├── search.py             # Tavily wrapper
│   │   ├── wikipedia.py          # Wikipedia wrapper
│   │   └── tool_selector.py      # Route query to best tool
│   ├── config.py                 # YAML config loader
│   └── guardrails.py             # PII scrubbing
├── evaluation/
│   ├── questions.yaml            # 10 test questions + expected key points
│   ├── judge_prompt.py           # LLM-as-judge (4 dimensions)
│   ├── baseline.py               # Single-agent baseline pipeline
│   ├── run_eval.py               # CLI runner + comparison table
│   └── results/                  # Saved JSON results (gitignored)
└── .streamlit/
    ├── config.toml               # Theme + server settings
    └── secrets.toml.example      # Secrets template
```

---

## Deploy to Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "deploy"
   git push origin main
   ```

2. **Connect at [share.streamlit.io](https://share.streamlit.io)**
   - New app → select your repo → main branch → `app.py`

3. **Add secrets** (App settings → Secrets):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   TAVILY_API_KEY    = "tvly-..."
   ```

4. **Deploy** — Streamlit installs dependencies from `pyproject.toml` automatically via `uv`.

> **Note on the embedding model:** The first cold start downloads `all-MiniLM-L6-v2` (~80MB). Subsequent deploys use the cached version. The model download only happens once per Streamlit Cloud instance.

---

## Observed Performance

| Stage | First Run | Cached Run |
|---|---|---|
| Planner | ~2.4s | ~2.4s |
| Researcher ×3 (parallel) | ~1.1s wall clock | 0s |
| Quality Gate | <1ms | <1ms |
| Analyst | ~2.8s | ~2.8s |
| Synthesizer | ~3.5s | ~3.5s |
| Writer | ~4.2s | ~4.2s |
| Reviewer | ~5.6s | ~5.6s |
| **Total** | **~20s** | **~12s** |
| **Cost** | **~$0.005** | **~$0.003** |

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key |
| `TAVILY_API_KEY` | Yes | Tavily search API key |
| `DATABASE_URL` | No | Postgres URL (defaults to SQLite) |

---

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run a single eval question
uv run python evaluation/run_eval.py --id faiss_overview --no-baseline
```
