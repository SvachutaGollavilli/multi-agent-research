# Multi-Agent Research Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-orchestration-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Claude](https://img.shields.io/badge/Claude-Haiku-orange.svg)](https://www.anthropic.com/)
[![CI](https://github.com/krishnagollavilli/multi-agent-research/actions/workflows/ci.yml/badge.svg)](https://github.com/krishnagollavilli/multi-agent-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 7-agent orchestrated research pipeline (10 graph nodes) with two-level semantic cache, async streaming, intra-node concurrent search, full observability, per-agent cost tracking, and `.docx` report generation.

---

## Problem Statement

Single LLM calls produce mediocre research reports: they hallucinate citations, miss key perspectives, and have no quality verification. This project builds a **production-grade multi-agent system** where 7 specialized agents collaborate through shared state, quality-gated routing, parallel fan-out, iterative refinement, and a semantic cache — delivering verified, well-cited research reports with full cost transparency.

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
│      │                (semantic similarity ≥ 0.60)  │          │
│      └── Cache MISS ──► researcher × N (parallel Send())       │
│                               │                                 │
│                         merge_research  (dedup + cache write)   │
│                               │         (background thread)     │
│                          quality_gate ── FAIL ──► retry_counter │
│                               │ PASS         ──► planner (fan)  │
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

### Key LangGraph Patterns

| Pattern | Where | What It Demonstrates |
|---------|-------|----------------------|
| `Send()` parallel fan-out | Planner → Researchers | Concurrent agent execution with state merging |
| `operator.add` reducer | `sources`, `token_count`, `cost_usd` | Parallel-safe state accumulation |
| Conditional edges (pure Python) | Quality Gate, Reviewer | Routing without extra LLM calls |
| Iterative refinement loop | Writer ↔ Reviewer | Bounded loops with max revision guard |
| `with_structured_output()` | Planner, Analyst, Reviewer | Pydantic-validated agent outputs |
| Two-level semantic cache | `fan_out_or_cache` | SHA-256 exact + cosine similarity fallback |
| Background thread cache writes | `merge_research_node` | Non-blocking pipeline with daemon threads |
| Async graph streaming | `stream_pipeline_async()` | Real-time Streamlit updates via aqueue bridge |
| Per-agent cost tracking | All LLM agents | USD pricing, `RunCostAccumulator` |
| Observability DB | logger + db.py | SQLite/Postgres `pipeline_runs`, `agent_events`, `cost_ledger` |

---

## Results

### Single-Agent vs Multi-Agent Comparison

| Metric | Single-Agent | Multi-Agent | Improvement |
|--------|-------------|-------------|-------------|
| Accuracy (0–3) | 2.14 | 2.46 | **+15%** |
| Completeness (0–3) | 1.84 | 2.42 | **+32%** |
| Citations (0–3) | 1.55 | 2.15 | **+39%** |
| Avg Cost/Report | $0.000281 | $0.000721 | 2.6× |

**Verdict:** Multi-agent justified for synthesis tasks (+32% completeness). Route simple factual lookups to single-agent to save 2.6× cost.

---

## Setup

```bash
# 1. Create virtual environment and install
uv sync

# 2. Set up API keys
cp .env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY=sk-ant-...    (https://console.anthropic.com)
#   TAVILY_API_KEY=tvly-...         (https://app.tavily.com, free tier)
```

### Running

```bash
# Streamlit UI (recommended)
make ui
# or: uv run streamlit run app.py

# CLI
make run
# or: uv run python main.py "What is FAISS and how does it work?"

# Evaluation (single vs multi-agent comparison)
make evaluate
```

---

## Testing

This project has **110 automated tests** across 8 test files. All tests run **without API keys** — LLM calls are mocked.

```bash
make test
# or: uv run pytest tests/ -v
```

Expected output:
```
tests/test_e2e_pipeline.py    37 passed  - Full pipeline (mocked LLM)
tests/test_ui_smoke.py        26 passed  - Module imports
tests/test_guardrails.py      14 passed  - PII, injection, rate limiter
tests/test_graph.py           12 passed  - Graph routing + utility nodes
tests/test_quality_gate.py    12 passed  - Source scoring heuristics
tests/test_cache.py           17 passed  - Two-level cache operations
tests/test_state.py           12 passed  - State schema + Pydantic models
tests/test_tools.py           12 passed  - Search + async tools
tests/test_tool_selector.py    7 passed  - Tool routing
tests/test_config.py           7 passed  - YAML config loader
──────────────────────────────────────────
                              110+ passed
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
   - New app → your repo → main → `app.py`

3. **Add secrets** (App settings → Secrets):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   TAVILY_API_KEY    = "tvly-..."
   ```

4. **Deploy** — Streamlit Cloud installs from `pyproject.toml` via `uv` automatically.

> **Note on embedding model:** The first cold start downloads `all-MiniLM-L6-v2` (~80MB) for the semantic cache. Subsequent restarts use the cached version. This is handled automatically in `src/cache/research_cache.py` with `HF_HUB_OFFLINE=1`.

---

## What Makes This Production-Grade

### 1. Two-Level Semantic Cache
Not just SHA-256 matching — the cache uses `all-MiniLM-L6-v2` embeddings and cosine similarity (threshold 0.60) to match paraphrased queries:
- `"what is FAISS"` hits `"explain FAISS"` (cosine ≈ 0.76)
- Cache writes happen on a **daemon background thread** so the pipeline never blocks on the ~40ms embed + SQLite write.

### 2. Fully Async Pipeline
The graph runs via `graph.astream()` in a background thread bridged to Streamlit's synchronous main thread via a `queue.Queue`. Within each researcher node, Tavily and Wikipedia are queried **concurrently** using `asyncio.gather()`, saving 600–900ms per node.

### 3. Full Observability
Every LLM call writes to three SQLite/Postgres tables:
- `pipeline_runs` — run lifecycle, final report, total cost
- `agent_events` — per-agent start/end/duration/error
- `cost_ledger` — per-call token counts and USD cost

All DB writes are **fire-and-forget via a background queue drain thread** — agents never wait for DB.

### 4. Per-Agent Cost Tracking
Model pricing is defined in `cost.py`. Every agent calls `calculate_cost()` after each LLM call and accumulates into `RunCostAccumulator`. Budget has a soft warning and a hard limit that gracefully skips LLM calls rather than crashing.

### 5. Config-Driven Behavior
All thresholds, domain trust lists, model assignments, and budget limits live in `configs/base.yaml`. Zero code changes needed to:
- Swap Claude Haiku → Sonnet for specific agents
- Add/remove trusted domains
- Tune quality gate sensitivity
- Change max revisions or sub-topics

---

## Project Structure

```
multi-agent-research/
├── app.py                          # Streamlit UI (async streaming)
├── main.py                         # CLI entry point
├── Makefile                        # make run / test / ui / evaluate
├── configs/
│   └── base.yaml                   # All settings — model, budget, cache, QG
├── src/
│   ├── agents/
│   │   ├── graph.py                # LangGraph StateGraph (10 nodes)
│   │   ├── planner.py              # Query decomposition (structured output)
│   │   ├── researcher.py           # Async concurrent search (Send()-aware)
│   │   ├── quality_gate.py         # Pure Python heuristic scorer (no LLM)
│   │   ├── analyst.py              # Claim extraction with evidence linking
│   │   ├── synthesizer.py          # Cross-source narrative synthesis
│   │   ├── writer.py               # Versioned reports + PII scrubbing
│   │   └── reviewer.py             # Draft scoring + revision routing
│   ├── cache/
│   │   └── research_cache.py       # Two-level semantic cache (SQLite)
│   ├── models/
│   │   └── state.py                # ResearchState TypedDict + Pydantic schemas
│   ├── observability/
│   │   ├── logger.py               # Background queue DB logger
│   │   ├── cost.py                 # Token pricing + RunCostAccumulator
│   │   └── db.py                   # SQLite / PostgreSQL backend
│   ├── output/
│   │   └── report_writer.py        # .docx generation (python-docx)
│   ├── tools/
│   │   ├── async_search.py         # aiohttp Tavily + asyncio Wikipedia
│   │   ├── search.py               # Sync Tavily wrapper
│   │   ├── wikipedia.py            # Wikipedia wrapper
│   │   └── tool_selector.py        # Route query → best tool
│   ├── config.py                   # YAML singleton loader
│   └── guardrails.py               # Injection, PII, budget gate, rate limiter
├── evaluation/
│   ├── questions.yaml              # 10 test questions + expected key points
│   ├── judge_prompt.py             # LLM-as-judge (4 dimensions + weights)
│   ├── baseline.py                 # Single-agent baseline
│   └── run_eval.py                 # CLI comparison runner
├── tests/                          # 110 tests across 10 files
│   ├── test_e2e_pipeline.py        # Full pipeline, mocked LLM (37 tests)
│   ├── test_ui_smoke.py            # Module imports (26 tests)
│   ├── test_guardrails.py          # Guardrail unit tests (14 tests)
│   ├── test_graph.py               # Graph routing (12 tests)
│   ├── test_quality_gate.py        # Quality scoring (12 tests)
│   ├── test_cache.py               # Two-level cache (17 tests)
│   ├── test_state.py               # State schema (12 tests)
│   ├── test_tools.py               # Search + async tools (12 tests)
│   ├── test_tool_selector.py       # Tool routing (7 tests)
│   └── test_config.py              # Config loader (7 tests)
├── artifacts/results/
│   ├── sample_report.md            # Example generated research report
│   └── evaluation_results.md       # Single vs multi-agent comparison
└── .streamlit/
    └── config.toml                 # Dark theme + streaming settings
```

---

## Observed Performance

| Stage | First Run | Cached Run |
|---|---|---|
| Planner | ~2.3s | ~2.3s |
| Researcher ×3 (parallel, async) | ~1.0s wall clock | 0s |
| Quality Gate | <1ms | <1ms |
| Analyst | ~2.9s | ~2.9s |
| Synthesizer | ~3.1s | ~3.1s |
| Writer | ~4.2s | ~4.2s |
| Reviewer | ~2.7s | ~2.7s |
| **Total** | **~18s** | **~10s** |
| **Cost** | **~$0.00072** | **~$0.00043** |

Researcher wall-clock time is ~1s even for 3 parallel instances because Tavily + Wikipedia are queried concurrently within each researcher using `asyncio.gather()`.

---

## Interview Guide

**Q: How many agents and nodes are in this system?**
> 7 agent functions across 10 graph nodes: planner, researcher, merge_research, cache_loader, quality_gate, retry_counter, analyst, synthesizer, writer, reviewer. The retry flow reuses the researcher function, so there are 10 nodes but only 7 unique agent implementations.

**Q: How does the two-level cache work?**
> Level 1 is an exact SHA-256 hash of the normalized query — O(1), instant. Level 2 is semantic similarity using `all-MiniLM-L6-v2` embeddings and cosine similarity at threshold 0.60. This catches paraphrased queries ("what is FAISS" → matches "explain FAISS" at cosine 0.76). Cache writes are dispatched to a daemon background thread from `merge_research_node` — the pipeline never blocks on the ~40ms embed+write.

**Q: How does parallel execution work in LangGraph?**
> The planner decomposes the query into 1-3 sub-topics. `fan_out_or_cache` creates `Send("researcher", {**state, "current_topic": topic})` objects — one per sub-topic. LangGraph dispatches these in parallel. Within each researcher, Tavily and Wikipedia are queried concurrently via `asyncio.gather()` — so there are two levels of parallelism. Sources merge via `Annotated[list[dict], operator.add]`.

**Q: Why use a background thread for DB writes?**
> The observability logger uses a `queue.Queue` + daemon drain thread. Agents call `_enqueue(item)` which is non-blocking — if the queue is full (DB slow), items are dropped rather than blocking agents. This keeps DB latency out of the critical path. The same pattern is used for cache writes: `merge_research_node` dispatches cache storage on a daemon thread.

**Q: How does cost tracking work?**
> Every agent calls `calculate_cost(model, input_tokens, output_tokens)` after each LLM call. Token counts come from `response.usage_metadata`. The `RunCostAccumulator` accumulates per-agent costs and checks against configurable soft ($0.08) and hard ($0.10) USD limits. `cost_usd` uses `operator.add` in the state so it accumulates safely across parallel agents.

**Q: How do you prevent the reviewer loop from running forever?**
> Two guards: (1) the reviewer scores 1-10, and the pipeline only loops if `score < review_pass_score (7)` from config; (2) `revision_count` is capped at `max_revisions=2` in base.yaml. The `_should_revise` routing function checks both: `score >= pass_score OR revision_count >= max_revisions → END`.

**Q: How is the Streamlit UI updated in real-time if the graph is async?**
> `stream_pipeline_async()` bridges async → sync: it runs `graph.astream()` inside a daemon thread with its own event loop, and passes node completion events back to Streamlit's synchronous main thread via a `queue.Queue`. The Streamlit thread drains this queue in a `while True` loop, yielding `(node_name, update, accumulated_state)` for each completed node.

**Q: What would you change for production deployment?**
> (1) Swap SQLite cache for Redis for concurrent multi-user access; (2) Use Postgres (already supported via `DATABASE_URL` + psycopg3) for the observability DB; (3) Add LangSmith tracing for distributed tracing across runs; (4) Implement human-in-the-loop checkpoints for low-confidence claims; (5) Route simple factual queries to single-agent (2.6× cheaper based on evaluation).

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Claude API key (console.anthropic.com) |
| `TAVILY_API_KEY` | Yes | Tavily search API (app.tavily.com, free tier) |
| `DATABASE_URL` | No | Postgres URL (defaults to `sqlite:///data/pipeline.db`) |

---

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Send() API](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [Claude API — Anthropic](https://docs.anthropic.com/)
- [Tavily Search API](https://tavily.com/)
- [sentence-transformers: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
