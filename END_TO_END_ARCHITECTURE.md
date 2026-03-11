# End-to-End Architecture: Multi-Agent Research System

> A production-grade 10-node LangGraph pipeline with two-level semantic cache,
> fully async streaming, per-agent cost tracking, and SQLite/PostgreSQL observability.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Complete Request Flows](#3-complete-request-flows)
   - 3.1 Happy Path (Cache Miss)
   - 3.2 Cache Hit Path
   - 3.3 Quality Gate Retry Path
   - 3.4 Reviewer Refinement Loop
4. [Sequence Diagrams](#4-sequence-diagrams)
5. [State Schema (24 Fields)](#5-state-schema-24-fields)
6. [Agent Deep-Dives](#6-agent-deep-dives)
   - 6.1 Planner
   - 6.2 Researcher (async concurrent)
   - 6.3 Merge Research
   - 6.4 Quality Gate
   - 6.5 Analyst
   - 6.6 Synthesizer
   - 6.7 Writer
   - 6.8 Reviewer
   - 6.9 Cache Loader
   - 6.10 Retry Counter
7. [Two-Level Semantic Cache](#7-two-level-semantic-cache)
8. [Async Architecture](#8-async-architecture)
9. [Observability: Database Schema](#9-observability-database-schema)
10. [Per-Agent Cost Tracking](#10-per-agent-cost-tracking)
11. [Guardrails (4 Layers)](#11-guardrails-4-layers)
12. [Tool Selection Logic](#12-tool-selection-logic)
13. [YAML-Driven Configuration](#13-yaml-driven-configuration)
14. [Testing Architecture (110+ Tests)](#14-testing-architecture-110-tests)
15. [Performance Profile](#15-performance-profile)
16. [Security Model (OWASP LLM Top 10)](#16-security-model-owasp-llm-top-10)
17. [Deployment](#17-deployment)
18. [Interview Q&A](#18-interview-qa)

---

## 1. System Overview

Single LLM calls produce mediocre research reports: they hallucinate citations,
miss key perspectives, and have no quality verification loop. This system solves
those problems by splitting the research pipeline into 7 specialised agent
functions (10 graph nodes) where each agent handles a distinct failure mode.

**Key differentiators over a naive single-agent pipeline:**

| Concern | Single-Agent | This System |
|---|---|---|
| Citation hallucination | Common (no grounding) | Blocked — writer constrained to `state.sources` only |
| Coverage | Single perspective | 1–3 parallel research threads |
| Quality control | None | Pure-Python quality gate + reviewer loop |
| Repeated API calls | Every run hits the API | Two-level semantic cache (SHA-256 + cosine) |
| Cost visibility | Black box | Per-agent USD tracking + hard budget cap |
| Observability | None | SQLite/Postgres tables for every run and LLM call |
| Latency | Sequential | Concurrent search (asyncio.gather), parallel fan-out (Send()) |

---

## 2. Architecture Diagram

```
User query
    │
    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Guardrails Layer (entry)                                              │
│  ① Prompt injection detection (OWASP LLM Top 10 #1)                   │
│  ② PII scrub on all LLM outputs                                        │
│  ③ Token budget gate ($0.08 soft / $0.10 hard)                         │
│  ④ Token-bucket rate limiter (50 RPM)                                  │
└────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────┐
│  Planner  │  with_structured_output(PlannerOutput)
│           │  Decomposes query → 1-3 focused sub-topics
└─────┬─────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  fan_out_or_cache (routing function — read-only)                     │
│                                                                      │
│  cache_fetch(query)                                                  │
│    ├── HIT  (exact SHA-256 or cosine ≥ 0.60) ──────► cache_loader   │
│    └── MISS ──► [Send("researcher", {topic_1}),                      │
│                  Send("researcher", {topic_2}),   parallel fan-out   │
│                  Send("researcher", {topic_3})]                      │
└──────────────────────────────────────────────────────────────────────┘
      │                              │
      ▼                              ▼
┌─────────────┐          ┌───────────────────────────────────────┐
│ cache_loader│          │  researcher × N  (parallel via Send)  │
│             │          │  asyncio.gather(Tavily, Wikipedia)     │
│ Loads cached│          │  → {title, url, content} per source   │
│ sources into│          └───────────────────────────────────────┘
│ state       │                         │
└──────┬──────┘                         ▼
       │                    ┌───────────────────────┐
       │                    │  merge_research_node  │
       │                    │  URL dedup            │
       │                    │  Background thread:   │
       │                    │  cache_store(sources) │
       │                    └──────────┬────────────┘
       │                               │
       └───────────────────────────────┤
                                       ▼
                           ┌───────────────────────┐
                           │    quality_gate        │
                           │    Pure Python (<1ms)  │
                           │    domain_trust_score  │
                           │  + snippet_quality     │
                           │  = composite score     │
                           └───────────┬───────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                    score < 0.4                  score ≥ 0.4
                    retries < max                     │
                         │                           │
                         ▼                           ▼
               ┌──────────────────┐        ┌────────────────┐
               │  retry_counter   │        │    analyst     │
               │  retries += 1    │        │                │
               │  force_research  │        │ with_structured│
               │  = True          │        │ _output(       │
               │  sources = []    │        │ AnalystOutput) │
               └────────┬─────────┘        │ 5 claims +     │
                        │                  │ conflicts      │
                        │ (loops back      └───────┬────────┘
                        │  to fan_out)             │
                        │                          ▼
                        │                 ┌────────────────┐
                        │                 │  synthesizer   │
                        │                 │                │
                        │                 │ Theme grouping │
                        │                 │ Conflict reso. │
                        │                 │ Source ranking │
                        │                 └───────┬────────┘
                        │                         │
                        │                         ▼
                        │                ┌─────────────────┐
                        │                │     writer      │◄──────────────┐
                        │                │                 │               │
                        │                │ v1: full report │               │
                        │                │ v2: revision    │               │
                        │                │ PII scrub on    │               │
                        │                │ output          │               │
                        │                └────────┬────────┘               │
                        │                         │                        │
                        │                         ▼                        │
                        │                ┌─────────────────┐               │
                        │                │    reviewer     │               │
                        │                │                 │               │
                        │                │ with_structured │               │
                        │                │ _output(Review  │               │
                        │                │ Output)         │               │
                        │                │ score 1-10      │               │
                        │                └────────┬────────┘               │
                        │                         │                        │
                        │              ┌──────────┴──────────┐             │
                        │              │                     │             │
                        │         score ≥ 7           score < 7 AND        │
                        │         OR max revisions    revision_count < 2 ──┘
                        │              │                     (loops back)
                        │              ▼
                        │           ┌─────┐
                        └──────────►│ END │
                                    │     │
                                    │ .docx saved
                                    │ DB updated
                                    └─────┘
```

---

## 3. Complete Request Flows

### 3.1 Happy Path (Cache Miss, Quality Pass, Review Pass)

```
Step  Agent              Action
────  ─────────────────  ────────────────────────────────────────────────────
 1    planner            Receives query "What is FAISS?"
                         Invokes Claude Haiku via with_structured_output()
                         Returns: sub_topics=["FAISS algorithm",
                                              "FAISS index types",
                                              "FAISS GPU support"]

 2    fan_out_or_cache   cache_fetch("What is FAISS?") → None (MISS)
                         Creates 3 Send() objects, one per sub-topic
                         LangGraph dispatches them in parallel

 3    researcher ×3      Each researcher receives its sub-topic via current_topic
                         Calls select_tool(topic) → "both"
                         asyncio.gather(
                           async_search_web(topic),       # Tavily via aiohttp
                           async_search_wikipedia(topic)  # asyncio.to_thread
                         )
                         Returns: sources (each list gets operator.add'd)

 4    merge_research     Global URL dedup across all 3 parallel results
                         Dispatches cache_store() on daemon background thread
                         Returns: deduplicated sources list

 5    quality_gate       Scores each source: domain_trust + snippet_quality
                         Composite = 0.5×domain + 0.5×snippet → 0.72
                         0.72 ≥ 0.40 threshold → quality_passed=True

 6    analyst            Caps to 5 sources (reduces latency/cost)
                         Extracts 5 claims with confidence + evidence
                         Returns: key_claims, conflicts (if any)

 7    synthesizer        Groups claims into themes
                         Writes 150-200 word narrative prose
                         Ranks sources by high-confidence claim count
                         Returns: synthesis, source_ranking

 8    writer             Builds prompt from synthesis (preferred) + claims
                         Writes markdown report (250-350 words)
                         PII scrubbed via check_output()
                         Returns: current_draft v1, drafts=[{version:1, ...}]

 9    reviewer           Reviews draft against research question
                         Score 8/10 → passed=True (≥ threshold of 7)
                         Returns: review={score:8, issues:[], passed:True}

10    _should_revise     score=8 ≥ pass_score=7 → route to "end"

11    END                write_report() saves .docx to results/
                         DB: pipeline_runs.status = "completed"
                         DB: cost_ledger rows written by drain thread
```

### 3.2 Cache Hit Path

```
Step  Agent              Action
────  ─────────────────  ────────────────────────────────────────────────────
 1    planner            Same as happy path

 2    fan_out_or_cache   cache_fetch(query)
                         Level 1: SHA-256 hash → exact match found
                         OR
                         Level 2: embed(query) → cosine ≥ 0.60 → semantic match
                         Returns: "cache_loader"

 3    cache_loader       Loads cached sources directly into state
                         Writes pipeline_trace entry with "Cache HIT"
                         Skips all researcher nodes entirely

 4    quality_gate       Scores cached sources
                         (Cached sources already passed once; likely still pass)

 5..9  (same as happy path from quality_gate onward)
```

**Speedup:** Researcher wall clock ~1s × 3 parallel = ~1s skipped.
Plus background thread cache write cost = 0 (pipeline never waits for it).

### 3.3 Quality Gate Retry Path

```
Step  Agent              Action
────  ─────────────────  ────────────────────────────────────────────────────
 1    planner            Decomposes query

 2    fan_out_or_cache   Cache miss → fan-out

 3    researcher ×N      Returns low-quality sources (e.g. reddit, quora only)

 4    merge_research     Deduplicates, stores in background

 5    quality_gate       Composite score = 0.25 < threshold 0.40
                         quality_passed = False
                         quality_retries = 0 < max_quality_retries = 1

 6    _should_retry      "retry_counter" (retries available)

 7    retry_counter      quality_retries = 1
                         force_research  = True (bypasses cache on retry)
                         sources         = []  (clears stale sources)

 8    fan_out_or_cache   force_research=True → skips cache_fetch
                         Fans out to fresh researchers

 9    researcher ×N      Fresh search (different results this time)

10    merge_research     Merges new sources

11    quality_gate       Composite score = 0.61 ≥ threshold → PASS
                         quality_retries = 1 = max_quality_retries

12..END (continues normally from analyst)
```

**Guard against infinite loop:** `quality_retries >= max_quality_retries` forces
routing to analyst regardless of score. Pipeline always terminates.

### 3.4 Reviewer Refinement Loop

```
Step  Agent         Action
────  ────────────  ───────────────────────────────────────────────────────
 8    writer v1     Produces first draft

 9    reviewer      Score 5/10 — issues: ["too brief", "missing GPU section"]
                    passed=False, revision_count=1 < max_revisions=2

10    _should_revise  score=5 < 7, revision_count=1 < 2 → "revise"

11    writer v2     Receives review.issues + review.suggestions
                    Rewrites with targeted fixes
                    revision_count=2

12    reviewer      Score 8/10 — passed=True

13    _should_revise  score=8 ≥ 7 → "end"

14    END
```

**Guard against infinite loop:** Two hard caps prevent runaway costs:
1. `revision_count >= max_revisions` routes to END regardless of score.
2. `review_pass_score=7` is configurable in base.yaml without code changes.

---

## 4. Sequence Diagrams

### 4.1 Happy Path

```
User   Streamlit   graph.py   planner   researcher×3   merge   qgate   analyst   synth   writer   reviewer   DB
 │        │           │          │           │           │        │       │         │        │          │       │
 │──run──►│           │          │           │           │        │       │         │        │          │       │
 │        │──stream──►│          │           │           │        │       │         │        │          │       │
 │        │           │──invoke─►│           │           │        │       │         │        │          │       │
 │        │           │◄─return──│           │           │        │       │         │        │          │       │
 │        │           │──Send()──────────────►│          │        │       │         │        │          │       │
 │        │           │          │    (×3 parallel)      │        │       │         │        │          │       │
 │        │           │          │    asyncio.gather     │        │       │         │        │          │       │
 │        │           │◄─sources─────────────┤           │        │       │         │        │          │       │
 │        │◄──node────│          │           │           │        │       │         │        │          │       │
 │        │           │──merge──────────────────────────►│        │       │         │        │          │       │
 │        │           │◄─deduped───────────────────────── │        │       │         │        │          │       │
 │        │◄──node────│          │           │            │        │       │         │        │          │       │
 │        │           │──score──────────────────────────────────► │       │         │        │          │       │
 │        │           │◄─0.72 PASS──────────────────────────────── │       │         │        │          │       │
 │        │◄──node────│          │           │            │        │       │         │        │          │       │
 │        │           │──claims─────────────────────────────────────────► │         │        │          │       │
 │        │           │◄─5 claims──────────────────────────────────────── │         │        │          │       │
 │        │◄──node────│          │           │            │        │       │         │        │          │       │
 │        │           │──synth──────────────────────────────────────────────────── ►│        │          │       │
 │        │           │◄─narrative──────────────────────────────────────────────── │        │          │       │
 │        │◄──node────│          │           │            │        │       │         │        │          │       │
 │        │           │──draft──────────────────────────────────────────────────────────────►│          │       │
 │        │           │◄─v1 report──────────────────────────────────────────────────────────│          │       │
 │        │◄──node────│          │           │            │        │       │         │        │          │       │
 │        │           │──review─────────────────────────────────────────────────────────────────────── ►│       │
 │        │           │◄─8/10 PASS─────────────────────────────────────────────────────────────────── │       │
 │        │◄──node────│          │           │            │        │       │         │        │          │       │
 │        │           │──────────────────────────────────────────────────────────────────────────────────────► DB
 │        │◄─report───│
```

### 4.2 Async Bridge (Streamlit ↔ graph.astream)

```
Streamlit main thread (sync)         Background daemon thread (async)
──────────────────────────           ──────────────────────────────────────
stream_pipeline_async(query)
  │
  ├─ Queue() created
  ├─ Thread(_thread_target).start()   asyncio.run(_astream())
  │                                     │
  │                                     ├─ graph.astream(initial)
  │                                     │
  │                                     ├─ Node "planner" completes
  │                                     │   event_queue.put(("event", {...}))
  │◄──────────────── event_queue.get()──┘
  │  yield ("planner", update, state)
  │
  ├─ st.status.write("Completed: Planner")
  ├─ render_step_log(steps)
  ├─ render_live_state(accumulated)
  │
  │                                     ├─ Node "researcher" completes
  │◄──────────────── event_queue.get()──┘
  │  yield ("researcher", update, state)
  │  ...
  │
  │                                     ├─ event_queue.put(("done", None))
  │◄──────────────── event_queue.get()──┘
  │  (kind == "done") → break
  │
  └─ end_run(), write_report()
```

---

## 5. State Schema (24 Fields)

`ResearchState` is a `TypedDict` with `total=False` (all fields optional at init,
required by the end). Fields annotated with `Annotated[T, operator.add]` are
**parallel-safe reducers**: when multiple `Send()` branches write to them, LangGraph
concatenates the values rather than overwriting.

```python
class ResearchState(TypedDict, total=False):

    # ── Run metadata (set at entry, never modified) ───────────────────────
    run_id:  str       # UUID4, used in all DB writes
    query:   str       # original user question — immutable throughout pipeline

    # ── Planner output ────────────────────────────────────────────────────
    sub_topics:    list[str]   # 1-3 focused sub-questions
    research_plan: str         # brief strategy description

    # ── Send() fan-out control fields ─────────────────────────────────────
    current_topic:  str    # injected per-Send(); which sub-topic to search
    force_research: bool   # True after quality gate failure — bypasses cache

    # ── Researcher output (parallel-safe via operator.add) ────────────────
    # Three parallel researchers each return a list of sources.
    # LangGraph's operator.add reducer concatenates all three lists
    # before merge_research_node runs, so merge sees the full combined list.
    sources:             Annotated[list[dict], operator.add]
    search_queries_used: Annotated[list[str],  operator.add]

    # Source dict shape: {title: str, url: str, content: str}

    # ── Quality gate output ───────────────────────────────────────────────
    quality_score:   float   # composite 0.0-1.0
    quality_passed:  bool    # composite >= quality_threshold
    quality_retries: int     # how many times we've retried

    # ── Analyst output ────────────────────────────────────────────────────
    key_claims: list[dict]   # [{claim, source_idx, confidence, evidence}]
    conflicts:  list[dict]   # [{description}]

    # ── Synthesizer output ────────────────────────────────────────────────
    synthesis:      str          # 150-200 word narrative prose
    source_ranking: list[dict]   # [{source_idx, title, url, high_conf_claims}]

    # ── Writer output ─────────────────────────────────────────────────────
    drafts:         list[dict]   # [{version, content, char_count, pii_scrubbed}]
    current_draft:  str          # latest draft (writer overwrites each revision)
    revision_count: int          # number of writer invocations

    # ── Reviewer output ───────────────────────────────────────────────────
    review: dict                 # {score, issues, suggestions, passed}

    # ── Final output ──────────────────────────────────────────────────────
    final_report: str            # currently same as current_draft after END

    # ── Observability (parallel-safe via operator.add) ────────────────────
    pipeline_trace: Annotated[list[dict], operator.add]
    # trace dict: {agent, duration_ms, tokens, summary}

    errors:      Annotated[list[str], operator.add]
    token_count: Annotated[int,       operator.add]   # accumulates across all agents
    cost_usd:    Annotated[float,     operator.add]   # accumulates across all agents
```

### Why `operator.add` matters for parallel execution

Without it, if researchers 1, 2, and 3 each write `state["sources"] = [...]`,
LangGraph would apply them sequentially and only the last writer's sources
would survive. With `Annotated[list[dict], operator.add]`, LangGraph calls
`operator.add(existing, new)` which is just list concatenation — so all three
researchers' results accumulate correctly in a single list before
`merge_research_node` sees them.

Same principle applies to `token_count` and `cost_usd`: parallel agents
accumulate them without race conditions because `operator.add` is applied
atomically by LangGraph's state merge logic.

---

## 6. Agent Deep-Dives

### 6.1 Planner

**Role:** Query decomposition  
**LLM:** Claude Haiku (claude-haiku-4-5-20251001)  
**Pattern:** `with_structured_output(PlannerOutput, include_raw=True)`

```
Input:  query = "What is FAISS?"
Output: PlannerOutput {
  sub_topics:    ["FAISS algorithm and data structures",
                  "FAISS index types IVF HNSW",
                  "FAISS performance and GPU support"]
  research_plan: "Search FAISS technical docs, Meta engineering blog,
                  and academic papers."
}
```

The `include_raw=True` flag returns `{"raw": AIMessage, "parsed": PlannerOutput}`.
The `raw` message carries `usage_metadata` (token counts) needed for cost tracking.
The `parsed` field is the Pydantic object with schema validation.

**Error fallback:** If structured parsing fails, returns `sub_topics=[query]`
(single-topic direct research) to keep the pipeline moving.

---

### 6.2 Researcher (async concurrent)

**Role:** Single-topic web research  
**LLM:** None (search tools only)  
**Pattern:** `Send()` fan-out + `asyncio.gather`

Each researcher receives `current_topic` from its `Send()` payload. It calls
`select_tool(current_topic)` to choose Tavily, Wikipedia, or both, then
runs them concurrently:

```python
results = await asyncio.gather(
    async_search_web(topic, max_results=5),      # Tavily via aiohttp
    async_search_wikipedia(topic, max_wiki=3),   # asyncio.to_thread (sync lib)
)
```

**Two parallelism levels:**
1. **Graph level:** LangGraph dispatches 3 researcher nodes simultaneously via `Send()`
2. **Intra-node level:** Each researcher queries Tavily + Wikipedia concurrently

For 3 sub-topics with "both" tool selection, the combined speedup vs sequential:
- Sequential would be: 3 topics × 2 tools × ~0.5s = ~3s
- Parallel (both levels): max(~0.5s × 3) = ~1s wall clock

**Sync/async bridge:** The LangGraph graph nodes are called synchronously.
The researcher bridges to async via:
```python
def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        # Already in async context (arun_pipeline) — use thread pool
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop (CLI, eval) — safe to call directly
        return asyncio.run(coro)
```

---

### 6.3 Merge Research

**Role:** Post-fan-out dedup + background cache write  
**LLM:** None (pure Python)

By the time this node runs, LangGraph has already applied `operator.add` to
combine all parallel researchers' sources. This node:

1. **URL dedup:** Iterates the combined list, tracking seen URLs. First occurrence
   wins. Removes duplication from overlapping search results across sub-topics.

2. **Background cache write:** Starts a daemon thread to call `cache_store()`.
   The pipeline never waits — the ~40ms embed+SQLite write happens asynchronously.

```python
t = threading.Thread(target=cache_store, args=(query, unique, "parallel_send"),
                     daemon=True, name="cache-write")
t.start()
# pipeline continues immediately
```

Why daemon=True? If the main process exits before the thread finishes, the thread
is killed automatically. We never want a cache write to delay pipeline shutdown.

---

### 6.4 Quality Gate

**Role:** Source quality scoring  
**LLM:** None (pure Python, < 1ms)  
**No API cost**

Scores two dimensions per source:

**Domain trust** (lookup from `configs/base.yaml`):
```
arxiv.org, nature.com, ieee.org → 1.0  (high trust)
medium.com, github.com          → 0.6  (medium trust)
reddit.com, quora.com           → 0.2  (low trust)
unknown domain                  → 0.4  (neutral)
*.gov, *.edu                    → 1.0  (automatic high trust)
```

Subdomain matching: `cs.mit.edu` matches `mit.edu` via `.endswith("." + trusted)`.

**Snippet quality** (length + boilerplate detection):
```python
# Length score: linear interpolation between min_chars=100 and max_chars=300
if length < 100:     length_score = 0.0
elif length >= 300:  length_score = 1.0
else:                length_score = (length - 100) / 200

# Boilerplate penalty: -0.3 per phrase found
# Phrases: "click here", "sign up", "cookie policy", "advertisement", ...
score = max(0.0, length_score - (penalties_hit × 0.3))
```

**Composite:**
```
composite = (avg_domain × 0.5) + (avg_snippet × 0.5)
```

**Routing (in graph.py, NOT in quality_gate.py):**
```
composite ≥ 0.40 → analyst
composite <  0.40 AND retries < 1 → retry_counter
composite <  0.40 AND retries ≥ 1 → analyst (exhausted, proceed anyway)
```

Routing is separated from scoring deliberately: the agent only writes to state,
the `_should_retry_research` routing function reads state. This keeps agents
side-effect-free (they never make routing decisions) which makes the graph
easier to test and reason about.

---

### 6.5 Analyst

**Role:** Claim extraction from sources  
**LLM:** Claude Haiku  
**Pattern:** `with_structured_output(AnalystOutput, include_raw=True)`

**Input cap:** Only the first 5 sources are sent to the LLM, with content
truncated to 150 chars each. This bounds input tokens to ~750 chars regardless
of how many sources the researcher returned.

```
AnalystOutput {
  claims: [
    ClaimOutput {
      claim:      "FAISS supports billion-scale vector search",
      source_idx: 1,            # 1-based index into state.sources
      confidence: "high",       # "high" | "medium" | "low"
      evidence:   "Facebook engineering blog: 'sets of vectors of any size'"
    },
    ...  (5-8 claims)
  ]
  conflicts: ["Source 2 claims O(log n) lookup; Source 3 claims O(1) with IVF"]
}
```

**Retry on parse failure:** If `raw_result["parsed"]` is None (intermittent schema
mismatch from Haiku), retries once before returning empty claims. This prevents
one bad LLM response from killing the pipeline.

---

### 6.6 Synthesizer

**Role:** Narrative synthesis from claims  
**LLM:** Claude Haiku

Unlike the analyst (which extracts facts in isolation), the synthesizer's job is
to "think through" the claims and produce a connected prose narrative:

- Groups claims by theme (what are the 2-3 central ideas?)
- Connects related claims ("FAISS supports GPU AND scales to billions — this is because...")
- Explicitly acknowledges conflicts ("Source A says O(log n), Source B says O(1) — the discrepancy is index type dependent")
- Ends with a one-sentence answer to the research question

**Source ranking:** The synthesizer also ranks sources by how many high-confidence
claims cite them. This gives the writer a signal about which sources are most
authoritative so they can lead citations with the strongest evidence.

---

### 6.7 Writer

**Role:** Markdown research report  
**LLM:** Claude Haiku  
**Two modes:** initial draft (v1) vs revision (v2+)

**Draft v1 prompt** uses `synthesis` (preferred) as the backbone, with `key_claims`
for detail and `source_ranking` for citation ordering:
```
## Executive Summary  (2-3 sentences)
## Key Findings       (4-6 bullet points)
## Analysis           (2-3 sentences drawing on synthesis)
## Conclusion         (1-2 sentences answering the research question)
## Sources            (numbered list from source_ranking)
```

**Revision prompt** (v2) is targeted: passes `review.issues` and
`review.suggestions` with the existing draft. The writer does not re-read
sources on revision — it fixes specific issues, not the whole report.

**PII scrubbing** is applied to every draft via `check_output()` before storing
in state. If an LLM accidentally includes email-like patterns or phone numbers,
they are replaced with `[REDACTED_EMAIL]` etc.

---

### 6.8 Reviewer

**Role:** Draft quality scoring  
**LLM:** Claude Haiku  
**Pattern:** `with_structured_output(ReviewOutput, include_raw=True)`

```
ReviewOutput {
  score:       8,                    # 1-10
  issues:      ["GPU section thin"],
  suggestions: ["Add IVF index example with code"],
  passed:      True                  # score >= review_pass_score (7)
}
```

The reviewer enforces `passed = (score >= review_pass_score)` from config,
regardless of what the LLM returns in the `passed` field. This prevents the LLM
from being overly lenient with its own `passed` judgment.

**Retry on parse failure:** Like the analyst, retries once. On second failure,
returns `score=pass_score, passed=True` — a neutral "pass" that prevents the
reviewer from blocking the pipeline indefinitely.

---

### 6.9 Cache Loader

**Role:** Write cached sources into state  
**LLM:** None

This is a utility node that exists because LangGraph routing functions are
read-only (they cannot write to state). `fan_out_or_cache` detects the cache
hit and returns the string `"cache_loader"`, but cannot itself write the sources
into state. The cache_loader node does that write.

```python
def cache_loader_node(state: ResearchState) -> dict:
    cached = cache_fetch(state["query"])
    return {"sources": cached or [], ...}
```

---

### 6.10 Retry Counter

**Role:** Increment retries, set force_research, clear stale sources  
**LLM:** None

Another utility node for the same reason as cache_loader: routing functions
cannot write state. When `_should_retry_research` returns `"retry_counter"`,
this node updates the three fields needed for the retry:

```python
return {
    "quality_retries": state.get("quality_retries", 0) + 1,
    "force_research":  True,    # ensures fan_out bypasses cache
    "sources":         [],      # clears stale low-quality sources
}
```

After this node, `fan_out_or_cache` is called again (via conditional edge from
retry_counter → fan_out_or_cache). The `force_research=True` flag ensures we
do fresh research instead of hitting the same cached low-quality results.

---

## 7. Two-Level Semantic Cache

The cache sits in `src/cache/research_cache.py` and uses a SQLite database at
`data/research_cache.db`.

### Schema

```sql
CREATE TABLE query_cache (
    query_hash  TEXT PRIMARY KEY,   -- SHA-256 of normalized query
    query       TEXT NOT NULL,      -- original query string
    results     TEXT NOT NULL,      -- JSON: list of {title, url, content}
    tool_used   TEXT NOT NULL,      -- "tavily" | "wikipedia" | "both" | "parallel_send"
    created_at  REAL NOT NULL,      -- Unix timestamp (for TTL enforcement)
    hit_count   INTEGER DEFAULT 0,  -- how many times this entry was served
    embedding   TEXT DEFAULT NULL   -- JSON: float[] from all-MiniLM-L6-v2
);

CREATE INDEX idx_cache_created ON query_cache(created_at);
```

### Level 1: Exact Match (SHA-256)

```python
key = hashlib.sha256(query.strip().lower().encode()).hexdigest()
row = conn.execute(
    "SELECT results, created_at FROM query_cache WHERE query_hash = ?",
    (key,)
).fetchone()
```

- O(1) — primary key lookup
- Catches exact repeats and case-normalized variants
- Zero compute — no embedding needed
- TTL enforced: `created_at >= time.time() - ttl_seconds (86400)`

### Level 2: Semantic Similarity (cosine)

When exact match fails:
1. Embed the incoming query with `all-MiniLM-L6-v2` (384-dim vectors)
2. Scan all fresh cache entries that have embeddings
3. Compute cosine similarity with each
4. Return best match if score ≥ threshold (0.60)

```python
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

**Example matches at threshold 0.60:**
```
"what is FAISS"   → "explain FAISS"          cosine ≈ 0.76  ✓ HIT
"how FAISS works" → "FAISS overview"         cosine ≈ 0.68  ✓ HIT
"what is FAISS"   → "what is Docker"         cosine ≈ 0.31  ✗ MISS
"FAISS algorithm" → "FAISS typo in qurey"    cosine ≈ 0.64  ✓ HIT
```

**Threshold guide:**
- 0.85 — too tight, misses most paraphrases
- 0.60 — sweet spot: catches paraphrases, word order, minor typos
- 0.45 — too loose, risks cross-topic matches
- < 0.40 — cosine of unrelated short queries can reach 0.4, don't use

### HF Hub Offline Mode

The model download is prevented on every restart via `HF_HUB_OFFLINE=1`:

```python
os.environ["HF_HUB_OFFLINE"] = "1"
try:
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")  # uses local cache
except Exception:
    # First-ever run: clear flag, download once (~80MB), then set flag back
    del os.environ["HF_HUB_OFFLINE"]
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    os.environ["HF_HUB_OFFLINE"] = "1"
```

`local_files_only=True` alone is insufficient in sentence-transformers v3: the
library routes through `huggingface_hub` which validates its cache via HEAD
requests regardless of that flag. `HF_HUB_OFFLINE=1` is the env var that
actually prevents all network calls.

### Background Thread Cache Writes

```python
# In merge_research_node:
t = threading.Thread(
    target=cache_store,
    args=(query, unique, "parallel_send"),
    daemon=True,
    name="cache-write",
)
t.start()
# pipeline returns immediately, never awaits the thread
```

Cache writes take ~40ms (embed + SQLite insert). Making the pipeline wait for
that on every run would add 40ms to every execution. The daemon thread approach:
- Never blocks the pipeline critical path
- Naturally handles failures (if the thread crashes, the pipeline is unaffected)
- Cleans up automatically when the process exits (daemon=True)

---

## 8. Async Architecture

### Why async?

The researchers make I/O-bound calls (Tavily REST API, Wikipedia API). Running
them synchronously wastes wall-clock time waiting for network responses. The
async design has two levels:

**Level 1 — Graph-level parallelism (Send):**
LangGraph dispatches all `Send()` objects returned by a routing function as
independent parallel node invocations. Three researchers run concurrently,
each in its own invocation context.

**Level 2 — Intra-node parallelism (asyncio.gather):**
Within each researcher node, Tavily and Wikipedia are queried concurrently:
```python
raw_lists = await asyncio.gather(
    async_search_web(topic),        # aiohttp → no blocking
    async_search_wikipedia(topic),  # asyncio.to_thread → thread pool
)
```

### Sync/Async Bridge

LangGraph graph nodes are called synchronously by default. The `researcher_agent`
bridges this gap with `_run_async()`:

```python
def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        # We're inside an existing event loop (e.g. arun_pipeline called from
        # an async context). Can't use asyncio.run() — it creates a new loop.
        # Use a thread with its own event loop instead.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running event loop (sync CLI, sync eval).
        # Safe to create a new one.
        return asyncio.run(coro)
```

### Streamlit Queue Bridge

Streamlit's main thread is synchronous. It cannot `await` or `async for`.
`stream_pipeline_async()` bridges the gap:

```
                    ┌──────────────────────────────────┐
Streamlit thread    │   Background daemon thread        │
(sync)              │   (has its own event loop)        │
──────────────────  │  ──────────────────────────────── │
                    │                                  │
stream_pipeline     │                                  │
_async()            │  asyncio.run(_astream())         │
  │                 │    │                             │
  │  Queue() ───────┼────┤ astream node events         │
  │                 │    │                             │
  │  while True:    │    ├─ event_queue.put(event)     │
  │    kind, data   │◄───┤                             │
  │    = queue.get()│    ├─ ...                        │
  │                 │    │                             │
  │  if kind="done" │    ├─ event_queue.put(("done"))  │
  │    break        │    │                             │
  │                 │    └─ thread exits               │
  │                 └──────────────────────────────────┘
  │
  yield (node_name, update, accumulated_state)
```

The queue is typed as `queue.Queue[Optional[dict]]`.
- `("event", {node: update})` — normal node completion
- `("error", Exception)` — pipeline failed, re-raise
- `("done", None)` — sentinel, stream complete

State accumulation happens on the Streamlit side: the generator maintains
an `accumulated` dict that merges each update, handling `operator.add` fields
manually (sources, pipeline_trace, token_count, cost_usd).

---

## 9. Observability: Database Schema

Three tables track every pipeline execution. The DB is SQLite by default
(no infra needed) and PostgreSQL/Supabase in production (set `DATABASE_URL`).

### pipeline_runs

```sql
CREATE TABLE pipeline_runs (
    run_id       TEXT    PRIMARY KEY,   -- UUID4
    query        TEXT    NOT NULL,      -- original user query
    status       TEXT    NOT NULL,      -- "running" | "completed" | "failed"
    started_at   REAL    NOT NULL,      -- Unix timestamp
    finished_at  REAL,                  -- NULL until pipeline ends
    total_tokens INTEGER DEFAULT 0,
    total_cost   REAL    DEFAULT 0.0,   -- USD, 6 decimal places
    final_report TEXT                   -- the markdown report output
);
```

### agent_events

```sql
CREATE TABLE agent_events (
    event_id    TEXT    PRIMARY KEY,   -- UUID4 per start/end pair
    run_id      TEXT    NOT NULL,      -- FK → pipeline_runs
    agent_name  TEXT    NOT NULL,      -- "planner" | "researcher" | ...
    status      TEXT    NOT NULL,      -- "started" | "completed" | "failed"
    started_at  REAL    NOT NULL,
    duration_ms INTEGER,               -- NULL for "started" rows
    tokens_used INTEGER DEFAULT 0,
    cost_usd    REAL    DEFAULT 0.0,
    input_hash  TEXT,                  -- SHA-256[:12] of agent input (for debugging)
    error       TEXT                   -- populated only on "failed"
);
CREATE INDEX idx_events_run ON agent_events(run_id);
```

### cost_ledger

```sql
CREATE TABLE cost_ledger (
    ledger_id     TEXT    PRIMARY KEY,   -- UUID4
    run_id        TEXT    NOT NULL,
    agent_name    TEXT    NOT NULL,
    model         TEXT    NOT NULL,      -- full model string
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd      REAL    NOT NULL DEFAULT 0.0,
    timestamp     REAL    NOT NULL
);
CREATE INDEX idx_ledger_run ON cost_ledger(run_id);
```

### Background Queue Pattern

All DB writes go through `src/observability/logger.py`'s write queue:

```
Agent calls          Queue           Drain thread (daemon)
─────────────        ─────────       ────────────────────
log_agent_start()  → put_nowait() → pop → db.execute(INSERT agent_events)
log_agent_end()    → put_nowait() → pop → db.execute(UPDATE agent_events)
log_cost()         → put_nowait() → pop → db.execute(INSERT cost_ledger)
start_run()        → put_nowait() → pop → db.execute(INSERT pipeline_runs)
end_run()          → put_nowait() → pop → db.execute(UPDATE pipeline_runs)
```

Agents never block on DB. If the queue fills up (max 1000 items), items are
dropped with a warning rather than blocking. Logging is non-critical — a missed
DB write is acceptable; blocking an LLM call is not.

### PostgreSQL / Supabase Support

Switch backends by setting `DATABASE_URL`:
```
DATABASE_URL=postgresql://postgres.[ref]:[pw]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
```

- Port 6543 = Supabase PgBouncer (transaction-mode connection pooling)
- SQLite uses `?` placeholders; PostgreSQL uses `%s`
- `db.execute()` translates automatically: `sql.replace("?", "%s")`
- Tables are created idempotently at import time (`CREATE TABLE IF NOT EXISTS`)

---

## 10. Per-Agent Cost Tracking

### Pricing Table (`src/observability/cost.py`)

```python
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},  # per 1k tokens
    "claude-sonnet-4-6":         {"input": 0.003,   "output": 0.015},
    "claude-opus-4-6":           {"input": 0.015,   "output": 0.075},
}
```

### Per-call calculation

```python
cost = (input_tokens / 1000 × price_per_1k_input) +
       (output_tokens / 1000 × price_per_1k_output)
```

Every agent calls `calculate_cost()` immediately after `llm.invoke()`:
```python
usage       = extract_token_usage(response)      # reads response.usage_metadata
cost_record = calculate_cost(model, usage.input_tokens, usage.output_tokens, ...)
log_cost(cost_record)                            # queues DB write
```

### Budget Enforcement

Two thresholds from `configs/base.yaml`:
```yaml
budget:
  soft_limit: 0.08   # USD — warn but continue
  hard_limit: 0.10   # USD — skip LLM call, return fallback
```

`RunCostAccumulator` tracks total cost across the run. The `budget_status()`
method returns `"ok"` / `"warn"` / `"exceeded"`. Agents can call
`check_budget(accumulator)` before any LLM call to gracefully degrade
rather than crash on budget overrun.

### Typical costs per run

| Agent | Input tokens | Output tokens | Cost (Haiku) |
|---|---|---|---|
| Planner | ~200 | ~100 | $0.000175 |
| Researcher ×3 | 0 | 0 | $0 (no LLM) |
| Analyst | ~600 | ~300 | $0.000525 |
| Synthesizer | ~800 | ~400 | $0.000700 |
| Writer v1 | ~1200 | ~700 | $0.001175 |
| Reviewer | ~1400 | ~300 | $0.000725 |
| Writer v2 (if needed) | ~800 | ~700 | $0.001075 |
| **Total (no revision)** | ~4200 | ~1800 | **~$0.00330** |
| **Total (with revision)** | ~5000 | ~2500 | **~$0.00438** |

---

## 11. Guardrails (4 Layers)

All guardrails are pure Python — zero LLM calls, zero API cost.

### Layer 1: Prompt Injection Detection

```python
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+|your\s+|previous\s+)?instructions", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"(reveal|show|print|output)\s+(your\s+)?(system\s+prompt|instructions)", re.I),
    re.compile(r"disregard\s+(the\s+|all\s+)?above", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
    re.compile(r"override\s+(system|safety)\s+", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a\s+)", re.I),
    re.compile(r"forget\s+(everything|all|your)", re.I),
    re.compile(r"(sudo|admin|root)\s*:", re.I),
]
```

Patterns are compiled once at import time for performance. Mitigates
OWASP LLM Top 10 #1 (Prompt Injection).

### Layer 2: PII Detection & Scrubbing

```python
_PII_PATTERNS = {
    "email":       re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "phone_us":    re.compile(r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
```

Applied to every LLM output via `check_output()`. Replaces matches with
`[REDACTED_EMAIL]`, `[REDACTED_PHONE_US]`, etc. The writer agent calls this
before storing any draft in state.

### Layer 3: Budget Gate

Thin wrapper around `cost.check_budget()`. Every agent calls this before
any LLM invocation. Hard limit ($0.10) returns `(False, "exceeded")` which
agents use to skip the LLM call and return a fallback response.

### Layer 4: Token-Bucket Rate Limiter

```python
class RateLimiter:
    def __init__(self, max_rpm=50):
        self._tokens     = float(max_rpm)    # start full
        self._refill_rate = max_rpm / 60.0   # tokens per second

    def acquire(self) -> float:
        # Consume 1 token. Block until available. Returns wait time.
        ...
```

Token bucket smooths request rate evenly (vs fixed window which allows bursts).
For Anthropic's API, 50 RPM limit means agents wait at most 1.2s between calls
at maximum throughput. Rate limit prevents `429 Too Many Requests` errors
entirely.

---

## 12. Tool Selection Logic

`src/tools/tool_selector.py` routes each sub-topic query to the best tool.

```
Wikipedia signals:               Tavily signals:
  "who is/was"                     "latest", "recent"
  "what is a/an/the"               "2023-2029" (year patterns)
  "history of"                     "current(ly)"
  "definition of"                  "news", "today"
  "explain"                        "best", "top N"
  "algorithm"                      "compare", "vs"
  "mathematical"                   "how to"
  "theory of"                      "tutorial", "guide"
```

Decision logic:
```python
if wiki_hits > 0 and tavily_hits == 0:   tool = "wikipedia"
elif tavily_hits > 0 and wiki_hits == 0: tool = "tavily"
elif wiki_hits > 0 and tavily_hits > 0:  tool = "both"
else:                                     tool = "tavily"  # safe default
```

Confidence is `min(0.6 + hits × 0.1, 0.95)`. Mixed signals get 0.5.

---

## 13. YAML-Driven Configuration

Everything configurable lives in `configs/base.yaml`. Zero code changes needed to:

```yaml
# Swap model for one agent (e.g. use Sonnet for writer for higher quality)
models:
  writer: "claude-sonnet-4-6"

# Tighten quality gate (stricter source acceptance)
pipeline:
  quality_threshold: 0.55    # was 0.40

# Add a trusted domain
quality_gate:
  high_trust_domains:
    - "openai.com"            # new addition

# Allow more revisions
pipeline:
  max_revisions: 3            # was 2

# Disable cache for debugging
cache:
  enabled: false

# Adjust budget for production
budget:
  soft_limit: 0.50
  hard_limit: 1.00
```

The config is loaded as a singleton (`_config` module-level cache) by
`src/config.py`. It is read lazily at agent invocation time, not at import time,
so changes during a test run take effect on the next `load_config()` call.

---

## 14. Testing Architecture (110+ Tests)

### Test Categories

| File | Tests | Category | What They Cover |
|---|---|---|---|
| `test_e2e_pipeline.py` | 37 | End-to-end | Full pipeline with all LLM calls mocked |
| `test_ui_smoke.py` | 26 | Smoke | All module imports, symbol accessibility |
| `test_guardrails.py` | 14 | Unit | PII, injection, budget, rate limiter |
| `test_graph.py` | 12 | Unit | Routing functions, utility nodes |
| `test_quality_gate.py` | 12 | Unit | Domain trust, snippet scoring, composites |
| `test_cache.py` | 17 | Unit | Two-level cache: fetch, store, stats, TTL |
| `test_state.py` | 12 | Unit | TypedDict fields, operator.add reducers |
| `test_tools.py` | 12 | Unit | search_web, search_wikipedia, async_search_all |
| `test_tool_selector.py` | 7 | Unit | Keyword matching, routing decisions |
| `test_config.py` | 7 | Unit | YAML loading, accessor functions |

All 110+ tests run **without any API keys**. LLM calls are mocked via
`unittest.mock.patch`. The test runner never makes network requests.

### Key Test Patterns

**Mocking LLM structured output:**
```python
mock_raw = {"parsed": PlannerOutput(sub_topics=["t1", "t2"], research_plan="plan"),
            "raw": make_mock_llm_response("...", input_tokens=100, output_tokens=50)}
with patch.object(_llm, "with_structured_output") as mock_sllm:
    mock_sllm.return_value.invoke.return_value = mock_raw
    result = planner_agent(state)
```

**Cache isolation:**
```python
@pytest.fixture
def patched_cache(tmp_db_path):
    with patch("src.cache.research_cache._DB_PATH", tmp_db_path), \
         patch("src.cache.research_cache._get_embedder", return_value=None):
        research_cache._ensure_table()
        yield
```
Redirects all cache DB operations to a temp file. Each test gets a clean slate.

**Graph routing tests:**
```python
def test_cache_hit_returns_string():
    from src.agents.graph import fan_out_or_cache
    fake_sources = [{"url": "https://a.com", "title": "a", "content": "c"}]
    with patch("src.agents.graph.cache_fetch", return_value=fake_sources):
        result = fan_out_or_cache(state)
    assert result == "cache_loader"   # routing function returns node name
```

---

## 15. Performance Profile

Measured on MacBook M2, Haiku model, 3 sub-topics, "both" tool selection.

| Stage | First Run | Cached Run | Notes |
|---|---|---|---|
| Planner | ~2.3s | ~2.3s | LLM call, not cacheable |
| Researcher ×3 (parallel) | ~1.0s wall | 0s | asyncio.gather within each; LangGraph Send() across all |
| Merge + background cache write | ~5ms | ~5ms | Dedup only; write is async |
| Quality Gate | <1ms | <1ms | Pure Python, no LLM |
| Analyst | ~2.9s | ~2.9s | LLM call |
| Synthesizer | ~3.1s | ~3.1s | LLM call |
| Writer v1 | ~4.2s | ~4.2s | LLM call, largest output |
| Reviewer | ~2.7s | ~2.7s | LLM call |
| **Total (no revision)** | **~16s** | **~10s** | 38% faster on cache hit |
| **Total (with revision)** | **~23s** | **~17s** | One extra writer + reviewer |

### Bottleneck Analysis

The pipeline is dominated by LLM latency (7 agents × avg 2.5s = ~17.5s).
The research stage adds only ~1s thanks to two-level parallelism. The cache
hit saves ~6s by eliminating the researcher + merge_research stages entirely.

### Optimization Opportunities (for production)

1. **Use `astream_events`** instead of `astream` — finer-grained streaming
   allows UI to update within agent calls, not just between them.
2. **Analyst streaming** — the analyst prompt is the best candidate for
   streaming partial output since it generates structured JSON.
3. **Redis cache** — for multi-user deployments, SQLite is single-writer.
   Redis supports concurrent cache writes with TTL natively.
4. **Single-agent routing** — simple factual queries ("what year was Python
   invented?") don't need 7 agents. Route by query length + complexity
   score to save 2.6× cost.

---

## 16. Security Model (OWASP LLM Top 10)

| Risk | Mitigation in This System |
|---|---|
| **LLM01 Prompt Injection** | `detect_injection()` at pipeline entry with 10 regex patterns. Rejects queries containing injection patterns before any LLM call. |
| **LLM02 Insecure Output Handling** | `check_output()` on every LLM response. PII (email, phone, SSN, CC, IP) replaced with redaction tokens before storing in state or displaying. |
| **LLM06 Sensitive Info Disclosure** | Writer is constrained: "Only use facts from `state.sources`". Sources are web search results, not private data. Cache stores search results, not user queries with PII (injection check runs first). |
| **LLM07 Insecure Plugin Design** | Tools (Tavily, Wikipedia) are sandboxed: read-only, no writes, no code execution. Async tools use `aiohttp` (no shell) and `asyncio.to_thread` (isolated). |
| **LLM08 Excessive Agency** | Agent capabilities are minimal: search + read only. No file writes, no network calls outside designated tools, no code execution. |
| **LLM09 Overreliance** | Reviewer loop provides independent quality check. Analyst conflict detection flags when sources disagree. Synthesis explicitly surfaces disagreements. |
| **LLM10 Model Theft** | N/A — uses Claude API (no local model). API keys in environment variables, never in source code. |

---

## 17. Deployment

### Local Development

```bash
# 1. Install dependencies
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY=sk-ant-api03-...
#   TAVILY_API_KEY=tvly-...

# 3. Run tests (no API keys needed)
make test

# 4. Launch Streamlit UI
make ui

# 5. Or run via CLI
make run
```

### Streamlit Community Cloud (Free)

```
1. Push to GitHub:
   git add . && git commit -m "deploy" && git push

2. Go to share.streamlit.io
   → New app → your repo → main → app.py

3. Add Secrets (App settings → Secrets):
   ANTHROPIC_API_KEY = "sk-ant-..."
   TAVILY_API_KEY    = "tvly-..."

4. Click Deploy
```

**Cold start note:** First deployment downloads `all-MiniLM-L6-v2` (~80MB).
The `HF_HUB_OFFLINE=1` logic in `research_cache.py` handles this automatically.
Subsequent restarts use the cached model (Streamlit Cloud caches between deploys).

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `TAVILY_API_KEY` | Yes | — | Tavily search API key (free tier: 1000/month) |
| `DATABASE_URL` | No | `sqlite:///data/pipeline.db` | Override with Postgres URL for production |
| `HF_HUB_OFFLINE` | Auto | Set by cache module | Prevents huggingface network calls after first model download |

### Production Upgrade Path

For serving multiple concurrent users:

1. **Replace SQLite cache with Redis:**
   ```python
   # research_cache.py: replace sqlite3 with redis-py
   # Natively handles concurrent writes + built-in TTL
   ```

2. **Use PostgreSQL for observability:**
   ```
   DATABASE_URL=postgresql://...  (already supported via psycopg3)
   ```

3. **Add LangSmith tracing:**
   ```python
   # In graph.py, wrap build_graph() with LangSmith callbacks
   # Gives distributed tracing across runs with full state inspection
   ```

4. **Add human-in-the-loop checkpoints:**
   ```python
   # LangGraph supports interrupt_before=["writer"] to pause for review
   ```

---

## 18. Interview Q&A

**Q: Why 10 graph nodes for 7 agent functions?**

> Seven unique agent functions but 10 graph nodes: planner, researcher,
> merge_research, cache_loader, quality_gate, retry_counter, analyst,
> synthesizer, writer, reviewer. The extra 3 are utility nodes:
> `merge_research` handles post-fan-out dedup, `cache_loader` writes
> cached sources into state (routing functions are read-only), and
> `retry_counter` increments retry count and sets force_research. These
> can't be done inside routing functions because LangGraph routing functions
> are read-only — they can return routing decisions but cannot modify state.

**Q: How does the two-level cache work and why does it matter?**

> Level 1 is SHA-256 of the normalized query — O(1), catches exact repeats
> and case differences. Level 2 embeds the query with `all-MiniLM-L6-v2`
> and finds the nearest cached entry by cosine similarity (threshold 0.60).
> This catches paraphrases: "what is FAISS" and "explain FAISS" have cosine
> 0.76, so the second query hits the cache and skips all research. Cache
> writes happen on a daemon background thread from `merge_research_node`
> — the pipeline never waits for the ~40ms embed+write.

**Q: How does parallel execution work and how are results merged?**

> The planner decomposes the query into 1-3 sub-topics. `fan_out_or_cache`
> returns a list of `Send("researcher", {**state, "current_topic": topic})`
> objects. LangGraph dispatches them in parallel. Within each researcher,
> Tavily and Wikipedia are queried concurrently via `asyncio.gather()`.
> The `sources` field uses `Annotated[list[dict], operator.add]` as its
> type annotation. LangGraph sees this annotation and, instead of overwriting
> `sources` on each parallel write, calls `operator.add(existing, new)` which
> is just list concatenation. So all three researchers' sources accumulate
> correctly before `merge_research_node` sees them.

**Q: Why is the quality gate pure Python instead of an LLM call?**

> Two reasons: cost and determinism. An LLM call for every set of search
> results would add ~$0.0001 and ~2.5s per quality check, which adds up across
> evaluation runs (10 questions × multiple pipeline stages). Pure Python heuristics
> (domain trust + snippet length/boilerplate) run in < 1ms at zero cost. The
> trade-off is less nuanced judgment — the LLM might recognize that a short
> snippet from arxiv.org is more trustworthy than a long snippet from reddit.com.
> In practice, the domain trust scores already encode this correctly.

**Q: How do you prevent the reviewer loop from running forever?**

> Two independent guards: (1) `review.score >= review_pass_score (7)` — the
> reviewer must actively fail the draft to trigger a revision; (2)
> `revision_count >= max_revisions (2)` — a hard cap checked in `_should_revise`
> before testing the score. The conditional edge logic is:
> `passed OR score >= pass_score OR revision_count >= max_revisions → END`.
> The pipeline always terminates within 3 writer invocations maximum.

**Q: How do you prevent citation hallucination?**

> Three layers: (1) The writer prompt explicitly says "only cite sources from
> the provided list" and the prompt includes the exact source URLs and titles.
> (2) The analyst's `ClaimOutput` schema requires a `source_idx` field (1-based
> index into `state.sources`) for every claim — the LLM must point to a real
> source. (3) The synthesizer detects cross-source conflicts and surfaces them
> explicitly, so the writer knows which claims are disputed.

**Q: How does the async Streamlit bridge work?**

> `stream_pipeline_async()` is a synchronous generator (Streamlit can't `await`).
> It starts a daemon thread that runs `graph.astream()` inside its own event
> loop via `asyncio.run(_astream())`. Node completion events are put onto a
> `queue.Queue`. The Streamlit main thread drains this queue in a `while True`
> loop, merges each update into an `accumulated` state dict, and `yield`s
> `(node_name, update, accumulated)` for each completed node. Streamlit calls
> `next()` on this generator on each Streamlit rerun cycle, updating the UI.

**Q: What would you change for a production multi-tenant deployment?**

> Five things: (1) Replace SQLite cache with Redis (concurrent writes, native TTL,
> horizontal scale). (2) Move observability DB to Postgres — already supported,
> just set DATABASE_URL. (3) Add LangSmith distributed tracing for per-run
> debugging across users. (4) Add a query router that sends simple factual
> queries to single-agent (2.6× cheaper per the evaluation data). (5) Add
> horizontal pipeline parallelism — run multiple `stream_pipeline_async()`
> calls concurrently with a `asyncio.gather()` wrapper for batch research jobs.

**Q: How are the tests structured to run without API keys?**

> Four patterns: (1) LLM singletons are mocked at import time with
> `patch("langchain_anthropic.ChatAnthropic")` in smoke tests.
> (2) Agent tests patch `llm.invoke()` or `llm.with_structured_output().invoke()`
> to return `make_mock_llm_response()` objects that mimic real AIMessage structure
> including `usage_metadata`. (3) Cache tests use `tmp_path` fixture to redirect
> `_DB_PATH` to a temp SQLite file and mock `_get_embedder` to return None
> (exact-match only, no embedding model needed). (4) E2E tests mock all network
> calls at the tool level (`search_web`, `search_wikipedia`) and run the full
> LangGraph pipeline with fake data flowing through real routing logic.

---

*Last updated: March 2026*
*Architecture version: 1.0 — 10-node async pipeline*
