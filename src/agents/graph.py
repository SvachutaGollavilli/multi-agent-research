# src/agents/graph.py

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Iterator

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.agents.analyst import analyst_agent
from src.agents.planner import planner_agent
from src.agents.quality_gate import quality_gate_agent
from src.agents.researcher import researcher_agent
from src.agents.reviewer import reviewer_agent
from src.agents.synthesizer import synthesizer_agent
from src.agents.writer import writer_agent
from src.cache.research_cache import fetch as cache_fetch
from src.cache.research_cache import store as cache_store
from src.config import get_pipeline_config
from src.models.state import ResearchState, default_state
from src.observability.cost import RunCostAccumulator
from src.observability.logger import end_run, start_logger, start_run
from src.output.report_writer import write_report

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================
# Routing functions (read-only -- cannot write state)
# =============================================================


def fan_out_or_cache(state: ResearchState):
    """
    After planner: check cache first, fan out to parallel researchers on miss.

    Cache HIT  -> "cache_loader"  (single node, loads sources into state)
    Cache MISS -> [Send("researcher", ...) x N]  (parallel fan-out)

    force_research=True skips cache -- set by retry_counter after a quality
    gate failure so retries don't return the same low-quality cached results.
    """
    query = state.get("query", "")
    sub_topics = state.get("sub_topics") or [query]
    force_research = state.get("force_research", False)

    if not force_research:
        cached = cache_fetch(query)
        if cached is not None:
            logger.info(
                f"[graph] cache HIT for '{query[:50]}' "
                f"({len(cached)} sources) -> cache_loader"
            )
            return "cache_loader"

    logger.info(
        f"[graph] {'forced fresh search' if force_research else 'cache MISS'} -- "
        f"fanning out to {len(sub_topics)} parallel researchers"
    )
    return [
        Send("researcher", {**state, "current_topic": topic}) for topic in sub_topics
    ]


def _should_retry_research(state: ResearchState) -> str:
    passed = state.get("quality_passed", False)
    quality_retries = state.get("quality_retries", 0)
    score = state.get("quality_score", 0.0)

    cfg = get_pipeline_config()
    max_quality_retries = int(cfg.get("max_quality_retries", 1))
    threshold = float(cfg.get("quality_threshold", 0.4))

    if passed:
        logger.info(
            f"[graph] quality PASS (score={score:.3f} >= {threshold}) -> analyst"
        )
        return "analyst"

    if quality_retries < max_quality_retries:
        logger.info(
            f"[graph] quality FAIL (score={score:.3f} < {threshold}) -- "
            f"retry {quality_retries + 1}/{max_quality_retries} -> retry_counter"
        )
        return "retry_counter"

    logger.info(
        f"[graph] quality FAIL (score={score:.3f}) retries exhausted "
        f"({quality_retries}/{max_quality_retries}) -- proceeding to analyst"
    )
    return "analyst"


def _should_revise(state: ResearchState) -> str:
    review = state.get("review", {})
    score = review.get("score", 0)
    passed = review.get("passed", False)
    revision_count = state.get("revision_count", 0)

    cfg = get_pipeline_config()
    max_revisions = int(cfg.get("max_revisions", 2))
    pass_score = int(cfg.get("review_pass_score", 7))

    if passed or score >= pass_score:
        logger.info(f"[graph] reviewer PASS (score={score}/{pass_score}) -> END")
        return "end"

    if revision_count >= max_revisions:
        logger.info(
            f"[graph] max revisions reached ({revision_count}/{max_revisions}), "
            f"score={score} -- proceeding to END"
        )
        return "end"

    logger.info(
        f"[graph] reviewer FAIL (score={score}/{pass_score}) -- "
        f"revision {revision_count}/{max_revisions} -> writer"
    )
    return "revise"


# =============================================================
# Utility nodes
# =============================================================


def cache_loader_node(state: ResearchState) -> dict:
    """Write cached sources into state. Needed because routing fns are read-only."""
    query = state.get("query", "")
    t0 = time.time()

    cached = cache_fetch(query)
    sources = cached or []
    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(f"[cache_loader] {len(sources)} cached sources | {elapsed_ms}ms")

    return {
        "sources": sources,
        "search_queries_used": [query],
        "pipeline_trace": [
            {
                "agent": "cache_loader",
                "duration_ms": elapsed_ms,
                "tokens": 0,
                "summary": f"Cache HIT: {len(sources)} sources for '{query[:40]}'",
            }
        ],
    }


def merge_research_node(state: ResearchState) -> dict:
    """
    Runs once after all parallel Send("researcher") branches finish.
    LangGraph has already concatenated sources via operator.add.

    1. Global URL dedup across the merged list.
    2. Cache write on a BACKGROUND DAEMON THREAD -- pipeline never waits
       for the ~40ms embed + write.
    """
    query = state.get("query", "")
    sources = state.get("sources", [])
    t0 = time.time()

    seen: set = set()
    unique: list[dict] = []
    for s in sources:
        url = s.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(s)

    duplicates = len(sources) - len(unique)
    if duplicates:
        logger.info(f"[merge_research] deduped {duplicates} duplicate URLs")

    if unique:
        t = threading.Thread(
            target=cache_store,
            args=(query, unique, "parallel_send"),
            daemon=True,
            name="cache-write",
        )
        t.start()
        logger.info("[merge_research] cache write dispatched to background thread")

    elapsed_ms = int((time.time() - t0) * 1000)
    logger.info(
        f"[merge_research] {len(unique)} unique sources "
        f"(removed {duplicates} dupes) | {elapsed_ms}ms"
    )

    return {
        "sources": unique,
        "pipeline_trace": [
            {
                "agent": "merge_research",
                "duration_ms": elapsed_ms,
                "tokens": 0,
                "summary": (
                    f"Merged to {len(unique)} unique sources "
                    f"(removed {duplicates} dupes)"
                ),
            }
        ],
    }


def retry_counter_node(state: ResearchState) -> dict:
    """Bump retry count, set force_research, clear stale sources."""
    retries = state.get("quality_retries", 0) + 1
    logger.info(
        f"[retry_counter] quality_retries -> {retries} | "
        "force_research=True | clearing sources"
    )
    return {
        "quality_retries": retries,
        "force_research": True,
        "sources": [],
    }


# =============================================================
# Graph assembly
# =============================================================


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_agent)
    graph.add_node("cache_loader", cache_loader_node)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("merge_research", merge_research_node)
    graph.add_node("quality_gate", quality_gate_agent)
    graph.add_node("retry_counter", retry_counter_node)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("reviewer", reviewer_agent)

    graph.set_entry_point("planner")

    graph.add_conditional_edges(
        "planner",
        fan_out_or_cache,
        {"cache_loader": "cache_loader", "researcher": "researcher"},
    )

    graph.add_edge("cache_loader", "quality_gate")
    graph.add_edge("researcher", "merge_research")
    graph.add_edge("merge_research", "quality_gate")

    graph.add_conditional_edges(
        "quality_gate",
        _should_retry_research,
        {"analyst": "analyst", "retry_counter": "retry_counter"},
    )

    graph.add_conditional_edges(
        "retry_counter",
        fan_out_or_cache,
        {"cache_loader": "cache_loader", "researcher": "researcher"},
    )

    graph.add_edge("analyst", "synthesizer")
    graph.add_edge("synthesizer", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        _should_revise,
        {"end": END, "revise": "writer"},
    )

    return graph.compile()


# =============================================================
# Sync pipeline runner (CLI, eval)
# =============================================================


def run_pipeline(query: str) -> dict:
    start_logger()
    run_id = start_run(query)
    acc = RunCostAccumulator(run_id=run_id)

    try:
        pipeline = build_graph()
        initial = default_state(query=query, run_id=run_id)
        result = pipeline.invoke(initial)

        end_run(
            run_id=run_id,
            accumulator=acc,
            final_report=result.get("final_report") or result.get("current_draft", ""),
            status="completed",
            total_tokens=result.get("token_count", 0),
            total_cost=result.get("cost_usd", 0.0),
        )

        report_path = write_report(result, run_id=run_id)
        logger.info(f"Report saved -> {report_path}")

        review = result.get("review", {})
        revision_count = result.get("revision_count", 0)
        quality_retries = result.get("quality_retries", 0)
        trace = result.get("pipeline_trace", [])

        logger.info("-" * 60)
        logger.info(f"Pipeline summary | run: {run_id[:8]}...")
        logger.info(f"  sub-topics      : {len(result.get('sub_topics', []))}")
        logger.info(f"  sources         : {len(result.get('sources', []))}")
        logger.info(
            f"  quality score   : {result.get('quality_score', 0.0):.3f} "
            f"({'PASS' if result.get('quality_passed') else 'FAIL'})"
        )
        logger.info(f"  quality retries : {quality_retries}")
        logger.info(f"  claims          : {len(result.get('key_claims', []))}")
        logger.info(f"  synthesis       : {len(result.get('synthesis', ''))} chars")
        logger.info(f"  revisions       : {revision_count}")
        logger.info(f"  final score     : {review.get('score', '?')}/10")
        logger.info(f"  tokens          : {result.get('token_count', 0)}")
        logger.info(f"  cost            : ${result.get('cost_usd', 0.0):.6f}")
        logger.info(f"  output          : {report_path}")
        logger.info("-" * 60)
        for step in trace:
            logger.info(
                f"  {step.get('agent', '?'):14s} | "
                f"{step.get('duration_ms', 0):5d}ms | "
                f"tokens: {step.get('tokens', 0):5d} | "
                f"{step.get('summary', '')}"
            )
        logger.info("-" * 60)

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        end_run(run_id=run_id, accumulator=acc, status="failed")
        raise


# =============================================================
# Async pipeline runner (Streamlit UI, arun_pipeline)
# =============================================================


async def arun_pipeline(query: str) -> dict:
    """
    Async version of run_pipeline() -- uses LangGraph's ainvoke() so
    the event loop is never blocked during LLM calls.

    Intended for contexts that already have a running event loop
    (e.g. a FastAPI server or an async test runner).
    For Streamlit use stream_pipeline_async() instead.
    """
    start_logger()
    run_id = start_run(query)
    acc = RunCostAccumulator(run_id=run_id)

    try:
        pipeline = build_graph()
        initial = default_state(query=query, run_id=run_id)
        result = await pipeline.ainvoke(initial)

        end_run(
            run_id=run_id,
            accumulator=acc,
            final_report=result.get("final_report") or result.get("current_draft", ""),
            status="completed",
            total_tokens=result.get("token_count", 0),
            total_cost=result.get("cost_usd", 0.0),
        )
        write_report(result, run_id=run_id)
        return result

    except Exception as e:
        logger.error(f"Async pipeline failed: {e}")
        end_run(run_id=run_id, accumulator=acc, status="failed")
        raise


def stream_pipeline_async(query: str) -> Iterator[tuple[str, dict, dict]]:
    """
    Sync generator that streams LangGraph node completions using the
    async graph.astream() internally.

    This is the primary entry point for the Streamlit UI.

    Why a sync generator over an async one?
      Streamlit's main thread is synchronous -- it cannot `await` or
      `async for`. We bridge by running the async stream in a background
      daemon thread with its own event loop, then passing events back to
      the Streamlit thread via a thread-safe queue.

    Yields: (node_name, state_update, accumulated_state)

    The accumulated_state dict is built incrementally here so the UI
    always has the full picture after each node completes.
    """
    start_logger()
    run_id = start_run(query)
    acc = RunCostAccumulator(run_id=run_id)

    # Queue bridges the async background thread -> sync Streamlit thread
    # Sentinel: None = stream finished, Exception instance = error
    event_queue: queue.Queue = queue.Queue()

    # Accumulated state -- built up as nodes complete
    initial = default_state(query=query, run_id=run_id)
    accumulated = dict(initial)

    async def _astream():
        """Run in the background thread's own event loop."""
        try:
            pipeline = build_graph()
            async for event in pipeline.astream(initial, stream_mode="updates"):
                event_queue.put(("event", event))
        except Exception as e:
            event_queue.put(("error", e))
        finally:
            event_queue.put(("done", None))

    # Start the async stream in a dedicated daemon thread
    def _thread_target():
        asyncio.run(_astream())

    t = threading.Thread(target=_thread_target, daemon=True, name="astream-thread")
    t.start()

    # Drain the queue and yield events to the Streamlit main thread
    run_error = None
    try:
        while True:
            kind, data = event_queue.get()

            if kind == "done":
                break
            elif kind == "error":
                run_error = data
                raise data
            else:
                # kind == "event": data is the {node_name: update} dict
                for node_name, update in data.items():
                    # Merge update into accumulated state
                    for k, v in update.items():
                        if k in (
                            "sources",
                            "pipeline_trace",
                            "errors",
                            "search_queries_used",
                        ):
                            existing = accumulated.get(k, [])
                            accumulated[k] = existing + (
                                v if isinstance(v, list) else []
                            )
                        elif k in ("token_count", "cost_usd"):
                            accumulated[k] = accumulated.get(k, 0) + (v or 0)
                        else:
                            accumulated[k] = v

                    yield node_name, update, accumulated

    finally:
        if run_error is None:
            end_run(
                run_id=run_id,
                accumulator=acc,
                final_report=(
                    accumulated.get("final_report")
                    or accumulated.get("current_draft", "")
                ),
                status="completed",
                total_tokens=accumulated.get("token_count", 0),
                total_cost=accumulated.get("cost_usd", 0.0),
            )
        else:
            end_run(run_id=run_id, accumulator=acc, status="failed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    query = input("\nResearch question: ")
    run_pipeline(query)
