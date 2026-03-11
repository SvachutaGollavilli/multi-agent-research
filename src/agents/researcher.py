# src/agents/researcher.py
# -----------------------------------------------------------------
# Researcher Agent -- single-topic, Send()-aware.
#
# Parallelism at two levels:
#
#   Graph level (existing):
#     LangGraph Send() dispatches one researcher per sub-topic in
#     parallel. Three sub-topics -> three concurrent researcher nodes.
#
#   Intra-node level (new):
#     Within each researcher, Tavily and Wikipedia are queried
#     CONCURRENTLY using asyncio.gather() via async_search_all().
#     For "both" tool selection this saves ~600-900ms per call.
#
# Cache: NOT checked or written here. Cache logic lives in:
#   - graph.py::fan_out_or_cache   (check before fan-out)
#   - graph.py::merge_research_node (write after merge)
# -----------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import time

from dotenv import load_dotenv

from src.config import get_search_config
from src.models.state import ResearchState
from src.observability.logger import log_agent_end, log_agent_start

# Async search -- Tavily via aiohttp, Wikipedia via asyncio.to_thread
from src.tools.async_search import async_search_all
from src.tools.tool_selector import select_tool

load_dotenv()
logger = logging.getLogger(__name__)


def researcher_agent(state: ResearchState) -> dict:
    """
    Researcher Agent -- searches a single topic and returns sources.

    Runs both Tavily and Wikipedia concurrently using asyncio.gather.
    The sync/async bridge is handled internally: the agent exposes a
    standard synchronous interface to LangGraph, but internally runs
    an async coroutine via _run_async().

    Receives current_topic from Send() -- the specific sub-topic to research.
    Falls back to query if current_topic is not set (direct/retry calls).
    """
    run_id = state.get("run_id", "")
    query = state.get("query", "")
    current_topic = state.get("current_topic", "") or query

    event_id, t0 = log_agent_start(run_id, "researcher", {"topic": current_topic[:60]})
    logger.info(f"[researcher] searching topic: '{current_topic[:60]}'")

    try:
        search_cfg = get_search_config()
        max_results = search_cfg.get("max_results", 5)
        max_wiki = search_cfg.get("max_wiki_results", 3)

        selection = select_tool(current_topic)
        tool = selection["tool"]
        logger.debug(
            f"[researcher] tool={tool} | conf={selection['confidence']} | "
            f"{selection['reason']}"
        )

        # Run searches concurrently via asyncio
        unique = _run_async(
            async_search_all(
                query=current_topic,
                tool=tool,
                max_results=max_results,
                max_wiki=max_wiki,
            )
        )

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[researcher] done | '{current_topic[:40]}' -> "
            f"{len(unique)} sources via {tool} | {elapsed_ms}ms"
        )
        log_agent_end(event_id, run_id, "researcher", t0)

        return {
            "sources": unique,
            "search_queries_used": [current_topic],
            "pipeline_trace": [
                {
                    "agent": "researcher",
                    "duration_ms": elapsed_ms,
                    "tokens": 0,
                    "summary": (
                        f"'{current_topic[:35]}' -> "
                        f"{len(unique)} sources via {tool} (async)"
                    ),
                }
            ],
        }

    except Exception as e:
        logger.error(f"[researcher] failed for '{current_topic[:40]}': {e}")
        log_agent_end(event_id, run_id, "researcher", t0, error=str(e))
        return {
            "sources": [],
            "search_queries_used": [current_topic],
            "errors": [f"Researcher error ({current_topic[:40]}): {e}"],
            "pipeline_trace": [
                {
                    "agent": "researcher",
                    "duration_ms": int((time.time() - t0) * 1000),
                    "tokens": 0,
                    "summary": f"Error on '{current_topic[:35]}': {e}",
                }
            ],
        }


# -----------------------------------------------------------------
# Sync/async bridge
# -----------------------------------------------------------------
# LangGraph graph nodes are called synchronously by default.
# We bridge to async by running the coroutine in a new event loop,
# or scheduling it on an existing one if we're already inside asyncio
# (e.g. when arun_pipeline() is used).
# -----------------------------------------------------------------


def _run_async(coro) -> any:
    """
    Run an async coroutine from a synchronous context.

    Strategy:
      1. If there is a running event loop (e.g. arun_pipeline called from
         an async context), schedule the coroutine as a task and block
         the thread until it finishes.
      2. If there is no running event loop (sync CLI or eval), use
         asyncio.run() which creates a fresh event loop.

    This bridges async_search_all() into the synchronous LangGraph graph
    without requiring the whole graph to be async.
    """
    try:
        asyncio.get_running_loop()
        # We ARE inside an async context -- run in a thread with its own event loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    except RuntimeError:
        # No running event loop -- safe to call asyncio.run() directly
        return asyncio.run(coro)
