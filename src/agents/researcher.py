# src/agents/researcher.py
# ─────────────────────────────────────────────
# Researcher Agent — single-topic, Send()-aware.
#
# In the parallel fan-out architecture, LangGraph dispatches one instance
# of this agent per sub-topic via Send(). Each instance receives its own
# sub-topic in state["current_topic"] and searches only that topic.
#
# Parallelism is now LangGraph's responsibility, not ours.
# ThreadPoolExecutor has been removed — it was doing the same job that
# Send() now does at the graph level, but invisibly.
#
# Cache: NOT checked or written here. Cache logic lives in:
#   - graph.py::fan_out_or_cache   (check before fan-out)
#   - graph.py::merge_research     (write after fan-out)
# This keeps the researcher stateless and focused on one job: fetch sources.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

from src.config import get_search_config
from src.models.state import ResearchState
from src.observability.logger import log_agent_end, log_agent_start
from src.tools.search import search_web
from src.tools.tool_selector import select_tool
from src.tools.wikipedia import search_wikipedia

load_dotenv()
logger = logging.getLogger(__name__)


def researcher_agent(state: ResearchState) -> dict:
    """
    Researcher Agent — searches a single topic and returns sources.

    Receives current_topic from Send() — the specific sub-topic to research.
    Falls back to query if current_topic is not set (direct/retry calls).

    Does NOT check or write cache — that is handled at the graph level
    (fan_out_or_cache checks before dispatch, merge_research writes after).
    """
    run_id        = state.get("run_id", "")
    query         = state.get("query", "")
    current_topic = state.get("current_topic", "") or query  # fallback to original query

    event_id, t0 = log_agent_start(run_id, "researcher", {"topic": current_topic[:60]})
    logger.info(f"[researcher] searching topic: '{current_topic[:60]}'")

    try:
        search_cfg  = get_search_config()
        max_results = search_cfg.get("max_results", 5)
        max_wiki    = search_cfg.get("max_wiki_results", 3)

        # Select tool based on the topic content
        selection = select_tool(current_topic)
        tool      = selection["tool"]
        logger.debug(
            f"[researcher] tool={tool} | conf={selection['confidence']} | "
            f"{selection['reason']}"
        )

        raw: list[dict] = []

        if tool in ("wikipedia", "both"):
            try:
                wiki_results = search_wikipedia(current_topic, max_results=max_wiki)
                raw.extend(wiki_results)
                logger.debug(f"[researcher] wikipedia: {len(wiki_results)} results")
            except Exception as e:
                logger.warning(f"[researcher] wikipedia failed: {e}")

        if tool in ("tavily", "both"):
            try:
                tavily_results = search_web(current_topic, max_results=max_results)
                raw.extend(tavily_results)
                logger.debug(f"[researcher] tavily: {len(tavily_results)} results")
            except Exception as e:
                logger.warning(f"[researcher] tavily failed: {e}")

        # URL dedup within this topic's results
        seen: set = set()
        unique: list[dict] = []
        for r in raw:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(r)

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[researcher] done | '{current_topic[:40]}' → "
            f"{len(unique)} sources via {tool} | {elapsed_ms}ms"
        )
        log_agent_end(event_id, run_id, "researcher", t0)

        return {
            "sources":             unique,
            "search_queries_used": [current_topic],
            "pipeline_trace": [{
                "agent":       "researcher",
                "duration_ms": elapsed_ms,
                "tokens":      0,
                "summary":     f"'{current_topic[:35]}' → {len(unique)} sources via {tool}",
            }],
        }

    except Exception as e:
        logger.error(f"[researcher] failed for '{current_topic[:40]}': {e}")
        log_agent_end(event_id, run_id, "researcher", t0, error=str(e))
        return {
            "sources":             [],
            "search_queries_used": [current_topic],
            "errors":              [f"Researcher error ({current_topic[:40]}): {e}"],
            "pipeline_trace": [{
                "agent":       "researcher",
                "duration_ms": int((time.time() - t0) * 1000),
                "tokens":      0,
                "summary":     f"Error on '{current_topic[:35]}': {e}",
            }],
        }
