# src/agents/researcher.py

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv

from src.config import get_search_config
from src.models.state import ResearchState
from src.observability.logger import log_agent_end, log_agent_start
from src.tools.search import search_web

load_dotenv()
logger = logging.getLogger(__name__)


def researcher_agent(state: ResearchState) -> dict:
    """
    Researcher Agent:
    - Receives a single sub-topic (or the full query) as state["query"]
    - Searches the web using Tavily
    - Returns sources list (operator.add merges parallel results)

    In parallel fan-out mode, LangGraph calls this once per sub-topic
    via Send(), each with its own sub-topic as the query.
    """
    run_id = state.get("run_id", "")
    query  = state.get("query", "")
    event_id, t0 = log_agent_start(run_id, "researcher", {"query": query})

    logger.info(f"[researcher] searching | query: '{query[:60]}'")

    try:
        search_cfg  = get_search_config()
        max_results = search_cfg.get("max_results", 5)

        results = search_web(query, max_results=max_results)

        # Deduplicate by URL — parallel researchers may find the same source
        seen_urls: set[str] = set()
        unique: list[dict]  = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)

        logger.info(f"[researcher] found {len(unique)} unique sources")
        log_agent_end(event_id, run_id, "researcher", t0)

        return {
            "sources":             unique,
            "search_queries_used": [query],
            "pipeline_trace": [{
                "agent":       "researcher",
                "duration_ms": int((time.time() - t0) * 1000),
                "tokens":      0,
                "summary":     f"Found {len(unique)} sources for '{query[:40]}'",
            }],
        }

    except Exception as e:
        logger.error(f"[researcher] failed: {e}")
        log_agent_end(event_id, run_id, "researcher", t0, error=str(e))
        return {
            "sources":             [],
            "search_queries_used": [query],
            "errors":              [f"Researcher error: {e}"],
            "pipeline_trace": [{
                "agent":   "researcher",
                "duration_ms": int((time.time() - t0) * 1000),
                "summary": f"Error: {e}",
            }],
        }