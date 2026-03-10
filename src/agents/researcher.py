# src/agents/researcher.py

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from src.cache.research_cache import fetch as cache_fetch, store as cache_store
from src.config import get_search_config
from src.models.state import ResearchState
from src.observability.logger import log_agent_end, log_agent_start
from src.tools.search import search_web
from src.tools.tool_selector import select_tool
from src.tools.wikipedia import search_wikipedia

load_dotenv()
logger = logging.getLogger(__name__)


def _search_one_topic(topic: str, max_results: int, max_wiki: int) -> dict:
    """
    Search a single topic. Stateless — runs in a thread.
    NOTE: Does NOT check/write cache here. Cache is managed at the
    original-query level in researcher_agent() to ensure cache keys
    are stable across runs (planner sub-topics are LLM-generated and
    non-deterministic, so they can't be used as reliable cache keys).
    """
    selection = select_tool(topic)
    tool      = selection["tool"]
    raw: list[dict] = []

    if tool in ("wikipedia", "both"):
        try:
            raw.extend(search_wikipedia(topic, max_results=max_wiki))
        except Exception as e:
            logger.warning(f"[researcher] wikipedia failed for '{topic[:40]}': {e}")

    if tool in ("tavily", "both"):
        try:
            raw.extend(search_web(topic, max_results=max_results))
        except Exception as e:
            logger.warning(f"[researcher] tavily failed for '{topic[:40]}': {e}")

    # URL dedup within this topic's results
    seen: set = set()
    unique: list[dict] = []
    for r in raw:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)

    return {"topic": topic, "results": unique, "tool": tool}


def researcher_agent(state: ResearchState) -> dict:
    """
    Researcher Agent — cache-first, then parallel sub-topic search.

    Cache strategy:
      Key  = original query (stable across runs — never changes)
      NOT  = sub-topic strings (LLM-generated, different every run)
      On hit  → return immediately, skip all searches
      On miss → fan-out parallel searches, store result under original query

    Parallel strategy:
      ThreadPoolExecutor fans out one thread per sub-topic.
      Wall-clock time = slowest single search, not their sum.
    """
    run_id     = state.get("run_id", "")
    query      = state.get("query", "")       # original query — stable cache key
    sub_topics = state.get("sub_topics") or [query]

    event_id, t0 = log_agent_start(run_id, "researcher", {
        "query":      query,
        "sub_topics": len(sub_topics),
    })

    try:
        search_cfg  = get_search_config()
        max_results = search_cfg.get("max_results", 5)
        max_wiki    = search_cfg.get("max_wiki_results", 3)

        # ── Layer 1: Cache check on ORIGINAL query ────────────────────────
        # The original query never changes between runs for the same question.
        # Sub-topic strings do change (LLM non-determinism) so we never key on those.
        cached = cache_fetch(query)
        if cached is not None:
            elapsed_ms = int((time.time() - t0) * 1000)
            logger.info(
                f"[researcher] cache HIT on original query | "
                f"{len(cached)} sources | {elapsed_ms}ms"
            )
            log_agent_end(event_id, run_id, "researcher", t0)
            return {
                "sources":             cached,
                "search_queries_used": [query],
                "pipeline_trace": [{
                    "agent":       "researcher",
                    "duration_ms": elapsed_ms,
                    "tokens":      0,
                    "summary":     f"Cache HIT: {len(cached)} sources for '{query[:40]}'",
                }],
            }

        # ── Layer 2: Parallel fan-out across sub-topics ───────────────────
        logger.info(f"[researcher] cache MISS — searching {len(sub_topics)} sub-topics in parallel")
        all_results: list[dict] = []
        queries_used: list[str] = []

        with ThreadPoolExecutor(max_workers=len(sub_topics)) as executor:
            futures = {
                executor.submit(_search_one_topic, topic, max_results, max_wiki): topic
                for topic in sub_topics
            }
            for future in as_completed(futures):
                topic = futures[future]
                try:
                    r = future.result()
                    all_results.extend(r["results"])
                    queries_used.append(r["topic"])
                    logger.debug(
                        f"[researcher] '{topic[:35]}' → "
                        f"{r['tool']} ({len(r['results'])} results)"
                    )
                except Exception as e:
                    logger.error(f"[researcher] sub-topic '{topic[:40]}' failed: {e}")

        # ── Global URL dedup ──────────────────────────────────────────────
        seen_urls: set = set()
        unique: list[dict] = []
        for r in all_results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(r)

        # ── Store under ORIGINAL query — stable key for future runs ───────
        if unique:
            cache_store(query, unique, tool_used="parallel")

        elapsed_ms = int((time.time() - t0) * 1000)
        logger.info(
            f"[researcher] done | {len(unique)} unique sources | "
            f"{elapsed_ms}ms | stored in cache"
        )
        log_agent_end(event_id, run_id, "researcher", t0)

        return {
            "sources":             unique,
            "search_queries_used": queries_used,
            "pipeline_trace": [{
                "agent":       "researcher",
                "duration_ms": elapsed_ms,
                "tokens":      0,
                "summary": (
                    f"{len(unique)} sources from {len(sub_topics)} parallel searches"
                ),
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
                "agent":       "researcher",
                "duration_ms": int((time.time() - t0) * 1000),
                "tokens":      0,
                "summary":     f"Error: {e}",
            }],
        }
