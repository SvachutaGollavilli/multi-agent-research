# src/tools/async_search.py
# -----------------------------------------------------------------
# Async versions of the search tools.
#
# async_search_web()       -- Tavily REST API via aiohttp (no blocking I/O)
# async_search_wikipedia() -- Wikipedia via asyncio.to_thread (library is sync,
#                             thread pool prevents blocking the event loop)
#
# Used by researcher_agent to run both searches concurrently:
#   results = await asyncio.gather(
#       async_search_web(topic),
#       async_search_wikipedia(topic),
#   )
#
# Speedup: for "both" tool selection, parallel search saves ~600-900ms
# per researcher call vs the sequential sync version.
# -----------------------------------------------------------------

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from src.config import get_search_config

load_dotenv()
logger = logging.getLogger(__name__)

# Tavily REST endpoint -- same API the SDK wraps
_TAVILY_URL = "https://api.tavily.com/search"


# -----------------------------------------------------------------
# Async Tavily search (aiohttp)
# -----------------------------------------------------------------

async def async_search_web(
    query:       str,
    max_results: int | None = None,
) -> list[dict]:
    """
    Call Tavily's REST API asynchronously using aiohttp.
    Returns the same {title, url, content} shape as the sync version.

    Falls back to an empty list (with a warning) if aiohttp is not
    installed or the request fails -- pipeline continues gracefully.
    """
    try:
        import aiohttp
    except ImportError:
        logger.warning(
            "[async_search] aiohttp not installed -- "
            "falling back to sync Tavily. Run: uv add aiohttp"
        )
        from src.tools.search import search_web
        return await asyncio.to_thread(search_web, query, max_results or 5)

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.error("[async_search] TAVILY_API_KEY not set")
        return []

    cfg         = get_search_config()
    n_results   = max_results or cfg.get("max_results", 5)
    depth       = cfg.get("search_depth", "basic")

    payload = {
        "api_key":      api_key,
        "query":        query,
        "max_results":  n_results,
        "search_depth": depth,
    }

    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(_TAVILY_URL, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results = []
        for r in data.get("results", []):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
            })

        logger.debug(f"[async_search] tavily: {len(results)} results for '{query[:50]}'")
        return results

    except Exception as e:
        logger.warning(f"[async_search] tavily error for '{query[:40]}': {e}")
        return []


# -----------------------------------------------------------------
# Async Wikipedia search (asyncio.to_thread)
# -----------------------------------------------------------------

async def async_search_wikipedia(
    query:       str,
    max_results: int | None = None,
) -> list[dict]:
    """
    Run the synchronous Wikipedia search in a thread pool so it doesn't
    block the event loop. Returns the same {title, url, content} shape.

    asyncio.to_thread() (Python 3.9+) executes the callable in the
    default ThreadPoolExecutor and awaits the result.
    """
    from src.tools.wikipedia import search_wikipedia
    cfg    = get_search_config()
    n_wiki = max_results or cfg.get("max_wiki_results", 3)
    try:
        results = await asyncio.to_thread(search_wikipedia, query, n_wiki)
        logger.debug(f"[async_search] wikipedia: {len(results)} results for '{query[:50]}'")
        return results
    except Exception as e:
        logger.warning(f"[async_search] wikipedia error for '{query[:40]}': {e}")
        return []


# -----------------------------------------------------------------
# Convenience: gather both sources concurrently
# -----------------------------------------------------------------

async def async_search_all(
    query:       str,
    tool:        str = "both",
    max_results: int | None = None,
    max_wiki:    int | None = None,
) -> list[dict]:
    """
    Run Tavily and/or Wikipedia concurrently and return merged, deduped results.

    Args:
        query:       search query
        tool:        "tavily" | "wikipedia" | "both"
        max_results: Tavily result cap
        max_wiki:    Wikipedia result cap

    Returns:
        Deduplicated list of {title, url, content} dicts
    """
    tasks = []
    if tool in ("tavily", "both"):
        tasks.append(async_search_web(query, max_results=max_results))
    if tool in ("wikipedia", "both"):
        tasks.append(async_search_wikipedia(query, max_results=max_wiki))

    if not tasks:
        return []

    # Run both concurrently -- key speedup vs sequential
    raw_lists = await asyncio.gather(*tasks, return_exceptions=True)

    combined: list[dict] = []
    for result in raw_lists:
        if isinstance(result, Exception):
            logger.warning(f"[async_search] gather task failed: {result}")
        elif isinstance(result, list):
            combined.extend(result)

    # URL dedup
    seen:   set        = set()
    unique: list[dict] = []
    for r in combined:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique.append(r)

    return unique


# -----------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        print("Testing async_search_all (both tools concurrently)...")
        results = await async_search_all("What is FAISS?", tool="both")
        print(f"Got {len(results)} unique results")
        for r in results[:3]:
            print(f"  {r['title'][:60]} -- {r['url'][:60]}")

    asyncio.run(main())
