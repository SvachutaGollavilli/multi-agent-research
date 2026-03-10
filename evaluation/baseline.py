# evaluation/baseline.py
# ─────────────────────────────────────────────
# Single-agent baseline for eval comparison.
#
# Mimics what a naive one-shot pipeline looks like:
#   1. Search one topic (the original query, no planner decomposition)
#   2. Pass raw results directly to an LLM that writes a report
#   3. No analyst, no synthesizer, no reviewer loop
#
# This is the "before" in the multi-agent "before/after" comparison.
# The delta shows exactly what the pipeline adds in quality terms.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_model, get_max_tokens, get_search_config
from src.tools.search import search_web
from src.tools.wikipedia import search_wikipedia
from src.tools.tool_selector import select_tool

load_dotenv()
logger = logging.getLogger(__name__)

# Baseline uses the same model as writer for fair comparison
_llm = ChatAnthropic(
    model=get_model("writer"),
    max_tokens=get_max_tokens("writer"),
)


def run_baseline(query: str) -> dict:
    """
    Single-agent baseline pipeline.

    Returns:
        {
          "query":      str,
          "report":     str,   # raw markdown text
          "sources":    list,
          "duration_s": float,
          "error":      str | None,
        }
    """
    t0 = time.time()
    logger.info(f"[baseline] starting | query: '{query[:60]}'")

    try:
        search_cfg  = get_search_config()
        max_results = search_cfg.get("max_results", 5)
        max_wiki    = search_cfg.get("max_wiki_results", 3)

        selection = select_tool(query)
        tool      = selection["tool"]
        raw: list[dict] = []

        if tool in ("wikipedia", "both"):
            try:
                raw.extend(search_wikipedia(query, max_results=max_wiki))
            except Exception as e:
                logger.warning(f"[baseline] wikipedia error: {e}")

        if tool in ("tavily", "both"):
            try:
                raw.extend(search_web(query, max_results=max_results))
            except Exception as e:
                logger.warning(f"[baseline] tavily error: {e}")

        # Simple dedup
        seen: set = set()
        sources: list[dict] = []
        for r in raw:
            url = r.get("url", "")
            if url and url not in seen:
                seen.add(url)
                sources.append(r)

        logger.info(f"[baseline] {len(sources)} sources retrieved via {tool}")

        # Build context block — same cap as analyst (5 sources × 150 chars)
        context_lines = []
        for i, s in enumerate(sources[:5], 1):
            title   = s.get("title", "Untitled")
            content = (s.get("content") or s.get("snippet") or "")[:150]
            url     = s.get("url", "")
            context_lines.append(f"[{i}] {title}\n{content}\nURL: {url}")

        context = "\n\n".join(context_lines) or "No sources available."

        prompt = (
            f"Research question: {query}\n\n"
            f"Sources:\n{context}\n\n"
            "Write a focused research report in markdown (250-350 words). "
            "Sections: "
            "## Executive Summary, "
            "## Key Findings, "
            "## Analysis, "
            "## Conclusion, "
            "## Sources. "
            "Only use facts from the provided sources."
        )

        response = _llm.invoke([
            SystemMessage(content=(
                "You are a professional research writer. "
                "Write accurate, concise markdown reports using only provided sources."
            )),
            HumanMessage(content=prompt),
        ])

        report    = response.content
        duration  = round(time.time() - t0, 2)

        logger.info(f"[baseline] done | {len(report)} chars | {duration}s")

        return {
            "query":      query,
            "report":     report,
            "sources":    sources,
            "duration_s": duration,
            "error":      None,
        }

    except Exception as e:
        logger.error(f"[baseline] failed: {e}")
        return {
            "query":      query,
            "report":     "",
            "sources":    [],
            "duration_s": round(time.time() - t0, 2),
            "error":      str(e),
        }
