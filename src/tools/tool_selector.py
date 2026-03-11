# src/tools/tool_selector.py

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Keyword sets ───────────────────────────────
# Wikipedia is better for: definitions, biographies, history, concepts
# Tavily is better for:    recent news, comparisons, how-tos, current state

_WIKIPEDIA_PATTERNS: list[str] = [
    r"\bwho (is|was|invented|discovered|founded|created)\b",
    r"\bwhat is (a |an |the )?\b",
    r"\bhistory of\b",
    r"\bbiography\b",
    r"\bborn in\b",
    r"\bdefinition of\b",
    r"\bdefined as\b",
    r"\borigin of\b",
    r"\bexplain\b",
    r"\boverview of\b",
    r"\bintroduction to\b",
    r"\bconcept of\b",
    r"\btheory of\b",
    r"\balgorithm\b",
    r"\bmathematical\b",
    r"\bformula for\b",
]

_TAVILY_PATTERNS: list[str] = [
    r"\blatest\b",
    r"\brecent\b",
    r"\b202[3-9]\b",  # years 2023-2029
    r"\bcurrent(ly)?\b",
    r"\bnews\b",
    r"\btoday\b",
    r"\bthis (week|month|year)\b",
    r"\bbest\b",
    r"\btop \d+\b",
    r"\bcompare\b",
    r"\bvs\.?\b",
    r"\bprice\b",
    r"\breview\b",
    r"\bhow to\b",
    r"\btutorial\b",
    r"\bstep[- ]by[- ]step\b",
    r"\bguide\b",
]

# Compiled once at import — never inside a function
_WIKI_RE = [re.compile(p, re.IGNORECASE) for p in _WIKIPEDIA_PATTERNS]
_TAVILY_RE = [re.compile(p, re.IGNORECASE) for p in _TAVILY_PATTERNS]


def select_tool(query: str) -> dict:
    """
    Decide which search tool best fits this query.

    Returns a dict:
        {
            "tool":        "wikipedia" | "tavily" | "both",
            "confidence":  0.0-1.0,
            "reason":      "human-readable explanation",
            "wiki_hits":   int,
            "tavily_hits": int,
        }

    Rules:
    - wiki_hits > 0 and tavily_hits == 0  → wikipedia (confident)
    - tavily_hits > 0 and wiki_hits == 0  → tavily    (confident)
    - both hit                            → both      (split signals)
    - neither hits                        → tavily    (default — live web safer)
    """
    wiki_hits = sum(1 for r in _WIKI_RE if r.search(query))
    tavily_hits = sum(1 for r in _TAVILY_RE if r.search(query))

    if wiki_hits > 0 and tavily_hits == 0:
        tool = "wikipedia"
        confidence = min(0.6 + wiki_hits * 0.1, 0.95)
        reason = f"matched {wiki_hits} wikipedia indicator(s)"

    elif tavily_hits > 0 and wiki_hits == 0:
        tool = "tavily"
        confidence = min(0.6 + tavily_hits * 0.1, 0.95)
        reason = f"matched {tavily_hits} tavily indicator(s)"

    elif wiki_hits > 0 and tavily_hits > 0:
        tool = "both"
        confidence = 0.5
        reason = f"mixed signals: {wiki_hits} wiki + {tavily_hits} tavily hits"

    else:
        # No keywords matched — default to Tavily (live web is safer default)
        tool = "tavily"
        confidence = 0.5
        reason = "no keyword match — defaulting to tavily"

    result = {
        "tool": tool,
        "confidence": round(confidence, 2),
        "reason": reason,
        "wiki_hits": wiki_hits,
        "tavily_hits": tavily_hits,
    }

    logger.debug(
        f"[tool_selector] '{query[:50]}' → {tool} "
        f"(conf: {confidence:.2f}, reason: {reason})"
    )
    return result


if __name__ == "__main__":
    test_queries = [
        "what is the attention mechanism",
        "latest AI research 2024",
        "history of neural networks",
        "best vector databases compared",
        "who invented the transformer architecture",
        "how to fine-tune a language model",
        "quantum computing",
    ]
    print(f"{'Query':<45} {'Tool':<12} {'Conf':<6} Reason")
    print("─" * 90)
    for q in test_queries:
        r = select_tool(q)
        print(f"{q:<45} {r['tool']:<12} {r['confidence']:<6} {r['reason']}")
