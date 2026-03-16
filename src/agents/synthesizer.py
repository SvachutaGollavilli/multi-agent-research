# src/agents/synthesizer.py
# ─────────────────────────────────────────────
# Synthesizer Agent
#
# Sits between Analyst and Writer in the pipeline.
#
# Input:  key_claims (list of extracted facts with confidence/source)
#         conflicts  (contradictions detected by analyst)
#         sources    (raw search results with title/url)
#
# Output: synthesis      — unified narrative paragraph(s) that cross-reference
#                          claims, acknowledge conflicts, and form a coherent
#                          story the writer can build a report around
#         source_ranking — sources ordered by relevance to the claims found,
#                          so the writer cites the strongest sources first
#
# Why this step exists:
#   The analyst extracts individual facts in isolation.
#   The writer needs a connected story, not a list of bullet points.
#   The synthesizer bridges that gap — it "thinks through" the claims,
#   spots themes and relationships, and hands the writer a pre-digested
#   narrative rather than raw data.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_max_tokens, get_model
from src.models.state import ResearchState
from src.observability.cost import calculate_cost, extract_token_usage
from src.observability.logger import log_agent_end, log_agent_start, log_cost

load_dotenv()
logger = logging.getLogger(__name__)

_llm = ChatAnthropic(
    model=get_model("synthesizer"),
    max_tokens=get_max_tokens("synthesizer"),
)


def synthesizer_agent(state: ResearchState) -> dict:
    """
    Synthesizer Agent:
    1. Groups claims by theme
    2. Resolves or surfaces conflicts explicitly
    3. Produces a synthesis paragraph (prose, not bullets)
    4. Ranks sources by how many high-confidence claims they support
    """
    run_id = state.get("run_id", "")
    query = state.get("query", "")
    key_claims = state.get("key_claims", [])
    conflicts = state.get("conflicts", [])
    sources = state.get("sources", [])

    event_id, t0 = log_agent_start(
        run_id,
        "synthesizer",
        {
            "claims": len(key_claims),
            "conflicts": len(conflicts),
        },
    )
    logger.info(
        f"[synthesizer] starting | claims: {len(key_claims)} | "
        f"conflicts: {len(conflicts)}"
    )

    # ── Fast path: nothing to synthesise ─────────────────────────────────
    if not key_claims:
        log_agent_end(
            event_id, run_id, "synthesizer", t0, error="No claims to synthesise"
        )
        return {
            "synthesis": "",
            "source_ranking": [],
            "errors": ["Synthesizer: no claims available"],
            "pipeline_trace": [
                {
                    "agent": "synthesizer",
                    "duration_ms": 0,
                    "tokens": 0,
                    "summary": "Skipped — no claims",
                }
            ],
        }

    try:
        # ── Build prompt ──────────────────────────────────────────────────
        claims_text = "\n".join(
            f"[{i + 1}] [{c.get('confidence', 'medium').upper()}] {c['claim']} "
            f"(Source {c.get('source_idx', '?')}): {c.get('evidence', '')[:120]}"
            for i, c in enumerate(key_claims)
        )
        conflicts_text = (
            "\n".join(f"- {c.get('description', c)}" for c in conflicts)
            if conflicts
            else "None detected."
        )
        sources_text = "\n".join(
            f"[{i + 1}] {s.get('title', 'Untitled')} — {s.get('url', '')}"
            for i, s in enumerate(sources[:10])
        )

        response = _llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a research synthesizer. Your job is to take a list of "
                        "extracted claims and produce a coherent, flowing narrative. "
                        "Do NOT just restate the claims as bullets — connect them, "
                        "find the underlying themes, and surface what they collectively mean. "
                        "Be concise and factual. Never invent information."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Research question: {query}\n\n"
                        f"Extracted claims:\n{claims_text}\n\n"
                        f"Conflicts/contradictions:\n{conflicts_text}\n\n"
                        f"Sources:\n{sources_text}\n\n"
                        "Write a synthesis of 150-200 words that:\n"
                        "1. Identifies the 2-3 central themes across the claims\n"
                        "2. Explains how the claims relate to and reinforce each other\n"
                        "3. Explicitly addresses any conflicts — do sources disagree? Why?\n"
                        "4. Ends with a one-sentence conclusion answering the research question\n\n"
                        "Write prose only — no bullet points, no headers."
                    )
                ),
            ]
        )

        synthesis = response.content.strip()

        # ── Source ranking ────────────────────────────────────────────────
        # Score each source by: how many HIGH-confidence claims cite it.
        # This tells the writer which sources to lead with.
        source_scores: dict[int, int] = {}
        for c in key_claims:
            idx = c.get("source_idx", 0)
            if idx and c.get("confidence", "").lower() == "high":
                source_scores[idx] = source_scores.get(idx, 0) + 1

        # Build ranked list (1-based source indices from analyst)
        ranked = sorted(source_scores.items(), key=lambda x: x[1], reverse=True)
        source_ranking = [
            {
                "source_idx": idx,
                "title": sources[idx - 1].get("title", "Untitled")
                if idx <= len(sources)
                else f"Source {idx}",
                "url": sources[idx - 1].get("url", "") if idx <= len(sources) else "",
                "high_conf_claims": count,
            }
            for idx, count in ranked
        ]

        # ── Observability ─────────────────────────────────────────────────
        usage = extract_token_usage(response)
        cost_record = calculate_cost(
            model=get_model("synthesizer"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            agent_name="synthesizer",
            run_id=run_id,
        )
        log_cost(cost_record)
        log_agent_end(
            event_id,
            run_id,
            "synthesizer",
            t0,
            tokens_used=usage.total_tokens,
            cost_usd=cost_record.cost_usd,
        )

        logger.info(
            f"[synthesizer] done | synthesis: {len(synthesis)} chars | "
            f"top sources ranked: {len(source_ranking)}"
        )

        return {
            "synthesis": synthesis,
            "source_ranking": source_ranking,
            "token_count": usage.total_tokens,
            "cost_usd": cost_record.cost_usd,
            "pipeline_trace": [
                {
                    "agent": "synthesizer",
                    "duration_ms": int((time.time() - t0) * 1000),
                    "tokens": usage.total_tokens,
                    "summary": (
                        f"Synthesised {len(key_claims)} claims into "
                        f"{len(synthesis)} chars | "
                        f"{len(source_ranking)} sources ranked"
                    ),
                }
            ],
        }

    except Exception as e:
        logger.error(f"[synthesizer] failed: {e}")
        log_agent_end(event_id, run_id, "synthesizer", t0, error=str(e))
        return {
            "synthesis": "",
            "source_ranking": [],
            "errors": [f"Synthesizer error: {e}"],
            "pipeline_trace": [
                {
                    "agent": "synthesizer",
                    "duration_ms": int((time.time() - t0) * 1000),
                    "tokens": 0,
                    "summary": f"Error: {e}",
                }
            ],
        }
