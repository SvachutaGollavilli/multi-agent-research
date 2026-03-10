# src/agents/analyst.py

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.config import get_max_tokens, get_model
from src.models.state import AnalystOutput, ResearchState
from src.observability.cost import calculate_cost, extract_token_usage
from src.observability.logger import log_agent_end, log_agent_start, log_cost

load_dotenv()
logger = logging.getLogger(__name__)

_llm = ChatAnthropic(
    model=get_model("analyst"),
    max_tokens=get_max_tokens("analyst"),
)

# Max sources fed to the LLM. More sources = more input tokens = slower.
# 5 sources × 150 chars ≈ 750 chars input — fast enough for Haiku.
# The parallel researcher may return 14+ sources but we only need the
# best 5 for claim extraction. Quality over quantity.
_MAX_ANALYST_SOURCES = 5
_MAX_CONTENT_CHARS   = 150


def analyst_agent(state: ResearchState) -> dict:
    """
    Analyst Agent:
    - Caps input to _MAX_ANALYST_SOURCES sources, _MAX_CONTENT_CHARS per source
    - Uses with_structured_output(AnalystOutput, include_raw=True)
    - Retries once on parsed=None (intermittent schema mismatch)
    """
    run_id  = state.get("run_id", "")
    query   = state.get("query", "")
    sources = state.get("sources", [])

    event_id, t0 = log_agent_start(run_id, "analyst", {"source_count": len(sources)})

    if not sources:
        log_agent_end(event_id, run_id, "analyst", t0, error="No sources to analyse")
        return {
            "key_claims": [],
            "conflicts":  [],
            "errors":     ["Analyst: no sources available"],
            "pipeline_trace": [{"agent": "analyst", "duration_ms": 0,
                                "summary": "Skipped — no sources"}],
        }

    # Cap input — take first N sources only.
    # The researcher already URL-deduped and quality-filtered; first N are best.
    capped   = sources[:_MAX_ANALYST_SOURCES]
    skipped  = len(sources) - len(capped)
    if skipped:
        logger.info(
            f"[analyst] capping input: using {len(capped)}/{len(sources)} sources "
            f"(skipped {skipped} to reduce latency)"
        )
    else:
        logger.info(f"[analyst] starting | sources: {len(capped)}")

    for attempt in range(2):
        try:
            structured_llm = _llm.with_structured_output(AnalystOutput, include_raw=True)

            # Truncate content per source — input token reduction
            source_text = "\n".join(
                f"[{i+1}] {s.get('title','Untitled')}: "
                f"{s.get('content','')[:_MAX_CONTENT_CHARS]}"
                for i, s in enumerate(capped)
            )

            raw_result = structured_llm.invoke(
                f"You are a research analyst. Extract 5 key claims from these sources.\n\n"
                f"For each claim: state the fact, cite the source number, "
                f"rate confidence (high/medium/low), include brief evidence.\n"
                f"Identify any contradictions between sources.\n\n"
                f"Research question: {query}\n\nSources:\n{source_text}"
            )

            result: AnalystOutput = raw_result["parsed"]
            raw_message           = raw_result["raw"]

            if result is None:
                parsing_error = raw_result.get("parsing_error", "unknown parse failure")
                logger.warning(
                    f"[analyst] attempt {attempt+1}: parsed=None ({parsing_error}) — "
                    f"{'retrying' if attempt == 0 else 'giving up'}"
                )
                if attempt == 0:
                    continue
                log_agent_end(event_id, run_id, "analyst", t0,
                              error=f"Structured parse failed: {parsing_error}")
                return {
                    "key_claims": [],
                    "conflicts":  [],
                    "errors":     [f"Analyst parse error: {parsing_error}"],
                    "pipeline_trace": [{
                        "agent":       "analyst",
                        "duration_ms": int((time.time() - t0) * 1000),
                        "tokens":      0,
                        "summary":     "Structured parse failed",
                    }],
                }

            claims = [
                {
                    "claim":      c.claim,
                    "source_idx": c.source_idx,
                    "confidence": c.confidence,
                    "evidence":   c.evidence,
                }
                for c in result.claims
            ]
            conflicts = [{"description": c} for c in result.conflicts]

            usage       = extract_token_usage(raw_message)
            cost_record = calculate_cost(
                model=get_model("analyst"),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                agent_name="analyst",
                run_id=run_id,
            )
            log_cost(cost_record)
            log_agent_end(event_id, run_id, "analyst", t0,
                          tokens_used=usage.total_tokens,
                          cost_usd=cost_record.cost_usd)

            logger.info(
                f"[analyst] extracted {len(claims)} claims, {len(conflicts)} conflicts"
            )

            return {
                "key_claims":  claims,
                "conflicts":   conflicts,
                "token_count": usage.total_tokens,
                "cost_usd":    cost_record.cost_usd,
                "pipeline_trace": [{
                    "agent":       "analyst",
                    "duration_ms": int((time.time() - t0) * 1000),
                    "tokens":      usage.total_tokens,
                    "summary":     f"Extracted {len(claims)} claims, "
                                   f"{len(conflicts)} conflicts",
                }],
            }

        except Exception as e:
            logger.error(f"[analyst] attempt {attempt+1} failed: {e}")
            if attempt == 1:
                log_agent_end(event_id, run_id, "analyst", t0, error=str(e))
                return {
                    "key_claims": [],
                    "conflicts":  [],
                    "errors":     [f"Analyst error: {e}"],
                    "pipeline_trace": [{
                        "agent":       "analyst",
                        "duration_ms": int((time.time() - t0) * 1000),
                        "tokens":      0,
                        "summary":     f"Error: {e}",
                    }],
                }

    return {"key_claims": [], "conflicts": [], "errors": ["Analyst: unexpected exit"]}
