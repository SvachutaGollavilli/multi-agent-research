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


def analyst_agent(state: ResearchState) -> dict:
    """
    Analyst Agent:
    - Reads sources from state (merged from parallel researchers)
    - Extracts structured claims with confidence + evidence
    - Uses with_structured_output(AnalystOutput) — no JSON parsing needed
    """
    run_id  = state.get("run_id", "")
    query   = state.get("query", "")
    sources = state.get("sources", [])

    event_id, t0 = log_agent_start(run_id, "analyst", {"source_count": len(sources)})
    logger.info(f"[analyst] starting | sources: {len(sources)}")

    if not sources:
        log_agent_end(event_id, run_id, "analyst", t0,
                      error="No sources to analyse")
        return {
            "key_claims": [],
            "conflicts":  [],
            "errors":     ["Analyst: no sources available"],
            "pipeline_trace": [{"agent": "analyst", "duration_ms": 0,
                                 "summary": "Skipped — no sources"}],
        }

    try:
        structured_llm = _llm.with_structured_output(AnalystOutput, include_raw = True)

        source_text = "\n".join(
            f"[{i+1}] {s.get('title','Untitled')}: {s.get('content','')[:300]}"
            for i, s in enumerate(sources[:10])
        )

        raw_result = structured_llm.invoke(
            f"You are a research analyst. Extract 5-8 key claims from these sources.\n\n"
            f"For each claim: state the fact, reference the source number, "
            f"rate confidence (high/medium/low), and include supporting evidence.\n"
            f"Also identify contradictions between sources.\n\n"
            f"Research question: {query}\n\nSources:\n{source_text}"
        )

        result: AnalystOutput = raw_result["parsed"]
        raw_message = raw_result["raw"]

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

        logger.info(f"[analyst] extracted {len(claims)} claims, "
                    f"{len(conflicts)} conflicts")

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
        logger.error(f"[analyst] failed: {e}")
        log_agent_end(event_id, run_id, "analyst", t0, error=str(e))
        return {
            "key_claims": [],
            "conflicts":  [],
            "errors":     [f"Analyst error: {e}"],
            "pipeline_trace": [{
                "agent":   "analyst",
                "duration_ms": int((time.time() - t0) * 1000),
                "summary": f"Error: {e}",
            }],
        }