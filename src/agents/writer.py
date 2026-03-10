# src/agents/writer.py

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_max_tokens, get_model
from src.guardrails import check_output
from src.models.state import ResearchState
from src.observability.cost import calculate_cost, extract_token_usage
from src.observability.logger import log_agent_end, log_agent_start, log_cost

load_dotenv()
logger = logging.getLogger(__name__)

_llm = ChatAnthropic(
    model=get_model("writer"),
    max_tokens=get_max_tokens("writer"),
)


def writer_agent(state: ResearchState) -> dict:
    """
    Writer Agent:
    - First call:    writes fresh report from key_claims + synthesis
    - Revision call: rewrites using reviewer feedback
    - Scrubs PII from output before storing
    - Tracks draft versions
    """
    run_id         = state.get("run_id", "")
    query          = state.get("query", "")
    revision_count = state.get("revision_count", 0)
    existing_drafts = state.get("drafts", [])

    event_id, t0 = log_agent_start(run_id, "writer",
                                    {"revision": revision_count})
    logger.info(f"[writer] starting | revision: {revision_count}")

    try:
        if revision_count == 0:
            prompt = _build_initial_prompt(state)
        else:
            prompt = _build_revision_prompt(state)

        response = _llm.invoke([
            SystemMessage(content=(
                "You are a professional research writer. Write clear, "
                "well-structured reports in markdown. Only use the facts "
                "provided — never invent citations."
            )),
            HumanMessage(content=prompt),
        ])

        # Scrub PII before storing
        guardrail_result = check_output(response.content)
        draft_text = guardrail_result.scrubbed_text or response.content

        version   = revision_count + 1
        new_draft = {
            "version":     version,
            "content":     draft_text,
            "char_count":  len(draft_text),
            "pii_scrubbed": guardrail_result.pii_found,
        }

        usage       = extract_token_usage(response)
        cost_record = calculate_cost(
            model=get_model("writer"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            agent_name="writer",
            run_id=run_id,
        )
        log_cost(cost_record)
        log_agent_end(event_id, run_id, "writer", t0,
                      tokens_used=usage.total_tokens,
                      cost_usd=cost_record.cost_usd)

        logger.info(f"[writer] draft v{version} written "
                    f"({len(draft_text)} chars)")

        return {
            "drafts":         existing_drafts + [new_draft],
            "current_draft":  draft_text,
            "revision_count": version,
            "token_count":    usage.total_tokens,
            "cost_usd":       cost_record.cost_usd,
            "pipeline_trace": [{
                "agent":       "writer",
                "duration_ms": int((time.time() - t0) * 1000),
                "tokens":      usage.total_tokens,
                "summary":     f"Draft v{version}: {len(draft_text)} chars",
            }],
        }

    except Exception as e:
        logger.error(f"[writer] failed: {e}")
        log_agent_end(event_id, run_id, "writer", t0, error=str(e))
        return {
            "revision_count": revision_count + 1,  # prevent infinite loop
            "errors":         [f"Writer error: {e}"],
            "pipeline_trace": [{
                "agent":   "writer",
                "duration_ms": int((time.time() - t0) * 1000),
                "summary": f"Error: {e}",
            }],
        }


def _build_initial_prompt(state: ResearchState) -> str:
    claims = state.get("key_claims", [])
    sources = state.get("sources", [])

    claims_text = "\n".join(
        f"- [{c.get('confidence','medium').upper()}] {c['claim']} "
        f"[Source {c.get('source_idx','?')}]"
        for c in claims
    )
    sources_text = "\n".join(
        f"[{i+1}] {s.get('title','Untitled')} — {s.get('url','')}"
        for i, s in enumerate(sources[:10])
    )

    return (
        f"Research question: {state.get('query','')}\n\n"
        f"Key claims:\n{claims_text}\n\n"
        f"Sources:\n{sources_text}\n\n"
        f"Write a structured report with: "
        f"## Executive Summary, ## Key Findings, ## Analysis, "
        f"## Conclusion, ## Sources"
    )


def _build_revision_prompt(state: ResearchState) -> str:
    review      = state.get("review", {})
    issues      = "\n".join(f"- {i}" for i in review.get("issues", []))
    suggestions = "\n".join(f"- {s}" for s in review.get("suggestions", []))
    current     = state.get("current_draft", "")

    return (
        f"Revise this research report to fix all issues listed below.\n\n"
        f"Current draft:\n{current[:3000]}\n\n"
        f"Issues to fix:\n{issues}\n\n"
        f"Suggestions:\n{suggestions}"
    )