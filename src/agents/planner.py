# src/agents/planner.py

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.config import get_max_tokens, get_model
from src.models.state import PlannerOutput, ResearchState
from src.observability.cost import calculate_cost, extract_token_usage
from src.observability.logger import log_agent_end, log_agent_start, log_cost

load_dotenv()
logger = logging.getLogger(__name__)

# ── LLM singleton ──────────────────────────────
# Created once at import — not inside the function.
# config is read here so if YAML changes, reload the module.
_llm = ChatAnthropic(
    model=get_model("planner"),
    max_tokens=get_max_tokens("planner"),
)


def planner_agent(state: ResearchState) -> dict:
    """
    Planner Agent:
    - Decomposes the query into 1-3 focused sub-topics
    - Returns sub_topics + research_plan for the parallel researchers
    - Uses with_structured_output() — guaranteed PlannerOutput schema
    """
    run_id = state.get("run_id", "")
    query  = state.get("query", "")
    event_id, t0 = log_agent_start(run_id, "planner", {"query": query})

    logger.info(f"[planner] starting | query: '{query[:60]}'")

    try:
        from src.config import get_pipeline_config
        max_topics = get_pipeline_config().get("max_sub_topics", 3)

        # include_raw=True returns {"raw": AIMessage, "parsed": PlannerOutput}
        structured_llm = _llm.with_structured_output(PlannerOutput, include_raw=True)

        raw_result = structured_llm.invoke(
            f"You are a research planner. Break this query into "
            f"1-{max_topics} focused, searchable sub-questions.\n\n"
            f"Query: {query}\n\n"
            f"Return {max_topics} sub-topics and a brief research strategy."
            )

        # parsed is the PlannerOutput Pydantic object
        # raw is the AIMessage that carries usage_metadata
        result: PlannerOutput = raw_result["parsed"]
        raw_message           = raw_result["raw"]

        sub_topics = result.sub_topics[:max_topics]
        logger.info(f"[planner] decomposed into {len(sub_topics)} sub-topics")

        # Now extract from the raw AIMessage — this has usage_metadata
        usage = extract_token_usage(raw_message)
        cost_record = calculate_cost(
            model=get_model("planner"),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            agent_name="planner",
            run_id=run_id,
        )
        log_cost(cost_record)
        log_agent_end(event_id, run_id, "planner", t0,
                      tokens_used=usage.total_tokens,
                      cost_usd=cost_record.cost_usd)

        return {
            "sub_topics":    sub_topics,
            "research_plan": result.research_plan,
            "token_count":   usage.total_tokens,
            "cost_usd":      cost_record.cost_usd,
            "pipeline_trace": [{
                "agent":    "planner",
                "duration_ms": int((time.time() - t0) * 1000),
                "tokens":   usage.total_tokens,
                "summary":  f"Decomposed into {len(sub_topics)} sub-topics",
            }],
        }

    except Exception as e:
        logger.error(f"[planner] failed: {e}")
        log_agent_end(event_id, run_id, "planner", t0, error=str(e))
        # Fallback — use original query as single sub-topic
        return {
            "sub_topics":    [query],
            "research_plan": "Direct research (planner error)",
            "errors":        [f"Planner error: {e}"],
            "pipeline_trace": [{
                "agent":   "planner",
                "duration_ms": int((time.time() - t0) * 1000),
                "summary": "Error — fallback to direct query",
            }],
        }
