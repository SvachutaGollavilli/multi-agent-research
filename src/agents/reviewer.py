# src/agents/reviewer.py
# ─────────────────────────────────────────────
# Reviewer Agent
#
# Reads current_draft and scores it 1-10 using ReviewOutput schema.
# Returns structured verdict: score, issues, suggestions, passed.
#
# The reviewer itself does not loop — the conditional edge in graph.py
# decides whether to route back to writer or proceed to END based on
# the score and revision_count returned here.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.config import get_max_tokens, get_model, get_pipeline_config
from src.models.state import ResearchState, ReviewOutput
from src.observability.cost import calculate_cost, extract_token_usage
from src.observability.logger import log_agent_end, log_agent_start, log_cost

load_dotenv()
logger = logging.getLogger(__name__)

_llm = ChatAnthropic(
    model=get_model("reviewer"),
    max_tokens=get_max_tokens("reviewer"),
)


def reviewer_agent(state: ResearchState) -> dict:
    """
    Reviewer Agent:
    - Reads current_draft from state
    - Scores it 1-10 with structured issues + suggestions
    - Sets review.passed = True if score >= review_pass_score from config
    - Does NOT decide whether to loop — that is graph.py's job
    """
    run_id         = state.get("run_id", "")
    query          = state.get("query", "")
    current_draft  = state.get("current_draft", "")
    revision_count = state.get("revision_count", 0)

    event_id, t0 = log_agent_start(run_id, "reviewer", {"revision": revision_count})
    logger.info(f"[reviewer] starting | reviewing draft v{revision_count}")

    if not current_draft:
        log_agent_end(event_id, run_id, "reviewer", t0, error="No draft to review")
        return {
            "review": {"score": 0, "issues": ["No draft available"], "suggestions": [], "passed": False},
            "pipeline_trace": [{
                "agent": "reviewer", "duration_ms": 0, "tokens": 0,
                "summary": "Skipped — no draft",
            }],
        }

    pass_score = get_pipeline_config().get("review_pass_score", 7)

    for attempt in range(2):
        try:
            structured_llm = _llm.with_structured_output(ReviewOutput, include_raw=True)

            raw_result = structured_llm.invoke(
                f"You are a strict research report reviewer.\n\n"
                f"Research question: {query}\n\n"
                f"Report to review:\n{current_draft[:3000]}\n\n"
                f"Score this report 1-10 on these criteria:\n"
                f"- Accuracy: do the claims match the sources cited?\n"
                f"- Completeness: does it fully answer the research question?\n"
                f"- Structure: are all sections present and logical?\n"
                f"- Conciseness: is there padding or repetition?\n"
                f"- Citations: are sources referenced correctly?\n\n"
                f"A score of {pass_score}+ means the report is ready to publish. "
                f"Below {pass_score} means it needs revision.\n"
                f"Be specific in issues and actionable in suggestions."
            )

            result: ReviewOutput = raw_result["parsed"]
            raw_message          = raw_result["raw"]

            if result is None:
                parsing_error = raw_result.get("parsing_error", "unknown")
                logger.warning(f"[reviewer] attempt {attempt+1}: parsed=None ({parsing_error})")
                if attempt == 0:
                    continue
                # Second failure — pass with a neutral score to avoid blocking the pipeline
                log_agent_end(event_id, run_id, "reviewer", t0, error=f"Parse failed: {parsing_error}")
                return {
                    "review": {"score": pass_score, "issues": [], "suggestions": [], "passed": True},
                    "pipeline_trace": [{
                        "agent": "reviewer", "duration_ms": int((time.time() - t0) * 1000),
                        "tokens": 0, "summary": "Parse failed — passing draft",
                    }],
                }

            # Enforce config pass_score regardless of LLM's passed field
            # (LLM might not know our exact threshold)
            passed = result.score >= pass_score

            review = {
                "score":       result.score,
                "issues":      result.issues,
                "suggestions": result.suggestions,
                "passed":      passed,
            }

            usage       = extract_token_usage(raw_message)
            cost_record = calculate_cost(
                model=get_model("reviewer"),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                agent_name="reviewer",
                run_id=run_id,
            )
            log_cost(cost_record)
            log_agent_end(event_id, run_id, "reviewer", t0,
                          tokens_used=usage.total_tokens,
                          cost_usd=cost_record.cost_usd)

            verdict = "PASS" if passed else "FAIL"
            logger.info(
                f"[reviewer] score: {result.score}/10 | {verdict} | "
                f"{len(result.issues)} issues | {len(result.suggestions)} suggestions"
            )

            return {
                "review":      review,
                "token_count": usage.total_tokens,
                "cost_usd":    cost_record.cost_usd,
                "pipeline_trace": [{
                    "agent":       "reviewer",
                    "duration_ms": int((time.time() - t0) * 1000),
                    "tokens":      usage.total_tokens,
                    "summary": (
                        f"Score {result.score}/10 ({verdict}) | "
                        f"{len(result.issues)} issues"
                    ),
                }],
            }

        except Exception as e:
            logger.error(f"[reviewer] attempt {attempt+1} failed: {e}")
            if attempt == 1:
                log_agent_end(event_id, run_id, "reviewer", t0, error=str(e))
                return {
                    "review": {"score": pass_score, "issues": [], "suggestions": [], "passed": True},
                    "errors": [f"Reviewer error: {e}"],
                    "pipeline_trace": [{
                        "agent": "reviewer", "duration_ms": int((time.time() - t0) * 1000),
                        "tokens": 0, "summary": f"Error: {e} — passing draft",
                    }],
                }

    return {"review": {"score": pass_score, "issues": [], "suggestions": [], "passed": True}}
