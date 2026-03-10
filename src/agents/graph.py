# src/agents/graph.py

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.agents.analyst     import analyst_agent
from src.agents.planner     import planner_agent
from src.agents.researcher  import researcher_agent
from src.agents.reviewer    import reviewer_agent
from src.agents.synthesizer import synthesizer_agent
from src.agents.writer      import writer_agent
from src.config             import get_pipeline_config
from src.models.state       import ResearchState, default_state
from src.observability.cost import RunCostAccumulator
from src.observability.logger import end_run, start_logger, start_run
from src.output.report_writer import write_report

load_dotenv()
logger = logging.getLogger(__name__)


def _should_revise(state: ResearchState) -> str:
    """
    Conditional routing function called after reviewer.

    Returns "end"    → pipeline proceeds to END (quality passed or budget exhausted)
    Returns "revise" → pipeline loops back to writer for another draft

    Two exit conditions prevent infinite loops:
      1. Quality exit:  review score >= review_pass_score (default 7)
      2. Safety exit:   revision_count >= max_revisions (default 2)
         Even if quality is still low, we stop after max_revisions attempts.
    """
    review         = state.get("review", {})
    score          = review.get("score", 0)
    passed         = review.get("passed", False)
    revision_count = state.get("revision_count", 0)

    cfg           = get_pipeline_config()
    max_revisions = cfg.get("max_revisions", 2)
    pass_score    = cfg.get("review_pass_score", 7)

    if passed or score >= pass_score:
        logger.info(f"[graph] reviewer PASS (score={score}/{pass_score}) → END")
        return "end"

    if revision_count >= max_revisions:
        logger.info(
            f"[graph] max revisions reached ({revision_count}/{max_revisions}), "
            f"score={score} — proceeding to END"
        )
        return "end"

    logger.info(
        f"[graph] reviewer FAIL (score={score}/{pass_score}) — "
        f"revision {revision_count}/{max_revisions} → writer"
    )
    return "revise"


def build_graph():
    """
    Research pipeline graph:

        planner → researcher → analyst → synthesizer → writer ─┐
                                                                 ↓
                                                             reviewer
                                                            /         \\
                                              (pass or max revisions)  (fail, retry)
                                                       ↓                    ↓
                                                      END               writer (loop)

    The writer→reviewer→writer loop runs at most max_revisions times
    (configured in base.yaml pipeline.max_revisions).
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner",     planner_agent)
    graph.add_node("researcher",  researcher_agent)
    graph.add_node("analyst",     analyst_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.add_node("writer",      writer_agent)
    graph.add_node("reviewer",    reviewer_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner",     "researcher")
    graph.add_edge("researcher",  "analyst")
    graph.add_edge("analyst",     "synthesizer")
    graph.add_edge("synthesizer", "writer")
    graph.add_edge("writer",      "reviewer")

    # Conditional edge: reviewer decides END or loop back to writer
    graph.add_conditional_edges(
        "reviewer",
        _should_revise,
        {
            "end":    END,
            "revise": "writer",
        },
    )

    return graph.compile()


def run_pipeline(query: str) -> dict:
    start_logger()
    run_id = start_run(query)
    acc    = RunCostAccumulator(run_id=run_id)

    try:
        pipeline = build_graph()
        initial  = default_state(query=query, run_id=run_id)
        result   = pipeline.invoke(initial)

        end_run(
            run_id=run_id,
            accumulator=acc,
            final_report=result.get("final_report") or result.get("current_draft", ""),
            status="completed",
            total_tokens=result.get("token_count", 0),
            total_cost=result.get("cost_usd", 0.0),
        )

        report_path = write_report(result, run_id=run_id)
        logger.info(f"Report saved → {report_path}")

        review         = result.get("review", {})
        revision_count = result.get("revision_count", 0)

        trace = result.get("pipeline_trace", [])
        logger.info("─" * 60)
        logger.info(f"Pipeline summary | run: {run_id[:8]}...")
        logger.info(f"  sub-topics  : {len(result.get('sub_topics', []))}")
        logger.info(f"  sources     : {len(result.get('sources', []))}")
        logger.info(f"  claims      : {len(result.get('key_claims', []))}")
        logger.info(f"  synthesis   : {len(result.get('synthesis', ''))} chars")
        logger.info(f"  revisions   : {revision_count}")
        logger.info(f"  final score : {review.get('score', '?')}/10")
        logger.info(f"  tokens      : {result.get('token_count', 0)}")
        logger.info(f"  cost        : ${result.get('cost_usd', 0.0):.6f}")
        logger.info(f"  output      : {report_path}")
        logger.info("─" * 60)
        for step in trace:
            logger.info(
                f"  {step.get('agent','?'):12s} | "
                f"{step.get('duration_ms', 0):5d}ms | "
                f"tokens: {step.get('tokens', 0):5d} | "
                f"{step.get('summary', '')}"
            )
        logger.info("─" * 60)

        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        end_run(run_id=run_id, accumulator=acc, status="failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    query = input("\n🔬 Enter your research question: ")
    run_pipeline(query)
