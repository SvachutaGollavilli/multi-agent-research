# src/agents/graph.py

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.agents.analyst    import analyst_agent
from src.agents.planner    import planner_agent
from src.agents.researcher import researcher_agent
from src.agents.writer     import writer_agent
from src.models.state      import ResearchState, default_state
from src.observability.cost   import RunCostAccumulator
from src.observability.logger import end_run, start_logger, start_run
from src.output.report_writer import write_report

load_dotenv()
logger = logging.getLogger(__name__)


def build_graph():
    """
    Build the research pipeline graph.
    Current flow (linear, observability wired):
        planner → researcher → analyst → writer → END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner",    planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyst",    analyst_agent)
    graph.add_node("writer",     writer_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst",    "writer")
    graph.add_edge("writer",     END)

    return graph.compile()


def run_pipeline(query: str) -> dict:
    """
    Run the full pipeline with observability.
    Saves report to results/ as a Word document.
    Returns the final state dict.
    """
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

        # Write report to Word document
        report_path = write_report(result, run_id=run_id)
        logger.info(f"Report saved → {report_path}")

        # ── Summary log (system info only — no report content on console) ──
        trace = result.get("pipeline_trace", [])
        logger.info("─" * 60)
        logger.info(f"Pipeline summary | run: {run_id[:8]}...")
        logger.info(f"  sub-topics : {len(result.get('sub_topics', []))}")
        logger.info(f"  sources    : {len(result.get('sources', []))}")
        logger.info(f"  claims     : {len(result.get('key_claims', []))}")
        logger.info(f"  tokens     : {result.get('token_count', 0)}")
        logger.info(f"  cost       : ${result.get('cost_usd', 0.0):.6f}")
        logger.info(f"  output     : {report_path}")
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
