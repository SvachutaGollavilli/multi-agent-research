# src/agents/graph.py

from __future__ import annotations

import logging

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from src.models.state import ResearchState, default_state
from src.agents.planner    import planner_agent
from src.agents.researcher import researcher_agent
from src.agents.analyst    import analyst_agent
from src.agents.writer     import writer_agent
from src.observability.logger import start_logger, start_run, end_run
from src.observability.cost   import RunCostAccumulator

load_dotenv()
logger = logging.getLogger(__name__)


def build_graph():
    """
    Build the research pipeline graph.
    Current flow (Phase 6 — linear, observability wired):
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
    Returns the final state dict.
    """
    start_logger()

    run_id = start_run(query)
    acc    = RunCostAccumulator(run_id=run_id)

    try:
        pipeline = build_graph()
        initial  = default_state(query=query, run_id=run_id)
        result   = pipeline.invoke(initial)

        # operator.add reducers accumulated token_count and cost_usd correctly
        # across all agents. Pass them directly — these are the authoritative totals.
        end_run(
            run_id=run_id,
            accumulator=acc,
            final_report=result.get("final_report") or result.get("current_draft", ""),
            status="completed",
            total_tokens=result.get("token_count", 0),
            total_cost=result.get("cost_usd", 0.0),
        )
        return result

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        end_run(run_id=run_id, accumulator=acc, status="failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    query  = input("\n🔬 Enter your research question: ")
    result = run_pipeline(query)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"\nSub-topics researched: {result.get('sub_topics', [])}")
    print(f"Sources found:         {len(result.get('sources', []))}")
    print(f"Claims extracted:      {len(result.get('key_claims', []))}")
    print(f"Total tokens:          {result.get('token_count', 0)}")
    print(f"Total cost:            ${result.get('cost_usd', 0.0):.6f}")
    print(f"\n{'─'*60}")
    print(result.get("current_draft", "No report generated"))

    trace = result.get("pipeline_trace", [])
    print(f"\n{'─'*60}")
    print(f"Pipeline trace ({len(trace)} steps):")
    for step in trace:
        print(f"  {step.get('agent','?'):12s} | "
              f"{step.get('duration_ms',0):5d}ms | "
              f"tokens: {step.get('tokens', 0):5d} | "
              f"{step.get('summary','')}")
