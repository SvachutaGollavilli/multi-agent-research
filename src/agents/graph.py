from langgraph.graph import StateGraph, END
from src.models.state import ResearchState
from src.agents.planner import planner_agent
from src.agents.researcher import researcher_agent
from src.agents.analyst import analyst_agent
from src.agents.writer import writer_agent


def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph research pipeline.

    Graph flow:
        planner → researcher → analyst -> writer -> END

    Each node receives the full ResearchState and returns
    an updated ResearchState.
    """

    # 1. Create a graph that uses ResearchState as its state schema
    graph = StateGraph(ResearchState)

    # 2. Register nodes — each node is just a function
    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("analyst", analyst_agent)
    graph.add_node("writer", writer_agent)

    # 3. Define the flow — edges connect nodes in order
    graph.set_entry_point("planner")       # always start here
    graph.add_edge("planner", "researcher") # planner → researcher
    graph.add_edge("researcher", "analyst") # researcher → analyst
    graph.add_edge("analyst", "writer")       # analyst → writer
    graph.add_edge("writer", END)  #writer → END

    # 4. Compile — turns the graph definition into a runnable object
    return graph.compile()


if __name__ == "__main__":
    # Manual end-to-end test
    # Run with: uv run python src/agents/graph.py

    pipeline = build_graph()

    query = input("\n Enter your research question: ")

    print("\n" + "=" * 60)
    print(" Running pipeline...")
    print("=" * 60)

    # LangGraph expects a dict matching your state fields
    result = pipeline.invoke({"query": query})

    print("\n" + "=" * 60)
    print(" Pipeline complete!")
    print("=" * 60)
    print(f"\n {result['report']}")
