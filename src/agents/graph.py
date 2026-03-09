from langgraph.graph import StateGraph, END
from src.models.state import ResearchState
from src.agents.planner import planner_agent
from src.agents.researcher import researcher_agent


def build_graph() -> StateGraph:
    """
    Builds and compiles the LangGraph research pipeline.

    Graph flow:
        planner → researcher → END

    Each node receives the full ResearchState and returns
    an updated ResearchState.
    """

    # 1. Create a graph that uses ResearchState as its state schema
    graph = StateGraph(ResearchState)

    # 2. Register nodes — each node is just a function
    graph.add_node("planner", planner_agent)
    graph.add_node("researcher", researcher_agent)

    # 3. Define the flow — edges connect nodes in order
    graph.set_entry_point("planner")       # always start here
    graph.add_edge("planner", "researcher") # planner → researcher
    graph.add_edge("researcher", END)       # researcher → done

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
    print(f" Final query used: {result['query']}")
    print(f" Search results found: {len(result['search_results'])}")
    print("=" * 60)

    for r in result["search_results"]:
        print(f"\n   {r['title']}")
        print(f"   {r['url']}")
        print(f"   {r['content'][:150]}...")
