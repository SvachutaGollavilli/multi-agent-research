from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from src.models.state import ResearchState
from src.tools.search import search_web
import os

load_dotenv()

# Initialize LLM once (not inside the function — avoids re-init on every call)
llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),  # ← fixed: api_key, not api_keys
    max_tokens=1024
)


def researcher_agent(state: ResearchState) -> ResearchState:
    """
    The Researcher Agent:
    - Takes the user query from state
    - Asks the LLM to refine the query for better search results
    - Searches the web using Tavily
    - Writes search results back to shared state
    """

    print(f"\n🔍 Researcher Agent starting for query: '{state.query}'")

    try:
        # Step 1: Search the web with the refined query
        results = search_web(state.query, max_results=5)
        print(f"  Found {len(results)} results")

        # Step 2: Write results back to shared state
        return ResearchState(
            query=state.query,
            search_results=results,
            report=state.report,
            error=None
        )

    except Exception as e:
        print(f"  Researcher failed: {e}")
        return ResearchState(
            query=state.query,
            search_results=[],
            report=None,
            error=str(e)
        )


if __name__ == "__main__":
    test_state = ResearchState(query=str(input("Enter your query: ")))
    result_state = researcher_agent(test_state)

    print(f"\n📊 Results found: {len(result_state.search_results)}")
    for r in result_state.search_results:
        print("=" * 50)
        print(f"  📰 {r['title']}")
        print(f"  🔗 {r['url']}")
