from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from src.models.state import ResearchState
import os

load_dotenv()

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024
)


def planner_agent(state: ResearchState) -> ResearchState:
    """
    The Planner Agent:
    - Takes the user's raw query
    - Uses the LLM to rewrite it into a clear, focused research question
    - Stores the refined question back in state as the query
    """

    print(f"\n  Planner Agent starting...")
    print(f"   Original query: '{state.query}'")

    try:
        messages = [
            SystemMessage(content=(
                "You are a research planner. Your job is to take a user's question "
                "and rewrite it as a single, focused, well-scoped research question. "
                "Make it specific and answerable. "
                "Return ONLY the rewritten question, nothing else."
            )),
            HumanMessage(content=f"User question: {state.query}")
        ]

        refined_question = llm.invoke(messages).content.strip()
        print(f"  Refined question: '{refined_question}'")

        # Write the improved question back to state
        return ResearchState(
            query=refined_question,        # ← overwrites with better question
            search_results=state.search_results,
            report=state.report,
            error=None
        )

    except Exception as e:
        print(f"   Planner failed: {e}")
        return ResearchState(
            query=state.query,             # ← fall back to original if error
            search_results=state.search_results,
            report=state.report,
            error=str(e)
        )
