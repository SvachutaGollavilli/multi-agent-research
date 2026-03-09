from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from src.models.state import ResearchState
import os

load_dotenv()

llm = ChatAnthropic(
    model = "claude-haiku-4-5-20251001",
    api_key = os.getenv("ANTHROPIC_API_KEY"),
    max_tokens = 2048
)

def analyst_agent(state:ResearchState) -> ResearchState:
    """
    The Analyst Agent:
        - Reads raw search results from state
        - Uses the LLM to extract key insights and patterns
        - Stores a structured analysis back in store
    """

    print(f" Analyst agent starting")


    try:
        #Format all search results into one block of text
        results_text = ""

        for i, r in enumerate(state.search_results, 1):
            results_text = f"\n [Source {i}]: {r['title']}\n{r['content']}\n"


        messages = [
            SystemMessage(content=(
                """You are an experienced research analyst. Given a set of web search results,
                extract the 5 most important insights relevant to the research question, format
                your response as a numbered list strting from the most important insight.
                Be concise, factful and attache the relevant source from where the insight
                has been made to eliminate halucination.
                """
            )),

            HumanMessage(content=(
                f"Research Query: {state.query}\n\n"
                f"Search results: \n {results_text}"
            ))
        ]


        analysis = llm.invoke(messages).content.strip()
        print(f" Analysis Complete")

        #store this analysis in a temporary report for the writer agent to turn it into a full report
        return ResearchState(
            query = state.query,
            search_results= state.search_results,
            analysis = analysis,
            report = state.report,
            error = None
        )
    
    except Exception as e:
        print(f" Analyst Failed: {e}")
        return ResearchState(
            query = state.query,
            search_results = state.search_results,
            analysis = state.analysis,
            report = state.report,
            error = str(e)
        )