from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from src.models.state import ResearchState
import os


load_dotenv()

llm = ChatAnthropic(
    model = "claude-haiku-4-5-20251001",
    api_key = os.getenv("ANTHROPIC_API_KEY"),
    max_tokens = 4096
)


def writer_agent(state:ResearchState) -> ResearchState:
    """
    The writer agent:
     -Reads the insights from analyst agent
     -Writes a clean, structured maekdown research report
     -stores the final report back in state
    """

    print(f"\n Writer agent starting...")

    try:
        messages = [
            SystemMessage(content=(
                """You are an experienced professional research writer. Given a research 
                question and key insights from an analysy, write a well structured research
                report in markdown format.
                Include: an executive summary, main findings, and a conclusion.
                Use headers, bullet points, clean and clear language.
                Aim for 500-600 words
                """
            )),
            HumanMessage(content=(
                f"Research Question: {state.query}\n\n"
                f"Analyst insights:\n{state.analysis}"
            ))
        ]

        report = llm.invoke(messages).content.strip()
        print(f" Report written")

        return ResearchState(
            query = state.query,
            search_results = state.search_results,
            analysis = state.analysis,
            report = report,
            error = None
        )
    
    except Exception as e:
        print(f"Writer Failed: {e}")
        return ResearchState(
            query = state.query,
            search_results = state.search_results,
            analysis = state.analysis,
            report = state.report,
            error = str(e)
        )