from pydantic import BaseModel, Field
from typing import Optional


class ResearchState(BaseModel):
    """
    The shared state object passed between all agents in the graph.
    Every agent reads from this and writes their output back to it.
    """

    # Input
    query: str = Field(description="The original research question from the user")

    # Agent outputs — populated as the pipeline runs
    search_results: list[dict] = Field(
        default_factory = list,
        description = "Raw search results from the researcher agent"
    )
    analysis: Optional[str] = Field(
        default = None,
        description = "key insights extracted from analysis agent"
    )
    report: Optional[str] = Field(
        default = None,
        description = "Final written report produced by writer agent later"
    )
    error: Optional[str] = Field(
        default = None,
        description = "Any error message if something goes wrong"
    )

    class Config:  
        arbitrary_types_allowed = True
