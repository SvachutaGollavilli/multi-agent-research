
# ─────────────────────────────────────────────
# Shared pipeline state — TypedDict with LangGraph reducers.
# Pydantic schemas for structured LLM output (with_structured_output).
# ─────────────────────────────────────────────

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════
# Reducer helpers
# ═══════════════════════════════════════════════
# Fields marked Annotated[list, operator.add] are MERGED (not overwritten)
# when multiple agents write to them concurrently via Send() fan-out.
# Fields with no annotation are simply overwritten by the latest writer.

class ResearchState(TypedDict, total=False):
    """
    Shared state flowing through the entire LangGraph pipeline.

    TypedDict rules:
      total=False  → all fields are optional at dict construction time.
                     LangGraph starts with default_state() and agents
                     return only the fields they actually changed.

    Annotated[list, operator.add] → parallel-safe accumulator.
      Multiple researchers writing concurrently will have their
      lists concatenated, not overwritten.
    """

    # ── Run metadata ──────────────────────────
    run_id:  str        # UUID assigned at pipeline entry
    query:   str        # original user question (never modified after planner)

    # ── Planner output ────────────────────────
    sub_topics:    list[str]   # 1-3 focused sub-questions to research in parallel
    research_plan: str         # brief strategy from planner

    # ── Researcher output (parallel-safe) ─────
    # operator.add means: if researcher_1 returns {"sources": [a,b]}
    # and researcher_2 returns {"sources": [c,d]}, LangGraph stores [a,b,c,d]
    sources:             Annotated[list[dict], operator.add]
    search_queries_used: Annotated[list[str],  operator.add]

    # ── Quality gate output ───────────────────
    quality_score:  float
    quality_passed: bool

    # ── Analyst output ────────────────────────
    key_claims: list[dict]   # structured claims with confidence + source ref
    conflicts:  list[dict]   # contradictions detected across sources

    # ── Synthesizer output ────────────────────
    synthesis:      str
    source_ranking: list[dict]

    # ── Writer output (versioned) ─────────────
    drafts:        list[dict]  # all versions — [{version, content, char_count}]
    current_draft: str
    revision_count: int

    # ── Reviewer output ───────────────────────
    review: dict               # {score, issues, suggestions, passed}

    # ── Final output ──────────────────────────
    final_report: str

    # ── Observability (parallel-safe) ─────────
    # operator.add lets parallel agents append their own trace entries
    pipeline_trace: Annotated[list[dict], operator.add]
    errors:         Annotated[list[str],  operator.add]
    token_count:    Annotated[int,        operator.add]  # accumulates across agents
    cost_usd:       Annotated[float,      operator.add]  # accumulates across agents


def default_state(query: str, run_id: str = "") -> dict:
    """
    Create a fully initialised state dict.
    Pass this to pipeline.invoke() — never pass a partial dict.
    All Annotated[list/int/float] fields must start as their
    identity element ([] or 0) so operator.add works correctly.
    """
    return {
        "run_id":             run_id,
        "query":              query,
        "sub_topics":         [],
        "research_plan":      "",
        "sources":            [],       # operator.add — must start as []
        "search_queries_used": [],      # operator.add — must start as []
        "quality_score":      0.0,
        "quality_passed":     False,
        "key_claims":         [],
        "conflicts":          [],
        "synthesis":          "",
        "source_ranking":     [],
        "drafts":             [],
        "current_draft":      "",
        "revision_count":     0,
        "review":             {},
        "final_report":       "",
        "pipeline_trace":     [],       # operator.add — must start as []
        "errors":             [],       # operator.add — must start as []
        "token_count":        0,        # operator.add — must start as 0
        "cost_usd":           0.0,      # operator.add — must start as 0.0
    }


# ═══════════════════════════════════════════════
# Pydantic schemas for with_structured_output()
# ═══════════════════════════════════════════════
# These are used by agents that need guaranteed JSON structure
# from the LLM. with_structured_output() enforces the schema
# at the LangChain level — no manual JSON parsing needed.

class PlannerOutput(BaseModel):
    """Structured output from the Planner agent."""
    sub_topics: list[str] = Field(
        description="1-3 focused, searchable sub-questions derived from the main query",
        min_length=1,
        max_length=3,
    )
    research_plan: str = Field(
        description="Brief 1-2 sentence strategy for how to approach this research",
    )


class ClaimOutput(BaseModel):
    """A single extracted claim with evidence and confidence."""
    claim:      str = Field(description="The factual claim extracted from sources")
    source_idx: int = Field(description="1-based index of the supporting source")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    evidence:   str = Field(description="Direct quote or paraphrase supporting this claim")


class AnalystOutput(BaseModel):
    """Structured output from the Analyst agent."""
    claims:    list[ClaimOutput] = Field(description="5-8 claims extracted with evidence")
    conflicts: list[str]         = Field(
        default_factory=list,
        description="Contradictions found between sources (if any)",
    )


class ReviewOutput(BaseModel):
    """Structured output from the Reviewer agent."""
    score:       int        = Field(description="Quality score 1-10", ge=1, le=10)
    issues:      list[str]  = Field(default_factory=list,
                                    description="Specific problems to fix")
    suggestions: list[str]  = Field(default_factory=list,
                                    description="Concrete improvement suggestions")
    passed:      bool       = Field(
        description="True if score >= 7 and report is ready for publication"
    )