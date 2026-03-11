
# ─────────────────────────────────────────────
# Shared pipeline state — TypedDict with LangGraph reducers.
# Pydantic schemas for structured LLM output (with_structured_output).
# ─────────────────────────────────────────────

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class ResearchState(TypedDict, total=False):
    # ── Run metadata ──────────────────────────
    run_id:  str
    query:   str          # original user question — stable, never modified

    # ── Planner output ────────────────────────
    sub_topics:    list[str]
    research_plan: str

    # ── Send() fan-out fields ─────────────────
    # current_topic is injected per-Send() invocation so each parallel
    # researcher knows which sub-topic to search. Not set in initial state.
    current_topic:  str
    # force_research bypasses cache on quality-gate retries so we don't
    # return the same low-quality cached results on a retry.
    force_research: bool

    # ── Researcher output (parallel-safe) ─────
    # operator.add: three parallel researchers each append their sources;
    # LangGraph concatenates all three lists before the next node runs.
    sources:             Annotated[list[dict], operator.add]
    search_queries_used: Annotated[list[str],  operator.add]

    # ── Quality gate output ───────────────────
    quality_score:   float
    quality_passed:  bool
    quality_retries: int

    # ── Analyst output ────────────────────────
    key_claims: list[dict]
    conflicts:  list[dict]

    # ── Synthesizer output ────────────────────
    synthesis:      str
    source_ranking: list[dict]

    # ── Writer output (versioned) ─────────────
    drafts:         list[dict]
    current_draft:  str
    revision_count: int

    # ── Reviewer output ───────────────────────
    review: dict

    # ── Final output ──────────────────────────
    final_report: str

    # ── Observability (parallel-safe) ─────────
    pipeline_trace: Annotated[list[dict], operator.add]
    errors:         Annotated[list[str],  operator.add]
    token_count:    Annotated[int,        operator.add]
    cost_usd:       Annotated[float,      operator.add]


def default_state(query: str, run_id: str = "") -> dict:
    return {
        "run_id":              run_id,
        "query":               query,
        "sub_topics":          [],
        "research_plan":       "",
        "current_topic":       "",
        "force_research":      False,
        "sources":             [],
        "search_queries_used": [],
        "quality_score":       0.0,
        "quality_passed":      False,
        "quality_retries":     0,
        "key_claims":          [],
        "conflicts":           [],
        "synthesis":           "",
        "source_ranking":      [],
        "drafts":              [],
        "current_draft":       "",
        "revision_count":      0,
        "review":              {},
        "final_report":        "",
        "pipeline_trace":      [],
        "errors":              [],
        "token_count":         0,
        "cost_usd":            0.0,
    }


# ═══════════════════════════════════════════════
# Pydantic schemas for with_structured_output()
# ═══════════════════════════════════════════════

class PlannerOutput(BaseModel):
    sub_topics: list[str] = Field(
        description="1-3 focused, searchable sub-questions derived from the main query",
        min_length=1, max_length=3,
    )
    research_plan: str = Field(
        description="Brief 1-2 sentence strategy for how to approach this research",
    )


class ClaimOutput(BaseModel):
    claim:      str = Field(description="The factual claim extracted from sources")
    source_idx: int = Field(description="1-based index of the supporting source")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    evidence:   str = Field(description="Direct quote or paraphrase supporting this claim")


class AnalystOutput(BaseModel):
    claims:    list[ClaimOutput] = Field(description="5-8 claims extracted with evidence")
    conflicts: list[str]         = Field(
        default_factory=list,
        description="Contradictions found between sources (if any)",
    )


class ReviewOutput(BaseModel):
    score:       int       = Field(description="Quality score 1-10", ge=1, le=10)
    issues:      list[str] = Field(default_factory=list,
                                   description="Specific problems to fix")
    suggestions: list[str] = Field(default_factory=list,
                                   description="Concrete improvement suggestions")
    passed:      bool      = Field(
        description="True if score >= threshold and report is ready for publication"
    )
