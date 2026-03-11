# tests/test_state.py
# Tests for ResearchState TypedDict, default_state(), and operator.add reducers.

from __future__ import annotations

import operator

import pytest

from src.models.state import (
    ClaimOutput,
    PlannerOutput,
    ReviewOutput,
    default_state,
)


class TestDefaultState:
    def test_returns_dict(self):
        s = default_state("test query")
        assert isinstance(s, dict)

    def test_query_is_set(self):
        s = default_state("What is FAISS?")
        assert s["query"] == "What is FAISS?"

    def test_run_id_stored(self):
        s = default_state("q", run_id="my-run-id")
        assert s["run_id"] == "my-run-id"

    def test_run_id_empty_by_default(self):
        s = default_state("q")
        assert s["run_id"] == ""

    def test_lists_are_empty(self):
        s = default_state("q")
        assert s["sources"]             == []
        assert s["sub_topics"]          == []
        assert s["pipeline_trace"]      == []
        assert s["errors"]              == []
        assert s["key_claims"]          == []
        assert s["drafts"]              == []
        assert s["search_queries_used"] == []

    def test_numeric_defaults(self):
        s = default_state("q")
        assert s["token_count"]    == 0
        assert s["cost_usd"]       == 0.0
        assert s["quality_score"]  == 0.0
        assert s["revision_count"] == 0
        assert s["quality_retries"] == 0

    def test_bool_defaults(self):
        s = default_state("q")
        assert s["quality_passed"]  is False
        assert s["force_research"]  is False

    def test_string_defaults(self):
        s = default_state("q")
        assert s["current_draft"]   == ""
        assert s["synthesis"]       == ""
        assert s["research_plan"]   == ""
        assert s["final_report"]    == ""
        assert s["current_topic"]   == ""

    def test_review_defaults(self):
        s = default_state("q")
        assert s["review"] == {}

    def test_all_required_keys_present(self):
        s = default_state("q")
        required = [
            "run_id", "query", "sub_topics", "research_plan", "current_topic",
            "force_research", "sources", "search_queries_used", "quality_score",
            "quality_passed", "quality_retries", "key_claims", "conflicts",
            "synthesis", "source_ranking", "drafts", "current_draft",
            "revision_count", "review", "final_report", "pipeline_trace",
            "errors", "token_count", "cost_usd",
        ]
        for key in required:
            assert key in s, f"Missing key: {key}"


class TestOperatorAddFields:
    """
    Verify that LangGraph's operator.add annotation works as expected
    on the list fields -- these get concatenated, not replaced, when
    multiple parallel nodes write to the same field.
    """

    def test_sources_concatenation(self):
        a = [{"url": "https://a.com"}]
        b = [{"url": "https://b.com"}]
        result = operator.add(a, b)
        assert len(result) == 2
        assert result[0]["url"] == "https://a.com"
        assert result[1]["url"] == "https://b.com"

    def test_token_count_addition(self):
        assert operator.add(100, 250) == 350

    def test_cost_usd_addition(self):
        assert abs(operator.add(0.001, 0.002) - 0.003) < 1e-9

    def test_pipeline_trace_concatenation(self):
        t1 = [{"agent": "planner",    "duration_ms": 100}]
        t2 = [{"agent": "researcher", "duration_ms": 200}]
        merged = operator.add(t1, t2)
        assert len(merged) == 2
        assert merged[0]["agent"] == "planner"
        assert merged[1]["agent"] == "researcher"

    def test_errors_concatenation(self):
        e1 = ["error from researcher 1"]
        e2 = ["error from researcher 2"]
        assert operator.add(e1, e2) == ["error from researcher 1", "error from researcher 2"]


class TestPydanticSchemas:
    def test_planner_output_valid(self):
        out = PlannerOutput(
            sub_topics=["FAISS algorithm", "FAISS GPU"],
            research_plan="Search technical docs.",
        )
        assert len(out.sub_topics) == 2
        assert "FAISS" in out.research_plan

    def test_planner_output_rejects_empty_topics(self):
        with pytest.raises(Exception):
            PlannerOutput(sub_topics=[], research_plan="plan")

    def test_planner_output_max_3_topics(self):
        # 3 is the upper bound in the schema
        out = PlannerOutput(
            sub_topics=["a", "b", "c"],
            research_plan="plan",
        )
        assert len(out.sub_topics) == 3

    def test_claim_output_valid(self):
        claim = ClaimOutput(
            claim="FAISS supports GPU search",
            source_idx=1,
            confidence="high",
            evidence="Direct quote from source",
        )
        assert claim.confidence == "high"
        assert claim.source_idx == 1

    def test_review_output_score_bounds(self):
        # Score must be 1-10
        out = ReviewOutput(score=7, issues=[], suggestions=[], passed=True)
        assert out.score == 7

    def test_review_output_rejects_score_0(self):
        with pytest.raises(Exception):
            ReviewOutput(score=0, issues=[], suggestions=[], passed=False)

    def test_review_output_rejects_score_11(self):
        with pytest.raises(Exception):
            ReviewOutput(score=11, issues=[], suggestions=[], passed=True)
