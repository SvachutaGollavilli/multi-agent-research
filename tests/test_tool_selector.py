# tests/test_tool_selector.py
# Tests for the keyword-based tool selector.
# Pure Python, zero network calls, runs in <1ms.

from __future__ import annotations

import pytest

from src.tools.tool_selector import select_tool


class TestToolSelection:
    def test_returns_dict(self):
        result = select_tool("What is Python?")
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = select_tool("What is Python?")
        for key in ("tool", "confidence", "reason", "wiki_hits", "tavily_hits"):
            assert key in result, f"Missing key: {key}"

    def test_tool_is_valid_choice(self):
        result = select_tool("anything")
        assert result["tool"] in ("wikipedia", "tavily", "both")

    def test_confidence_between_0_and_1(self):
        result = select_tool("anything")
        assert 0.0 <= result["confidence"] <= 1.0


class TestWikipediaRouting:
    """Queries that should route to Wikipedia."""

    @pytest.mark.parametrize(
        "query",
        [
            "what is the attention mechanism",
            "who invented the transformer",
            "history of neural networks",
            "definition of gradient descent",
            "explain backpropagation",
            "theory of relativity",
            "overview of reinforcement learning",
        ],
    )
    def test_wikipedia_queries(self, query):
        result = select_tool(query)
        assert result["tool"] in ("wikipedia", "both"), (
            f"Expected wikipedia or both for '{query}', got '{result['tool']}'"
        )
        assert result["wiki_hits"] > 0


class TestTavilyRouting:
    """Queries that should route to Tavily."""

    @pytest.mark.parametrize(
        "query",
        [
            "latest AI news today",
            "best vector databases 2024",
            "how to install FAISS",
            "top 5 LLM frameworks compared",
            "current state of LLM benchmarks",
        ],
    )
    def test_tavily_queries(self, query):
        result = select_tool(query)
        assert result["tool"] in ("tavily", "both"), (
            f"Expected tavily or both for '{query}', got '{result['tool']}'"
        )
        assert result["tavily_hits"] > 0


class TestDefaultRouting:
    """Ambiguous queries with no keyword signals should default to Tavily."""

    @pytest.mark.parametrize(
        "query",
        [
            "FAISS",
            "LangGraph",
            "multi-agent systems",
            "embeddings",
        ],
    )
    def test_no_keyword_defaults_to_tavily(self, query):
        result = select_tool(query)
        assert result["tool"] == "tavily"
        assert result["wiki_hits"] == 0
        assert result["tavily_hits"] == 0
        assert result["confidence"] == 0.5


class TestConfidenceScaling:
    def test_more_hits_means_higher_confidence(self):
        # "what is the definition of" hits multiple wiki patterns
        high_hit = select_tool("what is the definition of reinforcement learning")
        low_hit = select_tool("what is FAISS")

        if high_hit["tool"] == "wikipedia" and low_hit["tool"] == "wikipedia":
            assert high_hit["confidence"] >= low_hit["confidence"]

    def test_confidence_caps_at_0_95(self):
        # Even many hits shouldn't push confidence above 0.95
        result = select_tool(
            "what is the history biography definition of the origin theory algorithm"
        )
        assert result["confidence"] <= 0.95


class TestEdgeCases:
    def test_empty_string_defaults_to_tavily(self):
        result = select_tool("")
        assert result["tool"] == "tavily"

    def test_very_long_query(self):
        q = "what is " + " and ".join([f"concept{i}" for i in range(100)])
        result = select_tool(q)
        assert result["tool"] in ("wikipedia", "tavily", "both")

    def test_case_insensitive(self):
        lower = select_tool("what is python")
        upper = select_tool("WHAT IS PYTHON")
        assert lower["tool"] == upper["tool"]
