# tests/test_graph.py
# Tests for graph structure, routing functions, and utility nodes.
# LLM calls and search calls are all mocked -- no API spend.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.state import default_state


# -----------------------------------------------------------------
# Graph compilation
# -----------------------------------------------------------------

class TestBuildGraph:
    def test_compiles_without_error(self):
        from src.agents.graph import build_graph
        g = build_graph()
        assert g is not None

    def test_graph_has_expected_nodes(self):
        from src.agents.graph import build_graph
        g = build_graph()
        # LangGraph compiled graph exposes nodes via .nodes property
        nodes = set(g.nodes.keys())
        expected = {
            "planner", "cache_loader", "researcher", "merge_research",
            "quality_gate", "retry_counter", "analyst", "synthesizer",
            "writer", "reviewer",
        }
        assert expected.issubset(nodes), (
            f"Missing nodes: {expected - nodes}"
        )


# -----------------------------------------------------------------
# Routing: fan_out_or_cache
# -----------------------------------------------------------------

class TestFanOutOrCache:
    def _state(self, **kwargs):
        s = default_state("What is FAISS?", run_id="test")
        s["sub_topics"] = ["FAISS algorithm", "FAISS GPU", "FAISS index types"]
        s.update(kwargs)
        return s

    def test_cache_miss_returns_send_list(self):
        from src.agents.graph import fan_out_or_cache
        from langgraph.types import Send
        with patch("src.agents.graph.cache_fetch", return_value=None):
            result = fan_out_or_cache(self._state())
        assert isinstance(result, list)
        assert all(isinstance(r, Send) for r in result)
        assert len(result) == 3

    def test_cache_hit_returns_string(self):
        from src.agents.graph import fan_out_or_cache
        fake_sources = [{"url": "https://a.com", "title": "a", "content": "c"}]
        with patch("src.agents.graph.cache_fetch", return_value=fake_sources):
            result = fan_out_or_cache(self._state())
        assert result == "cache_loader"

    def test_force_research_bypasses_cache(self):
        from src.agents.graph import fan_out_or_cache
        from langgraph.types import Send
        fake_sources = [{"url": "https://a.com", "title": "a", "content": "c"}]
        with patch("src.agents.graph.cache_fetch", return_value=fake_sources):
            # Even with a cache hit, force_research=True must bypass it
            result = fan_out_or_cache(self._state(force_research=True))
        assert isinstance(result, list)
        assert all(isinstance(r, Send) for r in result)

    def test_send_injects_current_topic(self):
        from src.agents.graph import fan_out_or_cache
        with patch("src.agents.graph.cache_fetch", return_value=None):
            sends = fan_out_or_cache(self._state())
        topics = [s.arg["current_topic"] for s in sends]
        assert topics == ["FAISS algorithm", "FAISS GPU", "FAISS index types"]

    def test_falls_back_to_query_when_no_sub_topics(self):
        from src.agents.graph import fan_out_or_cache
        from langgraph.types import Send
        state = default_state("What is FAISS?")
        state["sub_topics"] = []  # no sub-topics
        with patch("src.agents.graph.cache_fetch", return_value=None):
            result = fan_out_or_cache(state)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].arg["current_topic"] == "What is FAISS?"


# -----------------------------------------------------------------
# Routing: _should_retry_research
# -----------------------------------------------------------------

class TestShouldRetryResearch:
    def test_passes_when_quality_passed(self):
        from src.agents.graph import _should_retry_research
        state = default_state("q")
        state["quality_passed"]  = True
        state["quality_score"]   = 0.75
        state["quality_retries"] = 0
        assert _should_retry_research(state) == "analyst"

    def test_retries_when_quality_failed_and_retries_available(self):
        from src.agents.graph import _should_retry_research
        state = default_state("q")
        state["quality_passed"]  = False
        state["quality_score"]   = 0.1
        state["quality_retries"] = 0
        assert _should_retry_research(state) == "retry_counter"

    def test_proceeds_when_retries_exhausted(self):
        from src.agents.graph import _should_retry_research
        state = default_state("q")
        state["quality_passed"]  = False
        state["quality_score"]   = 0.1
        state["quality_retries"] = 1  # max_quality_retries is 1 in config
        assert _should_retry_research(state) == "analyst"


# -----------------------------------------------------------------
# Routing: _should_revise
# -----------------------------------------------------------------

class TestShouldRevise:
    def test_ends_when_review_passes(self):
        from src.agents.graph import _should_revise
        state = default_state("q")
        state["review"]         = {"score": 8, "passed": True, "issues": [], "suggestions": []}
        state["revision_count"] = 1
        assert _should_revise(state) == "end"

    def test_revises_when_score_below_threshold(self):
        from src.agents.graph import _should_revise
        state = default_state("q")
        state["review"]         = {"score": 5, "passed": False, "issues": ["too short"],
                                   "suggestions": []}
        state["revision_count"] = 0
        assert _should_revise(state) == "revise"

    def test_ends_when_max_revisions_reached(self):
        from src.agents.graph import _should_revise
        state = default_state("q")
        state["review"]         = {"score": 5, "passed": False, "issues": [], "suggestions": []}
        state["revision_count"] = 2  # max_revisions = 2
        assert _should_revise(state) == "end"

    def test_ends_when_score_meets_threshold_even_if_not_passed(self):
        from src.agents.graph import _should_revise
        state = default_state("q")
        # score >= review_pass_score (7) even if passed=False
        state["review"]         = {"score": 7, "passed": False, "issues": [], "suggestions": []}
        state["revision_count"] = 0
        assert _should_revise(state) == "end"


# -----------------------------------------------------------------
# Utility nodes
# -----------------------------------------------------------------

class TestCacheLoaderNode:
    def test_returns_sources_on_hit(self):
        from src.agents.graph import cache_loader_node
        fake = [{"url": "https://a.com", "title": "A", "content": "c"}]
        state = default_state("What is FAISS?")
        with patch("src.agents.graph.cache_fetch", return_value=fake):
            result = cache_loader_node(state)
        assert result["sources"] == fake

    def test_returns_empty_on_miss(self):
        from src.agents.graph import cache_loader_node
        state = default_state("What is FAISS?")
        with patch("src.agents.graph.cache_fetch", return_value=None):
            result = cache_loader_node(state)
        assert result["sources"] == []

    def test_has_pipeline_trace(self):
        from src.agents.graph import cache_loader_node
        state = default_state("q")
        with patch("src.agents.graph.cache_fetch", return_value=[]):
            result = cache_loader_node(state)
        assert len(result["pipeline_trace"]) == 1
        assert result["pipeline_trace"][0]["agent"] == "cache_loader"


class TestMergeResearchNode:
    def test_deduplicates_sources_by_url(self):
        from src.agents.graph import merge_research_node
        state = default_state("q")
        state["sources"] = [
            {"url": "https://a.com", "title": "A", "content": "c"},
            {"url": "https://a.com", "title": "A dup", "content": "c"},  # dupe
            {"url": "https://b.com", "title": "B", "content": "c"},
        ]
        with patch("src.agents.graph.cache_store"):
            result = merge_research_node(state)
        assert len(result["sources"]) == 2

    def test_fires_cache_write_in_background(self):
        from src.agents.graph import merge_research_node
        state = default_state("q")
        state["sources"] = [{"url": "https://a.com", "title": "A", "content": "c"}]
        with patch("src.agents.graph.cache_store") as mock_store, \
             patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            merge_research_node(state)
        mock_thread.assert_called_once()

    def test_has_pipeline_trace(self):
        from src.agents.graph import merge_research_node
        state = default_state("q")
        state["sources"] = [{"url": "https://a.com", "title": "A", "content": "c"}]
        with patch("src.agents.graph.cache_store"):
            result = merge_research_node(state)
        assert result["pipeline_trace"][0]["agent"] == "merge_research"


class TestRetryCounterNode:
    def test_increments_quality_retries(self):
        from src.agents.graph import retry_counter_node
        state = default_state("q")
        state["quality_retries"] = 0
        result = retry_counter_node(state)
        assert result["quality_retries"] == 1

    def test_sets_force_research(self):
        from src.agents.graph import retry_counter_node
        state = default_state("q")
        result = retry_counter_node(state)
        assert result["force_research"] is True

    def test_clears_sources(self):
        from src.agents.graph import retry_counter_node
        state = default_state("q")
        state["sources"] = [{"url": "https://a.com", "title": "A", "content": "c"}]
        result = retry_counter_node(state)
        assert result["sources"] == []
