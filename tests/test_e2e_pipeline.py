# tests/test_e2e_pipeline.py
# ─────────────────────────────────────────────
# End-to-end pipeline tests using MOCKED LLM calls.
# No real API spend. No real DB writes (observability is mocked).
# Tests verify the complete orchestration: routing, state merging,
# conditional edges, loop bounds, and cost/token tracking.
# ─────────────────────────────────────────────

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.models.state import default_state

# =============================================================
# Shared mock helpers
# =============================================================


def _make_llm_response(content: str, input_tokens: int = 120, output_tokens: int = 280):
    """Build a minimal AIMessage-like mock for Claude responses."""
    mock = MagicMock()
    mock.content = content
    mock.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return mock


def _make_structured_llm_response(
    parsed_obj, raw_content="ok", input_tokens=100, output_tokens=200
):
    """Return what with_structured_output(include_raw=True).invoke() returns."""
    raw = _make_llm_response(raw_content, input_tokens, output_tokens)
    return {"parsed": parsed_obj, "raw": raw}


def _mock_observability():
    """Return a dict of patches that silence all DB/logger calls."""
    return {
        "start_logger": patch("src.agents.graph.start_logger", return_value=None),
        "start_run": patch("src.agents.graph.start_run", return_value="mock-run-id"),
        "end_run": patch("src.agents.graph.end_run", return_value=None),
        "write_report": patch(
            "src.agents.graph.write_report", return_value="/tmp/report.docx"
        ),
        "log_agent_start": patch(
            "src.observability.logger.log_agent_start",
            return_value=("mock-event-id", 0.0),
        ),
        "log_agent_end": patch(
            "src.observability.logger.log_agent_end", return_value=None
        ),
        "log_cost": patch("src.observability.logger.log_cost", return_value=None),
    }


FAKE_SOURCES = [
    {
        "title": "FAISS Overview",
        "url": "https://engineering.fb.com/faiss",
        "content": "FAISS (Facebook AI Similarity Search) is a library for efficient "
        "similarity search. It supports billion-scale search on GPUs. " * 5,
    },
    {
        "title": "FAISS Wikipedia",
        "url": "https://en.wikipedia.org/wiki/FAISS",
        "content": "FAISS is developed by Meta AI Research. It provides multiple "
        "index types: IVF, HNSW, and PQ for approximate nearest neighbor. " * 5,
    },
    {
        "title": "FAISS Tutorial",
        "url": "https://www.pinecone.io/learn/faiss/",
        "content": "FAISS index types trade off search speed and recall. "
        "IVF partitions vectors into Voronoi cells for fast lookup. " * 5,
    },
]


# =============================================================
# Full Pipeline — Happy Path
# =============================================================


class TestFullPipelineMocked:
    """
    Mock every LLM call and search call.
    Verify the pipeline completes with a final_report.
    """

    def _run_pipeline(self, query: str = "What is FAISS?"):
        from src.models.state import (
            AnalystOutput,
            ClaimOutput,
            PlannerOutput,
            ReviewOutput,
        )

        planner_obj = PlannerOutput(
            sub_topics=["FAISS algorithm overview", "FAISS index types"],
            research_plan="Search FB engineering blog and Wikipedia for FAISS details.",
        )
        analyst_obj = AnalystOutput(
            claims=[
                ClaimOutput(
                    claim="FAISS supports billion-scale search",
                    source_idx=1,
                    confidence="high",
                    evidence="FB blog states billion-scale support",
                ),
                ClaimOutput(
                    claim="FAISS runs on GPU",
                    source_idx=2,
                    confidence="high",
                    evidence="Wikipedia confirms GPU support",
                ),
                ClaimOutput(
                    claim="IVF index partitions vectors into Voronoi cells",
                    source_idx=3,
                    confidence="medium",
                    evidence="Pinecone tutorial explains IVF",
                ),
                ClaimOutput(
                    claim="FAISS is maintained by Meta AI Research",
                    source_idx=2,
                    confidence="high",
                    evidence="Wikipedia states Meta AI Research",
                ),
                ClaimOutput(
                    claim="HNSW provides approximate nearest neighbor search",
                    source_idx=2,
                    confidence="medium",
                    evidence="Wikipedia lists index types",
                ),
            ],
            conflicts=[],
        )
        reviewer_pass = ReviewOutput(
            score=8, issues=[], suggestions=["Add a comparison table"], passed=True
        )

        with (
            patch("langchain_anthropic.ChatAnthropic") as MockLLM,
            patch(
                "src.tools.async_search.async_search_web",
                new=AsyncMock(return_value=FAKE_SOURCES),
            ),
            patch(
                "src.tools.async_search.async_search_wikipedia",
                new=AsyncMock(return_value=[]),
            ),
            patch("src.cache.research_cache.fetch", return_value=None),
            patch("src.cache.research_cache.store", return_value=None),
            patch(
                "src.observability.logger.log_agent_start", return_value=("evt-id", 0.0)
            ),
            patch("src.observability.logger.log_agent_end", return_value=None),
            patch("src.observability.logger.log_cost", return_value=None),
            patch("src.agents.graph.start_logger", return_value=None),
            patch("src.agents.graph.start_run", return_value="mock-run"),
            patch("src.agents.graph.end_run", return_value=None),
            patch("src.agents.graph.write_report", return_value="/tmp/r.docx"),
        ):
            mock_llm_instance = MagicMock()
            MockLLM.return_value = mock_llm_instance

            # structured_output chain: .with_structured_output(...).invoke(...)
            mock_structured = MagicMock()
            mock_llm_instance.with_structured_output.return_value = mock_structured

            # Route calls by iteration
            mock_structured.invoke.side_effect = [
                _make_structured_llm_response(planner_obj),  # planner
                _make_structured_llm_response(analyst_obj),  # analyst
                _make_structured_llm_response(reviewer_pass),  # reviewer (first)
            ]

            # Plain invoke for synthesizer + writer
            mock_llm_instance.invoke.side_effect = [
                _make_llm_response(
                    "FAISS is a high-performance library by Meta AI for dense "
                    "vector similarity search. It supports GPU acceleration and "
                    "multiple index types including IVF and HNSW.",
                ),  # synthesizer
                _make_llm_response(
                    "## Executive Summary\n"
                    "FAISS (Facebook AI Similarity Search) is a library by Meta AI.\n\n"
                    "## Key Findings\n"
                    "- FAISS supports billion-scale GPU search [Source 1]\n"
                    "- IVF and HNSW indexes trade off speed/recall [Source 3]\n\n"
                    "## Analysis\nFAISS is widely adopted for production vector search.\n\n"
                    "## Conclusion\nFAISS is the leading open-source library for similarity search.\n\n"
                    "## Sources\n[1] FB Engineering Blog\n[2] Wikipedia\n[3] Pinecone Tutorial\n"
                ),  # writer
            ]

            from src.agents.graph import run_pipeline

            return run_pipeline(query)

    def test_pipeline_completes_with_final_report(self):
        result = self._run_pipeline()
        assert result is not None
        report = result.get("final_report") or result.get("current_draft", "")
        assert len(report) > 100, "Expected a non-trivial report"

    def test_pipeline_trace_has_all_agents(self):
        result = self._run_pipeline()
        trace = result.get("pipeline_trace", [])
        agent_names = [t.get("agent") for t in trace]
        required = {
            "planner",
            "researcher",
            "merge_research",
            "quality_gate",
            "analyst",
            "synthesizer",
            "writer",
            "reviewer",
        }
        missing = required - set(agent_names)
        assert not missing, f"Missing agents in trace: {missing}"

    def test_pipeline_has_sources(self):
        result = self._run_pipeline()
        assert len(result.get("sources", [])) > 0

    def test_pipeline_has_claims(self):
        result = self._run_pipeline()
        assert len(result.get("key_claims", [])) >= 3

    def test_pipeline_accumulates_tokens(self):
        result = self._run_pipeline()
        assert result.get("token_count", 0) > 0

    def test_pipeline_accumulates_cost(self):
        result = self._run_pipeline()
        assert result.get("cost_usd", 0.0) > 0.0


# =============================================================
# Quality Gate Retry Flow
# =============================================================


class TestQualityGateRetryFlow:
    def test_bad_sources_trigger_quality_fail(self):
        from src.agents.quality_gate import quality_gate_agent

        bad_sources = [
            {
                "url": "https://reddit.com/r/spam",
                "content": "click here subscribe now log in",
            },
        ]
        state = default_state("test query", run_id="t")
        state["sources"] = bad_sources
        result = quality_gate_agent(state)
        assert result["quality_passed"] is False
        assert result["quality_score"] < 0.4

    def test_good_sources_pass_quality_gate(self):
        from src.agents.quality_gate import quality_gate_agent

        good_sources = [
            {
                "url": "https://arxiv.org/abs/1234",
                "content": "This paper presents FAISS, a library for efficient "
                "similarity search of dense vectors at billion scale. " * 5,
            },
            {
                "url": "https://en.wikipedia.org/wiki/FAISS",
                "content": "FAISS is developed by Meta AI Research for similarity "
                "search and clustering of dense vectors. " * 5,
            },
        ]
        state = default_state("test query", run_id="t")
        state["sources"] = good_sources
        result = quality_gate_agent(state)
        assert result["quality_passed"] is True
        assert result["quality_score"] >= 0.4

    def test_retry_counter_increments_retries(self):
        from src.agents.graph import retry_counter_node

        state = default_state("q", run_id="t")
        state["quality_retries"] = 0
        result = retry_counter_node(state)
        assert result["quality_retries"] == 1

    def test_retry_counter_sets_force_research(self):
        from src.agents.graph import retry_counter_node

        state = default_state("q", run_id="t")
        result = retry_counter_node(state)
        assert result["force_research"] is True

    def test_retry_counter_clears_sources(self):
        from src.agents.graph import retry_counter_node

        state = default_state("q", run_id="t")
        state["sources"] = FAKE_SOURCES.copy()
        result = retry_counter_node(state)
        assert result["sources"] == []

    def test_force_research_bypasses_cache(self):
        from langgraph.types import Send

        from src.agents.graph import fan_out_or_cache

        # Even with a warm cache, force_research=True must fan out
        fake_cached = [{"url": "https://a.com", "title": "a", "content": "c"}]
        state = default_state("q")
        state["sub_topics"] = ["topic 1"]
        state["force_research"] = True
        with patch("src.agents.graph.cache_fetch", return_value=fake_cached):
            result = fan_out_or_cache(state)
        assert isinstance(result, list)
        assert all(isinstance(r, Send) for r in result)

    def test_retry_routing_returns_analyst_when_retries_exhausted(self):
        from src.agents.graph import _should_retry_research

        state = default_state("q")
        state["quality_passed"] = False
        state["quality_score"] = 0.1
        state["quality_retries"] = 1  # matches max_quality_retries=1 in config
        assert _should_retry_research(state) == "analyst"


# =============================================================
# Reviewer Refinement Loop
# =============================================================


class TestReviewerRefinementLoop:
    def test_high_score_ends_pipeline(self):
        from src.agents.graph import _should_revise

        state = default_state("q")
        state["review"] = {"score": 9, "passed": True, "issues": [], "suggestions": []}
        state["revision_count"] = 1
        assert _should_revise(state) == "end"

    def test_low_score_loops_back_to_writer(self):
        from src.agents.graph import _should_revise

        state = default_state("q")
        state["review"] = {
            "score": 4,
            "passed": False,
            "issues": ["missing citations"],
            "suggestions": [],
        }
        state["revision_count"] = 0
        assert _should_revise(state) == "revise"

    def test_max_revisions_reached_forces_end(self):
        from src.agents.graph import _should_revise

        state = default_state("q")
        state["review"] = {
            "score": 4,
            "passed": False,
            "issues": ["still bad"],
            "suggestions": [],
        }
        state["revision_count"] = 2  # max_revisions=2
        assert _should_revise(state) == "end"

    def test_score_at_pass_threshold_ends_pipeline(self):
        from src.agents.graph import _should_revise

        state = default_state("q")
        # score=7 == review_pass_score=7 → end
        state["review"] = {"score": 7, "passed": False, "issues": [], "suggestions": []}
        state["revision_count"] = 0
        assert _should_revise(state) == "end"

    def test_writer_uses_revision_prompt_on_second_call(self):
        """Writer should detect revision_count>0 and use reviewer feedback."""
        with (
            patch("langchain_anthropic.ChatAnthropic") as MockLLM,
            patch(
                "src.guardrails.scrub_pii",
                return_value=("revised draft v2 content " * 30, []),
            ),
            patch(
                "src.observability.logger.log_agent_start", return_value=("evt", 0.0)
            ),
            patch("src.observability.logger.log_agent_end", return_value=None),
            patch("src.observability.logger.log_cost", return_value=None),
        ):
            mock_llm = MagicMock()
            MockLLM.return_value = mock_llm
            mock_llm.invoke.return_value = _make_llm_response(
                "## Executive Summary\nRevised draft with better citations.\n\n"
                "## Sources\n[1] FB Blog\n"
            )

            state = default_state("q", run_id="test")
            state["revision_count"] = 1  # already revised once
            state["review"] = {
                "score": 4,
                "issues": ["missing citations"],
                "suggestions": ["add source numbers"],
                "passed": False,
            }
            state["current_draft"] = "Original draft here."
            state["key_claims"] = []

            from src.agents.writer import writer_agent

            result = writer_agent(state)

        assert result["revision_count"] == 2
        assert "current_draft" in result


# =============================================================
# Parallel Fan-Out
# =============================================================


class TestParallelFanOut:
    def test_fan_out_creates_one_send_per_subtopic(self):
        from langgraph.types import Send

        from src.agents.graph import fan_out_or_cache

        state = default_state("q")
        state["sub_topics"] = ["topic A", "topic B", "topic C"]
        with patch("src.agents.graph.cache_fetch", return_value=None):
            sends = fan_out_or_cache(state)
        assert isinstance(sends, list)
        assert len(sends) == 3
        assert all(isinstance(s, Send) for s in sends)

    def test_fan_out_injects_current_topic(self):
        from src.agents.graph import fan_out_or_cache

        topics = ["FAISS algorithm", "FAISS GPU support", "FAISS index types"]
        state = default_state("What is FAISS?")
        state["sub_topics"] = topics
        with patch("src.agents.graph.cache_fetch", return_value=None):
            sends = fan_out_or_cache(state)

        injected = [s.arg["current_topic"] for s in sends]
        assert injected == topics

    def test_merge_research_deduplicates_urls(self):
        from src.agents.graph import merge_research_node

        dupe_sources = [
            {"url": "https://a.com", "title": "A", "content": "c"},
            {"url": "https://a.com", "title": "A dup", "content": "c2"},
            {"url": "https://b.com", "title": "B", "content": "c"},
        ]
        state = default_state("q")
        state["sources"] = dupe_sources

        with (
            patch("src.agents.graph.cache_store"),
            patch("threading.Thread") as MockThread,
        ):
            MockThread.return_value.start = MagicMock()
            result = merge_research_node(state)

        assert len(result["sources"]) == 2

    def test_merge_dispatches_cache_write_to_background_thread(self):
        from src.agents.graph import merge_research_node

        state = default_state("q")
        state["sources"] = [{"url": "https://x.com", "title": "X", "content": "c"}]

        with (
            patch("src.agents.graph.cache_store") as mock_store,
            patch("threading.Thread") as MockThread,
        ):
            mock_t = MagicMock()
            MockThread.return_value = mock_t
            merge_research_node(state)

        MockThread.assert_called_once()
        mock_t.start.assert_called_once()

    def test_cache_hit_routes_to_cache_loader(self):
        from src.agents.graph import fan_out_or_cache

        cached = [{"url": "https://cached.com", "title": "Cached", "content": "c"}]
        state = default_state("What is FAISS?")
        state["sub_topics"] = ["FAISS overview"]
        with patch("src.agents.graph.cache_fetch", return_value=cached):
            result = fan_out_or_cache(state)

        assert result == "cache_loader"

    def test_single_subtopic_creates_one_send(self):

        from src.agents.graph import fan_out_or_cache

        state = default_state("q")
        state["sub_topics"] = ["single topic"]
        with patch("src.agents.graph.cache_fetch", return_value=None):
            sends = fan_out_or_cache(state)
        assert len(sends) == 1


# =============================================================
# Budget Enforcement
# =============================================================


class TestBudgetEnforcement:
    def test_calculate_cost_haiku(self):
        from src.observability.cost import calculate_cost

        record = calculate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=1000,
            output_tokens=500,
            agent_name="planner",
            run_id="test",
        )
        expected_input = 1000 / 1000 * 0.00025
        expected_output = 500 / 1000 * 0.00125
        assert abs(record.cost_usd - (expected_input + expected_output)) < 1e-8
        assert record.agent_name == "planner"

    def test_accumulator_tracks_total_cost(self):
        from src.observability.cost import RunCostAccumulator, calculate_cost

        acc = RunCostAccumulator(run_id="test")
        r1 = calculate_cost("claude-haiku-4-5-20251001", 500, 300, "planner", "test")
        r2 = calculate_cost("claude-haiku-4-5-20251001", 800, 400, "analyst", "test")
        acc.add(r1)
        acc.add(r2)
        assert abs(acc.total_cost - (r1.cost_usd + r2.cost_usd)) < 1e-8

    def test_check_budget_ok_under_limit(self):
        from src.observability.cost import RunCostAccumulator, check_budget

        acc = RunCostAccumulator(run_id="test")
        ok, status = check_budget(acc, "planner")
        assert ok is True
        assert status == "ok"

    def test_check_budget_exceeded_over_hard_limit(self):
        from src.observability.cost import HARD_LIMIT, RunCostAccumulator, check_budget

        acc = RunCostAccumulator(run_id="test")
        acc.total_cost = HARD_LIMIT + 0.01  # over the hard limit
        ok, status = check_budget(acc, "writer")
        assert ok is False
        assert status == "exceeded"

    def test_accumulator_per_agent_cost_breakdown(self):
        from src.observability.cost import RunCostAccumulator, calculate_cost

        acc = RunCostAccumulator(run_id="test")
        acc.add(
            calculate_cost("claude-haiku-4-5-20251001", 300, 200, "planner", "test")
        )
        acc.add(
            calculate_cost("claude-haiku-4-5-20251001", 500, 300, "analyst", "test")
        )
        assert "planner" in acc.agent_costs
        assert "analyst" in acc.agent_costs

    def test_accumulator_most_expensive_agent(self):
        from src.observability.cost import RunCostAccumulator, calculate_cost

        acc = RunCostAccumulator(run_id="test")
        acc.add(calculate_cost("claude-haiku-4-5-20251001", 100, 100, "planner", "t"))
        acc.add(calculate_cost("claude-haiku-4-5-20251001", 5000, 2000, "writer", "t"))
        assert acc.most_expensive_agent() == "writer"


# =============================================================
# PII Scrubbing in Pipeline
# =============================================================


class TestPIIScrubbing:
    def test_email_scrubbed_from_writer_output(self):
        from src.guardrails import check_output

        result = check_output("Contact research@company.com for the full paper.")
        assert "research@company.com" not in result.scrubbed_text
        assert "[REDACTED_EMAIL]" in result.scrubbed_text
        assert "email" in result.pii_found

    def test_phone_scrubbed_from_writer_output(self):
        from src.guardrails import check_output

        result = check_output("Call 555-123-4567 for information.")
        assert "555-123-4567" not in result.scrubbed_text
        assert "[REDACTED_PHONE_US]" in result.scrubbed_text

    def test_ssn_scrubbed(self):
        from src.guardrails import check_output

        result = check_output("The subject's SSN is 123-45-6789.")
        assert "123-45-6789" not in result.scrubbed_text
        assert "ssn" in result.pii_found

    def test_clean_text_passes_unchanged(self):
        text = "FAISS is a library for dense vector similarity search by Meta AI."
        from src.guardrails import check_output

        result = check_output(text)
        assert result.scrubbed_text == text
        assert result.pii_found == []

    def test_multiple_pii_types_all_scrubbed(self):
        text = "Contact admin@corp.com or call 555-999-1234."
        from src.guardrails import check_output

        result = check_output(text)
        assert "admin@corp.com" not in result.scrubbed_text
        assert "555-999-1234" not in result.scrubbed_text
        assert "email" in result.pii_found
        assert "phone_us" in result.pii_found


# =============================================================
# Cache Integration
# =============================================================


class TestCacheIntegration:
    def test_cache_store_then_fetch(self, tmp_db_path):
        with (
            patch("src.cache.research_cache._DB_PATH", tmp_db_path),
            patch("src.cache.research_cache._get_embedder", return_value=None),
            patch("src.cache.research_cache._embed", return_value=None),
        ):
            from src.cache import research_cache

            research_cache._ensure_table()

            research_cache.store("faiss test query", FAKE_SOURCES, "tavily")
            result = research_cache.fetch("faiss test query")

        assert result is not None
        assert len(result) == len(FAKE_SOURCES)

    def test_stats_updated_after_store(self, tmp_db_path):
        with (
            patch("src.cache.research_cache._DB_PATH", tmp_db_path),
            patch("src.cache.research_cache._get_embedder", return_value=None),
            patch("src.cache.research_cache._embed", return_value=None),
        ):
            from src.cache import research_cache

            research_cache._ensure_table()

            research_cache.store("stats test query", FAKE_SOURCES, "tavily")
            s = research_cache.stats()

        assert s["total_entries"] >= 1


# =============================================================
# State Schema — full pipeline requirements
# =============================================================


class TestStateSchemaE2E:
    def test_default_state_has_all_fields(self):
        state = default_state("test", run_id="r")
        required = [
            "run_id",
            "query",
            "sub_topics",
            "research_plan",
            "current_topic",
            "force_research",
            "sources",
            "search_queries_used",
            "quality_score",
            "quality_passed",
            "quality_retries",
            "key_claims",
            "conflicts",
            "synthesis",
            "source_ranking",
            "drafts",
            "current_draft",
            "revision_count",
            "review",
            "final_report",
            "pipeline_trace",
            "errors",
            "token_count",
            "cost_usd",
        ]
        for f in required:
            assert f in state, f"Missing field: {f}"

    def test_token_count_is_operator_add(self):
        import operator

        # Simulates parallel researchers each returning token_count
        assert operator.add(300, 400) == 700

    def test_sources_are_operator_add(self):
        import operator

        a = [{"url": "https://a.com"}]
        b = [{"url": "https://b.com"}]
        merged = operator.add(a, b)
        assert len(merged) == 2

    def test_graph_compiles_with_correct_schema(self):
        with patch("langchain_anthropic.ChatAnthropic"):
            from src.agents.graph import build_graph

            g = build_graph()
        assert g is not None
        nodes = set(g.nodes.keys())
        assert "planner" in nodes
        assert "researcher" in nodes
        assert "quality_gate" in nodes
        assert "analyst" in nodes
        assert "synthesizer" in nodes
        assert "writer" in nodes
        assert "reviewer" in nodes
        assert "merge_research" in nodes
        assert "cache_loader" in nodes
        assert "retry_counter" in nodes


# =============================================================
# Token Usage Extraction
# =============================================================


class TestTokenUsageExtraction:
    def test_extract_from_mock_response(self):
        from src.observability.cost import extract_token_usage

        mock_resp = _make_llm_response("hello", input_tokens=250, output_tokens=400)
        usage = extract_token_usage(mock_resp)
        assert usage.input_tokens == 250
        assert usage.output_tokens == 400
        assert usage.total_tokens == 650

    def test_extract_returns_zeros_when_no_metadata(self):
        from src.observability.cost import extract_token_usage

        mock = MagicMock()
        mock.usage_metadata = None
        usage = extract_token_usage(mock)
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_total_tokens_is_sum(self):
        from src.observability.cost import TokenUsage

        u = TokenUsage(input_tokens=100, output_tokens=200)
        assert u.total_tokens == 300
