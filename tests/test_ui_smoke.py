# tests/test_ui_smoke.py
# ─────────────────────────────────────────────
# Smoke tests: verify all modules import cleanly and key symbols
# are accessible. Does NOT launch a Streamlit server.
# All external API clients (Anthropic, Tavily) are mocked so
# LLM singletons don't fail on import with fake keys.
# ─────────────────────────────────────────────

from __future__ import annotations

import os
from unittest.mock import patch

# Ensure fake keys are present before any src import
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("TAVILY_API_KEY",    "test-tavily-not-real")


# =============================================================
# Agent modules
# =============================================================

class TestAgentImports:
    def test_planner_imports(self):
        with patch("langchain_anthropic.ChatAnthropic"):
            from src.agents.planner import planner_agent
        assert callable(planner_agent)

    def test_researcher_imports(self):
        from src.agents.researcher import researcher_agent
        assert callable(researcher_agent)

    def test_quality_gate_imports(self):
        from src.agents.quality_gate import quality_gate_agent
        assert callable(quality_gate_agent)

    def test_analyst_imports(self):
        with patch("langchain_anthropic.ChatAnthropic"):
            from src.agents.analyst import analyst_agent
        assert callable(analyst_agent)

    def test_synthesizer_imports(self):
        with patch("langchain_anthropic.ChatAnthropic"):
            from src.agents.synthesizer import synthesizer_agent
        assert callable(synthesizer_agent)

    def test_writer_imports(self):
        with patch("langchain_anthropic.ChatAnthropic"):
            from src.agents.writer import writer_agent
        assert callable(writer_agent)

    def test_reviewer_imports(self):
        with patch("langchain_anthropic.ChatAnthropic"):
            from src.agents.reviewer import reviewer_agent
        assert callable(reviewer_agent)

    def test_graph_imports(self):
        with patch("langchain_anthropic.ChatAnthropic"):
            from src.agents.graph import (
                build_graph,
                run_pipeline,
                stream_pipeline_async,
            )
        assert callable(build_graph)
        assert callable(run_pipeline)
        assert callable(stream_pipeline_async)


# =============================================================
# Tool modules
# =============================================================

class TestToolImports:
    def test_search_imports(self):
        from src.tools.search import search_web
        assert callable(search_web)

    def test_wikipedia_imports(self):
        from src.tools.wikipedia import search_wikipedia
        assert callable(search_wikipedia)

    def test_async_search_imports(self):
        from src.tools.async_search import async_search_all, async_search_web
        assert callable(async_search_all)
        assert callable(async_search_web)

    def test_tool_selector_imports(self):
        from src.tools.tool_selector import select_tool
        assert callable(select_tool)


# =============================================================
# Cache module
# =============================================================

class TestCacheImports:
    def test_cache_module_imports(self):
        from src.cache.research_cache import fetch, stats, store
        assert callable(fetch)
        assert callable(store)
        assert callable(stats)

    def test_stats_returns_dict(self):
        from src.cache.research_cache import stats
        result = stats()
        assert isinstance(result, dict)

    def test_stats_has_required_keys(self):
        from src.cache.research_cache import stats
        result = stats()
        assert "total_entries"   in result
        assert "fresh_entries"   in result
        assert "total_hits"      in result


# =============================================================
# Config module
# =============================================================

class TestConfigImports:
    def test_config_module_imports(self):
        from src.config import (
            get_model,
            get_pipeline_config,
            load_config,
        )
        assert callable(load_config)
        assert callable(get_model)
        assert callable(get_pipeline_config)

    def test_load_config_returns_dict(self):
        from src.config import load_config
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_pipeline_config_has_required_keys(self):
        from src.config import get_pipeline_config
        cfg = get_pipeline_config()
        assert "max_sub_topics"    in cfg
        assert "max_revisions"     in cfg
        assert "quality_threshold" in cfg
        assert "review_pass_score" in cfg

    def test_budget_config_has_limits(self):
        from src.config import get_budget_config
        cfg = get_budget_config()
        assert "soft_limit" in cfg
        assert "hard_limit" in cfg

    def test_get_model_returns_string(self):
        from src.config import get_model
        model = get_model("planner")
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_max_tokens_returns_int(self):
        from src.config import get_max_tokens
        tokens = get_max_tokens("writer")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_quality_gate_config_has_domain_lists(self):
        from src.config import get_quality_gate_config
        cfg = get_quality_gate_config()
        assert "high_trust_domains"   in cfg
        assert "medium_trust_domains" in cfg
        assert "low_trust_domains"    in cfg
        assert len(cfg["high_trust_domains"]) > 0


# =============================================================
# Guardrails module
# =============================================================

class TestGuardrailImports:
    def test_guardrails_module_imports(self):
        from src.guardrails import (
            check_input,
            check_output,
            detect_injection,
            scrub_pii,
        )
        assert callable(detect_injection)
        assert callable(scrub_pii)
        assert callable(check_input)
        assert callable(check_output)

    def test_global_rate_limiter_is_instance(self):
        from src.guardrails import RateLimiter, rate_limiter
        assert isinstance(rate_limiter, RateLimiter)

    def test_guardrail_result_is_dataclass(self):
        from src.guardrails import GuardrailResult
        result = GuardrailResult(safe=True, injection_safe=True)
        assert result.safe is True
        assert result.pii_found == []


# =============================================================
# Observability modules
# =============================================================

class TestObservabilityImports:
    def test_cost_module_imports(self):
        from src.observability.cost import (
            calculate_cost,
            check_budget,
            extract_token_usage,
        )
        assert callable(calculate_cost)
        assert callable(check_budget)
        assert callable(extract_token_usage)

    def test_logger_module_imports(self):
        from src.observability.logger import (
            end_run,
            log_agent_end,
            log_agent_start,
            start_run,
        )
        assert callable(start_run)
        assert callable(end_run)
        assert callable(log_agent_start)
        assert callable(log_agent_end)

    def test_db_module_imports(self):
        from src.observability.db import get_db_stats
        assert callable(get_db_stats)

    def test_db_stats_returns_dict(self):
        from src.observability.db import get_db_stats
        result = get_db_stats()
        assert isinstance(result, dict)


# =============================================================
# Output module
# =============================================================

class TestOutputImports:
    def test_report_writer_imports(self):
        from src.output.report_writer import write_report
        assert callable(write_report)


# =============================================================
# Evaluation module
# =============================================================

class TestEvaluationImports:
    def test_judge_prompt_imports(self):
        from evaluation.judge_prompt import build_judge_prompt
        assert callable(build_judge_prompt)

    def test_judge_prompt_has_required_criteria(self):
        from evaluation.judge_prompt import build_judge_prompt
        prompt = build_judge_prompt(
            query="What is FAISS?",
            report="FAISS is a library...",
            expected_points=["similarity search", "Meta AI"],
        )
        # Judge prompt should mention all scoring dimensions
        for keyword in ["accuracy", "completeness", "citation", "coherence"]:
            assert keyword.lower() in prompt.lower(), (
                f"Judge prompt missing keyword: {keyword}"
            )

    def test_questions_yaml_loads(self):
        import os

        import yaml
        questions_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "evaluation", "questions.yaml"
        )
        with open(questions_path) as f:
            data = yaml.safe_load(f)
        # questions.yaml is a dict with a top-level "questions" key
        questions = data.get("questions", []) if isinstance(data, dict) else data
        assert isinstance(questions, list)
        assert len(questions) >= 5  # at least 5 test questions
        for q in questions:
            assert "id"              in q
            assert "query"           in q
            assert "expected_points" in q


# =============================================================
# State module
# =============================================================

class TestStateModuleImports:
    def test_state_module_imports(self):
        from src.models.state import (
            default_state,
        )
        assert callable(default_state)

    def test_default_state_callable_with_query(self):
        from src.models.state import default_state
        s = default_state("test query")
        assert s["query"] == "test query"
