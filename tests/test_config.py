# tests/test_config.py
# Tests for the YAML config loader and all get_*_config() helpers.

from __future__ import annotations

import pytest

from src.config import (
    get_budget_config,
    get_cache_config,
    get_max_tokens,
    get_model,
    get_pipeline_config,
    get_quality_gate_config,
    get_search_config,
    load_config,
)


class TestLoadConfig:
    def test_returns_dict(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_has_top_level_sections(self):
        cfg = load_config()
        for section in ("models", "pipeline", "cache", "search", "budget"):
            assert section in cfg, f"Missing top-level key: {section}"

    def test_is_cached(self):
        """load_config() must return the same object on repeated calls (lru_cache)."""
        a = load_config()
        b = load_config()
        assert a is b


class TestGetModel:
    def test_returns_string(self):
        assert isinstance(get_model("writer"), str)

    def test_known_agents_return_model(self):
        for agent in ("planner", "researcher", "analyst", "synthesizer", "writer", "reviewer"):
            m = get_model(agent)
            assert m, f"Empty model for agent '{agent}'"
            assert "claude" in m.lower(), f"Unexpected model string: {m}"

    def test_unknown_agent_returns_default(self):
        m = get_model("nonexistent_agent_xyz")
        assert isinstance(m, str)
        assert len(m) > 0


class TestGetMaxTokens:
    def test_returns_int(self):
        assert isinstance(get_max_tokens("writer"), int)

    def test_writer_has_most_tokens(self):
        writer_tokens = get_max_tokens("writer")
        planner_tokens = get_max_tokens("planner")
        assert writer_tokens >= planner_tokens

    def test_unknown_agent_returns_default(self):
        tokens = get_max_tokens("nonexistent_agent")
        assert tokens > 0


class TestGetSearchConfig:
    def test_returns_dict(self):
        assert isinstance(get_search_config(), dict)

    def test_has_max_results(self):
        cfg = get_search_config()
        assert "max_results" in cfg
        assert isinstance(cfg["max_results"], int)
        assert cfg["max_results"] > 0

    def test_has_max_wiki_results(self):
        cfg = get_search_config()
        assert "max_wiki_results" in cfg


class TestGetPipelineConfig:
    def test_returns_dict(self):
        assert isinstance(get_pipeline_config(), dict)

    def test_has_required_keys(self):
        cfg = get_pipeline_config()
        for key in ("max_sub_topics", "max_revisions", "quality_threshold",
                    "review_pass_score", "max_quality_retries"):
            assert key in cfg, f"Missing pipeline config key: {key}"

    def test_max_sub_topics_is_positive_int(self):
        val = int(get_pipeline_config()["max_sub_topics"])
        assert 1 <= val <= 10

    def test_quality_threshold_is_between_0_and_1(self):
        val = float(get_pipeline_config()["quality_threshold"])
        assert 0.0 <= val <= 1.0

    def test_review_pass_score_is_between_1_and_10(self):
        val = int(get_pipeline_config()["review_pass_score"])
        assert 1 <= val <= 10


class TestGetCacheConfig:
    def test_returns_dict(self):
        assert isinstance(get_cache_config(), dict)

    def test_has_ttl(self):
        cfg = get_cache_config()
        assert "ttl_seconds" in cfg
        assert int(cfg["ttl_seconds"]) > 0

    def test_has_enabled_flag(self):
        cfg = get_cache_config()
        assert "enabled" in cfg
        assert isinstance(cfg["enabled"], bool)

    def test_has_similarity_threshold(self):
        cfg = get_cache_config()
        assert "similarity_threshold" in cfg
        threshold = float(cfg["similarity_threshold"])
        assert 0.0 < threshold < 1.0


class TestGetBudgetConfig:
    def test_returns_dict(self):
        assert isinstance(get_budget_config(), dict)

    def test_has_limits(self):
        cfg = get_budget_config()
        assert "soft_limit" in cfg
        assert "hard_limit" in cfg

    def test_soft_less_than_hard(self):
        cfg = get_budget_config()
        assert float(cfg["soft_limit"]) < float(cfg["hard_limit"])


class TestGetQualityGateConfig:
    def test_returns_dict(self):
        assert isinstance(get_quality_gate_config(), dict)

    def test_has_weight_keys(self):
        cfg = get_quality_gate_config()
        assert "domain_weight" in cfg
        assert "snippet_weight" in cfg

    def test_weights_sum_to_one(self):
        cfg = get_quality_gate_config()
        total = float(cfg["domain_weight"]) + float(cfg["snippet_weight"])
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_has_trust_lists(self):
        cfg = get_quality_gate_config()
        assert "high_trust_domains" in cfg
        assert isinstance(cfg["high_trust_domains"], list)
        assert len(cfg["high_trust_domains"]) > 0
