# tests/test_quality_gate.py
# Tests for the pure-Python quality gate heuristics.
# No mocking needed -- the agent has zero external dependencies.

from __future__ import annotations

import pytest

from src.agents.quality_gate import (
    _domain_trust_score,
    _extract_domain,
    _snippet_quality_score,
    quality_gate_agent,
)
from src.models.state import default_state

# -----------------------------------------------------------------
# _extract_domain
# -----------------------------------------------------------------

class TestExtractDomain:
    @pytest.mark.parametrize("url, expected", [
        ("https://en.wikipedia.org/wiki/FAISS",        "en.wikipedia.org"),
        ("https://www.wikipedia.org/",                  "wikipedia.org"),
        ("http://arxiv.org/abs/1702.08734",             "arxiv.org"),
        ("https://cs.mit.edu/research",                 "cs.mit.edu"),
        ("https://reddit.com/r/MachineLearning",        "reddit.com"),
        ("",                                            ""),
        ("not-a-url",                                   ""),
    ])
    def test_extract(self, url, expected):
        assert _extract_domain(url) == expected


# -----------------------------------------------------------------
# _domain_trust_score
# -----------------------------------------------------------------

class TestDomainTrustScore:
    @pytest.fixture(autouse=True)
    def cfg(self):
        from src.config import get_quality_gate_config
        self.cfg = get_quality_gate_config()

    def test_wikipedia_is_high_trust(self):
        score = _domain_trust_score("en.wikipedia.org", self.cfg)
        assert score == float(self.cfg["domain_trust_high"])

    def test_arxiv_is_high_trust(self):
        score = _domain_trust_score("arxiv.org", self.cfg)
        assert score == float(self.cfg["domain_trust_high"])

    def test_subdomain_inherits_parent_trust(self):
        # cs.mit.edu should match mit.edu -> high trust
        score = _domain_trust_score("cs.mit.edu", self.cfg)
        assert score == float(self.cfg["domain_trust_high"])

    def test_github_is_medium_trust(self):
        score = _domain_trust_score("github.com", self.cfg)
        assert score == float(self.cfg["domain_trust_medium"])

    def test_reddit_is_low_trust(self):
        score = _domain_trust_score("reddit.com", self.cfg)
        assert score == float(self.cfg["domain_trust_low"])

    def test_unknown_domain_is_neutral(self):
        score = _domain_trust_score("totally-unknown-xyz.com", self.cfg)
        assert score == float(self.cfg["domain_trust_neutral"])

    def test_gov_tld_is_high_trust(self):
        score = _domain_trust_score("cdc.gov", self.cfg)
        assert score == float(self.cfg["domain_trust_high"])

    def test_edu_tld_is_high_trust(self):
        score = _domain_trust_score("someuniversity.edu", self.cfg)
        assert score == float(self.cfg["domain_trust_high"])

    def test_score_ordering(self):
        high    = _domain_trust_score("wikipedia.org", self.cfg)
        medium  = _domain_trust_score("github.com", self.cfg)
        low     = _domain_trust_score("reddit.com", self.cfg)
        neutral = _domain_trust_score("unknown.xyz", self.cfg)
        assert high > medium > low
        assert high > neutral > low


# -----------------------------------------------------------------
# _snippet_quality_score
# -----------------------------------------------------------------

class TestSnippetQualityScore:
    @pytest.fixture(autouse=True)
    def cfg(self):
        from src.config import get_quality_gate_config
        self.cfg = get_quality_gate_config()

    def test_empty_content_scores_zero(self):
        assert _snippet_quality_score("", self.cfg) == 0.0

    def test_very_short_content_scores_zero(self):
        assert _snippet_quality_score("Short.", self.cfg) == 0.0

    def test_long_content_scores_one(self):
        long_text = "A" * 500
        assert _snippet_quality_score(long_text, self.cfg) == 1.0

    def test_medium_content_scores_between_0_and_1(self):
        # 200 chars is between min(100) and max(300)
        text  = "X" * 200
        score = _snippet_quality_score(text, self.cfg)
        assert 0.0 < score < 1.0

    def test_boilerplate_penalises_score(self):
        good_text = "X" * 300  # score = 1.0 before penalty
        bad_text  = "X" * 300 + " click here"
        good_score = _snippet_quality_score(good_text, self.cfg)
        bad_score  = _snippet_quality_score(bad_text,  self.cfg)
        assert bad_score < good_score

    def test_multiple_boilerplate_phrases_add_penalties(self):
        base = "X" * 300
        one_phrase   = base + " click here"
        two_phrases  = base + " click here subscribe now"
        assert _snippet_quality_score(two_phrases, self.cfg) <= _snippet_quality_score(one_phrase, self.cfg)

    def test_score_never_goes_below_zero(self):
        text = "short " + " ".join(["click here"] * 20)
        assert _snippet_quality_score(text, self.cfg) >= 0.0

    def test_score_never_above_one(self):
        text = "X" * 10_000
        assert _snippet_quality_score(text, self.cfg) <= 1.0


# -----------------------------------------------------------------
# quality_gate_agent (integration)
# -----------------------------------------------------------------

class TestQualityGateAgent:
    def _make_state(self, sources):
        s = default_state("test query", run_id="test-run")
        s["sources"] = sources
        return s

    def test_no_sources_fails(self):
        state  = self._make_state([])
        result = quality_gate_agent(state)
        assert result["quality_passed"] is False
        assert result["quality_score"]  == 0.0

    def test_high_quality_sources_pass(self):
        sources = [
            {
                "url":     "https://en.wikipedia.org/wiki/FAISS",
                "content": "FAISS is a library for efficient similarity search " * 10,
            },
            {
                "url":     "https://arxiv.org/abs/1702.08734",
                "content": "We present FAISS, a library for efficient similarity " * 10,
            },
        ]
        result = quality_gate_agent(self._make_state(sources))
        assert result["quality_score"] > 0.5

    def test_low_quality_sources_fail(self):
        sources = [
            {
                "url":     "https://reddit.com/r/spam",
                "content": "click here subscribe now log in to view this",
            },
        ]
        result = quality_gate_agent(self._make_state(sources))
        assert result["quality_score"] < 0.5

    def test_returns_required_keys(self):
        state  = self._make_state([{"url": "https://wikipedia.org/", "content": "A" * 200}])
        result = quality_gate_agent(state)
        assert "quality_score"  in result
        assert "quality_passed" in result
        assert "pipeline_trace" in result

    def test_pipeline_trace_has_one_entry(self):
        state  = self._make_state([{"url": "https://wikipedia.org/", "content": "A" * 200}])
        result = quality_gate_agent(state)
        assert len(result["pipeline_trace"]) == 1
        assert result["pipeline_trace"][0]["agent"] == "quality_gate"

    def test_composite_score_between_0_and_1(self):
        sources = [
            {"url": "https://medium.com/ai", "content": "B" * 150},
        ]
        result = quality_gate_agent(self._make_state(sources))
        assert 0.0 <= result["quality_score"] <= 1.0

    def test_mixed_sources_returns_average(self):
        # One high-trust + one low-trust source
        sources = [
            {"url": "https://arxiv.org/",   "content": "A" * 300},
            {"url": "https://reddit.com/r/spam", "content": "B" * 20},
        ]
        result = quality_gate_agent(self._make_state(sources))
        # Should be somewhere in the middle, not fully pass or fail
        assert 0.0 < result["quality_score"] < 1.0
