# tests/conftest.py
# -----------------------------------------------------------------
# Shared fixtures for the entire test suite.
# No real API calls. No real DB writes (all use tmp_path or patch).
# -----------------------------------------------------------------

from __future__ import annotations

import os
import sqlite3
from unittest.mock import MagicMock

import pytest

# ── Prevent any real network / DB activity during tests ──────────
# Set before importing src modules so load_dotenv(override=False)
# inside agents never clobbers these.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("TAVILY_API_KEY",    "test-tavily-not-real")


# -----------------------------------------------------------------
# Sample data fixtures
# -----------------------------------------------------------------

@pytest.fixture
def sample_sources() -> list[dict]:
    return [
        {
            "title":   "FAISS: A Library for Efficient Similarity Search",
            "url":     "https://engineering.fb.com/2017/03/29/data-infrastructure/faiss/",
            "content": (
                "FAISS (Facebook AI Similarity Search) is a library for efficient "
                "similarity search and clustering of dense vectors. It contains "
                "algorithms that search in sets of vectors of any size, up to ones "
                "that possibly do not fit in RAM. It also contains supporting code "
                "for evaluation and parameter tuning."
            ),
        },
        {
            "title":   "FAISS - Wikipedia",
            "url":     "https://en.wikipedia.org/wiki/FAISS",
            "content": (
                "FAISS is a library for similarity search and clustering of dense "
                "vectors developed by Meta AI. It supports both exact and approximate "
                "nearest neighbor search and can run on GPUs."
            ),
        },
        {
            "title":   "Vector Similarity Search with FAISS",
            "url":     "https://www.pinecone.io/learn/faiss/",
            "content": (
                "FAISS supports multiple index types including IVF (Inverted File Index) "
                "and HNSW (Hierarchical Navigable Small World). The choice of index "
                "affects the speed-recall tradeoff significantly."
            ),
        },
    ]


@pytest.fixture
def sample_state(sample_sources) -> dict:
    """A realistic mid-pipeline state dict."""
    return {
        "run_id":          "test-run-001",
        "query":           "What is FAISS and how does it work?",
        "sub_topics":      [
            "FAISS similarity search algorithm",
            "FAISS index types IVF HNSW",
            "FAISS GPU support performance",
        ],
        "research_plan":   "Search FAISS documentation and technical articles.",
        "current_topic":   "",
        "force_research":  False,
        "sources":         sample_sources,
        "search_queries_used": ["FAISS similarity search algorithm"],
        "quality_score":   0.72,
        "quality_passed":  True,
        "quality_retries": 0,
        "key_claims":      [
            {
                "claim":      "FAISS supports billion-scale vector search",
                "source_idx": 1,
                "confidence": "high",
                "evidence":   "Direct quote from FB engineering blog",
            },
            {
                "claim":      "FAISS can run on GPUs",
                "source_idx": 2,
                "confidence": "high",
                "evidence":   "Stated in Wikipedia summary",
            },
        ],
        "conflicts":       [],
        "synthesis":       "FAISS is a highly optimised library for dense vector similarity search.",
        "source_ranking":  [
            {"source_idx": 1, "title": "FAISS FB Blog", "url": "https://engineering.fb.com/..."},
        ],
        "drafts":          [],
        "current_draft":   "## Executive Summary\nFAISS is a library by Meta AI...",
        "revision_count":  1,
        "review":          {"score": 8, "issues": [], "suggestions": [], "passed": True},
        "final_report":    "",
        "pipeline_trace":  [],
        "errors":          [],
        "token_count":     0,
        "cost_usd":        0.0,
    }


# -----------------------------------------------------------------
# Temp SQLite DB fixture
# Used by cache tests to avoid touching the real data/ directory.
# -----------------------------------------------------------------

@pytest.fixture
def tmp_db_path(tmp_path) -> str:
    db_path = str(tmp_path / "test_cache.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash  TEXT PRIMARY KEY,
            query       TEXT NOT NULL,
            results     TEXT NOT NULL,
            tool_used   TEXT NOT NULL,
            created_at  REAL NOT NULL,
            hit_count   INTEGER DEFAULT 0,
            embedding   TEXT DEFAULT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cache_created ON query_cache(created_at)"
    )
    conn.commit()
    conn.close()
    return db_path


# -----------------------------------------------------------------
# Mock LLM response builder
# -----------------------------------------------------------------

def make_mock_llm_response(content: str, input_tokens: int = 100, output_tokens: int = 200):
    """Build a minimal mock that looks like a LangChain AIMessage."""
    mock = MagicMock()
    mock.content = content
    mock.usage_metadata = {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }
    return mock


@pytest.fixture
def mock_llm_response():
    return make_mock_llm_response
