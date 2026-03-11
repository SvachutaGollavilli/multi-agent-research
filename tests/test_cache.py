# tests/test_cache.py
# Tests for the two-level research cache.
# All tests use a temp SQLite DB and mock the embedder -- no real files touched.

from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import patch

import pytest

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _write_cache_row(db_path: str, query: str, results: list, embedding=None,
                     created_at: float | None = None):
    """Insert a row directly into the temp cache DB for test setup."""
    import hashlib
    key = hashlib.sha256(query.strip().lower().encode()).hexdigest()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO query_cache
                (query_hash, query, results, tool_used, created_at, hit_count, embedding)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (
                key,
                query.strip(),
                json.dumps(results),
                "tavily",
                created_at or time.time(),
                json.dumps(embedding) if embedding else None,
            ),
        )


# -----------------------------------------------------------------
# Patch the cache module to use the tmp DB
# -----------------------------------------------------------------

@pytest.fixture
def patched_cache(tmp_db_path):
    """
    Patch research_cache._DB_PATH to point at the temp DB.
    Also disables the embedder (returns None) so semantic search falls
    back to exact-only -- keeps tests fast and deterministic.
    """
    with patch("src.cache.research_cache._DB_PATH", tmp_db_path), \
         patch("src.cache.research_cache._get_embedder", return_value=None), \
         patch("src.cache.research_cache._embed", return_value=None):
        # Force re-init of the table in the tmp DB
        from src.cache import research_cache
        research_cache._ensure_table()
        yield


# -----------------------------------------------------------------
# fetch()
# -----------------------------------------------------------------

class TestCacheFetch:
    def test_miss_on_empty_cache(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import fetch
        assert fetch("What is FAISS?") is None

    def test_exact_hit_returns_results(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import fetch
        data = [{"title": "T1", "url": "https://example.com", "content": "c"}]
        _write_cache_row(tmp_db_path, "What is FAISS?", data)
        result = fetch("What is FAISS?")
        assert result is not None
        assert len(result) == 1
        assert result[0]["title"] == "T1"

    def test_exact_hit_is_case_insensitive(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import fetch
        data = [{"title": "T1", "url": "https://example.com", "content": "c"}]
        _write_cache_row(tmp_db_path, "what is faiss?", data)
        # Query with different casing
        result = fetch("WHAT IS FAISS?")
        assert result is not None

    def test_stale_entry_not_returned(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import fetch
        data = [{"title": "T1", "url": "https://example.com", "content": "c"}]
        old_time = time.time() - (86400 * 2)  # 2 days ago = beyond TTL
        _write_cache_row(tmp_db_path, "What is FAISS?", data, created_at=old_time)
        result = fetch("What is FAISS?")
        assert result is None

    def test_miss_returns_none(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import fetch
        data = [{"title": "T1", "url": "https://a.com", "content": "c"}]
        _write_cache_row(tmp_db_path, "What is FAISS?", data)
        result = fetch("Completely different query about Docker")
        assert result is None

    def test_disabled_cache_always_misses(self, tmp_db_path):
        data = [{"title": "T1", "url": "https://example.com", "content": "c"}]
        with patch("src.cache.research_cache._DB_PATH", tmp_db_path), \
             patch("src.cache.research_cache.get_cache_config",
                   return_value={"enabled": False, "ttl_seconds": 86400,
                                 "similarity_threshold": 0.60}):
            from src.cache.research_cache import fetch
            _write_cache_row(tmp_db_path, "What is FAISS?", data)
            assert fetch("What is FAISS?") is None


# -----------------------------------------------------------------
# store()
# -----------------------------------------------------------------

class TestCacheStore:
    def test_store_then_fetch(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import fetch, store
        data = [{"title": "Stored", "url": "https://stored.com", "content": "content here"}]
        store("my unique query", data, "tavily")
        result = fetch("my unique query")
        assert result is not None
        assert result[0]["title"] == "Stored"

    def test_store_deduplicates_by_url(self, patched_cache, tmp_db_path):
        import hashlib
        import json
        import sqlite3

        from src.cache.research_cache import store
        data = [
            {"title": "A", "url": "https://dupe.com", "content": "x"},
            {"title": "B", "url": "https://dupe.com", "content": "y"},  # duplicate URL
            {"title": "C", "url": "https://unique.com", "content": "z"},
        ]
        store("dedup test query", data, "tavily")

        key = hashlib.sha256(b"dedup test query").hexdigest()
        with sqlite3.connect(tmp_db_path) as conn:
            row = conn.execute(
                "SELECT results FROM query_cache WHERE query_hash=?", (key,)
            ).fetchone()
        stored = json.loads(row[0])
        assert len(stored) == 2  # dupe removed

    def test_store_upserts_on_same_query(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import fetch, store
        store("upsert query", [{"title": "v1", "url": "https://a.com", "content": "c"}], "t")
        store("upsert query", [{"title": "v2", "url": "https://b.com", "content": "c"}], "t")
        result = fetch("upsert query")
        assert result[0]["title"] == "v2"

    def test_disabled_cache_does_not_store(self, tmp_db_path):
        with patch("src.cache.research_cache._DB_PATH", tmp_db_path), \
             patch("src.cache.research_cache.get_cache_config",
                   return_value={"enabled": False, "ttl_seconds": 86400,
                                 "similarity_threshold": 0.60}), \
             patch("src.cache.research_cache._get_embedder", return_value=None), \
             patch("src.cache.research_cache._embed", return_value=None):
            from src.cache.research_cache import store
            store("disabled store test", [{"title": "x", "url": "https://x.com",
                                           "content": "y"}], "t")

        with sqlite3.connect(tmp_db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
        assert count == 0


# -----------------------------------------------------------------
# stats()
# -----------------------------------------------------------------

class TestCacheStats:
    def test_stats_on_empty_db(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import stats
        s = stats()
        assert s["total_entries"] == 0
        assert s["fresh_entries"] == 0
        assert s["total_hits"]    == 0

    def test_stats_counts_entries(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import stats
        _write_cache_row(tmp_db_path, "query one",   [{"url": "https://a.com", "title": "a", "content": "c"}])
        _write_cache_row(tmp_db_path, "query two",   [{"url": "https://b.com", "title": "b", "content": "c"}])
        s = stats()
        assert s["total_entries"] == 2
        assert s["fresh_entries"] == 2

    def test_stats_counts_stale_separately(self, patched_cache, tmp_db_path):
        from src.cache.research_cache import stats
        old = time.time() - (86400 * 3)
        _write_cache_row(tmp_db_path, "fresh query", [{"url": "https://f.com", "title": "f", "content": "c"}])
        _write_cache_row(tmp_db_path, "stale query", [{"url": "https://s.com", "title": "s", "content": "c"}],
                         created_at=old)
        s = stats()
        assert s["total_entries"] == 2
        assert s["fresh_entries"] == 1
        assert s["stale_entries"] == 1


# -----------------------------------------------------------------
# Semantic similarity (unit test the pure math, no embedder needed)
# -----------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        from src.cache.research_cache import _cosine_similarity
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_score_zero(self):
        from src.cache.research_cache import _cosine_similarity
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector_returns_zero(self):
        from src.cache.research_cache import _cosine_similarity
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_opposite_vectors_score_minus_one(self):
        from src.cache.research_cache import _cosine_similarity
        score = _cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(score - (-1.0)) < 1e-6
