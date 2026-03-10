# src/cache/research_cache.py
# ─────────────────────────────────────────────
# Two-level cache with semantic similarity fallback.
#
# Level 1 — exact hash match (O(1), instant):
#   SHA256(normalised_query) → results
#
# Level 2 — semantic similarity (O(n) over cached entries):
#   embed(query) → cosine_similarity against all stored embeddings
#   → return best match above SIMILARITY_THRESHOLD
#
# Embedding model: all-MiniLM-L6-v2 via sentence-transformers
#   - Local inference, no API cost, no network latency
#   - ~30-50ms on CPU per query
#   - 384-dimensional vectors, purpose-built for semantic similarity
# ─────────────────────────────────────────────

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Optional

from src.config import get_cache_config

logger = logging.getLogger(__name__)

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "research_cache.db",
)

# Cosine similarity threshold for semantic cache hits.
# 0.85 catches paraphrases ("what is X" / "explain X" / "define X")
# without false-positives on genuinely different queries.
SIMILARITY_THRESHOLD = 0.85

# Lazy-loaded — only imported on first cache miss to avoid adding
# ~200ms startup overhead to every run.
_embedder = None


def _get_embedder():
    """
    Load the sentence-transformer model once, cache in module-level variable.
    all-MiniLM-L6-v2: 80MB, 384-dim, ~30-50ms per encode on CPU.
    """
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[cache] loading embedding model all-MiniLM-L6-v2...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[cache] embedding model loaded")
        except ImportError:
            logger.warning(
                "[cache] sentence-transformers not installed — "
                "semantic similarity disabled. Run: uv add sentence-transformers"
            )
    return _embedder


def _embed(text: str) -> Optional[list[float]]:
    """
    Embed a query string → list of 384 floats.
    Returns None if embedder unavailable (graceful degradation).
    """
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        vec = embedder.encode(text, convert_to_numpy=True)
        return vec.tolist()
    except Exception as e:
        logger.warning(f"[cache] embed error: {e}")
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity between two equal-length float vectors.
    Pure Python — fast enough for small caches (<1000 entries).
    Returns value in [-1, 1]; 1.0 = identical direction.
    """
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── DB setup ──────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    """
    Create/migrate the cache table.
    Adds 'embedding' column to existing tables if missing (backward-compatible).
    """
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash  TEXT PRIMARY KEY,
                query       TEXT NOT NULL,
                results     TEXT NOT NULL,
                tool_used   TEXT NOT NULL,
                created_at  REAL NOT NULL,
                hit_count   INTEGER DEFAULT 0,
                embedding   TEXT    DEFAULT NULL  -- JSON float array, 384-dim
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_created ON query_cache(created_at)"
        )
        # Migrate existing tables that don't have the embedding column yet
        existing_cols = [
            row[1]
            for row in conn.execute("PRAGMA table_info(query_cache)").fetchall()
        ]
        if "embedding" not in existing_cols:
            conn.execute("ALTER TABLE query_cache ADD COLUMN embedding TEXT DEFAULT NULL")
            logger.info("[cache] migrated: added 'embedding' column to query_cache")


_ensure_table()


def _hash_query(query: str) -> str:
    """SHA256 of lowercased, stripped query — exact match key."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


# ── Public API ────────────────────────────────

def fetch(query: str) -> Optional[list[dict]]:
    """
    Two-level cache lookup:
      1. Exact SHA256 match — O(1), zero compute
      2. Semantic similarity — embeds query, scans all fresh entries

    Returns list[dict] on any hit, None on complete miss.
    """
    cfg = get_cache_config()
    if not cfg.get("enabled", True):
        return None

    ttl    = cfg.get("ttl_seconds", 86400)
    now    = time.time()
    cutoff = now - ttl

    # ── Level 1: Exact match ─────────────────────────────────────────────
    exact_key = _hash_query(query)
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT results, created_at FROM query_cache WHERE query_hash = ?",
                (exact_key,),
            ).fetchone()

        if row and row["created_at"] >= cutoff:
            with _get_conn() as conn:
                conn.execute(
                    "UPDATE query_cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                    (exact_key,),
                )
            age_min = int((now - row["created_at"]) / 60)
            logger.info(
                f"[cache] EXACT HIT ({age_min}min old): '{query[:50]}'"
            )
            return json.loads(row["results"])
    except Exception as e:
        logger.error(f"[cache] exact fetch error: {e}")

    # ── Level 2: Semantic similarity ─────────────────────────────────────
    query_vec = _embed(query)
    if query_vec is None:
        logger.debug(f"[cache] MISS (embedder unavailable): '{query[:50]}'")
        return None

    try:
        with _get_conn() as conn:
            # Only load fresh entries that have embeddings stored
            rows = conn.execute(
                """
                SELECT query_hash, query, results, created_at, embedding
                FROM query_cache
                WHERE created_at >= ? AND embedding IS NOT NULL
                """,
                (cutoff,),
            ).fetchall()
    except Exception as e:
        logger.error(f"[cache] semantic scan error: {e}")
        return None

    if not rows:
        logger.debug(f"[cache] MISS (no candidates for semantic search): '{query[:50]}'")
        return None

    # Find best semantic match
    best_score = 0.0
    best_row   = None

    for row in rows:
        try:
            cached_vec = json.loads(row["embedding"])
            score      = _cosine_similarity(query_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_row   = row
        except Exception:
            continue

    if best_score >= SIMILARITY_THRESHOLD and best_row is not None:
        # Bump hit count on the matched entry
        with _get_conn() as conn:
            conn.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                (best_row["query_hash"],),
            )
        age_min = int((now - best_row["created_at"]) / 60)
        logger.info(
            f"[cache] SEMANTIC HIT (score={best_score:.3f}, {age_min}min old) | "
            f"'{query[:40]}' → matched '{best_row['query'][:40]}'"
        )
        return json.loads(best_row["results"])

    logger.info(
        f"[cache] MISS (best_score={best_score:.3f} < {SIMILARITY_THRESHOLD}): "
        f"'{query[:50]}'"
    )
    return None


def store(query: str, results: list[dict], tool_used: str = "unknown") -> None:
    """
    Store search results + query embedding in the cache.
    Embedding is computed here so fetch() gets instant similarity on next lookup.
    """
    if not get_cache_config().get("enabled", True):
        return

    key = _hash_query(query)

    # URL dedup
    seen_urls: set = set()
    unique: list[dict] = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)

    # Compute embedding at write time — pays cost once, not on every fetch
    embedding_json: Optional[str] = None
    vec = _embed(query.strip())
    if vec is not None:
        embedding_json = json.dumps(vec)

    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_cache
                    (query_hash, query, results, tool_used, created_at, hit_count, embedding)
                VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    key,
                    query.strip(),
                    json.dumps(unique),
                    tool_used,
                    time.time(),
                    embedding_json,
                ),
            )
        embedded_label = "with embedding" if embedding_json else "no embedding"
        logger.debug(
            f"[cache] STORE ({len(unique)} results, {embedded_label}): '{query[:50]}'"
        )
    except Exception as e:
        logger.error(f"[cache] store error: {e}")


def invalidate(query: str) -> None:
    """Force-expire a single cache entry."""
    key = _hash_query(query)
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM query_cache WHERE query_hash = ?", (key,))
        logger.info(f"[cache] invalidated: '{query[:50]}'")
    except Exception as e:
        logger.error(f"[cache] invalidate error: {e}")


def evict_stale() -> int:
    """Delete all entries older than TTL. Returns count deleted."""
    ttl    = get_cache_config().get("ttl_seconds", 86400)
    cutoff = time.time() - ttl
    try:
        with _get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM query_cache WHERE created_at < ?", (cutoff,)
            )
            deleted = cursor.rowcount
        logger.info(f"[cache] evicted {deleted} stale entries")
        return deleted
    except Exception as e:
        logger.error(f"[cache] evict error: {e}")
        return 0


def stats() -> dict:
    """Return cache statistics for the dashboard."""
    try:
        with _get_conn() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            ttl     = get_cache_config().get("ttl_seconds", 86400)
            cutoff  = time.time() - ttl
            fresh   = conn.execute(
                "SELECT COUNT(*) FROM query_cache WHERE created_at >= ?", (cutoff,)
            ).fetchone()[0]
            hits    = conn.execute(
                "SELECT SUM(hit_count) FROM query_cache"
            ).fetchone()[0] or 0
            with_emb = conn.execute(
                "SELECT COUNT(*) FROM query_cache WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {
            "total_entries":    total,
            "fresh_entries":    fresh,
            "stale_entries":    total - fresh,
            "total_hits":       hits,
            "with_embeddings":  with_emb,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }
    except Exception as e:
        logger.error(f"[cache] stats error: {e}")
        return {}
