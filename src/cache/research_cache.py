# src/cache/research_cache.py
# ─────────────────────────────────────────────
# Two-level cache with semantic similarity fallback.
#
# Level 1 — exact hash match (O(1), instant)
# Level 2 — semantic similarity via all-MiniLM-L6-v2 embeddings
#
# Performance design:
#   - HF_HUB_OFFLINE=1 is set before model load so huggingface_hub
#     never makes network calls after the first download.
#     local_files_only=True is NOT sufficient — sentence-transformers v3
#     routes through huggingface_hub which does its own cache validation
#     HEAD requests regardless of that flag.
#   - store() is called from a daemon background thread (see graph.py
#     merge_research_node) so the pipeline never waits for the write.
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

_embedder = None


def _get_similarity_threshold() -> float:
    return float(get_cache_config().get("similarity_threshold", 0.60))


def _get_embedder():
    """
    Load sentence-transformer model once per process with zero network calls.

    HF_HUB_OFFLINE=1 is set before loading — this is the env var that
    huggingface_hub actually checks before every network call. Setting it
    prevents all HEAD/GET requests to huggingface.co on subsequent runs.

    If the model is not yet cached locally (first-ever run), the offline
    attempt raises an exception, we clear the env var, download the model
    once (~80MB), then the next run will use offline mode.
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        # Suppress INFO noise from sentence-transformers, transformers, HF hub
        for noisy_logger in (
            "sentence_transformers",
            "transformers",
            "huggingface_hub",
            "huggingface_hub.utils._http",
        ):
            logging.getLogger(noisy_logger).setLevel(logging.ERROR)

        from sentence_transformers import SentenceTransformer

        # ── Fast path: model already on disk, no network ──────────────────
        # Set HF_HUB_OFFLINE=1 before load — huggingface_hub checks this flag
        # before every HTTP call. local_files_only=True alone is insufficient
        # in sentence-transformers v3 (library still validates cache via HEAD).
        prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[cache] embedding model loaded (offline, no network)")
        except Exception:
            # ── Slow path: first-ever run, model not cached ───────────────
            # Restore previous offline setting (probably unset) and download
            if prev_offline is None:
                del os.environ["HF_HUB_OFFLINE"]
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_offline

            logger.info(
                "[cache] downloading embedding model all-MiniLM-L6-v2 (one-time ~80MB)..."
            )
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[cache] embedding model downloaded and cached")

            # Set offline for remainder of this process now that it's cached
            os.environ["HF_HUB_OFFLINE"] = "1"

    except ImportError:
        logger.warning(
            "[cache] sentence-transformers not installed — "
            "semantic similarity disabled. Run: uv add sentence-transformers"
        )

    return _embedder


def _embed(text: str) -> Optional[list[float]]:
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        vec = embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.tolist()
    except Exception as e:
        logger.warning(f"[cache] embed error: {e}")
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
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
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                query_hash  TEXT PRIMARY KEY,
                query       TEXT NOT NULL,
                results     TEXT NOT NULL,
                tool_used   TEXT NOT NULL,
                created_at  REAL NOT NULL,
                hit_count   INTEGER DEFAULT 0,
                embedding   TEXT    DEFAULT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_created ON query_cache(created_at)"
        )
        existing_cols = [
            row[1] for row in conn.execute("PRAGMA table_info(query_cache)").fetchall()
        ]
        if "embedding" not in existing_cols:
            conn.execute(
                "ALTER TABLE query_cache ADD COLUMN embedding TEXT DEFAULT NULL"
            )
            logger.info("[cache] migrated: added 'embedding' column to query_cache")


_ensure_table()


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


# ── Public API ────────────────────────────────


def fetch(query: str) -> Optional[list[dict]]:
    """
    Two-level cache lookup.
    Level 1: exact SHA256 — O(1), zero compute.
    Level 2: semantic similarity — embeds query, scans fresh entries.
    """
    cfg = get_cache_config()
    if not cfg.get("enabled", True):
        return None

    ttl = cfg.get("ttl_seconds", 86400)
    now = time.time()
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
            logger.info(f"[cache] EXACT HIT ({age_min}min old): '{query[:50]}'")
            return json.loads(row["results"])
    except Exception as e:
        logger.error(f"[cache] exact fetch error: {e}")

    # ── Level 2: Semantic similarity ─────────────────────────────────────
    query_vec = _embed(query)
    if query_vec is None:
        logger.debug(f"[cache] MISS (embedder unavailable): '{query[:50]}'")
        return None

    threshold = _get_similarity_threshold()

    try:
        with _get_conn() as conn:
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
        logger.debug(f"[cache] MISS (no candidates): '{query[:50]}'")
        return None

    best_score = 0.0
    best_row = None
    for row in rows:
        try:
            score = _cosine_similarity(query_vec, json.loads(row["embedding"]))
            if score > best_score:
                best_score = score
                best_row = row
        except Exception:
            continue

    if best_score >= threshold and best_row is not None:
        with _get_conn() as conn:
            conn.execute(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE query_hash = ?",
                (best_row["query_hash"],),
            )
        age_min = int((now - best_row["created_at"]) / 60)
        logger.info(
            f"[cache] SEMANTIC HIT (score={best_score:.3f} >= threshold={threshold}, "
            f"{age_min}min old) | "
            f"'{query[:40]}' → matched '{best_row['query'][:40]}'"
        )
        return json.loads(best_row["results"])

    logger.info(
        f"[cache] MISS (best_score={best_score:.3f} < threshold={threshold}): "
        f"'{query[:50]}'"
    )
    return None


def store(query: str, results: list[dict], tool_used: str = "unknown") -> None:
    """
    Store search results + embedding in cache.
    Thread-safe — called from a daemon background thread in graph.py so the
    pipeline never blocks on the ~40ms embedding compute + SQLite write.
    """
    if not get_cache_config().get("enabled", True):
        return

    key = _hash_query(query)

    seen_urls: set = set()
    unique: list[dict] = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)

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
        label = "with embedding" if embedding_json else "no embedding"
        logger.debug(f"[cache] STORE ({len(unique)} results, {label}): '{query[:50]}'")
    except Exception as e:
        logger.error(f"[cache] store error: {e}")


def invalidate(query: str) -> None:
    key = _hash_query(query)
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM query_cache WHERE query_hash = ?", (key,))
        logger.info(f"[cache] invalidated: '{query[:50]}'")
    except Exception as e:
        logger.error(f"[cache] invalidate error: {e}")


def evict_stale() -> int:
    ttl = get_cache_config().get("ttl_seconds", 86400)
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
    try:
        with _get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            ttl = get_cache_config().get("ttl_seconds", 86400)
            cutoff = time.time() - ttl
            fresh = conn.execute(
                "SELECT COUNT(*) FROM query_cache WHERE created_at >= ?", (cutoff,)
            ).fetchone()[0]
            hits = (
                conn.execute("SELECT SUM(hit_count) FROM query_cache").fetchone()[0]
                or 0
            )
            with_emb = conn.execute(
                "SELECT COUNT(*) FROM query_cache WHERE embedding IS NOT NULL"
            ).fetchone()[0]
        return {
            "total_entries": total,
            "fresh_entries": fresh,
            "stale_entries": total - fresh,
            "total_hits": hits,
            "with_embeddings": with_emb,
            "similarity_threshold": _get_similarity_threshold(),
        }
    except Exception as e:
        logger.error(f"[cache] stats error: {e}")
        return {}
