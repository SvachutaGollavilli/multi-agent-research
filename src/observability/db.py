# src/observability/db.py
# -----------------------------------------------------------------
# Database layer -- SQLite (local dev) or PostgreSQL/Supabase (production).
# Backend is selected automatically via DATABASE_URL env var.
#
# Supabase setup:
#   1. Create a Supabase project at supabase.com
#   2. Run supabase_migrations.sql in the Supabase SQL Editor
#   3. Set DATABASE_URL to your connection string (use port 6543 for
#      transaction-mode PgBouncer -- handles connection pooling for you)
#   4. Example: postgresql://postgres.[ref]:[pw]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
#
# psycopg3 (the `psycopg` package) is used for PostgreSQL.
# The placeholder translation (? -> %s) is handled automatically.
# -----------------------------------------------------------------

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Backend detection
# -----------------------------------------------------------------

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/pipeline.db")

_IS_POSTGRES: bool = DATABASE_URL.startswith(
    "postgresql://"
) or DATABASE_URL.startswith("postgres://")

_BASE_DIR: str = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_SQLITE_PATH: str = (
    os.path.join(_BASE_DIR, DATABASE_URL.replace("sqlite:///", ""))
    if not _IS_POSTGRES
    else os.path.join(_BASE_DIR, "data", "pipeline.db")
)

# Thread-local storage for SQLite connections (one per thread)
_local = threading.local()

# -----------------------------------------------------------------
# Postgres connection pool (psycopg3)
# -----------------------------------------------------------------
# We use a simple module-level lock + connection-per-call pattern.
# Supabase's built-in PgBouncer (port 6543) handles server-side
# pooling, so we don't need a client-side pool here.
# -----------------------------------------------------------------
_PG_LOCK = threading.Lock()


def _pg_connect() -> Any:
    """
    Open a psycopg3 connection to Postgres/Supabase.
    Each call returns a fresh connection -- Supabase PgBouncer reuses
    the underlying server connections automatically when port 6543 is used.
    """
    import psycopg  # psycopg3 -- installed as the `psycopg` package

    # autocommit=False: we commit explicitly after each write
    conn = psycopg.connect(DATABASE_URL, autocommit=False)
    return conn


# -----------------------------------------------------------------
# SQLite connections (thread-local, one per thread)
# -----------------------------------------------------------------


def _get_sqlite_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(_SQLITE_PATH), exist_ok=True)
        conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _create_tables_sqlite(conn)
        _local.conn = conn
        logger.debug(f"SQLite connected: {_SQLITE_PATH}")
    return _local.conn  # type: ignore[return-value]


def close_connection() -> None:
    """Close the thread-local SQLite connection. No-op for PostgreSQL."""
    if not _IS_POSTGRES:
        conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
        if conn:
            conn.close()
            _local.conn = None


# -----------------------------------------------------------------
# Schema -- SQLite
# -----------------------------------------------------------------


def _create_tables_sqlite(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id       TEXT    PRIMARY KEY,
            query        TEXT    NOT NULL,
            status       TEXT    NOT NULL DEFAULT 'running',
            started_at   REAL    NOT NULL,
            finished_at  REAL,
            total_tokens INTEGER DEFAULT 0,
            total_cost   REAL    DEFAULT 0.0,
            final_report TEXT
        );

        CREATE TABLE IF NOT EXISTS agent_events (
            event_id    TEXT    PRIMARY KEY,
            run_id      TEXT    NOT NULL,
            agent_name  TEXT    NOT NULL,
            status      TEXT    NOT NULL,
            started_at  REAL    NOT NULL,
            duration_ms INTEGER,
            tokens_used INTEGER DEFAULT 0,
            cost_usd    REAL    DEFAULT 0.0,
            input_hash  TEXT,
            error       TEXT
        );

        CREATE TABLE IF NOT EXISTS cost_ledger (
            ledger_id     TEXT    PRIMARY KEY,
            run_id        TEXT    NOT NULL,
            agent_name    TEXT    NOT NULL,
            model         TEXT    NOT NULL,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd      REAL    NOT NULL DEFAULT 0.0,
            timestamp     REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_events_run
            ON agent_events(run_id);
        CREATE INDEX IF NOT EXISTS idx_ledger_run
            ON cost_ledger(run_id);
        CREATE INDEX IF NOT EXISTS idx_runs_started
            ON pipeline_runs(started_at DESC);
    """)
    conn.commit()
    logger.debug("SQLite tables verified")


# -----------------------------------------------------------------
# Schema -- PostgreSQL
# (idempotent -- safe to run multiple times)
# -----------------------------------------------------------------


def _ensure_tables_postgres() -> None:
    """Create tables on Postgres if they don't exist yet."""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id       TEXT    PRIMARY KEY,
            query        TEXT    NOT NULL,
            status       TEXT    NOT NULL DEFAULT 'running',
            started_at   DOUBLE PRECISION NOT NULL,
            finished_at  DOUBLE PRECISION,
            total_tokens INTEGER DEFAULT 0,
            total_cost   DOUBLE PRECISION DEFAULT 0.0,
            final_report TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS agent_events (
            event_id    TEXT    PRIMARY KEY,
            run_id      TEXT    NOT NULL,
            agent_name  TEXT    NOT NULL,
            status      TEXT    NOT NULL,
            started_at  DOUBLE PRECISION NOT NULL,
            duration_ms INTEGER,
            tokens_used INTEGER DEFAULT 0,
            cost_usd    DOUBLE PRECISION DEFAULT 0.0,
            input_hash  TEXT,
            error       TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cost_ledger (
            ledger_id     TEXT    PRIMARY KEY,
            run_id        TEXT    NOT NULL,
            agent_name    TEXT    NOT NULL,
            model         TEXT    NOT NULL,
            input_tokens  INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            timestamp     DOUBLE PRECISION NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_events_run    ON agent_events(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_ledger_run    ON cost_ledger(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_runs_started  ON pipeline_runs(started_at DESC)",
    ]
    with _pg_connect() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
        conn.commit()
    logger.debug("PostgreSQL tables verified")


# Ensure tables exist at import time for both backends
if _IS_POSTGRES:
    try:
        _ensure_tables_postgres()
        logger.info(f"[db] PostgreSQL backend active: {DATABASE_URL[:40]}...")
    except Exception as e:
        logger.error(f"[db] PostgreSQL table init failed: {e}")
else:
    # SQLite tables are created lazily on first connection (in _get_sqlite_conn)
    logger.debug(f"[db] SQLite backend: {_SQLITE_PATH}")


# -----------------------------------------------------------------
# Unified query helpers
# -----------------------------------------------------------------


def execute(sql: str, params: tuple = ()) -> None:
    """
    Execute a write (INSERT / UPDATE / DELETE) on the active backend.
    SQLite uses ? placeholders -- auto-translated to %s for Postgres.
    """
    if _IS_POSTGRES:
        pg_sql = sql.replace("?", "%s")
        with _pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(pg_sql, params)
            conn.commit()
    else:
        conn = _get_sqlite_conn()
        conn.execute(sql, params)
        conn.commit()


def fetchall(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SELECT and return all rows as plain dicts."""
    if _IS_POSTGRES:
        pg_sql = sql.replace("?", "%s")
        with _pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(pg_sql, params)
                # psycopg3: cur.description is a list of Column namedtuples
                cols = [col.name for col in (cur.description or [])]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
    else:
        conn = _get_sqlite_conn()
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def fetchone(sql: str, params: tuple = ()) -> Optional[dict]:
    """Execute a SELECT and return the first row, or None."""
    rows = fetchall(sql, params)
    return rows[0] if rows else None


# -----------------------------------------------------------------
# Stats helper
# -----------------------------------------------------------------


def get_db_stats() -> dict:
    try:
        runs_row = fetchone("SELECT COUNT(*) as n FROM pipeline_runs") or {}
        events_row = fetchone("SELECT COUNT(*) as n FROM agent_events") or {}
        cost_row = (
            fetchone(
                "SELECT COALESCE(SUM(total_cost), 0.0) as total FROM pipeline_runs"
            )
            or {}
        )
        return {
            "total_runs": runs_row.get("n", 0),
            "total_agent_events": events_row.get("n", 0),
            "total_cost_usd": round(float(cost_row.get("total", 0.0)), 6),
            "backend": "postgresql" if _IS_POSTGRES else "sqlite",
            "db_url": DATABASE_URL[:50] + "..."
            if len(DATABASE_URL) > 50
            else DATABASE_URL,
        }
    except Exception as e:
        logger.error(f"get_db_stats error: {e}")
        return {}


# -----------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------

if __name__ == "__main__":
    print(f"Backend : {'PostgreSQL' if _IS_POSTGRES else 'SQLite'}")
    print(f"URL     : {DATABASE_URL}")
    stats = get_db_stats()
    print("Database ready")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    close_connection()
