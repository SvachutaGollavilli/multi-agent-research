# src/observability/db.py
# ─────────────────────────────────────────────
# Database foundation — supports SQLite (local) and PostgreSQL (production).
# Backend selected via DATABASE_URL environment variable.
# Rule: nothing in this file imports from other src/ modules.
# ─────────────────────────────────────────────

from __future__ import annotations

import os
import sqlite3
import threading
import logging
from typing import Any, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# Backend detection
# ═══════════════════════════════════════════════

# os.getenv with a default always returns str — safe to use directly
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/pipeline.db")

_IS_POSTGRES: bool = DATABASE_URL.startswith("postgresql://") or \
                     DATABASE_URL.startswith("postgres://")

# ── SQLite path ────────────────────────────────
# os.path.abspath guarantees __file__ resolves to a real str,
# eliminating the "str | None" ambiguity Pylance raises.
_BASE_DIR: str = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Always a str — when postgres, we just never use it.
# This avoids the str | None type that caused issue #2.
_SQLITE_PATH: str = os.path.join(
    _BASE_DIR,
    DATABASE_URL.replace("sqlite:///", "")
) if not _IS_POSTGRES else os.path.join(_BASE_DIR, "data", "pipeline.db")

# ── Thread-local pool (SQLite only) ───────────
_local = threading.local()


# ═══════════════════════════════════════════════
# Connections
# ═══════════════════════════════════════════════

def get_connection() -> Any:
    """
    Return a live database connection.
    SQLite  → thread-local (one connection per thread, reused).
    Postgres → new connection per call (use pgBouncer at real scale).
    Return type is Any because sqlite3.Connection and
    psycopg2.connection are unrelated types.
    """
    if _IS_POSTGRES:
        return _get_postgres_conn()
    return _get_sqlite_conn()


def _get_sqlite_conn() -> sqlite3.Connection:
    """Return (or create) the thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(_SQLITE_PATH), exist_ok=True)
        conn = sqlite3.connect(_SQLITE_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _create_tables_sqlite(conn)
        _local.conn = conn
        logger.debug(f"SQLite connected: {_SQLITE_PATH}")
    return _local.conn  # type: ignore[return-value]


def _get_postgres_conn() -> Any:
    """Open a new psycopg2 connection. Caller must close it."""
    import psycopg2  # lazy import — only installed when actually needed
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    _create_tables_postgres(conn)
    logger.debug("PostgreSQL connected")
    return conn


def close_connection() -> None:
    """Close the thread-local SQLite connection (no-op for PostgreSQL)."""
    if not _IS_POSTGRES:
        conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
        if conn is not None:
            conn.close()
            _local.conn = None


# ═══════════════════════════════════════════════
# Schema — SQLite
# ═══════════════════════════════════════════════

def _create_tables_sqlite(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes for SQLite backend."""
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


# ═══════════════════════════════════════════════
# Schema — PostgreSQL
# ═══════════════════════════════════════════════

def _create_tables_postgres(conn: Any) -> None:
    """Create all tables and indexes for PostgreSQL backend."""
    # psycopg2 connections require an explicit cursor for all queries.
    # Unlike sqlite3, there is no conn.execute() shortcut.
    cur = conn.cursor()

    # Each CREATE TABLE must be a separate execute call in psycopg2
    # (executescript is SQLite-only).
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
        "CREATE INDEX IF NOT EXISTS idx_events_run ON agent_events(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_ledger_run ON cost_ledger(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_runs_started ON pipeline_runs(started_at DESC)",
    ]

    for stmt in statements:
        cur.execute(stmt)

    conn.commit()
    cur.close()
    logger.debug("PostgreSQL tables verified")


# ═══════════════════════════════════════════════
# Query helpers — unified API for both backends
# ═══════════════════════════════════════════════

def execute(sql: str, params: tuple = ()) -> None:
    """
    Execute a write (INSERT / UPDATE / DELETE) on the active backend.

    SQLite  uses  ?  placeholders  →  kept as-is.
    Postgres uses %s placeholders  →  auto-translated.
    """
    conn = get_connection()
    if _IS_POSTGRES:
        sql = sql.replace("?", "%s")
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close()
        conn.close()
    else:
        # sqlite3.Connection.execute() is a valid shortcut
        sqlite_conn: sqlite3.Connection = conn
        sqlite_conn.execute(sql, params)
        sqlite_conn.commit()


def fetchall(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read query and return all rows as plain dicts."""
    conn = get_connection()
    if _IS_POSTGRES:
        import psycopg2.extras  # lazy import
        sql = sql.replace("?", "%s")
        # Use a plain cursor + manual dict conversion to avoid
        # the RealDictCursor overload issue Pylance raises
        cur = conn.cursor()
        cur.execute(sql, params)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    else:
        sqlite_conn: sqlite3.Connection = conn
        return [dict(row) for row in sqlite_conn.execute(sql, params).fetchall()]


def fetchone(sql: str, params: tuple = ()) -> Optional[dict]:
    """Execute a read query and return the first row, or None."""
    rows = fetchall(sql, params)
    return rows[0] if rows else None


# ═══════════════════════════════════════════════
# Stats helper — used by the UI dashboard
# ═══════════════════════════════════════════════

def get_db_stats() -> dict:
    """Return a quick health summary of the pipeline database."""
    try:
        # fetchone can return None — guard every access with `or {}`
        runs_row    = fetchone("SELECT COUNT(*) as n FROM pipeline_runs") or {}
        events_row  = fetchone("SELECT COUNT(*) as n FROM agent_events") or {}
        cost_row    = fetchone(
            "SELECT COALESCE(SUM(total_cost), 0.0) as total FROM pipeline_runs"
        ) or {}

        return {
            "total_runs":       runs_row.get("n", 0),
            "total_agent_events": events_row.get("n", 0),
            "total_cost_usd":   round(float(cost_row.get("total", 0.0)), 6),
            "backend":          "postgresql" if _IS_POSTGRES else "sqlite",
            "db_path":          DATABASE_URL,
        }
    except Exception as e:
        logger.error(f"get_db_stats error: {e}")
        return {}


# ═══════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Backend : {'PostgreSQL' if _IS_POSTGRES else 'SQLite'}")
    print(f"URL     : {DATABASE_URL}")
    stats = get_db_stats()
    print(f" Database ready")
    for k, v in stats.items():
        print(f"   {k}: {v}")
    close_connection()