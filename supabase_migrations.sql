-- supabase_migrations.sql
-- -----------------------------------------------------------------
-- Paste this entire file into the Supabase SQL Editor and click Run.
-- It is idempotent -- safe to run multiple times.
--
-- Tables:
--   pipeline_runs   one row per pipeline execution
--   agent_events    one row per agent start/end event
--   cost_ledger     one row per LLM API call
-- -----------------------------------------------------------------

-- ── pipeline_runs ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       TEXT             PRIMARY KEY,
    query        TEXT             NOT NULL,
    status       TEXT             NOT NULL DEFAULT 'running',
    started_at   DOUBLE PRECISION NOT NULL,
    finished_at  DOUBLE PRECISION,
    total_tokens INTEGER          DEFAULT 0,
    total_cost   DOUBLE PRECISION DEFAULT 0.0,
    final_report TEXT
);

-- ── agent_events ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_events (
    event_id    TEXT             PRIMARY KEY,
    run_id      TEXT             NOT NULL REFERENCES pipeline_runs(run_id),
    agent_name  TEXT             NOT NULL,
    status      TEXT             NOT NULL,   -- 'started' | 'completed' | 'failed'
    started_at  DOUBLE PRECISION NOT NULL,
    duration_ms INTEGER,
    tokens_used INTEGER          DEFAULT 0,
    cost_usd    DOUBLE PRECISION DEFAULT 0.0,
    input_hash  TEXT,
    error       TEXT
);

-- ── cost_ledger ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS cost_ledger (
    ledger_id     TEXT             PRIMARY KEY,
    run_id        TEXT             NOT NULL REFERENCES pipeline_runs(run_id),
    agent_name    TEXT             NOT NULL,
    model         TEXT             NOT NULL,
    input_tokens  INTEGER          NOT NULL DEFAULT 0,
    output_tokens INTEGER          NOT NULL DEFAULT 0,
    cost_usd      DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    timestamp     DOUBLE PRECISION NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_events_run
    ON agent_events(run_id);

CREATE INDEX IF NOT EXISTS idx_events_agent
    ON agent_events(agent_name);

CREATE INDEX IF NOT EXISTS idx_ledger_run
    ON cost_ledger(run_id);

CREATE INDEX IF NOT EXISTS idx_ledger_agent
    ON cost_ledger(agent_name);

CREATE INDEX IF NOT EXISTS idx_runs_started
    ON pipeline_runs(started_at DESC);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON pipeline_runs(status);

-- ── Row-Level Security (optional but recommended for Supabase) ────
-- Uncomment and adapt if you want per-user isolation.
-- By default these tables are private (no public access).

-- ALTER TABLE pipeline_runs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE agent_events  ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE cost_ledger   ENABLE ROW LEVEL SECURITY;

-- ── Helpful views ─────────────────────────────────────────────────

-- Cost summary per run
CREATE OR REPLACE VIEW run_cost_summary AS
SELECT
    r.run_id,
    r.query,
    r.status,
    r.started_at,
    r.finished_at,
    r.total_tokens,
    ROUND(r.total_cost::numeric, 6) AS total_cost_usd,
    COUNT(DISTINCT e.event_id)      AS agent_calls,
    COUNT(DISTINCT l.ledger_id)     AS llm_calls
FROM pipeline_runs r
LEFT JOIN agent_events e ON e.run_id = r.run_id
LEFT JOIN cost_ledger  l ON l.run_id = r.run_id
GROUP BY r.run_id
ORDER BY r.started_at DESC;

-- Agent cost breakdown across all runs
CREATE OR REPLACE VIEW agent_cost_breakdown AS
SELECT
    agent_name,
    model,
    COUNT(*)                           AS calls,
    SUM(input_tokens)                  AS total_input_tokens,
    SUM(output_tokens)                 AS total_output_tokens,
    ROUND(SUM(cost_usd)::numeric, 6)   AS total_cost_usd,
    ROUND(AVG(cost_usd)::numeric, 6)   AS avg_cost_per_call
FROM cost_ledger
GROUP BY agent_name, model
ORDER BY total_cost_usd DESC;

-- ── Sanity check ──────────────────────────────────────────────────
-- After running, you should see three tables and two views:
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('pipeline_runs', 'agent_events', 'cost_ledger');
