# src/observability/logger.py
# ─────────────────────────────────────────────
# Background queue logger — agents fire-and-forget.
# Owns: run lifecycle, agent event tracking, async DB writes.
# Imports: db (writes), cost (CostRecord type)
# ─────────────────────────────────────────────

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, Optional

from src.observability import db
from src.observability.cost import CostRecord, RunCostAccumulator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# Write queue — the core of the async pattern
# ═══════════════════════════════════════════════
# Agents push dicts onto this queue.
# The drain thread below pops and writes to DB.
# maxsize=1000: if the queue fills up (DB very slow),
# we drop rather than block agents. Logging is non-critical.

_write_queue: queue.Queue[Optional[dict]] = queue.Queue(maxsize=1000)
_drain_thread: Optional[threading.Thread] = None
_shutdown_event = threading.Event()


def _drain_worker() -> None:
    """
    Background daemon thread — runs for the lifetime of the process.
    Drains the write queue and persists each event to the database.
    Shuts down cleanly when it receives a None sentinel.
    """
    logger.debug("DB drain thread started")
    while not _shutdown_event.is_set():
        try:
            # block up to 0.5s — keeps CPU near 0% when idle
            item = _write_queue.get(timeout=0.5)
            if item is None:            # sentinel — time to shut down
                break
            _persist(item)
            _write_queue.task_done()
        except queue.Empty:
            continue                    # timeout — loop and check shutdown
        except Exception as e:
            logger.error(f"Drain worker error: {e}")
    logger.debug("DB drain thread stopped")


def _persist(item: dict) -> None:
    """
    Write one queued item to the correct DB table.
    Each item carries a 'type' key that routes it.
    """
    try:
        item_type = item.get("type")

        if item_type == "run_start":
            db.execute(
                """
                INSERT OR REPLACE INTO pipeline_runs
                    (run_id, query, status, started_at)
                VALUES (?, ?, 'running', ?)
                """,
                (item["run_id"], item["query"], item["started_at"]),
            )

        elif item_type == "run_end":
            db.execute(
                """
                UPDATE pipeline_runs
                SET status       = ?,
                    finished_at  = ?,
                    total_tokens = ?,
                    total_cost   = ?,
                    final_report = ?
                WHERE run_id = ?
                """,
                (
                    item["status"],
                    item["finished_at"],
                    item["total_tokens"],
                    item["total_cost"],
                    item.get("final_report", ""),
                    item["run_id"],
                ),
            )

        elif item_type == "agent_event":
            db.execute(
                """
                INSERT INTO agent_events
                    (event_id, run_id, agent_name, status,
                     started_at, duration_ms, tokens_used,
                     cost_usd, input_hash, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["event_id"],
                    item["run_id"],
                    item["agent_name"],
                    item["status"],
                    item["started_at"],
                    item.get("duration_ms"),
                    item.get("tokens_used", 0),
                    item.get("cost_usd", 0.0),
                    item.get("input_hash"),
                    item.get("error"),
                ),
            )

        elif item_type == "cost_ledger":
            db.execute(
                """
                INSERT INTO cost_ledger
                    (ledger_id, run_id, agent_name, model,
                     input_tokens, output_tokens, cost_usd, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item["ledger_id"],
                    item["run_id"],
                    item["agent_name"],
                    item["model"],
                    item["input_tokens"],
                    item["output_tokens"],
                    item["cost_usd"],
                    item["timestamp"],
                ),
            )

        else:
            logger.warning(f"Unknown queue item type: {item_type}")

    except Exception as e:
        logger.error(f"_persist error ({item.get('type')}): {e}")


# ═══════════════════════════════════════════════
# Lifecycle — call once at app startup / shutdown
# ═══════════════════════════════════════════════

def start_logger() -> None:
    """
    Start the background drain thread.
    Safe to call multiple times — only starts one thread.
    Call this at app startup (in app.py or graph.py).
    """
    global _drain_thread
    if _drain_thread is not None and _drain_thread.is_alive():
        return                          # already running

    _shutdown_event.clear()
    _drain_thread = threading.Thread(
        target=_drain_worker,
        name="db-drain",
        daemon=True,                    # dies automatically when main process exits
    )
    _drain_thread.start()
    logger.info("Observability logger started")


def stop_logger(timeout: float = 5.0) -> None:
    """
    Flush remaining queue items and stop the drain thread.
    Call this on clean shutdown (optional — daemon thread dies anyway).
    """
    _write_queue.put(None)              # send sentinel
    _shutdown_event.set()
    if _drain_thread:
        _drain_thread.join(timeout=timeout)
    logger.info("Observability logger stopped")


# ═══════════════════════════════════════════════
# Public API — what agents call
# ═══════════════════════════════════════════════

def start_run(query: str) -> str:
    """
    Register a new pipeline run in the DB.
    Returns a fresh run_id (UUID) — store this in state["run_id"].
    Call at the very beginning of graph execution.
    """
    run_id = str(uuid.uuid4())
    _enqueue({
        "type":       "run_start",
        "run_id":     run_id,
        "query":      query,
        "started_at": time.time(),
    })
    logger.info(f"Run started: {run_id[:8]}... | query: '{query[:60]}'")
    return run_id


def end_run(
    run_id: str,
    accumulator: RunCostAccumulator,
    final_report: str = "",
    status: str = "completed",
    total_tokens: Optional[int] = None,
    total_cost: Optional[float] = None,
) -> None:
    """
    Mark a pipeline run as finished in the DB.

    total_tokens / total_cost: pass these directly from the final
    LangGraph result dict (result["token_count"], result["cost_usd"]).
    When provided, they override the accumulator's running totals —
    this is the authoritative source since operator.add reducers
    accumulated them correctly across all parallel agents.

    When NOT provided, falls back to accumulator.summary() values.
    """
    summary = accumulator.summary()

    # Direct values from the result dict are authoritative.
    # The accumulator is a fallback for cases where we don't have result.
    final_tokens = total_tokens if total_tokens is not None else summary["total_tokens"]
    final_cost   = total_cost   if total_cost   is not None else summary["total_cost"]

    _enqueue({
        "type":         "run_end",
        "run_id":       run_id,
        "status":       status,
        "finished_at":  time.time(),
        "total_tokens": final_tokens,
        "total_cost":   round(final_cost, 8),
        "final_report": final_report,
    })

    logger.info(
        f"Run ended: {run_id[:8]}... | status: {status} | "
        f"tokens: {final_tokens} | "
        f"cost: ${final_cost:.6f}"
    )


def log_agent_start(
    run_id: str,
    agent_name: str,
    input_state: Any = None,
) -> tuple[str, float]:
    """
    Log that an agent has started.
    Returns (event_id, start_time) — pass both to log_agent_end().
    """
    event_id   = str(uuid.uuid4())
    start_time = time.time()

    input_hash = _hash_state(input_state) if input_state is not None else None

    _enqueue({
        "type":        "agent_event",
        "event_id":    event_id,
        "run_id":      run_id,
        "agent_name":  agent_name,
        "status":      "started",
        "started_at":  start_time,
        "input_hash":  input_hash,
    })
    logger.debug(f"[{agent_name}] started (run={run_id[:8]}...)")
    return event_id, start_time


def log_agent_end(
    event_id: str,
    run_id: str,
    agent_name: str,
    start_time: float,
    tokens_used: int = 0,
    cost_usd: float = 0.0,
    error: Optional[str] = None,
) -> None:
    """
    Log that an agent has finished (or failed).
    duration_ms is computed automatically from start_time.
    """
    duration_ms = int((time.time() - start_time) * 1000)
    status      = "failed" if error else "completed"

    _enqueue({
        "type":        "agent_event",
        "event_id":    event_id + "_end",
        "run_id":      run_id,
        "agent_name":  agent_name,
        "status":      status,
        "started_at":  start_time,
        "duration_ms": duration_ms,
        "tokens_used": tokens_used,
        "cost_usd":    cost_usd,
        "error":       error,
    })

    if error:
        logger.error(f"[{agent_name}] FAILED in {duration_ms}ms — {error}")
    else:
        logger.info(
            f"[{agent_name}] completed | "
            f"{duration_ms}ms | "
            f"tokens: {tokens_used} | "
            f"cost: ${cost_usd:.6f}"
        )


def log_cost(record: CostRecord) -> None:
    """
    Write one LLM API call to the cost_ledger table.
    Call this immediately after every llm.invoke().
    """
    _enqueue({
        "type":          "cost_ledger",
        "ledger_id":     str(uuid.uuid4()),
        "run_id":        record.run_id,
        "agent_name":    record.agent_name,
        "model":         record.model,
        "input_tokens":  record.input_tokens,
        "output_tokens": record.output_tokens,
        "cost_usd":      record.cost_usd,
        "timestamp":     time.time(),
    })


# ═══════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════

def _enqueue(item: dict) -> None:
    """
    Push an item onto the write queue.
    If full, drop rather than block the agent.
    """
    try:
        _write_queue.put_nowait(item)
    except queue.Full:
        logger.warning(
            f"Write queue full — dropping {item.get('type')} event. "
            f"DB may be slow or drain thread is stuck."
        )


def _hash_state(state: Any) -> str:
    """SHA256 hash of input state for debugging — stored in agent_events."""
    try:
        raw = json.dumps(state, default=str, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
    except Exception:
        return "unhashable"


# ═══════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    from src.observability.cost import RunCostAccumulator, calculate_cost

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    print("Starting logger...")
    start_logger()

    acc    = RunCostAccumulator(run_id="")
    run_id = start_run("How does LangGraph handle parallel execution?")
    acc.run_id = run_id

    for agent_name in ["planner", "researcher", "writer"]:
        event_id, t0 = log_agent_start(run_id, agent_name)
        time.sleep(0.05)

        rec = calculate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=800,
            output_tokens=400,
            agent_name=agent_name,
            run_id=run_id,
        )
        acc.add(rec)
        log_cost(rec)
        log_agent_end(event_id, run_id, agent_name, t0,
                      tokens_used=rec.total_tokens,
                      cost_usd=rec.cost_usd)

    # Pass totals directly — simulates what graph.py does with result dict
    end_run(
        run_id, acc,
        final_report="Test report.",
        status="completed",
        total_tokens=acc.total_tokens,
        total_cost=acc.total_cost,
    )

    print("Waiting for queue to flush...")
    _write_queue.join()
    stop_logger()

    print("\nDB stats:")
    stats = db.get_db_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
