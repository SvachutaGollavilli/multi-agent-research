# src/observability/cost.py
# ─────────────────────────────────────────────
# Token pricing, cost calculation, and budget enforcement.
# Pure Python — no DB, no LangChain imports.
# Rule: agents call calculate_cost() after every LLM call.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# Pricing table
# USD per 1,000 tokens (input / output priced separately)
# Update this dict when Anthropic changes pricing.
# ═══════════════════════════════════════════════

PRICING: dict[str, dict[str, float]] = {
    # Claude Haiku — fast, cheap, used for most agents
    "claude-haiku-4-5-20251001": {
        "input":  0.00025,   # $0.25  per 1M input tokens
        "output": 0.00125,   # $1.25  per 1M output tokens
    },
    # Claude Sonnet — smarter, used for writer/reviewer
    "claude-sonnet-4-6": {
        "input":  0.003,     # $3.00  per 1M input tokens
        "output": 0.015,     # $15.00 per 1M output tokens
    },
    # Claude Opus — most capable, use sparingly
    "claude-opus-4-6": {
        "input":  0.015,     # $15.00 per 1M input tokens
        "output": 0.075,     # $75.00 per 1M output tokens
    },
}

# Fallback pricing for unknown models
_FALLBACK_PRICING: dict[str, float] = {"input": 0.003, "output": 0.015}

# ── Budget limits (USD) ────────────────────────
def _get_limits() -> tuple[float, float]:
    """Read budget limits from config at runtime (not import time)."""
    from src.config import get_budget_config   # local import avoids circular dep
    budget = get_budget_config()
    return float(budget["soft_limit"]), float(budget["hard_limit"])

# Read once at module load — config is cached so this is free
SOFT_LIMIT, HARD_LIMIT = _get_limits()



# ═══════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════

@dataclass
class TokenUsage:
    """
    Token counts from one LLM API call.
    Agents populate this from the LangChain response metadata.
    """
    input_tokens:  int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostRecord:
    """
    Full cost record for one LLM API call.
    Passed to the logger which writes it to cost_ledger.
    """
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    agent_name:    str
    run_id:        str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunCostAccumulator:
    """
    Tracks cumulative cost across all agents in one pipeline run.
    One instance per run — passed through state or held in the logger.
    """
    run_id:        str
    total_cost:    float = 0.0
    total_tokens:  int   = 0
    agent_costs:   dict[str, float] = field(default_factory=dict)
    _warned:       bool  = field(default=False, repr=False)

    def add(self, record: CostRecord) -> None:
        """Add a cost record to the accumulator."""
        self.total_cost   += record.cost_usd
        self.total_tokens += record.total_tokens
        self.agent_costs[record.agent_name] = (
            self.agent_costs.get(record.agent_name, 0.0) + record.cost_usd
        )

    def budget_status(self) -> str:
        """Return 'ok', 'warn', or 'exceeded' based on current spend."""
        if self.total_cost >= HARD_LIMIT:
            return "exceeded"
        if self.total_cost >= SOFT_LIMIT:
            return "warn"
        return "ok"

    def most_expensive_agent(self) -> Optional[str]:
        """Return the agent name with the highest cost so far."""
        if not self.agent_costs:
            return None
        return max(self.agent_costs, key=lambda k: self.agent_costs[k])

    def summary(self) -> dict:
        """Dict summary — used by the UI and end-of-run logging."""
        return {
            "run_id":       self.run_id,
            "total_cost":   round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "agent_costs":  {k: round(v, 6) for k, v in self.agent_costs.items()},
            "status":       self.budget_status(),
            "soft_limit":   SOFT_LIMIT,
            "hard_limit":   HARD_LIMIT,
        }


# ═══════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    agent_name: str = "unknown",
    run_id: str = "",
) -> CostRecord:
    """
    Calculate the USD cost of one LLM API call.

    Usage in an agent:
        record = calculate_cost(
            model="claude-haiku-4-5-20251001",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            agent_name="researcher",
            run_id=state["run_id"],
        )

    Returns a CostRecord — pass it to log_cost() in the logger.
    """
    pricing = PRICING.get(model, _FALLBACK_PRICING)
    if model not in PRICING:
        logger.warning(f"Unknown model '{model}' — using fallback pricing")

    # Cost = tokens / 1000 * price_per_1k
    input_cost  = (input_tokens  / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost  = round(input_cost + output_cost, 8)

    return CostRecord(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=total_cost,
        agent_name=agent_name,
        run_id=run_id,
    )


def check_budget(
    accumulator: RunCostAccumulator,
    agent_name: str = "unknown",
) -> tuple[bool, str]:
    """
    Check if a pipeline run is within budget BEFORE making an LLM call.

    Returns:
        (True, "ok")        → safe to proceed
        (True, "warn")      → over soft limit, log warning but proceed
        (False, "exceeded") → over hard limit, skip the LLM call

    Usage in every agent before calling llm.invoke():
        ok, status = check_budget(accumulator, agent_name="analyst")
        if not ok:
            return fallback_response(state)
    """
    status = accumulator.budget_status()

    if status == "exceeded":
        logger.error(
            f"[{agent_name}] HARD budget exceeded: "
            f"${accumulator.total_cost:.4f} >= ${HARD_LIMIT} — skipping LLM call"
        )
        return False, "exceeded"

    if status == "warn" and not accumulator._warned:
        logger.warning(
            f"[{agent_name}] Soft budget warning: "
            f"${accumulator.total_cost:.4f} >= ${SOFT_LIMIT}"
        )
        accumulator._warned = True  # only warn once per run

    return True, status


def extract_token_usage(response: object) -> TokenUsage:
    """
    Safely extract token counts from a LangChain AIMessage response.

    LangChain stores token usage in response.usage_metadata:
        {
          "input_tokens": 123,
          "output_tokens": 456,
        }

    Falls back to 0s if metadata is missing (avoids crashes on
    models that don't return usage).
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        logger.debug("No usage_metadata in response — token counts will be 0")
        return TokenUsage()

    return TokenUsage(
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
    )


# ═══════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    # Test 1 — basic cost calculation
    record = calculate_cost(
        model="claude-haiku-4-5-20251001",
        input_tokens=1000,
        output_tokens=500,
        agent_name="researcher",
        run_id="test-run-001",
    )
    print(f"Cost record: {record}")
    print(f"  input:  1000 tokens @ $0.00025/1k = ${1000/1000 * 0.00025:.6f}")
    print(f"  output:  500 tokens @ $0.00125/1k = ${500/1000 * 0.00125:.6f}")
    print(f"  total: ${record.cost_usd:.6f}")

    # Test 2 — accumulator + budget check
    print("\n--- Budget accumulation test ---")
    acc = RunCostAccumulator(run_id="test-run-001")

    agents = [
        ("planner",    200,  300),
        ("researcher", 800,  600),
        ("analyst",    1200, 900),
        ("writer",     1500, 1800),
    ]

    for agent, inp, out in agents:
        rec = calculate_cost("claude-haiku-4-5-20251001", inp, out, agent, "test-run-001")
        acc.add(rec)
        ok, status = check_budget(acc, agent)
        print(f"  {agent:12s} | cost: ${rec.cost_usd:.6f} | "
              f"running total: ${acc.total_cost:.4f} | "
              f"budget: {status} | proceed: {ok}")

    print(f"\nRun summary: {acc.summary()}")
    print(f"Most expensive agent: {acc.most_expensive_agent()}")