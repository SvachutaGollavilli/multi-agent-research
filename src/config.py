# src/config.py
# ─────────────────────────────────────────────
# Singleton YAML config loader.
# Loads configs/base.yaml once, caches forever.
# Provides typed accessor functions used by every module.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ── Singleton cache ────────────────────────────
_config: Optional[dict] = None

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "base.yaml",
)


# ═══════════════════════════════════════════════
# Core loader
# ═══════════════════════════════════════════════

def load_config(path: Optional[str] = None) -> dict:
    """
    Load and cache the YAML config.
    Thread-safe for reads (writes only happen once at startup).
    Pass a custom path only for testing.
    """
    global _config
    if _config is not None and path is None:
        return _config                  # cache hit — zero disk I/O

    config_path = path or _CONFIG_PATH
    try:
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f) or {}
        logger.info(f"Config loaded from {config_path}")
    except FileNotFoundError:
        logger.warning(f"Config not found at {config_path} — using defaults")
        _config = _default_config()
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in {config_path}: {e} — using defaults")
        _config = _default_config()

    return _config


def reload_config() -> dict:
    """Force a fresh reload from disk (useful in tests)."""
    global _config
    _config = None
    return load_config()


# ═══════════════════════════════════════════════
# Typed accessors — agents call these, not load_config()
# ═══════════════════════════════════════════════

def get_model(agent_name: str) -> str:
    """
    Return the configured LLM model for a given agent.

    Usage in every agent:
        model = get_model("researcher")
        llm = ChatAnthropic(model=model, ...)
    """
    cfg = load_config()
    models = cfg.get("models", {})
    return models.get(agent_name, models.get("default", "claude-haiku-4-5-20251001"))


def get_max_tokens(agent_name: str) -> int:
    """Return the max_tokens limit for a given agent."""
    cfg = load_config()
    limits = cfg.get("max_tokens", {})
    return int(limits.get(agent_name, limits.get("default", 1024)))


def get_budget_config() -> dict[str, float]:
    """
    Return budget thresholds.
    Keys: soft_limit, hard_limit (both in USD).
    """
    cfg = load_config()
    return cfg.get("budget", {"soft_limit": 0.08, "hard_limit": 0.10})


def get_search_config() -> dict[str, Any]:
    """Return search tool settings (max_results, depth, etc.)."""
    cfg = load_config()
    return cfg.get("search", {
        "max_results":      5,
        "search_depth":     "basic",
        "max_wiki_results": 3,
    })


def get_pipeline_config() -> dict[str, Any]:
    """
    Return pipeline behaviour settings.
    Keys: max_sub_topics, max_revisions, quality_threshold, review_pass_score.
    """
    cfg = load_config()
    return cfg.get("pipeline", {
        "max_sub_topics":    3,
        "max_revisions":     2,
        "quality_threshold": 0.4,
        "review_pass_score": 7,
    })


def get_cache_config() -> dict[str, Any]:
    """Return cache settings (ttl_seconds, enabled)."""
    cfg = load_config()
    return cfg.get("cache", {"ttl_seconds": 86400, "enabled": True})


def get_rate_limit_rpm() -> int:
    """Return the requests-per-minute rate limit."""
    cfg = load_config()
    return int(cfg.get("rate_limiter", {}).get("requests_per_minute", 50))


def get_log_level() -> str:
    """Return the configured log level string."""
    cfg = load_config()
    return cfg.get("logging", {}).get("level", "INFO")


def get_queue_max_size() -> int:
    """Return the max size of the observability write queue."""
    cfg = load_config()
    return int(cfg.get("logging", {}).get("queue_max_size", 1000))


# ═══════════════════════════════════════════════
# Defaults — used if YAML is missing or broken
# ═══════════════════════════════════════════════

def _default_config() -> dict:
    """Hardcoded fallback — mirrors configs/base.yaml exactly."""
    return {
        "models": {
            "planner":     "claude-haiku-4-5-20251001",
            "researcher":  "claude-haiku-4-5-20251001",
            "analyst":     "claude-haiku-4-5-20251001",
            "synthesizer": "claude-haiku-4-5-20251001",
            "writer":      "claude-haiku-4-5-20251001",
            "reviewer":    "claude-haiku-4-5-20251001",
            "default":     "claude-haiku-4-5-20251001",
        },
        "max_tokens": {
            "planner": 512, "researcher": 512, "analyst": 2048,
            "synthesizer": 2048, "writer": 4096, "reviewer": 1024,
            "default": 1024,
        },
        "budget":   {"soft_limit": 0.08, "hard_limit": 0.10},
        "search":   {"max_results": 5, "search_depth": "basic", "max_wiki_results": 3},
        "pipeline": {
            "max_sub_topics": 3, "max_revisions": 2,
            "quality_threshold": 0.4, "review_pass_score": 7,
        },
        "cache":        {"ttl_seconds": 86400, "enabled": True},
        "rate_limiter": {"requests_per_minute": 50},
        "logging":      {"level": "INFO", "queue_max_size": 1000},
    }


# ═══════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    cfg = load_config()

    print(" Config loaded\n")

    print("── Models per agent ──")
    for agent in ["planner", "researcher", "analyst", "synthesizer", "writer", "reviewer"]:
        print(f"  {agent:12s} → {get_model(agent)}  (max_tokens: {get_max_tokens(agent)})")

    print("\n── Budget ──")
    budget = get_budget_config()
    print(f"  soft limit: ${budget['soft_limit']}")
    print(f"  hard limit: ${budget['hard_limit']}")

    print("\n── Pipeline ──")
    pipeline = get_pipeline_config()
    for k, v in pipeline.items():
        print(f"  {k}: {v}")

    print("\n── Search ──")
    search = get_search_config()
    for k, v in search.items():
        print(f"  {k}: {v}")

    print(f"\n── Rate limit: {get_rate_limit_rpm()} rpm ──")
    print(f"── Log level:  {get_log_level()} ──")