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

_config: Optional[dict] = None

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "base.yaml",
)


def load_config(path: Optional[str] = None) -> dict:
    global _config
    if _config is not None and path is None:
        return _config
    config_path = path or _CONFIG_PATH
    try:
        with open(config_path) as f:
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
    global _config
    _config = None
    return load_config()


def get_model(agent_name: str) -> str:
    cfg = load_config()
    models = cfg.get("models", {})
    return models.get(agent_name, models.get("default", "claude-haiku-4-5-20251001"))


def get_max_tokens(agent_name: str) -> int:
    cfg = load_config()
    limits = cfg.get("max_tokens", {})
    return int(limits.get(agent_name, limits.get("default", 1024)))


def get_budget_config() -> dict[str, float]:
    cfg = load_config()
    return cfg.get("budget", {"soft_limit": 0.08, "hard_limit": 0.10})


def get_search_config() -> dict[str, Any]:
    cfg = load_config()
    return cfg.get(
        "search",
        {
            "max_results": 5,
            "search_depth": "basic",
            "max_wiki_results": 3,
        },
    )


def get_pipeline_config() -> dict[str, Any]:
    cfg = load_config()
    return cfg.get(
        "pipeline",
        {
            "max_sub_topics": 3,
            "max_revisions": 2,
            "quality_threshold": 0.4,
            "review_pass_score": 7,
            "max_quality_retries": 1,
        },
    )


def get_cache_config() -> dict[str, Any]:
    cfg = load_config()
    return cfg.get(
        "cache",
        {
            "ttl_seconds": 86400,
            "enabled": True,
            "similarity_threshold": 0.85,
        },
    )


def get_quality_gate_config() -> dict[str, Any]:
    """
    Return quality gate heuristic settings.
    All domain lists and scoring weights live in base.yaml — no code changes
    needed to add/remove trusted domains or tune scoring.
    """
    cfg = load_config()
    return cfg.get("quality_gate", _default_quality_gate())


def get_rate_limit_rpm() -> int:
    cfg = load_config()
    return int(cfg.get("rate_limiter", {}).get("requests_per_minute", 50))


def get_log_level() -> str:
    cfg = load_config()
    return cfg.get("logging", {}).get("level", "INFO")


def get_queue_max_size() -> int:
    cfg = load_config()
    return int(cfg.get("logging", {}).get("queue_max_size", 1000))


def _default_quality_gate() -> dict:
    return {
        "domain_weight": 0.5,
        "snippet_weight": 0.5,
        "snippet_min_chars": 100,
        "snippet_max_chars": 300,
        "boilerplate_penalty": 0.3,
        "domain_trust_high": 1.0,
        "domain_trust_medium": 0.6,
        "domain_trust_low": 0.2,
        "domain_trust_neutral": 0.4,
        "high_trust_domains": [
            "wikipedia.org",
            "arxiv.org",
            "nature.com",
            "science.org",
            "pubmed.ncbi.nlm.nih.gov",
            "scholar.google.com",
            "britannica.com",
            "mit.edu",
            "stanford.edu",
            "harvard.edu",
            "ieee.org",
            "acm.org",
            "bbc.com",
            "reuters.com",
            "apnews.com",
        ],
        "medium_trust_domains": [
            "medium.com",
            "towardsdatascience.com",
            "github.com",
            "stackoverflow.com",
            "techcrunch.com",
            "wired.com",
            "theverge.com",
            "zdnet.com",
            "venturebeat.com",
            "analyticsvidhya.com",
            "kdnuggets.com",
            "neptune.ai",
        ],
        "low_trust_domains": [
            "reddit.com",
            "quora.com",
            "yahoo.com",
            "answers.com",
        ],
        "boilerplate_phrases": [
            "click here",
            "subscribe now",
            "sign up",
            "log in to",
            "cookie policy",
            "privacy policy",
            "all rights reserved",
            "terms of service",
            "javascript is disabled",
            "enable javascript",
            "advertisement",
        ],
    }


def _default_config() -> dict:
    return {
        "models": {
            "planner": "claude-haiku-4-5-20251001",
            "researcher": "claude-haiku-4-5-20251001",
            "analyst": "claude-haiku-4-5-20251001",
            "synthesizer": "claude-haiku-4-5-20251001",
            "writer": "claude-haiku-4-5-20251001",
            "reviewer": "claude-haiku-4-5-20251001",
            "default": "claude-haiku-4-5-20251001",
        },
        "max_tokens": {
            "planner": 256,
            "researcher": 256,
            "analyst": 768,
            "synthesizer": 1024,
            "writer": 1500,
            "reviewer": 512,
            "default": 512,
        },
        "budget": {"soft_limit": 0.08, "hard_limit": 0.10},
        "search": {"max_results": 5, "search_depth": "basic", "max_wiki_results": 3},
        "pipeline": {
            "max_sub_topics": 3,
            "max_revisions": 2,
            "quality_threshold": 0.4,
            "review_pass_score": 7,
            "max_quality_retries": 1,
        },
        "quality_gate": _default_quality_gate(),
        "cache": {
            "ttl_seconds": 86400,
            "enabled": True,
            "similarity_threshold": 0.85,
        },
        "rate_limiter": {"requests_per_minute": 50},
        "logging": {"level": "INFO", "queue_max_size": 1000},
    }


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded\n")
    print("── Models per agent ──")
    for agent in [
        "planner",
        "researcher",
        "analyst",
        "synthesizer",
        "writer",
        "reviewer",
    ]:
        print(
            f"  {agent:12s} → {get_model(agent)}  (max_tokens: {get_max_tokens(agent)})"
        )
    print("\n── Pipeline ──")
    for k, v in get_pipeline_config().items():
        print(f"  {k}: {v}")
    print("\n── Quality gate ──")
    qg = get_quality_gate_config()
    print(f"  domain_weight:       {qg['domain_weight']}")
    print(f"  snippet_weight:      {qg['snippet_weight']}")
    print(f"  snippet_min_chars:   {qg['snippet_min_chars']}")
    print(f"  snippet_max_chars:   {qg['snippet_max_chars']}")
    print(f"  boilerplate_penalty: {qg['boilerplate_penalty']}")
    print(f"  high_trust_domains:  {len(qg['high_trust_domains'])} domains")
    print(f"  medium_trust_domains:{len(qg['medium_trust_domains'])} domains")
    print(f"  low_trust_domains:   {len(qg['low_trust_domains'])} domains")
    print("\n── Cache ──")
    for k, v in get_cache_config().items():
        print(f"  {k}: {v}")
