# src/agents/quality_gate.py
# ─────────────────────────────────────────────
# Quality Gate Agent
#
# Pure-Python heuristic scorer — NO LLM call, NO API cost, runs in <1ms.
#
# Sits between researcher and analyst. Scores raw sources on two axes:
#   1. Domain trust  — is the source from a credible domain?
#   2. Snippet quality — is the content long enough and junk-free?
#
# Composite score = (domain_trust × domain_weight) + (snippet_quality × snippet_weight)
# Both weights and all domain lists are read from configs/base.yaml.
#
# Routing (handled by graph.py, not here):
#   score >= quality_threshold → proceed to analyst
#   score <  quality_threshold AND retries < max_quality_retries → retry researcher
#   score <  quality_threshold AND retries >= max_quality_retries → proceed anyway
# ─────────────────────────────────────────────

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

from src.config import get_pipeline_config, get_quality_gate_config
from src.models.state import ResearchState
from src.observability.logger import log_agent_end, log_agent_start

logger = logging.getLogger(__name__)


def _extract_domain(url: str) -> str:
    """Parse domain from URL, stripping www. prefix."""
    try:
        host = urlparse(url).netloc.lower()
        return host.removeprefix("www.")
    except Exception:
        return ""


def _domain_trust_score(domain: str, cfg: dict) -> float:
    """
    Look up domain in trust tier lists from config.
    Returns the configured trust score for its tier, or neutral if unknown.
    Checks suffix match so subdomains work:
      e.g. "cs.mit.edu" matches "mit.edu" → high trust
    """
    high_trust    = cfg.get("high_trust_domains", [])
    medium_trust  = cfg.get("medium_trust_domains", [])
    low_trust     = cfg.get("low_trust_domains", [])

    score_high    = float(cfg.get("domain_trust_high",    1.0))
    score_medium  = float(cfg.get("domain_trust_medium",  0.6))
    score_low     = float(cfg.get("domain_trust_low",     0.2))
    score_neutral = float(cfg.get("domain_trust_neutral", 0.4))

    # Suffix match — handles subdomains like "cs.mit.edu" → "mit.edu"
    for trusted in high_trust:
        if domain == trusted or domain.endswith("." + trusted):
            return score_high
    for trusted in medium_trust:
        if domain == trusted or domain.endswith("." + trusted):
            return score_medium
    for trusted in low_trust:
        if domain == trusted or domain.endswith("." + trusted):
            return score_low

    # Also grant high trust to any .gov or .edu TLD not explicitly listed
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return score_high

    return score_neutral


def _snippet_quality_score(content: str, cfg: dict) -> float:
    """
    Score the content snippet for length and absence of boilerplate.

    Length scoring (linear interpolation):
      < snippet_min_chars → 0.0
      > snippet_max_chars → 1.0
      between             → proportional

    Boilerplate penalty:
      Each boilerplate phrase found deducts boilerplate_penalty from the score.
      Score is clamped to 0.0 minimum.
    """
    if not content:
        return 0.0

    min_chars   = int(cfg.get("snippet_min_chars", 100))
    max_chars   = int(cfg.get("snippet_max_chars", 300))
    penalty     = float(cfg.get("boilerplate_penalty", 0.3))
    boilerplate = cfg.get("boilerplate_phrases", [])

    # Length score
    length = len(content.strip())
    if length < min_chars:
        length_score = 0.0
    elif length >= max_chars:
        length_score = 1.0
    else:
        length_score = (length - min_chars) / (max_chars - min_chars)

    # Boilerplate penalty
    content_lower = content.lower()
    penalties_hit = sum(1 for phrase in boilerplate if phrase in content_lower)
    final_score   = max(0.0, length_score - (penalties_hit * penalty))

    return final_score


def quality_gate_agent(state: ResearchState) -> dict:
    """
    Quality Gate Agent — pure Python, no LLM.

    Scores each source on domain trust + snippet quality.
    Stores composite score and pass/fail in state.
    The routing decision (retry vs proceed) is made by _should_retry_research()
    in graph.py, not here — this agent only scores and reports.
    """
    run_id  = state.get("run_id", "")
    sources = state.get("sources", [])

    event_id, t0 = log_agent_start(run_id, "quality_gate", {"sources": len(sources)})
    logger.info(f"[quality_gate] scoring {len(sources)} sources")

    cfg      = get_quality_gate_config()
    pipe_cfg = get_pipeline_config()
    threshold       = float(pipe_cfg.get("quality_threshold", 0.4))
    domain_weight   = float(cfg.get("domain_weight",  0.5))
    snippet_weight  = float(cfg.get("snippet_weight", 0.5))

    if not sources:
        log_agent_end(event_id, run_id, "quality_gate", t0, error="No sources to score")
        return {
            "quality_score":   0.0,
            "quality_passed":  False,
            "pipeline_trace": [{
                "agent": "quality_gate", "duration_ms": 0, "tokens": 0,
                "summary": "Skipped — no sources",
            }],
        }

    # ── Score every source ────────────────────────────────────────────────
    domain_scores:  list[float] = []
    snippet_scores: list[float] = []
    per_source_log: list[str]   = []

    for s in sources:
        url     = s.get("url", "")
        content = s.get("content", "")
        domain  = _extract_domain(url)

        d_score = _domain_trust_score(domain, cfg)
        s_score = _snippet_quality_score(content, cfg)

        domain_scores.append(d_score)
        snippet_scores.append(s_score)
        per_source_log.append(
            f"  {domain or 'unknown':35s} domain={d_score:.2f} snippet={s_score:.2f}"
        )
        logger.debug(per_source_log[-1])

    avg_domain  = sum(domain_scores)  / len(domain_scores)
    avg_snippet = sum(snippet_scores) / len(snippet_scores)
    composite   = (avg_domain * domain_weight) + (avg_snippet * snippet_weight)

    passed = composite >= threshold
    verdict = "PASS" if passed else "FAIL"

    elapsed_ms = int((time.time() - t0) * 1000)
    log_agent_end(event_id, run_id, "quality_gate", t0)

    logger.info(
        f"[quality_gate] {verdict} | composite={composite:.3f} "
        f"(domain={avg_domain:.3f} × {domain_weight}, "
        f"snippet={avg_snippet:.3f} × {snippet_weight}) | "
        f"threshold={threshold} | {elapsed_ms}ms"
    )

    return {
        "quality_score":   round(composite, 4),
        "quality_passed":  passed,
        "pipeline_trace": [{
            "agent":       "quality_gate",
            "duration_ms": elapsed_ms,
            "tokens":      0,
            "summary": (
                f"{verdict} score={composite:.3f} "
                f"(domain={avg_domain:.3f}, snippet={avg_snippet:.3f}) "
                f"| {len(sources)} sources"
            ),
        }],
    }
