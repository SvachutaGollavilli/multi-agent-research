# evaluation/judge_prompt.py
# ─────────────────────────────────────────────
# LLM-as-Judge for evaluating research reports.
#
# Scores each report on 4 dimensions (1-10 each):
#   accuracy     — claims are factually correct for the topic
#   completeness — all expected key points are addressed
#   citations    — sources are cited, relevant, not hallucinated
#   coherence    — well-structured, clear, no padding
#
# Public API:
#   build_judge_prompt()  — pure function, returns the prompt string (testable)
#   judge_report()        — invokes the LLM judge, returns scored dict
#
# Dimension weights are read from configs/base.yaml eval.weights.
# Composite score = weighted average of all 4 dimensions.
# ─────────────────────────────────────────────

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from src.config import load_config, get_model, get_max_tokens

logger = logging.getLogger(__name__)


def _get_judge_model() -> str:
    cfg = load_config()
    return cfg.get("evaluation", {}).get("judge_model", get_model("default"))


def _get_judge_max_tokens() -> int:
    cfg = load_config()
    return int(cfg.get("evaluation", {}).get("judge_max_tokens", 1024))


def _get_weights() -> dict:
    cfg = load_config()
    return cfg.get("evaluation", {}).get("weights", {
        "accuracy":     0.35,
        "completeness": 0.35,
        "citations":    0.15,
        "coherence":    0.15,
    })


# ── Pydantic output schema ────────────────────

class JudgeOutput(BaseModel):
    accuracy_score: int = Field(
        description="Factual accuracy score 1-10. Are the claims correct?",
        ge=1, le=10,
    )
    accuracy_reasoning: str = Field(
        description="1-2 sentences explaining the accuracy score",
    )
    completeness_score: int = Field(
        description="Completeness score 1-10. Are all expected key points covered?",
        ge=1, le=10,
    )
    completeness_reasoning: str = Field(
        description="Which expected points are present or missing",
    )
    citations_score: int = Field(
        description="Citation quality score 1-10. Are sources cited and relevant?",
        ge=1, le=10,
    )
    citations_reasoning: str = Field(
        description="1-2 sentences on citation quality",
    )
    coherence_score: int = Field(
        description="Coherence score 1-10. Well-structured, clear, no padding?",
        ge=1, le=10,
    )
    coherence_reasoning: str = Field(
        description="1-2 sentences on structure and clarity",
    )
    overall_summary: str = Field(
        description="2-3 sentence overall assessment of the report quality",
    )


# ── Judge LLM singleton ───────────────────────
_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatAnthropic(
            model=_get_judge_model(),
            max_tokens=_get_judge_max_tokens(),
        )
    return _judge_llm


# ════════════════════════════════════════════════
# build_judge_prompt — pure function, no LLM call
# ════════════════════════════════════════════════

def build_judge_prompt(
    query:           str,
    report:          str,
    expected_points: list[str],
) -> str:
    """
    Build the LLM-as-judge evaluation prompt string.

    Pure function — no LLM call, no side effects.
    Separated from judge_report() so tests can verify prompt content
    (keywords, structure) without making real API calls.

    Args:
        query:           the original research question
        report:          the report text to evaluate
        expected_points: key facts/concepts the report should cover

    Returns:
        Full prompt string with accuracy, completeness, citations,
        and coherence scoring instructions embedded.
    """
    expected_str = "\n".join(f"  - {p}" for p in expected_points)

    return (
        "You are an expert evaluator of research reports. "
        "Score the following report objectively on 4 dimensions.\n\n"
        f"Research question: {query}\n\n"
        "Expected key points (a complete report should address ALL of these):\n"
        f"{expected_str}\n\n"
        "Report to evaluate:\n"
        f"{'─' * 60}\n"
        f"{report[:4000]}\n"
        f"{'─' * 60}\n\n"
        "Score each dimension 1-10 with reasoning. Be strict and fair.\n"
        "1-3 = poor, 4-6 = adequate, 7-8 = good, 9-10 = excellent.\n\n"
        "Dimensions:\n"
        "  accuracy     — Are the claims factually correct and supported by sources?\n"
        "  completeness — Are all expected key points covered?\n"
        "  citations    — Are sources cited correctly (not hallucinated)?\n"
        "  coherence    — Is the report well-structured, clear, and free of padding?"
    )


# ════════════════════════════════════════════════
# judge_report — calls the LLM judge
# ════════════════════════════════════════════════

def judge_report(
    query:           str,
    report:          str,
    expected_points: list[str],
    pipeline_label:  str = "pipeline",
) -> dict:
    """
    Score a research report on 4 dimensions using an LLM judge.

    Args:
        query:           the original research question
        report:          the full report text to evaluate
        expected_points: key facts/concepts the report should cover
        pipeline_label:  "multi_agent" or "baseline" — for logging only

    Returns:
        {
          "accuracy":     int,
          "completeness": int,
          "citations":    int,
          "coherence":    int,
          "composite":    float,  # weighted average
          "reasoning":    dict,   # per-dimension explanation strings
          "summary":      str,
          "weights":      dict,
        }
    """
    if not report or not report.strip():
        logger.warning(f"[judge] empty report for '{query[:40]}' ({pipeline_label})")
        return _zero_result()

    weights = _get_weights()
    prompt  = build_judge_prompt(query, report, expected_points)

    for attempt in range(2):
        try:
            structured_llm = _get_judge_llm().with_structured_output(
                JudgeOutput, include_raw=True
            )
            raw    = structured_llm.invoke(prompt)
            result: JudgeOutput = raw["parsed"]

            if result is None:
                if attempt == 0:
                    logger.warning("[judge] parse=None attempt 1, retrying...")
                    continue
                logger.error(f"[judge] parse failed twice for '{query[:40]}'")
                return _zero_result()

            composite = (
                result.accuracy_score     * weights.get("accuracy",     0.35) +
                result.completeness_score * weights.get("completeness", 0.35) +
                result.citations_score    * weights.get("citations",    0.15) +
                result.coherence_score    * weights.get("coherence",    0.15)
            )

            logger.info(
                f"[judge] {pipeline_label} | '{query[:40]}' | "
                f"acc={result.accuracy_score} comp={result.completeness_score} "
                f"cite={result.citations_score} coh={result.coherence_score} "
                f"composite={composite:.2f}"
            )

            return {
                "accuracy":     result.accuracy_score,
                "completeness": result.completeness_score,
                "citations":    result.citations_score,
                "coherence":    result.coherence_score,
                "composite":    round(composite, 2),
                "reasoning": {
                    "accuracy":     result.accuracy_reasoning,
                    "completeness": result.completeness_reasoning,
                    "citations":    result.citations_reasoning,
                    "coherence":    result.coherence_reasoning,
                },
                "summary": result.overall_summary,
                "weights": weights,
            }

        except Exception as e:
            logger.error(f"[judge] attempt {attempt+1} error: {e}")
            if attempt == 1:
                return _zero_result()

    return _zero_result()


def _zero_result() -> dict:
    weights = _get_weights()
    return {
        "accuracy": 0, "completeness": 0, "citations": 0, "coherence": 0,
        "composite": 0.0,
        "reasoning": {
            "accuracy":     "evaluation failed",
            "completeness": "evaluation failed",
            "citations":    "evaluation failed",
            "coherence":    "evaluation failed",
        },
        "summary": "Evaluation failed — report was empty or could not be parsed.",
        "weights": weights,
    }
