# evaluation/run_eval.py
# -----------------------------------------------------------------
# Evaluation runner -- compares multi-agent pipeline vs single-agent baseline.
#
# Usage:
#   uv run python evaluation/run_eval.py                           # full eval
#   uv run python evaluation/run_eval.py --id faiss_overview       # one question
#   uv run python evaluation/run_eval.py --category ml_tools       # one category
#   uv run python evaluation/run_eval.py --no-baseline             # skip baseline
#   uv run python evaluation/run_eval.py --list                    # show saved results
#   uv run python evaluation/run_eval.py --load 2026-03-10T20-30-34.json
#
# Outputs:
#   evaluation/results/{timestamp}.json   full results with all scores + reasoning
#   Console: comparison table with delta column
# -----------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import yaml

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from evaluation.baseline     import run_baseline
from evaluation.judge_prompt import judge_report
from src.agents.graph        import run_pipeline
from src.config              import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

_QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.yaml")
_RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")


# -- Config helpers -------------------------------------------------------

def _get_eval_config() -> dict:
    cfg = load_config()
    return cfg.get("evaluation", {
        "judge_model":      "claude-haiku-4-5-20251001",
        "judge_max_tokens": 1024,
        "weights": {
            "accuracy": 0.35, "completeness": 0.35,
            "citations": 0.15, "coherence": 0.15,
        },
    })


# -- Question loading -------------------------------------------------------

def load_questions(
    question_id: str | None = None,
    category:    str | None = None,
) -> list[dict]:
    with open(_QUESTIONS_PATH) as f:
        data = yaml.safe_load(f)
    questions = data.get("questions", [])

    if question_id:
        questions = [q for q in questions if q["id"] == question_id]
    if category:
        questions = [q for q in questions if q.get("category") == category]

    if not questions:
        raise ValueError(
            f"No questions matched -- id={question_id!r}, category={category!r}"
        )
    return questions


# -- Single-question evaluation --------------------------------------------

def evaluate_question(
    question:      dict,
    run_baseline_: bool = True,
) -> dict:
    qid      = question["id"]
    query    = question["query"]
    expected = question.get("expected_points", [])

    logger.info("=" * 60)
    logger.info(f"[eval] question: {qid} | '{query[:60]}'")
    logger.info("=" * 60)

    result = {
        "id":            qid,
        "query":         query,
        "category":      question.get("category", ""),
        "expected_points": expected,
        "multi_agent":   {},
        "baseline":      {},
        "delta":         {},
        "timestamp":     datetime.utcnow().isoformat(),
    }

    # Multi-agent pipeline
    logger.info("[eval] running multi-agent pipeline...")
    ma_t0 = time.time()
    try:
        ma_state  = run_pipeline(query)
        ma_report = ma_state.get("final_report") or ma_state.get("current_draft", "")
        ma_duration = round(time.time() - ma_t0, 2)

        ma_scores = judge_report(
            query=query,
            report=ma_report,
            expected_points=expected,
            pipeline_label="multi_agent",
        )

        result["multi_agent"] = {
            "report":      ma_report,
            "scores":      ma_scores,
            "duration_s":  ma_duration,
            "sources":     len(ma_state.get("sources", [])),
            "claims":      len(ma_state.get("key_claims", [])),
            "revisions":   ma_state.get("revision_count", 0),
            "final_score": ma_state.get("review", {}).get("score", 0),
            "cost_usd":    ma_state.get("cost_usd", 0.0),
            "tokens":      ma_state.get("token_count", 0),
        }
        logger.info(
            f"[eval] multi-agent done | composite={ma_scores['composite']:.2f} | {ma_duration}s"
        )
    except Exception as e:
        logger.error(f"[eval] multi-agent failed for {qid}: {e}")
        result["multi_agent"] = {"error": str(e), "scores": {"composite": 0}}

    # Baseline pipeline
    if run_baseline_:
        logger.info("[eval] running baseline pipeline...")
        bl_result = run_baseline(query)
        bl_scores = judge_report(
            query=query,
            report=bl_result["report"],
            expected_points=expected,
            pipeline_label="baseline",
        )

        result["baseline"] = {
            "report":     bl_result["report"],
            "scores":     bl_scores,
            "duration_s": bl_result["duration_s"],
            "sources":    len(bl_result.get("sources", [])),
            "error":      bl_result.get("error"),
        }
        logger.info(
            f"[eval] baseline done | composite={bl_scores['composite']:.2f} | {bl_result['duration_s']}s"
        )

        ma_s = result["multi_agent"].get("scores", {})
        result["delta"] = {
            "accuracy":     ma_s.get("accuracy", 0)     - bl_scores.get("accuracy", 0),
            "completeness": ma_s.get("completeness", 0) - bl_scores.get("completeness", 0),
            "citations":    ma_s.get("citations", 0)    - bl_scores.get("citations", 0),
            "coherence":    ma_s.get("coherence", 0)    - bl_scores.get("coherence", 0),
            "composite":    round(
                ma_s.get("composite", 0.0) - bl_scores.get("composite", 0.0), 2
            ),
        }
        sign = "+" if result["delta"]["composite"] >= 0 else ""
        logger.info(f"[eval] delta composite: {sign}{result['delta']['composite']:.2f}")

    return result


# -- Full eval run ---------------------------------------------------------

def run_eval(
    question_id:   str | None = None,
    category:      str | None = None,
    run_baseline_: bool = True,
) -> dict:
    questions = load_questions(question_id=question_id, category=category)
    logger.info(
        f"[eval] starting | {len(questions)} questions | "
        f"baseline={'yes' if run_baseline_ else 'no'}"
    )

    all_results = []
    for q in questions:
        res = evaluate_question(q, run_baseline_=run_baseline_)
        all_results.append(res)

    def avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    ma_composites = [
        r["multi_agent"]["scores"]["composite"]
        for r in all_results
        if "scores" in r.get("multi_agent", {})
    ]
    bl_composites = [
        r["baseline"]["scores"]["composite"]
        for r in all_results
        if run_baseline_ and "scores" in r.get("baseline", {})
    ]

    summary = {
        "total_questions": len(all_results),
        "ran_baseline":    run_baseline_,
        "multi_agent_avg": avg(ma_composites),
        "baseline_avg":    avg(bl_composites) if run_baseline_ else None,
        "avg_delta":       round(avg(ma_composites) - avg(bl_composites), 2)
                           if run_baseline_ and bl_composites else None,
        "by_category":     _category_breakdown(all_results, run_baseline_),
    }

    run_data = {
        "eval_timestamp": datetime.utcnow().isoformat(),
        "summary":        summary,
        "results":        all_results,
    }

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    ts_safe  = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    out_path = os.path.join(_RESULTS_DIR, f"{ts_safe}.json")
    with open(out_path, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    logger.info(f"[eval] results saved -> {out_path}")

    _print_table(all_results, summary, run_baseline_)
    print(f"\n  Saved: {out_path}")
    print(f"  Reload: uv run python evaluation/run_eval.py --load {os.path.basename(out_path)}\n")
    return run_data


def _category_breakdown(results: list[dict], run_baseline_: bool) -> dict:
    cats: dict = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in cats:
            cats[cat] = {"questions": 0, "ma_scores": [], "bl_scores": []}
        cats[cat]["questions"] += 1
        ma_c = r.get("multi_agent", {}).get("scores", {}).get("composite", 0)
        cats[cat]["ma_scores"].append(ma_c)
        if run_baseline_:
            bl_c = r.get("baseline", {}).get("scores", {}).get("composite", 0)
            cats[cat]["bl_scores"].append(bl_c)

    breakdown = {}
    for cat, data in cats.items():
        ma_avg = round(sum(data["ma_scores"]) / len(data["ma_scores"]), 2)
        bl_avg = (
            round(sum(data["bl_scores"]) / len(data["bl_scores"]), 2)
            if run_baseline_ and data["bl_scores"] else None
        )
        breakdown[cat] = {
            "questions":       data["questions"],
            "multi_agent_avg": ma_avg,
            "baseline_avg":    bl_avg,
            "delta":           round(ma_avg - bl_avg, 2) if bl_avg is not None else None,
        }
    return breakdown


def _print_table(results: list[dict], summary: dict, run_baseline_: bool) -> None:
    W = 90
    print("\n" + "=" * W)
    print("  EVALUATION RESULTS")
    print("=" * W)

    if run_baseline_:
        print(
            f"  {'Question':<25} {'Category':<16} "
            f"{'Multi':<7} {'Base':<7} {'Delta':<8} "
            f"{'Acc':>4} {'Cmp':>4} {'Cit':>4} {'Coh':>4}"
        )
    else:
        print(
            f"  {'Question':<25} {'Category':<16} "
            f"{'Multi':<7} "
            f"{'Acc':>4} {'Cmp':>4} {'Cit':>4} {'Coh':>4}"
        )
    print("  " + "-" * (W - 2))

    for r in results:
        qid  = r["id"][:24]
        cat  = r.get("category", "")[:15]
        ma   = r.get("multi_agent", {}).get("scores", {})
        ma_c = f"{ma.get('composite', 0):.1f}"

        if run_baseline_:
            bl    = r.get("baseline", {}).get("scores", {})
            bl_c  = f"{bl.get('composite', 0):.1f}"
            dc    = r.get("delta", {}).get("composite", 0)
            sign  = "+" if dc >= 0 else ""
            arrow = "^" if dc > 0 else ("v" if dc < 0 else "=")
            print(
                f"  {qid:<25} {cat:<16} "
                f"{ma_c:<7} {bl_c:<7} {arrow}{sign}{dc:<6.1f} "
                f"{ma.get('accuracy',0):>4} "
                f"{ma.get('completeness',0):>4} "
                f"{ma.get('citations',0):>4} "
                f"{ma.get('coherence',0):>4}"
            )
        else:
            print(
                f"  {qid:<25} {cat:<16} "
                f"{ma_c:<7} "
                f"{ma.get('accuracy',0):>4} "
                f"{ma.get('completeness',0):>4} "
                f"{ma.get('citations',0):>4} "
                f"{ma.get('coherence',0):>4}"
            )

    print("  " + "-" * (W - 2))
    ma_avg = summary.get("multi_agent_avg", 0)
    if run_baseline_:
        bl_avg    = summary.get("baseline_avg", 0) or 0
        avg_delta = summary.get("avg_delta",    0) or 0
        sign      = "+" if avg_delta >= 0 else ""
        arrow     = "^" if avg_delta > 0 else ("v" if avg_delta < 0 else "=")
        print(
            f"  {'AVERAGE':<25} {'':<16} "
            f"{ma_avg:<7.2f} {bl_avg:<7.2f} {arrow}{sign}{avg_delta:<6.2f}"
        )
    else:
        print(f"  {'AVERAGE':<25} {'':<16} {ma_avg:<7.2f}")

    print("\n  By category:")
    print("  " + "-" * 50)
    for cat, data in summary.get("by_category", {}).items():
        ma = data["multi_agent_avg"]
        if run_baseline_ and data.get("baseline_avg") is not None:
            bl    = data["baseline_avg"]
            delta = data.get("delta") or 0
            sign  = "+" if delta >= 0 else ""
            print(f"  {cat:<18} multi={ma:.2f}  base={bl:.2f}  d={sign}{delta:.2f}")
        else:
            print(f"  {cat:<18} multi={ma:.2f}")

    print("=" * W + "\n")


# -- Load and display a previous run ---------------------------------------

def _resolve_result_path(path: str) -> str:
    """
    Three-strategy path resolution so users don't have to type the full path:
      1. Exact path as given
      2. Bare filename inside evaluation/results/
      3. Basename of any longer path inside evaluation/results/
    Prints available files and exits if nothing matches.
    """
    if os.path.isfile(path):
        return path

    candidate = os.path.join(_RESULTS_DIR, os.path.basename(path))
    if os.path.isfile(candidate):
        return candidate

    print(f"\n  File not found: {path!r}\n")
    _list_results()
    sys.exit(1)


def _list_results() -> None:
    """Print available result files, newest first, with a one-line summary each."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    files = sorted(
        [f for f in os.listdir(_RESULTS_DIR) if f.endswith(".json")],
        reverse=True,
    )
    if not files:
        print("  No evaluation results yet. Run one first:\n"
              "    uv run python evaluation/run_eval.py --id faiss_overview\n")
        return

    print(f"  Available results ({len(files)} files):")
    print("  " + "-" * 55)
    for fname in files:
        fpath = os.path.join(_RESULTS_DIR, fname)
        try:
            with open(fpath) as fp:
                d = json.load(fp)
            n_q  = len(d.get("results", []))
            ma   = d.get("summary", {}).get("multi_agent_avg", 0)
            bl   = d.get("summary", {}).get("baseline_avg")
            info = f"{n_q}q  multi={ma:.2f}"
            if bl is not None:
                info += f"  base={bl:.2f}"
        except Exception:
            info = f"{os.path.getsize(fpath) // 1024}KB"
        print(f"  {fname}  ({info})")

    print(
        f"\n  Load: uv run python evaluation/run_eval.py --load <filename>\n"
        f"  e.g.: uv run python evaluation/run_eval.py --load {files[0]}\n"
    )


def load_and_print(path: str) -> None:
    resolved = _resolve_result_path(path)
    with open(resolved) as f:
        data = json.load(f)
    print(f"\n  Loaded: {resolved}")
    _print_table(
        data["results"],
        data["summary"],
        data["summary"].get("ran_baseline", True),
    )


# -- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-agent research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python evaluation/run_eval.py\n"
            "  uv run python evaluation/run_eval.py --id faiss_overview\n"
            "  uv run python evaluation/run_eval.py --category ml_tools\n"
            "  uv run python evaluation/run_eval.py --no-baseline\n"
            "  uv run python evaluation/run_eval.py --list\n"
            "  uv run python evaluation/run_eval.py --load 2026-03-10T20-30-34.json\n"
        ),
    )
    parser.add_argument("--id",          help="Run single question by id")
    parser.add_argument("--category",    help="Run all questions in a category")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline (faster, only scores multi-agent)")
    parser.add_argument("--load",        metavar="FILE",
                        help="Load and display a previous results JSON "
                             "(filename or full path)")
    parser.add_argument("--list",        action="store_true",
                        help="List saved result files and exit")
    args = parser.parse_args()

    if args.list:
        _list_results()
        return

    if args.load:
        load_and_print(args.load)
        return

    run_eval(
        question_id=args.id,
        category=args.category,
        run_baseline_=not args.no_baseline,
    )


if __name__ == "__main__":
    main()
