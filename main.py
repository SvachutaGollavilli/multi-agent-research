#!/usr/bin/env python
# main.py -- CLI entry point
# Run the pipeline from the terminal without the Streamlit UI.
#
# Usage:
#   uv run python main.py
#   uv run python main.py "What is FAISS and how does it work?"

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    from src.agents.graph import run_pipeline

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("\nResearch question: ").strip()

    if not query:
        print("No query provided. Exiting.")
        sys.exit(1)

    print(f"\nRunning pipeline for: '{query}'\n")
    result = run_pipeline(query)

    report = result.get("final_report") or result.get("current_draft", "")
    if report:
        print("\n" + "=" * 70)
        print(report)
        print("=" * 70)

    print(
        f"\nDone. "
        f"Tokens: {result.get('token_count', 0):,}  |  "
        f"Cost: ${result.get('cost_usd', 0):.5f}"
    )


if __name__ == "__main__":
    main()
