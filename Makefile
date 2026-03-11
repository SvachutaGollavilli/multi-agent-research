# Makefile for multi-agent-research
# Usage: make <target>

.PHONY: help install run ui test evaluate clean all

# ── Default: show help ─────────────────────────────────────────
help:
	@echo ""
	@echo "  Multi-Agent Research System"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make install      Install all dependencies via uv"
	@echo "  make run          Run the pipeline from CLI (prompts for query)"
	@echo "  make ui           Launch the Streamlit UI"
	@echo "  make test         Run all tests (no API keys needed)"
	@echo "  make evaluate     Run single vs multi-agent evaluation (~15 min)"
	@echo "  make clean        Delete __pycache__, .pytest_cache, results/"
	@echo "  make all          install → test → run"
	@echo ""

# ── Install ────────────────────────────────────────────────────
install:
	uv sync

# ── Run pipeline (CLI) ─────────────────────────────────────────
run:
	uv run python main.py

# ── Streamlit UI ───────────────────────────────────────────────
ui:
	uv run streamlit run app.py

# ── Test suite (all 110+ tests, no API keys needed) ────────────
test:
	uv run pytest tests/ -v

# ── Run tests with coverage ────────────────────────────────────
test-cov:
	uv run pytest tests/ -v --tb=short

# ── Evaluation: single-agent vs multi-agent ────────────────────
evaluate:
	uv run python evaluation/run_eval.py

# ── Evaluate one question (fast sanity check) ──────────────────
eval-one:
	uv run python evaluation/run_eval.py --id faiss_overview --no-baseline

# ── Clean build artifacts ──────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f data/research_cache.db data/pipeline.db 2>/dev/null || true
	@echo "Clean complete."

# ── Full run: install → test → pipeline ───────────────────────
all: install test run
