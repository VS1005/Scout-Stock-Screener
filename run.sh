#!/usr/bin/env bash
# One-command local demo: API on :8000, UI on :8501.
set -euo pipefail
cd "$(dirname "$0")"
[ -d .venv ] || { python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
[ -f data/market.duckdb ] || .venv/bin/python -m etl.run          # first run: ~5 min data build
export PYTHONWARNINGS=ignore
.venv/bin/uvicorn api.main:app --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
.venv/bin/streamlit run ui/app.py --server.port 8501
