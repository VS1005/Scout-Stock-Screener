"""Central configuration. Everything tunable lives here or in .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = Path(os.getenv("SCOUT_DB_PATH", DATA_DIR / "market.duckdb"))  # analytical store (read-only at query time)
APP_DB_PATH = DATA_DIR / "app.sqlite"     # OLTP store: sessions, saved screens, alerts, LLM cache

# LLM
LLM_API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip("\"'").strip()  # tolerate pasted quotes/spaces
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b").split(",") if m.strip()]
PROMPT_GUARD_MODEL = os.getenv("PROMPT_GUARD_MODEL", "meta-llama/llama-prompt-guard-2-22m")
PROMPT_GUARD_ENABLED = os.getenv("PROMPT_GUARD_ENABLED", "true").lower() == "true"

# API
API_KEYS = {k.strip() for k in os.getenv("SCREENER_API_KEYS", "demo-key-123").split(",") if k.strip()}
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Alerts
ALERT_INTERVAL_MIN = int(os.getenv("ALERT_INTERVAL_MIN", "60"))
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")

# Query-engine guard rails
MAX_CONDITIONS = 12
MAX_ANY_GROUP_SIZE = 6
MAX_LIMIT = 100
MAX_QUERY_CHARS = 500
QUERY_TIMEOUT_S = 5

# SEBI LODR Reg. 33: quarterly results within 45 days of quarter end, annual within 60 days
# of FY end. Used to make fundamentals point-in-time (no look-ahead in backtests).
QUARTERLY_REPORT_LAG_DAYS = 45
ANNUAL_REPORT_LAG_DAYS = 60

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "*/*",
}

for _d in (DATA_DIR, RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)
