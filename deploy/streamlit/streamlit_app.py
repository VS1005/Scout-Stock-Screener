"""Streamlit Community Cloud entrypoint.

Community Cloud runs a single Streamlit process, so the FastAPI app is started once in a background
thread of that process, on a localhost port, and the normal UI (ui/app.py) talks to it as usual.
Data is the prebuilt snapshot shipped next to this file (the ETL is not run in the cloud).

Deploy: share.streamlit.io → repo VS1005/Scout-Stock-Screener, branch main,
        main file deploy/streamlit/streamlit_app.py, secret GROQ_API_KEY = "...".
"""
import os
import runpy
import sys
import threading
import time
import urllib.request
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCOUT_DB_PATH", str(HERE / "data" / "market.duckdb"))
os.environ.setdefault("API_URL", "http://127.0.0.1:8000")
try:  # Streamlit secrets → env, where screener.config reads them
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:  # noqa: BLE001 - no secrets file (local run): fall back to .env / rule-based parser
    pass


@st.cache_resource(show_spinner=False)
def start_api() -> bool:
    import uvicorn

    from api.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(60):  # the API loads the snapshot on startup
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


start_api()
runpy.run_path(str(ROOT / "ui" / "app.py"), run_name="__main__")
