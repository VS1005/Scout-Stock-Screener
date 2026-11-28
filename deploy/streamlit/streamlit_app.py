"""Streamlit Community Cloud entrypoint.

Community Cloud runs a single Streamlit process, so the FastAPI app is started once in a background
thread of that process, on a localhost port, and the normal UI (ui/app.py) talks to it as usual.
Data is the prebuilt snapshot shipped next to this file (the ETL is not run in the cloud).

Deploy: share.streamlit.io → repo VS1005/Scout-Stock-Screener, branch main,
        main file deploy/streamlit/streamlit_app.py, secret GROQ_API_KEY = "...".
"""
import os
import runpy
import socket
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
try:  # Streamlit secrets → env, where screener.config reads them
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:  # noqa: BLE001 - no secrets file (local run): fall back to .env / rule-based parser
    pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@st.cache_resource(show_spinner=False)
def start_api() -> dict:
    """Start the API once per process on a free localhost port; report why if it can't start."""
    state = {"port": _free_port(), "error": None}

    def serve():
        try:
            import uvicorn

            from api.main import app

            uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=state["port"], log_level="warning")).run()
        except BaseException as e:  # noqa: BLE001 - surfaced on the page below
            state["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=serve, daemon=True).start()
    for _ in range(120):  # the API loads the snapshot on startup
        if state["error"]:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{state['port']}/health", timeout=1)
            break
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return state


api_state = start_api()
os.environ["API_URL"] = f"http://127.0.0.1:{api_state['port']}"
if api_state["error"]:
    st.error(f"The screener API failed to start: {api_state['error']}")
runpy.run_path(str(ROOT / "ui" / "app.py"), run_name="__main__")
