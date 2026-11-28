"""SQLite persistence: chat sessions, saved screens, alerts, alert events, LLM cache, query telemetry."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from screener.config import APP_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, owner TEXT, screen TEXT, updated TEXT);
CREATE TABLE IF NOT EXISTS screens (id TEXT PRIMARY KEY, owner TEXT, name TEXT, query TEXT, screen TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS alerts (id TEXT PRIMARY KEY, owner TEXT, screen_id TEXT, webhook TEXT,
    last_members TEXT, last_run TEXT, last_as_of TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS alert_events (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT, ts TEXT, as_of TEXT,
    symbol TEXT, event TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response TEXT, model TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS query_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, owner TEXT, message TEXT,
    path TEXT, latency_ms REAL, tokens INTEGER, ok INTEGER);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path=APP_DB_PATH):
        self._con = sqlite3.connect(str(path), check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._con.executescript(SCHEMA)

    def _q(self, sql: str, args: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            rows = self._con.execute(sql, args).fetchall()  # fetch first: DELETE ... RETURNING is still open
            self._con.commit()
            return rows

    # sessions --------------------------------------------------------------
    def get_session(self, sid: str, owner: str) -> Optional[Dict[str, Any]]:
        r = self._q("SELECT * FROM sessions WHERE id=? AND owner=?", (sid, owner))
        if not r:
            return None
        return {"id": sid, "screen": json.loads(r[0]["screen"]) if r[0]["screen"] else None}

    def save_session(self, sid: Optional[str], owner: str, screen: Optional[dict]) -> str:
        """A session is just the conversation's current screen; follow-ups are patches applied to it."""
        sid = sid or uuid.uuid4().hex[:12]
        self._q("INSERT OR REPLACE INTO sessions (id, owner, screen, updated) VALUES (?,?,?,?)",
                (sid, owner, json.dumps(screen) if screen else None, now()))
        return sid

    # saved screens ----------------------------------------------------------
    def save_screen(self, owner: str, name: str, query: str, screen: dict) -> str:
        sid = uuid.uuid4().hex[:10]
        self._q("INSERT INTO screens VALUES (?,?,?,?,?,?)", (sid, owner, name, query, json.dumps(screen), now()))
        return sid

    def list_screens(self, owner: str) -> List[dict]:
        return [{**dict(r), "screen": json.loads(r["screen"])}
                for r in self._q("SELECT * FROM screens WHERE owner=? ORDER BY created DESC", (owner,))]

    def get_screen(self, sid: str, owner: Optional[str] = None) -> Optional[dict]:
        rows = (self._q("SELECT * FROM screens WHERE id=? AND owner=?", (sid, owner)) if owner
                else self._q("SELECT * FROM screens WHERE id=?", (sid,)))
        return {**dict(rows[0]), "screen": json.loads(rows[0]["screen"])} if rows else None

    def delete_screen(self, sid: str, owner: str) -> bool:
        self._q("DELETE FROM alerts WHERE screen_id=? AND owner=?", (sid, owner))
        return bool(self._q("DELETE FROM screens WHERE id=? AND owner=? RETURNING id", (sid, owner)))

    # alerts -------------------------------------------------------------------
    def create_alert(self, owner: str, screen_id: str, webhook: Optional[str], members: List[str], as_of: str) -> str:
        aid = uuid.uuid4().hex[:10]
        self._q("INSERT INTO alerts (id, owner, screen_id, webhook, last_members, last_run, last_as_of, created) "
                "VALUES (?,?,?,?,?,?,?,?)", (aid, owner, screen_id, webhook, json.dumps(members), now(), as_of, now()))
        return aid

    def list_alerts(self, owner: Optional[str] = None) -> List[dict]:
        sql = "SELECT a.*, s.name AS screen_name FROM alerts a JOIN screens s ON s.id=a.screen_id"
        rows = (self._q(sql + " WHERE a.owner=? ORDER BY a.created DESC", (owner,)) if owner else self._q(sql))
        return [{**dict(r), "last_members": json.loads(r["last_members"] or "[]")} for r in rows]

    def delete_alert(self, aid: str, owner: str) -> bool:
        return bool(self._q("DELETE FROM alerts WHERE id=? AND owner=? RETURNING id", (aid, owner)))

    def update_alert_state(self, aid: str, members: List[str], as_of: str) -> None:
        self._q("UPDATE alerts SET last_members=?, last_run=?, last_as_of=? WHERE id=?",
                (json.dumps(members), now(), as_of, aid))

    def add_event(self, aid: str, as_of: str, symbol: str, event: str, detail: str) -> None:
        self._q("INSERT INTO alert_events (alert_id, ts, as_of, symbol, event, detail) VALUES (?,?,?,?,?,?)",
                (aid, now(), as_of, symbol, event, detail))

    def events(self, owner: str, limit: int = 50) -> List[dict]:
        return [dict(r) for r in self._q(
            "SELECT e.*, s.name AS screen_name FROM alert_events e JOIN alerts a ON a.id=e.alert_id "
            "JOIN screens s ON s.id=a.screen_id WHERE a.owner=? ORDER BY e.id DESC LIMIT ?", (owner, limit))]

    # LLM cache + telemetry ----------------------------------------------------
    def cache_get(self, key: str) -> Optional[str]:
        r = self._q("SELECT response FROM llm_cache WHERE key=?", (key,))
        return r[0]["response"] if r else None

    def cache_put(self, key: str, response: str, model: str) -> None:
        self._q("INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?)", (key, response, model, now()))

    def log_query(self, owner: str, message: str, path: str, latency_ms: float, tokens: int, ok: bool) -> None:
        self._q("INSERT INTO query_log (ts, owner, message, path, latency_ms, tokens, ok) VALUES (?,?,?,?,?,?,?)",
                (now(), owner, message[:500], path, latency_ms, tokens, int(ok)))

    def usage_today(self) -> Dict[str, int]:
        r = self._q("SELECT COUNT(*) FILTER (WHERE path='llm') llm, COUNT(*) FILTER (WHERE path='cache') cache, "
                    "COUNT(*) FILTER (WHERE path IN ('fastpath','diagnose')) rules, COALESCE(SUM(tokens),0) tok "
                    "FROM query_log WHERE substr(ts,1,10)=?", (now()[:10],))[0]
        return {"llm_queries": r["llm"], "cached": r["cache"], "rule_based": r["rules"], "tokens": r["tok"]}

    def stats(self) -> Dict[str, Any]:
        rows = self._q("SELECT path, COUNT(*) n, AVG(latency_ms) lat, SUM(tokens) tok FROM query_log GROUP BY path")
        return {r["path"]: {"queries": r["n"], "avg_latency_ms": round(r["lat"] or 0, 1), "tokens": r["tok"] or 0}
                for r in rows}
