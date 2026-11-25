"""Alerts = saved screens re-evaluated on a schedule; we alert on *membership changes*.

A stock ENTERING a screen gets the same evidence-based explanation as a search result; a stock
EXITING names the criterion it now fails ("D/E rose to 0.62, above 0.5"). Delivery: stored in an
inbox (API/UI) and optionally POSTed to a webhook (Slack/Discord/Teams compatible JSON).
"""
from __future__ import annotations

from typing import Dict, List

import requests

from screener.config import ALERT_WEBHOOK_URL
from screener.dsl import Screen
from screener.engine import Engine
from screener.store import Store


def _explain_exit(engine: Engine, screen: Screen, symbol: str) -> str:
    diag = engine.diagnose(screen, symbol)
    if diag is None:
        return "no longer in the data universe"
    failed = [d["text"] for d in diag if not d["passed"]]
    return "now fails: " + "; ".join(failed) if failed else "now fails one or more criteria"


def evaluate_alert(engine: Engine, store: Store, alert: dict, force: bool = False) -> List[Dict]:
    saved = store.get_screen(alert["screen_id"])
    if not saved:
        return []
    screen = Screen.model_validate(saved["screen"])
    as_of = str(engine.as_of)
    if not force and alert.get("last_as_of") == as_of:
        return []  # data hasn't changed since the last evaluation
    now, before = set(engine.members(screen)), set(alert["last_members"])
    events = []
    entered = sorted(now - before)
    if entered:
        res = engine.run(screen.model_copy(update={"limit": 100}))
        by_sym = {r["symbol"]: r for r in res["results"]}
    for sym in entered:
        reasons = ("; ".join(x["text"] for x in by_sym[sym]["reasons"] if x["passed"])
                   if sym in by_sym else "now meets every criterion")
        events.append({"symbol": sym, "event": "entered", "detail": reasons})
    for sym in sorted(before - now):
        events.append({"symbol": sym, "event": "exited", "detail": _explain_exit(engine, screen, sym)})
    for ev in events:
        store.add_event(alert["id"], as_of, ev["symbol"], ev["event"], ev["detail"])
    store.update_alert_state(alert["id"], sorted(now), as_of)
    hook = alert.get("webhook") or ALERT_WEBHOOK_URL
    if events and hook:
        text = f"*{saved['name']}* ({as_of}): " + ", ".join(f"{e['symbol']} {e['event']}" for e in events[:20])
        try:
            requests.post(hook, json={"text": text, "content": text, "events": events}, timeout=5)
        except requests.RequestException:
            pass  # delivery is best-effort; events remain in the inbox
    return events


def run_all(engine: Engine, store: Store, force: bool = False) -> Dict[str, int]:
    engine.reload()
    out = {}
    for a in store.list_alerts():
        out[a["id"]] = len(evaluate_alert(engine, store, a, force=force))
    return out
