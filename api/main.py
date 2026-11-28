"""REST API for the natural-language screener.

    uvicorn api.main:app --reload

Auth: `X-API-Key` header (keys in .env). LLM-backed endpoints are rate-limited per key.
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
import socket
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import urlparse

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from screener import config
from screener.alerts import evaluate_alert, run_all
from screener.dsl import Condition, PatchOp, Screen, apply_patch, describe
from screener.engine import Engine
from screener.llm import LLM
from screener.parser import ParseError, Parser
from screener.registry import CONCEPTS, FIELDS, UNIVERSES
from screener.store import Store

STATE: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = Engine()
    store = Store()
    llm = LLM(store, engine.categories()["sector"])
    STATE.update(engine=engine, store=store, parser=Parser(llm), llm=llm)
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(lambda: run_all(engine, store), "interval", minutes=config.ALERT_INTERVAL_MIN, id="alerts")
    sched.start()
    yield
    sched.shutdown(wait=False)


# No CORS middleware: the Streamlit UI calls this API from its Python server, not from the browser.
app = FastAPI(title="Scout API", version="1.0", lifespan=lifespan,
              description="Natural-language stock screening over NSE data with a validated query DSL.")


# ------------------------------------------------------------------ security
def owner(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key not in config.API_KEYS:
        raise HTTPException(401, "Invalid API key")
    return hashlib.sha256(x_api_key.encode()).hexdigest()[:16]


_hits: Dict[str, Deque[float]] = defaultdict(deque)


def rate_limited(who: str = Depends(owner)) -> str:
    now, q = time.time(), _hits[who]
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        raise HTTPException(429, f"Rate limit: {config.RATE_LIMIT_PER_MIN} natural-language queries per minute")
    q.append(now)
    return who


def safe_webhook(url: Optional[str]) -> Optional[str]:
    """SSRF protection: https only, and the host must resolve to public IPs."""
    if not url:
        return None
    u = urlparse(url)
    if u.scheme != "https" or not u.hostname:
        raise HTTPException(422, "Webhook must be an https URL")
    try:
        ips = {ai[4][0] for ai in socket.getaddrinfo(u.hostname, None)}
    except socket.gaierror:
        raise HTTPException(422, "Webhook host does not resolve")
    for ip in ips:
        a = ipaddress.ip_address(ip)
        if a.is_private or a.is_loopback or a.is_link_local or a.is_reserved or a.is_multicast:
            raise HTTPException(422, "Webhook must point to a public host")
    return url


# ------------------------------------------------------------------ schemas
class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=config.MAX_QUERY_CHARS)
    session_id: Optional[str] = Field(None, max_length=32)


class ScreenIn(BaseModel):
    screen: Screen
    session_id: Optional[str] = Field(None, max_length=32)


class SaveIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    screen: Screen
    query: str = Field("", max_length=config.MAX_QUERY_CHARS)


class AlertIn(BaseModel):
    screen_id: str = Field(..., max_length=32)
    webhook_url: Optional[str] = Field(None, max_length=300)


class BacktestIn(BaseModel):
    screen: Screen
    as_of: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    horizon_days: int = Field(90, ge=20, le=365)


class ExplainIn(BaseModel):
    screen: Screen
    symbol: str = Field(..., max_length=20, pattern=r"^[A-Za-z0-9&\-]+$")


def interpretation(screen: Screen) -> List[dict]:
    """What the system understood — rendered as editable chips in the UI."""
    out = []
    for i, c in enumerate(screen.all):
        leaves = getattr(c, "any", None) or [c]
        out.append({"index": i, "text": describe(c),
                    "assumed": any(l.assumed for l in leaves),
                    "source_phrase": next((l.source_phrase for l in leaves if l.source_phrase), None),
                    "rationale": next((l.rationale for l in leaves if l.rationale), None)})
    return out


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    e: Engine = STATE["engine"]
    return {"status": "ok", "data_as_of": str(e.as_of), "stocks": len(e.snap), "llm_enabled": STATE["llm"].enabled}


@app.get("/v1/meta")
def meta(_: str = Depends(owner)):
    e: Engine = STATE["engine"]
    return {
        "as_of": str(e.as_of), "universes": list(UNIVERSES),
        "fields": [{"key": f.key, "label": f.label, "kind": f.kind, "group": f.group, "unit": f.unit,
                    "description": f.desc} for f in FIELDS.values()],
        "concepts": [{"phrase": c.phrases[0], "means": [describe(Condition(**d)) for d in c.conditions],
                      "rationale": c.rationale} for c in CONCEPTS],
        "categories": e.categories(),
    }


WHY_RX = re.compile(r"\bwhy\b.*?\b(isn'?t|is not|not|is|did)\b", re.I)


@app.post("/v1/chat")
def chat(body: ChatIn, who: str = Depends(rate_limited)):
    engine: Engine = STATE["engine"]
    store: Store = STATE["store"]
    sess = store.get_session(body.session_id, who) if body.session_id else None
    current = Screen.model_validate(sess["screen"]) if sess and sess.get("screen") else None
    t0 = time.perf_counter()

    # Zero-token diagnostic: "why isn't TCS in the list?" → evidence per criterion
    if current is not None and WHY_RX.search(body.message):
        syms = [w.upper() for w in re.findall(r"[A-Za-z0-9&-]{2,}", body.message) if engine.stock(w.upper())]
        if syms:
            diag = engine.diagnose(current, syms[0]) or []
            failed = [d["text"] for d in diag if not d["passed"]]
            if failed:
                reply = f"{syms[0]} is excluded because it fails: " + "; ".join(failed)
            else:
                ranked = [r["symbol"] for r in engine.run(current.model_copy(update={"limit": 100}))["results"]]
                pos = ranked.index(syms[0]) + 1 if syms[0] in ranked else None
                reply = (f"{syms[0]} passes every criterion - it is #{pos} of the matches"
                         + (f", below your visible limit of {current.limit}." if pos and pos > current.limit else "."))
            store.log_query(who, body.message, "diagnose", (time.perf_counter() - t0) * 1000, 0, True)
            return {"session_id": body.session_id, "action": "diagnose", "reply": reply, "symbol": syms[0],
                    "diagnosis": diag, "screen": current.model_dump(), "path": "fastpath", "tokens": 0}

    try:
        pr = STATE["parser"].parse(body.message, current)
    except ParseError as e:
        store.log_query(who, body.message, "error", (time.perf_counter() - t0) * 1000, 0, False)
        raise HTTPException(422, str(e))
    result = engine.run(pr.screen) if pr.screen is not None else None
    kept = pr.screen or current  # clarify/unsupported replies keep the conversation's current screen
    sid = store.save_session(body.session_id, who, kept.model_dump() if kept else None)
    store.log_query(who, body.message, pr.path, pr.latency_ms, pr.tokens, True)
    return {
        "session_id": sid, "action": pr.action, "reply": pr.reply,
        "screen": pr.screen.model_dump() if pr.screen else None,
        "interpretation": interpretation(pr.screen) if pr.screen else [],
        "changes": pr.changes, "unmapped": pr.unmapped, "notes": pr.notes,
        "path": pr.path, "model": pr.model, "tokens": pr.tokens, "parse_ms": round(pr.latency_ms, 1),
        "result": result,
    }


@app.post("/v1/screen")
def run_screen(body: ScreenIn, who: str = Depends(owner)):
    """Deterministic execution of an explicit DSL (no LLM) — used when users edit chips."""
    store: Store = STATE["store"]
    if body.session_id:
        store.save_session(body.session_id, who, body.screen.model_dump())
    return {"screen": body.screen.model_dump(), "interpretation": interpretation(body.screen),
            "result": STATE["engine"].run(body.screen)}


class PatchIn(BaseModel):
    screen: Screen
    ops: List[PatchOp] = Field(..., max_length=10)
    session_id: Optional[str] = Field(None, max_length=32)


@app.post("/v1/screen/patch")
def patch_screen(body: PatchIn, who: str = Depends(owner)):
    """Apply patch ops (e.g. a one-click relaxation suggestion) and re-run, without the LLM."""
    try:
        new, changes = apply_patch(body.screen, body.ops)
    except (ValidationError, ValueError) as e:
        raise HTTPException(422, str(e))
    out = run_screen(ScreenIn(screen=new, session_id=body.session_id), who)
    return {**out, "changes": changes}


@app.post("/v1/explain")
def explain(body: ExplainIn, _: str = Depends(owner)):
    diag = STATE["engine"].diagnose(body.screen, body.symbol)
    if diag is None:
        raise HTTPException(404, "Unknown symbol")
    return {"symbol": body.symbol.upper(), "passes_all": all(d["passed"] for d in diag), "criteria": diag}


@app.get("/v1/stocks/{symbol}")
def stock(symbol: str, _: str = Depends(owner)):
    row = STATE["engine"].stock(symbol)
    if row is None:
        raise HTTPException(404, "Unknown symbol")
    return row


@app.post("/v1/screens")
def save_screen(body: SaveIn, who: str = Depends(owner)):
    sid = STATE["store"].save_screen(who, body.name, body.query, body.screen.model_dump())
    return {"id": sid}


@app.get("/v1/screens")
def list_screens(who: str = Depends(owner)):
    return [{**s, "description": [describe(c) for c in Screen.model_validate(s["screen"]).all]}
            for s in STATE["store"].list_screens(who)]


@app.delete("/v1/screens/{sid}")
def delete_screen(sid: str, who: str = Depends(owner)):
    if not STATE["store"].delete_screen(sid, who):
        raise HTTPException(404, "Not found")
    return {"deleted": sid}


@app.post("/v1/alerts")
def create_alert(body: AlertIn, who: str = Depends(owner)):
    store: Store = STATE["store"]
    engine: Engine = STATE["engine"]
    webhook = safe_webhook(body.webhook_url)  # validate before touching storage
    saved = store.get_screen(body.screen_id, who)
    if not saved:
        raise HTTPException(404, "Saved screen not found")
    members = engine.members(Screen.model_validate(saved["screen"]))
    aid = store.create_alert(who, body.screen_id, webhook, members, str(engine.as_of))
    return {"id": aid, "baseline_members": len(members), "as_of": str(engine.as_of)}


@app.get("/v1/alerts")
def list_alerts(who: str = Depends(owner)):
    return [{**{k: v for k, v in a.items() if k != "last_members"}, "members": len(a["last_members"])}
            for a in STATE["store"].list_alerts(who)]


@app.delete("/v1/alerts/{aid}")
def delete_alert(aid: str, who: str = Depends(owner)):
    if not STATE["store"].delete_alert(aid, who):
        raise HTTPException(404, "Not found")
    return {"deleted": aid}


@app.get("/v1/alerts/events")
def alert_events(who: str = Depends(owner), limit: int = 50):
    return STATE["store"].events(who, min(limit, 200))


@app.post("/v1/alerts/run")
def trigger_alerts(who: str = Depends(owner), force: bool = False):
    """Manually evaluate this user's alerts now (the scheduler also runs them periodically)."""
    engine: Engine = STATE["engine"]
    store: Store = STATE["store"]
    engine.reload()
    out = {a["id"]: evaluate_alert(engine, store, a, force=force) for a in store.list_alerts(who)}
    return {"as_of": str(engine.as_of), "events": out}


@app.post("/v1/backtest")
def backtest(body: BacktestIn, _: str = Depends(owner)):
    try:
        return STATE["engine"].backtest(body.screen, body.as_of, body.horizon_days)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/v1/stats")
def stats(_: str = Depends(owner)):
    return STATE["store"].stats()


@app.exception_handler(ValidationError)
async def _pydantic_error(_: Request, exc: ValidationError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
