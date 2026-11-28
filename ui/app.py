"""Conversational stock-discovery UI. A pure client of the REST API (no direct DB/LLM access).

    streamlit run ui/app.py
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API = os.getenv("API_URL", "http://127.0.0.1:8000")
KEY = os.getenv("SCREENER_UI_KEY", (os.getenv("SCREENER_API_KEYS", "demo-key-123").split(",")[0]).strip())

st.set_page_config(page_title="Scout · stock screener", page_icon="🔎", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, .stTabs button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero h1 { font-size: 2.1rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg, #E7EAF0 0%, #A5A9FF 60%, #7CE0C3 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #9AA3B5; margin: .35rem 0 1.2rem 0; font-size: .98rem; }
.brand { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; margin-top: .4rem; }
.muted { color: #8A93A6; font-size: .82rem; }
.pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .76rem; font-weight: 500;
  border: 1px solid rgba(255,255,255,.10); background: rgba(255,255,255,.04); color: #C9CEDA; }
.pill.llm { background: rgba(124,131,253,.14); border-color: rgba(124,131,253,.45); color: #C4C7FF; }
.pill.rules { background: rgba(124,224,195,.12); border-color: rgba(124,224,195,.40); color: #9FEBD3; }
.pill.warn { background: rgba(245,158,11,.14); border-color: rgba(245,158,11,.45); color: #F8C46B; }
.chip { display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: .72rem; margin-left: 8px; }
.assumed { background: rgba(245,158,11,.14); color: #F8C46B; border: 1px solid rgba(245,158,11,.40); }
.explicit { background: rgba(124,224,195,.12); color: #9FEBD3; border: 1px solid rgba(124,224,195,.35); }
.cond { font-weight: 600; font-size: .95rem; }
.why { color: #8A93A6; font-size: .8rem; margin-top: 2px; }
.quota-num { font-size: 1.5rem; font-weight: 700; line-height: 1.1; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,.06); }
[data-testid="stSidebar"] .stButton button { text-align: left; justify-content: flex-start; font-size: .85rem; }
[data-testid="stSidebar"] .stButton button[kind="primary"] { justify-content: center; font-weight: 600;
  font-size: .95rem; padding: .6rem 0; box-shadow: 0 4px 18px rgba(124,131,253,.35); }
[data-testid="stMetricValue"] { font-size: 1.55rem; }
[data-testid="stExpander"] details { border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { padding: 6px 14px; border-radius: 8px 8px 0 0; }
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API helpers
def api(method: str, path: str, quiet: bool = False, **kw):
    """Call the API; on failure remember the reason (so it survives a rerun) and show it unless quiet."""
    try:
        r = httpx.request(method, f"{API}{path}", headers={"X-API-Key": KEY}, timeout=60, **kw)
    except httpx.HTTPError as e:
        st.session_state["api_error"] = f"API unreachable at {API}: {e}"
        if not quiet:
            st.error(st.session_state["api_error"])
        return None
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = r.text
        if isinstance(detail, list):
            detail = "; ".join(d.get("msg", str(d)) for d in detail[:3])
        st.session_state["api_error"] = str(detail)
        if not quiet:
            st.error(detail)
        return None
    return r.json()


def meta() -> dict:
    """Field labels/units and the glossary; fetched once per session (only kept once it succeeds)."""
    if not st.session_state.get("meta"):
        st.session_state["meta"] = api("GET", "/v1/meta", quiet=True) or {}
    return st.session_state["meta"]


ss = st.session_state
ss.setdefault("session_id", None)
ss.setdefault("turns", [])        # [{"user": str, "resp": dict}]
ss.setdefault("current", None)    # latest response with a result
ss.setdefault("pending", None)
ss.setdefault("mode", "Always AI")

EXAMPLES = [
    "Find profitable companies with low debt and increasing revenue",
    "Undervalued banks with ROE above 15",
    "Quality mid caps in an uptrend",
    "IT or pharma stocks near 52 week high with rising delivery",
    "Companies that make money, don't borrow much and keep growing sales",
    "Small caps with high growth and positive free cash flow, not in financial services",
]
MODES = {"Always AI": "llm", "Smart": "auto"}


def send(message: str):
    ss.pop("api_error", None)
    resp = api("POST", "/v1/chat", json={"message": message, "session_id": ss.session_id,
                                          "mode": MODES.get(ss.mode, "llm")})
    if resp is None:  # keep the question and the reason visible after the rerun
        ss.turns.append({"user": message, "resp": {"error": ss.get("api_error", "Request failed")}})
        return
    ss.session_id = resp.get("session_id") or ss.session_id
    ss.turns.append({"user": message, "resp": resp})
    if resp.get("result"):
        ss.current = resp


def rerun_screen(screen: dict, note: str):
    resp = api("POST", "/v1/screen", json={"screen": screen, "session_id": ss.session_id})
    if resp:
        resp.update(path="edit", tokens=0, changes=[note], unmapped=[], notes=[], reply="", action="patch")
        ss.turns.append({"user": f"✎ {note}", "resp": resp})
        ss.current = resp


def apply_ops(ops: list, note: str):
    resp = api("POST", "/v1/screen/patch", json={"screen": ss.current["screen"], "ops": ops,
                                                  "session_id": ss.session_id})
    if resp:
        resp.update(path="edit", tokens=0, unmapped=[], notes=[], reply="", action="patch")
        ss.turns.append({"user": f"✎ {note}", "resp": resp})
        ss.current = resp


# ------------------------------------------------------------------ sidebar
health = api("GET", "/health", quiet=True) or {}


def quota_card():
    q = api("GET", "/v1/quota", quiet=True)
    st.markdown("**LLM budget** <span class='muted'>· Groq free tier</span>", unsafe_allow_html=True)
    if not q or not q.get("enabled"):
        st.caption("No LLM key configured: rule-based parsing only.")
        return
    if q.get("key_error"):
        st.error(q["key_error"])
        return
    models = q.get("models") or {}
    main = next((v for k, v in models.items() if "guard" not in k), None)
    with st.container(border=True):
        if main and main.get("requests_per_day"):
            left, cap = main["requests_left_today"], main["requests_per_day"]
            st.markdown(f"<div class='quota-num'>{left:,}<span class='muted'> / {cap:,}</span></div>"
                        f"<div class='muted'>LLM requests left today</div>", unsafe_allow_html=True)
            st.progress(max(0.0, min(1.0, left / cap)))
            tl, tc = main.get("tokens_left_this_minute"), main.get("tokens_per_minute")
            if tl is not None and tc:
                st.caption(f"{tl:,} / {tc:,} tokens left this minute (≈ {max(tl, 0) // 2500} queries)")
        else:
            st.caption("Quota appears after the first LLM call.")
        t = q.get("today") or {}
        st.caption(f"Today on this server: {t.get('llm_queries', 0)} LLM · {t.get('cached', 0)} cached · "
                   f"{t.get('rule_based', 0)} rule-based · {t.get('tokens', 0):,} tokens")


with st.sidebar:
    if st.button("＋  New conversation", type="primary", width="stretch"):
        ss.session_id, ss.turns, ss.current = None, [], None
        st.rerun()
    st.markdown("<div class='brand'>🔎 Scout</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Natural-language screener · NSE Nifty 500<br>"
                f"Data as of <b>{health.get('data_as_of', '?')}</b> · {health.get('stocks', '?')} stocks</div>",
                unsafe_allow_html=True)
    st.write("")
    st.radio("Understanding", list(MODES), key="mode", horizontal=True)
    st.caption("**Always AI**: every query is read by the LLM (rules as fallback).  \n"
               "**Smart**: rules answer common phrasings for free; the LLM handles the rest.")
    quota_card()
    st.markdown("**Try asking**")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}", width="stretch"):
            ss.pending = ex
    stats = api("GET", "/v1/stats", quiet=True) or {}
    if stats:
        with st.expander("Query routing on this server"):
            st.dataframe(pd.DataFrame(stats).T.rename(columns={"queries": "Queries", "avg_latency_ms": "Avg ms",
                                                               "tokens": "Tokens"}), width="stretch")

st.markdown("<div class='hero'><h1>Find stocks by describing them</h1>"
            "<p>Plain English in, a validated screen out, with the evidence for every match. "
            "Follow up with “only banks”, “drop the debt filter”, “sort by ROE” or “why isn't TCS here?”</p></div>",
            unsafe_allow_html=True)
tab_chat, tab_saved, tab_bt, tab_gloss = st.tabs(["💬  Screener", "🔔  Saved & alerts", "📈  Backtest",
                                                   "📖  Definitions"])


# ------------------------------------------------------------------ rendering helpers
def path_badge(r: dict) -> str:
    p = r.get("path")
    if p == "llm":
        return f"<span class='pill llm'>🧠 LLM · {r.get('model')} · {r.get('tokens', 0):,} tokens · {r.get('parse_ms', 0):.0f} ms</span>"
    if p == "cache":
        return "<span class='pill llm'>🧠 LLM answer from cache · 0 tokens</span>"
    if p == "fastpath":
        return f"<span class='pill rules'>⚡ rule-based parser · 0 tokens · {r.get('parse_ms', 0):.0f} ms</span>"
    if p == "degraded":
        return "<span class='pill warn'>⚠ LLM unavailable · partial rule-based parse</span>"
    if p == "edit":
        return "<span class='pill'>✎ direct edit · no LLM</span>"
    return f"<span class='pill'>{p or ''}</span>"


def _fmt_col(key: str, f: dict):
    label = f.get("label", key)
    unit = f.get("unit", "")
    if f.get("kind") == "bool":
        return st.column_config.CheckboxColumn(label)
    if f.get("kind") == "category":
        return st.column_config.TextColumn(label)
    fmt = {"%": "%.1f%%", "pp": "%.1f pp", "x": "%.2fx", "yrs": "%d yrs", "₹": "₹%.2f"}.get(unit)
    if unit == "₹ Cr":
        return st.column_config.NumberColumn(f"{label} (₹ Cr)", format="localized")
    return st.column_config.NumberColumn(label, format=fmt or "%.2f")


def render_chips(resp: dict):
    res, screen = resp["result"], resp["screen"]
    st.markdown("##### What I understood")
    st.caption("Edit or remove any condition: it re-runs instantly without the LLM.")
    stats = {c["id"]: c for c in res["criteria"]}
    for item in resp["interpretation"]:
        i = item["index"]
        crit = stats.get(f"__c{i}", {})
        with st.container(border=True):
            c1, c2, c3 = st.columns([6, 3, 1], vertical_alignment="center")
            tag = (f"<span class='chip assumed'>assumed from “{item['source_phrase']}”</span>"
                   if item["assumed"] and item.get("source_phrase") else "<span class='chip explicit'>as you said</span>")
            why = f"<div class='why'>{item['rationale']}</div>" if item.get("rationale") and item["assumed"] else ""
            c1.markdown(f"<span class='cond'>{item['text']}</span>{tag}{why}", unsafe_allow_html=True)
            c2.progress(min(crit.get("pass_rate", 0) / 100, 1.0),
                        text=f"{crit.get('pass_count', 0)} of {res['universe_size']} stocks pass")
            cond = screen["all"][i]
            with c3:
                if isinstance(cond, dict) and "field" in cond and isinstance(cond.get("value"), (int, float)) \
                        and not isinstance(cond.get("value"), bool):
                    with st.popover("✎"):
                        v = st.number_input("Threshold", value=float(cond["value"]), key=f"edit_{i}_{len(ss.turns)}")
                        if st.button("Apply", key=f"apply_{i}_{len(ss.turns)}", type="primary"):
                            new = {**screen, "all": [dict(c) for c in screen["all"]]}
                            new["all"][i] = {**cond, "value": v, "assumed": False}
                            rerun_screen(new, f"{item['text']} → {v:g}")
                            st.rerun()
                if st.button("✕", key=f"rm_{i}_{len(ss.turns)}", help="Remove this condition"):
                    new = {**screen, "all": [c for j, c in enumerate(screen["all"]) if j != i]}
                    rerun_screen(new, f"removed: {item['text']}")
                    st.rerun()


def results_table(res: dict):
    fields = {f["key"]: f for f in meta().get("fields", [])}
    base = {"symbol", "name", "sector", "industry", "cap_category", "market_cap_cr", "close"}
    shown = [k for k in res["results"][0]["values"] if k not in base]
    rows = [{"Symbol": r["symbol"], "Company": r["name"], "Sector": r["sector"], "Cap": r.get("cap_category"),
             "Strength": r["match_strength"], "Mkt cap": r["market_cap_cr"],
             **{k: r["values"].get(k) for k in shown},
             "Borderline on": ", ".join(r["borderline"])} for r in res["results"]]
    cfg = {"Strength": st.column_config.ProgressColumn(
               "Match strength", min_value=0, max_value=100, format="%d",
               help="How comfortably the stock clears your conditions (margin of safety, peer-normalised)"),
           "Mkt cap": st.column_config.NumberColumn("Mkt cap (₹ Cr)", format="localized")}
    cfg.update({k: _fmt_col(k, fields.get(k, {})) for k in shown})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=cfg)


def render_result(resp: dict):
    res = resp["result"]
    st.markdown(path_badge(resp), unsafe_allow_html=True)
    if resp.get("reply"):
        st.markdown(resp["reply"])
    for ch in resp.get("changes") or []:
        st.markdown(f"↳ _{ch}_")
    if resp.get("unmapped"):
        st.warning("Couldn't map these to available data, so they were **not** applied: "
                   + ", ".join(f"“{u}”" for u in resp["unmapped"]))
    for n in (resp.get("notes") or []) + (res.get("warnings") or []):
        st.info(n)
    render_chips(resp)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matches", f"{res['total_matches']}", help="stocks passing every condition")
    m2.metric("Universe", f"{res['universe']} · {res['universe_size']}")
    m3.metric("Data as of", res["as_of"])
    m4.metric("Query time", f"{res['elapsed_ms']:.0f} ms")

    if res.get("suggestions"):
        with st.container(border=True):
            st.markdown("**Too few matches?** The single change that unlocks the most stocks:")
            cols = st.columns(len(res["suggestions"]))
            for k, (s, col) in enumerate(zip(res["suggestions"], cols)):
                label = (f"Relax to {s['relax_to']} → {s['matches_if_relaxed']} stocks" if s.get("relax_to")
                         else f"Drop “{s['criterion']}” → {s['matches_if_removed']} stocks")
                if col.button(label, key=f"sugg_{k}_{len(ss.turns)}", width="stretch"):
                    apply_ops(s["patch"], label)
                    st.rerun()

    n_near = len(res.get("near_misses") or [])
    t_res, t_why, t_near, t_save, t_sql = st.tabs([f"Results ({min(res['total_matches'], len(res['results']))})",
                                                   "Why they match", f"Near misses ({n_near})",
                                                   "Save & alert", "Under the hood"])
    with t_res:
        if res["results"]:
            results_table(res)
            if res["total_matches"] > len(res["results"]):
                st.caption(f"Showing the top {len(res['results'])} of {res['total_matches']} by match strength. "
                           f"Ask for “top 50” to see more.")
        else:
            st.error("No stock passes every condition. Try one of the suggestions above.")
    with t_why:
        if not res["results"]:
            st.caption("Nothing matched.")
        for r in res["results"][:15]:
            with st.expander(f"**{r['symbol']}** · {r['name']} · strength {r['match_strength']:.0f}"):
                for x in r["reasons"]:
                    st.markdown(f"{'✅' if x['passed'] else '❌'} {x['text']}")
                (st.warning if r["borderline"] else st.success)(r["summary"])
    with t_near:
        if n_near:
            st.caption("Stocks that failed exactly one numeric condition, closest first.")
            st.dataframe(pd.DataFrame([{"Symbol": n["symbol"], "Company": n["name"], "Missed on": n["missed"]}
                                       for n in res["near_misses"]]), hide_index=True, width="stretch")
        else:
            st.caption("No near misses for this screen.")
    with t_save:
        name = st.text_input("Name", value=(ss.turns[-1]["user"][:60] if ss.turns else "My screen"),
                             key=f"name_{len(ss.turns)}")
        hook = st.text_input("Webhook URL (optional · https · Slack/Discord/Teams)", key=f"hook_{len(ss.turns)}")
        alert = st.toggle("Alert me when stocks enter or leave this screen", value=True, key=f"al_{len(ss.turns)}")
        if st.button("Save screen", key=f"save_{len(ss.turns)}", type="primary"):
            saved = api("POST", "/v1/screens", json={"name": name, "screen": resp["screen"],
                                                     "query": ss.turns[-1]["user"] if ss.turns else ""})
            if saved and alert:
                a = api("POST", "/v1/alerts", json={"screen_id": saved["id"], "webhook_url": hook or None})
                if a:
                    st.success(f"Saved. Alert is live (baseline {a['baseline_members']} stocks as of {a['as_of']}).")
            elif saved:
                st.success("Saved.")
    with t_sql:
        st.caption("The LLM only ever produces the JSON on the left. It is validated against a whitelist, then "
                   "compiled to parameterised SQL that runs in a sandboxed in-memory database.")
        a, b = st.columns(2)
        a.json(resp["screen"])
        b.code(res["sql"], language="sql")


# ------------------------------------------------------------------ chat tab
with tab_chat:
    for t in ss.turns[:-1]:
        with st.chat_message("user"):
            st.markdown(t["user"])
        with st.chat_message("assistant"):
            r = t["resp"]
            if r.get("error"):
                st.error(r["error"])
                continue
            txt = r.get("reply") or "; ".join(r.get("changes") or []) or ""
            if r.get("result"):
                txt += f"  \n→ **{r['result']['total_matches']}** matches"
            st.markdown(path_badge(r), unsafe_allow_html=True)
            st.markdown(txt or "…")
    if ss.turns:
        last = ss.turns[-1]
        with st.chat_message("user"):
            st.markdown(last["user"])
        with st.chat_message("assistant"):
            r = last["resp"]
            if r.get("error"):
                st.error(r["error"])
            elif r.get("action") == "diagnose":
                st.markdown("<span class='pill rules'>🩺 diagnosis · 0 tokens</span>", unsafe_allow_html=True)
                st.markdown(r["reply"])
                for d in r.get("diagnosis", []):
                    st.markdown(f"{'✅' if d['passed'] else '❌'} {d['text']}")
            elif r.get("result"):
                render_result(r)
            else:
                st.markdown(path_badge(r), unsafe_allow_html=True)
                st.markdown(r.get("reply") or "I couldn't turn that into a stock screen. Try describing "
                                                "the kind of companies you want.")
                if r.get("unmapped"):
                    st.warning("Not available in the data: " + ", ".join(r["unmapped"]))
    q = st.chat_input("Describe the stocks you want, e.g. profitable companies with low debt and increasing revenue")
    if q or ss.pending:
        msg = q or ss.pending
        ss.pending = None
        with st.spinner("Reading your request…"):
            send(msg)
        st.rerun()

# ------------------------------------------------------------------ saved & alerts
with tab_saved:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("#### Saved screens")
        screens = api("GET", "/v1/screens", quiet=True) or []
        if not screens:
            st.caption("Nothing saved yet. Run a screen, then use **Save & alert** under its results.")
        for s in screens:
            with st.container(border=True):
                st.markdown(f"**{s['name']}** <span class='muted'>· saved {s['created'][:10]}</span>",
                            unsafe_allow_html=True)
                st.caption(" AND ".join(s["description"]) or "no filters")
                b1, b2, b3 = st.columns(3)
                if b1.button("Open", key=f"open_{s['id']}", width="stretch"):
                    rerun_screen(s["screen"], f"opened saved screen “{s['name']}”")
                    st.rerun()
                if b2.button("Alert me", key=f"alert_{s['id']}", width="stretch"):
                    a = api("POST", "/v1/alerts", json={"screen_id": s["id"]})
                    if a:
                        st.success(f"Alert created (baseline {a['baseline_members']} stocks)")
                if b3.button("Delete", key=f"del_{s['id']}", width="stretch"):
                    api("DELETE", f"/v1/screens/{s['id']}")
                    st.rerun()
    with c2:
        st.markdown("#### Alerts")
        alerts = api("GET", "/v1/alerts", quiet=True) or []
        if not alerts:
            st.caption("No alerts yet.")
        for a in alerts:
            st.markdown(f"🔔 **{a['screen_name']}** <span class='muted'>· {a['members']} stocks · "
                        f"checked on data of {a['last_as_of']}</span>", unsafe_allow_html=True)
        if alerts and st.button("Check alerts now", help="The scheduler also runs this automatically"):
            out = api("POST", "/v1/alerts/run", params={"force": True})
            if out:
                n = sum(len(v) for v in out["events"].values())
                st.success(f"Evaluated on data as of {out['as_of']}: {n} membership change(s).")
        st.markdown("#### Inbox")
        events = api("GET", "/v1/alerts/events", quiet=True) or []
        if not events:
            st.caption("Alerts fire when stocks enter or leave a saved screen after a data refresh.")
        for e in events:
            icon = "🟢 entered" if e["event"] == "entered" else "🔴 left"
            st.markdown(f"{icon} **{e['symbol']}** <span class='muted'>· {e['screen_name']} · data {e['as_of']}</span>",
                        unsafe_allow_html=True)
            st.caption(e["detail"])

# ------------------------------------------------------------------ backtest
with tab_bt:
    st.markdown("#### How would the current screen have done?")
    st.caption("Point-in-time: the screen is re-run as of the chosen date using only prices up to that day and "
               "financial results whose SEBI filing deadline had passed (45 days after quarter end, 60 after FY end).")
    if not ss.current:
        st.info("Run a screen in the Screener tab first.")
    else:
        with st.container(border=True):
            st.markdown("**Screen:** " + " AND ".join(i["text"] for i in ss.current["interpretation"]))
            c1, c2 = st.columns(2)
            try:  # anchor on the data's date, not the clock: a deployed snapshot may be weeks old
                anchor = date.fromisoformat(health["data_as_of"])
            except (KeyError, TypeError, ValueError):
                anchor = date.today()
            as_of = c1.date_input("Screen as of", value=anchor - timedelta(days=200),
                                  min_value=anchor - timedelta(days=560), max_value=anchor - timedelta(days=30))
            horizon = c2.slider("Hold for (days)", 30, 365, 180, step=15)
            run_bt = st.button("Run backtest", type="primary")
        if run_bt:
            bt = api("POST", "/v1/backtest", json={"screen": ss.current["screen"], "as_of": str(as_of),
                                                   "horizon_days": horizon})
            if bt:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Screen (equal-weight)", f"{bt['screen_return']}%" if bt["screen_return"] is not None else "-",
                          delta=f"{(bt['screen_return'] or 0) - bt['nifty50_return']:+.1f} pp vs Nifty 50")
                k2.metric("Nifty 50", f"{bt['nifty50_return']}%")
                k3.metric("Nifty 500 stocks (equal-weight)", f"{bt['universe_ew_return']}%")
                k4.metric("Members beating Nifty", f"{bt['hit_rate_vs_nifty']}%" if bt["hit_rate_vs_nifty"] is not None else "-")
                st.caption(f"{bt['n_members']} stocks qualified on {bt['as_of']}, held to {bt['end']} ({bt['horizon_days']} days)")
                st.bar_chart(pd.DataFrame({"Return %": [bt["screen_return"] or 0, bt["nifty50_return"], bt["universe_ew_return"]]},
                                          index=["Screen", "Nifty 50", "Universe EW"]))
                b, w = st.columns(2)
                b.markdown("**Best**  \n" + "  \n".join(f"{x['symbol']}: {x['return']:+.1f}%" for x in bt["best"]))
                w.markdown("**Worst**  \n" + "  \n".join(f"{x['symbol']}: {x['return']:+.1f}%" for x in bt["worst"]))
                st.markdown("**Caveats**  \n" + "  \n".join(f"- {c}" for c in bt["caveats"]))

# ------------------------------------------------------------------ definitions
with tab_gloss:
    m = meta()
    st.markdown("#### Vague words have explicit definitions")
    st.caption("The rule-based parser and the LLM both use this table, so “low debt” always means the same thing. "
               "Anything assumed shows as an amber chip you can edit.")
    st.dataframe(pd.DataFrame([{"You say": c["phrase"], "Scout applies": " AND ".join(c["means"]), "Why": c["rationale"]}
                               for c in m.get("concepts", [])]), hide_index=True, width="stretch")
    st.markdown("#### Every screenable field")
    st.dataframe(pd.DataFrame([{"Field": f["label"], "Group": f["group"], "Unit": f["unit"], "Meaning": f["description"]}
                               for f in m.get("fields", [])]), hide_index=True, width="stretch")
