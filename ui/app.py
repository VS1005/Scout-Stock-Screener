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

st.set_page_config(page_title="Scout · natural-language stock screener", page_icon="🔎", layout="wide")
st.markdown("""
<style>
.chip{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.78rem;margin-left:6px}
.assumed{background:rgba(245,158,11,.15);color:#b45309;border:1px solid rgba(245,158,11,.4)}
.explicit{background:rgba(16,185,129,.12);color:#047857;border:1px solid rgba(16,185,129,.35)}
.path{font-size:0.8rem;opacity:.75}
.block-container{padding-top:1.6rem}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ API helpers
def api(method: str, path: str, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", headers={"X-API-Key": KEY}, timeout=60, **kw)
    except httpx.HTTPError as e:
        st.session_state["api_error"] = f"API unreachable at {API}: {e}"
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
        st.error(detail)
        return None
    return r.json()


ss = st.session_state
ss.setdefault("session_id", None)
ss.setdefault("turns", [])        # [{"user": str, "resp": dict}]
ss.setdefault("current", None)    # latest response with a result
ss.setdefault("pending", None)

EXAMPLES = [
    "Find profitable companies with low debt and increasing revenue",
    "Undervalued banks with ROE above 15",
    "Quality mid caps in an uptrend",
    "IT or pharma stocks near 52 week high with rising delivery",
    "Oversold large caps that pay dividends",
    "Small caps with high growth and positive free cash flow, not in financial services",
]


def send(message: str):
    ss.pop("api_error", None)
    resp = api("POST", "/v1/chat", json={"message": message, "session_id": ss.session_id})
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
health = api("GET", "/health") or {}
with st.sidebar:
    st.markdown("### 🔎 Scout")
    st.caption(f"NSE · Nifty 500 · data as of **{health.get('data_as_of', '?')}** · {health.get('stocks', '?')} stocks")
    st.markdown("**Try asking**")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex_{ex}", width="stretch"):
            ss.pending = ex
    if st.button("🧹 New conversation", width="stretch"):
        ss.session_id, ss.turns, ss.current = None, [], None
        st.rerun()
    stats = api("GET", "/v1/stats") or {}
    if stats:
        st.markdown("**Query routing (this server)**")
        st.dataframe(pd.DataFrame(stats).T[["queries", "avg_latency_ms", "tokens"]], width="stretch")

tab_chat, tab_saved, tab_bt, tab_gloss = st.tabs(["💬 Screener", "🔔 Saved screens & alerts", "📈 Backtest", "📖 How terms are defined"])


# ------------------------------------------------------------------ rendering helpers
def path_badge(r: dict) -> str:
    p = r.get("path")
    if p == "fastpath":
        return f"⚡ deterministic parser · 0 tokens · {r.get('parse_ms', 0):.0f} ms"
    if p == "cache":
        return "💾 cached interpretation · 0 tokens"
    if p == "llm":
        return f"🧠 LLM ({r.get('model')}) · {r.get('tokens', 0):,} tokens · {r.get('parse_ms', 0):.0f} ms"
    if p == "degraded":
        return "⚠️ LLM unavailable: partial deterministic parse (see notes)"
    if p == "edit":
        return "✎ direct edit · no LLM"
    return p or ""


def render_chips(resp: dict):
    res = resp["result"]
    screen = resp["screen"]
    st.markdown("**What I understood** · edit any condition; changes re-run instantly without the LLM")
    stats = {c["id"]: c for c in res["criteria"]}
    for item in resp["interpretation"]:
        i = item["index"]
        crit = stats.get(f"__c{i}", {})
        c1, c2, c3 = st.columns([6, 2, 1])
        tag = (f"<span class='chip assumed' title='{item.get('rationale') or ''}'>assumed from “{item['source_phrase']}”</span>"
               if item["assumed"] and item.get("source_phrase")
               else "<span class='chip explicit'>as you stated</span>")
        c1.markdown(f"**{item['text']}** {tag}", unsafe_allow_html=True)
        if item.get("rationale") and item["assumed"]:
            c1.caption(f"Why: {item['rationale']}")
        c2.progress(min(crit.get("pass_rate", 0) / 100, 1.0),
                    text=f"{crit.get('pass_count', 0)} of {res['universe_size']} pass")
        cond = screen["all"][i]
        with c3:
            if isinstance(cond, dict) and "field" in cond and isinstance(cond.get("value"), (int, float)) \
                    and not isinstance(cond.get("value"), bool):
                with st.popover("✎"):
                    v = st.number_input("Threshold", value=float(cond["value"]), key=f"edit_{i}_{len(ss.turns)}")
                    if st.button("Apply", key=f"apply_{i}_{len(ss.turns)}"):
                        new = {**screen, "all": [dict(c) for c in screen["all"]]}
                        new["all"][i] = {**cond, "value": v, "assumed": False}
                        rerun_screen(new, f"{item['text']} → {v:g}")
                        st.rerun()
            if st.button("✕", key=f"rm_{i}_{len(ss.turns)}", help="Remove this condition"):
                new = {**screen, "all": [c for j, c in enumerate(screen["all"]) if j != i]}
                rerun_screen(new, f"removed: {item['text']}")
                st.rerun()


def render_result(resp: dict):
    res = resp["result"]
    st.caption(path_badge(resp))
    if resp.get("reply"):
        st.markdown(resp["reply"])
    for ch in resp.get("changes") or []:
        st.markdown(f"↳ _{ch}_")
    if resp.get("unmapped"):
        st.warning("I couldn't map these to available data, so they were **not** applied: "
                   + ", ".join(f"“{u}”" for u in resp["unmapped"]))
    for n in (resp.get("notes") or []) + (res.get("warnings") or []):
        st.info(n)
    render_chips(resp)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matches", f"{res['total_matches']}", help="stocks passing every condition")
    m2.metric("Universe", f"{res['universe']} ({res['universe_size']})")
    m3.metric("Data as of", res["as_of"])
    m4.metric("Query time", f"{res['elapsed_ms']:.0f} ms")

    if res["results"]:
        fields = [k for k in res["results"][0]["values"] if k not in ("market_cap_cr", "close")]
        rows = [{"Symbol": r["symbol"], "Company": r["name"], "Sector": r["sector"],
                 "Match strength": r["match_strength"], "Mkt cap (₹ Cr)": r["market_cap_cr"],
                 **{k: r["values"].get(k) for k in fields},
                 "⚠ Borderline": ", ".join(r["borderline"])} for r in res["results"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Match strength": st.column_config.ProgressColumn(
                         min_value=0, max_value=100, format="%d",
                         help="How comfortably the stock clears your conditions (margin of safety, peer-normalised)"),
                         "Mkt cap (₹ Cr)": st.column_config.NumberColumn(format="%,.0f")})
        st.markdown("**Why each stock matches**")
        for r in res["results"][:12]:
            with st.expander(f"{r['symbol']} · {r['name']} · strength {r['match_strength']:.0f}"):
                for x in r["reasons"]:
                    st.markdown(f"{'✅' if x['passed'] else '❌'} {x['text']}")
                (st.warning if r["borderline"] else st.success)(r["summary"])
    else:
        st.error("No stock passes every condition.")

    if res.get("suggestions"):
        st.markdown("**Bottleneck analysis** · the single change that unlocks the most results")
        for k, s in enumerate(res["suggestions"]):
            label = (f"Relax to {s['relax_to']} → {s['matches_if_relaxed']} stocks" if s.get("relax_to")
                     else f"Drop “{s['criterion']}” → {s['matches_if_removed']} stocks")
            if st.button(label, key=f"sugg_{k}_{len(ss.turns)}"):
                apply_ops(s["patch"], label)
                st.rerun()
    if res.get("near_misses"):
        with st.expander(f"🎯 Near misses: {len(res['near_misses'])} stocks failed exactly one condition"):
            st.dataframe(pd.DataFrame([{"Symbol": n["symbol"], "Company": n["name"], "Missed on": n["missed"]}
                                       for n in res["near_misses"]]), hide_index=True, width="stretch")

    with st.expander("💾 Save this screen / create an alert"):
        name = st.text_input("Name", value=(ss.turns[-1]["user"][:60] if ss.turns else "My screen"),
                             key=f"name_{len(ss.turns)}")
        hook = st.text_input("Webhook URL (optional, https - Slack/Discord/Teams)", key=f"hook_{len(ss.turns)}")
        alert = st.checkbox("Alert me when stocks enter or leave this screen", value=True, key=f"al_{len(ss.turns)}")
        if st.button("Save", key=f"save_{len(ss.turns)}"):
            saved = api("POST", "/v1/screens", json={"name": name, "screen": resp["screen"],
                                                     "query": ss.turns[-1]["user"] if ss.turns else ""})
            if saved and alert:
                a = api("POST", "/v1/alerts", json={"screen_id": saved["id"], "webhook_url": hook or None})
                if a:
                    st.success(f"Saved, and the alert is live (baseline {a['baseline_members']} stocks as of {a['as_of']}).")
            elif saved:
                st.success("Saved.")

    with st.expander("🔍 Under the hood: validated DSL and compiled SQL"):
        st.caption("The LLM only ever produces the JSON on the left. It is validated against a whitelist, then "
                   "compiled to parameterised SQL that runs in a sandboxed in-memory database.")
        a, b = st.columns(2)
        a.json(resp["screen"])
        b.code(res["sql"], language="sql")


# ------------------------------------------------------------------ chat tab
with tab_chat:
    st.markdown("## Describe the stocks you're looking for")
    st.caption("Plain English. Ask follow-ups like “only banks”, “drop the debt filter”, “sort by ROE”, "
               "or “why isn't TCS here?”")
    for t in ss.turns[:-1]:
        with st.chat_message("user"):
            st.markdown(t["user"])
        with st.chat_message("assistant"):
            r = t["resp"]
            if r.get("error"):
                st.error(r["error"])
                continue
            st.caption(path_badge(r))
            txt = r.get("reply") or "; ".join(r.get("changes") or []) or ""
            if r.get("result"):
                txt += f"  \n→ {r['result']['total_matches']} matches"
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
                st.caption("🩺 diagnosis · 0 tokens")
                st.markdown(r["reply"])
                for d in r.get("diagnosis", []):
                    st.markdown(f"{'✅' if d['passed'] else '❌'} {d['text']}")
            elif r.get("result"):
                render_result(r)
            else:
                st.caption(path_badge(r))
                st.markdown(r.get("reply") or "I couldn't turn that into a stock screen. Try describing "
                                                "the kind of companies you want.")
                if r.get("unmapped"):
                    st.warning("Not available in the data: " + ", ".join(r["unmapped"]))
    q = st.chat_input("e.g. profitable companies with low debt and increasing revenue")
    if q or ss.pending:
        msg = q or ss.pending
        ss.pending = None
        with st.spinner("Understanding your request…"):
            send(msg)
        st.rerun()

# ------------------------------------------------------------------ saved & alerts
with tab_saved:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("### Saved screens")
        for s in api("GET", "/v1/screens") or []:
            with st.container(border=True):
                st.markdown(f"**{s['name']}** · saved {s['created'][:10]}")
                st.caption(" AND ".join(s["description"]) or "no filters")
                b1, b2, b3 = st.columns(3)
                if b1.button("Open", key=f"open_{s['id']}"):
                    rerun_screen(s["screen"], f"opened saved screen “{s['name']}”")
                    st.rerun()
                if b2.button("Alert me", key=f"alert_{s['id']}"):
                    a = api("POST", "/v1/alerts", json={"screen_id": s["id"]})
                    if a:
                        st.success(f"Alert created (baseline {a['baseline_members']} stocks)")
                if b3.button("Delete", key=f"del_{s['id']}"):
                    api("DELETE", f"/v1/screens/{s['id']}")
                    st.rerun()
    with c2:
        st.markdown("### Alerts")
        alerts = api("GET", "/v1/alerts") or []
        for a in alerts:
            st.markdown(f"🔔 **{a['screen_name']}** · {a['members']} members · last checked on data of {a['last_as_of']}")
        if alerts and st.button("Check alerts now", help="The scheduler also runs this automatically"):
            out = api("POST", "/v1/alerts/run", params={"force": True})
            if out:
                n = sum(len(v) for v in out["events"].values())
                st.success(f"Evaluated on data as of {out['as_of']}: {n} membership change(s).")
        st.markdown("### Inbox")
        events = api("GET", "/v1/alerts/events") or []
        if not events:
            st.caption("No membership changes yet. Alerts fire when stocks enter or leave a saved screen after a data refresh.")
        for e in events:
            icon = "🟢 entered" if e["event"] == "entered" else "🔴 left"
            st.markdown(f"{icon} **{e['symbol']}**, {e['screen_name']} · data {e['as_of']}")
            st.caption(e["detail"])

# ------------------------------------------------------------------ backtest
with tab_bt:
    st.markdown("### How would the current screen have done?")
    st.caption("Point-in-time: the screen is re-run as of the chosen date using only prices up to that day and "
               "financial results whose SEBI filing deadline had passed (45 days after quarter end, 60 after FY end).")
    if not ss.current:
        st.info("Run a screen in the Screener tab first.")
    else:
        st.markdown("Screen: " + " AND ".join(i["text"] for i in ss.current["interpretation"]))
        c1, c2 = st.columns(2)
        try:  # anchor on the data's date, not the clock: a deployed snapshot may be weeks old
            anchor = date.fromisoformat(health["data_as_of"])
        except (KeyError, TypeError, ValueError):
            anchor = date.today()
        as_of = c1.date_input("Screen as of", value=anchor - timedelta(days=200),
                              min_value=anchor - timedelta(days=560), max_value=anchor - timedelta(days=30))
        horizon = c2.slider("Hold for (days)", 30, 365, 180, step=15)
        if st.button("Run backtest", type="primary"):
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

# ------------------------------------------------------------------ glossary
with tab_gloss:
    meta = api("GET", "/v1/meta") or {}
    st.markdown("### Vague terms have explicit, inspectable definitions")
    st.caption("The deterministic parser and the LLM both use this table, so “low debt” always means the same thing. "
               "Anything assumed is shown as an amber chip you can edit.")
    st.dataframe(pd.DataFrame([{"You say": c["phrase"], "We apply": " AND ".join(c["means"]), "Why": c["rationale"]}
                               for c in meta.get("concepts", [])]), hide_index=True, width="stretch")
    st.markdown("### Every screenable field")
    st.dataframe(pd.DataFrame(meta.get("fields", [])), hide_index=True, width="stretch")
