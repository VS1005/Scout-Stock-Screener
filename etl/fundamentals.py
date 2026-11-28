"""Company fundamentals from yfinance (NSE tickers with .NS suffix).

Stored in long format (symbol, freq, period_end, item, value) so the snapshot builder can
pick only the statements that were *public* as of any date (point-in-time correctness).
Each symbol is cached to data/raw/fund/<SYMBOL>.json → the job is resumable.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd

from screener.config import RAW_DIR

ITEMS = {
    "Total Revenue": "revenue",
    "Net Income Common Stockholders": "net_income",
    "Net Income": "net_income_alt",
    "Operating Income": "operating_income",
    "EBIT": "ebit",
    "Interest Expense": "interest_expense",
    "Diluted EPS": "eps",
    "Total Debt": "total_debt",
    "Stockholders Equity": "equity",
    "Total Assets": "total_assets",
    "Operating Cash Flow": "operating_cf",
    "Free Cash Flow": "free_cf",
}
CACHE = RAW_DIR / "fund"


def _frame_to_rows(df: pd.DataFrame, freq: str) -> List[dict]:
    rows = []
    if df is None or df.empty:
        return rows
    for raw_item, item in ITEMS.items():
        if raw_item not in df.index:
            continue
        for period, val in df.loc[raw_item].items():
            if pd.notna(val):
                rows.append({"freq": freq, "period_end": str(pd.Timestamp(period).date()),
                             "item": item, "value": float(val)})
    return rows


def _fetch_one(symbol: str) -> Dict:
    import yfinance as yf

    path = CACHE / f"{symbol}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = yf.Ticker(f"{symbol}.NS")
    rows: List[dict] = []
    for attr, freq in [("income_stmt", "A"), ("quarterly_income_stmt", "Q"), ("balance_sheet", "A"),
                       ("quarterly_balance_sheet", "Q"), ("cashflow", "A")]:
        try:
            rows += _frame_to_rows(getattr(t, attr), freq)
        except Exception:  # noqa: BLE001 - yfinance raises many types; a missing statement is not fatal
            pass
    meta = {}
    try:
        info = t.info or {}
        meta = {k: info.get(k) for k in ("industry", "sharesOutstanding", "impliedSharesOutstanding",
                                          "dividendYield", "longName", "heldPercentInsiders",
                                          "financialCurrency")}
    except Exception:  # noqa: BLE001
        pass
    out = {"symbol": symbol, "meta": meta, "rows": rows}
    if rows or meta:  # don't cache total failures (likely rate-limit) so a rerun retries them
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return out


MONEY_ITEMS = {"revenue", "net_income", "net_income_alt", "operating_income", "ebit", "interest_expense",
               "eps", "total_debt", "equity", "total_assets", "operating_cf", "free_cf"}


def _to_inr(fin: pd.DataFrame, currency: Dict[str, str]) -> pd.DataFrame:
    """Some NSE companies (e.g. Infosys, Wipro) report in USD on Yahoo. Convert at the FX rate on
    each statement's period-end date so market-cap/earnings ratios aren't off by ~85x."""
    import yfinance as yf

    fin["period_end"] = pd.to_datetime(fin["period_end"])
    foreign = {s: c for s, c in currency.items() if c and c != "INR"}
    if not foreign:
        return fin
    print(f"  converting {len(foreign)} non-INR reporters to INR: {sorted(set(foreign.values()))}")
    for cur in set(foreign.values()):
        fx = yf.Ticker(f"{cur}INR=X").history(period="6y")["Close"]
        fx.index = fx.index.tz_localize(None).normalize()
        mask = fin["symbol"].isin([s for s, c in foreign.items() if c == cur]) & fin["item"].isin(MONEY_ITEMS)
        rows = fin[mask].sort_values("period_end")
        rate = pd.merge_asof(rows[["period_end"]], fx.rename("rate").reset_index().rename(columns={"Date": "period_end"}),
                             on="period_end", direction="nearest")["rate"].values
        fin.loc[rows.index, "value"] = rows["value"].values * rate
    return fin


def fetch_fundamentals(symbols: List[str], workers: int = 4) -> tuple:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_one, s) for s in symbols]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                print("  fundamentals error:", e)
            if i % 50 == 0:
                print(f"  fundamentals: {i}/{len(symbols)}")
                time.sleep(1)
    fin, meta = [], []
    for r in results:
        for row in r["rows"]:
            fin.append({"symbol": r["symbol"], **row})
        m = r.get("meta") or {}
        meta.append({
            "symbol": r["symbol"],
            "industry": m.get("industry"),
            "shares_out": m.get("sharesOutstanding") or m.get("impliedSharesOutstanding"),
            # yfinance reports Indian dividend yield already in percent (1.95 = 1.95%)
            "dividend_yield": m.get("dividendYield"),
            "insider_pct": m["heldPercentInsiders"] * 100 if m.get("heldPercentInsiders") is not None else None,
        })
    fin_df = pd.DataFrame(fin)
    currency = {r["symbol"]: (r.get("meta") or {}).get("financialCurrency") or "INR" for r in results}
    if not fin_df.empty:
        fin_df = _to_inr(fin_df, currency)
        # Prefer 'Net Income Common Stockholders', fall back to 'Net Income'
        alt = fin_df[fin_df["item"] == "net_income_alt"].copy()
        alt["item"] = "net_income"
        fin_df = pd.concat([fin_df[fin_df["item"] != "net_income_alt"], alt])
        fin_df = fin_df.drop_duplicates(["symbol", "freq", "period_end", "item"], keep="first")
    return fin_df, pd.DataFrame(meta)
