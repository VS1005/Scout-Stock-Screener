"""Point-in-time feature snapshot: one row per stock, every screenable field, as of a given date.

The same function powers:
  * the live screener (as_of = latest session),
  * alert demos (rebuild for an earlier date, then the latest, and diff membership),
  * backtests (screen as of T using only data public at T, measure returns after T).

Point-in-time rules
  * prices: only sessions <= as_of
  * financial statements: only periods whose filing deadline (SEBI LODR Reg. 33: quarter end + 45d,
    FY end + 60d) is <= as_of — so a backtest never sees results that weren't published yet.
Known approximations (documented in the README): share count, promoter holding and dividend yield
are current values from yfinance, not historical.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from screener.config import ANNUAL_REPORT_LAG_DAYS, QUARTERLY_REPORT_LAG_DAYS

CR = 1e7  # 1 crore = 10 million INR
FINANCIAL_SECTORS = {"Financial Services"}
FINANCIAL_NA_FIELDS = ("debt_to_equity", "interest_coverage")  # leverage is a lender's raw material
YEAR = 252  # trading sessions in a year


def _pct(a, b):
    if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1.0) * 100.0


def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up.iloc[-1] / dn.iloc[-1] if dn.iloc[-1] else np.inf
    return float(100 - 100 / (1 + rs))


def _technical(g: pd.DataFrame, bench_ret: pd.Series, bench_6m: float) -> Dict[str, float]:
    c = g["close"].reset_index(drop=True)
    n = len(c)
    last = c.iloc[-1]

    def ret(k):
        return _pct(last, c.iloc[-1 - k]) if n > k else np.nan

    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    yr = c.iloc[-YEAR:]
    daily = c.pct_change()
    out = {
        "close": last,
        "ret_1w": ret(5), "ret_1m": ret(21), "ret_3m": ret(63), "ret_6m": ret(126), "ret_1y": ret(YEAR),
        "sma_50": sma50.iloc[-1], "sma_200": sma200.iloc[-1],
        "rsi_14": _rsi(c) if n > 20 else np.nan,
        "high_52w": yr.max(), "low_52w": yr.min(),
        "pct_from_52w_high": _pct(last, yr.max()),
        "pct_from_52w_low": _pct(last, yr.min()),
        "volatility_1y": float(daily.iloc[-YEAR:].std() * np.sqrt(YEAR) * 100) if n > 60 else np.nan,
        "max_drawdown_1y": float(((yr / yr.cummax()) - 1).min() * 100) if n > 20 else np.nan,
        "avg_turnover_cr_20d": float(g["turnover_lacs"].iloc[-20:].mean() / 100),
        "volume_ratio_20d": float(g["volume"].iloc[-1] / g["volume"].iloc[-21:-1].mean())
        if n > 21 and g["volume"].iloc[-21:-1].mean() > 0 else np.nan,
        "delivery_pct_20d": float(g["deliv_pct"].iloc[-20:].mean()),
        "delivery_trend": float(g["deliv_pct"].iloc[-20:].mean() - g["deliv_pct"].iloc[-60:].mean())
        if n >= 60 else np.nan,
    }
    out["price_vs_sma200_pct"] = _pct(last, out["sma_200"])
    out["above_sma_200"] = bool(last > out["sma_200"]) if pd.notna(out["sma_200"]) else None
    out["above_sma_50"] = bool(last > out["sma_50"]) if pd.notna(out["sma_50"]) else None
    if n >= 221:
        above = (sma50 > sma200).iloc[-21:]
        out["golden_cross_20d"] = bool(above.iloc[-1] and not above.iloc[0])
    else:
        out["golden_cross_20d"] = None
    out["rel_strength_6m"] = out["ret_6m"] - bench_6m if pd.notna(out["ret_6m"]) else np.nan
    # beta vs Nifty 50 on overlapping daily returns
    r = pd.Series(daily.values, index=g["date"].values).iloc[-YEAR:]
    j = pd.concat([r, bench_ret], axis=1, join="inner").dropna()
    out["beta_1y"] = float(np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1] / j.iloc[:, 1].var()) if len(j) > 60 else np.nan
    return out


def _latest(fin: pd.DataFrame, item: str, freq: str, k: int = 1) -> pd.Series:
    s = fin[(fin["item"] == item) & (fin["freq"] == freq)].sort_values("period_end")
    return s.set_index("period_end")["value"].iloc[-k:] if len(s) else pd.Series(dtype=float)


def _ttm(fin: pd.DataFrame, item: str):
    q = _latest(fin, item, "Q", 4)
    if len(q) == 4 and (q.index[-1] - q.index[0]).days <= 290:  # 4 consecutive quarters
        return float(q.sum()), q.index[-1]
    a = _latest(fin, item, "A", 1)
    return (float(a.iloc[-1]), a.index[-1]) if len(a) else (np.nan, None)


def _streak(series: pd.Series) -> float:
    """Number of consecutive year-on-year increases, counting back from the latest year."""
    v = series.dropna().values
    if len(v) < 2:
        return np.nan
    s = 0
    for i in range(len(v) - 1, 0, -1):
        if v[i] > v[i - 1]:
            s += 1
        else:
            break
    return float(s)


def _fundamental(fin: pd.DataFrame) -> Dict[str, float]:
    if fin.empty:
        return {}
    out: Dict[str, float] = {}
    rev_ttm, rev_end = _ttm(fin, "revenue")
    np_ttm, _ = _ttm(fin, "net_income")
    out["revenue_ttm_cr"] = rev_ttm / CR
    out["net_profit_ttm_cr"] = np_ttm / CR
    out["net_margin_pct"] = np_ttm / rev_ttm * 100 if rev_ttm and rev_ttm > 0 else np.nan

    rev_a = _latest(fin, "revenue", "A", 5)
    ni_a = _latest(fin, "net_income", "A", 5)
    eps_a = _latest(fin, "eps", "A", 2)
    out["revenue_growth_yoy"] = _pct(rev_a.iloc[-1], rev_a.iloc[-2]) if len(rev_a) >= 2 else np.nan
    out["profit_growth_yoy"] = (
        (ni_a.iloc[-1] - ni_a.iloc[-2]) / abs(ni_a.iloc[-2]) * 100
        if len(ni_a) >= 2 and ni_a.iloc[-2] != 0 else np.nan)
    out["eps_growth_yoy"] = (
        (eps_a.iloc[-1] - eps_a.iloc[-2]) / abs(eps_a.iloc[-2]) * 100
        if len(eps_a) >= 2 and eps_a.iloc[-2] != 0 else np.nan)
    out["revenue_cagr_3y"] = (
        ((rev_a.iloc[-1] / rev_a.iloc[-4]) ** (1 / 3) - 1) * 100
        if len(rev_a) >= 4 and rev_a.iloc[-4] > 0 and rev_a.iloc[-1] > 0 else np.nan)
    out["revenue_growth_streak"] = _streak(rev_a)
    out["profit_growth_streak"] = _streak(ni_a)

    # latest quarter vs same quarter last year (seasonality-safe)
    rq = _latest(fin, "revenue", "Q", 8)
    out["revenue_growth_qoq_yoy"] = np.nan
    if len(rq) >= 2:
        target = rq.index[-1] - pd.DateOffset(years=1)
        prior = rq[(rq.index > target - pd.Timedelta(days=20)) & (rq.index < target + pd.Timedelta(days=20))]
        if len(prior):
            out["revenue_growth_qoq_yoy"] = _pct(rq.iloc[-1], prior.iloc[-1])

    # balance sheet: most recent public one (quarterly if available, else annual)
    def bs(item):
        cands = fin[(fin["item"] == item)].sort_values("period_end")
        return float(cands["value"].iloc[-1]) if len(cands) else np.nan

    debt, equity, assets = bs("total_debt"), bs("equity"), bs("total_assets")
    out["equity_cr"] = equity / CR
    out["debt_to_equity"] = debt / equity if equity and equity > 0 and pd.notna(debt) else np.nan
    ni_last = ni_a.iloc[-1] if len(ni_a) else np.nan
    eq_a = _latest(fin, "equity", "A", 2)
    avg_eq = eq_a.mean() if len(eq_a) else np.nan
    out["roe"] = ni_last / avg_eq * 100 if avg_eq and avg_eq > 0 else np.nan
    out["roa"] = ni_last / assets * 100 if assets and assets > 0 else np.nan

    op_a = _latest(fin, "operating_income", "A", 1)
    rev_last = rev_a.iloc[-1] if len(rev_a) else np.nan
    out["operating_margin_pct"] = (op_a.iloc[-1] / rev_last * 100
                                   if len(op_a) and rev_last and rev_last > 0 else np.nan)
    ebit = _latest(fin, "ebit", "A", 1)
    intr = _latest(fin, "interest_expense", "A", 1)
    if len(ebit) and len(intr):
        i = abs(intr.iloc[-1])  # Yahoo reports interest expense with either sign
        out["interest_coverage"] = min(ebit.iloc[-1] / i, 100.0) if i > 0 else 100.0  # no interest = fully covered
    else:
        out["interest_coverage"] = np.nan
    fcf = _latest(fin, "free_cf", "A", 1)
    ocf = _latest(fin, "operating_cf", "A", 1)
    out["free_cash_flow_cr"] = fcf.iloc[-1] / CR if len(fcf) else np.nan
    out["cash_conversion"] = (ocf.iloc[-1] / ni_last
                              if len(ocf) and pd.notna(ni_last) and ni_last > 0 else np.nan)
    out["fundamentals_as_of"] = rev_end
    return out


def build_snapshot(con, as_of: pd.Timestamp, symbols: Optional[list] = None) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    start = as_of - pd.Timedelta(days=400)
    prices = con.execute(
        "SELECT * FROM prices WHERE date BETWEEN ? AND ? ORDER BY symbol, date", [start, as_of]).df()
    if prices.empty:
        raise ValueError(f"No price data on or before {as_of.date()}")
    eff_date = prices["date"].max()
    uni = con.execute("SELECT * FROM universe").df()
    if symbols:
        uni = uni[uni["symbol"].isin(symbols)]
    fin = con.execute("SELECT * FROM financials").df()
    lag = np.where(fin["freq"] == "Q", QUARTERLY_REPORT_LAG_DAYS, ANNUAL_REPORT_LAG_DAYS)
    fin = fin[fin["period_end"] + pd.to_timedelta(lag, unit="D") <= as_of]  # point-in-time filter
    fin_by = dict(tuple(fin.groupby("symbol")))

    idx = con.execute(
        "SELECT date, close FROM index_prices WHERE index_name='Nifty 50' AND date BETWEEN ? AND ? ORDER BY date",
        [start, as_of]).df().set_index("date")["close"]
    bench_ret = idx.pct_change()
    idx500 = con.execute(
        "SELECT date, close FROM index_prices WHERE index_name='Nifty 500' AND date BETWEEN ? AND ? ORDER BY date",
        [start, as_of]).df()["close"]
    bench_6m = _pct(idx500.iloc[-1], idx500.iloc[-127]) if len(idx500) > 127 else np.nan

    rows, members = [], set(uni["symbol"])
    for sym, g in prices.groupby("symbol"):
        if sym not in members or g["date"].iloc[-1] < eff_date - pd.Timedelta(days=10):
            continue  # not in universe or suspended / delisted
        row = {"symbol": sym}
        row.update(_technical(g, bench_ret, bench_6m))
        row.update(_fundamental(fin_by.get(sym, pd.DataFrame())))
        rows.append(row)
    snap = pd.DataFrame(rows).merge(uni, on="symbol", how="inner")

    snap["market_cap_cr"] = snap["close"] * snap["shares_out"] / CR
    snap["mcap_rank"] = snap["market_cap_cr"].rank(ascending=False, method="first")
    # AMFI/SEBI categorisation: top 100 = large, 101–250 = mid, rest = small
    snap["cap_category"] = np.select(
        [snap["mcap_rank"] <= 100, snap["mcap_rank"] <= 250, snap["mcap_rank"].notna()], ["Large", "Mid", "Small"], None)
    np_ttm = snap["net_profit_ttm_cr"]
    snap["pe"] = np.where(np_ttm > 0, snap["market_cap_cr"] / np_ttm, np.nan)
    snap["pb"] = np.where(snap["equity_cr"] > 0, snap["market_cap_cr"] / snap["equity_cr"], np.nan)
    snap["earnings_yield"] = np.where(snap["market_cap_cr"] > 0, np_ttm / snap["market_cap_cr"] * 100, np.nan)
    # Leverage ratios are meaningless for banks/NBFCs (deposits & borrowings are their raw material)
    snap.loc[snap["sector"].isin(FINANCIAL_SECTORS), list(FINANCIAL_NA_FIELDS)] = np.nan
    snap = snap.rename(columns={"insider_pct": "promoter_holding_pct"})
    snap["as_of"] = eff_date
    for c in ("above_sma_200", "above_sma_50", "golden_cross_20d"):
        snap[c] = snap[c].astype("boolean")
    return snap.reset_index(drop=True)
