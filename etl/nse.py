"""NSE public-archive loaders: index constituents, full bhavcopy (with delivery %), index closes.

All files are cached per trading day under data/raw so the ETL is resumable and polite.
"""
from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import requests

from screener.config import NSE_HEADERS, RAW_DIR

ARCHIVE = "https://nsearchives.nseindia.com"
INDEX_LISTS = {
    "NIFTY50": "ind_nifty50list.csv",
    "NIFTY100": "ind_nifty100list.csv",
    "NIFTY200": "ind_nifty200list.csv",
    "NIFTY500": "ind_nifty500list.csv",
}
BENCHMARKS = ["Nifty 50", "Nifty 500"]

_session = requests.Session()
_session.headers.update(NSE_HEADERS)


def _get(url: str, retries: int = 3) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            r = _session.get(url, timeout=20)
            if r.status_code == 404:
                return None  # holiday / weekend / not yet published
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_universe() -> pd.DataFrame:
    """One row per Nifty-500 stock with NSE's own sector classification + index membership flags."""
    frames = {}
    for name, fname in INDEX_LISTS.items():
        raw = _get(f"{ARCHIVE}/content/indices/{fname}")
        if raw is None:
            raise RuntimeError(f"Could not download {fname}")
        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip() for c in df.columns]
        frames[name] = df
    uni = frames["NIFTY500"].rename(columns={
        "Company Name": "name", "Industry": "sector", "Symbol": "symbol", "ISIN Code": "isin", "Series": "series"})
    uni = uni[["symbol", "name", "sector", "isin"]].copy()
    for name, df in frames.items():
        uni[f"in_{name.lower()}"] = uni["symbol"].isin(set(df["Symbol"]))
    return uni


def _bhav_one(d: date, symbols: set) -> Optional[pd.DataFrame]:
    # cache the FULL equity file (not universe-filtered) so the cache never depends on the universe
    cache = RAW_DIR / "bhav" / f"{d:%Y%m%d}.parquet"
    if cache.exists():
        out = pd.read_parquet(cache)
        return out[out["symbol"].isin(symbols)]
    raw = _get(f"{ARCHIVE}/products/content/sec_bhavdata_full_{d:%d%m%Y}.csv")
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip().isin(["EQ", "BE", "BZ"])]
    # Use the session date printed INSIDE the file, not the requested date: NSE sometimes serves a
    # neighbouring session's file for a requested date, which would otherwise create duplicate bars.
    out = pd.DataFrame({
        "date": pd.to_datetime(df["DATE1"].str.strip(), format="%d-%b-%Y").values,
        "symbol": df["SYMBOL"].values,
        "prev_close": pd.to_numeric(df["PREV_CLOSE"], errors="coerce").values,
        "open": pd.to_numeric(df["OPEN_PRICE"], errors="coerce").values,
        "high": pd.to_numeric(df["HIGH_PRICE"], errors="coerce").values,
        "low": pd.to_numeric(df["LOW_PRICE"], errors="coerce").values,
        "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce").values,
        "volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce").values,
        "turnover_lacs": pd.to_numeric(df["TURNOVER_LACS"], errors="coerce").values,
        "deliv_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce").values,
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out[out["symbol"].isin(symbols)]


def _index_one(d: date) -> Optional[pd.DataFrame]:
    cache = RAW_DIR / "index" / f"{d:%Y%m%d}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raw = _get(f"{ARCHIVE}/content/indices/ind_close_all_{d:%d%m%Y}.csv")
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Index Name"].isin(BENCHMARKS)]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Index Date"], format="%d-%m-%Y").values,
        "index_name": df["Index Name"].values,
        "close": pd.to_numeric(df["Closing Index Value"], errors="coerce").values,
        "pe": pd.to_numeric(df["P/E"], errors="coerce").values,
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def weekdays_back(n_calendar_days: int, end: Optional[date] = None) -> List[date]:
    """All calendar days (weekends included: NSE holds special weekend sessions, e.g. Budget day and
    Diwali Muhurat trading; skipping them breaks the PREV_CLOSE chain). Non-sessions simply 404."""
    end = end or date.today()
    return [end - timedelta(days=i) for i in range(n_calendar_days)]


def fetch_prices(symbols: set, n_calendar_days: int, workers: int = 4) -> tuple:
    """Download (or load from cache) daily bhavcopy + index closes. Returns (prices_df, index_df)."""
    days = weekdays_back(n_calendar_days)
    bhav, idx = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_bhav_one, d, symbols): ("b", d) for d in days}
        futs.update({ex.submit(_index_one, d): ("i", d) for d in days})
        done = 0
        for f in as_completed(futs):
            kind, _ = futs[f]
            res = f.result()
            if res is not None and len(res):
                (bhav if kind == "b" else idx).append(res)
            done += 1
            if done % 100 == 0:
                print(f"  nse: {done}/{len(futs)} files processed")
    prices = pd.concat(bhav, ignore_index=True).sort_values(["symbol", "date"])
    # A symbol can trade in EQ and BE series on different days but not both on the same day.
    prices = prices.drop_duplicates(["symbol", "date"], keep="last")
    index = pd.concat(idx, ignore_index=True).drop_duplicates(["index_name", "date"]).sort_values(["index_name", "date"])
    return prices, index


# Price ratios produced by common Indian corporate actions:
#   splits 1:k → 1/k;  bonus a:b → b/(a+b) (1:1 → 1/2, 2:1 → 1/3, 1:2 → 2/3, 4:1 → 1/5);  combos (e.g. 1/10)
CLEAN_RATIOS = [1 / 2, 1 / 3, 2 / 3, 1 / 4, 1 / 5, 1 / 6, 2 / 5, 1 / 8, 1 / 10, 1 / 20, 2, 5, 10]


def adjust_for_corporate_actions(prices: pd.DataFrame) -> pd.DataFrame:
    """Back-adjust OHLCV for splits & bonuses without an external corporate-actions feed.

    NSE's price-band regime (circuit filters of <=20% for most stocks, staged dynamic bands for F&O
    names) makes a one-day close/prev-close ratio below 0.65 or above 1.8 practically impossible as a
    *market* move, so such a jump is treated as a corporate action. Its factor is snapped to the
    nearest standard split/bonus ratio when within 6% (removing that day's genuine market move from
    the factor). Earlier bars are multiplied by the cumulative factor of all later events.
    A same-day event across >=10% (and >=5) of stocks is a data gap, not a corporate action, and is ignored.
    """
    p = prices.sort_values(["symbol", "date"]).copy()
    r = p["close"] / p.groupby("symbol")["close"].shift(1)
    is_ca = (r < 0.65) | (r > 1.8)
    share = is_ca.groupby(p["date"]).transform("mean")
    count = is_ca.groupby(p["date"]).transform("sum")
    is_ca &= ~((share >= 0.10) & (count >= 5))

    def snap(x: float) -> float:
        best = min(CLEAN_RATIOS, key=lambda c: abs(x / c - 1))
        return best if abs(x / best - 1) < 0.06 else x

    factor = pd.Series(1.0, index=p.index)
    factor[is_ca] = r[is_ca].map(snap)
    p["_f"] = factor
    # product of factors strictly AFTER each row = reverse cumprod / own factor
    p["adj_factor"] = p.iloc[::-1].groupby("symbol")["_f"].cumprod().iloc[::-1] / p["_f"]
    for c in ("open", "high", "low", "close", "prev_close"):
        p[c] = p[c] * p["adj_factor"]
    p["volume"] = p["volume"] / p["adj_factor"]
    return p.drop(columns=["_f"])
