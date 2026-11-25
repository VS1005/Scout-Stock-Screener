"""ETL entrypoint.

    python -m etl.run                      # full: download + load + build latest snapshot
    python -m etl.run --download-only      # just warm the raw cache
    python -m etl.run --snapshot-only --as-of 2026-11-10   # rebuild snapshot for a past date (alerts demo)
"""
from __future__ import annotations

import argparse
import time
from datetime import date

import duckdb
import pandas as pd

from etl.fundamentals import fetch_fundamentals
from etl.nse import adjust_for_corporate_actions, fetch_prices, fetch_universe
from screener.config import DB_PATH
from screener.snapshot import build_snapshot


def load_raw(days: int, limit: int = 0) -> None:
    t0 = time.time()
    print("[1/3] NSE index constituents ...")
    uni = fetch_universe()
    if limit:
        uni = uni.head(limit)
    symbols = set(uni["symbol"])
    print(f"      {len(uni)} stocks")

    print(f"[2/3] NSE bhavcopy + index closes for last {days} calendar days ...")
    prices, index = fetch_prices(symbols, days)
    prices = adjust_for_corporate_actions(prices)
    n_ca = int((prices["adj_factor"].round(4) != 1).groupby(prices["symbol"]).any().sum())
    print(f"      {len(prices):,} price rows, {prices['date'].nunique()} sessions, "
          f"{n_ca} symbols back-adjusted for corporate actions")

    print("[3/3] Fundamentals (yfinance) ...")
    fin, meta = fetch_fundamentals(sorted(symbols))
    print(f"      {len(fin):,} statement line items for {fin['symbol'].nunique() if len(fin) else 0} stocks")

    uni = uni.merge(meta, on="symbol", how="left")
    con = duckdb.connect(str(DB_PATH))
    for name, df in [("universe", uni), ("prices", prices), ("index_prices", index), ("financials", fin)]:
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.close()
    print(f"Raw tables loaded in {time.time() - t0:.0f}s → {DB_PATH}")


def write_snapshot(as_of: date) -> None:
    con = duckdb.connect(str(DB_PATH))
    snap = build_snapshot(con, pd.Timestamp(as_of))
    con.register("_snap", snap)
    con.execute("CREATE OR REPLACE TABLE snapshot AS SELECT * FROM _snap")
    con.execute("CREATE OR REPLACE TABLE snapshot_meta AS SELECT ? AS as_of, now() AS built_at",
                [snap["as_of"].iloc[0]])
    con.close()
    print(f"Snapshot as of {snap['as_of'].iloc[0].date()}: {len(snap)} stocks × {snap.shape[1]} fields")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=600, help="calendar days of price history")
    ap.add_argument("--limit", type=int, default=0, help="only first N stocks (quick test)")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--snapshot-only", action="store_true")
    ap.add_argument("--as-of", type=str, default=None)
    a = ap.parse_args()
    if not a.snapshot_only:
        load_raw(a.days, a.limit)
    if not a.download_only:
        write_snapshot(date.fromisoformat(a.as_of) if a.as_of else date.today())


if __name__ == "__main__":
    main()
