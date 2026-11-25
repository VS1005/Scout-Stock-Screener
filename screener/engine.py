"""Execution engine: run a compiled screen, then explain, rank, find near-misses and suggest relaxations.

Everything here is deterministic — no LLM. Explanations are generated from the AST and the actual
numbers, so they cannot hallucinate.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from screener.compiler import Compiled, compile_screen
from screener.config import DB_PATH, QUERY_TIMEOUT_S
from screener.dsl import AnyGroup, Condition, PatchOp, RefValue, Screen, describe, fmt_num, fmt_value
from screener.registry import CAT, FIELDS, NUM
from screener.snapshot import FINANCIAL_SECTORS


def _sandboxed_connection(snap: pd.DataFrame, table: str = "snapshot") -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB holding only the snapshot; filesystem/network access disabled and locked."""
    con = duckdb.connect(":memory:")
    con.register("_s", snap)
    con.execute(f"CREATE TABLE {table} AS SELECT * FROM _s")
    con.unregister("_s")
    con.execute("SET enable_external_access = false")
    con.execute("SET lock_configuration = true")
    return con


class Engine:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._mtime = None
        self.snap: Optional[pd.DataFrame] = None
        self.con = None
        self.as_of = None
        self.reload()

    # -------------------------------------------------------------- data lifecycle
    def reload(self, force: bool = False) -> None:
        mtime = self.db_path.stat().st_mtime if self.db_path.exists() else None
        if not force and mtime == self._mtime and self.snap is not None:
            return
        with self._lock:
            src = duckdb.connect(str(self.db_path), read_only=True)
            try:
                snap = src.execute("SELECT * FROM snapshot").df()
            finally:
                src.close()
            self.snap = snap
            self.con = _sandboxed_connection(snap)
            self.as_of = pd.Timestamp(snap["as_of"].iloc[0]).date()
            self._mtime = mtime
            self._scales = self._robust_scales(snap)

    @staticmethod
    def _robust_scales(snap: pd.DataFrame) -> Dict[str, float]:
        """IQR per numeric field → puts 'margin of safety' for P/E and ROE on a common scale."""
        out = {}
        for k, f in FIELDS.items():
            if f.kind == NUM and k in snap:
                s = pd.to_numeric(snap[k], errors="coerce").dropna()
                iqr = float(s.quantile(0.75) - s.quantile(0.25)) if len(s) > 10 else 0
                out[k] = iqr if iqr > 0 else f.scale
        return out

    def categories(self) -> Dict[str, List[str]]:
        return {k: sorted(self.snap[k].dropna().unique().tolist()) for k in ("sector", "industry", "cap_category")}

    def stock(self, symbol: str) -> Optional[dict]:
        row = self.snap[self.snap["symbol"] == symbol.upper()]
        return None if row.empty else _clean(row.iloc[0].to_dict())

    # -------------------------------------------------------------- validation against live data
    def check_categories(self, screen: Screen) -> List[str]:
        """Category values must exist in the data; fix case/near-misses, report unknowns."""
        notes = []
        cats = self.categories()
        import difflib
        for leaf in screen.leaves():
            f = FIELDS[leaf.field]
            if f.kind != CAT or leaf.op == "contains":
                continue
            fixed = []
            for v in leaf.value:
                lower = {c.lower(): c for c in cats.get(leaf.field, [])}
                if v.lower() in lower:
                    fixed.append(lower[v.lower()])
                    continue
                m = difflib.get_close_matches(v, cats.get(leaf.field, []), n=1, cutoff=0.6)
                if m:
                    fixed.append(m[0])
                    notes.append(f"'{v}' isn't an exact {f.label}; using '{m[0]}'")
                else:
                    notes.append(f"'{v}' is not a known {f.label} - it will match nothing")
                    fixed.append(v)
            leaf.value = fixed
        return notes

    # -------------------------------------------------------------- execution
    def _execute(self, con, comp: Compiled) -> pd.DataFrame:
        timer = threading.Timer(QUERY_TIMEOUT_S, con.interrupt)
        timer.start()
        try:
            return con.execute(comp.sql, comp.params).df()
        finally:
            timer.cancel()

    def run(self, screen: Screen, con=None, scales=None, explain_limit: int = 100) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if con is None:
            self.reload()
            con, scales = self.con, self._scales
        notes = self.check_categories(screen) if con is self.con else []
        comp = compile_screen(screen)
        with self._lock:
            df = self._execute(con, comp)
        n_crit = len(comp.criteria)
        crit_ids = [c["id"] for c in comp.criteria]
        df["__passed"] = df[crit_ids].sum(axis=1) if n_crit else 0
        df["match_strength"] = self._strength(df, comp, scales)

        matches = df[df["__passed"] == n_crit].copy()
        if screen.sort and screen.sort.field in matches:
            matches = matches.sort_values(screen.sort.field, ascending=screen.sort.dir == "asc", na_position="last")
        elif n_crit:
            matches = matches.sort_values(["match_strength", "market_cap_cr"], ascending=[False, False])
        else:
            matches = matches.sort_values("market_cap_cr", ascending=False)
        total = len(matches)
        matches = matches.head(screen.limit)

        result = {
            "as_of": str(self.as_of),
            "universe": screen.universe,
            "universe_size": int(len(df)),
            "total_matches": int(total),
            "criteria": [self._criterion_stats(df, c) for c in comp.criteria],
            "results": [self._explain_row(r, comp, scales) for _, r in matches.head(explain_limit).iterrows()],
            "near_misses": self._near_misses(df, comp, scales) if n_crit >= 2 else [],
            "suggestions": self._relaxations(df, comp, screen) if total < 5 and n_crit else [],
            "warnings": screen.warnings() + notes,
            "sql": comp.sql,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return result

    def members(self, screen: Screen) -> List[str]:
        """Full (un-limited) set of matching symbols — used for alert membership diffs."""
        self.reload()
        comp = compile_screen(screen)
        with self._lock:
            df = self._execute(self.con, comp)
        ids = [c["id"] for c in comp.criteria]
        return sorted(df[df[ids].all(axis=1)]["symbol"]) if ids else sorted(df["symbol"])

    def diagnose(self, screen: Screen, symbol: str) -> Optional[List[dict]]:
        """Per-criterion pass/fail evidence for ONE stock (why is / isn't X in this screen?)."""
        self.reload()
        comp = compile_screen(screen)
        with self._lock:
            df = self._execute(self.con, comp)
        row = df[df["symbol"] == symbol.upper()]
        if row.empty:
            return None
        r = row.iloc[0]
        return [self._leaf_text(r, leaf, refcol, bool(r[col]))
                for entry in comp.criteria for col, leaf, refcol in entry["leaves"]]

    # -------------------------------------------------------------- explanation helpers
    @staticmethod
    def _margin(leaf: Condition, value: Any, ref: Any, scale: float) -> float:
        """Signed distance by which a value clears its threshold, in IQR units (negative = fails)."""
        if leaf.op in ("is", "in", "not_in", "contains"):
            return 1.0
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return -1.0
        if isinstance(leaf.value, RefValue):
            t = ref
        elif leaf.op == "between":
            lo, hi = leaf.value
            return min(value - lo, hi - value) / scale
        else:
            t = leaf.value
        if t is None or (isinstance(t, float) and math.isnan(t)):
            return -1.0
        d = (value - t) if leaf.op in (">", ">=") else (t - value)
        return d / scale

    def _strength(self, df: pd.DataFrame, comp: Compiled, scales) -> pd.Series:
        """0-100: average margin of safety across numeric criteria (how robustly a stock qualifies)."""
        if df.empty or not comp.criteria:
            return pd.Series(0.0, index=df.index)
        margins = []
        for entry in comp.criteria:
            best = None
            for col, leaf, refcol in entry["leaves"]:
                if FIELDS[leaf.field].kind != NUM:
                    continue
                vals = df[leaf.field].astype(float)
                ref = df[refcol].astype(float) if refcol else None
                if isinstance(leaf.value, RefValue):
                    t = ref
                    m = ((vals - t) if leaf.op in (">", ">=") else (t - vals)) / scales.get(leaf.field, 1)
                elif leaf.op == "between":
                    lo, hi = leaf.value
                    m = np.minimum(vals - lo, hi - vals) / scales.get(leaf.field, 1)
                else:
                    t = float(leaf.value)
                    m = ((vals - t) if leaf.op in (">", ">=") else (t - vals)) / scales.get(leaf.field, 1)
                m = m.fillna(-1).clip(-2, 2)
                best = m if best is None else np.maximum(best, m)
            if best is not None:
                margins.append(best)
        if not margins:
            return pd.Series(50.0, index=df.index)
        avg = sum(margins) / len(margins)
        return (50 + 25 * avg).clip(0, 100).round(0)

    def _leaf_text(self, row, leaf: Condition, refcol: Optional[str], passed: bool) -> dict:
        f = FIELDS[leaf.field]
        val = row.get(leaf.field)
        missing = val is None or (isinstance(val, float) and math.isnan(val)) or (val is pd.NA)
        if missing:
            reason = "no data"
            if f.na_note and row.get("sector") in FINANCIAL_SECTORS and "bank" in f.na_note.lower():
                reason = f.na_note
            elif f.na_note:
                reason = f"no data ({f.na_note})"
            return {"field": leaf.field, "passed": False, "text": f"{f.label}: {reason}", "missing": True}
        if f.kind == NUM:
            shown = fmt_num(leaf.field, val)
            if refcol:
                ref = row.get(refcol)
                target = f"{fmt_value(leaf.field, leaf.value)} ({fmt_num(leaf.field, ref)})"
            else:
                target = fmt_value(leaf.field, leaf.value)
            from screener.dsl import OP_WORDS
            text = f"{f.label} {shown} {OP_WORDS[leaf.op] if passed else 'fails ' + OP_WORDS[leaf.op]} {target}"
        elif f.kind == CAT:
            text = f"{f.label}: {val}"
        else:
            text = f"{f.label}: {'yes' if bool(val) else 'no'}"
        if leaf.assumed and leaf.source_phrase:
            text += f"  ·  from “{leaf.source_phrase}”"
        return {"field": leaf.field, "passed": bool(passed), "text": text, "missing": False}

    def _explain_row(self, r: pd.Series, comp: Compiled, scales) -> dict:
        reasons, tight = [], []
        for entry in comp.criteria:
            leaf_out = [self._leaf_text(r, leaf, refcol, bool(r[col])) for col, leaf, refcol in entry["leaves"]]
            if entry["kind"] == "any":
                ok = [l for l in leaf_out if l["passed"]]
                reasons.append({"passed": bool(r[entry["id"]]), "text": " OR ".join(l["text"] for l in (ok or leaf_out)),
                                "field": "any"})
            else:
                reasons.extend(leaf_out)
            for col, leaf, refcol in entry["leaves"]:
                if FIELDS[leaf.field].kind == NUM and bool(r[col]):
                    m = self._margin(leaf, r.get(leaf.field), r.get(refcol) if refcol else None,
                                     scales.get(leaf.field, 1))
                    if m < 0.1:
                        tight.append(FIELDS[leaf.field].label)
        summary = (f"Qualifies, but only just on {', '.join(tight)} - a small change could drop it."
                   if tight else "Clears every criterion with room to spare.")
        return {
            "symbol": r["symbol"], "name": r["name"], "sector": r["sector"], "industry": r.get("industry"),
            "cap_category": r.get("cap_category"),
            "market_cap_cr": _num(r.get("market_cap_cr")), "close": _num(r.get("close")),
            "match_strength": _num(r.get("match_strength")),
            "values": {k: _num(r.get(k)) if FIELDS.get(k) and FIELDS[k].kind == NUM else _clean_v(r.get(k))
                       for k in comp.fields if k in FIELDS},
            "reasons": reasons, "summary": summary, "borderline": tight,
        }

    def _criterion_stats(self, df: pd.DataFrame, entry: dict) -> dict:
        leaves = entry["leaves"]
        return {
            "id": entry["id"],
            "description": " OR ".join(describe(l) for _, l, _ in leaves),
            "pass_count": int(df[entry["id"]].sum()),
            "pass_rate": round(float(df[entry["id"]].mean() * 100), 1) if len(df) else 0,
            "assumed": any(l.assumed for _, l, _ in leaves),
            "source_phrase": next((l.source_phrase for _, l, _ in leaves if l.source_phrase), None),
            "rationale": next((l.rationale for _, l, _ in leaves if l.rationale), None),
        }

    def _near_misses(self, df: pd.DataFrame, comp: Compiled, scales, k: int = 8) -> List[dict]:
        """Stocks failing exactly ONE criterion, ranked by how close they came."""
        n = len(comp.criteria)
        cand = df[df["__passed"] == n - 1]
        out = []
        for _, r in cand.iterrows():
            failed = next(e for e in comp.criteria if not r[e["id"]])
            col, leaf, refcol = failed["leaves"][0]
            if FIELDS[leaf.field].kind != NUM or pd.isna(r.get(leaf.field)):
                continue  # wrong sector / missing data is not "close" - only numeric misses are near-misses
            gap = self._margin(leaf, r.get(leaf.field), r.get(refcol) if refcol else None, scales.get(leaf.field, 1))
            out.append({
                "symbol": r["symbol"], "name": r["name"], "sector": r["sector"],
                "missed": self._leaf_text(r, leaf, refcol, False)["text"],
                "missed_field": leaf.field, "gap": round(float(gap), 3),
            })
        out.sort(key=lambda x: -x["gap"])  # closest misses first (gap is negative)
        return out[:k]

    def _relaxations(self, df: pd.DataFrame, comp: Compiled, screen: Screen, target: int = 5) -> List[dict]:
        """For an empty/thin result: which single criterion is the bottleneck, and how far to relax it."""
        sugg = []
        ids = [e["id"] for e in comp.criteria]
        for i, entry in enumerate(comp.criteria):
            others = [x for x in ids if x != entry["id"]]
            pool = df[df[others].all(axis=1)] if others else df
            if len(pool) == 0:
                continue
            col, leaf, refcol = entry["leaves"][0]
            s = {"criterion": describe(leaf), "matches_if_removed": int(len(pool)),
                 "patch": [PatchOp(op="remove", field=leaf.field).model_dump(exclude_none=True)]}
            if (entry["kind"] == "leaf" and FIELDS[leaf.field].kind == NUM and leaf.op in (">", ">=", "<", "<=")
                    and not isinstance(leaf.value, RefValue)):
                vals = pool[leaf.field].dropna().astype(float)
                if len(vals):
                    asc = leaf.op in ("<", "<=")
                    ordered = vals.sort_values(ascending=asc).reset_index(drop=True)
                    new_t = ordered.iloc[min(target, len(ordered)) - 1]
                    new_t = float(np.round(new_t, 2))
                    new_leaf = leaf.model_copy(update={"value": new_t, "op": "<=" if asc else ">=", "assumed": True,
                                                       "rationale": f"relaxed from {fmt_value(leaf.field, leaf.value)}"})
                    s.update({
                        "relax_to": f"{FIELDS[leaf.field].label} {'≤' if asc else '≥'} {fmt_num(leaf.field, new_t)}",
                        "matches_if_relaxed": int(min(target, len(ordered))),
                        "patch": [PatchOp(op="replace", field=leaf.field, condition=new_leaf).model_dump(exclude_none=True)],
                    })
            sugg.append(s)
        sugg.sort(key=lambda s: -s.get("matches_if_relaxed", s["matches_if_removed"]))
        return sugg[:3]

    # -------------------------------------------------------------- backtest
    def backtest(self, screen: Screen, as_of: str, horizon_days: int = 90) -> Dict[str, Any]:
        """Run the screen on a point-in-time snapshot at `as_of`, measure forward returns.

        Equal-weight portfolio of matches vs Nifty 50 and vs the equal-weight universe.
        """
        from screener.snapshot import build_snapshot
        t = pd.Timestamp(as_of)
        src = duckdb.connect(str(self.db_path), read_only=True)
        try:
            snap = build_snapshot(src, t)
            start = pd.Timestamp(snap["as_of"].iloc[0])
            end_target = start + pd.Timedelta(days=horizon_days)
            px = src.execute(
                "SELECT symbol, date, close FROM prices WHERE date >= ? AND date <= ? ORDER BY date",
                [start, end_target + pd.Timedelta(days=7)]).df()
            idx = src.execute(
                "SELECT date, close FROM index_prices WHERE index_name='Nifty 50' AND date >= ? AND date <= ? ORDER BY date",
                [start, end_target + pd.Timedelta(days=7)]).df()
        finally:
            src.close()
        if px.empty or idx.empty:
            raise ValueError("not enough price history after as_of")
        end = px[px["date"] <= end_target]["date"].max()
        if (end - start).days < horizon_days * 0.8:
            raise ValueError(f"only {(end - start).days} days of data after {start.date()} - choose an earlier date")
        p0 = px[px["date"] == start].set_index("symbol")["close"]
        p1 = px[px["date"] == end].set_index("symbol")["close"]
        fwd = ((p1 / p0 - 1) * 100).rename("fwd_ret")

        # same compiler, same sandbox — only the snapshot differs (point-in-time at `as_of`)
        con = _sandboxed_connection(snap)
        comp = compile_screen(screen)
        df = self._execute(con, comp)
        con.close()
        ids = [c["id"] for c in comp.criteria]
        members = df[df[ids].all(axis=1)]["symbol"] if ids else df["symbol"]
        uni_syms = df["symbol"]
        mret = fwd.reindex(members).dropna()
        uret = fwd.reindex(uni_syms).dropna()
        i0 = idx[idx["date"] == start]["close"].iloc[0]
        i1 = idx[idx["date"] <= end]["close"].iloc[-1]
        nifty = (i1 / i0 - 1) * 100
        return {
            "as_of": str(start.date()), "end": str(end.date()), "horizon_days": int((end - start).days),
            "n_members": int(len(mret)),
            "screen_return": round(float(mret.mean()), 2) if len(mret) else None,
            "screen_median_return": round(float(mret.median()), 2) if len(mret) else None,
            "nifty50_return": round(float(nifty), 2),
            "universe_ew_return": round(float(uret.mean()), 2),
            "hit_rate_vs_nifty": round(float((mret > nifty).mean() * 100), 1) if len(mret) else None,
            "best": _top(mret, True), "worst": _top(mret, False),
            "caveats": [
                "Fundamentals are point-in-time using SEBI filing deadlines (45d quarterly / 60d annual).",
                "Universe is today's Nifty 500 → survivorship bias flatters results.",
                "Share count / promoter holding / dividend yield are current values, not historical.",
                "Equal-weight, no transaction costs or slippage.",
            ],
        }


def _top(s: pd.Series, best: bool, k: int = 3) -> List[dict]:
    s = s.sort_values(ascending=not best).head(k)
    return [{"symbol": i, "return": round(float(v), 2)} for i, v in s.items()]


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _clean_v(v):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _num(v)
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    return v


def _clean(d: dict) -> dict:
    return {k: _clean_v(v) for k, v in d.items()}
