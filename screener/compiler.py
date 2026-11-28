"""Compile a validated Screen AST into ONE parameterised DuckDB query.

Security properties
  * identifiers come only from the registry whitelist (never from user/LLM text);
  * every user-supplied value is a bound parameter (`?`), except percentile p which is a
    pydantic-validated float in [0,1] formatted by us;
  * the query runs on an in-memory DB that has external access disabled (see engine.py).

Instead of `WHERE a AND b AND c`, every criterion is projected as its own boolean column
(__c0, __c1, ...). One scan therefore yields matches, near-misses (fail exactly one), and
per-criterion explanations, with no extra queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List

from screener.dsl import AnyGroup, Condition, RefValue, Screen
from screener.registry import FIELDS, UNIVERSES

BASE_COLS = ["symbol", "name", "sector", "industry", "cap_category", "market_cap_cr", "close"]


def _ident(name: str) -> str:
    if name not in FIELDS and name not in BASE_COLS and name not in UNIVERSES.values():
        raise ValueError(f"identifier not whitelisted: {name}")
    return f'"{name}"'


@dataclass
class Compiled:
    sql: str
    params: List[Any]
    criteria: List[dict]                  # [{id, kind: leaf|any, leaves: [(col, Condition, refcol)]}]
    fields: List[str] = field(default_factory=list)


def _ref_expr(c: Condition, v: RefValue) -> str:
    col = _ident(c.field)
    if v.ref == "field":
        return _ident(v.field)
    part = "PARTITION BY sector" if v.ref.startswith("sector") else ""
    if v.ref.endswith("median"):
        return f"median({col}) OVER ({part})"
    p = f"{float(v.p):.4f}"  # validated float in [0,1]
    return f"quantile_cont({col}, {p}) OVER ({part})"


def _leaf_expr(c: Condition, refcol: str, params: List[Any]) -> str:
    col = _ident(c.field)
    v = c.value
    if isinstance(v, RefValue):
        return f"({col} {c.op} {refcol})"
    if c.op == "between":
        params.extend(v)
        return f"({col} BETWEEN ? AND ?)"
    if c.op in (">", ">=", "<", "<="):
        params.append(float(v))
        return f"({col} {c.op} ?)"
    if c.op == "is":
        params.append(bool(v))
        return f"({col} = ?)"
    if c.op in ("in", "not_in"):
        params.extend(v)
        ph = ", ".join("?" for _ in v)
        return f"({col} {'NOT ' if c.op == 'not_in' else ''}IN ({ph}))"
    if c.op == "contains":
        escaped = str(v).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params.append(f"%{escaped}%")
        return f"({col} ILIKE ? ESCAPE '\\')"
    raise ValueError(f"unsupported op {c.op}")


def compile_screen(screen: Screen, table: str = "snapshot") -> Compiled:
    if table not in ("snapshot", "bt_snapshot"):
        raise ValueError("unknown table")
    params: List[Any] = []
    ref_selects, pass_selects, criteria = [], [], []
    used = list(dict.fromkeys(
        BASE_COLS + [l.field for l in screen.leaves()]
        + [l.value.field for l in screen.leaves() if isinstance(l.value, RefValue) and l.value.field]
        + ([screen.sort.field] if screen.sort and screen.sort.field in FIELDS else [])))

    # Reference values (sector medians, percentiles) are computed over the FULL Nifty-500 so that
    # "cheaper than sector peers" is stable regardless of the chosen sub-universe.
    for i, crit in enumerate(screen.all):
        leaves = crit.any if isinstance(crit, AnyGroup) else [crit]
        entry = {"id": f"__c{i}", "kind": "any" if isinstance(crit, AnyGroup) else "leaf", "leaves": []}
        for j, leaf in enumerate(leaves):
            refcol = f"__r{i}_{j}"
            if isinstance(leaf.value, RefValue):
                ref_selects.append(f"{_ref_expr(leaf, leaf.value)} AS {refcol}")
            entry["leaves"].append((f"__c{i}_{j}", leaf, refcol if isinstance(leaf.value, RefValue) else None))
        criteria.append(entry)

    # pass columns (params appended in SELECT order — must match placeholder order)
    for entry in criteria:
        exprs = []
        for col, leaf, refcol in entry["leaves"]:
            e = f"COALESCE({_leaf_expr(leaf, refcol, params)}, FALSE)"
            pass_selects.append(f"{e} AS {col}")
            exprs.append(col)
        pass_selects.append(f"({' OR '.join(exprs)}) AS {entry['id']}")

    uni_col = _ident(UNIVERSES[screen.universe])
    cols = ", ".join(_ident(c) for c in used)
    base = f"SELECT {cols}, {uni_col} AS __in_universe" + (", " + ", ".join(ref_selects) if ref_selects else "")
    sql = (
        f"WITH base AS ({base} FROM {table})\n"
        f"SELECT *{', ' + ', '.join(pass_selects) if pass_selects else ''}\n"
        f"FROM base WHERE __in_universe"
    )
    return Compiled(sql=sql, params=params, criteria=criteria, fields=used)
