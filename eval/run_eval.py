"""Evaluate NL → DSL translation against hand-labelled gold conditions.

    python -m eval.run_eval            # full pipeline (fast path + LLM; LLM answers are cached → reruns are free)
    python -m eval.run_eval --no-llm   # fast-path only: measures zero-token coverage & precision

Gold format per condition: "<field> <op> <value>" where value '*' = any threshold (vague terms),
category values are matched case-insensitively, and > / >= (and < / <=) count as the same direction.
Also: "sort <field> <dir>", "limit <n>", "universe <U>".
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from screener.dsl import AnyGroup, RefValue, Screen
from screener.engine import Engine
from screener.fastpath import parse as fast_parse
from screener.llm import LLM
from screener.parser import ParseError, Parser
from screener.store import Store

DIR = {">": "gt", ">=": "gt", "<": "lt", "<=": "lt"}


def leaf_matches(gold: str, leaf) -> bool:
    f, op, *rest = gold.split(" ", 2)
    val = rest[0] if rest else "*"
    if leaf.field != f:
        return False
    if op in DIR:
        if DIR.get(leaf.op) != DIR[op]:
            return False
        if val == "*":
            return True
        return not isinstance(leaf.value, RefValue) and abs(float(leaf.value) - float(val)) <= 1e-6 + 0.01 * abs(float(val))
    if op != leaf.op:
        return False
    if val == "*":
        return True
    if op == "is":
        return str(leaf.value).lower() == val
    if op == "contains":
        return val.lower() in str(leaf.value).lower()
    vals = [v.lower() for v in (leaf.value if isinstance(leaf.value, list) else [leaf.value])]
    return val.lower() in vals


def score(screen: Screen, gold: list) -> tuple:
    leaves = []
    for c in screen.all:
        leaves.extend(c.any if isinstance(c, AnyGroup) else [c])
    extras = []
    if screen.sort:
        extras.append(f"sort {screen.sort.field} {screen.sort.dir}")
    if screen.limit != 25:
        extras.append(f"limit {screen.limit}")
    if screen.universe != "NIFTY500":
        extras.append(f"universe {screen.universe}")
    g_cond = [g for g in gold if not g.startswith(("sort ", "limit ", "universe "))]
    g_meta = [g for g in gold if g.startswith(("sort ", "limit ", "universe "))]
    used, tp = set(), 0
    for g in g_cond:
        for i, l in enumerate(leaves):
            if i not in used and leaf_matches(g, l):
                used.add(i)
                tp += 1
                break
    tp_meta = sum(1 for g in g_meta if g in extras)
    n_pred = len(leaves) + len(extras)
    n_gold = len(gold)
    return tp + tp_meta, n_pred, n_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    items = [json.loads(l) for l in Path(__file__).with_name("queries.jsonl").read_text().splitlines() if l.strip()]
    engine = Engine()
    parser = Parser(LLM(Store(), engine.categories()["sector"]))
    tot = defaultdict(float)
    by_cat = defaultdict(lambda: [0, 0, 0])
    rows = []
    for it in items:
        t0 = time.perf_counter()
        path, screen, unmapped, rejected = "-", None, [], False
        try:
            if a.no_llm:
                fp = fast_parse(it["q"])
                if fp and fp.consumed_all:
                    screen, path = fp.screen, "fastpath"
                else:
                    path = "handoff"
            else:
                pr = parser.parse(it["q"])
                screen, path, unmapped = pr.screen, pr.path, pr.unmapped
                rejected = pr.action in ("unsupported", "clarify")
                tot["tokens"] += pr.tokens
        except ParseError:
            rejected, path = True, "rejected"
        ms = (time.perf_counter() - t0) * 1000
        tot["n"] += 1
        tot[f"path_{path}"] += 1
        if it.get("reject"):
            ok = rejected or screen is None or not screen.all
            tot["reject_total"] += 1
            tot["reject_ok"] += ok
            rows.append((it["id"], it["cat"], path, "REJECTED OK" if ok else "SHOULD REJECT", ms))
            continue
        if it.get("safe_either") and (rejected or screen is None):
            tot["reject_total"] += 1
            tot["reject_ok"] += 1
            rows.append((it["id"], it["cat"], path, "REFUSED (safe)", ms))
            continue
        if path == "handoff":
            rows.append((it["id"], it["cat"], path, "-", ms))
            continue
        tp, n_pred, n_gold = score(screen, it["gold"]) if screen else (0, 0, len(it["gold"]))
        tot["tp"] += tp
        tot["pred"] += n_pred
        tot["gold"] += n_gold
        c = by_cat[it["cat"]]
        c[0] += tp
        c[1] += n_pred
        c[2] += n_gold
        if it.get("unmapped"):
            tot["unmapped_total"] += 1
            tot["unmapped_ok"] += bool(unmapped)
        rows.append((it["id"], it["cat"], path, f"{tp}/{n_gold} gold, {n_pred} predicted", ms))

    print(f"{'id':>3} {'category':12} {'path':9} {'result':32} {'ms':>7}")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:12} {r[2]:9} {r[3]:32} {r[4]:7.0f}")
    p = tot["tp"] / tot["pred"] if tot["pred"] else 0
    rcl = tot["tp"] / tot["gold"] if tot["gold"] else 0
    print("\n=== Summary ===")
    print(f"condition precision {p:.1%} · recall {rcl:.1%} · F1 {2 * p * rcl / (p + rcl) if p + rcl else 0:.1%}"
          f"  (over answered queries)")
    for k in sorted(k for k in tot if k.startswith("path_")):
        print(f"{k[5:]:>10}: {int(tot[k])}/{int(tot['n'])} queries")
    if tot["reject_total"]:
        print(f"off-topic/adversarial correctly refused: {int(tot['reject_ok'])}/{int(tot['reject_total'])}")
    if tot["unmapped_total"]:
        print(f"unmappable concepts surfaced to user: {int(tot['unmapped_ok'])}/{int(tot['unmapped_total'])}")
    print(f"LLM tokens used this run: {int(tot['tokens']):,}")
    print("\nper category (precision / recall):")
    for cat, (tp, pr_, g) in by_cat.items():
        print(f"  {cat:12} {tp / pr_ if pr_ else 0:6.1%} / {tp / g if g else 0:6.1%}")


if __name__ == "__main__":
    main()
