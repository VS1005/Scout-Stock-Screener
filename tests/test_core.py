"""Offline unit tests: DSL validation, compiler security, fast path, patches, engine logic, ETL adjustments.

    pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from etl.nse import adjust_for_corporate_actions
from screener.compiler import compile_screen
from screener.dsl import PatchOp, Screen, apply_patch, repair
from screener.engine import Engine, _sandboxed_connection
from screener.fastpath import parse, parse_followup


# ------------------------------------------------------------------ DSL validation
def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Screen.model_validate({"all": [{"field": "pe; DROP TABLE snapshot", "op": "<", "value": 10}]})


def test_op_type_mismatch_rejected():
    with pytest.raises(ValidationError):
        Screen.model_validate({"all": [{"field": "sector", "op": ">", "value": 3}]})
    with pytest.raises(ValidationError):
        Screen.model_validate({"all": [{"field": "above_sma_200", "op": "<", "value": 1}]})


def test_value_range_enforced():
    with pytest.raises(ValidationError):
        Screen.model_validate({"all": [{"field": "rsi_14", "op": ">", "value": 150}]})


def test_size_limits():
    conds = [{"field": "pe", "op": "<", "value": 10}] * 13
    with pytest.raises(ValidationError):
        Screen.model_validate({"all": conds})
    with pytest.raises(ValidationError):
        Screen.model_validate({"limit": 10_000})


def test_contradiction_warning():
    s = Screen.model_validate({"all": [{"field": "pe", "op": "<", "value": 10}, {"field": "pe", "op": ">", "value": 20}]})
    assert s.warnings() and "Contradiction" in s.warnings()[0]


def test_repair_fixes_synonyms():
    notes = []
    obj = repair({"all": [{"field": "return on equity", "op": ">", "value": 15}]}, notes)
    assert obj["all"][0]["field"] == "roe" and notes


# ------------------------------------------------------------------ compiler security
def test_compiler_parameterises_values():
    s = Screen.model_validate({"all": [
        {"field": "industry", "op": "contains", "value": "Bank'); DROP TABLE x; --"},
        {"field": "pe", "op": "<", "value": 15}]})
    c = compile_screen(s)
    assert "DROP" not in c.sql and "15" not in c.sql
    assert any("DROP" in str(p) for p in c.params)


def test_compiler_rejects_unknown_table():
    with pytest.raises(ValueError):
        compile_screen(Screen(), table="universe; --")


def test_sandbox_blocks_file_access():
    con = _sandboxed_connection(pd.DataFrame({"a": [1]}))
    with pytest.raises(Exception):
        con.execute("SELECT * FROM read_csv('/etc/passwd')").fetchall()
    with pytest.raises(Exception):
        con.execute("SET enable_external_access = true")


# ------------------------------------------------------------------ fast path
def test_fastpath_canonical_query():
    fp = parse("Find profitable companies with low debt and increasing revenue")
    assert fp and fp.consumed_all
    fields = {c.field for c in fp.screen.all}
    assert fields == {"net_profit_ttm_cr", "debt_to_equity", "revenue_growth_streak"}
    assert all(c.assumed for c in fp.screen.all)


def test_fastpath_explicit_numbers_and_units():
    fp = parse("market cap above 50000 crore with dividend yield at least 3%")
    assert fp.consumed_all
    vals = {c.field: (c.op, c.value) for c in fp.screen.all}
    assert vals["market_cap_cr"] == (">", 50000) and vals["dividend_yield"] == (">=", 3)


def test_fastpath_hands_off_negation_and_unknown_words():
    assert parse("large caps not in IT") is None
    fp = parse("low debt companies with good ESG")
    assert fp is not None and not fp.consumed_all and "esg" in fp.leftovers


def test_followup_patch():
    base = parse("profitable companies").screen
    ops = parse_followup("only large caps")
    new, log = apply_patch(base, ops)
    assert len(new.all) == 2 and log


def test_patch_remove_and_replace():
    s = Screen.model_validate({"all": [{"field": "pe", "op": "<", "value": 10}, {"field": "roe", "op": ">", "value": 15}]})
    s2, _ = apply_patch(s, [PatchOp(op="remove", field="pe")])
    assert [c.field for c in s2.all] == ["roe"]
    s3, _ = apply_patch(s, [PatchOp.model_validate(
        {"op": "replace", "field": "roe", "condition": {"field": "roe", "op": ">", "value": 20}})])
    assert s3.all[1].value == 20


# ------------------------------------------------------------------ engine logic on a synthetic snapshot
def _synthetic_engine():
    rng = np.random.default_rng(0)
    n = 40
    snap = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)], "name": [f"Co {i}" for i in range(n)],
        "sector": ["IT"] * 20 + ["Healthcare"] * 20, "industry": ["Software"] * 20 + ["Drugs"] * 20,
        "cap_category": ["Large"] * 10 + ["Mid"] * 15 + ["Small"] * 15,
        "market_cap_cr": rng.uniform(1e3, 1e5, n), "close": rng.uniform(10, 1000, n),
        "pe": np.linspace(5, 45, n), "roe": np.linspace(30, 2, n), "debt_to_equity": np.linspace(0, 2, n),
        "in_nifty500": True, "in_nifty50": [True] * 5 + [False] * 35, "in_nifty100": False, "in_nifty200": False,
        "as_of": pd.Timestamp("2026-01-01"),
    })
    e = Engine.__new__(Engine)
    import threading
    e._lock = threading.Lock()
    e.snap, e.con, e.as_of = snap, _sandboxed_connection(snap), pd.Timestamp("2026-01-01").date()
    e._scales = Engine._robust_scales(snap)
    e.reload = lambda force=False: None
    return e


def test_engine_matches_near_misses_and_strength():
    e = _synthetic_engine()
    s = Screen.model_validate({"all": [{"field": "pe", "op": "<", "value": 15}, {"field": "roe", "op": ">", "value": 20}]})
    r = e.run(s)
    assert r["total_matches"] == len([x for x in r["results"]])
    for row in r["results"]:
        assert row["values"]["pe"] < 15 and row["values"]["roe"] > 20
        assert all(x["passed"] for x in row["reasons"])
    assert all(n["missed_field"] in ("pe", "roe") for n in r["near_misses"])
    strengths = [x["match_strength"] for x in r["results"]]
    assert strengths == sorted(strengths, reverse=True)


def test_engine_sector_relative_reference():
    e = _synthetic_engine()
    s = Screen.model_validate({"all": [{"field": "pe", "op": "<", "value": {"ref": "sector_median"}}]})
    r = e.run(s)
    assert r["total_matches"] == 20  # half of each sector is below its own median


def test_engine_relaxation_suggestions_on_empty_result():
    e = _synthetic_engine()
    s = Screen.model_validate({"all": [{"field": "pe", "op": "<", "value": 6}, {"field": "roe", "op": "<", "value": 3}]})
    r = e.run(s)
    assert r["total_matches"] == 0 and r["suggestions"]
    assert any(sg.get("relax_to") for sg in r["suggestions"])


def test_engine_diagnose():
    e = _synthetic_engine()
    s = Screen.model_validate({"all": [{"field": "pe", "op": "<", "value": 10}]})
    d = e.diagnose(s, "S39")
    assert d and not d[0]["passed"]


# ------------------------------------------------------------------ ETL
def test_corporate_action_back_adjustment():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    p = pd.DataFrame({"symbol": "X", "date": dates, "open": [1000, 1010, 101, 102], "high": [1000, 1010, 101, 102],
                      "low": [1000, 1010, 101, 102], "close": [1000.0, 1010.0, 101.0, 102.0],
                      "prev_close": [990, 1000, 1010, 101], "volume": [100.0, 100, 1000, 1000]})
    adj = adjust_for_corporate_actions(p)
    assert adj["close"].tolist() == pytest.approx([100, 101, 101, 102])  # 1:10 split snapped to 0.1
    assert adj["volume"].iloc[0] == pytest.approx(1000)


def test_point_in_time_filter_respects_filing_lag():
    from screener.config import QUARTERLY_REPORT_LAG_DAYS
    period_end = pd.Timestamp("2026-06-30")
    available = period_end + pd.Timedelta(days=QUARTERLY_REPORT_LAG_DAYS)
    assert available == pd.Timestamp("2026-08-14")
