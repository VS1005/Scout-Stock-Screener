"""Field registry + concept glossary: the single source of truth.

Everything downstream reads from here:
  validator (allowed fields/ops/ranges) · compiler (column whitelist) · LLM prompt (field catalogue)
  fast-path parser (synonyms) · explainer (labels/units/formatting) · UI (editable chips).
Adding a screenable field = one entry here + one column in snapshot.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

NUM, BOOL, CAT = "number", "bool", "category"


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str
    group: str                      # fundamental | technical | market | profile
    unit: str = ""                  # %, x, ₹ Cr, pp, ₹ ...
    desc: str = ""
    synonyms: Tuple[str, ...] = ()
    vrange: Tuple[float, float] = (-1e12, 1e12)
    higher_is_better: Optional[bool] = None
    scale: float = 1.0              # typical spread, used to normalise "margin of safety"
    na_note: str = ""               # why a value may be missing


F = Field
FIELDS: Dict[str, Field] = {f.key: f for f in [
    # --- profile ---
    F("sector", "Sector", CAT, "profile", desc="NSE industry classification",
      synonyms=("industry group", "sector")),
    F("industry", "Industry", CAT, "profile", desc="Granular industry, e.g. 'Banks - Regional', 'Software'",
      synonyms=("sub-industry", "business")),
    F("cap_category", "Market-cap class", CAT, "market",
      desc="SEBI/AMFI class by market-cap rank: Large (1-100), Mid (101-250), Small (251+)",
      synonyms=("large cap", "mid cap", "small cap")),
    # --- market ---
    F("market_cap_cr", "Market cap", NUM, "market", "₹ Cr", "Market capitalisation",
      ("market cap", "mcap", "size", "market value"), (0, 1e8), True, 50000),
    F("close", "Price", NUM, "market", "₹", "Last close price", ("price", "share price", "ltp"), (0, 1e7), None, 1000),
    F("avg_turnover_cr_20d", "Avg daily turnover (20d)", NUM, "market", "₹ Cr",
      "Average traded value per day over 20 sessions - a liquidity measure",
      ("liquidity", "liquid", "turnover", "traded value"), (0, 1e6), True, 50),
    F("promoter_holding_pct", "Promoter holding", NUM, "market", "%", "Promoter/insider shareholding",
      ("promoter holding", "promoter stake", "insider holding"), (0, 100), True, 20),
    F("dividend_yield", "Dividend yield", NUM, "market", "%", "Trailing dividend yield",
      ("dividend", "yield", "dividend paying"), (0, 100), True, 1.5),
    # --- valuation ---
    F("pe", "P/E", NUM, "fundamental", "x", "Price to TTM earnings (only for profitable companies)",
      ("pe", "p/e", "pe ratio", "price to earnings", "valuation", "cheap", "expensive"), (0, 10000), False, 20,
      "not meaningful when earnings are negative"),
    F("pb", "P/B", NUM, "fundamental", "x", "Price to book value", ("pb", "p/b", "price to book"),
      (0, 1000), False, 3),
    F("earnings_yield", "Earnings yield", NUM, "fundamental", "%", "TTM profit / market cap (inverse P/E)",
      ("earnings yield",), (-1000, 1000), True, 3),
    # --- profitability ---
    F("net_profit_ttm_cr", "Net profit (TTM)", NUM, "fundamental", "₹ Cr", "Trailing-12-month net profit",
      ("profit", "net profit", "earnings", "net income", "profitable"), (-1e7, 1e7), True, 1000),
    F("revenue_ttm_cr", "Revenue (TTM)", NUM, "fundamental", "₹ Cr", "Trailing-12-month revenue",
      ("revenue", "sales", "topline", "turnover of company"), (0, 1e8), True, 5000),
    F("net_margin_pct", "Net margin", NUM, "fundamental", "%", "TTM net profit / revenue",
      ("net margin", "profit margin", "margins"), (-1000, 1000), True, 8),
    F("operating_margin_pct", "Operating margin", NUM, "fundamental", "%", "Operating income / revenue (last FY)",
      ("operating margin", "ebit margin", "opm"), (-1000, 1000), True, 10),
    F("roe", "ROE", NUM, "fundamental", "%", "Return on (average) equity, last FY",
      ("roe", "return on equity"), (-1000, 1000), True, 8),
    F("roa", "ROA", NUM, "fundamental", "%", "Return on assets, last FY", ("roa", "return on assets"),
      (-1000, 1000), True, 4),
    F("cash_conversion", "Cash conversion", NUM, "fundamental", "x",
      "Operating cash flow / net profit - earnings quality (>1 means profits are backed by cash)",
      ("cash conversion", "earnings quality", "cash flow quality"), (-100, 100), True, 0.5),
    F("free_cash_flow_cr", "Free cash flow", NUM, "fundamental", "₹ Cr", "Free cash flow, last FY",
      ("free cash flow", "fcf", "cash generation"), (-1e7, 1e7), True, 1000),
    # --- growth ---
    F("revenue_growth_yoy", "Revenue growth (YoY)", NUM, "fundamental", "%", "Last FY revenue vs previous FY",
      ("revenue growth", "sales growth", "growing revenue", "topline growth"), (-100, 10000), True, 12),
    F("revenue_growth_qoq_yoy", "Latest-quarter revenue growth (YoY)", NUM, "fundamental", "%",
      "Latest quarter revenue vs same quarter last year", ("quarterly revenue growth", "recent sales growth"),
      (-100, 10000), True, 15),
    F("revenue_cagr_3y", "Revenue CAGR (3y)", NUM, "fundamental", "%", "3-year revenue compound annual growth",
      ("revenue cagr", "3 year growth", "long term growth"), (-100, 1000), True, 10),
    F("revenue_growth_streak", "Revenue growth streak", NUM, "fundamental", "yrs",
      "Consecutive years of rising revenue (max ~3 with available history)",
      ("increasing revenue", "consistent revenue growth", "revenue rising every year"), (0, 10), True, 1),
    F("profit_growth_yoy", "Profit growth (YoY)", NUM, "fundamental", "%", "Last FY net profit vs previous FY",
      ("profit growth", "earnings growth", "growing profits"), (-10000, 100000), True, 25),
    F("profit_growth_streak", "Profit growth streak", NUM, "fundamental", "yrs",
      "Consecutive years of rising net profit", ("increasing profit", "consistent profit growth"), (0, 10), True, 1),
    F("eps_growth_yoy", "EPS growth (YoY)", NUM, "fundamental", "%", "Diluted EPS growth, last FY",
      ("eps growth",), (-10000, 100000), True, 25),
    # --- balance sheet ---
    F("debt_to_equity", "Debt / Equity", NUM, "fundamental", "x", "Total debt / shareholders' equity",
      ("debt", "leverage", "d/e", "debt to equity", "debt free", "low debt"), (0, 1000), False, 0.5,
      "not meaningful for banks/NBFCs (Financial Services)"),
    F("interest_coverage", "Interest coverage", NUM, "fundamental", "x", "EBIT / interest expense (capped at 100)",
      ("interest coverage", "icr"), (-1000, 100), True, 5, "not meaningful for banks/NBFCs"),
    # --- technical ---
    F("ret_1w", "1-week return", NUM, "technical", "%", "", ("1 week return", "this week"), (-100, 1000), True, 4),
    F("ret_1m", "1-month return", NUM, "technical", "%", "", ("1 month return", "this month"), (-100, 1000), True, 8),
    F("ret_3m", "3-month return", NUM, "technical", "%", "", ("3 month return", "quarter return"), (-100, 1000), True, 12),
    F("ret_6m", "6-month return", NUM, "technical", "%", "", ("6 month return",), (-100, 10000), True, 18),
    F("ret_1y", "1-year return", NUM, "technical", "%", "", ("1 year return", "annual return", "12 month return"),
      (-100, 10000), True, 25),
    F("rel_strength_6m", "Relative strength vs Nifty 500 (6m)", NUM, "technical", "pp",
      "6-month return minus Nifty 500's 6-month return", ("outperforming", "relative strength", "beating the market"),
      (-1000, 10000), True, 15),
    F("rsi_14", "RSI (14)", NUM, "technical", "", "Wilder RSI; <30 oversold, >70 overbought",
      ("rsi", "oversold", "overbought"), (0, 100), None, 10),
    F("price_vs_sma200_pct", "Price vs 200-DMA", NUM, "technical", "%", "Distance of price above the 200-day average",
      ("200 dma distance",), (-100, 1000), None, 10),
    F("above_sma_200", "Above 200-DMA", BOOL, "technical", desc="Price above its 200-day moving average (long-term uptrend)",
      synonyms=("uptrend", "above 200 dma", "long term uptrend")),
    F("above_sma_50", "Above 50-DMA", BOOL, "technical", desc="Price above its 50-day moving average",
      synonyms=("above 50 dma", "short term uptrend")),
    F("golden_cross_20d", "Golden cross (last 20 sessions)", BOOL, "technical",
      desc="50-DMA crossed above 200-DMA within the last 20 sessions", synonyms=("golden cross",)),
    F("pct_from_52w_high", "Distance from 52-week high", NUM, "technical", "%",
      "0 = at the high, -20 = 20% below it", ("52 week high", "near high", "breakout"), (-100, 0), None, 10),
    F("pct_from_52w_low", "Distance from 52-week low", NUM, "technical", "%", "% above the 52-week low",
      ("52 week low", "near low"), (0, 10000), None, 20),
    F("volatility_1y", "Volatility (1y, annualised)", NUM, "technical", "%", "Std-dev of daily returns × √252",
      ("volatility", "risky", "stable", "low risk"), (0, 1000), False, 10),
    F("beta_1y", "Beta vs Nifty 50", NUM, "technical", "", "Sensitivity to the market (1 = moves with Nifty)",
      ("beta", "defensive", "high beta"), (-10, 10), None, 0.3),
    F("max_drawdown_1y", "Max drawdown (1y)", NUM, "technical", "%", "Worst peak-to-trough fall in a year",
      ("drawdown",), (-100, 0), True, 10),
    F("volume_ratio_20d", "Volume vs 20-day avg", NUM, "technical", "x", "Today's volume / 20-day average",
      ("volume spike", "unusual volume", "volume surge"), (0, 1000), None, 1),
    F("delivery_pct_20d", "Delivery % (20d avg)", NUM, "technical", "%",
      "Share of traded volume actually delivered (NSE) - high = investors, not intraday traders",
      ("delivery", "delivery percentage", "investor buying"), (0, 100), True, 12),
    F("delivery_trend", "Delivery % trend", NUM, "technical", "pp",
      "20-day minus 60-day average delivery % - rising = accumulation",
      ("accumulation", "smart money", "rising delivery"), (-100, 100), True, 5),
]}

UNIVERSES = {"NIFTY50": "in_nifty50", "NIFTY100": "in_nifty100", "NIFTY200": "in_nifty200", "NIFTY500": "in_nifty500"}


@dataclass(frozen=True)
class Concept:
    """A vague, plain-English idea pinned to an explicit, explainable definition."""
    phrases: Tuple[str, ...]
    conditions: Tuple[dict, ...]
    rationale: str


# Deterministic definitions of fuzzy terms. Used by the fast-path parser *and* injected into the
# LLM prompt so "low debt" means the same thing whether or not an LLM was involved.
CONCEPTS: List[Concept] = [
    Concept(("profitable", "making profit", "makes money", "profit making"),
            ({"field": "net_profit_ttm_cr", "op": ">", "value": 0},), "TTM net profit above zero"),
    Concept(("loss making", "unprofitable"),
            ({"field": "net_profit_ttm_cr", "op": "<", "value": 0},), "TTM net profit below zero"),
    Concept(("low debt", "little debt", "less debt", "low leverage", "conservatively financed"),
            ({"field": "debt_to_equity", "op": "<", "value": 0.5},), "D/E under 0.5 is conventionally low"),
    Concept(("debt free", "zero debt", "no debt"),
            ({"field": "debt_to_equity", "op": "<", "value": 0.05},), "D/E under 0.05 ≈ practically debt free"),
    Concept(("increasing revenue", "growing revenue", "rising revenue", "revenue growth", "growing sales",
             "increasing sales", "rising sales"),
            ({"field": "revenue_growth_streak", "op": ">=", "value": 2},),
            "revenue rose in each of the last 2+ financial years (not just one lucky year)"),
    Concept(("increasing profit", "growing profit", "rising profit", "growing earnings", "earnings growth"),
            ({"field": "profit_growth_streak", "op": ">=", "value": 2},), "net profit rose 2+ years in a row"),
    Concept(("high growth", "fast growing", "high-growth"),
            ({"field": "revenue_cagr_3y", "op": ">=", "value": 15},), "3-year revenue CAGR of at least 15%"),
    Concept(("high roe", "good roe", "efficient", "high return on equity"),
            ({"field": "roe", "op": ">=", "value": 15},), "ROE of 15%+ is a common quality bar"),
    Concept(("undervalued", "cheap", "value stocks", "inexpensive", "low valuation", "attractively valued"),
            ({"field": "pe", "op": "<", "value": {"ref": "sector_median"}},),
            "P/E below its own sector's median - valuation is only meaningful relative to peers"),
    Concept(("expensive", "overvalued"),
            ({"field": "pe", "op": ">", "value": {"ref": "sector_median"}},), "P/E above sector median"),
    Concept(("high quality", "quality companies", "quality stocks", "quality"),
            ({"field": "roe", "op": ">=", "value": 15}, {"field": "debt_to_equity", "op": "<", "value": 0.5},
             {"field": "cash_conversion", "op": ">=", "value": 0.8}),
            "high ROE, low debt, and profits backed by operating cash flow"),
    Concept(("oversold",), ({"field": "rsi_14", "op": "<", "value": 30},), "RSI(14) below 30"),
    Concept(("overbought",), ({"field": "rsi_14", "op": ">", "value": 70},), "RSI(14) above 70"),
    Concept(("uptrend", "in an uptrend", "trending up", "bullish trend"),
            ({"field": "above_sma_200", "op": "is", "value": True},), "price above its 200-day average"),
    Concept(("momentum", "strong momentum", "outperforming", "market leaders"),
            ({"field": "rel_strength_6m", "op": ">", "value": 0}, {"field": "above_sma_200", "op": "is", "value": True}),
            "beating Nifty 500 over 6 months while in a long-term uptrend"),
    Concept(("near 52 week high", "near 52-week high", "near their high", "near highs", "close to 52 week high"),
            ({"field": "pct_from_52w_high", "op": ">=", "value": -5},), "within 5% of the 52-week high"),
    Concept(("near 52 week low", "near 52-week low", "near lows"),
            ({"field": "pct_from_52w_low", "op": "<=", "value": 10},), "within 10% of the 52-week low"),
    Concept(("dividend paying", "pays dividends", "dividend payers", "good dividend", "high dividend"),
            ({"field": "dividend_yield", "op": ">=", "value": 2},), "dividend yield of 2%+"),
    Concept(("low volatility", "stable stocks", "less volatile", "low risk", "defensive"),
            ({"field": "volatility_1y", "op": "<", "value": {"ref": "universe_pctile", "p": 0.33}},),
            "in the least volatile third of the universe"),
    Concept(("accumulation", "being accumulated", "smart money buying", "rising delivery"),
            ({"field": "delivery_trend", "op": ">", "value": 3}, {"field": "delivery_pct_20d", "op": ">", "value": 45}),
            "delivery % rising (20d vs 60d) and above 45% - real investors taking delivery"),
    Concept(("volume spike", "unusual volume", "volume surge"),
            ({"field": "volume_ratio_20d", "op": ">=", "value": 2},), "volume at least 2× its 20-day average"),
    Concept(("liquid", "highly liquid", "easily tradable"),
            ({"field": "avg_turnover_cr_20d", "op": ">=", "value": 10},), "₹10 Cr+ traded per day on average"),
    Concept(("high promoter holding", "promoter backed", "high promoter stake"),
            ({"field": "promoter_holding_pct", "op": ">=", "value": 50},), "promoters hold 50%+"),
    Concept(("good cash flow", "cash generating", "positive cash flow", "positive free cash flow"),
            ({"field": "free_cash_flow_cr", "op": ">", "value": 0},), "positive free cash flow last FY"),
    Concept(("large caps", "large cap", "large-cap", "largecap", "bluechips", "bluechip", "blue chips", "blue chip"),
            ({"field": "cap_category", "op": "in", "value": ["Large"]},), "SEBI large-cap: top 100 by market cap"),
    Concept(("mid caps", "mid cap", "mid-cap", "midcaps", "midcap"),
            ({"field": "cap_category", "op": "in", "value": ["Mid"]},), "SEBI mid-cap: rank 101-250"),
    Concept(("small caps", "small cap", "small-cap", "smallcaps", "smallcap"),
            ({"field": "cap_category", "op": "in", "value": ["Small"]},), "SEBI small-cap: rank 251+"),
]

# Plain-language sector aliases → NSE sector names (validated against the live snapshot at runtime)
SECTOR_ALIASES = {
    "bank": ("industry", "Bank"), "banks": ("industry", "Bank"), "banking": ("industry", "Bank"),
    "nbfc": ("industry", "Credit Services"), "insurance": ("industry", "Insurance"),
    "it": ("sector", "Information Technology"), "tech": ("sector", "Information Technology"),
    "software": ("sector", "Information Technology"), "pharma": ("sector", "Healthcare"),
    "healthcare": ("sector", "Healthcare"), "auto": ("sector", "Automobile and Auto Components"),
    "fmcg": ("sector", "Fast Moving Consumer Goods"), "metal": ("sector", "Metals & Mining"),
    "metals": ("sector", "Metals & Mining"), "power": ("sector", "Power"), "energy": ("sector", "Oil Gas & Consumable Fuels"),
    "oil and gas": ("sector", "Oil Gas & Consumable Fuels"), "chemicals": ("sector", "Chemicals"),
    "cement": ("sector", "Construction Materials"), "realty": ("sector", "Realty"), "real estate": ("sector", "Realty"),
    "telecom": ("sector", "Telecommunication"), "capital goods": ("sector", "Capital Goods"),
    "financials": ("sector", "Financial Services"), "financial": ("sector", "Financial Services"),
    "consumer durables": ("sector", "Consumer Durables"), "textiles": ("sector", "Textiles"),
    "media": ("sector", "Media Entertainment & Publication"), "construction": ("sector", "Construction"),
}


# Only fields whose meaning isn't obvious from key + label carry a description into the prompt.
# (Free-tier LLM quotas are per-token: every line here is paid on every uncached call.)
_PROMPT_DESC = {"cap_category", "industry", "revenue_growth_streak", "profit_growth_streak", "cash_conversion",
                "rel_strength_6m", "pct_from_52w_high", "pct_from_52w_low", "delivery_pct_20d", "delivery_trend",
                "golden_cross_20d", "volume_ratio_20d", "price_vs_sma200_pct", "revenue_growth_qoq_yoy"}


def catalogue_for_prompt() -> str:
    """Compact field catalogue for the LLM system prompt: `key [kind, unit]: label (- desc)`."""
    lines = []
    for f in FIELDS.values():
        kind = {"number": "num", "bool": "bool", "category": "cat"}[f.kind]
        unit = f", {f.unit}" if f.unit else ""
        desc = f" - {f.desc}" if f.key in _PROMPT_DESC and f.desc else ""
        lines.append(f"{f.key} [{kind}{unit}]: {f.label}{desc}")
    return "\n".join(lines)


def concepts_for_prompt() -> str:
    out = []
    for c in CONCEPTS:
        conds = " AND ".join(f"{d['field']} {d['op']} {d['value']}" for d in c.conditions)
        out.append(f"'{c.phrases[0]}' => {conds}")
    return "\n".join(out)
