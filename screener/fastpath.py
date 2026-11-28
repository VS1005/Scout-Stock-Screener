"""Deterministic, zero-token parser for common phrasings.

Precision over recall: it only answers when EVERY meaningful word in the query is accounted for
by a known concept, field comparison, sector, universe or sort instruction. Anything with negation,
OR-logic or unexplained words is handed to the LLM. Same concept definitions as the LLM prompt, so
both paths interpret "low debt" identically.
"""
from __future__ import annotations

import re
from typing import List, Optional

from screener.dsl import Condition, PatchOp, Screen, Sort
from screener.registry import CONCEPTS, FIELDS, NUM, SECTOR_ALIASES

STOPWORDS = set("""
a an the and with that have has having are is which who whose of in on for to me show find list give get
stocks stock companies company shares share firms firm businesses names scrips equities please screen screener
all some any good strong solid nice best top i want looking look need would like can you also only just now
them those these their its it's sector sectors industry space from among within where by
""".split())
HAND_OFF = re.compile(r"\b(not|no|without|except|excluding|exclude|or|but|instead|remove|drop|unless|rather|"
                      r"compared|vs|versus|why|what|how|explain|similar|like)\b")
# longest unit first ("crore" before "cr"), and a unit must not be the start of a longer word
NUMBER = r"(-?\d+(?:\.\d+)?)\s*(crores|crore|cr|percent|lakhs|lakh|%|x|k)?(?![a-z])"
CMP = {
    "below": "<", "under": "<", "less than": "<", "lower than": "<", "smaller than": "<", "<": "<", "<=": "<=",
    "at most": "<=", "upto": "<=", "up to": "<=", "max": "<=", "maximum": "<=",
    "above": ">", "over": ">", "more than": ">", "greater than": ">", "higher than": ">", "exceeding": ">",
    ">": ">", ">=": ">=", "at least": ">=", "min": ">=", "minimum": ">=",
}


def _syn_map():
    """Numeric-field names/synonyms → field key (synonyms never overlap concept phrases; see registry)."""
    m = {}
    for k, f in FIELDS.items():
        if f.kind == NUM:
            for s in (k.replace("_", " "), f.label.lower(), *f.synonyms):
                m.setdefault(s.lower(), k)
    return m


SYN = _syn_map()
_syn_alt = "|".join(sorted(map(re.escape, SYN), key=len, reverse=True))
_cmp_alt = "|".join(sorted(map(re.escape, CMP), key=len, reverse=True))
RX_CMP = re.compile(rf"\b({_syn_alt})\s*(?:is|of|ratio|value)?\s*({_cmp_alt})\s*{NUMBER}")
RX_BETWEEN = re.compile(rf"\b({_syn_alt})\s*(?:is|of)?\s*between\s*{NUMBER}\s*(?:and|-|to)\s*{NUMBER}")
RX_TOP = re.compile(r"\b(?:top|first)\s+(\d{1,3})\b")
RX_SORT = re.compile(rf"\b(?:sort(?:ed)?\s+by|order(?:ed)?\s+by|ranked\s+by|highest|lowest|most|least)\s+({_syn_alt})"
                     rf"(?:\s+(asc|ascending|desc|descending))?")
RX_UNIVERSE = re.compile(r"\bnifty\s*-?\s*(50|100|200|500)\b")


def _unit_scale(field: str, num: float, unit: Optional[str]) -> float:
    if unit in ("k",) and FIELDS[field].unit == "₹ Cr":
        return num * 1000
    if unit in ("lakh", "lakhs") and FIELDS[field].unit == "₹ Cr":
        return num / 100
    return num


def _normalise(q: str) -> str:
    q = q.lower().replace("’", "'").replace("–", "-")
    q = re.sub(r"(\d),(\d)", r"\1\2", q)
    q = re.sub(r"[^\w\s%<>=.\-/']", " ", q)
    return re.sub(r"\s+", " ", q).strip()


class FastParse:
    def __init__(self, screen: Screen, consumed_all: bool, leftovers: List[str]):
        self.screen, self.consumed_all, self.leftovers = screen, consumed_all, leftovers


def parse(query: str) -> Optional[FastParse]:
    q = _normalise(query)
    if not q or HAND_OFF.search(q):
        return None
    mask = list(q)
    conds: List[Condition] = []
    universe, sort, limit = "NIFTY500", None, 25

    def consume(m: re.Match):
        for i in range(m.start(), m.end()):
            mask[i] = " "

    for m in RX_BETWEEN.finditer(q):
        f = SYN[m.group(1)]
        lo = _unit_scale(f, float(m.group(2)), m.group(3))
        hi = _unit_scale(f, float(m.group(4)), m.group(5))
        conds.append(Condition(field=f, op="between", value=[lo, hi], source_phrase=m.group(0)))
        consume(m)
    for m in RX_CMP.finditer("".join(mask)):
        f = SYN[m.group(1)]
        conds.append(Condition(field=f, op=CMP[m.group(2)], value=_unit_scale(f, float(m.group(3)), m.group(4)),
                               source_phrase=m.group(0).strip()))
        consume(m)

    # concepts: longest phrase first
    phrases = sorted(((p, c) for c in CONCEPTS for p in c.phrases), key=lambda x: -len(x[0]))
    for phrase, concept in phrases:
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", "".join(mask)):
            for d in concept.conditions:
                conds.append(Condition(**d, source_phrase=phrase, assumed=True, rationale=concept.rationale))
            consume(m)

    m = RX_UNIVERSE.search("".join(mask))
    if m:
        universe = f"NIFTY{m.group(1)}"
        consume(m)
    m = RX_TOP.search("".join(mask))
    if m:
        limit = max(1, min(100, int(m.group(1))))
        consume(m)
    m = RX_SORT.search("".join(mask))
    if m:
        word = m.group(0).split()[0]
        d = "asc" if word in ("lowest", "least") or (m.group(2) or "").startswith("asc") else "desc"
        sort = Sort(field=SYN[m.group(1)], dir=d)
        consume(m)

    # sectors / industries (require context for the ambiguous alias "it")
    text = "".join(mask)
    sector_vals, industry_vals, said = [], [], []
    for alias in sorted(SECTOR_ALIASES, key=len, reverse=True):
        pat = (r"\bit\s+(?=stocks|companies|sector|firms|shares|services)" if alias == "it"
               else rf"\b{re.escape(alias)}\b")
        for m in re.finditer(pat, text):
            col, val = SECTOR_ALIASES[alias]
            if col == "sector":
                sector_vals.append(val)
                said.append(alias)
            else:
                industry_vals.append(val)
            text = text[:m.start()] + " " * (m.end() - m.start()) + text[m.end():]
    mask = list(text)
    if sector_vals:
        conds.append(Condition(field="sector", op="in", value=sorted(set(sector_vals)),
                               source_phrase=" / ".join(dict.fromkeys(said))))
    for v in sorted(set(industry_vals)):
        conds.append(Condition(field="industry", op="contains", value=v, source_phrase=v.lower()))

    leftovers = [w for w in re.findall(r"[a-z0-9%]+", "".join(mask)) if w not in STOPWORDS]
    if not conds and sort is None:
        return None
    # de-duplicate identical conditions (e.g. "profitable" + "quality" both adding the same rule)
    seen, uniq = set(), []
    for c in conds:
        key = (c.field, c.op, str(c.value))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    screen = Screen(universe=universe, all=uniq, sort=sort, limit=limit)
    return FastParse(screen, consumed_all=not leftovers, leftovers=leftovers)


REFINE_CUE = re.compile(r"^(only|also|and|add|plus|now|just|filter|with|then)\b")


def parse_followup(message: str) -> Optional[List[PatchOp]]:
    """Purely additive follow-ups ('only banks', 'also RSI below 40') → patch ops without the LLM."""
    q = _normalise(message)
    if not REFINE_CUE.match(q):
        return None
    fp = parse(q)
    if fp is None or not fp.consumed_all:
        return None
    ops = [PatchOp(op="add", condition=c) for c in fp.screen.all]
    if fp.screen.sort:
        ops.append(PatchOp(op="set_sort", sort=fp.screen.sort))
    if fp.screen.universe != "NIFTY500":
        ops.append(PatchOp(op="set_universe", universe=fp.screen.universe))
    if fp.screen.limit != 25:
        ops.append(PatchOp(op="set_limit", limit=fp.screen.limit))
    return ops or None
