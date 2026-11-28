"""Screen DSL: the only contract between language understanding and data access.

The LLM (or fast-path parser) produces this JSON; nothing else ever reaches the database.
Validation guarantees: whitelisted fields, operator/type compatibility, sane value ranges,
bounded size. Patches let a conversation refine a screen without regenerating it.
"""
from __future__ import annotations

import copy
import difflib
from typing import Any, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from screener.config import MAX_ANY_GROUP_SIZE, MAX_CONDITIONS, MAX_LIMIT
from screener.registry import BOOL, CAT, FIELDS, NUM, UNIVERSES

NUM_OPS = {">", ">=", "<", "<=", "between"}
CAT_OPS = {"in", "not_in", "contains"}
BOOL_OPS = {"is"}
Op = Literal[">", ">=", "<", "<=", "between", "in", "not_in", "contains", "is"]


class RefValue(BaseModel):
    """A threshold defined relative to peers or another field instead of a constant."""
    model_config = ConfigDict(extra="forbid")
    ref: Literal["sector_median", "universe_median", "sector_pctile", "universe_pctile", "field"]
    p: Optional[float] = Field(None, ge=0.0, le=1.0, description="percentile for *_pctile refs (0.8 = top 20% cutoff)")
    field: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.ref.endswith("pctile") and self.p is None:
            raise ValueError(f"ref '{self.ref}' needs p between 0 and 1")
        if self.ref == "field":
            if self.field not in FIELDS or FIELDS[self.field].kind != NUM:
                raise ValueError(f"ref field '{self.field}' is not a numeric screenable field")
        return self


class Condition(BaseModel):
    model_config = ConfigDict(extra="ignore")
    field: str
    op: Op
    value: Union[bool, float, RefValue, List[float], List[str], str]
    source_phrase: Optional[str] = Field(None, description="the user's words this came from")
    assumed: bool = Field(False, description="threshold chosen by the system, not stated by the user")
    rationale: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.field not in FIELDS:
            raise ValueError(f"unknown field '{self.field}'")
        f = FIELDS[self.field]
        v = self.value
        if f.kind == NUM:
            if self.op not in NUM_OPS:
                raise ValueError(f"op '{self.op}' not allowed on numeric field {self.field}")
            if self.op == "between":
                if not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
                    raise ValueError("'between' needs [low, high]")
                lo, hi = sorted(float(x) for x in v)
                self.value = [lo, hi]
                vals = [lo, hi]
            elif isinstance(v, RefValue):
                if v.ref == "field" and FIELDS[v.field].unit != f.unit:  # e.g. P/E (x) vs ROE (%)
                    raise ValueError(f"cannot compare {self.field} with {v.field}: different units")
                vals = []
            elif isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{self.field} needs a number or a reference value")
            else:
                self.value = float(v)
                vals = [float(v)]
            for x in vals:
                if not (f.vrange[0] <= x <= f.vrange[1]):
                    raise ValueError(f"{x} is outside the valid range {f.vrange} for {self.field}")
        elif f.kind == BOOL:
            if self.op != "is" or not isinstance(v, bool):
                raise ValueError(f"{self.field} is a yes/no field: use op 'is' with true/false")
        elif f.kind == CAT:
            if self.op not in CAT_OPS:
                raise ValueError(f"op '{self.op}' not allowed on category field {self.field}")
            if self.op == "contains":
                if not isinstance(v, str) or not (1 <= len(v) <= 40):
                    raise ValueError("'contains' needs a short text value")
            else:
                if isinstance(v, str):
                    self.value = [v]
                if not (isinstance(self.value, list) and 1 <= len(self.value) <= 20
                        and all(isinstance(x, str) and len(x) <= 60 for x in self.value)):
                    raise ValueError(f"'{self.op}' needs a list of 1-20 category names")
        return self


class AnyGroup(BaseModel):
    """OR-group: passes if any member condition passes."""
    model_config = ConfigDict(extra="forbid")
    any: List[Condition] = Field(..., min_length=2, max_length=MAX_ANY_GROUP_SIZE)
    source_phrase: Optional[str] = None


class Sort(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    dir: Literal["asc", "desc"] = "desc"

    @field_validator("field")
    @classmethod
    def _known(cls, v):
        if v != "match_strength" and (v not in FIELDS or FIELDS[v].kind != NUM):
            raise ValueError(f"cannot sort by '{v}'")
        return v


Criterion = Union[Condition, AnyGroup]


class Screen(BaseModel):
    model_config = ConfigDict(extra="forbid")
    universe: Literal["NIFTY50", "NIFTY100", "NIFTY200", "NIFTY500"] = "NIFTY500"
    all: List[Criterion] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    sort: Optional[Sort] = None
    limit: int = Field(25, ge=1, le=MAX_LIMIT)

    def leaves(self) -> List[Condition]:
        out = []
        for c in self.all:
            out.extend(c.any if isinstance(c, AnyGroup) else [c])
        return out

    def warnings(self) -> List[str]:
        """Detect logically impossible combinations on the same field (e.g. P/E < 10 AND P/E > 20)."""
        lo, hi, out = {}, {}, []
        for c in self.all:
            if not isinstance(c, Condition) or not isinstance(c.value, (int, float)) or isinstance(c.value, bool):
                continue
            if c.op in (">", ">="):
                lo[c.field] = max(lo.get(c.field, -1e18), c.value)
            elif c.op in ("<", "<="):
                hi[c.field] = min(hi.get(c.field, 1e18), c.value)
        for k in lo.keys() & hi.keys():
            if lo[k] >= hi[k]:
                out.append(f"Contradiction: {FIELDS[k].label} must be > {lo[k]:g} and < {hi[k]:g} - nothing can match.")
        return out


assert set(UNIVERSES) == {"NIFTY50", "NIFTY100", "NIFTY200", "NIFTY500"}

# ---------------------------------------------------------------- repair (LLM robustness)
_SYN = {s: k for k, f in FIELDS.items() for s in (*f.synonyms, f.label.lower(), k)}


def resolve_field(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Map a possibly-wrong field name to a registry key. Returns (key, note-if-repaired)."""
    if not isinstance(name, str):
        return None, None
    if name in FIELDS:
        return name, None
    n = name.lower().strip()
    if n in _SYN:
        return _SYN[n], f"interpreted field '{name}' as '{_SYN[n]}'"
    m = difflib.get_close_matches(n, list(_SYN), n=1, cutoff=0.8)
    if m:
        return _SYN[m[0]], f"interpreted field '{name}' as '{_SYN[m[0]]}'"
    return None, None


def repair(obj: Any, notes: List[str]) -> Any:
    """Walk a raw dict and fix field names in place where we're confident; record what we changed."""
    if isinstance(obj, dict):
        f = obj.get("field")
        if isinstance(f, str) and f not in FIELDS and f != "match_strength":
            fixed, note = resolve_field(f)
            if fixed:
                obj["field"] = fixed
                notes.append(note)
        for v in obj.values():
            repair(v, notes)
    elif isinstance(obj, list):
        for v in obj:
            repair(v, notes)
    return obj


# ---------------------------------------------------------------- patches (conversation)
class PatchOp(BaseModel):
    model_config = ConfigDict(extra="ignore")
    op: Literal["add", "remove", "replace", "set_sort", "set_limit", "set_universe", "clear"]
    field: Optional[str] = None          # for remove/replace: which field to target
    condition: Optional[Criterion] = None
    sort: Optional[Sort] = None
    limit: Optional[int] = Field(None, ge=1, le=MAX_LIMIT)
    universe: Optional[str] = None


def _touches(c: Criterion, field: str) -> bool:
    return any(l.field == field for l in (c.any if isinstance(c, AnyGroup) else [c]))


def apply_patch(screen: Screen, ops: List[PatchOp]) -> Tuple[Screen, List[str]]:
    """Apply patch ops; returns the new (re-validated) screen and a human-readable change log."""
    s = copy.deepcopy(screen)
    log: List[str] = []
    for p in ops:
        if p.op == "clear":
            s.all = []
            log.append("cleared all filters")
        elif p.op == "add" and p.condition is not None:
            s.all.append(p.condition)
            log.append(f"added {describe(p.condition)}")
        elif p.op == "remove" and p.field:
            before = len(s.all)
            s.all = [c for c in s.all if not _touches(c, p.field)]
            if len(s.all) < before:
                log.append(f"removed {FIELDS[p.field].label if p.field in FIELDS else p.field} filter")
        elif p.op == "replace" and p.condition is not None:
            target = p.field or (p.condition.field if isinstance(p.condition, Condition) else None)
            idx = next((i for i, c in enumerate(s.all) if target and _touches(c, target)), None)
            if idx is None:
                s.all.append(p.condition)
            else:
                s.all[idx] = p.condition
            log.append(f"set {describe(p.condition)}")
        elif p.op == "set_sort" and p.sort:
            s.sort = p.sort
            log.append(f"sorted by {p.sort.field} {p.sort.dir}")
        elif p.op == "set_limit" and p.limit:
            s.limit = p.limit
            log.append(f"limit {p.limit}")
        elif p.op == "set_universe" and p.universe:
            s.universe = p.universe  # type: ignore[assignment]
            log.append(f"universe {p.universe}")
    return Screen.model_validate(s.model_dump()), log


# ---------------------------------------------------------------- human-readable rendering
def fmt_value(field: str, v: Any) -> str:
    f = FIELDS.get(field)
    if isinstance(v, RefValue):
        if v.ref == "field":
            return FIELDS[v.field].label
        if v.ref.endswith("median"):
            return f"{v.ref.split('_')[0]} median"
        return f"{v.ref.split('_')[0]} {int(round(v.p * 100))}th percentile"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        if f and f.kind == NUM:
            return f"{fmt_num(field, v[0])} and {fmt_num(field, v[1])}"
        return ", ".join(map(str, v))
    if isinstance(v, (int, float)):
        return fmt_num(field, v)
    return str(v)


def fmt_num(field: str, x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x != x:  # NaN
        return "n/a"
    unit = FIELDS[field].unit if field in FIELDS else ""
    if unit == "₹ Cr":
        return f"₹{x:,.0f} Cr" if abs(x) >= 100 else f"₹{x:,.1f} Cr"
    if unit == "₹":
        return f"₹{x:,.2f}"
    if unit in ("%", "pp"):
        return f"{x:.1f}{'%' if unit == '%' else ' pp'}"
    if unit == "x":
        return f"{x:.2f}x"
    if unit == "yrs":
        return f"{x:.0f} yrs"
    return f"{x:.2f}" if abs(x) < 100 else f"{x:,.0f}"


OP_WORDS = {">": ">", ">=": "≥", "<": "<", "<=": "≤", "between": "between", "in": "in", "not_in": "not in",
            "contains": "contains", "is": "is"}


def describe(c: Criterion) -> str:
    if isinstance(c, AnyGroup):
        return "(" + " OR ".join(describe(x) for x in c.any) + ")"
    label = FIELDS[c.field].label if c.field in FIELDS else c.field
    if c.op == "is":
        return label if c.value else f"NOT {label}"
    return f"{label} {OP_WORDS[c.op]} {fmt_value(c.field, c.value)}"
