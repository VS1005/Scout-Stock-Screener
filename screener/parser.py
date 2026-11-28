"""Natural language → validated Screen. Routing: fast path → cache → LLM (→ repair retry → fallback model)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import ValidationError

from screener import config
from screener.dsl import PatchOp, Screen, apply_patch, repair
from screener.fastpath import parse as fast_parse, parse_followup
from screener.llm import LLM, LLMError


class ParseError(ValueError):
    pass


@dataclass
class ParseResult:
    action: str                         # new | patch | clarify | unsupported
    screen: Optional[Screen]
    path: str                           # fastpath | cache | llm | degraded
    reply: str = ""
    changes: List[str] = field(default_factory=list)
    unmapped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    tokens: int = 0
    latency_ms: float = 0.0
    model: Optional[str] = None


def _short_err(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])


def _validate_llm(obj: dict, current: Optional[Screen], notes: List[str]) -> Optional[str]:
    """Validator handed to the LLM loop: returns an error string for the repair prompt, or None."""
    if not isinstance(obj, dict) or obj.get("action") not in ("new", "patch", "clarify", "unsupported"):
        return "action must be one of new|patch|clarify|unsupported"
    repair(obj, notes)
    try:
        if obj["action"] == "new":
            Screen.model_validate(obj.get("screen") or {})
        elif obj["action"] == "patch":
            ops = [PatchOp.model_validate(o) for o in obj.get("ops") or []]
            if current is None:  # a "patch" with nothing to patch is a new screen built from those ops
                obj["action"], obj["screen"] = "new", apply_patch(Screen(), ops)[0].model_dump(exclude_none=True)
            else:
                apply_patch(current, ops)
    except ValidationError as e:
        return _short_err(e)
    except (TypeError, ValueError, KeyError) as e:
        return str(e)
    return None


class Parser:
    def __init__(self, llm: LLM):
        self.llm = llm

    def parse(self, message: str, current: Optional[Screen] = None) -> ParseResult:
        t0 = time.perf_counter()
        message = (message or "").strip()
        if not message:
            raise ParseError("Empty query")
        if len(message) > config.MAX_QUERY_CHARS:
            raise ParseError(f"Query too long (max {config.MAX_QUERY_CHARS} characters)")

        # 1) zero-token fast path
        if current is not None:
            ops = parse_followup(message)
            if ops:
                screen, changes = apply_patch(current, ops)
                return ParseResult("patch", screen, "fastpath", changes=changes,
                                   latency_ms=(time.perf_counter() - t0) * 1000)
        fp = fast_parse(message)
        if fp and fp.consumed_all:
            return ParseResult("new", fp.screen, "fastpath",
                               latency_ms=(time.perf_counter() - t0) * 1000)

        # 2) LLM path - guard first (skipped for cache hits: those inputs were already screened)
        cur_json = current.model_dump(exclude_none=True) if current else None
        score = None if self.llm.is_cached(message, cur_json) else self.llm.injection_score(message)
        if score is not None and score > 0.9:
            raise ParseError("This request looks like an attempt to manipulate the assistant, so it was not processed. "
                             "Please describe the stocks you are looking for.")
        notes: List[str] = []
        try:
            obj, meta = self.llm.translate(message, cur_json, lambda o: _validate_llm(o, current, notes))
        except LLMError as e:
            if fp and fp.screen.all:  # graceful degradation: partial fast-path parse + honest note
                return ParseResult("new", fp.screen, "degraded", unmapped=fp.leftovers,
                                   notes=[f"LLM unavailable ({e}); showing the parts I could parse deterministically."],
                                   latency_ms=(time.perf_counter() - t0) * 1000)
            raise ParseError(str(e)) from e
        # the cached object was validated when stored, but re-run repair so notes are populated
        _validate_llm(obj, current, notes)

        action = obj["action"]
        res = ParseResult(action, None, meta["path"], reply=obj.get("reply") or "",
                          unmapped=[u for u in obj.get("unmapped") or [] if isinstance(u, str)][:10],
                          notes=list(dict.fromkeys(notes)), tokens=meta.get("tokens", 0), model=meta.get("model"))
        if action == "new":
            res.screen = Screen.model_validate(obj["screen"])
        elif action == "patch":
            ops = [PatchOp.model_validate(o) for o in obj.get("ops") or []]
            res.screen, res.changes = apply_patch(current or Screen(), ops)
        res.latency_ms = (time.perf_counter() - t0) * 1000
        return res
