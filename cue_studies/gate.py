"""Programmatic factuality/structure gate (0 API) for candidate descriptions.

Checks:
  - structure_sanitize : reject prompt injection / card structure / meta-instructions
  - length_band        : length within ±X% of the neutral description (or explicit word bounds)
  - number_preservation: no invented numbers (only pool numbers allowed)
  - required_facts     : no dropped mandatory facts. The cue-study runners mark every pool
                         fact as required=False, so this check never triggers in the reported runs.
  - provenance         : every building block in the pool carries a provenance (fact|stipulated|real)

The SEMANTIC part (pool membership "no new claims", valence bound) is an optional LLM judge
(judge_fn); it is not used in the reported runs, and the result is marked 'judge_pending'.
Non-response is measured NOT here but in the PE evaluation.

A pool is:
  {"facts": [PoolItem, ...], "cues": [PoolItem, ...]}
PoolItem = {"id","text","provenance","required"(facts only),"numbers":[...],"keywords":[...]}
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class GateResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)   # rejection reasons (empty = ok)
    checks: dict = field(default_factory=dict)          # per-check pass/fail
    judge_pending: bool = False                         # semantic check still pending


# --- reserved / injection patterns -------------------------------------------------
_CARD_FIELDS = re.compile(r"(?im)^\s*(name|price|rating|description|option\s*[ab])\s*:")
_INSTRUCTION = re.compile(
    r"(?i)(reply with|answer with|respond with|choose (the|option|left|right)|"
    r"select (the|option|left|right)|ignore (the|all|previous|above)|disregard|"
    r"you must|as an ai|system:|assistant:|user:|prefer (the|this) (left|right|option))")
_REPLY_TOKENS = re.compile(r"(?i)\breply\b[^.]{0,20}\b(left|right|option\s*[ab]|[ab])\b")
_PRICE_LIKE   = re.compile(r"\$\s?\d")
_MOCK_TOKENS  = re.compile(r"<<[^>]*>>")
_NUM          = re.compile(r"\d[\d,\.]*")


def _norm_num(s: str) -> str:
    return s.replace(",", "").rstrip(".").lstrip("0") or "0"


def _has_control_chars(text: str) -> bool:
    for ch in text:
        if ch in "\t\r":
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch != "\n":     # control/format chars (zero-width etc.)
            return True
    return False


def structure_sanitize(text: str) -> list[str]:
    bad = []
    if _MOCK_TOKENS.search(text):        bad.append("mock/reveal tokens (<<...>>)")
    if _CARD_FIELDS.search(text):        bad.append("card-structure field label (Name/Price/Rating/Option)")
    if _INSTRUCTION.search(text):        bad.append("instruction / meta / role directive")
    if _REPLY_TOKENS.search(text):       bad.append("reply-with-choice injection")
    if _PRICE_LIKE.search(text):         bad.append("literal price ($n) in description")
    if _has_control_chars(text):         bad.append("control / zero-width unicode")
    if text.count("\n") > 1:             bad.append("multiple newlines (fakes card structure)")
    return bad


def _allowed_numbers(pool: dict) -> set[str]:
    out = set()
    for item in pool.get("facts", []) + pool.get("cues", []):
        for n in item.get("numbers", []):
            out.add(_norm_num(str(n)))
    return out


def check(text: str, pool: dict, neutral_desc: str,
          length_band_pct: int = 20, min_words=None, max_words=None,
          judge_fn=None) -> GateResult:
    reasons, checks = [], {}

    # 1) structure / injection
    sbad = structure_sanitize(text)
    checks["structure"] = not sbad
    reasons += sbad

    # 2) length: explicit word bounds if given (facts+cues are longer than facts-only),
    #    else a ±band around the neutral length.
    cn = len(text.split())
    if min_words is not None or max_words is not None:
        lo = min_words if min_words is not None else 0
        hi = max_words if max_words is not None else 10 ** 9
        checks["length"] = (lo <= cn <= hi)
        if not checks["length"]:
            reasons.append(f"length {cn}w outside [{lo}, {hi}]")
    else:
        nn = len(neutral_desc.split())
        lo, hi = nn * (1 - length_band_pct / 100), nn * (1 + length_band_pct / 100)
        checks["length"] = (lo <= cn <= hi)
        if not checks["length"]:
            reasons.append(f"length {cn}w outside ±{length_band_pct}% of {nn}w")

    # 3) number preservation: no invented numbers
    allowed = _allowed_numbers(pool)
    used = {_norm_num(m.group()) for m in _NUM.finditer(text)}
    invented = sorted(used - allowed - {"0"})
    checks["numbers"] = not invented
    if invented:
        reasons.append(f"invented numbers not in pool: {invented}")

    # 4) required facts present (no dropped mandatory fact — bidirectional coverage)
    missing = []
    for f in pool.get("facts", []):
        if f.get("required"):
            kws = f.get("keywords") or [f.get("text", "")]
            if not any(kw.lower() in text.lower() for kw in kws if kw):
                missing.append(f.get("id", f.get("text")))
    checks["required_facts"] = not missing
    if missing:
        reasons.append(f"dropped required facts: {missing}")

    # 5) provenance present on every pool item (config-level guard)
    prov_ok = all(it.get("provenance") in ("fact", "stipulated", "real")
                  for it in pool.get("facts", []) + pool.get("cues", []))
    checks["provenance"] = prov_ok
    if not prov_ok:
        reasons.append("pool item missing provenance tag")

    # 6) semantic membership / valence -> optional judge (API); else pending
    judge_pending = judge_fn is None
    if judge_fn is not None:
        jres = judge_fn(text, pool)          # expected: {"ok":bool,"reason":str}
        checks["judge"] = bool(jres.get("ok"))
        if not jres.get("ok"):
            reasons.append("judge: " + jres.get("reason", "semantic membership failed"))

    # ok = all PROGRAMMATIC checks passed. judge_pending = True means no judge_fn was given
    # (optional LLM judge; not used in the reported runs).
    return GateResult(ok=(len(reasons) == 0), reasons=reasons,
                      checks=checks, judge_pending=judge_pending)
