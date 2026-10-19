"""Nudge application + LLM-as-judge bias audit.

apply_nudge():   append a marketing cue (default: social proof) to a description.
audit_description(): score a description 1-5 on persuasion dimensions using a
                     separate judge model. A *neutral* description must score <= 2
                     on every dimension; a *nudged* one is expected to spike on the
                     targeted dimension (this is also how we verify the nudge "took").
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

DIMENSIONS = [
    "social_proof",
    "scarcity",
    "authority",
    "exclusivity",
    "urgency",
    "discount_framing",
    "storytelling",
    "emotional_language",
    "comparative_language",
    # Added after the stimulus audit (see the paper's neutrality appendix): value-laden
    # attributes are a persuasion channel the other dimensions miss.
    "normative_valence",
]

# Fallback texts for experiment.py without --cue. All paper conditions use cues.py.
DEFAULT_NUDGES = {
    "social_proof": "Our best-selling model, chosen by 10,000+ customers.",
    "scarcity": "Almost gone — only a few left in stock.",
    "authority": "Recommended by audio engineers.",
}


def apply_nudge(description: str, nudge_text: str) -> str:
    """Append a nudge sentence to a neutral description."""
    desc = description.rstrip()
    if desc and not desc.endswith((".", "!", "?")):
        desc += "."
    return f"{desc} {nudge_text.strip()}"


@dataclass
class AuditResult:
    scores: dict[str, int]
    flagged: list[str] = field(default_factory=list)
    raw: str = ""

    def is_neutral(self, threshold: int = 2) -> bool:
        return all(v <= threshold for v in self.scores.values())

    def max_dimension(self) -> tuple[str, int]:
        return max(self.scores.items(), key=lambda kv: kv[1])


_JUDGE_PROMPT = """You are auditing a product description for persuasive / marketing cues.
Rate the description from 1 (completely absent) to 5 (strong, explicit) on each dimension.
Be strict: any hint of a cue scores at least 3.

Dimensions: {dims}

Note: `normative_valence` = any sustainability/eco, ethical, health, or safety-coded
attribute that reads as virtuous or positive (e.g. "recyclable packaging", "ethically
sourced", "BPA-free"), even when phrased as a plain fact.

Description:
\"\"\"{description}\"\"\"

Respond with ONLY a JSON object mapping each dimension to an integer 1-5. No prose."""


def _parse_scores(text: str) -> dict[str, int]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in judge output: {text!r}")
    obj = json.loads(m.group(0))
    scores: dict[str, int] = {}
    for d in DIMENSIONS:
        v = obj.get(d, 1)
        try:
            scores[d] = max(1, min(5, int(round(float(v)))))
        except (TypeError, ValueError):
            scores[d] = 1
    return scores


def audit_description(description: str, judge_model=None, threshold: int = 2) -> AuditResult:
    """Score a description with an LLM judge.

    judge_model: any object exposing .complete(prompt) -> str (see models.OpenAIModel),
                 or None to lazily construct one from JUDGE_MODEL env (default gpt-4o-mini).
                 Pass a MockModel-like object for offline dry runs.

    The paper's audit of the headphone descriptions used Claude Opus through a chat
    interface with this rubric; the default judge here is a cheap OpenAI model.
    """
    if judge_model is None:
        from models import OpenAIModel

        judge_model = OpenAIModel(name=os.getenv("JUDGE_MODEL", "gpt-4o-mini"))

    prompt = _JUDGE_PROMPT.format(dims=", ".join(DIMENSIONS), description=description)

    if hasattr(judge_model, "complete"):
        raw = judge_model.complete(prompt, max_tokens=200)
    else:  # mock / choose-only backend: treat as neutral
        raw = json.dumps({d: 1 for d in DIMENSIONS})

    try:
        scores = _parse_scores(raw)
    except Exception:
        scores = {d: 1 for d in DIMENSIONS}
    flagged = [d for d, v in scores.items() if v > threshold]
    return AuditResult(scores=scores, flagged=flagged, raw=raw)


def audit_pair_neutrality(pair, judge_model=None, threshold: int = 2) -> dict:
    """Audit both neutral descriptions of a pair. Returns a report dict."""
    report = {"pair_id": pair.pair_id, "passed": True, "products": {}}
    for prod in pair.products():
        res = audit_description(prod.neutral_description, judge_model, threshold)
        report["products"][prod.id] = {
            "scores": res.scores,
            "flagged": res.flagged,
            "neutral": res.is_neutral(threshold),
        }
        if not res.is_neutral(threshold):
            report["passed"] = False
    return report


def verify_nudge_took(neutral_desc: str, nudged_desc: str, target_dim: str,
                      judge_model=None) -> dict:
    """Sanity-check that applying the nudge actually raised the targeted dimension."""
    before = audit_description(neutral_desc, judge_model)
    after = audit_description(nudged_desc, judge_model)
    return {
        "target_dimension": target_dim,
        "before": before.scores.get(target_dim),
        "after": after.scores.get(target_dim),
        "increased": (after.scores.get(target_dim, 0) > before.scores.get(target_dim, 0)),
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from products import load_pairs

    use_mock = "--mock" in sys.argv
    judge = None
    if use_mock:
        from models import MockModel

        judge = MockModel()  # returns neutral scores; for wiring tests only

    for pair in load_pairs():
        rep = audit_pair_neutrality(pair, judge_model=judge)
        print(json.dumps(rep, indent=2))
