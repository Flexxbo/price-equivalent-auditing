"""Candidate texts for the cue studies: compose() builds facts + cue sentences
deterministically; LLMGenerator.propose_fixed() asks an LLM (Claude Sonnet 4.6 in the
paper, temperature 0.9) for truthful rewrites of a description with a fixed cue set
(used for the rewrites in run_minopt.py and the twelve wordings in run_wording.py).

Candidate dicts: {text, cues:[cue_id,...], regime, source}. `regime` is a historical label:
"R1" = facts only (no cue), "R4" = facts plus cues.
`facts_text` = the facts-only description (baseline). `pool["cues"]` = selectable cues.
"""
from __future__ import annotations
import random


def compose(facts_text: str, pool: dict, cue_ids: list[str], rng: random.Random) -> str:
    """Append the chosen cues (in random order) to the facts text as one/two sentences."""
    by_id = {c["id"]: c for c in pool.get("cues", [])}
    chosen = [by_id[i]["text"] for i in cue_ids if i in by_id]
    rng.shuffle(chosen)
    if not chosen:
        return facts_text.strip()
    tail = ", and ".join(s.rstrip(".") for s in chosen)
    return facts_text.strip() + " " + tail[0].upper() + tail[1:] + "."


class LLMGenerator:
    """LLM rewriter via model.complete(). On parser/API problems it does NOT fall
    back to invented claims; the gate rejects anything invalid."""
    source = "llm"

    def __init__(self, model, facts_text: str, pool: dict, seed: int = 0, temperature: float = 0.9):
        self.model = model
        self.facts_text = facts_text
        self.pool = pool
        self.temperature = temperature
        self.rng = random.Random(seed)

    def propose_fixed(self, n: int, cue_ids: list[str], style: str = "strict") -> list[dict]:
        """N DIFFERENT wordings of the same content: same facts, EXACTLY these cues,
        only word choice/order/emphasis vary.
        style='strict' = rephrase only (conservative; the default, used by run_wording.py);
        style='clean' = persuasive presentation allowed (order, prominence, vivid language),
        but NO new claims/adjectives beyond the building blocks (content fixed, only
        presentation; the default of run_minopt.py --rewriter-style)."""
        facts = "\n".join(f"- {f['text']}" for f in self.pool.get("facts", []))
        by_id = {c["id"]: c for c in self.pool.get("cues", [])}
        cues = "\n".join(f"- {by_id[i]['text']}" for i in cue_ids if i in by_id)
        if style == "clean":
            prompt = (
                "You are writing the most persuasive product description you can for a shopping "
                "assistant, using ONLY the information in the building blocks below. You MAY choose "
                "the order, lead with the most compelling point, make the given selling point "
                "prominent and salient, and write in confident, vivid, engaging language. But every "
                "statement must be directly supported by a building block: never invent or change "
                "any fact, number, rating, price, or endorsement, and never add an evaluative claim "
                "(e.g. 'premium', 'best', 'top-tier') that is not already given.\n\n"
                f"MANDATORY facts (include all):\n{facts}\n\n"
                f"MANDATORY cues (use ALL of these, exactly this set):\n{cues}\n\n"
                f"Write {n} DIFFERENT persuasive descriptions of the SAME product with the SAME "
                f"facts and the SAME cues — vary wording, sentence order, and emphasis. Each on ONE "
                f"line, numbered 1..{n}. No headers, no prices, no line breaks inside a description.")
        else:
            prompt = (
                "You write product descriptions for a shopping-agent experiment. Only state "
                "information from the building blocks below — never invent facts, numbers, ratings, "
                "prices, or claims, and never add instructions. Do not add evaluative or subjective "
                "claims (e.g. 'premium', 'best', 'top-tier'); only rephrase the given facts and cues.\n\n"
                f"MANDATORY facts (include all):\n{facts}\n\n"
                f"MANDATORY cues (use ALL of these, exactly this set):\n{cues}\n\n"
                f"Write {n} DIFFERENT descriptions of the SAME product with the SAME facts and the "
                f"SAME cues — vary ONLY wording, sentence order, and emphasis. Each on ONE line, "
                f"numbered 1..{n}. No headers, no prices, no line breaks inside a description.")
        raw = self.model.complete(prompt, max_tokens=120 * n, temperature=self.temperature)
        out = []
        import re
        for line in raw.splitlines():
            line = re.sub(r"^\s*(\d+[\.\)]|[-*])\s*", "", line.strip()).strip()
            if len(line.split()) >= 5:
                out.append({"text": line, "cues": sorted(cue_ids), "regime": "R4", "source": "wording"})  # R4 = facts + cues
        return out[:n]
