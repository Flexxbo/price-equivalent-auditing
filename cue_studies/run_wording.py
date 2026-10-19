"""Wording test (Sec. 4.6): a fixed cue set, N LLM-generated truthful wordings of the full
description, each measured at confirmation resolution. Is wording a systematic lever?

Mock:  python cue_studies/run_wording.py --mock
Paper: python cue_studies/run_wording.py --eval-model gpt-4o --gen-model claude-sonnet \
           --cues sp_strong,scarcity --pair-index 0 --n-wordings 12 --confirm-n 30 --max-budget 8 --yes
"""
from __future__ import annotations
import argparse, os, re, sys, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from models import get_model                                    # noqa: E402
from products import resolve_category, load_pairs               # noqa: E402
from budget import BudgetGuard, estimate_chat, preflight        # noqa: E402
from generator import LLMGenerator, compose                     # noqa: E402
import wording as wording_mod                                   # noqa: E402
import manifest as manifest_mod                                 # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "data", "results", "cue_studies")
os.makedirs(OUT_DIR, exist_ok=True)

# Cue pool for the wording test (headphones; stipulated true). numbers/keywords feed the gate.
CUES_HEADPHONES = [
    {"id": "sp_strong", "text": "chosen by over 10,000 customers", "provenance": "stipulated",
     "numbers": ["10,000"], "keywords": ["customers", "chosen"]},
    {"id": "authority", "text": "recommended by audio experts", "provenance": "stipulated",
     "numbers": [], "keywords": ["experts", "recommended"]},
    {"id": "scarcity", "text": "only a few left in stock", "provenance": "stipulated",
     "numbers": [], "keywords": ["few left", "stock"]},
    {"id": "bandwagon", "text": "a popular choice this season", "provenance": "stipulated",
     "numbers": [], "keywords": ["popular"]},
]


def build_pool(pair):
    facts_text = pair.product_a.neutral_description
    nums = [m.group() for m in re.finditer(r"\d[\d,\.]*", facts_text)]
    pool = {
        "facts": [{"id": "facts", "text": facts_text, "provenance": "fact",
                   "required": False, "numbers": nums, "keywords": []}],
        "cues": CUES_HEADPHONES,
    }
    return facts_text, pool


def main(argv=None):
    p = argparse.ArgumentParser(description="Wording test (Sec. 4.6)")
    p.add_argument("--category", default="headphones")
    p.add_argument("--pair-index", type=int, default=0)
    p.add_argument("--eval-model", default="gpt-4o")
    p.add_argument("--gen-model", default="claude-sonnet")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--cues", default="authority,sp_strong", help="fixed cue set (comma-separated)")
    p.add_argument("--n-wordings", type=int, default=12)
    p.add_argument("--confirm-n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-budget", type=float, default=8.0)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--run-id", default="wording")
    p.add_argument("--yes", action="store_true")
    a = p.parse_args(argv)

    pair = load_pairs(resolve_category(a.category))[a.pair_index]
    facts_text, pool = build_pool(pair)
    pool_ids = [c["id"] for c in pool["cues"]]

    cue_ids = [c.strip() for c in a.cues.split(",") if c.strip()]
    bad = [c for c in cue_ids if c not in pool_ids]
    if bad:
        print(f"Unknown cues {bad}. Available: {pool_ids}")
        return

    n_word = a.n_wordings

    guard = None
    if a.mock:
        model, eval_alias, reveal = get_model("mock"), "mock", True
        rng = random.Random(a.seed)
        cands = [{"id": f"w{i}", "text": compose(facts_text, pool, cue_ids, rng),
                  "cues": sorted(cue_ids), "regime": "R4"} for i in range(a.n_wordings)]
    else:
        eval_calls = n_word * a.confirm_n * (len(wording_mod.confirm_mod.FINE_GRID) + 4)
        gen_cost = estimate_chat(1, 400, 120 * a.n_wordings, a.gen_model)
        projected = estimate_chat(eval_calls, 340, 4, a.eval_model) + gen_cost
        bd = f"{n_word} wordings x {a.confirm_n}/point x ~15 points = ~{eval_calls} calls ({a.eval_model})"
        if not preflight(projected, a.max_budget, label="wording", assume_yes=a.yes, breakdown=bd):
            print("Aborted before the first API call.")
            return
        guard = BudgetGuard(a.max_budget, label="wording")
        model, eval_alias, reveal = get_model(a.eval_model, budget=guard), a.eval_model, False
        gen = LLMGenerator(get_model(a.gen_model, budget=guard), facts_text, pool, seed=a.seed)
        cands = [{"id": f"w{i}", "text": pr["text"], "cues": sorted(cue_ids), "regime": "R4"}
                 for i, pr in enumerate(gen.propose_fixed(a.n_wordings, cue_ids))]
        if not cands:
            print("The generator returned no valid wordings.")
            return

    out = os.path.join(OUT_DIR, f"wording_{a.run_id}.md")
    s = wording_mod.run_wording(cands, pair, model, pool, facts_text, model_name=eval_alias,
                                n=a.confirm_n, seed=a.seed, out_path=out, guard=guard,
                                reveal=reveal, concurrency=(1 if a.mock else a.concurrency),
                                run_id=a.run_id)

    aliases = ["mock"] if a.mock else [a.eval_model, a.gen_model]
    manifest_mod.write(os.path.join(OUT_DIR, f"manifest_{a.run_id}.json"),
                       manifest_mod.build(run_id=a.run_id, model_aliases=aliases,
                                          cfg={"stage": "wording", "cues": cue_ids,
                                               "n_wordings": a.n_wordings, "confirm_n": a.confirm_n,
                                               "seed": a.seed}, pool=pool, guard=guard, extra=s))

    print("\n===== WORDING SUMMARY =====")
    print(f"  valid:            {s['n_valid']}/{a.n_wordings}")
    print(f"  PE spread:        {s['pe_range_pp']:.1f} pp" if s['pe_range_pp'] is not None else "  PE spread: —")
    print(f"  median CI width:  {s['median_ci_pp']:.1f} pp" if s['median_ci_pp'] is not None else "  CI width: —")
    print(f"  best vs worst CIs separated: {s['best_worst_separated']}")
    print(f"  -> wording effect systematic: {s['systematic']}")
    print(f"\n  Report: {s['report']}")


if __name__ == "__main__":
    main()
