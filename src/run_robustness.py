"""Prompt-paraphrase robustness runner: the SAME cues measured under several semantically
equivalent task phrasings (prompt variants), under one shared budget.

If the price-equivalent is a stable measurement, it should be at least ordinally
invariant to harmless paraphrases of the task.

Run:
    python src/run_robustness.py --mock --n 8          # offline pipeline check
    python src/run_robustness.py --model gpt-4o --n 12 --max-budget 15 --yes
Then:
    python src/analyze_robustness.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import budget as B  # noqa: E402
from cues import get_cue  # noqa: E402
from experiment import (EST_IN_TOKENS, EST_OUT_TOKENS, FINE_SWEEP, PROMPT_VARIANTS,  # noqa: E402
                        RESULTS_DIR, ExperimentConfig, run_experiment)
from products import load_pairs, resolve_category  # noqa: E402

# default cue ladder: well-separated intensities so the cross-variant rank
# correlation is meaningful (a single adjacent swap can't dominate Kendall tau).
DEFAULT_CUES = ["neutral", "sp_weak", "sp_medium", "sp_strong", "sp_vstrong",
                "framing_gainloss"]
DEFAULT_SWEEP_E1 = list(FINE_SWEEP)  # the paper's eight-point price grid


def main(argv=None) -> int:
    from models import _load_env_file
    _load_env_file()  # so LLM_AUDIT_HARD_CAP_* from .env applies at preflight
    p = argparse.ArgumentParser(description="Prompt-paraphrase robustness of the price-equivalent")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--n", type=int, default=10, help="trials per price point per pair (the paper used 12)")
    p.add_argument("--cues", default=",".join(DEFAULT_CUES),
                   help="comma-separated cue ids (>=3 recommended for rank stability)")
    p.add_argument("--variants", default=",".join(PROMPT_VARIANTS.keys()),
                   help="comma-separated prompt variants (v0..v3)")
    p.add_argument("--sweep", default=None,
                   help="comma-separated gaps (default: dense low-end FINE_SWEEP)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-budget", type=float, default=4.0)
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel in-flight API calls per pair (1 = sequential; try 4-8 "
                        "for OpenAI, 2-3 for throttled Bedrock)")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="skip (cue,variant) cells whose CSV is already complete")
    p.add_argument("--category", default="headphones",
                   help="product category (headphones|coffee) or a pairs-file path")
    a = p.parse_args(argv)

    cues = [c.strip() for c in a.cues.split(",") if c.strip()]
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in PROMPT_VARIANTS]
    if bad:
        print(f"Unknown prompt variant(s): {bad}. Known: {list(PROMPT_VARIANTS)}")
        return 1
    sweep = [float(x) for x in a.sweep.split(",")] if a.sweep else list(DEFAULT_SWEEP_E1)
    model = "mock" if a.mock else a.model
    ppath = resolve_category(a.category)

    n_pairs, n_gaps = len(load_pairs(ppath)), len(sweep)
    total_calls = len(cues) * len(variants) * n_pairs * n_gaps * a.n

    guard = None
    if model != "mock":
        projected = B.estimate_chat(total_calls, EST_IN_TOKENS, EST_OUT_TOKENS, model)
        bd = (f"{len(cues)} cues x {len(variants)} variants x {n_pairs} pairs x "
              f"{n_gaps} gaps x {a.n} trials = {total_calls} calls on {model}")
        if not B.preflight(projected, a.max_budget, label="robustness", assume_yes=a.yes, breakdown=bd):
            print("Aborted before any API call.")
            return 1
        guard = B.BudgetGuard(a.max_budget, label="robustness")

    sub = "E1_robustness" if a.category == "headphones" else f"E1_robustness_{a.category}"
    out_dir = RESULTS_DIR / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    bar = None
    try:
        from tqdm import tqdm
        bar = tqdm(total=total_calls, desc="robustness", dynamic_ncols=True)
    except Exception:
        bar = None

    expected_rows = n_pairs * len(sweep) * a.n
    for cid in cues:
        c = get_cue(cid)
        for v in variants:
            if a.resume:
                complete = False
                for f in sorted(out_dir.glob(f"e1_{cid}_{v}_{model}_*.csv")):
                    with open(f, encoding="utf-8") as fh:
                        n_rows = sum(1 for _ in fh) - 1  # minus header
                    if n_rows >= expected_rows:
                        complete = True
                        break
                if complete:
                    if bar is not None:
                        bar.update(expected_rows)
                    continue
            cfg = ExperimentConfig(
                model_name=model, n_per_point=a.n, sweep=list(sweep),
                condition=f"e1_{cid}", temperature=a.temperature,
                cued_text=c.cued_text, control_text=c.control_text,
                nudge_target="balanced", prompt_variant=v,
                assume_yes=True, max_budget=a.max_budget,
                products_path=str(ppath), concurrency=a.concurrency,
            )
            if bar is not None:
                bar.set_description(f"robustness {cid}/{v}")
            else:
                print(f"\n=== robustness {cid} / {v}: \"{c.cued_text}\" ===")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"e1_{cid}_{v}_{model}_{stamp}.csv"
            run_experiment(cfg, out_path=out_path, guard=guard, progress=bar)
            if guard is not None and guard.spent >= guard.max:
                if bar is not None:
                    bar.close()
                print(f"\nBudget reached; stopping the robustness run. "
                      f"Spend: ${guard.spent:.4f} over {guard.calls} calls.")
                return 0

    if bar is not None:
        bar.close()
    if guard is not None:
        print(f"\nRobustness total spend: ${guard.spent:.4f} over {guard.calls} calls "
              f"(budget ${guard.max:.2f}).")
    print("\nNext: python src/analyze_robustness.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
