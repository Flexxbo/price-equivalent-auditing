"""Main-audit runner: runs every cue condition of the main audit (cues.M4_CONDITIONS;
'M4' is the internal label of the main audit) as its own price-sweep curve / CSV
under one shared budget.

A single BudgetGuard is shared across conditions, so the whole battery is capped.

Run:
    python src/run_m4.py --mock --n 16                 # offline pipeline check
    python src/run_m4.py --model gpt-4o --n 24 --max-budget 15 --yes   # full battery
    python src/run_m4.py --model gpt-4o --conditions sp_weak,sp_strong   # subset
Then:
    python src/analyze_m4.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import budget as B  # noqa: E402
from cues import M4_CONDITIONS, get_cue  # noqa: E402
from datetime import datetime  # noqa: E402

from experiment import (FINE_SWEEP, EST_IN_TOKENS, EST_OUT_TOKENS, RESULTS_DIR,  # noqa: E402
                        ExperimentConfig, run_experiment)
from products import load_pairs, resolve_category  # noqa: E402


def _count_data_rows(path: str) -> int:
    """Number of trial rows (excluding header) in an existing result CSV."""
    try:
        with open(path, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def _resume_target(out_dir, cid: str, model: str, expected: int):
    """For --resume: find the newest existing CSV for this (condition, model).

    Returns (action, path):
      ('resume', path) if it exists with 0 < rows < expected  -> continue the deficit
      ('skip',   path) if it is already complete (rows >= expected)
      ('fresh',  None) if there is no usable partial (missing, or empty w/o header)
    """
    files = sorted(glob.glob(str(out_dir / f"m4_{cid}_{model}_*.csv")))
    if not files:
        return "fresh", None
    newest = files[-1]
    rows = _count_data_rows(newest)
    if rows >= expected:
        return "skip", newest
    if rows == 0:                      # empty file has no header -> start clean instead
        return "fresh", None
    return "resume", newest


def main(argv=None) -> int:
    from models import _load_env_file
    _load_env_file()  # so LLM_AUDIT_HARD_CAP_* from .env applies at preflight
    p = argparse.ArgumentParser(description="Run the main-audit cue conditions")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--n", type=int, default=24, help="trials per price point per pair")
    p.add_argument("--conditions", default="", help="comma-separated cue ids (default: all main-audit conditions)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--sweep", default=None,
                   help="comma-separated gaps (default: dense low-end FINE_SWEEP)")
    p.add_argument("--category", default="headphones",
                   help="product category (headphones|coffee) or a pairs-file path")
    p.add_argument("--max-budget", type=float, default=8.0)
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel in-flight API calls per pair (1 = sequential; try 4-8 "
                        "for OpenAI, 2-3 for throttled Bedrock)")
    p.add_argument("--resume", action="store_true",
                   help="continue the newest incomplete CSV per condition (correct cue + "
                        "fine grid) instead of starting fresh; complete conditions are skipped")
    p.add_argument("--yes", action="store_true")
    a = p.parse_args(argv)

    conds = a.conditions.split(",") if a.conditions else list(M4_CONDITIONS)
    model = "mock" if a.mock else a.model
    sweep = [float(x) for x in a.sweep.split(",")] if a.sweep else list(FINE_SWEEP)
    ppath = resolve_category(a.category)
    n_pairs, n_gaps = len(load_pairs(ppath)), len(sweep)
    total_calls = len(conds) * n_pairs * n_gaps * a.n

    guard = None
    if model != "mock":
        projected = B.estimate_chat(total_calls, EST_IN_TOKENS, EST_OUT_TOKENS, model)
        bd = (f"{len(conds)} conditions x {n_pairs} pairs x {n_gaps} gaps x {a.n} trials "
              f"= {total_calls} calls on {model}")
        if not B.preflight(projected, a.max_budget, label="M4", assume_yes=a.yes, breakdown=bd):
            print("Aborted before any API call.")
            return 1
        guard = B.BudgetGuard(a.max_budget, label="M4")

    sub = "M4_validity" if a.category == "headphones" else f"M4_validity_{a.category}"
    out_dir = RESULTS_DIR / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = n_pairs * n_gaps * a.n
    for cid in conds:
        c = get_cue(cid)
        resume_path = ""
        if a.resume:
            action, path = _resume_target(out_dir, cid, model, expected)
            if action == "skip":
                print(f"\n=== M4 {cid}: already complete ({expected} trials) -> skipped ===")
                continue
            if action == "resume":
                resume_path = path

        cfg = ExperimentConfig(
            model_name=model, n_per_point=a.n, sweep=list(sweep),
            condition=f"m4_{cid}", temperature=a.temperature,
            cued_text=c.cued_text, control_text=c.control_text,
            nudge_target="balanced", assume_yes=True, max_budget=a.max_budget,
            products_path=str(ppath), concurrency=a.concurrency,
            resume_path=resume_path, no_cue=(c.type == "none"),
        )
        ctl = f'  (control: "{c.control_text}")' if c.control_text else ""
        tag = f"  [resume {os.path.basename(resume_path)}]" if resume_path else ""
        print(f"\n=== M4 [{c.type}] {cid}: \"{c.cued_text}\"{ctl}{tag} ===")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"m4_{cid}_{model}_{stamp}.csv"  # ignored when resuming
        run_experiment(cfg, out_path=out_path, guard=guard)
        if guard is not None and guard.spent >= guard.max:
            print("Budget reached; stopping M4.")
            break

    if guard is not None:
        print(f"\nM4 total spend: ${guard.spent:.4f} over {guard.calls} calls "
              f"(budget ${guard.max:.2f}).")
    print("\nNext: python src/analyze_m4.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
