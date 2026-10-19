"""Single source of truth for every price-equivalent + CI reported in the paper.

Recomputes all PEs and pair-clustered bootstrap 95% CIs ONCE, with a fixed seed and
fixed n_boot, using the model-free local-crossing estimator. It prints the canonical
values. All PE and CI values in the battery, cross-model and framing tables are taken
from here.

    python src/make_paper_numbers.py        (run from the repository root)

Conventions:
  * PE = model-free local crossing of the pooled (pair- & position-averaged) curve, in %.
  * CI = 2.5/97.5 percentiles over 2000 pair-cluster bootstrap resamples, seed 0.
  * base@0 < 0.5  -> cue dispreferred at equal price -> PE floor-censored to 0, mark '‡'.
  * curve never crosses 0.5 in the sweep -> right-censored, PE = '>=<ceiling>' , mark '†'.
  * a '†' inside a CI upper bound = more than 10% of resamples reach the sweep ceiling.
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from analysis import load_results  # noqa: E402

SEED, NBOOT = 0, 2000

def newest(cond, model, sub="M4_validity"):
    fs = glob.glob(f"data/results/{sub}/m4_{cond}_{model}_*.csv")
    return sorted(fs)[-1] if fs else None

def _prep(df):
    g = df.groupby(["pair_id", "price_gap"])["chose_nudged"].agg(["sum", "count"])
    gaps = np.array(sorted(df["price_gap"].unique()))
    pairs = sorted(df["pair_id"].unique())
    K = {p: np.array([g.loc[(p, x), "sum"] if (p, x) in g.index else 0 for x in gaps], float) for p in pairs}
    N = {p: np.array([g.loc[(p, x), "count"] if (p, x) in g.index else 0 for x in gaps], float) for p in pairs}
    return gaps, pairs, K, N

def _cross(gaps, ps):
    if ps[0] < 0.5:
        return 0.0, "floor"
    for i in range(len(gaps) - 1):
        if ps[i] >= 0.5 > ps[i + 1]:
            x0, x1, p0, p1 = gaps[i], gaps[i + 1], ps[i], ps[i + 1]
            return (x0 + (p0 - 0.5) / (p0 - p1) * (x1 - x0)) * 100, None
    return gaps[-1] * 100, "ceil"

def cell(cond, model, sub="M4_validity"):
    f = newest(cond, model, sub)
    if not f:
        return None
    gaps, pairs, K, N = _prep(load_results(f))
    pt, flag = _cross(gaps, sum(K.values()) / sum(N.values()))
    if flag == "floor":
        return {"pe": "0", "mark": "‡", "ci": None}
    if flag == "ceil":
        return {"pe": f">={round(pt)}", "mark": "†", "ci": None}
    rng = np.random.default_rng(SEED)
    es, nc = [], 0
    for _ in range(NBOOT):
        t = rng.choice(pairs, len(pairs), replace=True)
        e, fl = _cross(gaps, sum(K[p] for p in t) / sum(N[p] for p in t))
        if fl == "ceil":
            nc += 1
        es.append(e)
    lo, hi = np.percentile(es, 2.5), np.percentile(es, 97.5)
    tag = "†" if nc / NBOOT > 0.1 else ""
    return {"pe": f"{pt:.1f}", "mark": "", "ci": f"[{lo:.1f}, {hi:.1f}{tag}]"}

BATTERY = ["sp_weak", "sp_medium", "sp_strong", "sp_vstrong", "authority", "scarcity",
           "framing_gainloss", "neutral", "vacuous", "framing_pct_frac", "framing_rating",
           "sp_strong_lenctrl", "presence_ctrl"]
CORE = ["sp_weak", "sp_medium", "sp_strong", "sp_vstrong", "scarcity", "authority",
        "neutral", "framing_gainloss"]
MODELS = ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "nova-micro",
          "llama-70b", "llama-8b"]

def fmt(c):
    if c is None:
        return "--"
    return f"{c['pe']}{c['mark']}" + (f" {c['ci']}" if c["ci"] else "")

def main():
    print(f"=== CANONICAL PAPER NUMBERS  (seed={SEED}, nboot={NBOOT}) ===\n")
    print("--- Cue battery (GPT-4o; headphones | coffee) ---")
    for cue in BATTERY:
        h = cell(cue, "gpt-4o")
        cc = "authority_coffee" if cue == "authority" else cue
        co = cell(cc, "gpt-4o", "M4_validity_coffee")
        print(f"  {cue:20} H: {fmt(h):24}  C: {fmt(co)}")
    print("\n--- Cross-model PEs ---")
    for m in MODELS:
        cells = [cell(cue, m) for cue in CORE]
        print(f"  {m:13} " + " | ".join(f"{c['pe']}{c['mark']}" if c else "--" for c in cells))
    print("\n--- Cross-model PEs with CIs ---")
    for m in MODELS:
        print(f"  {m:13} " + " | ".join(fmt(cell(cue, m)) for cue in CORE))
    print("\n--- Framing at constant information (GPT-4o) ---")
    for cue in ["framing_rating", "framing_pct_frac", "framing_gainloss"]:
        print(f"  {cue:18} {fmt(cell(cue,'gpt-4o'))}")

if __name__ == "__main__":
    raise SystemExit(main())
