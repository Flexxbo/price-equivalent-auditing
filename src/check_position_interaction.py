"""Position-control specification check (appendix on position interaction): does a
price x position INTERACTION move the PE?

The additive logistic assumes a constant first-option
advantage; the primacy bias, however, sharpens near indifference, so the position effect
may depend on the price gap. We refit each condition with an interaction term

    P = sigma(b0 + b1*g + b2*pos + b3*(g*pos)),   pos = nudged_primary in {0,1}

and read the position-neutral crossing at the mean order pos=0.5:

    PE_interaction = -(b0 + 0.5*b2) / (b1 + 0.5*b3)

compared to the additive PE = -(b0 + 0.5*b2)/b1. We report b3 and the signed
shift PE_int - PE_add with a cluster (by product pair) bootstrap 95% CI. The model-free
local-crossing estimator is unaffected by construction and is printed for reference.

    python src/check_position_interaction.py data/results/M4_validity/m4_sp_strong_gpt-4o_*.csv
    python src/check_position_interaction.py <glob> --nboot 2000
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analysis import (ridge_irls, position_controlled_wtp,  # noqa: E402
                      local_crossing_wtp, load_results)


def _clean(path: str) -> pd.DataFrame:
    d = load_results(path)  # derives nudged_primary from nudged_position
    d["price_gap"] = d["price_gap"].astype(float)
    d["nudged_primary"] = d["nudged_primary"].astype(float)
    return d


def interaction_wtp(df: pd.DataFrame) -> dict:
    g = df["price_gap"].values
    pos = df["nudged_primary"].values
    X = np.column_stack([np.ones(len(df)), g, pos, g * pos])
    b = ridge_irls(X, df["chose_nudged"].values)
    denom = b[1] + 0.5 * b[3]
    pe = -(b[0] + 0.5 * b[2]) / denom if denom != 0 else float("nan")
    return {"b0": b[0], "b1": b[1], "b2": b[2], "b3": b[3], "pe": pe}


def _pe_int_point(df: pd.DataFrame) -> float:
    return interaction_wtp(df)["pe"]


def cluster_bootstrap_delta(df: pd.DataFrame, nboot: int, seed: int = 0) -> dict:
    """Bootstrap by resampling product pairs; report b3, PE_add, PE_int, and the shift."""
    rng = np.random.default_rng(seed)
    pairs = df["pair_id"].unique()
    b3s, deltas, pe_ints = [], [], []
    for _ in range(nboot):
        take = rng.choice(pairs, size=len(pairs), replace=True)
        sub = pd.concat([df[df["pair_id"] == p] for p in take], ignore_index=True)
        try:
            it = interaction_wtp(sub)
            add = position_controlled_wtp(sub)["wtp"]
            if np.isfinite(it["pe"]) and np.isfinite(add):
                b3s.append(it["b3"])
                pe_ints.append(it["pe"])
                deltas.append(it["pe"] - add)
        except Exception:
            continue
    def ci(a):
        a = np.array(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a.size else [float("nan")] * 2
    return {
        "b3_ci": ci(b3s), "pe_int_ci": ci(pe_ints),
        "delta_mean_pp": float(np.mean(deltas) * 100) if deltas else float("nan"),
        "delta_ci_pp": [c * 100 for c in ci(deltas)], "n_valid": len(deltas),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("globs", nargs="+", help="one or more CSV paths / globs")
    p.add_argument("--nboot", type=int, default=2000)
    a = p.parse_args(argv)

    files = []
    for gl in a.globs:
        files.extend(sorted(glob.glob(gl)))
    if not files:
        print("No files matched."); return 1

    hdr = f"{'condition':32} {'PE_free':>8} {'PE_add':>8} {'PE_int':>8} {'shift_pp':>9} {'b3':>8} {'b3 95% CI':>18}"
    print(hdr); print("-" * len(hdr))
    for f in files:
        df = _clean(f)
        if df.empty:
            print(f"{os.path.basename(f)[:32]:32}  (no parsable rows)"); continue
        free = local_crossing_wtp(df) * 100 if np.isfinite(local_crossing_wtp(df)) else float("nan")
        add = position_controlled_wtp(df)["wtp"] * 100
        it = interaction_wtp(df)
        pe_int = it["pe"] * 100
        boot = cluster_bootstrap_delta(df, a.nboot)
        name = os.path.basename(f)[:32]
        print(f"{name:32} {free:8.1f} {add:8.1f} {pe_int:8.1f} "
              f"{boot['delta_mean_pp']:9.2f} {it['b3']:8.3f} "
              f"[{boot['b3_ci'][0]:6.3f},{boot['b3_ci'][1]:6.3f}]")
    print("\nPE_free = model-free local crossing (primary; interaction-immune).")
    print("PE_add  = additive logistic, position-neutral -(b0+.5 b2)/b1.")
    print("PE_int  = interaction logistic, -(b0+.5 b2)/(b1+.5 b3).")
    print("shift_pp = PE_int - PE_add (cluster-bootstrap mean, pp).  b3 CI spanning 0 => no interaction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
