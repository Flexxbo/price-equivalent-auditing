"""Prompt-paraphrase robustness analysis: is the price-equivalent stable across prompt paraphrases?

Reads the per-(cue x variant) CSVs from data/results/E1_robustness/, computes the
position-controlled price-equivalent (PE) for every cell, and reports TWO graded
robustness criteria:

  * Level robustness  -- per (model, cue): how far does PE move across variants?
                         verdict STABLE if the span <= --level-thresh pp.
  * Rank robustness   -- per model: does the ORDERING of cues survive paraphrasing?
                         average pairwise Kendall tau across variants;
                         verdict STABLE if tau >= --tau-thresh (needs >=3 cues).

A rank-stable but level-drifting result implies an ordinal/interval reading.

Diagnostic: logistic PE with a trial-level bootstrap. The paper's robustness table
and figure use the model-free estimator (make_paper_outputs.py).

Run:
    python src/analyze_robustness.py
    python src/analyze_robustness.py --dir data/results/E1_robustness --tau-thresh 0.8
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analysis import (load_results, position_controlled_bootstrap,  # noqa: E402
                      position_controlled_wtp)

PE_CAP = 1.5  # |PE| above 150% -> treat as "no identifiable crossing"


def _kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b for two equal-length numeric vectors (no scipy dependency)."""
    n = len(x)
    if n < 2:
        return float("nan")
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                c += 1
            else:
                d += 1
    denom = np.sqrt((c + d + tx) * (c + d + ty))
    return float((c - d) / denom) if denom > 0 else float("nan")


def _cue_key(condition: str, variant: str) -> str:
    """Stable cue id across variants (strip a trailing _<variant> if present)."""
    if variant and condition.endswith(f"_{variant}"):
        condition = condition[: -(len(variant) + 1)]
    return re.sub(r"^(e1_|cue_|m4_)", "", condition)


def load_cells(paths: list[str]) -> pd.DataFrame:
    """One row per (model, cue, variant) with its position-controlled PE (%)."""
    frames = []
    for p in paths:
        if "_mock_" in os.path.basename(p):
            continue
        try:
            d = load_results(p)
        except Exception as e:
            print(f"  (skipped {os.path.basename(p)}: {e})")
            continue
        if d.empty:
            continue
        if "prompt_variant" not in d.columns:
            d["prompt_variant"] = "v0"
        d["prompt_variant"] = d["prompt_variant"].fillna("v0")
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # robust against leftover mock/demo test files: exclude by the model column,
    # not just the filename, so they can never contaminate a real analysis.
    df = df[df["model"].astype(str).str.lower() != "mock"]
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (model, cond, variant), sub in df.groupby(["model", "condition", "prompt_variant"]):
        pe = position_controlled_wtp(sub)["wtp"]
        ok = np.isfinite(pe) and abs(pe) <= PE_CAP
        boot = position_controlled_bootstrap(sub, n_boot=500) if ok else None
        rows.append({
            "model": model, "cue": _cue_key(str(cond), str(variant)),
            "variant": variant, "n": len(sub),
            "pe_pct": pe * 100 if ok else np.nan,
            "pe_lo": boot["ci95"][0] * 100 if boot else np.nan,
            "pe_hi": boot["ci95"][1] * 100 if boot else np.nan,
        })
    return pd.DataFrame(rows)


def level_report(cells: pd.DataFrame, thresh_pp: float) -> list[dict]:
    out = []
    for (model, cue), sub in cells.groupby(["model", "cue"]):
        pes = sub.dropna(subset=["pe_pct"])
        vals = pes["pe_pct"].values
        if len(vals) < 2:
            out.append({"model": model, "cue": cue, "k": len(vals),
                        "mean": float(vals[0]) if len(vals) else np.nan,
                        "span": np.nan, "verdict": "n/a (<2 variants)"})
            continue
        span = float(vals.max() - vals.min())
        # is the drift bigger than sampling noise? if ANY pair of variant CIs is
        # disjoint, at least one paraphrase genuinely differs (not just noise).
        vv = pes[["pe_pct", "pe_lo", "pe_hi"]].values
        disjoint = any(
            (vv[i][1] > vv[j][2]) or (vv[j][1] > vv[i][2])
            for i in range(len(vv)) for j in range(i + 1, len(vv))
        )
        verdict = "STABLE" if span <= thresh_pp else "DRIFT"
        if verdict == "DRIFT":
            verdict += " (real: some CIs disjoint)" if disjoint else " (within noise: CIs overlap)"
        out.append({
            "model": model, "cue": cue, "k": len(vals),
            "mean": float(vals.mean()), "span": span, "verdict": verdict,
        })
    return out


def rank_report(cells: pd.DataFrame, tau_thresh: float) -> list[dict]:
    out = []
    for model, sub in cells.groupby("model"):
        wide = sub.pivot_table(index="cue", columns="variant", values="pe_pct")
        wide = wide.dropna(axis=0, how="any")  # cues present (finite) in every variant
        variants = list(wide.columns)
        if wide.shape[0] < 3 or len(variants) < 2:
            out.append({"model": model, "n_cues": wide.shape[0], "n_variants": len(variants),
                        "mean_tau": np.nan, "verdict": "n/a (need >=3 cues, >=2 variants)"})
            continue
        taus = []
        for v1, v2 in combinations(variants, 2):
            t = _kendall_tau(wide[v1].values, wide[v2].values)
            if np.isfinite(t):
                taus.append(t)
        mean_tau = float(np.mean(taus)) if taus else np.nan
        out.append({
            "model": model, "n_cues": wide.shape[0], "n_variants": len(variants),
            "mean_tau": mean_tau,
            "verdict": "STABLE" if (mean_tau == mean_tau and mean_tau >= tau_thresh) else "UNSTABLE",
        })
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Prompt-paraphrase robustness of the price-equivalent")
    p.add_argument("--dir", default=None, help="folder with robustness CSVs (default data/results/E1_robustness)")
    p.add_argument("--glob", default=None, help="explicit glob instead of --dir")
    p.add_argument("--level-thresh", type=float, default=5.0, help="max PE span (pp) to call STABLE")
    p.add_argument("--tau-thresh", type=float, default=0.8, help="min mean Kendall tau to call STABLE")
    a = p.parse_args(argv)

    if a.glob:
        paths = sorted(glob.glob(a.glob, recursive=True))
    else:
        base = a.dir or os.path.join(os.path.dirname(__file__), "..", "data", "results", "E1_robustness")
        paths = sorted(glob.glob(os.path.join(base, "**", "*.csv"), recursive=True))
    if not paths:
        print("No robustness CSVs found. Run: python src/run_robustness.py --model <m> --yes")
        return 1

    cells = load_cells(paths)
    if cells.empty:
        print("No usable (non-mock) cells found.")
        return 1

    print("=" * 70)
    print("PRICE-EQUIVALENT per cell (position-controlled, %):")
    pivot = cells.pivot_table(index=["model", "cue"], columns="variant", values="pe_pct")
    with pd.option_context("display.float_format", lambda x: f"{x:6.1f}"):
        print(pivot.to_string())

    print("\nPer-cell 95% CI (bootstrap) -- for telling real drift from sampling noise:")
    for (model, cue), sub in cells.groupby(["model", "cue"]):
        parts = []
        for _, r in sub.sort_values("variant").iterrows():
            if r["pe_pct"] == r["pe_pct"]:
                parts.append(f"{r['variant']} {r['pe_pct']:5.1f} [{r['pe_lo']:4.1f},{r['pe_hi']:5.1f}]")
            else:
                parts.append(f"{r['variant']}  n/a")
        print(f"  {model:12s} {cue:18s} " + "   ".join(parts))

    print("\n" + "=" * 70)
    print(f"LEVEL robustness (STABLE if span <= {a.level_thresh:.0f} pp):")
    for r in level_report(cells, a.level_thresh):
        span = f"{r['span']:5.1f}" if r["span"] == r["span"] else "  n/a"
        print(f"  {r['model']:16s} {r['cue']:18s} k={r['k']} "
              f"mean={r['mean']:6.1f}%  span={span} pp  -> {r['verdict']}")

    print("\n" + "=" * 70)
    print(f"RANK robustness (STABLE if mean Kendall tau >= {a.tau_thresh}):")
    for r in rank_report(cells, a.tau_thresh):
        mt = f"{r['mean_tau']:.2f}" if r["mean_tau"] == r["mean_tau"] else "n/a"
        print(f"  {r['model']:16s} cues={r['n_cues']} variants={r['n_variants']} "
              f"mean_tau={mt}  -> {r['verdict']}")

    print("\nReading: rank STABLE + level DRIFT = ordinal/interval reading; "
          "both STABLE = cardinal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
