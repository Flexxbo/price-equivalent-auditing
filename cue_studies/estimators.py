"""Estimators for the cue studies; uses the position-controlled logistic model from src/analysis.py.

ONE fit yields two quantities:
  - PE (logistic) = -(b0 + 0.5*b_pos)/b_gap   -> price premium until indifference (money)
  - pull β        =  b0 + 0.5*b_pos           -> log-odds of choosing the product at EQUAL price
                                                 (position-controlled) = 'pull'
PE = pull / (-b_gap) decomposes PE into pull / price sensitivity.
The logistic PE uses ALL sweep points -> much tighter CIs than the model-free
local crossing. β is the second, standardized lens (log-odds effect size).

`wtp` in identifiers = PE (historical name; not a willingness to pay).
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from analysis import PRIMARY_LABELS, position_controlled_wtp, local_crossing_wtp  # noqa: E402


def _prep(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows).dropna(subset=["chose_nudged"]).copy()
    if len(df):
        df["chose_nudged"] = df["chose_nudged"].astype(int)
        df["nudged_primary"] = df["nudged_position"].isin(PRIMARY_LABELS).astype(int)
    return df


def _ci(vals):
    # discard degenerate resamples (price sensitivity ~0 -> PE explodes);
    # otherwise a few outliers inflate the CI absurdly (e.g. [-200%, +166%]).
    a = np.array([v for v in vals if np.isfinite(v) and -1.5 <= v <= 3.0])
    if a.size < 20:
        return None
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def estimate_all(rows, n_boot: int = 1500, seed: int = 0) -> dict:
    """PE (model-free + logistic, with CI) and pull β (with CI) from one bootstrap."""
    df = _prep(rows)
    out = {"pe_local": float("nan"), "pe_logit": float("nan"), "pe_logit_ci": None,
           "pull_beta": float("nan"), "pull_beta_ci": None, "n": len(df)}
    if len(df) < 3:
        return out
    out["pe_local"] = local_crossing_wtp(df)
    r0 = position_controlled_wtp(df)
    out["pe_logit"] = r0["wtp"]
    out["pull_beta"] = r0["b0"] + 0.5 * r0["b_pos"]

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    pe_b, be_b = [], []
    for _ in range(n_boot):
        d = df.iloc[rng.choice(idx, size=len(idx), replace=True)]
        r = position_controlled_wtp(d)
        if np.isfinite(r["wtp"]):
            pe_b.append(r["wtp"])
        be_b.append(r["b0"] + 0.5 * r["b_pos"])
    out["pe_logit_ci"] = _ci(pe_b)
    out["pull_beta_ci"] = _ci(be_b)
    return out
