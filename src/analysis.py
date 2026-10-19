"""Analysis: baseline effect, price-equivalent (PE) curve, and a POSITION-CONTROLLED estimate.

Why position control: the agent has a slot bias (it over-picks the first option,
Left / A). When the nudge and price roughly cancel, the agent falls back on that
bias, which — because the nudge slot is counterbalanced — pins P(nudged) at ~0.5
and masquerades as indifference. We therefore fit

    P(chose nudged) = sigma(b0 + b_gap * gap + b_pos * nudged_is_primary)

and report the PE at a *neutral* position (nudged_is_primary = 0.5) as a
confirmation of the model-free local crossing, which is the primary estimate. A
diagnostic plot of P(picked primary slot) vs gap is produced by default so the bias
is always visible.

Identifiers containing 'wtp' are historical names for the price-equivalent (PE),
which is not a willingness to pay. The paper's primary estimator is the model-free
local crossing (local_crossing_wtp); the logistic model is a confirmation. The
bootstraps in this file resample trials and serve as quick per-file diagnostics;
the paper's pair-clustered CIs (2000 resamples, seed 0) come from
make_paper_numbers.py.

Run:
    python src/analysis.py data/results/<file>.csv --base-price 129.99
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# The "primary" slot is the one shown first: Left in left_right mode, A in ab mode.
PRIMARY_LABELS = {"Left", "A"}

# |b_pos| above this flags a pair whose choices are nearly separated by position;
# the position-controlled PE then extrapolates and should not be trusted per-pair.
QUASI_SEP_BPOS = 8.0


def load_results(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["chose_nudged"]).copy()
    df["chose_nudged"] = df["chose_nudged"].astype(int)
    df["nudged_primary"] = df["nudged_position"].isin(PRIMARY_LABELS).astype(int)
    df["picked_primary"] = df["parsed_choice"].isin(PRIMARY_LABELS).astype(int)
    return df


# ---------- basic tests ----------

def binomial_test(successes: int, n: int, p: float = 0.5) -> float:
    from scipy import stats

    return float(stats.binomtest(successes, n, p).pvalue)


def baseline_effect(df: pd.DataFrame) -> dict:
    base = df[df["price_gap"] == 0.0]
    n = len(base)
    k = int(base["chose_nudged"].sum())
    p_hat = k / n if n else float("nan")
    return {
        "n": n, "chose_nudged": k, "p_nudged": p_hat,
        "p_value_vs_0.5": binomial_test(k, n) if n else float("nan"),
        "lift_pp": (p_hat - 0.5) * 100 if n else float("nan"),
    }


def position_bias(df: pd.DataFrame) -> dict:
    """P(agent picks the primary slot) across all trials; ~0.5 means no slot bias."""
    k = int(df["picked_primary"].sum())
    n = len(df)
    p = k / n if n else float("nan")
    return {"n": n, "p_primary": p,
            "p_value_vs_0.5": binomial_test(k, n) if n else float("nan")}


# ---------- logistic fits ----------

def ridge_irls(X: np.ndarray, y: np.ndarray, ridge: float = 1e-3,
               iters: int = 200, tol: float = 1e-9) -> np.ndarray:
    """Logistic regression by iteratively reweighted least squares with a small
    ridge penalty (keeps coefficients finite under quasi-separation)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        z = eta + (y - mu) / W
        A = (X.T * W) @ X + ridge * np.eye(X.shape[1])
        nb = np.linalg.solve(A, (X.T * W) @ z)
        if np.max(np.abs(nb - beta)) < tol:
            beta = nb
            break
        beta = nb
    return beta


def naive_wtp(df: pd.DataFrame) -> dict:
    """PE ignoring position (chose_nudged ~ gap)."""
    X = np.column_stack([np.ones(len(df)), df["price_gap"].astype(float).values])
    b = ridge_irls(X, df["chose_nudged"].values)
    wtp = -b[0] / b[1] if b[1] != 0 else float("nan")
    return {"b0": float(b[0]), "b_gap": float(b[1]), "wtp": float(wtp)}


def position_controlled_wtp(df: pd.DataFrame) -> dict:
    """PE controlling for slot bias (chose_nudged ~ gap + nudged_primary),
    evaluated at a neutral position (nudged_primary = 0.5)."""
    X = np.column_stack([
        np.ones(len(df)),
        df["price_gap"].astype(float).values,
        df["nudged_primary"].values.astype(float),
    ])
    b = ridge_irls(X, df["chose_nudged"].values)
    wtp = -(b[0] + 0.5 * b[2]) / b[1] if b[1] != 0 else float("nan")
    return {"b0": float(b[0]), "b_gap": float(b[1]), "b_pos": float(b[2]), "wtp": float(wtp)}


def _pos_controlled_point(gap: np.ndarray, y: np.ndarray, prim: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(gap)), gap, prim])
    b = ridge_irls(X, y)
    return -(b[0] + 0.5 * b[2]) / b[1] if b[1] != 0 else float("nan")


def position_controlled_bootstrap(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    gap = df["price_gap"].astype(float).values
    y = df["chose_nudged"].values
    prim = df["nudged_primary"].values.astype(float)
    idx = np.arange(len(df))
    ests = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        e = _pos_controlled_point(gap[b], y[b], prim[b])
        if np.isfinite(e):
            ests.append(e)
    ests = np.array(ests)
    return {
        "mean": float(np.mean(ests)) if ests.size else float("nan"),
        "ci95": [float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))]
        if ests.size else [float("nan"), float("nan")],
        "n_boot_valid": int(ests.size),
    }


def local_crossing_wtp(df: pd.DataFrame) -> float:
    """Model-free indifference gap by local linear interpolation of the empirical
    P(chose cued) curve around 0.5. Because position is counterbalanced within each
    price point, the empirical curve is already position-neutral in expectation, so
    this needs no logistic assumption -- it is robust to the plateau/cliff shapes
    that distort a global logistic fit (esp. for weak cues near zero).

    Returns the gap where P first crosses 0.5 downward; 0.0 if the cue is already not
    preferred at the smallest gap; nan if P never drops below 0.5 within the sweep."""
    g = empirical_curve(df).sort_values("price_gap")
    xs = g["price_gap"].astype(float).values
    ps = g["p_nudged"].astype(float).values
    if len(xs) < 2:
        return float("nan")
    if ps[0] < 0.5:
        return 0.0
    for i in range(len(xs) - 1):
        if ps[i] >= 0.5 > ps[i + 1]:
            x0, x1, p0, p1 = xs[i], xs[i + 1], ps[i], ps[i + 1]
            return float(x0 + (p0 - 0.5) / (p0 - p1) * (x1 - x0))
    return float("nan")  # no crossing within the swept range


def local_crossing_bootstrap(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True)
    idx = np.arange(len(d))
    ests = []
    for _ in range(n_boot):
        e = local_crossing_wtp(d.iloc[rng.choice(idx, size=len(idx), replace=True)])
        if np.isfinite(e):
            ests.append(e)
    ests = np.array(ests)
    return {
        "mean": float(np.mean(ests)) if ests.size else float("nan"),
        "ci95": [float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))]
        if ests.size else [float("nan"), float("nan")],
    }


def per_pair_wtp(df: pd.DataFrame) -> list[dict]:
    out = []
    for pid, sub in df.groupby("pair_id"):
        base_price = float(sub["control_price"].iloc[0])
        pc = position_controlled_wtp(sub)
        nv = naive_wtp(sub)
        boot = position_controlled_bootstrap(sub, n_boot=800)
        out.append({
            "pair_id": pid, "base_price": base_price,
            "wtp_pos_controlled_pct": pc["wtp"] * 100,
            "wtp_pos_controlled_usd": base_price * pc["wtp"],
            "wtp_ci_pct": [boot["ci95"][0] * 100, boot["ci95"][1] * 100],
            "wtp_naive_pct": nv["wtp"] * 100,
            "b_pos": pc["b_pos"],
            "quasi_separation": bool(abs(pc["b_pos"]) > QUASI_SEP_BPOS),
        })
    return out


def empirical_curve(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("price_gap")["chose_nudged"].agg(["mean", "count", "sum"]).reset_index()
    g = g.rename(columns={"mean": "p_nudged", "count": "n", "sum": "k"})
    g["se"] = np.sqrt(g["p_nudged"] * (1 - g["p_nudged"]) / g["n"])
    return g


# ---------- plots ----------

def plot_wtp(df: pd.DataFrame, b0: float, b_gap: float, wtp_gap: float,
             base_price: float | None, out_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = empirical_curve(df)
    xs = np.linspace(0, max(0.3, df["price_gap"].max()), 200)
    ys = 1 / (1 + np.exp(-(b0 + b_gap * xs)))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(curve["price_gap"] * 100, curve["p_nudged"], yerr=curve["se"],
                fmt="o", capsize=3, label="Empirical P(nudged)", color="#2b6cb0")
    ax.plot(xs * 100, ys, "-", color="#e53e3e", label="Naive logistic fit")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    if np.isfinite(wtp_gap):
        ax.axvline(wtp_gap * 100, ls="--", color="#38a169",
                   label=f"Pos-controlled PE {wtp_gap*100:.1f}%")
    ax.set_xlabel("Price premium on nudged product (%)")
    ax.set_ylabel("P(agent chooses nudged product)")
    title = "Price-equivalent (PE) of the cue"
    if base_price and np.isfinite(wtp_gap):
        title += f"   (≈ ${base_price*wtp_gap:.2f} on a ${base_price:.2f} item)"
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_diagnostic(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Per-pair diagnostic: P(chose nudged) and P(picked primary slot) vs gap.
    The slot-bias curve makes the position confound visible at a glance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pids = list(df["pair_id"].unique())
    ncols = min(3, len(pids))
    nrows = (len(pids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows),
                             squeeze=False, sharey=True)
    for ax_i, pid in enumerate(pids):
        ax = axes[ax_i // ncols][ax_i % ncols]
        sub = df[df["pair_id"] == pid]
        g = sub.groupby("price_gap").agg(P_nudged=("chose_nudged", "mean"),
                                         P_primary=("picked_primary", "mean")).reset_index()
        pc = position_controlled_wtp(sub)
        bp = float(sub["control_price"].iloc[0])
        ax.plot(g["price_gap"] * 100, g["P_nudged"], "o-", color="#2b6cb0",
                label="P(chose nudged)")
        ax.plot(g["price_gap"] * 100, g["P_primary"], "s--", color="#dd6b20",
                alpha=0.8, label="P(picked first slot)")
        ax.axhline(0.5, ls=":", color="gray")
        if np.isfinite(pc["wtp"]):
            ax.axvline(pc["wtp"] * 100, ls="--", color="#38a169",
                       label=f"PE {pc['wtp']*100:.1f}%")
        ax.set_title(f"{pid} (${bp:.2f})")
        ax.set_xlabel("Price premium (%)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
    for j in range(len(pids), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].set_ylabel("Probability")
    fig.suptitle("Position-bias diagnostic: slot preference inflates the 0.50 plateau")
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------- orchestration ----------

def analyze(csv_path: str | Path, base_price: float | None = None, plot: bool = True) -> dict:
    df = load_results(csv_path)
    report: dict = {"file": str(csv_path), "n_trials": len(df)}
    report["baseline"] = baseline_effect(df)
    report["position_bias"] = position_bias(df)

    nv = naive_wtp(df)
    pc = position_controlled_wtp(df)
    report["naive"] = {"wtp_pct": nv["wtp"] * 100}
    report["position_controlled"] = {
        "b0": pc["b0"], "b_gap": pc["b_gap"], "b_pos": pc["b_pos"],
        "wtp_pct": pc["wtp"] * 100,
        "wtp_usd": base_price * pc["wtp"] if base_price else None,
        "bootstrap": position_controlled_bootstrap(df),
    }
    lc = local_crossing_wtp(df)
    report["local_crossing"] = {
        "wtp_pct": lc * 100 if np.isfinite(lc) else float("nan"),
        "bootstrap": local_crossing_bootstrap(df),
    }
    report["per_pair"] = per_pair_wtp(df)
    report["empirical_curve"] = empirical_curve(df).to_dict(orient="records")

    if plot:
        try:
            report["plot"] = str(plot_wtp(df, nv["b0"], nv["b_gap"], pc["wtp"],
                                          base_price, Path(csv_path).with_suffix(".wtp.png")))
        except Exception as e:
            report["plot_error"] = str(e)
        try:
            report["diagnostic_plot"] = str(
                plot_diagnostic(df, Path(csv_path).with_suffix(".diagnostic.png")))
        except Exception as e:
            report["diagnostic_plot_error"] = str(e)
    return report


def _pretty(report: dict) -> str:
    b = report["baseline"]
    pc = report["position_controlled"]
    lo, hi = pc["bootstrap"]["ci95"]
    lines = [
        f"File: {report['file']}",
        f"Trials (parseable): {report['n_trials']}",
        "",
        f"Baseline (gap=0): P(nudged)={b['p_nudged']:.3f} "
        f"(+{b['lift_pp']:.1f} pp vs 50%, p={b['p_value_vs_0.5']:.4g}, n={b['n']})",
        f"Slot bias: P(first slot)={report['position_bias']['p_primary']:.3f} "
        f"(p={report['position_bias']['p_value_vs_0.5']:.4g})  "
        f"-- 0.5 = no bias; departure means the position control matters",
        "",
        f">> PE (position-controlled logistic, confirmation): {pc['wtp_pct']:.1f}%"
        + (f"  (~${pc['wtp_usd']:.2f})" if pc["wtp_usd"] is not None else ""),
        f"   trial-level bootstrap 95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]",
        f"   local-crossing (model-free, primary): {report['local_crossing']['wtp_pct']:.1f}% "
        f"[{report['local_crossing']['bootstrap']['ci95'][0]*100:.1f}%, "
        f"{report['local_crossing']['bootstrap']['ci95'][1]*100:.1f}%]"
        f"  (preferred, esp. when the curve is not S-shaped)",
        f"   (naive, ignoring position: {report['naive']['wtp_pct']:.1f}%)",
        "",
        "Per pair (position-controlled, 95% CI; [!] = position quasi-separation):",
    ]
    for pp in report["per_pair"]:
        lo2, hi2 = pp["wtp_ci_pct"]
        flag = "  [!]" if pp["quasi_separation"] else ""
        lines.append(
            f"   {pp['pair_id']} (${pp['base_price']:.2f}): "
            f"{pp['wtp_pos_controlled_pct']:.1f}% [{lo2:.1f}%, {hi2:.1f}%] "
            f"(b_pos={pp['b_pos']:.1f}){flag}"
        )
    if report.get("plot"):
        lines.append(f"\nPlot: {report['plot']}")
    if report.get("diagnostic_plot"):
        lines.append(f"Diagnostic: {report['diagnostic_plot']}")
    return "\n".join(lines)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="results CSV from experiment.py / run_m4.py / run_robustness.py")
    p.add_argument("--base-price", type=float, default=None,
                   help="base price for $-denominated PE (e.g. 129.99)")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    print(_pretty(analyze(a.csv, base_price=a.base_price, plot=not a.no_plot)))
