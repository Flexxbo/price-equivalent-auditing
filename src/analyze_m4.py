"""Main-audit diagnostic summary across cue conditions.

Reads all m4_*.csv runs and reports, per condition, the position-controlled
price-equivalent (PE) with a bootstrap CI, the baseline P(cued) at equal price, and the
first-slot rate. It then checks the three validity questions:
  1. Dose-response: does the PE rise monotonically with social-proof intensity?
  2. Controls: are neutral / vacuous cues near zero?
  3. Framing: do same-information / different-form cues move the PE (a pure form effect)?

Uses the additive-logistic cross-check with a trial-level bootstrap. The paper's
primary PEs and pair-clustered CIs come from make_paper_numbers.py.

Run:
    python src/analyze_m4.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402

import analysis  # noqa: E402
from cues import DOSE_FAMILY, get_cue  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "data" / "results"


def _cid_from(df) -> str:
    cond = str(df["condition"].iloc[0])
    return cond[3:] if cond.startswith("m4_") else cond


def summarize(csv_path: str) -> dict:
    df = analysis.load_results(csv_path)
    cid = _cid_from(df)
    pc = analysis.position_controlled_wtp(df)
    boot = analysis.position_controlled_bootstrap(df, n_boot=2000)
    base = df[df["price_gap"] == 0.0]["chose_nudged"].mean()
    try:
        cue = get_cue(cid)
        ctype, intensity = cue.type, cue.intensity
    except Exception:
        ctype, intensity = "?", None
    return {
        "cid": cid, "type": ctype, "intensity": intensity,
        "wtp": pc["wtp"] * 100, "ci": [boot["ci95"][0] * 100, boot["ci95"][1] * 100],
        "baseline": base, "p_first": df["picked_primary"].mean(), "n": len(df),
    }


def main() -> int:
    files = glob.glob(str(RESULTS / "**" / "m4_*.csv"), recursive=True)
    files = [f for f in files if "_mock_" not in os.path.basename(f)]  # ignore mock dry-runs
    files.sort(key=os.path.getmtime)  # newest last -> the latest run wins per condition
    if not files:
        print("No real m4_*.csv found. Run: python src/run_m4.py --model gpt-4o")
        return 1
    rows = {}
    for f in files:
        s = summarize(f)
        rows[s["cid"]] = s  # latest wins if duplicates

    def line(s):
        lo, hi = s["ci"]
        return (f"  {s['cid']:18s} [{s['type']:11s}] PE={s['wtp']:6.1f}%  "
                f"CI[{lo:5.1f},{hi:5.1f}]  base@0={s['baseline']:.2f}  "
                f"P(first)={s['p_first']:.2f}  n={s['n']}")

    print("=== Main-audit condition summary (position-controlled PE) ===")
    for s in sorted(rows.values(), key=lambda r: -r["wtp"]):
        print(line(s))

    # 1) Dose-response monotonicity
    print("\n=== 1. Dose-response (social proof intensity) ===")
    dose = [rows[c] for c in DOSE_FAMILY if c in rows]
    if len(dose) >= 2:
        dose.sort(key=lambda r: r["intensity"])
        seq = [round(d["wtp"], 1) for d in dose]
        print("  intensity ->", " < ".join(f"{d['cid']}={d['wtp']:.1f}%" for d in dose))
        mono = all(b >= a for a, b in zip(seq, seq[1:]))
        print(f"  monotonically non-decreasing: {mono}  (expected if PE tracks cue intensity)")
        _plot_dose(dose)
    else:
        print("  (need the sp_* conditions; run them first)")

    # 2) Controls near zero
    print("\n=== 2. Controls (should be near 0) ===")
    for c in ("neutral", "vacuous"):
        if c in rows:
            print("  " + line(rows[c]).strip())

    # 3) Framing at constant information -- judge by the equal-price preference,
    # not the extrapolated PE (which degenerates when base@0 ~ 0.50).
    print("\n=== 3. Framing at constant information ===")
    print("    base@0 = P(form A chosen at equal price); 0.50 = no form preference.")
    for c in ("framing_pct_frac", "framing_rating", "framing_gainloss"):
        if c in rows:
            s = rows[c]
            base = s["baseline"]
            verdict = ("prefers form A" if base > 0.55 else
                       "prefers form B" if base < 0.45 else "NO clear form preference")
            print(f"  {s['cid']:18s} base@0={base:.2f} -> {verdict}"
                  f"   (extrapolated PE={s['wtp']:.1f}%, treat as artifact if base~0.5)")
    return 0


def _plot_dose(dose):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [d["intensity"] for d in dose]
        ys = [d["wtp"] for d in dose]
        err = [[d["wtp"] - d["ci"][0] for d in dose], [d["ci"][1] - d["wtp"] for d in dose]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(xs, ys, yerr=err, fmt="o-", capsize=4, color="#2b6cb0")
        ax.set_xticks(xs)
        ax.set_xticklabels([d["cid"] for d in dose], rotation=15)
        ax.set_ylabel("Price-equivalent (%)")
        ax.set_title("Dose-response: PE vs social-proof intensity")
        ax.axhline(0, ls=":", color="gray")
        fig.tight_layout()
        out = RESULTS / "m4_dose_response.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  plot: {out}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


if __name__ == "__main__":
    raise SystemExit(main())
