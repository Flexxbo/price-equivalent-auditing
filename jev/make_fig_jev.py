"""Figure for the Jev results (Jev appendix figure): argmax choice rate of the cued product vs. price
premium, split by display position of the cued card (shown first vs. shown second).

Source: jev/_out/<run>/trials.jsonl (raw log; error rows skipped). Per pair mean over the rows
of one slot at each premium, then mean over the 6 pairs (36 trials per point in v1: 6 pairs x
2 product assignments x 3 repeats; 72 in v2). Price-only control: matched-pair rows (product A vs
product B, the main paper's design); in v1 the two-copies-of-A rows are left out.

Usage:  python jev/make_fig_jev.py                                   (v1, social proof strong + authority)
        python jev/make_fig_jev.py jev_audit_v2                      (aligned run, same two cues)
        python jev/make_fig_jev.py jev_audit_v2 sp_vstrong,authority (choose the two cues)
Paper figure: python jev/make_fig_jev.py jev_audit_v2 sp_vstrong,authority
Writes: figures/fig_jev_position[_<run>].pdf (+ .png preview), relative to the release root
"""
import json, os, statistics as st, collections, math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUN = sys.argv[1] if len(sys.argv) > 1 else "jev_audit_v1"
LOG = HERE / "_out" / RUN / "trials.jsonl"
FIG_DIR = HERE.parent / "figures"
os.makedirs(FIG_DIR, exist_ok=True)
OUT = FIG_DIR / (
    "fig_jev_position" if RUN == "jev_audit_v1" else f"fig_jev_position_{RUN}")

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8a85"
INK, INK2 = "#1f1f1e", "#5f5e58"
NAMES = {"sp_weak": "Social proof (weak)", "sp_medium": "Social proof (moderate)",
         "sp_strong": "Social proof (strong)", "sp_vstrong": "Social proof (very strong)",
         "authority": "Authority", "scarcity": "Scarcity", "framing_gainloss": "Gain/loss framing",
         "neutral": "Neutral sentence", "sp_strong_lenctrl": "Social proof, length-matched"}
_sel = (sys.argv[2].split(",") if len(sys.argv) > 2 else ["sp_strong", "authority"])[:2]
CUES = [(c, NAMES.get(c, c), col, mk) for c, col, mk in zip(_sel, (BLUE, ORANGE), ("o", "s"))]
if len(sys.argv) > 2:
    OUT = OUT.with_name(OUT.name + "_" + "_".join(_sel))
CATS = [("headphones", "Headphones"), ("coffee_machines", "Coffee machines")]


def load():
    rows = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if not r.get("error") and r.get("category")]


def curve(sel, key="choice_treat"):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in sel:
        per[r["pair_id"]][r["g"]].append(r[key])
    gs = sorted({r["g"] for r in sel})
    ys = [st.fmean([st.fmean(per[p][g]) for p in per if per[p][g]]) for g in gs]
    return [g * 100 for g in gs], ys


# x axis: log(1 + premium in %) keeps the fine low grid (0.5/1/1.5 %) readable up to 60 %
def _f(x):
    import numpy as np
    return np.log1p(np.clip(x, 0, None))
def _i(y):
    import numpy as np
    return np.expm1(y)


def main():
    rows = load()
    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(2, 1, figsize=(3.33, 3.9), sharex=True, sharey=True)
    for ax, (cat, cat_lbl) in zip(axes, CATS):
        ax.axhline(0.5, color=GRAY, lw=0.7, ls=(0, (2, 2)), zorder=1)
        po = [r for r in rows if r["category"] == cat and r["cue_id"] == "price_only"
              and r.get("prompt_variant", "v0") == "v0"
              and not ("stimuli" in r and r["stimuli"]["first"]["name"] == r["stimuli"]["second"]["name"])]
        x, y = curve(po)
        gmax = max(r["g"] for r in rows if r["cue_id"] != "price_only") * 100
        keep = [i for i, a in enumerate(x) if a <= gmax]
        x, y = [x[i] for i in keep], [y[i] for i in keep]
        ax.plot(x, y, color=GRAY, lw=1.2, zorder=2, label="Price-only control (pooled)")
        for cue, lbl, col, mk in CUES:
            # v2 coffee: the domain-matched sentence is 'authority_coffee' ('authority' there is the
            # off-domain audio-experts control); v1 coffee used the coffee sentence under 'authority'.
            ids = {r["cue_id"] for r in rows if r["category"] == cat}
            cid = cue + "_coffee" if (cue + "_coffee") in ids else cue
            sel = [r for r in rows if r["category"] == cat and r["cue_id"] == cid
                   and r.get("prompt_variant", "v0") == "v0"]
            for first, ls, face in [(True, "-", col), (False, (0, (4, 2)), "white")]:
                x, y = curve([r for r in sel if r["treat_first"] == first])
                ax.plot(x, y, color=col, lw=1.5, ls=ls, marker=mk, ms=3.6, mfc=face,
                        mec=col, mew=1.0, zorder=3)
        ax.set_xscale("function", functions=(_f, _i))
        ax.set_xlim(-0.03, gmax * 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(axis="y", color="#e6e5df", lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.text(0.99, 0.97, cat_lbl, transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=INK, fontweight="bold")
        ax.set_ylabel("Cued product chosen")
    ticks = [t for t in [0, 0.5, 1, 2, 4, 8, 12, 26, 60] if t <= gmax + 1e-9]
    axes[1].set_xticks(ticks)
    axes[1].set_xticklabels([f"{t:g}" for t in ticks])
    axes[1].set_xlabel("Price premium on cued product (%)")

    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=3.6), Line2D([], [], color=ORANGE, lw=1.5, marker="s", ms=3.6),
         Line2D([], [], color=GRAY, lw=1.2),
         Line2D([], [], color=INK2, lw=1.2, ls="-", marker="o", ms=3.6),
         Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), marker="o", ms=3.6, mfc="white")]
    l = [CUES[0][1], CUES[1][1], "Price-only control", "cued card shown first", "cued card shown second"]
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=6.5, handlelength=2.2,
               columnspacing=0.8, bbox_to_anchor=(0.52, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.86), h_pad=0.6)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
