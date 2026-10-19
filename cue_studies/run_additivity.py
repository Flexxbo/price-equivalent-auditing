"""Cue-combination additivity (App. H): a length-matched 2^3 factorial.

Three families, one cue each (Social Proof, Authority, Scarcity), each in a fixed SLOT that
carries EITHER the family's cue OR a fixed neutral filler. Every condition therefore has exactly
three appended sentences, so the length/presence confound cancels in every interaction contrast.

8 conditions per pair = all subsets of {SP, Auth, Scar}. From these we get all pairwise
interactions (SP:Auth, SP:Scar, Auth:Scar) and the three-way (SP:Auth:Scar).

Two read-outs:
  * PE scale (primary, interpretable): per-condition model-free PE; interaction contrasts
    (Delta PE) relative to the all-filler baseline, mean and 95% t-CI over pairs. Censored cells
    enter at the grid ceiling.
  * Log-odds scale (ceiling-robust cross-check): one Firth-penalised logistic pooled over the
    trials, P = sigma(b0 + bg*g + bpos*pos + main effects + all interactions); interaction
    coefficients with a pair-cluster bootstrap CI. Firth handles near-separation from strong combos.

Decision rule: an interaction counts as present if its Firth coefficient's 95% pair-cluster
bootstrap CI excludes 0 after Holm correction over the four interaction terms.

Mock (0 API):  python cue_studies/run_additivity.py --mock --pairs 2
Real:          python cue_studies/run_additivity.py --eval-model gpt-4o --pairs 6 --n 24 \
                   --max-budget 14 --yes
"""
from __future__ import annotations
import argparse, itertools, math, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from models import get_model                                    # noqa: E402
from products import resolve_category, load_pairs               # noqa: E402
from budget import BudgetGuard, BudgetExceeded, estimate_chat, preflight  # noqa: E402
from cues import CUES                                           # noqa: E402
import gate as gate_mod                                         # noqa: E402
import pe_eval                                                  # noqa: E402
import checkpoint as ckpt_mod                                   # noqa: E402
import manifest as manifest_mod                                 # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "data", "results", "cue_studies")
os.makedirs(OUT_DIR, exist_ok=True)
FIXED_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.26, 0.40, 0.60]   # shared, extended
FAMILIES = ["SP", "Auth", "Scar"]
# fixed neutral filler per slot (value-free, ~0% PE in the main audit); they cancel in interactions
FILLERS = {"SP": "Each unit has a printed serial number on the housing.",
           "Auth": "The packaging carries a printed barcode.",
           "Scar": "A printed user manual is included in the box."}
DEFAULT_CUES = {"SP": "mo_sp_med", "Auth": "mo_auth_weak", "Scar": "mo_scar_med"}
_TCRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _fin(x): return x is not None and not (isinstance(x, float) and math.isnan(x))


class _Progress:
    def __init__(self, total): self.total = max(1, int(total)); self.t0 = time.time()
    def tick(self, done, label=""):
        el = time.time() - self.t0; frac = min(0.999, done / self.total)
        eta = el * (1 - frac) / frac if frac > 0 else 0.0
        print(f"[add] {label:26s} | {done:5d}/{self.total} ({frac*100:4.1f}%) | "
              f"{el:5.0f}s elapsed | ETA ~{eta:5.0f}s", flush=True)


def firth_logit(X, y, iters=100, tol=1e-7):
    """Firth (Jeffreys-penalised) logistic MLE, stable under separation."""
    n, k = X.shape; b = np.zeros(k)
    for _ in range(iters):
        eta = X @ b; p = 1 / (1 + np.exp(-eta)); p = np.clip(p, 1e-9, 1 - 1e-9)
        W = p * (1 - p)
        XtWX = (X.T * W) @ X + 1e-8 * np.eye(k)
        try: inv = np.linalg.inv(XtWX)
        except np.linalg.LinAlgError: break
        h = W * np.sum((X @ inv) * X, axis=1)          # hat diag
        U = X.T @ (y - p + h * (0.5 - p))              # Firth-adjusted score
        step = inv @ U; b = b + step
        if np.max(np.abs(step)) < tol: break
    return b


def build_add_pool(pair, cue_ids):
    facts_text = pair.product_a.neutral_description
    import re
    fnums = [m.group() for m in re.finditer(r"\d[\d,\.]*", facts_text)]
    cues = []
    for fam, cid in cue_ids.items():
        t = CUES[cid].cued_text
        cues.append({"id": cid, "text": t, "provenance": "stipulated", "required": False,
                     "numbers": [m.group() for m in re.finditer(r"\d[\d,\.]*", t)], "keywords": []})
    pool = {"facts": [{"id": "facts", "text": facts_text, "provenance": "fact", "required": False,
                       "numbers": fnums, "keywords": []}], "cues": cues}
    return facts_text, pool


def condition_text(facts_text, present, cue_ids):
    """present = set of families carrying their cue; others carry the fixed filler. Fixed slot order."""
    parts = []
    for fam in FAMILIES:
        parts.append(CUES[cue_ids[fam]].cued_text if fam in present else FILLERS[fam])
    return facts_text.strip() + " " + " ".join(parts)


def subsets():
    for r in range(4):
        for combo in itertools.combinations(FAMILIES, r):
            yield frozenset(combo)


def _mean_ci(vals):
    v = [x for x in vals if _fin(x)]; n = len(v)
    if n == 0: return None, None, None, 0
    m = sum(v) / n
    if n == 1: return m, None, None, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    half = _TCRIT.get(n - 1, 2.1) * sd / math.sqrt(n)
    return m, m - half, m + half, n


def run(argv=None):
    p = argparse.ArgumentParser(description="Cue-combination additivity (length-matched 2^3)")
    p.add_argument("--category", default="headphones"); p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--eval-model", default="gpt-4o"); p.add_argument("--mock", action="store_true")
    p.add_argument("--n", type=int, default=24); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=6); p.add_argument("--max-budget", type=float, default=14.0)
    p.add_argument("--run-id", default="additivity"); p.add_argument("--yes", action="store_true")
    p.add_argument("--sp", default=DEFAULT_CUES["SP"]); p.add_argument("--auth", default=DEFAULT_CUES["Auth"])
    p.add_argument("--scar", default=DEFAULT_CUES["Scar"]); p.add_argument("--nboot", type=int, default=2000)
    a = p.parse_args(argv)
    cue_ids = {"SP": a.sp, "Auth": a.auth, "Scar": a.scar}
    pairs = load_pairs(resolve_category(a.category))[:a.pairs]
    conds = list(subsets())                                   # 8
    conc = 1 if a.mock else a.concurrency
    total = int(len(pairs) * len(conds) * len(FIXED_GRID) * a.n * 1.02)
    prog = _Progress(total)
    print(f"[add] {len(pairs)} pairs x {len(conds)} conditions | ~{total} trials "
          f"(~${total/1152:.1f}) | cues={cue_ids}", flush=True)

    guard = None
    if a.mock:
        model, alias, reveal = get_model("mock"), "mock", True
    else:
        proj = estimate_chat(total, 340, 4, a.eval_model)
        if not preflight(proj, a.max_budget, label="additivity",
                         breakdown=f"{len(pairs)}x8 conditions x {len(FIXED_GRID)} pts x n={a.n}",
                         assume_yes=a.yes):
            return
        guard = BudgetGuard(a.max_budget, label="additivity")
        model, alias, reveal = get_model(a.eval_model, budget=guard), a.eval_model, False

    ckpt_path = ckpt_mod.path_for(os.path.join(OUT_DIR, f"add_{a.run_id}.md"))
    ckpt = ckpt_mod.load(ckpt_path)
    trials_ct = [0]
    pe_by_pair = []            # list of dict cond->pe
    all_rows = []              # pooled trial rows with indicators + pair id
    stopped = False
    for pi, pair in enumerate(pairs):
        facts_text, pool = build_add_pool(pair, cue_ids)
        pe_cond = {}
        try:
            for present in conds:
                text = condition_text(facts_text, present, cue_ids)
                g = gate_mod.check(text, pool, facts_text, min_words=1, max_words=100000)
                if not g.ok:
                    print(f"[add] GATE FAIL p{pi+1} {sorted(present)}: {g.reasons}"); continue
                k = ckpt_mod.key(alias, a.n, a.seed, FIXED_GRID, text)
                hk = f"{pair.pair_id}:{'+'.join(sorted(present)) or 'none'}"
                cached = ckpt.get(hk)
                if cached and cached.get("key") == k:
                    pe_cond[present] = cached["pe"]; rows = cached["rows"]
                else:
                    r = pe_eval.evaluate_candidate(text, pair, model, model_name=alias, n=a.n, seed=a.seed,
                                                   base_grid=FIXED_GRID, max_ceiling=FIXED_GRID[-1],
                                                   step=0.06, reveal_for_mock=reveal, concurrency=conc,
                                                   return_rows=True)
                    trials_ct[0] += r.get("n_trials", 0)
                    pe = r["pe"] if _fin(r["pe"]) else (r["ceiling"] if r.get("censored") else float("nan"))
                    rows = [{"g": float(x["price_gap"]), "y": float(x["chose_nudged"]),
                             "pos": 1.0 if str(x.get("nudged_position", "")).lower().startswith("l") else 0.0}
                            for x in r["rows"] if x.get("chose_nudged") is not None
                            and not (isinstance(x.get("chose_nudged"), float) and math.isnan(x["chose_nudged"]))]
                    pe_cond[present] = pe
                    ckpt[hk] = {"key": k, "pe": pe, "rows": rows}
                    ckpt_mod.save(ckpt_path, ckpt)
                    prog.tick(trials_ct[0], f"p{pi+1} {'+'.join(sorted(present)) or 'baseline'}")
                for row in rows:
                    row2 = dict(row); row2["pair"] = pi
                    row2["SP"] = 1.0 if "SP" in present else 0.0
                    row2["Auth"] = 1.0 if "Auth" in present else 0.0
                    row2["Scar"] = 1.0 if "Scar" in present else 0.0
                    all_rows.append(row2)
        except (BudgetExceeded, KeyboardInterrupt):
            stopped = True; print("[add] Stopped: partial result."); break
        pe_by_pair.append(pe_cond)

    # ---------- PE-scale interaction contrasts (per pair, clustered) ----------
    def pe(d, fams): return d.get(frozenset(fams))
    def two_way(d, a1, b1):
        vals = [pe(d, [a1, b1]), pe(d, [a1]), pe(d, [b1]), pe(d, [])]
        return (vals[0] - vals[1] - vals[2] + vals[3]) if all(_fin(v) for v in vals) else None
    def three_way(d):
        v = {fs: pe(d, list(fs)) for fs in subsets()}
        if not all(_fin(x) for x in v.values()): return None
        return (v[frozenset(FAMILIES)] - v[frozenset(["SP", "Auth"])] - v[frozenset(["SP", "Scar"])]
                - v[frozenset(["Auth", "Scar"])] + v[frozenset(["SP"])] + v[frozenset(["Auth"])]
                + v[frozenset(["Scar"])] - v[frozenset()])
    pe_int = {"SP:Auth": [], "SP:Scar": [], "Auth:Scar": [], "SP:Auth:Scar": []}
    for d in pe_by_pair:
        pe_int["SP:Auth"].append(two_way(d, "SP", "Auth"))
        pe_int["SP:Scar"].append(two_way(d, "SP", "Scar"))
        pe_int["Auth:Scar"].append(two_way(d, "Auth", "Scar"))
        pe_int["SP:Auth:Scar"].append(three_way(d))

    # ---------- Firth factorial logistic (pooled) + pair-cluster bootstrap ----------
    firth = {}
    if all_rows:
        df = pd.DataFrame(all_rows)
        terms = ["SP", "Auth", "Scar"]
        df["SP:Auth"] = df.SP * df.Auth; df["SP:Scar"] = df.SP * df.Scar
        df["Auth:Scar"] = df.Auth * df.Scar; df["SP:Auth:Scar"] = df.SP * df.Auth * df.Scar
        cols = ["g", "pos"] + terms + ["SP:Auth", "SP:Scar", "Auth:Scar", "SP:Auth:Scar"]
        def fit(frame):
            X = np.column_stack([np.ones(len(frame))] + [frame[c].values for c in cols])
            return firth_logit(X, frame["y"].values)
        b_full = fit(df)
        idx = {c: 3 + i for i, c in enumerate(cols[2:])}     # +1 intercept,+g,+pos
        rng = np.random.default_rng(a.seed)
        pair_ids = sorted(df.pair.unique())
        boots = {c: [] for c in ["SP:Auth", "SP:Scar", "Auth:Scar", "SP:Auth:Scar"]}
        nb = 400 if a.mock else a.nboot
        for _ in range(nb):
            samp = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            frame = pd.concat([df[df.pair == pid] for pid in samp], ignore_index=True)
            try: bb = fit(frame)
            except Exception: continue
            for c in boots: boots[c].append(bb[idx[c]])
        for c in ["SP:Auth", "SP:Scar", "Auth:Scar", "SP:Auth:Scar"]:
            est = b_full[idx[c]]; arr = np.array(boots[c])
            ci = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))) if len(arr) > 10 else None
            pv = 2 * min((arr < 0).mean(), (arr > 0).mean()) if len(arr) > 10 else 1.0
            firth[c] = {"beta": float(est), "ci": ci, "p": float(pv)}
        # Holm over the four interaction terms
        order = sorted(firth, key=lambda c: firth[c]["p"]); m = len(order)
        for i, c in enumerate(order):
            firth[c]["holm_sig"] = firth[c]["p"] <= 0.05 / (m - i)
            if not firth[c]["holm_sig"]:
                for c2 in order[i:]:
                    firth[c2]["holm_sig"] = False
                break

    # ---------- report ----------
    def excl0(ci): return ci is not None and (ci[0] > 0 or ci[1] < 0)
    L = [f"# Cue-combination additivity: length-matched 2^3 factorial ({a.eval_model})\n",
         f"*{len(pe_by_pair)} pairs; pair = cluster. Cues: SP=`{a.sp}`, Auth=`{a.auth}`, Scar=`{a.scar}`. "
         f"Each condition carries 3 sentences (cue or fixed neutral filler) → length cancels.*\n",
         "## Firth log-odds interactions (ceiling-robust cross-check; pair-cluster bootstrap 95% CI)",
         "| term | beta | 95% CI | p | Holm sig.? |", "|---|--:|---|--:|:--:|"]
    for c in ["SP:Auth", "SP:Scar", "Auth:Scar", "SP:Auth:Scar"]:
        f = firth.get(c, {})
        ci = f.get("ci"); cis = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "—"
        L.append(f"| {c} | {f.get('beta', float('nan')):+.2f} | {cis} | {f.get('p', float('nan')):.3f} "
                 f"| {'yes' if f.get('holm_sig') else 'no'} |")
    L += ["", "## PE-scale interaction contrasts (primary; mean ± 95% t-CI over pairs)",
          "| term | Δ (pp) | 95% CI |", "|---|--:|---|"]
    for c in ["SP:Auth", "SP:Scar", "Auth:Scar", "SP:Auth:Scar"]:
        m, lo, hi, nn = _mean_ci(pe_int[c])
        cis = f"[{lo*100:+.1f}, {hi*100:+.1f}]" if lo is not None else "(n<2)"
        L.append(f"| {c} | {m*100:+.1f} | {cis} |" if m is not None else f"| {c} | — | — |")
    any_int = any(firth.get(c, {}).get("holm_sig") for c in firth)
    L += ["", "### Decision rule (Firth CI excludes 0 after Holm correction)",
          f"- **{'Interaction present: at least one Firth term is Holm-significant, so combinations are not purely additive.' if any_int else 'Consistent with additivity at current precision: no Firth interaction survives Holm correction.'}**",
          "*Length cancels by design (fixed filler per slot). Conditional on these pairs and the evaluator model.*"]
    if stopped: L.append("\n*(stopped early: partial)*")
    est_cost = trials_ct[0] / 1152.0
    L.append(f"\n**Accounting:** {trials_ct[0]} evaluator trials ≈ ${est_cost:.2f}" + (" [MOCK]" if a.mock else ""))
    Path(os.path.join(OUT_DIR, f"summary_{a.run_id}.md")).write_text("\n".join(L), encoding="utf-8")

    manifest_mod.write(os.path.join(OUT_DIR, f"manifest_{a.run_id}.json"),
                       manifest_mod.build(run_id=a.run_id, model_aliases=(["mock"] if a.mock else [a.eval_model]),
                                          cfg={"stage": "additivity", "pairs": len(pe_by_pair), "n": a.n,
                                               "cues": cue_ids, "grid": FIXED_GRID},
                                          pool=build_add_pool(pairs[0], cue_ids)[1] if pairs else {}, guard=guard,
                                          extra={"firth": firth, "trials": trials_ct[0], "any_interaction": bool(any_int)}))
    print(f"\n[add] interactions: " + " | ".join(
        f"{c} {firth.get(c, {}).get('beta', float('nan')):+.2f}" for c in
        ["SP:Auth", "SP:Scar", "Auth:Scar", "SP:Auth:Scar"]))
    print(f"[add] {trials_ct[0]} trials ≈ ${est_cost:.2f}{' [MOCK]' if a.mock else ''}")
    print(f"[add] Summary: {os.path.join(OUT_DIR, f'summary_{a.run_id}.md')}")
    if guard is not None: print(f"[add] Spend: ${guard.spent:.3f} / cap ${guard.max:.0f}")


if __name__ == "__main__":
    run()
