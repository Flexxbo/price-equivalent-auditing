"""PE-guided cue selection (App. I), with an optional truthful-rewrite step.

Does PE-guided cue selection beat a cue chosen without PE, and does a truthfulness-constrained
rewrite of the selected cue change its PE? Reports TWO gains per product pair, clustered over
pairs (pair = generalization unit).

Per pair:  select -> rewrite -> FREEZE -> independent out-of-sample (OOS) confirmation.
  Step A) Screen the 9 cues (single-cue candidates) at a coarse grid, seed A. Pick argmax PE.
          Baseline = mean PE of the 9 (unbiased; a random draw is also recorded for reference).
  Step B) Rewrite the selected cue into R truthful wordings (facts+cue fixed), gate them,
          screen at seed A, pick the best rewrite.
  Step C) FREEZE the two winners and re-measure them out-of-sample (fine grid, fresh seed B,
          bootstrap CIs). The baseline stays the Step A mean (not selected -> no winner's curse).

  selection_gain = PE_oos(selected cue) - mean_A(9 cues)
  rewrite_gain   = PE_oos(best rewrite) - PE_oos(selected cue)

The out-of-sample PEs come from confirm.confirm_candidate (logistic PE on the fine grid); the
mean(9) baseline is the model-free screening PE.

Aggregate: mean +/- 95% t-CI over the P pairs; a gain is reported as positive iff its lower 95%
CI bound > 0. The gains are read only from the frozen, OOS-confirmed winners; screening PEs are
never reported as the result.

Mock (0 API):  python cue_studies/run_minopt.py --mock
Real:          python cue_studies/run_minopt.py --eval-model gpt-4o --gen-model claude-sonnet \
                   --pairs 5 --rewrites 5 --screen-n 10 --confirm-n 24 --max-budget 8 --yes
"""
from __future__ import annotations
import argparse, math, os, random, re, sys, time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "src"))
from models import get_model                                    # noqa: E402
from products import resolve_category, load_pairs               # noqa: E402
from budget import BudgetGuard, BudgetExceeded, estimate_chat, preflight  # noqa: E402
from generator import LLMGenerator, compose                     # noqa: E402
from cues import CUES, MINOPT_CUES                              # noqa: E402
import gate as gate_mod                                         # noqa: E402
import pe_eval                                                  # noqa: E402
import confirm as confirm_mod                                   # noqa: E402
import checkpoint as ckpt_mod                                   # noqa: E402
import manifest as manifest_mod                                 # noqa: E402

OUT_DIR = os.path.join(HERE, "..", "data", "results", "cue_studies")
os.makedirs(OUT_DIR, exist_ok=True)
SCREEN_GRID = [0.0, 0.04, 0.08, 0.12]           # coarse: selection only
# two-sided 95% t critical values by df (fallback 1.96 for df>=30)
_TCRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
          7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086}


def _fin(x): return x is not None and not (isinstance(x, float) and math.isnan(x))


def _tcrit(df): return _TCRIT.get(df, 2.09 if df >= 11 else 12.71) if df < 30 else 1.96


def _mean_ci(vals):
    """mean and 95% t-CI over the per-pair values (cluster = pair)."""
    v = [x for x in vals if _fin(x)]
    n = len(v)
    if n == 0: return None, None, None, 0
    m = sum(v) / n
    if n == 1: return m, None, None, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    half = _tcrit(n - 1) * sd / math.sqrt(n)
    return m, m - half, m + half, n


def build_minopt_pool(pair):
    """Pool = product facts + the 9 minopt cues (numbers registered for the gate)."""
    facts_text = pair.product_a.neutral_description
    fnums = [m.group() for m in re.finditer(r"\d[\d,\.]*", facts_text)]
    cues = []
    for cid in MINOPT_CUES:
        t = CUES[cid].cued_text
        cues.append({"id": cid, "text": t, "provenance": "stipulated", "required": False,
                     "numbers": [m.group() for m in re.finditer(r"\d[\d,\.]*", t)], "keywords": []})
    pool = {"facts": [{"id": "facts", "text": facts_text, "provenance": "fact",
                       "required": False, "numbers": fnums, "keywords": []}],
            "cues": cues}
    return facts_text, pool


def _wc(t): return len(t.split())


class _Counter:
    def __init__(self): self.trials = 0
    def add(self, k): self.trials += int(k or 0)


class _Progress:
    """Rough live status with ETA based on trials done vs. an analytical total."""
    def __init__(self, total): self.total = max(1, int(total)); self.t0 = time.time()
    def tick(self, done, label=""):
        el = time.time() - self.t0
        frac = min(0.999, done / self.total)
        eta = el * (1 - frac) / frac if frac > 0 else 0.0
        print(f"[minopt] {label:30s} | {done:5d}/{self.total} trials ({frac*100:4.1f}%) | "
              f"elapsed {el:5.0f}s | ETA ~{eta:5.0f}s", flush=True)


def _screen_pe(text, pair, model, *, alias, n, seed, reveal, conc, ckpt, ctr, prog=None, label=""):
    """Coarse screening PE with checkpointing; returns a PE float (censored -> ceiling)."""
    k = ckpt_mod.key(alias, n, seed, SCREEN_GRID, text)
    hk = "S:" + text[:70]
    c = ckpt.get(hk)
    if c and c.get("key") == k:
        return c["pe"]
    r = pe_eval.evaluate_candidate(text, pair, model, model_name=alias, n=n, seed=seed,
                                   base_grid=SCREEN_GRID, reveal_for_mock=reveal, concurrency=conc)
    ctr.add(r.get("n_trials"))
    if prog: prog.tick(ctr.trials, label)
    pe = r["pe"] if _fin(r["pe"]) else (r["ceiling"] if r.get("censored") else float("nan"))
    ckpt[hk] = {"key": k, "pe": pe}
    return pe


def _confirm_pe(text, pair, model, *, alias, n, seed, reveal, conc, ckpt, ctr, prog=None, label=""):
    """Fine-grid OOS PE with bootstrap CI; checkpointed. Returns {'pe','ci'}."""
    k = ckpt_mod.key(alias, n, seed, confirm_mod.FINE_GRID, text)
    hk = "C:" + text[:70]
    c = ckpt.get(hk)
    if c and c.get("key") == k:
        return c["res"]
    res = confirm_mod.confirm_candidate(text, pair, model, model_name=alias, n=n, seed=seed,
                                        reveal=reveal, concurrency=conc)
    ctr.add(len(confirm_mod.FINE_GRID) * n)     # accounting (adaptive extension may add a little)
    if prog: prog.tick(ctr.trials, label)
    out = {"pe": res["pe"], "ci": res.get("ci")}
    ckpt[hk] = {"key": k, "res": out}
    return out


def run_pair_experiment(pair, idx, *, eval_model, gen_ctor, alias, reveal, conc, rewrites,
                        screen_n, confirm_n, seed_a, seed_b, rand_seed, ckpt, ctr, mock,
                        prog=None, rewriter_style="clean"):
    facts_text, pool = build_minopt_pool(pair)
    tag = f"p{idx+1}"

    # --- Step A: screen the 9 single-cue candidates, pick strongest; baseline = mean ---
    cue_texts = {cid: compose(facts_text, pool, [cid], random.Random(1)) for cid in MINOPT_CUES}
    pe_a = {}
    for cid, txt in cue_texts.items():
        g = gate_mod.check(txt, pool, facts_text, min_words=1, max_words=100000)
        pe_a[cid] = _screen_pe(txt, pair, eval_model, alias=alias, n=screen_n, seed=seed_a,
                               reveal=reveal, conc=conc, ckpt=ckpt, ctr=ctr,
                               prog=prog, label=f"{tag} A {cid}") if g.ok else float("nan")
    finite = {cid: v for cid, v in pe_a.items() if _fin(v)}
    if not finite:
        return {"pair": pair.pair_id, "error": "no finite cue PE"}
    selected_cid = max(finite, key=finite.get)
    mean_a = sum(finite.values()) / len(finite)
    rnd = random.Random(rand_seed + idx)
    random_cid = rnd.choice([c for c in finite])          # reference only; baseline = mean

    # --- Step B: rewrite the selected cue, gate, screen, pick best ---
    sel_text = cue_texts[selected_cid]
    lo_w, hi_w = max(6, int(0.7 * _wc(sel_text))), int(1.4 * _wc(sel_text))
    if mock:
        rw_texts = [sel_text for _ in range(rewrites)]     # 0-API: identical (wiring only)
    else:
        gen = gen_ctor(facts_text, pool)
        rw_texts = [w["text"] for w in gen.propose_fixed(rewrites, [selected_cid], style=rewriter_style)]
    rw_scored = []
    for j, t in enumerate(rw_texts):
        g = gate_mod.check(t, pool, facts_text, min_words=lo_w, max_words=hi_w)
        if not g.ok:
            continue
        rw_scored.append((t, _screen_pe(t, pair, eval_model, alias=alias, n=screen_n, seed=seed_a,
                                        reveal=reveal, conc=conc, ckpt=ckpt, ctr=ctr,
                                        prog=prog, label=f"{tag} B rewrite {j+1}/{len(rw_texts)}")))
    rw_valid = [(t, p) for t, p in rw_scored if _fin(p)]
    best_rw_text = max(rw_valid, key=lambda x: x[1])[0] if rw_valid else sel_text

    # --- Step C: freeze the two winners, OOS confirm on a fresh seed ---
    oos_sel = _confirm_pe(sel_text, pair, eval_model, alias=alias, n=confirm_n, seed=seed_b,
                          reveal=reveal, conc=conc, ckpt=ckpt, ctr=ctr,
                          prog=prog, label=f"{tag} C confirm cue")
    oos_rw = _confirm_pe(best_rw_text, pair, eval_model, alias=alias, n=confirm_n, seed=seed_b,
                         reveal=reveal, conc=conc, ckpt=ckpt, ctr=ctr,
                         prog=prog, label=f"{tag} C confirm rewrite")

    sel_gain = (oos_sel["pe"] - mean_a) if _fin(oos_sel["pe"]) else None
    rw_gain = (oos_rw["pe"] - oos_sel["pe"]) if _fin(oos_rw["pe"]) and _fin(oos_sel["pe"]) else None
    return {"pair": pair.pair_id, "selected_cue": selected_cid, "random_cue": random_cid,
            "mean_a": mean_a, "n_valid_rewrites": len(rw_valid),
            "pe_oos_selected": oos_sel["pe"], "pe_oos_rewrite": oos_rw["pe"],
            "selection_gain": sel_gain, "rewrite_gain": rw_gain}


def main(argv=None):
    p = argparse.ArgumentParser(description="PE-guided cue selection: selection gain and rewrite gain")
    p.add_argument("--category", default="headphones")
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--eval-model", default="gpt-4o"); p.add_argument("--gen-model", default="claude-sonnet")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--rewrites", type=int, default=5)
    p.add_argument("--screen-n", type=int, default=10); p.add_argument("--confirm-n", type=int, default=24)
    p.add_argument("--seed-a", type=int, default=0); p.add_argument("--seed-b", type=int, default=1000)
    p.add_argument("--rand-seed", type=int, default=7)
    p.add_argument("--concurrency", type=int, default=6); p.add_argument("--max-budget", type=float, default=8.0)
    p.add_argument("--run-id", default="minopt"); p.add_argument("--yes", action="store_true")
    p.add_argument("--rewriter-style", default="clean", choices=["clean", "strict"],
                   help="clean = persuasive presentation allowed (no new claims); strict = only rephrase")
    a = p.parse_args(argv)

    pairs = load_pairs(resolve_category(a.category))[:a.pairs]
    conc = 1 if a.mock else a.concurrency
    ctr = _Counter()
    total_expected = int((a.pairs * (9 + a.rewrites) * len(SCREEN_GRID) * a.screen_n
                          + a.pairs * 2 * len(confirm_mod.FINE_GRID) * a.confirm_n) * 1.12)
    prog = _Progress(total_expected)
    print(f"[minopt] style={a.rewriter_style} | ~{total_expected} trials expected "
          f"(~${total_expected/1152:.1f}); live status below.", flush=True)

    guard = None
    if a.mock:
        eval_model, alias, reveal = get_model("mock"), "mock", True
        gen_ctor = None
    else:
        # rough preflight: P*(9 cues + R rewrites) screens (4 pts x screen_n) + P*2 confirms (11 pts x confirm_n)
        screen_trials = a.pairs * (9 + a.rewrites) * len(SCREEN_GRID) * a.screen_n
        confirm_trials = a.pairs * 2 * len(confirm_mod.FINE_GRID) * a.confirm_n
        proj = estimate_chat(screen_trials + confirm_trials, 340, 4, a.eval_model) \
            + estimate_chat(a.pairs, 400, 120 * a.rewrites, a.gen_model)
        bd = f"{a.pairs} pairs x (9 cues + {a.rewrites} rewrites) screen + {a.pairs}x2 OOS confirm"
        if not preflight(proj, a.max_budget, label="minopt", breakdown=bd, assume_yes=a.yes):
            return
        guard = BudgetGuard(a.max_budget, label="minopt")
        eval_model, alias, reveal = get_model(a.eval_model, budget=guard), a.eval_model, False
        gmodel = get_model(a.gen_model, budget=guard)
        gen_ctor = lambda facts, pool: LLMGenerator(gmodel, facts, pool, seed=a.seed_a)

    ckpt_path = ckpt_mod.path_for(os.path.join(OUT_DIR, f"minopt_{a.run_id}.md"))
    ckpt = ckpt_mod.load(ckpt_path)
    rows, stopped = [], False
    for i, pair in enumerate(pairs):
        try:
            r = run_pair_experiment(pair, i, eval_model=eval_model, gen_ctor=gen_ctor, alias=alias,
                                    reveal=reveal, conc=conc, rewrites=a.rewrites, screen_n=a.screen_n,
                                    confirm_n=a.confirm_n, seed_a=a.seed_a, seed_b=a.seed_b,
                                    rand_seed=a.rand_seed, ckpt=ckpt, ctr=ctr, mock=a.mock,
                                    prog=prog, rewriter_style=a.rewriter_style)
            ckpt_mod.save(ckpt_path, ckpt)
            rows.append(r)
            print(f"[minopt] {pair.pair_id}: sel={r.get('selected_cue')} "
                  f"selGain={r.get('selection_gain')} rwGain={r.get('rewrite_gain')}")
        except (BudgetExceeded, KeyboardInterrupt):
            stopped = True; print("[minopt] Stopped: partial result."); break

    # --- aggregate over pairs (pair = cluster) ---
    sel_gains = [r["selection_gain"] for r in rows if _fin(r.get("selection_gain"))]
    rw_gains = [r["rewrite_gain"] for r in rows if _fin(r.get("rewrite_gain"))]
    sm, slo, shi, sn = _mean_ci(sel_gains)
    rm, rlo, rhi, rn = _mean_ci(rw_gains)

    def pct(x): return "—" if not _fin(x) else f"{x*100:+.1f}"
    L = [f"# PE-guided cue selection: selection gain and rewrite gain ({a.eval_model})\n",
         f"*{len(rows)} pairs; pair = cluster. Screen@seed {a.seed_a} (select), OOS confirm@seed "
         f"{a.seed_b} (frozen winners). Baseline = mean PE of the 9 cues.*\n",
         "| Pair | selected cue | mean(9) | PE_oos sel | PE_oos rewrite | sel gain | rewrite gain |",
         "|---|---|--:|--:|--:|--:|--:|"]
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['pair']} | ERROR: {r['error']} | | | | | |"); continue
        L.append(f"| {r['pair']} | {r['selected_cue']} | {r['mean_a']*100:.1f} | "
                 f"{pct(r['pe_oos_selected'])[1:] if _fin(r['pe_oos_selected']) else '—'} | "
                 f"{pct(r['pe_oos_rewrite'])[1:] if _fin(r['pe_oos_rewrite']) else '—'} | "
                 f"{pct(r['selection_gain'])} | {pct(r['rewrite_gain'])} |")
    def ci_str(m, lo, hi, n):
        if m is None: return "—"
        if lo is None: return f"{m*100:+.1f} pp (n=1)"
        return f"{m*100:+.1f} pp  95% CI [{lo*100:+.1f}, {hi*100:+.1f}] (n={n})"
    def verdict(lo, hi):
        if lo is None: return "n<2 (no CI)"
        if lo > 0: return "positive (lower 95% CI bound > 0)"
        if hi < 0: return "significantly NEGATIVE"
        return "not separated from 0"
    sel_go = slo is not None and slo > 0
    rw_go = rlo is not None and rlo > 0
    L += ["",
          f"**Selection gain (PE-selected cue − mean(9)):** {ci_str(sm, slo, shi, sn)} → "
          f"{verdict(slo, shi)}",
          f"**Rewrite gain (best rewrite − selected cue):** {ci_str(rm, rlo, rhi, rn)} → "
          f"{verdict(rlo, rhi)}",
          "",
          "*A gain is positive if the lower 95% t-CI bound over pairs is > 0. Gains read only from "
          "frozen, OOS-confirmed winners. Conditional on these pairs and the evaluator model.*"]
    if stopped: L.append("\n*(stopped early: partial)*")
    est_cost = ctr.trials / 1152.0
    L.append(f"\n**Accounting:** {ctr.trials} evaluator trials ≈ ${est_cost:.2f}"
             + (" [MOCK]" if a.mock else ""))
    L.insert(2, f"*Rewriter style: **{a.rewriter_style}**.*\n")
    Path(os.path.join(OUT_DIR, f"summary_{a.run_id}_{a.rewriter_style}.md")).write_text("\n".join(L), encoding="utf-8")

    manifest_mod.write(os.path.join(OUT_DIR, f"manifest_{a.run_id}_{a.rewriter_style}.json"),
                       manifest_mod.build(run_id=a.run_id,
                                          model_aliases=(["mock"] if a.mock else [a.eval_model, a.gen_model]),
                                          cfg={"stage": "minopt", "pairs": len(rows), "rewrites": a.rewrites,
                                               "screen_n": a.screen_n, "confirm_n": a.confirm_n,
                                               "seed_a": a.seed_a, "seed_b": a.seed_b},
                                          pool=build_minopt_pool(pairs[0])[1] if pairs else {},
                                          guard=guard,
                                          extra={"sel_gain_mean": sm, "sel_go": bool(sel_go),
                                                 "rw_gain_mean": rm, "rw_go": bool(rw_go),
                                                 "trials": ctr.trials}))
    print(f"\n[minopt] selection gain {ci_str(sm, slo, shi, sn)} | rewrite gain {ci_str(rm, rlo, rhi, rn)}")
    print(f"[minopt] {ctr.trials} trials ≈ ${est_cost:.2f}{' [MOCK]' if a.mock else ''}")
    print(f"[minopt] Summary: {os.path.join(OUT_DIR, f'summary_{a.run_id}_{a.rewriter_style}.md')}")
    if guard is not None:
        print(f"[minopt] Spend: ${guard.spent:.3f} / cap ${guard.max:.0f}")


if __name__ == "__main__":
    main()
