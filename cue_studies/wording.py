"""Wording test (Sec. 4.6): fixed content (same facts and the SAME cues), only the
wording varies. Measures N wordings at confirmation resolution (logistic PE and pull beta,
bootstrap CIs) and checks whether the SPREAD across wordings is large relative to the
noise, i.e. whether wording is a systematic lever.
"""
from __future__ import annotations
import os, sys, json, math
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import gate as gate_mod                                   # noqa: E402
import confirm as confirm_mod                             # noqa: E402
import checkpoint as ckpt_mod                             # noqa: E402
from generator import compose                             # noqa: E402


def _finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _fmt_pe(c):
    if c.get("censored"):
        return f"≥{c['ceiling']*100:.0f}% (censored)"
    pe, ci = c.get("pe"), c.get("pe_ci")
    if not _finite(pe):
        return "—"
    return f"{pe*100:.1f}%" + (f" [{ci[0]*100:.1f}, {ci[1]*100:.1f}]" if ci else "")


def run_wording(cands, pair, model, pool, facts_text, *, model_name, n, seed, out_path,
                guard=None, reveal=False, concurrency=1, run_id="wording"):
    cue_ids = cands[0]["cues"] if cands else []
    import random as _r
    ref_words = len(compose(facts_text, pool, cue_ids, _r.Random(1)).split())
    min_w, max_w = int(ref_words * 0.6), int(ref_words * 1.6)

    try:
        from tqdm import tqdm
        bar = tqdm(total=len(cands) * n * (len(confirm_mod.FINE_GRID) + 4), desc="wording",
                   unit="call", dynamic_ncols=True)
    except Exception:
        bar = None

    ckpt_path = ckpt_mod.path_for(out_path)
    ckpt = ckpt_mod.load(ckpt_path)
    n_loaded = 0
    for i, c in enumerate(cands, 1):
        g = gate_mod.check(c["text"], pool, facts_text, min_words=min_w, max_words=max_w)
        c["gate_ok"] = g.ok
        c["gate_reasons"] = g.reasons
        if g.ok:
            k = ckpt_mod.key(model_name, n, seed, confirm_mod.FINE_GRID, c["text"])
            cached = ckpt.get(c["id"])
            if cached and cached.get("key") == k:       # already measured -> load, 0 API
                res = cached["res"]; n_loaded += 1
            else:
                res = confirm_mod.confirm_candidate(c["text"], pair, model, model_name=model_name,
                                                    n=n, seed=seed, reveal=reveal, progress=bar,
                                                    concurrency=concurrency)
                ckpt[c["id"]] = {"key": k, "res": res}
                ckpt_mod.save(ckpt_path, ckpt)          # persist immediately
            c.update(pe=res["pe"], pe_ci=res["ci"], beta=res["beta"], beta_ci=res["beta_ci"],
                     censored=res["censored"], ceiling=res["ceiling"],
                     non_response_rate=res["non_response_rate"])
            c["_rows"] = res["rows"]
        else:
            c.update(pe=None, pe_ci=None, beta=None, censored=False)
        sp = f"${guard.spent:.3f}/{guard.max:.0f}" if guard is not None else "mock"
        if bar is not None:
            bar.set_postfix_str(f"{i}/{len(cands)} | spent {sp}")
        else:
            print(f"[{i}/{len(cands)}] {_fmt_pe(c)} | spent {sp}", flush=True)
    if bar is not None:
        bar.close()
    if n_loaded:
        print(f"[wording] {n_loaded}/{len(cands)} candidates loaded from checkpoint (0 API).")
    if guard is not None:
        print(f"\n[wording] Actual spend: ${guard.spent:.4f} over {guard.calls} calls "
              f"(cap ${guard.max:.2f}).")

    # --- Analysis: spread vs. noise -----------------------------------------
    ok = [c for c in cands if c.get("gate_ok") and _finite(c.get("pe")) and not c.get("censored")]
    pes = [c["pe"] for c in ok]
    widths = [(c["pe_ci"][1] - c["pe_ci"][0]) for c in ok if c.get("pe_ci")]
    pe_range = (max(pes) - min(pes)) if pes else float("nan")
    med_ci = (sorted(widths)[len(widths) // 2]) if widths else float("nan")
    best = max(ok, key=lambda c: c["pe"]) if ok else None
    worst = min(ok, key=lambda c: c["pe"]) if ok else None
    sep = None
    if best and worst and best.get("pe_ci") and worst.get("pe_ci"):
        sep = bool(best["pe_ci"][0] > worst["pe_ci"][1])
    systematic = (_finite(pe_range) and _finite(med_ci) and pe_range > 2 * med_ci) or bool(sep)

    # --- Report ----------------------------------------------------------------
    L = [f"# Wording test — {run_id}\n",
         f"*Fixed content: facts + cues {cue_ids}. Only the wording varies. "
         f"{len(cands)} wordings, n={n} per point, logistic PE + β, bootstrap CIs.*\n",
         "| # | PE % [95% CI] | pull β | words | gate |", "|--:|---|--:|--:|---|"]
    for c in sorted(cands, key=lambda c: -(c["pe"] if _finite(c.get("pe")) else -1)):
        b = f"{c['beta']:+.2f}" if _finite(c.get("beta")) else "—"
        L.append(f"| {c.get('id')} | {_fmt_pe(c)} | {b} | {len(c['text'].split())} | "
                 f"{'ok' if c.get('gate_ok') else 'REJECT'} |")
    L.append("")
    L.append("## Summary: is wording a systematic lever?\n")
    L.append(f"- valid wordings: **{len(ok)}/{len(cands)}**")
    L.append(f"- PE spread (max−min): **{pe_range*100:.1f} pp**" if _finite(pe_range) else "- PE spread: —")
    L.append(f"- typical CI width (median): {med_ci*100:.1f} pp" if _finite(med_ci) else "- CI width: —")
    if best and worst:
        d = (best["pe"] - worst["pe"]) * 100
        s = "separated" if sep else ("overlap" if sep is False else "?")
        L.append(f"- best vs. worst wording: Δ **{d:.1f} pp**, CIs {s}")
    L.append(f"- **Wording effect: {'systematic (spread well above noise)' if systematic else 'not above noise'}.**")
    if best:
        L.append(f"\n### Best wording ({_fmt_pe(best)})\n> {best['text'].strip()}")
    if worst:
        L.append(f"\n### Worst wording ({_fmt_pe(worst)})\n> {worst['text'].strip()}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(L), encoding="utf-8")
    # persist raw rows
    raw = [{k: v for k, v in c.items() if k != "_rows"} | {"rows": c.get("_rows")} for c in cands]
    with open(str(out_path).replace(".md", "_raw.json"), "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "cue_ids": cue_ids, "candidates": raw}, f, ensure_ascii=False)

    return {"report": str(out_path), "n_valid": len(ok), "pe_range_pp": pe_range * 100 if _finite(pe_range) else None,
            "median_ci_pp": med_ci * 100 if _finite(med_ci) else None,
            "best_worst_separated": sep, "systematic": systematic}
