r"""Jev PE audit — run harness (collector + analyzer). Real pairs, JSONL persistence, resume,
retry/backoff, enforced budget cap, pair-clustered bootstrap, price-only control.

TWO PROTOCOLS (chosen by the config file):

  * "cards_v1"   (jev/audit_config_v1_cards.json, run jev_audit_v1)
      Structured JSON cards, option keys first_option/second_option with the criteria
      "The product listed first/second", no category sentence, fine grid 0-26 % (+0.5/1/1.5/3),
      3 repeats per cell (12 per pair x price). NOT the prompt the chat models got.
  * "paper_text" (jev/audit_config_v2.json, run jev_audit_v2): METHODOLOGICALLY ALIGNED
      Jev's `state` is the byte-identical user prompt of src/experiment.build_prompt (the one
      GPT-4o & co. got: "A customer wants to buy ... / Left option ... / Right option ... /
      Reply with only 'Left' or 'Right'."), options are named "Left"/"Right", cue texts come
      from src/cues.py, 24 trials per pair x price (4 counterbalance cells x 6). Cue grid =
      the paper grid (0,2,4,6,8,12,18,26 %) plus fine points 0.5/1/1.5/3 % (+40/60 % with
      --extend). The price-only control uses the paper's design (matched pair A vs B, both
      directions) on the paper's control grid (0,2,5,10,18,30,50,80 %) plus fine points 0.5/1/1.5 %.

ANALYSIS (both protocols). results.json holds, per condition:
  * "paper": the numbers to compare with the other models: the paper's estimator
    (run_jev_pe.paper_cross == src/make_paper_numbers._cross), trial-weighted pooling, paper
    grid (+ extension points where collected), numpy default_rng(0) bootstrap with 2000
    resamples, formatted exactly like make_paper_numbers ('0‡', '>=X†', 'x.x [lo, hi]');
    plus the position-specific crossings (cued card first / second) as in the paper's
    per-position table, and for the price-only control p_up = P(choose pricier) at +10 %.
  * "position": argmax curves split by slot, base rates per slot, mean slot gap over g > 0 and
    two flags: equal-price choice fully position-determined; pooled crossing produced by
    per-slot curves on opposite sides of 0.5.
  * the Jev-native diagnostics (pe_argmax / pe_prob with the statuses
    left-censored / plateau / ... from run_jev_pe.crossing, fine grid), NOT for cross-model
    comparison.
  Also written: summary.md (tables for the paper) and per_pair.md.

Usage (from the repository root; see the top-level README.md):
  python jev/run_audit.py --config jev/audit_config_v2.json --backend mock --limit-pairs 1   # offline test (-> jev/_out/jev_audit_v2_mock)
  python jev/run_audit.py --config jev/audit_config_v2.json --backend openrouter             # aligned run (resumes)
  python jev/run_audit.py --config jev/audit_config_v2.json --analyze-only
  python jev/run_audit.py --analyze-only --extend                                            # re-analyse v1
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, sys, time, urllib.error
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient, load_env, InvalidJevResponse
from run_jev_pe import crossing, paper_cross

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # repository root
SRC = ROOT / "src"
MOCK_PE = {"sp_weak": 0.007, "sp_medium": 0.013, "sp_strong": 0.13, "sp_vstrong": 0.21,
           "authority": 0.20, "authority_coffee": 0.20, "scarcity": 0.016, "neutral": 0.02,
           "framing_gainloss": 0.22, "framing_pct_frac": 0.0, "price_only": None}
# The chat models' system prompt (src/models.py, default of every model wrapper). Jev has no system
# prompt, so it goes into the Choice `instructions` together with the variant's question stem.
SYSTEM_PROMPT = "You are a shopping assistant helping a customer choose a product."
RETRY_HTTP = (429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529)   # 52x = transient gateway errors


# ---------------------------------------------------------------- config + stimuli
def load_config(path=HERE / "audit_config_v2.json"):
    raw = Path(path).read_bytes()
    cfg = json.loads(raw)
    cfg["_hash"] = hashlib.sha256(raw).hexdigest()[:16]
    cfg.setdefault("protocol", "cards_v1")
    return cfg


def load_pairs(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))["pairs"]


def cue_text(cfg, cue_id, category, which):
    t = cfg["cues"][cue_id].get(which)
    if t == "__AUTHORITY_PER_CATEGORY__":
        t = cfg["categories"][category]["authority"]
    return t


def build_cards(pair, treat_prod, treat_first, g, treat_txt, ctrl_txt, identical=False):
    """v1 cards. treat card = treatment product + treat_txt + (1+g) premium; ctrl = base price.
    Price-only control (identical=True): no cue on either card and, like the main paper's
    no-cue design in src/experiment.run_pair, the MATCHED PAIR (A vs B, pricier product
    counterbalanced), NOT two copies of product A.
    v1 price-only rows are labelled by what was shown (identical copies vs matched pair)."""
    base = float(pair["base_price"])
    a, b = pair["product_a"], pair["product_b"]
    tp = a if treat_prod == "a" else b
    cp = b if treat_prod == "a" else a

    def card(prod, price, extra):
        return {"name": prod["name"], "price": round(price, 2), "rating": pair["rating"],
                "rating_count": pair["rating_count"],
                "description": prod["neutral_description"] + ((" " + extra) if extra else "")}

    treat = card(tp, base * (1.0 + g), "" if identical else treat_txt)
    ctrl = card(cp, base, "" if identical else ctrl_txt)
    return (treat, ctrl, "first_option") if treat_first else (ctrl, treat, "second_option")


# ---------------------------------------------------------------- collector
import threading
RETRY_STATS = {"429": 0, "5xx": 0, "network": 0}      # shown in the status line (rate limiting visible)
_RETRY_LOCK = threading.Lock()


def _count(kind):
    with _RETRY_LOCK:
        RETRY_STATS[kind] += 1


def _call(fn, tries=6):
    """Call fn() with backoff on rate limits (429, honouring Retry-After), transient 5xx/52x
    gateway errors and timeouts. Every retry is counted in RETRY_STATS."""
    for i in range(tries):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code in RETRY_HTTP and i < tries - 1:
                _count("429" if e.code == 429 else "5xx")
                wait = min(30, 2 ** i)
                try:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = min(60, max(wait, float(ra))) if ra else wait
                except (TypeError, ValueError):
                    pass
                time.sleep(wait); continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i < tries - 1:
                _count("network")
                time.sleep(min(30, 2 ** i)); continue
            raise


def _open_log(out_dir, retry_errors):
    """Read the log: keys already done (+ cost). With retry_errors, keys that only have an error
    row are NOT counted as done, so a resume re-collects them (successful rows win in analysis)."""
    log = out_dir / "trials.jsonl"
    done, n_cost, ok_keys, err_keys = set(), 0.0, set(), set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            (err_keys if r.get("error") else ok_keys).add(r["key"])
            n_cost += float((r.get("usage") or {}).get("cost", 0) or 0)
    done = ok_keys | (set() if retry_errors else err_keys)
    return log, done, n_cost


def cells_v1(cfg, category, pairs, cue_ids, grid):
    for pair in pairs:
        for cue_id in cue_ids:
            identical = cue_id == "price_only"
            tt = "" if identical else cue_text(cfg, cue_id, category, "treat")
            ct = "" if identical else (cue_text(cfg, cue_id, category, "ctrl") or "")
            for g in grid:
                for treat_prod in ("a", "b"):
                    for treat_first in (True, False):
                        for rep in range(cfg["n_repeat"]):
                            yield (pair, cue_id, identical, tt, ct, g, treat_prod, treat_first, rep)


def collect_v1(cfg, backend, out_dir, limit_pairs, only_cues, grid, retry_errors=False):
    load_env()
    if backend == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing (checked .env) — aborting."); return
    exp = cfg["expected_model_prefix"] if backend == "openrouter" else cfg["model"]
    client = JevClient(backend=backend, model=cfg["model"], expected_model=exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    log, done, n_cost = _open_log(out_dir, retry_errors)
    print(f"resume: {len(done)} trials already logged; cost so far ${n_cost:.4f}")
    cue_ids = only_cues or (list(cfg["cues"]) + (["price_only"] if cfg["controls"]["price_only"] else []))
    n_new = 0
    with log.open("a", encoding="utf-8") as fh:
        for category, cc in cfg["categories"].items():
            pairs = load_pairs(cc["pairs_file"])[:limit_pairs] if limit_pairs else load_pairs(cc["pairs_file"])
            for (pair, cue_id, identical, tt, ct, g, tp, tf, rep) in cells_v1(cfg, category, pairs, cue_ids, grid):
                key = f"{category}|{pair['pair_id']}|{cue_id}|{g}|{tp}|{int(tf)}|{rep}"
                if key in done:
                    continue
                if n_cost >= cfg["budget"]["max_cost_usd"] or len(done) >= cfg["budget"]["max_calls"]:
                    print(f"BUDGET CAP reached (cost ${n_cost:.4f}, calls {len(done)}) — stopping; rerun to resume.")
                    return
                cf, cs, slot = build_cards(pair, tp, tf, g, tt, ct, identical)
                if backend == "mock" and MOCK_PE.get(cue_id) is not None:      # give the mock a signal
                    (cf if slot == "first_option" else cs)["_mock_cued"] = True
                    (cf if slot == "first_option" else cs)["_mock_pe"] = MOCK_PE[cue_id]
                try:
                    r = _call(lambda: client.choose(cf, cs))
                except (urllib.error.URLError, TimeoutError, ConnectionError, InvalidJevResponse) as e:
                    fh.write(json.dumps({"key": key, "error": f"{type(e).__name__}: {e}"}) + "\n"); fh.flush()
                    done.add(key); continue
                p_treat = r["p_first"] if slot == "first_option" else r["p_second"]
                rec = {"key": key, "run_id": cfg["run_id"], "config_hash": cfg["_hash"],
                       "protocol": "cards_v1",
                       "ts": datetime.now(timezone.utc).isoformat(), "category": category,
                       "pair_id": pair["pair_id"], "cue_id": cue_id, "g": g, "treat_prod": tp,
                       "treat_first": tf, "rep": rep, "treat_slot": slot, "p_treat": p_treat,
                       "choice_treat": 1.0 if r["choice"] == slot else 0.0, "confidence": r["confidence"],
                       "model": r["model"], "usage": r["usage"],
                       "stimuli": {"first": cf, "second": cs}}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                done.add(key); n_new += 1
                n_cost += float(r["usage"].get("cost", 0) or 0)
                if n_new % 50 == 0:
                    print(f"  {n_new} new trials; cost ${n_cost:.4f}")
    print(f"done: {n_new} new trials; total cost ${n_cost:.4f}")


def _v2_tasks(cfg, only_cues, extend, limit_pairs):
    """All trials of the aligned protocol, in a fixed order (like src/experiment.run_pair + the
    E1 runner src/run_robustness.py): pair x cue x prompt variant x premium x cue target (A/B) x
    slot (Left/Right) x repeat. Prompts come from src/experiment.build_prompt, cue texts from
    src/cues.py, pairs/prices from src/products.py — the chat models' code, not a copy."""
    sys.path.insert(0, str(SRC))
    from experiment import build_prompt, PROMPT_VARIANTS                  # noqa: E402
    from products import load_pairs as load_pair_objs, resolve_category  # noqa: E402
    from nudges import apply_nudge                                        # noqa: E402
    from cues import get_cue                                              # noqa: E402

    variants = cfg.get("prompt_variants", ["v0"])
    assert all(v in PROMPT_VARIANTS for v in variants), variants
    first_label, second_label = "Left", "Right"            # models.LABELS["left_right"]
    for category, cc in cfg["categories"].items():
        pairs = load_pair_objs(resolve_category(cc["category"]))
        pairs = pairs[:limit_pairs] if limit_pairs else pairs
        cue_ids = [c for c in cc["cues"] if not only_cues or c in only_cues]
        for pair in pairs:
            for cue_id in cue_ids:
                c = get_cue(cue_id)
                no_cue = c.type == "none"
                if no_cue:
                    grid = [x / 100 for x in cfg["price_only_grid_percent"]]
                else:
                    grid = [x / 100 for x in cfg["grid_percent"] + (cfg["grid_extend_percent"] if extend else [])]
                for variant in variants:
                    for g in grid:
                        nudged_price = round(pair.base_price * (1 + g), 2)
                        control_price = pair.base_price
                        for tp in ("a", "b"):
                            nudged = pair.product_a if tp == "a" else pair.product_b
                            control = pair.product_b if tp == "a" else pair.product_a
                            nudged_desc = nudged.neutral_description if no_cue else \
                                apply_nudge(nudged.neutral_description, c.cued_text)
                            control_desc = (apply_nudge(control.neutral_description, c.control_text)
                                            if (c.control_text and not no_cue) else control.neutral_description)
                            for pos in (first_label, second_label):
                                if pos == first_label:
                                    args = (nudged, control, nudged_desc, control_desc, nudged_price, control_price)
                                else:
                                    args = (control, nudged, control_desc, nudged_desc, control_price, nudged_price)
                                prompt = build_prompt(pair, *args, label_mode="left_right",
                                                      prompt_variant=variant)
                                instr = cfg["jev_question"]["instructions"]
                                if instr == "auto":          # system prompt + this variant's question stem
                                    instr = f"{SYSTEM_PROMPT} {PROMPT_VARIANTS[variant][1]}"
                                for rep in range(cfg["n_repeat"]):
                                    key = (f"{category}|{pair.pair_id}|{cue_id}|{variant}|{g}|{tp}|"
                                           f"{int(pos == first_label)}|{rep}")
                                    yield {"key": key, "category": category, "pair_id": pair.pair_id,
                                           "cue_id": cue_id, "prompt_variant": variant, "g": g,
                                           "treat_prod": tp, "treat_first": pos == first_label, "rep": rep,
                                           "treat_slot": pos, "nudged_product_id": nudged.id,
                                           "control_product_id": control.id, "nudged_price": nudged_price,
                                           "control_price": control_price, "prompt": prompt,
                                           "instructions": instr,
                                           "_hint": {"cued": None if no_cue else pos, "pe": MOCK_PE.get(cue_id),
                                                     "gap": g, "pricier": pos}}


def _estimate_cost(cfg, todo):
    """Pre-flight estimate from the request sizes, using the input-token price in the config's
    "pricing" block. Tokens are predicted with a linear fit in request characters (config
    tokens_fixed / tokens_per_char) times a safety factor."""
    from jev_client import build_text_request
    pr = cfg.get("pricing", {})
    fixed, per_char = pr.get("tokens_fixed", 385), pr.get("tokens_per_char", 0.172)
    price_in, safety = pr.get("input_usd_per_mtok", 0.042), pr.get("safety_factor", 1.5)
    crit = cfg["jev_question"]["criteria"]
    toks = sum(fixed + per_char * len(json.dumps(build_text_request(t["prompt"], t["instructions"], crit,
                                                                     cfg["model"]))) for t in todo)
    return toks * price_in / 1e6 * safety, toks / max(1, len(todo))


def collect_v2(cfg, backend, out_dir, limit_pairs, only_cues, extend, retry_errors=False, concurrency=1,
               max_budget=None, assume_yes=False):
    """Aligned protocol ("paper_text"). Same three spend-safety layers as the chat-model runs
    (src/budget.py): (1) pre-flight cost estimate with confirmation, aborting if it exceeds the budget;
    (2) runtime cap on the ACTUAL cost logged by the API (usage.cost) and on the number of calls;
    (3) the global hard caps LLM_AUDIT_HARD_CAP_USD / _CALLS from .env. Budget and call cap count the
    whole run folder, i.e. they include earlier (resumed) calls.
    concurrency > 1 sends that many calls in parallel; rows are still written in task order.
    Check the provider's rate limits before raising concurrency. Retries (429 / 5xx / network)
    are shown in the status."""
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(SRC))
    import budget as B                                                     # noqa: E402
    load_env()
    if backend == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY missing (checked .env) — aborting."); return
    exp = cfg["expected_model_prefix"] if backend == "openrouter" else cfg["model"]
    client = JevClient(backend=backend, model=cfg["model"], expected_model=exp)
    q = cfg["jev_question"]
    assert list(q["criteria"]) == ["Left", "Right"], "criteria keys must be Left, Right"
    out_dir.mkdir(parents=True, exist_ok=True)
    log, done, n_cost = _open_log(out_dir, retry_errors)
    todo = [t for t in _v2_tasks(cfg, only_cues, extend, limit_pairs) if t["key"] not in done]
    total = len(done) + len(todo)
    cap_usd = min(float(cfg["budget"]["max_cost_usd"]), float(max_budget) if max_budget else 1e9, B.hard_cap_usd())
    cap_calls = min(int(cfg["budget"]["max_calls"]), B.hard_cap_calls())
    print(f"[{cfg['run_id']}] {len(done)} trials already logged (cost ${n_cost:.4f}); {len(todo)} to go "
          f"(of {total}); concurrency {concurrency}; caps ${cap_usd:.2f} / {cap_calls} calls")
    if not todo:
        return
    est, avg_tok = _estimate_cost(cfg, todo)
    if not B.preflight(n_cost + est, cap_usd, label=cfg["run_id"], assume_yes=assume_yes or backend == "mock",
                       breakdown=f"{len(todo)} remaining calls x ~{avg_tok:.0f} input tokens on Jev "
                                 f"(+ ${n_cost:.4f} already spent); estimate x{cfg.get('pricing', {}).get('safety_factor', 1.5)} safety"):
        print("Aborted before any API call.")
        return

    def work(t):
        try:
            return t, _call(lambda: client.choose_text(t["prompt"], t["instructions"], q["criteria"], t["_hint"])), None
        except (urllib.error.URLError, TimeoutError, ConnectionError, InvalidJevResponse) as e:
            return t, None, f"{type(e).__name__}: {e}"

    try:
        from tqdm import tqdm
        bar = tqdm(total=total, initial=len(done), desc=cfg["run_id"], dynamic_ncols=True)
    except Exception:
        bar = None
    t0, n_new, n_err, chunk = time.time(), 0, 0, max(1, concurrency) * 25

    def status():
        el = max(1e-9, time.time() - t0)
        rate = n_new / el * 60
        left = len(todo) - n_new - n_err
        eta = left / (rate / 60) if rate > 0 else float("nan")
        proj = n_cost + (n_cost / max(1, len(done))) * left if done else float("nan")
        return {"cost": f"${n_cost:.3f}", "proj": f"${proj:.2f}", "per_min": f"{rate:.0f}",
                "eta": (f"{eta / 3600:.1f}h" if eta == eta else "?"), "err": n_err,
                "retry": f"429:{RETRY_STATS['429']} 5xx:{RETRY_STATS['5xx']} net:{RETRY_STATS['network']}"}

    with log.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for i in range(0, len(todo), chunk):
            if n_cost >= cap_usd or len(done) >= cap_calls:
                print(f"\nBUDGET CAP reached (cost ${n_cost:.4f} of ${cap_usd:.2f}, calls {len(done)} of "
                      f"{cap_calls}) — stopping; rerun to resume.")
                break
            for t, r, err in ex.map(work, todo[i:i + chunk]):
                if err:
                    fh.write(json.dumps({"key": t["key"], "error": err}) + "\n")
                    done.add(t["key"]); n_err += 1
                else:
                    pos = t["treat_slot"]
                    rec = {k: v for k, v in t.items() if not k.startswith("_")}
                    rec.update({"run_id": cfg["run_id"], "config_hash": cfg["_hash"], "protocol": "paper_text",
                                "ts": datetime.now(timezone.utc).isoformat(),
                                "p_treat": r["probs"][pos], "choice_treat": 1.0 if r["choice"] == pos else 0.0,
                                "probs": r["probs"], "confidence": r["confidence"], "model": r["model"],
                                "usage": r["usage"]})
                    fh.write(json.dumps(rec) + "\n")
                    done.add(t["key"]); n_new += 1
                    n_cost += float(r["usage"].get("cost", 0) or 0)
                if bar is not None:
                    bar.update(1)
            fh.flush()
            st = status()
            if bar is not None:
                bar.set_postfix(st, refresh=True)
            else:
                print(f"  [{len(done):>6}/{total}] " + "  ".join(f"{k} {v}" for k, v in st.items()), flush=True)
    if bar is not None:
        bar.close()
    print(f"done: {n_new} new trials, {n_err} errors; total cost ${n_cost:.4f}; retries {RETRY_STATS}")
    if n_err:
        print(f"  {n_err} calls failed after retries -> rerun with --retry-errors to fetch them again.")


# ---------------------------------------------------------------- analyzer helpers
def _mean(x):
    return statistics.fmean(x) if x else float("nan")


def _pooled_curve(rows, pair_ids, grid, key):
    """Jev-native pooling: per pair mean over cells x reps at each g, then mean
    over pairs. Returns (curve, pos_gap, per) with pos_gap = |P_first - P_second| of p_treat at g=0."""
    per = {pid: {g: [] for g in grid} for pid in pair_ids}
    pos = {True: [], False: []}
    for r in rows:
        if r["g"] not in per[r["pair_id"]]:
            continue
        per[r["pair_id"]][r["g"]].append(r[key])
        if r["g"] == grid[0]:
            pos[r["treat_first"]].append(r["p_treat"])
    curve = []
    for g in grid:
        pair_means = [statistics.fmean(per[pid][g]) for pid in pair_ids if per[pid][g]]
        curve.append(statistics.fmean(pair_means) if pair_means else float("nan"))
    pos_gap = abs(statistics.fmean(pos[True]) - statistics.fmean(pos[False])) if pos[True] and pos[False] else 0.0
    return curve, pos_gap, per


def _bootstrap(per, pair_ids, grid, pos_gap, resamples, seed):
    """Jev-native bootstrap. Returns lo, hi, frac_not_identified and the share of replicates in
    each status (so censoring is reported)."""
    import random
    rng = random.Random(seed)
    pes, stat = [], {}
    for _ in range(resamples):
        sample = [rng.choice(pair_ids) for _ in pair_ids]
        curve = []
        for g in grid:
            vals = [statistics.fmean(per[pid][g]) for pid in sample if per[pid][g]]
            curve.append(statistics.fmean(vals) if vals else float("nan"))
        pe, st, _ = crossing(grid, curve, pos_gap)
        key = st.split(" (")[0]
        stat[key] = stat.get(key, 0) + 1
        if pe is not None:
            pes.append(pe)
    shares = {k: round(v / resamples, 3) for k, v in sorted(stat.items())}
    if not pes:
        return None, None, 1.0, shares
    pes.sort()
    lo = pes[int(0.025 * len(pes))]
    hi = pes[min(len(pes) - 1, int(0.975 * len(pes)))]
    return lo, hi, 1.0 - len(pes) / resamples, shares


def _paper_block(rows, grid, key, seed=0, nboot=2000):
    """Paper estimator, reproducing src/make_paper_numbers.cell(): trial-weighted pooled curve
    sum(K)/sum(N), paper_cross, numpy default_rng(seed) pair bootstrap. Returns dict with the
    point value (fraction), flag, CI (fractions), share of ceil replicates and the formatted cell."""
    import numpy as np
    pairs = sorted({r["pair_id"] for r in rows})
    idx = {g: i for i, g in enumerate(grid)}
    K = {p: np.zeros(len(grid)) for p in pairs}
    N = {p: np.zeros(len(grid)) for p in pairs}
    for r in rows:
        if r["g"] in idx:
            K[r["pair_id"]][idx[r["g"]]] += r[key]
            N[r["pair_id"]][idx[r["g"]]] += 1
    curve = sum(K.values()) / sum(N.values())
    pt, flag = paper_cross(grid, list(curve))
    out = {"grid_percent": [round(g * 100, 3) for g in grid], "curve": [round(float(x), 3) for x in curve],
           "pe": pt, "flag": flag, "ci": None, "frac_ceil": None}
    if flag == "floor":
        out["fmt"] = "0‡"; return out
    if flag == "ceil":
        out["fmt"] = f">={round(pt * 100)}†"; return out
    if nboot <= 0:
        out["fmt"] = f"{pt * 100:.1f}"; return out
    rng = np.random.default_rng(seed)
    es, nc = [], 0
    for _ in range(nboot):
        t = rng.choice(pairs, len(pairs), replace=True)
        e, fl = paper_cross(grid, list(sum(K[p] for p in t) / sum(N[p] for p in t)))
        nc += fl == "ceil"
        es.append(e)
    lo, hi = float(np.percentile(es, 2.5)), float(np.percentile(es, 97.5))
    out.update(ci=[lo, hi], frac_ceil=round(nc / nboot, 3))
    out["fmt"] = f"{pt * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}{'†' if nc / nboot > 0.1 else ''}]"
    return out


def _point(rows, grid, key):
    """Paper estimator point value only (for position-specific crossings, as in the per-position table)."""
    b = _paper_block(rows, grid, key, nboot=0) if rows else None
    if not b:
        return None
    return {"pe": b["pe"], "flag": b["flag"],
            "fmt": "0‡" if b["flag"] == "floor" else (f">={round(b['pe'] * 100)}†" if b["flag"] == "ceil"
                                                      else f"{b['pe'] * 100:.1f}")}


def _slot_curve(rows, grid, first):
    return [_mean([r["choice_treat"] for r in rows if r["g"] == g and r["treat_first"] == first]) for g in grid]


def _condition_label(r):
    """price_only rows are split by what was actually shown: two copies of one product
    ('price_only_identical', v1 treat_prod=a cells) vs the matched pair ('price_only' = paper design).
    Task paraphrases other than v0 get their own condition '<cue>@<variant>' (as the E1 check)."""
    v = r.get("prompt_variant", "v0")
    if r["cue_id"] != "price_only":
        return r["cue_id"] + ("" if v == "v0" else f"@{v}")
    st = r.get("stimuli")
    if st and st["first"]["name"] == st["second"]["name"] and st["first"]["description"] == st["second"]["description"]:
        return "price_only_identical"
    return "price_only"


def _paper_grid(cfg, present, cue_label):
    """The premia the CHAT MODELS were measured on (for like-for-like comparison): the paper's cue
    grid 0/2/4/6/8/12/18/26 % (+ extension points where collected), and for the price-only control
    the paper's control grid 0/2/5/10/18/30/50/80 %. Jev's extra fine points (0.5/1/1.5/3 %) are
    used in the 'fine' block instead."""
    pg = set(round(x / 100, 6) for x in cfg.get("paper_grid_percent", [0, 2, 4, 6, 8, 12, 18, 26]))
    ext = set(round(x / 100, 6) for x in cfg.get("grid_extend_percent", []))
    if cue_label.startswith("price_only"):
        pg = set(round(x / 100, 6) for x in cfg.get("paper_price_only_grid_percent", [])) or pg
    return [g for g in present if round(g, 6) in pg or round(g, 6) in ext]


# ---------------------------------------------------------------- analyzer
def analyze(cfg, out_dir, grid):
    log = out_dir / "trials.jsonl"
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    determinism_report(out_dir)
    # a key can have an error row followed by a successful retry -> keep successful rows only
    good = [r for r in rows if not r.get("error") and r.get("category") is not None]
    ok_keys = {r["key"] for r in good}
    n_err = sum(1 for r in rows if (r.get("error") or r.get("category") is None) and r.get("key") not in ok_keys)
    if n_err:
        print(f"note: {n_err} keys only have error/blank rows and are skipped in analysis.")
    by = {}
    for r in good:
        by.setdefault((r["category"], _condition_label(r)), []).append(r)
    R, S = cfg["bootstrap"]["resamples"], cfg["bootstrap"]["seed"]
    results = {}
    for (category, cue), rs in sorted(by.items()):
        pair_ids = sorted({r["pair_id"] for r in rs})
        present = sorted({r["g"] for r in rs})
        g_cond = [g for g in grid if g in present] if cfg["protocol"] == "cards_v1" and not cue.startswith("price_only") else present
        # --- Jev-native diagnostics (fine grid; NOT for cross-model comparison)
        P, pos_gap, perP = _pooled_curve(rs, pair_ids, g_cond, "p_treat")
        A, _, perA = _pooled_curve(rs, pair_ids, g_cond, "choice_treat")
        pe_p, st_p, _ = crossing(g_cond, P, pos_gap)
        pe_a, st_a, _ = crossing(g_cond, A, pos_gap)
        lo_a, hi_a, ni_a, sh_a = _bootstrap(perA, pair_ids, g_cond, pos_gap, R, S)
        lo_p, hi_p, ni_p, sh_p = _bootstrap(perP, pair_ids, g_cond, pos_gap, R, S)
        # --- paper-comparable block
        pgrid = _paper_grid(cfg, present, cue)
        paper = {"argmax": _paper_block(rs, pgrid, "choice_treat", seed=0),
                 "prob": _paper_block(rs, pgrid, "p_treat", seed=0),
                 # same estimator on ALL collected premia (incl. Jev's fine 0.5/1/1.5/3 % steps)
                 "fine": {"argmax": _paper_block(rs, present, "choice_treat", seed=0),
                          "prob": _paper_block(rs, present, "p_treat", seed=0),
                          "first": _point([r for r in rs if r["treat_first"]], present, "choice_treat"),
                          "second": _point([r for r in rs if not r["treat_first"]], present, "choice_treat")},
                 "first": {"argmax": _point([r for r in rs if r["treat_first"]], pgrid, "choice_treat"),
                           "prob": _point([r for r in rs if r["treat_first"]], pgrid, "p_treat")},
                 "second": {"argmax": _point([r for r in rs if not r["treat_first"]], pgrid, "choice_treat"),
                            "prob": _point([r for r in rs if not r["treat_first"]], pgrid, "p_treat")},
                 "base_rate": round(_mean([r["choice_treat"] for r in rs if r["g"] == 0]), 3),
                 "n_per_pair_price": min(sum(1 for r in rs if r["pair_id"] == p and r["g"] == g)
                                         for p in pair_ids for g in pgrid)}
        if cue.startswith("price_only"):
            at10 = [r["choice_treat"] for r in rs if abs(r["g"] - 0.10) < 1e-9]
            paper["p_up_10"] = round(_mean(at10), 3) if at10 else None
            near = sorted(present, key=lambda g: abs(g - 0.10))[:2]
            paper["p_up_near_10"] = {f"{g * 100:g}": round(_mean([r["choice_treat"] for r in rs if r["g"] == g]), 3) for g in near}
        # --- position diagnostics (argmax)
        Af, As = _slot_curve(rs, g_cond, True), _slot_curve(rs, g_cond, False)
        gaps_gt0 = [abs(a - b) for g, a, b in zip(g_cond, Af, As) if g > 0]
        mixed = False
        pa = paper["argmax"]
        if pa["flag"] is None:
            i = next(k for k, g in enumerate(pgrid) if g >= pa["pe"] - 1e-12)
            gi = pgrid[i]
            fi = _mean([r["choice_treat"] for r in rs if r["g"] == gi and r["treat_first"]])
            si = _mean([r["choice_treat"] for r in rs if r["g"] == gi and not r["treat_first"]])
            mixed = fi >= 0.5 > si
        position = {"grid_percent": [round(g * 100, 3) for g in g_cond],
                    "curve_first": [round(x, 3) for x in Af], "curve_second": [round(x, 3) for x in As],
                    "base_first": round(Af[0], 3), "base_second": round(As[0], 3),
                    "mean_slot_gap_g_gt_0": round(_mean(gaps_gt0), 3) if gaps_gt0 else None,
                    "equal_price_position_determined": Af[0] >= 0.99 and As[0] <= 0.01,
                    "pooled_crossing_from_opposite_slots": mixed}
        results[f"{category}/{cue}"] = {
            "n_pairs": len(pair_ids),
            "paper": paper,
            "position": position,
            # ---- Jev-native diagnostics
            "base_choice_rate": round(A[0], 3), "base_p_treat": round(P[0], 3),
            "position_gap": round(pos_gap, 3),
            "pe_argmax": pe_a, "status_argmax": st_a, "pe_argmax_ci": [lo_a, hi_a],
            "frac_not_identified_argmax": round(ni_a, 3), "bootstrap_status_shares_argmax": sh_a,
            "pe_prob": pe_p, "status_prob": st_p, "pe_prob_ci": [lo_p, hi_p],
            "frac_not_identified_prob": round(ni_p, 3), "bootstrap_status_shares_prob": sh_p,
            "curve_A": [round(x, 3) for x in A], "curve_P": [round(x, 3) for x in P]}
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(cfg, out_dir, results)
    write_per_pair(cfg, out_dir, by, grid)
    print(f"\nRESULTS -> {out_dir/'results.json'}, summary.md, per_pair.md")
    print(f"{'condition':34s} {'base':>5s}  {'PE paper (argmax)':24s} {'first':>7s} {'second':>7s}  flags")
    for k, v in results.items():
        p, pos = v["paper"], v["position"]
        fl = ("eq-price=position " if pos["equal_price_position_determined"] else "") + \
             ("pooled-from-opposite-slots" if pos["pooled_crossing_from_opposite_slots"] else "")
        print(f"  {k:32s} {p['base_rate']:5.2f}  {p['argmax']['fmt']:24s} {p['first']['argmax']['fmt']:>7s} "
              f"{p['second']['argmax']['fmt']:>7s}  {fl}")


LABEL = {"vacuous": "Vacuous claim", "framing_rating": "Rating-rescale framing",
         "sp_strong_lenctrl": "Social proof, length-matched", "presence_ctrl": "Presence control",
         "sp_weak": "Social proof — weak", "sp_medium": "Social proof — moderate",
         "sp_strong": "Social proof — strong", "sp_vstrong": "Social proof — very strong",
         "authority": "Authority", "authority_coffee": "Authority (coffee experts)",
         "scarcity": "Scarcity", "neutral": "Neutral sentence", "framing_gainloss": "Gain/loss framing",
         "framing_pct_frac": "Percent vs. fraction framing", "price_only": "Price-only (matched pair A vs B)",
         "price_only_identical": "Price-only (two copies of A)"}


def write_summary(cfg, out_dir, results):
    cats = sorted({k.split("/")[0] for k in results}, key=lambda c: (c != "headphones", c))
    L = [f"# Jev results — {cfg['run_id']} (protocol {cfg['protocol']})", "",
         "Generated by `jev/run_audit.py --analyze-only`. **Paper-comparable** numbers: estimator of "
         "`src/make_paper_numbers.py` (first crossing p[i] >= 0.5 > p[i+1], linear interpolation; ‡ base < 0.5; "
         "† no crossing), trial-weighted pooling, paper grid 0/2/4/6/8/12/18/26 % (+40/60 % where collected), "
         "pair bootstrap 2000x, seed 0. PE in % of base price, argmax choice (primary) and Jev's stated "
         "probability (secondary). 'first'/'second' = cued card in the first/second slot (as in the per-position table). "
         "'fine grid' = same estimator on every collected premium incl. Jev's extra 0.5/1/1.5/3 % steps "
         "(first/second in brackets); use the paper-grid column for tables next to the chat models.", ""]
    for cat in cats:
        L += [f"## {cat}", "", "| Condition | base rate | slot 1 / slot 2 at g=0 | PE argmax [95% CI] (paper grid) | PE stated prob. | first | second | PE argmax, fine grid | flags |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, v in results.items():
            if not k.startswith(cat + "/"):
                continue
            cue = k.split("/")[1]
            p, pos = v["paper"], v["position"]
            flags = []
            if pos["equal_price_position_determined"]:
                flags.append("equal-price choice = slot (100/0)")
            if pos["pooled_crossing_from_opposite_slots"]:
                flags.append("pooled crossing from opposite slots")
            if (p["argmax"].get("frac_ceil") or 0) >= 0.005:
                flags.append(f"{p['argmax']['frac_ceil']:.0%} of resamples uncrossed")
            if cue.startswith("price_only") and p.get("p_up_near_10"):
                flags.append("p_up " + ", ".join(f"+{g}%: {x:.2f}" for g, x in p["p_up_near_10"].items()))
            base_cue, _, var = cue.partition("@")
            lab = LABEL.get(base_cue, base_cue)
            if cfg["protocol"] == "paper_text" and cat.startswith("coffee") and base_cue == "authority":
                lab = "Authority (audio experts — off-domain for coffee)"
            lab += f" [{var}]" if var else ""
            L.append(f"| {lab} | {p['base_rate']:.2f} | {pos['base_first']:.2f} / {pos['base_second']:.2f} | "
                     f"{p['argmax']['fmt']} | {p['prob']['fmt']} | {p['first']['argmax']['fmt']} | "
                     f"{p['second']['argmax']['fmt']} | {p['fine']['argmax']['fmt']} "
                     f"({p['fine']['first']['fmt']} / {p['fine']['second']['fmt']}) | {'; '.join(flags)} |")
        L.append("")
    if cfg["protocol"] == "cards_v1":
        L += ["Note: n per pair x price = 12 in v1 (4 cells x 3 repeats) vs 24 for the chat models; "
              "the v1 protocol used structured cards, not the chat prompt (see the README).", ""]
    else:
        L += [f"Note: aligned protocol — the chat models' prompt (src/experiment.build_prompt), cue catalog and "
              f"pairs; {4 * cfg['n_repeat']} trials per pair x price and prompt variant(s) "
              f"{', '.join(cfg.get('prompt_variants', ['v0']))}.", ""]
    (out_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")


def write_per_pair(cfg, out_dir, by, grid):
    L = ["# Per-pair view (generated by jev/run_audit.py)", "",
         "`Jev status` = Jev-native diagnostic run_jev_pe.crossing on the pair's argmax curve (fine grid); "
         "`PE paper` = paper estimator on the paper grid (+ extension). `first/second` = mean argmax "
         "choice of the cued card when shown in slot 1 / slot 2, over g >= 2 % and at the largest premium.", ""]
    for (category, cue), rs in sorted(by.items()):
        if cue not in ("sp_strong", "authority", "authority_coffee"):
            continue
        L += [f"## {category} — {LABEL.get(cue, cue)}", "",
              "| Pair | Jev status | PE paper | first @g≥2% | second @g≥2% | first @max | second @max | argmax curve |",
              "|---|---|---|---|---|---|---|---|"]
        for pid in sorted({r["pair_id"] for r in rs}):
            pr = [r for r in rs if r["pair_id"] == pid]
            present = sorted({r["g"] for r in pr})
            A = [_mean([r["choice_treat"] for r in pr if r["g"] == g]) for g in present]
            pf = _mean([r["p_treat"] for r in pr if r["g"] == 0 and r["treat_first"]])
            ps = _mean([r["p_treat"] for r in pr if r["g"] == 0 and not r["treat_first"]])
            pe, st, _ = crossing(present, A, abs(pf - ps))
            pt = _point(pr, _paper_grid(cfg, present, cue), "choice_treat")
            gm = present[-1]
            f2 = _mean([r["choice_treat"] for r in pr if r["g"] >= 0.02 and r["treat_first"]])
            s2 = _mean([r["choice_treat"] for r in pr if r["g"] >= 0.02 and not r["treat_first"]])
            fm = _mean([r["choice_treat"] for r in pr if r["g"] == gm and r["treat_first"]])
            sm = _mean([r["choice_treat"] for r in pr if r["g"] == gm and not r["treat_first"]])
            st_s = st + ("" if pe is None else f" ({pe * 100:.1f}%)")
            L.append(f"| {pid} | {st_s} | {pt['fmt']} | {f2:.2f} | {s2:.2f} | {fm:.2f} (+{gm * 100:g}%) | "
                     f"{sm:.2f} | {' '.join(f'{a:.2f}' for a in A)} |")
        L.append("")
    (out_dir / "per_pair.md").write_text("\n".join(L), encoding="utf-8")


def _determinism(rows):
    """Max spread of p_treat across the identical-input repeats of each cell."""
    groups = {}
    for r in rows:
        if r.get("error") or "p_treat" not in r:
            continue
        groups.setdefault(r["key"].rsplit("|", 1)[0], []).append(r["p_treat"])
    spreads = [max(v) - min(v) for v in groups.values() if len(v) > 1]
    if not spreads:
        return None
    return {"cells": len(spreads), "max_spread": max(spreads),
            "mean_spread": statistics.fmean(spreads),
            "frac_deterministic": sum(1 for s in spreads if s <= 1e-3) / len(spreads)}


def determinism_report(out_dir):
    log = out_dir / "trials.jsonl"
    if not log.exists():
        print("no trials.jsonl yet."); return
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
    d = _determinism(rows)
    if not d:
        print("determinism: no cells with >1 repeat (n_repeat=1?)."); return
    rec = "report the spread across repeats"
    print(f"determinism: {d['cells']} repeated cells; max rep spread={d['max_spread']:.4f}, "
          f"mean={d['mean_spread']:.4f}, deterministic(<=0.001)={d['frac_deterministic']*100:.0f}%  -> {rec}")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "audit_config_v2.json"),
                    help="audit_config_v2.json (main run, paper_text; default), audit_config_v2_e1.json (paraphrase robustness) or audit_config_v1_cards.json (earlier structured-cards run)")
    ap.add_argument("--backend", default="openrouter", choices=["openrouter", "typesafe", "mock"])
    ap.add_argument("--limit-pairs", type=int, default=0, help="use only the first N pairs per category")
    ap.add_argument("--cues", default="", help="comma-separated subset (e.g. authority,neutral); '' = all")
    ap.add_argument("--extend", action="store_true", help="append the extended grid (40,60%%); intended for "
                    "conditions that are right-censored at 26%%")
    ap.add_argument("--retry-errors", action="store_true", help="re-collect keys that only have an error row")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel calls (paper_text protocol only); test e.g. 4 on a small subset first")
    ap.add_argument("--max-budget", type=float, default=None, help="USD cap for this run folder (<= config cap)")
    ap.add_argument("--yes", action="store_true", help="skip the interactive cost confirmation")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--determinism", action="store_true", help="report repeat spread from the log and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    grid = [g / 100 for g in cfg["grid_percent"] + (cfg.get("grid_extend_percent", []) if args.extend else [])]
    # mock runs never write into a real run's log (a later real run would otherwise "resume" from
    # mock rows and skip those keys): they get their own folder <run_id>_mock.
    out_dir = HERE / "_out" / (cfg["run_id"] + ("_mock" if args.backend == "mock" else ""))
    only = [c.strip() for c in args.cues.split(",") if c.strip()] or None

    if args.determinism:
        determinism_report(out_dir); return
    if not args.analyze_only:
        if cfg["protocol"] == "paper_text":
            collect_v2(cfg, args.backend, out_dir, args.limit_pairs or None, only, args.extend,
                       args.retry_errors, args.concurrency, args.max_budget, args.yes)
        else:
            collect_v1(cfg, args.backend, out_dir, args.limit_pairs or None, only, grid, args.retry_errors)
    if not (out_dir / "trials.jsonl").exists() or (out_dir / "trials.jsonl").stat().st_size == 0:
        print(f"no data in {out_dir} yet — nothing to analyse."); return
    if cfg["protocol"] == "paper_text":   # analysis always sees every grid point that was collected
        grid = sorted({g / 100 for g in cfg["grid_percent"] + cfg.get("grid_extend_percent", [])
                       + cfg.get("price_only_grid_percent", [])})
    analyze(cfg, out_dir, grid)


if __name__ == "__main__":
    main()
