"""Jev PE estimators and self-test harness (mock-runnable, 0 API).

Features:
 * crossing() returns an explicit status: ok / left-censored / aversive(‡) / right-censored(†) /
   plateau at 0.5 / position-driven(not identified). A first-option model does not yield a
   silent PE=0%.
 * FULL counterbalancing: cue-target (A/B) x position (first/second) = 4 cells, equally weighted.
 * two-sided cues (gain/loss, percent/fraction) carry a control text on the OTHER card.
 * keeps raw per-call records (choice, probs, confidence, model, usage, repeat, cell, g), computes
   BOTH the probability-PE and the argmax-PE, and a determinism spread.
 * paper_cross(): the main paper's crossing estimator, used by run_audit.py for the reported numbers.

Estimands are DISTINCT and reported separately:
  - probability-threshold PE : premium where Jev's stated P(treatment) crosses 0.5.
  - behavioral-threshold PE  : premium where the argmax choice rate for the treatment crosses 0.5.
The argmax PE via paper_cross is the number compared with the chat models in the paper; the
probability-threshold PE is a secondary check (in the paper the two agree).

GRID / N_REPEAT / CUES below belong to the self-test harness; the audit itself takes its grid,
repeats and cues from jev/audit_config*.json (see run_audit.py).

Run offline:  python jev/run_jev_pe.py
"""
from __future__ import annotations
import sys, os, math, statistics
sys.path.insert(0, os.path.dirname(__file__))
from jev_client import JevClient

# ---- self-test harness (not the audit configuration) ----
GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.26]
N_REPEAT = 3
EPS_POS = 0.90           # position gap at g=0: reported as a DIAGNOSTIC (not an auto-exclusion)

# A cue is either one-sided ({treat}: only the treatment card gets a sentence) or two-sided
# ({treat, ctrl}: both cards get their respective framing).
CUES = {
    "sp_strong":        {"treat": "Our best-selling model, chosen by over 10,000 customers."},
    "authority":        {"treat": "Recommended by audio experts."},
    "scarcity":         {"treat": "Only a few left in stock."},
    "neutral":          {"treat": "Each unit has a printed serial number on the housing."},
    "framing_gainloss": {"treat": "9 out of 10 buyers keep it.", "ctrl": "1 out of 10 buyers returns it."},
    "framing_pct_frac": {"treat": "Chosen by 25% of shoppers.", "ctrl": "Chosen by 1 in 4 shoppers."},
}


def _card(name, price, pair, extra):
    d = {"name": name, "price": round(price, 2), "rating": pair["rating"],
         "rating_count": pair["rating_count"],
         "description": pair["facts"] + ((" " + extra) if extra else "")}
    return d


def make_cards(pair, cue, treat_is_a, treat_first, g, mock_pe=None):
    """Return (card_first, card_second, treat_slot). Treatment card carries `treat` text and a +g
    premium; for two-sided cues the other card carries `ctrl`. `treat_is_a` and `treat_first`
    give the 4 counterbalance cells."""
    base = float(pair["price"])
    t_name = pair["name_a"] if treat_is_a else pair["name_b"]
    c_name = pair["name_b"] if treat_is_a else pair["name_a"]
    treat = _card(t_name, base * (1.0 + g), pair, cue["treat"])
    ctrl = _card(c_name, base, pair, cue.get("ctrl"))        # ctrl text only for two-sided cues
    if mock_pe is not None:
        treat["_mock_cued"] = True
        treat["_mock_pe"] = mock_pe
    if treat_first:
        return treat, ctrl, "first_option"
    return ctrl, treat, "second_option"


def eval_cell(client, pair, cue, treat_is_a, treat_first, g, mock_pe=None):
    """One counterbalance cell x N_REPEAT. Returns (mean P(treat), [choice==treat ...], raw records)."""
    ps, hits, raw = [], [], []
    for rep in range(N_REPEAT):
        cf, cs, slot = make_cards(pair, cue, treat_is_a, treat_first, g, mock_pe)
        r = client.choose(cf, cs)
        p_treat = r["p_first"] if slot == "first_option" else r["p_second"]
        ps.append(p_treat)
        hits.append(1.0 if r["choice"] == slot else 0.0)
        raw.append({"g": g, "treat_is_a": treat_is_a, "treat_first": treat_first, "rep": rep,
                    "p_treat": p_treat, "choice_treat": hits[-1], "confidence": r["confidence"],
                    "model": r["model"]})
    return statistics.fmean(ps), statistics.fmean(hits), raw


def curves(client, pair, cue, mock_pe=None):
    """Position/target-averaged curves + raw + position gap at g=0 + determinism spread."""
    P, A, raw = [], [], []
    spreads = []
    pos_at0 = {True: [], False: []}     # treat_first -> P(treat) at g=0 (for position-gap)
    for g in GRID:
        cell_p, cell_a = [], []
        for treat_is_a in (True, False):
            for treat_first in (True, False):
                mp, ma, rec = eval_cell(client, pair, cue, treat_is_a, treat_first, g, mock_pe)
                cell_p.append(mp); cell_a.append(ma); raw.extend(rec)
                spreads.append(max(r["p_treat"] for r in rec) - min(r["p_treat"] for r in rec))
                if g == GRID[0]:
                    pos_at0[treat_first].append(mp)
        P.append(statistics.fmean(cell_p))
        A.append(statistics.fmean(cell_a))
    pos_gap = abs(statistics.fmean(pos_at0[True]) - statistics.fmean(pos_at0[False]))
    return P, A, pos_gap, max(spreads), raw


def crossing(gs, ys, pos_gap):
    """Return (pe, status, diag). Computes the crossing whenever one exists; position_gap and
    price_response are DIAGNOSTICS, not auto-exclusions. Any exclusion filter is a separate
    decision applied to (pe, diag) downstream.
    Statuses: ok | left-censored (≤ first grid step, unresolved) | aversive(‡) | right-censored(†) |
              plateau at 0.5 (ambiguous) | position-driven (not identified) [only when the crossing
              itself sits on a 0.5 plateau and the position gap is large] | invalid data (non-finite)."""
    finite = all(y is not None and math.isfinite(y) for y in ys)
    diag = {"position_gap": pos_gap, "price_response": (ys[0] - ys[-1]) if finite else None}
    if not finite:
        return None, "invalid data (non-finite)", diag
    if ys[0] < 0.5 - 1e-9:
        return 0.0, "aversive (‡)", diag
    for i in range(1, len(ys)):
        if ys[i] <= 0.5 + 1e-12:
            # crossing lands on a 0.5 plateau (this point AND the next are 0.5) -> ambiguous interval
            if abs(ys[i] - 0.5) < 1e-6 and i + 1 < len(ys) and abs(ys[i + 1] - 0.5) < 1e-6:
                status = ("position-driven (not identified)" if pos_gap >= EPS_POS
                          else "plateau at 0.5 (ambiguous)")
                return None, status, diag
            y0, y1, g0, g1 = ys[i - 1], ys[i], gs[i - 1], gs[i]
            t = 0.0 if y0 == y1 else (y0 - 0.5) / (y0 - y1)
            pe = g0 + t * (g1 - g0)
            # A crossing already in the FIRST grid interval [g0, g1] is unresolved: the true PE lies
            # anywhere in that interval, so the interpolated point is only an upper marker, not a
            # measurement. Flag it left-censored instead of reporting a spurious point estimate.
            status = ("left-censored (≤%g%%)" % (gs[1] * 100)) if i == 1 else "ok"
            return pe, status, diag
    return gs[-1], "right-censored (†)", diag


def paper_cross(gaps, ps):
    """The MAIN PAPER's estimator, verbatim logic of src/make_paper_numbers._cross (kept here so
    the Jev folder stays self-contained; `_selftest` checks it against the original when src/ is
    importable). Returns (pe_fraction, flag) with flag None | "floor" (base < 0.5 -> 0, ‡) |
    "ceil" (no crossing -> grid max, †).

    Differences to `crossing()` above (Jev-native diagnostics), which make the two NOT
    interchangeable:
      * a crossing needs p[i] >= 0.5 > p[i+1] (strictly below 0.5); `crossing()` already stops at
        p <= 0.5, so an exact 0.5 plateau counts as crossed there but NOT here;
      * no 'left-censored' / 'plateau' statuses: a crossing in the first interval is a point value,
        exactly as for GPT-4o & co. in the paper (e.g. SP-weak 0.7 %).
    Use THIS function for every number that is compared with the other models."""
    if ps[0] < 0.5:
        return 0.0, "floor"
    for i in range(len(gaps) - 1):
        if ps[i] >= 0.5 > ps[i + 1]:
            x0, x1, p0, p1 = gaps[i], gaps[i + 1], ps[i], ps[i + 1]
            return x0 + (p0 - 0.5) / (p0 - p1) * (x1 - x0), None
    return gaps[-1], "ceil"


def pe_for(client, pair, cue, mock_pe=None):
    P, A, pos_gap, det_spread, raw = curves(client, pair, cue, mock_pe)
    pe_p, st_p, diag_p = crossing(GRID, P, pos_gap)
    pe_a, st_a, diag_a = crossing(GRID, A, pos_gap)
    return {"pe_prob": pe_p, "status_prob": st_p, "diag_prob": diag_p,
            "pe_argmax": pe_a, "status_argmax": st_a, "diag_argmax": diag_a,
            "base_prob": P[0], "pos_gap": pos_gap, "determinism_spread": det_spread,
            "P": P, "A": A, "raw": raw}


# ------------------------------------------------------------------ mock smoke + edge cases
def _pair():
    return {"name_a": "Aeris ANC700", "name_b": "Voltic Studio One", "price": 129.99,
            "rating": 4.5, "rating_count": 1200,
            "facts": "Over-ear headphones. Noise cancelling. Wireless. 30-hour battery."}


def _fmt(pe):
    return "  n/a " if pe is None else f"{pe*100:5.1f}%"


def _selftest():
    from jev_client import parse_answer, InvalidJevResponse, QUESTION_ID
    pair = _pair()

    # A) normal mock recovers the injected PE (assertions with tolerance)
    r = pe_for(JevClient(backend="mock", mock_mode="normal"), pair, CUES["sp_strong"], mock_pe=0.13)
    assert r["status_prob"] == "ok" and abs(r["pe_prob"] - 0.13) < 0.02, r
    assert r["determinism_spread"] == 0.0, r                    # mock is deterministic
    r = pe_for(JevClient(backend="mock", mock_mode="normal"), pair,
               {"treat": "A budget option some buyers return."}, mock_pe=-0.05)
    assert r["pe_prob"] == 0.0 and "aversive" in r["status_prob"], r

    # B) edge models: flagged as not-identified, NOT a silent PE=0%
    r = pe_for(JevClient(backend="mock", mock_mode="always_first"), pair, CUES["sp_strong"], mock_pe=0.13)
    assert r["pe_prob"] is None and "position-driven" in r["status_prob"] and r["pos_gap"] > 0.9, r
    r = pe_for(JevClient(backend="mock", mock_mode="flat"), pair, CUES["sp_strong"], mock_pe=0.13)
    assert r["pe_prob"] is None and "plateau" in r["status_prob"], r

    # C) valid crossings must NOT be discarded; diagnostics are separate.
    #    A crossing inside the FIRST grid interval is reported as "left-censored (≤ first step)"
    #    with the interpolated value as upper marker; the first three cases below cross there.
    pe, st, dg = crossing([0.0, 0.1, 0.2], [0.51, 0.50, 0.49], 0.0)               # small but real
    assert st.startswith("left-censored") and abs(pe - 0.10) < 1e-9, (pe, st)
    pe, st, dg = crossing([0.0, 0.02, 0.04], [0.52, 0.499, 0.247], 0.94)          # big position gap, still crosses
    assert st.startswith("left-censored") and pe is not None and dg["position_gap"] == 0.94, (pe, st, dg)
    pe, st, dg = crossing([0.0, 0.1, 0.2, 0.3, 0.4], [0.9, 0.4, 0.5, 0.5, 0.1], 0.0)  # clean early crossing
    assert st.startswith("left-censored") and abs(pe - 0.08) < 1e-9, (pe, st)
    pe, st, dg = crossing([0.0, 0.1, 0.2, 0.3], [0.9, 0.7, 0.4, 0.1], 0.0)        # crossing later -> ok
    assert st == "ok" and abs(pe - (0.1 + 0.2 / 0.3 * 0.1)) < 1e-9, (pe, st)

    # C2) paper estimator: exact-0.5 plateaus do NOT count as crossed; first interval is a point value
    pe, fl = paper_cross([0, .02, .04], [1.0, 0.5, 0.4]); assert fl is None and abs(pe - 0.02) < 1e-12
    assert paper_cross([0, .02, .04], [0.4, 0.3, 0.2]) == (0.0, "floor")
    assert paper_cross([0, .02, .04], [1.0, 0.5, 0.5]) == (0.04, "ceil")
    pe, fl = paper_cross([0, .02, .04], [1.0, 0.25, 0.1]); assert fl is None and abs(pe - 0.0133333333) < 1e-6
    try:                                   # identical to the paper's own implementation?
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from make_paper_numbers import _cross as _paper_ref
        for gs_, ps_ in [([0, .02, .04, .06], [1.0, .6, .5, .3]), ([0, .02], [.5, 0.0]),
                         ([0, .02, .04], [.9, .7, .6]), ([0, .02, .04], [.3, .2, .1])]:
            ref, ours = _paper_ref(gs_, ps_), paper_cross(gs_, ps_)
            assert abs(ref[0] - ours[0] * 100) < 1e-9 and ref[1] == ours[1], (gs_, ps_, ref, ours)
    except ImportError:
        print("(note: src/make_paper_numbers not importable here - paper_cross checked stand-alone)")
    pe, st, dg = crossing([0.0, 0.1, 0.2, 0.3], [1.0, 0.5, 0.5, 0.0], 0.0)        # crossing sits on a plateau
    assert st.startswith("plateau"), (pe, st)
    pe, st, dg = crossing(GRID, [0.8, float("nan"), 0.2] + [0.2] * 5, 0.0)
    assert (pe, st) == (None, "invalid data (non-finite)"), (pe, st)

    # D) two-sided framing puts a control text on the OTHER card
    cf, cs, _ = make_cards(pair, CUES["framing_gainloss"], True, True, 0.0)
    assert "keep it" in cf["description"] and "returns it" in cs["description"], (cf, cs)

    # E) parser rejects malformed responses (values, container types, version)
    def resp(probs, choice="first_option", conf=0.5, typ="choice", model="jev-1.13.0"):
        return {"model": model, "answers": {QUESTION_ID: {"type": typ, "choice": choice,
                "probabilities": probs, "confidence": conf}}}
    ok = {"first_option": 0.7, "second_option": 0.3}
    parse_answer(resp(ok))                                            # valid -> no raise
    for bad in [resp({"first_option": 0.7}),                          # missing option
                resp({"first_option": 1.2, "second_option": -0.2}),  # out of range
                resp({"first_option": 0.9, "second_option": 0.7}),   # sum != 1
                resp({"first_option": float("nan"), "second_option": 0.5}),  # non-finite
                resp(ok, choice="third_option"),                     # unknown choice
                resp(ok, choice="second_option"),                    # choice != argmax
                {"model": "x", "answers": None},                     # answers not a map
                {"model": "x", "answers": {QUESTION_ID: None}}]:      # answer not a map
        try:
            parse_answer(bad); raise AssertionError(f"should have rejected: {bad}")
        except InvalidJevResponse:
            pass

    # version mismatch must be caught THROUGH choose(), not only in a direct parser call
    cf, cs, _ = make_cards(pair, CUES["neutral"], True, True, 0.0, mock_pe=0.0)
    try:
        JevClient(backend="mock", model="jev-1.13.0", expected_model="jev-2.0.0").choose(cf, cs)
        raise AssertionError("version mismatch not caught via choose()")
    except InvalidJevResponse:
        pass
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
