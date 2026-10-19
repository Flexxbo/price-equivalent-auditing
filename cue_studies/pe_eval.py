"""PE evaluation of one candidate description.

Sets the candidate text on the nudged product via the experiment.py hook
(custom_nudged_desc) and runs the price sweep.
ADAPTIVE sweep: if the curve does not cross within the start grid (strong candidates),
the grid is extended upward until it crosses or the ceiling is reached
-> no NaN censoring of the best candidates.

Returns: pe (price share), base0 (choice rate at equal price), non_response_rate,
grid_used, n_trials. Non-response = share of unparseable answers; non-responses are
excluded from the PE fit and reported as a rate.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pandas as pd                                  # noqa: E402
from experiment import ExperimentConfig, run_pair    # noqa: E402
from analysis import local_crossing_wtp              # noqa: E402  (no scipy at this point)

BASE_GRID = [0.0, 0.04, 0.08]
MAX_CEILING = 0.60
STEP = 0.06


def _run(text, pair, model, model_name, n, seed, grid, nudge_target, reveal_for_mock,
         progress, concurrency):
    cfg = ExperimentConfig(model_name=model_name, n_per_point=n, sweep=list(grid),
                           seed=seed, nudge_target=nudge_target,
                           custom_nudged_desc=text, assume_yes=True,
                           reveal_for_mock=reveal_for_mock, concurrency=concurrency)
    return run_pair(model, pair, cfg, progress=progress)


def evaluate_candidate(text, pair, model, *, model_name="mock", n=20, seed=0,
                       base_grid=BASE_GRID, max_ceiling=MAX_CEILING, step=STEP,
                       nudge_target="a", reveal_for_mock=False, progress=None,
                       return_rows=False, concurrency=1):
    if not text or not text.strip():
        raise ValueError("evaluate_candidate: empty candidate text (hook would fall back to default).")
    grid = list(base_grid)
    rows = []
    while True:
        rows = _run(text, pair, model, model_name, n, seed, grid, nudge_target,
                    reveal_for_mock, progress, concurrency)
        df = pd.DataFrame(rows)
        nonresp = float(df["chose_nudged"].isna().mean()) if len(df) else 1.0
        d = df.dropna(subset=["chose_nudged"]).copy()
        if len(d):
            d["chose_nudged"] = d["chose_nudged"].astype(int)
        pe = local_crossing_wtp(d) if len(d) else float("nan")
        crossed = not (isinstance(pe, float) and pd.isna(pe))
        if crossed or grid[-1] >= max_ceiling:
            break
        grid = grid + [round(grid[-1] + step, 2), round(grid[-1] + 2 * step, 2)]   # extend upward
    base0 = float(d.loc[d["price_gap"] == 0.0, "chose_nudged"].mean()) if len(d) else float("nan")
    censored = not crossed and grid[-1] >= max_ceiling      # never crossed -> PE >= ceiling
    out = {"pe": pe, "base0": base0, "non_response_rate": nonresp,
           "grid_used": grid, "n_trials": len(rows), "censored": censored, "ceiling": grid[-1]}
    if return_rows:
        out["rows"] = rows
    return out
