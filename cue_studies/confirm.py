"""Out-of-sample re-measurement on the fine grid (0-30% in 3-point steps, extended
adaptively in 6-point steps until the curve crosses or the 60% ceiling is reached): logistic
PE and pull beta with bootstrap CIs, plus the model-free crossing for comparison."""
from __future__ import annotations
import os, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import pe_eval                                             # noqa: E402
import estimators                                          # noqa: E402

FINE_GRID = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30]
CEILING = 0.60
STEP = 0.06


def _finite(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def confirm_candidate(text, pair, model, *, model_name, n, seed, reveal, progress=None,
                      concurrency=1):
    r = pe_eval.evaluate_candidate(text, pair, model, model_name=model_name, n=n, seed=seed,
                                   base_grid=FINE_GRID, max_ceiling=CEILING, step=STEP,
                                   reveal_for_mock=reveal, progress=progress, return_rows=True,
                                   concurrency=concurrency)
    est = estimators.estimate_all(r["rows"], n_boot=1500, seed=0)   # logistic PE + pull β, with CIs
    ci = tuple(est["pe_logit_ci"]) if est["pe_logit_ci"] else None
    bci = tuple(est["pull_beta_ci"]) if est["pull_beta_ci"] else None
    return {"pe": est["pe_logit"], "ci": ci,               # headline = logistic PE (tighter CIs)
            "pe_local": est["pe_local"],                    # model-free, for comparison
            "beta": est["pull_beta"], "beta_ci": bci,       # second lens: log-odds pull
            "censored": r["censored"], "ceiling": r["ceiling"],
            "non_response_rate": r["non_response_rate"],
            "rows": r["rows"]}                              # raw rows -> later re-analysis = 0 API
