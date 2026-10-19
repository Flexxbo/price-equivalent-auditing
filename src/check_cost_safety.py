"""Cost-safety self-check: exercises the BudgetGuard's hard limits WITHOUT any API
calls, so you can prove the safety nets actually fire before a long unattended run:

    python src/check_cost_safety.py

Exits non-zero if any safety property fails.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import budget as B  # noqa: E402


def usage(pt, ct):
    return {"prompt_tokens": pt, "completion_tokens": ct}


results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def _clear_env():
    os.environ.pop("LLM_AUDIT_HARD_CAP_USD", None)
    os.environ.pop("LLM_AUDIT_HARD_CAP_CALLS", None)


print("Cost-safety self-check (no API calls):")
_clear_env()

# 1) dollar cap stops the run mid-flight
g = B.BudgetGuard(0.05, label="t1")            # 5 cents
stopped = False
try:
    for _ in range(1_000_000):
        g.add_chat(usage(340, 8), "gpt-4o")     # ~$0.00093 each
except B.BudgetExceeded:
    stopped = True
check("dollar cap stops the run", stopped and 0.05 <= g.spent < 0.06)

# 2) global hard cap clamps a mistyped huge --max-budget
os.environ["LLM_AUDIT_HARD_CAP_USD"] = "25"
g2 = B.BudgetGuard(1_000_000.0, label="t2")
check("global hard cap clamps mistyped --max-budget", g2.max == 25.0)
_clear_env()

# 3) a usage-less / unknown model still accrues spend (never unbounded)
g3 = B.BudgetGuard(0.05, label="t3")
stopped3 = False
try:
    for _ in range(1_000_000):
        g3.add_chat(None, "some-unknown-model")  # no usage + unknown price -> fallbacks
except B.BudgetExceeded:
    stopped3 = True
check("usage-less/unknown model still stops (fallback tokens+price)", stopped3 and g3.spent > 0)

# 4) call ceiling fires independently of price (protects against wrong/zero pricing)
os.environ["LLM_AUDIT_HARD_CAP_USD"] = "1000000"
os.environ["LLM_AUDIT_HARD_CAP_CALLS"] = "5"
g4 = B.BudgetGuard(1_000_000.0, label="t4")
stopped4, n4 = False, 0
try:
    for _ in range(100):
        g4.add_chat(usage(0, 0), "gpt-4o")
        n4 += 1
except B.BudgetExceeded:
    stopped4 = True
check("call ceiling stops even with (near) zero price", stopped4 and n4 <= 6)
_clear_env()

# 5) preflight aborts BEFORE any call when projection exceeds budget
ok5 = B.preflight(projected_usd=10.0, budget_usd=4.0, label="t5", assume_yes=True)
check("preflight aborts when projected > budget", ok5 is False)

# 6) preflight enforces the global hard cap (clamp 1000 -> 25, projected 30 -> abort)
os.environ["LLM_AUDIT_HARD_CAP_USD"] = "25"
ok6 = B.preflight(projected_usd=30.0, budget_usd=1000.0, label="t6", assume_yes=True)
check("preflight enforces global hard cap", ok6 is False)
_clear_env()

failed = [n for n, ok in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
print("All cost-safety nets verified.")
