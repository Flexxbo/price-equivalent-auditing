"""Spend safety: pre-flight cost estimate + a hard runtime budget cap.

Concept (three layers):
  1) Pre-flight estimate  - every run computes a worst-case $ cost before any call.
  2) Hard cap             - a BudgetGuard tallies *actual* spend (from token usage /
                            per-image price) and raises BudgetExceeded the moment it
                            crosses --max-budget, aborting the run mid-flight.
  3) Bounded retries      - callers cap retries (image text-check uses 1).

USD list prices as of mid-2026; update as needed.
  Chat:  per 1M tokens (input, output).
  Image: per image, 1024x1024, by quality (gpt-image-1).
"""
from __future__ import annotations

import os
import sys
import threading

PRICING_CHAT = {                 # (input_per_1M, output_per_1M), USD
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    # --- Amazon Bedrock (keyed by our short alias) ---
    "claude-sonnet": (3.00, 15.0),    # Anthropic Claude Sonnet (strong)
    "claude-haiku": (1.00, 5.00),     # Anthropic Claude Haiku (weak)
    "nova-pro": (0.80, 3.20),         # Amazon Nova Pro (strong)
    "nova-micro": (0.035, 0.14),      # Amazon Nova Micro (weak)
    "llama-70b": (0.72, 0.72),        # Meta Llama 3.3 70B (strong)
    "llama-8b": (0.30, 0.30),         # Meta Llama 8B (weak), approximate
}
PRICING_IMAGE = {                # per image, 1024x1024
    "gpt-image-1": {"low": 0.011, "medium": 0.042, "high": 0.167, "auto": 0.167},
}
# Conservative fallbacks (used for unknown models so we never UNDER-estimate).
_CHAT_FALLBACK = (2.50, 10.0)
_IMAGE_FALLBACK = 0.167

# --- GLOBAL hard ceilings (independent of any per-run --max-budget) ---
# Absolute safety net for unattended runs: even a mistyped --max-budget
# cannot exceed these. Override via environment only if you truly need more.
DEFAULT_HARD_CAP_USD = 60.0
DEFAULT_HARD_CAP_CALLS = 200_000
# If a provider returns no token usage, charge this so spend still accrues and the
# guard can never run unbounded on a usage-less model.
_FALLBACK_TOKENS = (340, 8)


def hard_cap_usd() -> float:
    try:
        return float(os.getenv("LLM_AUDIT_HARD_CAP_USD", DEFAULT_HARD_CAP_USD))
    except ValueError:
        return DEFAULT_HARD_CAP_USD


def hard_cap_calls() -> int:
    try:
        return int(os.getenv("LLM_AUDIT_HARD_CAP_CALLS", DEFAULT_HARD_CAP_CALLS))
    except ValueError:
        return DEFAULT_HARD_CAP_CALLS


class BudgetExceeded(RuntimeError):
    pass


def chat_cost(in_tok: int, out_tok: int, model: str) -> float:
    pin, pout = PRICING_CHAT.get(model, _CHAT_FALLBACK)
    return (in_tok / 1e6) * pin + (out_tok / 1e6) * pout


def image_cost(model: str, quality: str, n: int = 1) -> float:
    per = PRICING_IMAGE.get(model, {}).get(quality, _IMAGE_FALLBACK)
    return per * n


def _usage_tokens(usage) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens", 0) or 0), int(usage.get("completion_tokens", 0) or 0)
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(getattr(usage, "completion_tokens", 0) or 0)


class BudgetGuard:
    """Tracks actual spend and aborts when it exceeds max_usd."""

    def __init__(self, max_usd: float, label: str = "run"):
        req = float(max_usd)
        cap = hard_cap_usd()
        self.max = min(req, cap)
        if req > cap:
            print(f"[{label}] NOTE: --max-budget ${req:.2f} clamped to global hard "
                  f"cap ${cap:.2f} (raise LLM_AUDIT_HARD_CAP_USD to change).")
        self.max_calls = hard_cap_calls()
        self.label = label
        self.spent = 0.0
        self.calls = 0
        # Guards spent/calls so parallel workers (--concurrency) can't corrupt the
        # tally or race the budget check. All mutation goes through the lock.
        self._lock = threading.Lock()

    @property
    def remaining(self) -> float:
        return self.max - self.spent

    def _check(self) -> None:
        if self.spent > self.max:
            raise BudgetExceeded(
                f"[{self.label}] budget ${self.max:.2f} reached "
                f"(spent ${self.spent:.4f} over {self.calls} calls). Run stopped."
            )
        if self.calls > self.max_calls:
            raise BudgetExceeded(
                f"[{self.label}] call ceiling {self.max_calls} reached "
                f"({self.calls} calls, ${self.spent:.4f}). Run stopped."
            )

    def add_chat(self, usage, model: str) -> None:
        it, ot = _usage_tokens(usage)
        if it == 0 and ot == 0:            # provider reported no usage -> assume worst
            it, ot = _FALLBACK_TOKENS
        with self._lock:
            self.spent += chat_cost(it, ot, model)
            self.calls += 1
            self._check()

    def add_image(self, model: str, quality: str, n: int = 1) -> None:
        with self._lock:
            self.spent += image_cost(model, quality, n)
            self.calls += n
            self._check()

    def add_usd(self, amount: float) -> None:
        with self._lock:
            self.spent += float(amount)
            self.calls += 1
            self._check()


# ---- pre-flight estimates (worst case) ----

def estimate_chat(n_calls: int, in_tok: int, out_tok: int, model: str) -> float:
    return n_calls * chat_cost(in_tok, out_tok, model)


def estimate_images(n_slots: int, model: str, quality: str, max_retries: int,
                    check_model: str | None = None) -> float:
    """Worst case: each of n_slots images is regenerated max_retries times, and each
    attempt is text-checked once with check_model."""
    attempts = n_slots * (1 + max_retries)
    cost = attempts * image_cost(model, quality, 1)
    if check_model:
        cost += attempts * chat_cost(1200, 4, check_model)  # ~image-input tokens for the check
    return cost


def preflight(projected_usd: float, budget_usd: float, label: str = "run",
              assume_yes: bool = False, breakdown: str = "") -> bool:
    """Print the projected worst-case cost; return True if the run may proceed.

    Refuses (False) if the projection exceeds the budget. Otherwise proceeds, asking
    for interactive confirmation when attached to a terminal (unless assume_yes)."""
    cap = hard_cap_usd()
    if budget_usd > cap:
        print(f"[{label}] NOTE: budget ${budget_usd:.2f} clamped to global hard cap "
              f"${cap:.2f}.")
        budget_usd = cap
    print(f"[{label}] projected worst-case cost: ${projected_usd:.2f}  "
          f"(budget ${budget_usd:.2f})")
    if breakdown:
        print(f"  {breakdown}")
    if projected_usd > budget_usd:
        print("  ABORT: projected cost exceeds --max-budget. "
              "Raise --max-budget or reduce --n / --sweep.")
        return False
    if assume_yes:
        return True
    try:
        if sys.stdin.isatty():
            return input("  Proceed? [y/N] ").strip().lower().startswith("y")
    except Exception:
        pass
    return True  # non-interactive and within budget -> proceed (hard cap still applies)
