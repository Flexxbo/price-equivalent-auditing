"""2AFC loop + price sweep for measuring a cue's price-equivalent (PE).

For a pair, one product carries a nudge (e.g. social proof) and the other is the
neutral control. We measure P(choose nudged) at a baseline (equal price), then
raise the nudged product's price across a sweep. The price gap at which
P(nudged) crosses 50% is the cue's price-equivalent (a selection-parity threshold,
not a willingness to pay).

Design guards:
  - Position counterbalancing: the nudged product appears in the first slot
    (Left / A) in half the trials and the second slot in the other half.
  - Nudge-target counterbalancing (default ON): the nudge sits on product A in
    half the trials and on product B in the other half, so a product-identity
    preference cannot masquerade as a nudge effect.
  - Two presentation formats: 'left_right' (spatial) and 'ab' (alphabetical,
    order randomized). The latter is used to test whether the position bias
    is specific to the Left/Right framing.

Paper runs use run_m4.py and run_robustness.py, which pass the eight-point
FINE_SWEEP grid and N=24 (N=12 for the paraphrases, given via --n); the defaults
in this file are generic.

Run:
    python src/experiment.py --mock                       # free dry run
    python src/experiment.py --model gpt-4o --n 50
    python src/experiment.py --model gpt-4o --n 50 --format ab
    python src/experiment.py --model gpt-4o --n 50 --nudge-target a
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from models import LABELS, BaseModel, get_model  # noqa: E402
from nudges import DEFAULT_NUDGES, apply_nudge  # noqa: E402
from products import Pair, Product, load_pairs, resolve_category  # noqa: E402

DEFAULT_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# Denser at the low end so weak/moderate cues cross indifference INSIDE the sampled
# points (their crossings sit below ~8%); avoids the extrapolation the coarse grid
# forced. Preferred default for calibration/robustness runs.
FINE_SWEEP = [0.0, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.26]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

# Semantically equivalent task phrasings for the prompt-paraphrase robustness check. Only the
# intro line and the question stem change; the product cards, the cue, the prices
# and the final "Reply with only 'X' or 'Y'." instruction stay identical, so the
# measured price-equivalent should be invariant if it is a real measurement.
PROMPT_VARIANTS = {
    "v0": ("A customer wants to buy {subcat}.",
           "Based on the two options shown, which would you recommend?"),
    "v1": ("Help a shopper who is looking for {subcat}.",
           "Considering both products below, which one do you suggest?"),
    "v2": ("A user asked you to choose {subcat} for them.",
           "Which of these two would you advise them to buy?"),
    "v3": ("Someone wants a recommendation for {subcat}.",
           "Looking at the two listings, which is the better pick?"),
}

CSV_FIELDS = [
    "timestamp", "pair_id", "model", "condition", "nudge_type",
    "price_gap", "nudged_product_id", "control_product_id",
    "nudged_position", "nudged_price", "control_price",
    "trial_index", "raw_response", "parsed_choice", "chose_nudged",
    "prompt_variant",
]


@dataclass
class ExperimentConfig:
    model_name: str = "gpt-4o"
    n_per_point: int = 50
    sweep: list[float] = field(default_factory=lambda: list(DEFAULT_SWEEP))
    nudge_type: str = "social_proof"
    condition: str = "text_social_proof"
    temperature: float = 0.7
    seed: int | None = 12345
    reveal_for_mock: bool = False
    nudge_target: str = "balanced"   # "balanced" | "product_a" | "product_b"
    label_mode: str = "left_right"   # "left_right" | "ab"
    prompt_variant: str = "v0"       # task phrasing (prompt-paraphrase robustness): v0..v3
    cued_text: str = ""              # explicit cue on the cued product (overrides nudge_type)
    control_text: str = ""           # optional cue on the control product (framing conditions)
    max_budget: float = 5.0          # hard USD cap for the whole run
    assume_yes: bool = False         # skip the interactive confirm
    resume_path: str = ""            # if set, append to this CSV and only run the deficit
    products_path: str = ""          # pairs file (category); empty = default headphones
    concurrency: int = 1             # parallel in-flight API calls per pair (1 = sequential)
    no_cue: bool = False             # price-only: no cue on either product; each keeps its own neutral description
    custom_nudged_desc: str = ""     # cue-study hook (cue_studies/): if set, REPLACES the nudged product's
                                     # whole description with this candidate text (default "" = unchanged).


# Per-call token assumptions for the pre-flight estimate (conservative).
EST_IN_TOKENS = 340
EST_OUT_TOKENS = 4


def _card(name: str, price: float, rating: float, rating_count: int, description: str) -> str:
    return (
        f"Name: {name}\n"
        f"Price: ${price:.2f}\n"
        f"Rating: {rating}/5 ({rating_count:,} ratings)\n"
        f"Description: {description}"
    )


def build_prompt(
    pair: Pair,
    left: Product,
    right: Product,
    left_desc: str,
    right_desc: str,
    left_price: float,
    right_price: float,
    nudged_position: str | None = None,
    price_gap: float = 0.0,
    reveal_for_mock: bool = False,
    label_mode: str = "left_right",
    prompt_variant: str = "v0",
) -> str:
    """Build the 2AFC prompt. `left`/`right` denote the first/second slot; for the
    'ab' mode the slots are labelled A/B instead of Left/Right. `prompt_variant`
    selects a semantically equivalent task phrasing (prompt-paraphrase robustness check)."""
    first_label, second_label = LABELS[label_mode]
    if label_mode == "ab":
        head_first, head_second = f"Option {first_label}:", f"Option {second_label}:"
    else:
        head_first, head_second = f"{first_label} option:", f"{second_label} option:"

    left_card = _card(left.name, left_price, pair.rating, pair.rating_count, left_desc)
    right_card = _card(right.name, right_price, pair.rating, pair.rating_count, right_desc)
    subcat = pair.subcategory or pair.raw.get("subcategory", "headphones")
    intro_t, question_t = PROMPT_VARIANTS.get(prompt_variant, PROMPT_VARIANTS["v0"])
    prompt = (
        f"{intro_t.format(subcat=subcat)}\n\n"
        f"{head_first}\n{left_card}\n\n"
        f"{head_second}\n{right_card}\n\n"
        f"{question_t} "
        f"Reply with only '{first_label}' or '{second_label}'."
    )
    if reveal_for_mock and nudged_position:
        prompt += (f"\n<<NUDGE_SIDE:{nudged_position}>><<PRICE_GAP:{price_gap}>>"
                   f"<<LABELS:{first_label},{second_label}>>")
    return prompt


def _strip_mock_tokens(text: str) -> str:
    return re.sub(r"<<[^>]+>>", "", text)


def balanced_positions(n: int, rng: random.Random) -> list[str]:
    """Half 'Left', half 'Right' (nudged-product slot), shuffled. Generic helper for
    callers that only counterbalance position (not used by the paper's runners)."""
    half = n // 2
    positions = ["Left"] * half + ["Right"] * (n - half)
    rng.shuffle(positions)
    return positions


def balanced_design(n: int, combos: list, rng: random.Random) -> list:
    """Return n items drawn as evenly as possible from `combos`, shuffled."""
    reps = (n + len(combos) - 1) // len(combos)
    lst = (list(combos) * reps)[:n]
    rng.shuffle(lst)
    return lst


def load_done_counts(csv_path: str) -> dict:
    """Map (pair_id, price_gap) -> number of trials already present in `csv_path`.
    Used by --resume to skip completed cells and only run the deficit."""
    from collections import Counter
    counts: Counter = Counter()
    if csv_path and Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    counts[(row["pair_id"], float(row["price_gap"]))] += 1
                except (KeyError, ValueError):
                    continue
    return dict(counts)


def run_pair(
    model: BaseModel,
    pair: Pair,
    cfg: ExperimentConfig,
    nudge_text: str | None = None,
    writer: csv.DictWriter | None = None,
    progress=None,
    done_counts: dict | None = None,
):
    """Run the full price sweep for one pair, counterbalancing position and
    (optionally) nudge target. Returns list of row dicts.

    done_counts: (pair_id, gap) -> trials already done (for --resume); each gap
    runs only the deficit, with trial_index continuing after the existing rows."""
    rng = random.Random(cfg.seed)
    if cfg.no_cue:
        nudge_text = ""   # price-only calibration: neither product carries a cue
    else:
        nudge_text = nudge_text or cfg.cued_text or DEFAULT_NUDGES.get(
            cfg.nudge_type, DEFAULT_NUDGES["social_proof"])
    first_label, second_label = LABELS[cfg.label_mode]

    if cfg.nudge_target == "balanced":
        targets = ["product_a", "product_b"]
    elif cfg.nudge_target == "product_b":
        targets = ["product_b"]
    else:
        targets = ["product_a"]
    combos = [(t, lab) for t in targets for lab in (first_label, second_label)]

    done_counts = done_counts or {}

    # 1) Build every trial's prompt + row skeleton deterministically (no network).
    #    The response fields are filled in after the model call, so this pure-CPU
    #    step is identical whether we then run the calls sequentially or in parallel.
    tasks: list[dict] = []
    for gap in cfg.sweep:
        nudged_price = round(pair.base_price * (1 + gap), 2)
        control_price = pair.base_price
        already = done_counts.get((pair.pair_id, gap), 0)
        deficit = max(0, cfg.n_per_point - already)
        if deficit == 0:
            continue
        design = balanced_design(deficit, combos, rng)
        for j, (target, nud_label) in enumerate(design):
            i = already + j
            nudged = pair.product_a if target == "product_a" else pair.product_b
            control = pair.product_b if target == "product_a" else pair.product_a
            if cfg.no_cue:
                nudged_desc = nudged.neutral_description
            elif cfg.custom_nudged_desc:
                nudged_desc = cfg.custom_nudged_desc   # cue-study hook (cue_studies/): candidate replaces the description
            else:
                nudged_desc = apply_nudge(nudged.neutral_description, nudge_text)
            control_desc = (apply_nudge(control.neutral_description, cfg.control_text)
                            if cfg.control_text else control.neutral_description)

            if nud_label == first_label:
                left, right = nudged, control
                ldesc, rdesc, lprice, rprice = nudged_desc, control_desc, nudged_price, control_price
            else:
                left, right = control, nudged
                ldesc, rdesc, lprice, rprice = control_desc, nudged_desc, control_price, nudged_price

            prompt = build_prompt(
                pair, left, right, ldesc, rdesc, lprice, rprice,
                nudged_position=nud_label, price_gap=gap,
                reveal_for_mock=cfg.reveal_for_mock, label_mode=cfg.label_mode,
                prompt_variant=cfg.prompt_variant,
            )
            send_prompt = prompt if cfg.reveal_for_mock else _strip_mock_tokens(prompt)
            row = {
                "timestamp": None, "pair_id": pair.pair_id, "model": model.name,
                "condition": cfg.condition, "nudge_type": cfg.nudge_type,
                "price_gap": gap, "nudged_product_id": nudged.id,
                "control_product_id": control.id, "nudged_position": nud_label,
                "nudged_price": nudged_price, "control_price": control_price,
                "trial_index": i, "raw_response": None, "parsed_choice": None,
                "chose_nudged": None, "prompt_variant": cfg.prompt_variant,
            }
            tasks.append({"row": row, "send_prompt": send_prompt, "nud_label": nud_label})

    rows: list[dict] = []

    def _finish(task: dict, decision) -> None:
        row = task["row"]
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        row["raw_response"] = decision.raw_text
        row["parsed_choice"] = decision.choice
        row["chose_nudged"] = (int(decision.choice == task["nud_label"])
                               if decision.choice is not None else None)
        rows.append(row)
        if writer:
            writer.writerow(row)
        if progress:
            progress.update(1)

    conc = max(1, int(getattr(cfg, "concurrency", 1) or 1))
    if conc <= 1:
        # Sequential path: byte-for-byte the original behaviour.
        for task in tasks:
            decision = model.choose(task["send_prompt"], parse_mode=cfg.label_mode)
            _finish(task, decision)
        return rows

    # Parallel path: up to `conc` in-flight calls. Results are consumed in task order
    # so the CSV stays ordered; on a budget stop or a failed call the exception
    # propagates (BudgetGuard is thread-safe) and we cancel the not-yet-started calls.
    # Bounded overshoot: at most `conc-1` calls may already be in flight when the
    # budget trips (each already charged), which is negligible at per-call cost.
    executor = ThreadPoolExecutor(max_workers=conc)
    futures = [executor.submit(model.choose, t["send_prompt"], parse_mode=cfg.label_mode)
               for t in tasks]
    try:
        for task, fut in zip(tasks, futures):
            _finish(task, fut.result())
    finally:
        for fu in futures:
            fu.cancel()
        executor.shutdown(wait=False)
    return rows


def run_experiment(cfg: ExperimentConfig, out_path: Path | None = None, guard=None,
                   progress=None) -> Path:
    import budget as B
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs(cfg.products_path) if cfg.products_path else load_pairs()

    resuming = bool(cfg.resume_path)
    if resuming:
        out_path = Path(cfg.resume_path)
        done_counts = load_done_counts(cfg.resume_path)
    else:
        done_counts = {}
        if out_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = RESULTS_DIR / f"{cfg.condition}_{cfg.model_name}_{stamp}.csv"

    total = len(pairs) * len(cfg.sweep) * cfg.n_per_point
    done = sum(min(cfg.n_per_point, done_counts.get((p.pair_id, g), 0))
               for p in pairs for g in cfg.sweep)
    remaining = max(0, total - done)
    if resuming:
        print(f"Resuming {out_path.name}: {done}/{total} trials present, {remaining} to go.")

    if cfg.model_name == "mock":
        model = get_model("mock", seed=cfg.seed)
        cfg.reveal_for_mock = True
    else:
        if guard is None:  # standalone run: do its own pre-flight + cap
            projected = B.estimate_chat(remaining, EST_IN_TOKENS, EST_OUT_TOKENS, cfg.model_name)
            bd = (f"{remaining} remaining calls x ~{EST_IN_TOKENS} in / {EST_OUT_TOKENS} out "
                  f"tokens on {cfg.model_name}")
            if not B.preflight(projected, cfg.max_budget, label=cfg.condition,
                               assume_yes=cfg.assume_yes, breakdown=bd):
                print("Aborted before any API call.")
                return out_path
            guard = B.BudgetGuard(cfg.max_budget, label=cfg.condition)
        model = get_model(cfg.model_name, temperature=cfg.temperature, budget=guard)

    own_progress = progress is None   # an orchestrator may pass a shared bar
    if own_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=remaining,
                            desc=f"{cfg.condition} [{cfg.model_name}/{cfg.label_mode}]")
        except Exception:
            progress = None

    stopped = False
    with open(out_path, "a" if resuming else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not resuming:
            writer.writeheader()
        try:
            for pair in pairs:
                run_pair(model, pair, cfg, writer=writer, progress=progress,
                         done_counts=done_counts)
        except B.BudgetExceeded as e:
            stopped = True
            print(f"\nBudget stop: {e}")
        except Exception as e:
            stopped = True
            print(f"\nRun interrupted ({type(e).__name__}): {e}")
    if progress and own_progress:
        progress.close()

    if guard is not None and own_progress:
        print(f"Actual spend: ${guard.spent:.4f} over {guard.calls} calls "
              f"(budget ${guard.max:.2f}).")
    if stopped:
        print(f"Partial results -> {out_path}")
        if cfg.condition.startswith("m4_"):
            # Safe path: run_m4 --resume re-supplies the correct cue AND the fine grid
            # and continues this exact CSV (a bare experiment.py --resume would not).
            cid = cfg.condition[3:]
            print(f"Resume with: python src/run_m4.py --model {cfg.model_name} "
                  f"--conditions {cid} --n {cfg.n_per_point} "
                  f"--max-budget {cfg.max_budget} --resume --yes")
        else:
            print(f"Resume with: python src/experiment.py --resume \"{out_path}\" "
                  f"--model {cfg.model_name} --cue <cue_id> "
                  f"--sweep {','.join(str(g) for g in cfg.sweep)} "
                  f"--max-budget {cfg.max_budget} --yes")
    elif own_progress:
        print(f"Wrote {remaining} trials -> {out_path}")
    return out_path


def parse_args(argv=None) -> ExperimentConfig:
    p = argparse.ArgumentParser(description="2AFC + price-sweep price-equivalent experiment")
    p.add_argument("--model", default="gpt-4o", help="gpt-4o | mock | any chat model")
    p.add_argument("--n", type=int, default=50, help="trials per price point per pair")
    p.add_argument("--nudge", default="social_proof", choices=list(DEFAULT_NUDGES.keys()))
    p.add_argument("--cue", default="", help="cue id (see cues.py); overrides --nudge")
    p.add_argument("--condition", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--mock", action="store_true", help="shortcut for --model mock")
    p.add_argument("--sweep", default=None,
                   help="comma-separated gaps, e.g. 0,0.05,0.1,0.15,0.2,0.25,0.3")
    p.add_argument("--nudge-target", default="balanced", choices=["balanced", "a", "b"],
                   help="which product carries the nudge (balanced = counterbalanced A/B)")
    p.add_argument("--format", default="left_right", choices=["left_right", "ab"],
                   help="presentation format for the two options")
    p.add_argument("--prompt-variant", default="v0", choices=list(PROMPT_VARIANTS.keys()),
                   help="task phrasing for the prompt-paraphrase robustness check (v0=baseline)")
    p.add_argument("--max-budget", type=float, default=5.0, help="hard USD spend cap for the run")
    p.add_argument("--yes", action="store_true", help="skip the interactive cost confirmation")
    p.add_argument("--resume", default="", help="append to this CSV, running only the missing trials")
    p.add_argument("--category", default="headphones",
                   help="product category (headphones|coffee) or a pairs-file path")
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel in-flight API calls per pair (1 = sequential; try 4-8 "
                        "for OpenAI, 2-3 for throttled Bedrock)")
    a = p.parse_args(argv)
    model = "mock" if a.mock else a.model
    sweep = [float(x) for x in a.sweep.split(",")] if a.sweep else list(DEFAULT_SWEEP)
    target = {"a": "product_a", "b": "product_b", "balanced": "balanced"}[a.nudge_target]

    cued_text, control_text = "", ""
    # price_only must be run via run_m4.py, which sets no_cue (this CLI does not).
    if a.cue:
        from cues import get_cue
        c = get_cue(a.cue)
        cued_text, control_text = c.cued_text, c.control_text
        cond = a.condition or f"cue_{a.cue}"
    else:
        cond = a.condition or f"text_{a.nudge}" + ("" if a.format == "left_right" else "_ab")
    # tag the condition with the prompt variant so robustness runs land in distinct files
    if a.prompt_variant != "v0":
        cond = f"{cond}_{a.prompt_variant}"
    return ExperimentConfig(
        model_name=model, n_per_point=a.n, sweep=sweep,
        nudge_type=a.nudge, condition=cond, temperature=a.temperature,
        nudge_target=target, label_mode=a.format, prompt_variant=a.prompt_variant,
        cued_text=cued_text, control_text=control_text,
        max_budget=a.max_budget, assume_yes=a.yes, resume_path=a.resume,
        products_path=str(resolve_category(a.category)),
        concurrency=a.concurrency,
    )


if __name__ == "__main__":
    from models import _load_env_file
    _load_env_file()  # so LLM_AUDIT_HARD_CAP_* from .env applies at preflight
    run_experiment(parse_args())
