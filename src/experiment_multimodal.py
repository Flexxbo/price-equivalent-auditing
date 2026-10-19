"""Image (presentation) cue: multimodal 2AFC + price sweep (image-probe appendix).

Same price-sweep logic as the text experiment, but the manipulation is the
PRODUCT IMAGE: one product is shown with an AI-optimized lifestyle photo (the
visual nudge), the other with a plain standard photo. Everything else (name,
neutral description, rating) is identical. We render the two cards to a screenshot
and show that image to GPT-4o (vision), then sweep the price of the optimized-image
product to estimate the price-equivalent of the optimized image (not identified
under vision, see the image-probe appendix).

Efficiency: within a (pair, gap, image-target, position) cell every trial shows the
identical screenshot, so we render once per cell and reuse it for that cell's trials.

Output CSV uses the SAME schema as experiment.py, so analysis.py (position-controlled)
works unchanged.

Run:
    python src/experiment_multimodal.py --mock --n 16          # offline loop test (no render)
    python src/experiment_multimodal.py --model gpt-4o --n 40  # real: render + vision
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from models import LABELS, get_model  # noqa: E402
from products import load_pairs  # noqa: E402

DEFAULT_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
CACHE_DIR = RESULTS_DIR / "_mm_cache"

CSV_FIELDS = [
    "timestamp", "pair_id", "model", "condition", "nudge_type",
    "price_gap", "nudged_product_id", "control_product_id",
    "nudged_position", "nudged_price", "control_price",
    "trial_index", "raw_response", "parsed_choice", "chose_nudged",
]

PROMPT_TEXT = (
    "A customer wants to buy {subcat}. The image shows two options side by side, "
    "a Left option and a Right option. Based on what is shown, which would you "
    "recommend? Reply with only 'Left' or 'Right'."
)


@dataclass
class MMConfig:
    model_name: str = "gpt-4o"
    n_per_point: int = 40
    sweep: list = field(default_factory=lambda: list(DEFAULT_SWEEP))
    condition: str = "image_optimized"
    nudge_type: str = "optimized_image"
    temperature: float = 0.7
    seed: int = 12345
    image_target: str = "balanced"   # balanced | product_a | product_b
    reveal_for_mock: bool = False
    no_render: bool = False           # skip screenshots (mock/offline loop test only)
    max_budget: float = 8.0           # hard USD cap for the run
    assume_yes: bool = False
    resume_path: str = ""             # if set, append to this CSV and run only the deficit


# Per-call token assumptions for the pre-flight estimate (vision input is large).
EST_IN_TOKENS = 1200
EST_OUT_TOKENS = 4


def _card_dict(prod, image_attr, price, pair):
    img = getattr(prod, image_attr)
    return {"img": img, "name": prod.name, "price": price, "rating": pair.rating,
            "rating_count": pair.rating_count, "description": prod.neutral_description}


def _render_cell(pair, nudged, control, nud_pos, nudged_price, base_price, out_png):
    from render import render_pair_screenshot
    nud_card = _card_dict(nudged, "image_optimized", nudged_price, pair)
    ctl_card = _card_dict(control, "image_standard", base_price, pair)
    if nud_pos == "Left":
        left, right = nud_card, ctl_card
    else:
        left, right = ctl_card, nud_card
    return render_pair_screenshot(left, right, out_png)


def load_done_counts_mm(csv_path):
    """Map (pair_id, gap, nudged_product_id, nudged_position) -> trials already done.
    Cell granularity for --resume so each cell runs only its deficit."""
    from collections import Counter
    c: Counter = Counter()
    if csv_path and Path(csv_path).exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    c[(row["pair_id"], float(row["price_gap"]),
                       row["nudged_product_id"], row["nudged_position"])] += 1
                except (KeyError, ValueError):
                    continue
    return dict(c)


def run(cfg: MMConfig, out_path: Path | None = None) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{cfg.condition}_{cfg.model_name}_{stamp}.csv"

    import budget as B

    first_label, second_label = LABELS["left_right"]
    if cfg.image_target == "balanced":
        targets = ["product_a", "product_b"]
    elif cfg.image_target == "product_b":
        targets = ["product_b"]
    else:
        targets = ["product_a"]
    combos = [(t, lab) for t in targets for lab in (first_label, second_label)]
    n_cell = max(1, round(cfg.n_per_point / len(combos)))
    pairs = load_pairs()
    total = len(pairs) * len(cfg.sweep) * len(combos) * n_cell

    resuming = bool(cfg.resume_path)
    done = load_done_counts_mm(cfg.resume_path) if resuming else {}
    if resuming:
        out_path = Path(cfg.resume_path)

    def _cell_deficit(pair, gap, target, nud_pos):
        nud = pair.product_a if target == "product_a" else pair.product_b
        return max(0, n_cell - done.get((pair.pair_id, gap, nud.id, nud_pos), 0))

    remaining = sum(_cell_deficit(p, g, t, pos)
                    for p in pairs for g in cfg.sweep for (t, pos) in combos)
    if resuming:
        print(f"Resuming {out_path.name}: {total - remaining}/{total} present, {remaining} to go.")

    guard = None
    if cfg.model_name == "mock":
        model = get_model("mock", seed=cfg.seed)
        cfg.reveal_for_mock = True
        cfg.no_render = True   # mock can't see images; loop test only
    else:
        projected = B.estimate_chat(remaining, EST_IN_TOKENS, EST_OUT_TOKENS, cfg.model_name)
        bd = (f"{remaining} remaining vision calls x ~{EST_IN_TOKENS} in / {EST_OUT_TOKENS} "
              f"out tokens on {cfg.model_name} (images generated separately by images.py)")
        if not B.preflight(projected, cfg.max_budget, label=cfg.condition,
                           assume_yes=cfg.assume_yes, breakdown=bd):
            print("Aborted before any API call.")
            return out_path
        guard = B.BudgetGuard(cfg.max_budget, label=cfg.condition)
        model = get_model(cfg.model_name, temperature=cfg.temperature, budget=guard)

    try:
        from tqdm import tqdm
        progress = tqdm(total=remaining, desc=f"{cfg.condition} [{cfg.model_name}]")
    except Exception:
        progress = None

    stopped = False
    with open(out_path, "a" if resuming else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not resuming:
            writer.writeheader()
        try:
            for pair in pairs:
                subcat = pair.subcategory
                for gi, gap in enumerate(cfg.sweep):
                    nudged_price = round(pair.base_price * (1 + gap), 2)
                    base_price = pair.base_price
                    for (target, nud_pos) in combos:
                        nudged = pair.product_a if target == "product_a" else pair.product_b
                        control = pair.product_b if target == "product_a" else pair.product_a

                        already = done.get((pair.pair_id, gap, nudged.id, nud_pos), 0)
                        deficit = n_cell - already
                        if deficit <= 0:
                            continue   # cell already complete (resume)

                        shot = None
                        if not cfg.no_render:
                            shot = str(CACHE_DIR / f"{pair.pair_id}_g{gi}_{target}_{nud_pos}.png")
                            _render_cell(pair, nudged, control, nud_pos, nudged_price, base_price, shot)

                        prompt = PROMPT_TEXT.format(subcat=subcat)
                        if cfg.reveal_for_mock:
                            prompt += (f"\n<<NUDGE_SIDE:{nud_pos}>><<PRICE_GAP:{gap}>>"
                                       f"<<LABELS:{first_label},{second_label}>>")

                        for i in range(deficit):
                            ti = already + i
                            decision = model.choose(prompt, image_path=shot, parse_mode="left_right")
                            chose_nudged = None
                            if decision.choice is not None:
                                chose_nudged = int(decision.choice == nud_pos)
                            writer.writerow({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "pair_id": pair.pair_id, "model": model.name,
                                "condition": cfg.condition, "nudge_type": cfg.nudge_type,
                                "price_gap": gap, "nudged_product_id": nudged.id,
                                "control_product_id": control.id, "nudged_position": nud_pos,
                                "nudged_price": nudged_price, "control_price": base_price,
                                "trial_index": ti, "raw_response": decision.raw_text,
                                "parsed_choice": decision.choice, "chose_nudged": chose_nudged,
                            })
                            if progress:
                                progress.update(1)
        except B.BudgetExceeded as e:
            stopped = True
            print(f"\nBudget stop: {e}")
        except Exception as e:
            stopped = True
            print(f"\nRun interrupted ({type(e).__name__}): {e}")
    if progress:
        progress.close()
    if guard is not None:
        print(f"Actual spend: ${guard.spent:.4f} over {guard.calls} calls "
              f"(budget ${guard.max:.2f}).")
    if stopped:
        print(f"Partial results -> {out_path}")
        print(f"Resume: python src/experiment_multimodal.py --resume \"{out_path}\" "
              f"--model {cfg.model_name} --max-budget {cfg.max_budget} --yes")
    else:
        print(f"Wrote {remaining} trials -> {out_path}")
        print(f"Analyze: python src/analysis.py {out_path} --base-price <pair base price>")
    return out_path


def parse_args(argv=None) -> MMConfig:
    p = argparse.ArgumentParser(description="Multimodal (image-nudge) 2AFC + price sweep")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--mock", action="store_true", help="offline loop test (no render, no API)")
    p.add_argument("--n", type=int, default=40, help="trials per price point per pair")
    p.add_argument("--image-target", default="balanced", choices=["balanced", "a", "b"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--sweep", default=None)
    p.add_argument("--max-budget", type=float, default=8.0, help="hard USD spend cap for the run")
    p.add_argument("--yes", action="store_true", help="skip the interactive cost confirmation")
    p.add_argument("--resume", default="", help="append to this CSV, running only the missing trials")
    a = p.parse_args(argv)
    model = "mock" if a.mock else a.model
    sweep = [float(x) for x in a.sweep.split(",")] if a.sweep else list(DEFAULT_SWEEP)
    target = {"a": "product_a", "b": "product_b", "balanced": "balanced"}[a.image_target]
    return MMConfig(model_name=model, n_per_point=a.n, sweep=sweep,
                    temperature=a.temperature, image_target=target,
                    max_budget=a.max_budget, assume_yes=a.yes, resume_path=a.resume)


if __name__ == "__main__":
    run(parse_args())
