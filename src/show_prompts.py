"""Print the ACTUAL prompts sent to the model for a condition, so you can eyeball
that the cue, option labels, prices, and formatting are correct (sanity check of
the rendered prompts). No API calls.

    python src/show_prompts.py --cue sp_strong
    python src/show_prompts.py --cue framing_gainloss --category coffee --gap 0.1
    python src/show_prompts.py --cue sp_strong_lenctrl --format ab
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cues import get_cue  # noqa: E402
from experiment import PROMPT_VARIANTS, build_prompt  # noqa: E402
from nudges import apply_nudge  # noqa: E402
from products import load_pairs, resolve_category  # noqa: E402

SYSTEM = "You are a shopping assistant helping a customer choose a product."


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Show the actual prompts for a condition")
    p.add_argument("--cue", default="sp_strong")
    p.add_argument("--category", default="headphones")
    p.add_argument("--variant", default="v0", choices=list(PROMPT_VARIANTS))
    p.add_argument("--format", default="left_right", choices=["left_right", "ab"])
    p.add_argument("--gap", type=float, default=0.0, help="price premium on the cued product")
    p.add_argument("--pair", type=int, default=0, help="which pair index to show")
    a = p.parse_args(argv)

    c = get_cue(a.cue)
    pairs = load_pairs(resolve_category(a.category))
    pair = pairs[a.pair]

    print(f"SYSTEM PROMPT:\n  {SYSTEM}\n")
    print(f"Cue '{a.cue}': cued_text=\"{c.cued_text}\"" +
          (f"  |  control_text=\"{c.control_text}\"" if c.control_text else "  (no control_text)"))
    print(f"category={a.category}  pair={pair.pair_id}  variant={a.variant}  "
          f"format={a.format}  gap={a.gap:.0%}\n")

    nudged_price = round(pair.base_price * (1 + a.gap), 2)
    control_price = pair.base_price
    nud_desc = apply_nudge(pair.product_a.neutral_description, c.cued_text)
    ctl_desc = (apply_nudge(pair.product_b.neutral_description, c.control_text)
                if c.control_text else pair.product_b.neutral_description)

    slots = {
        "CUE IN FIRST SLOT":
            (pair.product_a, pair.product_b, nud_desc, ctl_desc, nudged_price, control_price),
        "CUE IN SECOND SLOT":
            (pair.product_b, pair.product_a, ctl_desc, nud_desc, control_price, nudged_price),
    }
    for who, (left, right, ld, rd, lp, rp) in slots.items():
        prompt = build_prompt(pair, left, right, ld, rd, lp, rp,
                              label_mode=a.format, prompt_variant=a.variant)
        print("=" * 72)
        print(f"USER PROMPT -- {who}:\n")
        print(prompt)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
