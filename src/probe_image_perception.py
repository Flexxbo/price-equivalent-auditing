"""Diagnostic: does the model PERCEIVE the optimized image as more premium?

This isolates the image manipulation from the position/price confounds that
saturate the 2AFC: each product image is shown ALONE and rated 1-10 for
premium/appeal. We then compare the optimized vs standard rating per product.

  - If optimized > standard, the manipulation is perceptible, and the failure to
    identify a PE in the 2AFC reflects position saturation rather than a perception
    failure.
  - If optimized ~= standard, the manipulation is too weak to register.

No side-by-side, so there is no Left/Right bias here at all.

Run:
    python src/probe_image_perception.py --only in_ear_anc_01 --n 10
    python src/probe_image_perception.py --n 8 --max-budget 1 --yes
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from products import load_pairs  # noqa: E402
from models import _load_env_file  # noqa: E402
import budget as B  # noqa: E402

_load_env_file()

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
RATE_PROMPT = (
    "Here is a product photo of a pair of headphones. On a scale of 1 to 10, how "
    "premium, high-quality and appealing does the product look in THIS photo? "
    "Judge only the visual impression. Reply with only a single number from 1 to 10."
)


def _num(text: str):
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    if not m:
        return None
    return max(1.0, min(10.0, float(m.group(0))))


def rate_image(client, path: Path, model: str, guard) -> float | None:
    b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    resp = client.chat.completions.create(
        model=model, temperature=0.7, max_tokens=4,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": RATE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    if guard is not None:
        guard.add_chat(getattr(resp, "usage", None), model)
    return _num(resp.choices[0].message.content or "")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Single-image perception probe (no position confound)")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--n", type=int, default=10, help="ratings per image")
    p.add_argument("--only", default=None, help="restrict to one pair_id")
    p.add_argument("--max-budget", type=float, default=2.0)
    p.add_argument("--yes", action="store_true")
    a = p.parse_args(argv)

    from openai import OpenAI
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set.")
        return 1
    client = OpenAI(api_key=key)

    pairs = load_pairs()
    if a.only:
        pairs = [pp for pp in pairs if pp.pair_id == a.only]
    products = [pr for pp in pairs for pr in pp.products()]

    n_calls = len(products) * 2 * a.n
    projected = B.estimate_chat(n_calls, 1100, 4, a.model)
    if not B.preflight(projected, a.max_budget, label="perception-probe",
                       assume_yes=a.yes, breakdown=f"{n_calls} vision rating calls"):
        return 1
    guard = B.BudgetGuard(a.max_budget, label="perception-probe")

    import statistics
    rows = []
    try:
        for prod in products:
            res = {}
            for kind in ("standard", "optimized"):
                path = IMAGES_DIR / f"{prod.id}_{kind}.png"
                if not path.exists():
                    print(f"  missing {path.name}; run images.py first.")
                    continue
                vals = [v for _ in range(a.n) if (v := rate_image(client, path, a.model, guard)) is not None]
                res[kind] = statistics.mean(vals) if vals else float("nan")
            delta = res.get("optimized", float("nan")) - res.get("standard", float("nan"))
            rows.append((prod.id, res.get("standard"), res.get("optimized"), delta))
            print(f"  {prod.id:20s} standard={res.get('standard'):.2f}  "
                  f"optimized={res.get('optimized'):.2f}  delta={delta:+.2f}")
    except B.BudgetExceeded as e:
        print(f"\n{e}")

    deltas = [d for _, _, _, d in rows if d == d]  # drop NaN
    if deltas:
        n_pos = sum(1 for d in deltas if d > 0)
        print("\n=== SUMMARY ===")
        print(f"mean rating delta (optimized - standard): {statistics.mean(deltas):+.2f}")
        print(f"products where optimized rated higher: {n_pos}/{len(deltas)}")
        print("Interpretation: a clearly positive delta means the model PERCEIVES the "
              "optimized image as more premium -> the unidentified 2AFC PE reflects position "
              "saturation, not a perception failure.")
    if guard is not None:
        print(f"Spend: ${guard.spent:.4f} over {guard.calls} calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
