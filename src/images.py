"""Product image generation for the image probe.

Two images per product, of the SAME product:
  - standard:  a plain, UNBRANDED catalog shot on a white background (control).
  - optimized: an AI-staged lifestyle version *of that exact standard image* (the
               visual nudge), produced by EDITING the standard image so the product
               itself is unchanged and only its presentation differs.

Avoiding spurious text/logos (the experiment must not let the agent "read" a label):
  Following the OpenAI image-prompting guide, we (1) never put the product NAME in
  the image prompt — that is what makes the model write text on the product —
  (2) describe the product as plain/unbranded and enumerate explicit exclusions,
  (3) for the edit, change only the scene and keep the product identical, and
  (4) verify each generated image with a vision check and regenerate if any text /
  logo / watermark is detected.

Real backend: OpenAI `gpt-image-1` (generate + edit). Text check: a vision model.

Run:
    python src/images.py --offline                 # placeholders, no API cost
    python src/images.py                           # real, with text-check + retry
    python src/images.py --only over_ear_anc_01 --quality high
    python src/images.py --no-check-text           # skip the vision text check
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from products import load_pairs  # noqa: E402
# Load the project .env so OPENAI_API_KEY is available (this module doesn't import
# models.py, where the .env loader normally runs).
from models import _load_env_file  # noqa: E402

_load_env_file()

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"
IMAGE_MODEL = "gpt-image-1"

# NOTE: deliberately NO product name in these prompts (the name is rendered on the
# card, not baked into the photo). Strong, enumerated exclusions per the guide.
NO_TEXT = ("Absolutely no text, letters, words, numbers, logos, brand names, labels, "
           "captions, stickers or watermarks anywhere in the image; the product has "
           "smooth, blank, unbranded surfaces with no markings.")

STANDARD_PROMPT = (
    "Professional e-commerce catalog photo of a plain, unbranded pair of {subcat}. "
    "A single pair centered on a pure-white seamless background, soft even studio "
    "lighting, sharp focus. " + NO_TEXT
)

# Edit prompt applied to the standard image: change ONLY the scene, keep product identical.
OPTIMIZE_EDIT_PROMPT = (
    "Change ONLY the scene around this exact pair of headphones. Re-stage the SAME "
    "product as a premium lifestyle advertising photograph: a warm wooden desk, soft "
    "natural window light, shallow depth of field, a tasteful out-of-focus plant and "
    "coffee cup in the background. Keep the product itself pixel-identical — same "
    "shape, colour, materials and geometry — and do not add anything to it. " + NO_TEXT
)

TEXT_CHECK_PROMPT = (
    "Does this image contain ANY visible text, letters, words, numbers, logos, brand "
    "names, labels or watermarks, however small? Answer with only 'yes' or 'no'."
)


def _b64_to_file(b64: str, out_path: Path) -> None:
    out_path.write_bytes(base64.b64decode(b64))


def _gen_standard(client, prompt: str, out_path: Path, size: str, quality: str) -> None:
    resp = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size=size,
                                  quality=quality, n=1)
    _b64_to_file(resp.data[0].b64_json, out_path)


def _edit_to_optimized(client, base_path: Path, prompt: str, out_path: Path,
                       size: str, quality: str) -> None:
    with open(base_path, "rb") as f:
        resp = client.images.edit(model=IMAGE_MODEL, image=[f], prompt=prompt,
                                  size=size, quality=quality, n=1)
    _b64_to_file(resp.data[0].b64_json, out_path)


def _has_text(client, image_path: Path, vision_model: str) -> bool:
    """Vision check: True if the image appears to contain text/logo/watermark."""
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    resp = client.chat.completions.create(
        model=vision_model, temperature=0, max_tokens=4,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": TEXT_CHECK_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )
    return (resp.choices[0].message.content or "").strip().lower().startswith("y")


def _make_with_retry(make_fn, out_path: Path, label: str, client, check: bool,
                     vision_model: str, max_retries: int, guard, quality: str) -> None:
    import budget as B
    for attempt in range(max_retries + 1):
        make_fn(out_path)
        if guard is not None:
            guard.add_image(IMAGE_MODEL, quality, 1)        # bill + cap the image
        has = False
        if check:
            has = _has_text(client, out_path, vision_model)
            if guard is not None:
                guard.add_usd(B.chat_cost(1200, 4, vision_model))  # bill + cap the check
        if not has:
            print(f"  {label}: {out_path.name}" + ("" if not check else " (text-check OK)"))
            return
        print(f"  {label}: text detected, regenerating ({attempt + 1}/{max_retries})")
    print(f"  WARNING: {label} still shows text after {max_retries} retries "
          f"-> inspect {out_path.name} manually")


def _placeholder_standard(out_path: Path) -> None:
    from PIL import Image, ImageDraw
    W = H = 512
    img = Image.new("RGB", (W, H), (255, 255, 255))
    _draw_product(ImageDraw.Draw(img), W, H)
    img.save(out_path)


def _placeholder_optimized(out_path: Path) -> None:
    from PIL import Image, ImageDraw
    W = H = 512
    img = Image.new("RGB", (W, H), (224, 206, 184))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(H * 0.72), W, H], fill=(206, 188, 166))
    d.ellipse([W - 150, H - 170, W - 40, H - 60], fill=(176, 156, 132))
    _draw_product(d, W, H)
    img.save(out_path)


def _draw_product(d, W, H, accent=(70, 70, 75)) -> None:
    cx = W // 2
    d.arc([cx - 150, 120, cx + 150, 420], start=200, end=340, fill=accent, width=14)
    for dx in (-150, 150):
        d.rounded_rectangle([cx + dx - 45, 250, cx + dx + 45, 380], radius=28,
                            outline=accent, width=12)


def generate_for_product(product, subcat, offline, client, size, quality,
                         check_text, vision_model, max_retries, guard=None) -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    std = IMAGES_DIR / f"{product.id}_standard.png"
    opt = IMAGES_DIR / f"{product.id}_optimized.png"
    if offline:
        _placeholder_standard(std)
        _placeholder_optimized(opt)
        print(f"  (offline) {std.name} + {opt.name} (same glyph, two backgrounds)")
        return
    std_prompt = STANDARD_PROMPT.format(subcat=subcat)
    _make_with_retry(lambda o: _gen_standard(client, std_prompt, o, size, quality),
                     std, "standard", client, check_text, vision_model, max_retries, guard, quality)
    _make_with_retry(lambda o: _edit_to_optimized(client, std, OPTIMIZE_EDIT_PROMPT, o, size, quality),
                     opt, "optimized", client, check_text, vision_model, max_retries, guard, quality)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate standard + optimized (edited) product images")
    p.add_argument("--offline", action="store_true", help="PIL placeholders, no API cost")
    p.add_argument("--only", default=None, help="restrict to one pair_id")
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--quality", default="medium", choices=["low", "medium", "high", "auto"])
    p.add_argument("--no-check-text", dest="check_text", action="store_false",
                   help="skip the vision text/logo check")
    p.add_argument("--max-retries", type=int, default=1,
                   help="regenerations allowed if text is detected (kept low to bound cost)")
    p.add_argument("--max-budget", type=float, default=3.0, help="hard USD spend cap")
    p.add_argument("--yes", action="store_true", help="skip the interactive cost confirmation")
    a = p.parse_args(argv)

    client = None
    vision_model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    if not a.offline:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            print("OPENAI_API_KEY not set; use --offline for placeholders.")
            return 1
        client = OpenAI(api_key=key)

    pairs = load_pairs()
    if a.only:
        pairs = [pp for pp in pairs if pp.pair_id == a.only]

    guard = None
    if not a.offline:
        import budget as B
        n_slots = len(pairs) * 2 * 2  # 2 products/pair x {standard, optimized}
        projected = B.estimate_images(n_slots, IMAGE_MODEL, a.quality, a.max_retries,
                                      check_model=(vision_model if a.check_text else None))
        bd = (f"{n_slots} images (up to {1 + a.max_retries} attempts each) "
              f"@ {a.quality} on {IMAGE_MODEL}")
        if not B.preflight(projected, a.max_budget, label="images",
                           assume_yes=a.yes, breakdown=bd):
            print("Aborted before any API call.")
            return 1
        guard = B.BudgetGuard(a.max_budget, label="images")

    try:
        for pair in pairs:
            print(f"{pair.pair_id}:")
            for prod in pair.products():
                generate_for_product(prod, pair.subcategory, a.offline, client, a.size,
                                     a.quality, a.check_text, vision_model, a.max_retries, guard)
    except Exception as e:
        from budget import BudgetExceeded
        if isinstance(e, BudgetExceeded):
            print(f"\n{e}")
            if guard:
                print(f"Spent ${guard.spent:.4f}. Images created so far are saved.")
            return 2
        raise
    if guard is not None:
        print(f"Spend: ${guard.spent:.4f} over {guard.calls} billed items "
              f"(budget ${guard.max:.2f}).")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
