"""Render two product cards side by side to a screenshot (image probe).

The multimodal experiment shows the agent a screenshot (not text). This module
builds an HTML page with two product cards — each with an image, name, price,
star rating and the neutral description — and rasterises it to PNG with a headless
browser (Playwright). Images are base64-embedded so the browser needs no file
access.

Setup (one-time):
    pip install playwright pillow
    python -m playwright install chromium

Demo:
    python src/render.py --demo      # renders one example card to data/results/
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"

CARD_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: Arial, Helvetica, sans-serif; margin:0; padding:28px; background:#fff; }}
  .row {{ display:flex; gap:36px; justify-content:center; align-items:stretch; }}
  .card {{ width:340px; border:1px solid #e3e3e3; border-radius:12px; padding:18px;
           box-shadow:0 1px 4px rgba(0,0,0,.06); display:flex; flex-direction:column; }}
  .card img {{ width:100%; height:240px; object-fit:cover; border-radius:8px; background:#f6f6f6; }}
  .name {{ font-size:19px; font-weight:700; margin:12px 0 2px; color:#111; }}
  .price {{ font-size:22px; font-weight:700; color:#111; margin:2px 0; }}
  .rating {{ color:#f5a623; font-size:14px; margin:2px 0; }}
  .rating span {{ color:#666; }}
  .desc {{ font-size:13px; color:#333; line-height:1.45; margin-top:8px; }}
</style></head><body><div class="row">{cards}</div></body></html>"""

CARD = """<div class="card">
  <img src="{img}" alt="{name}">
  <div class="name">{name}</div>
  <div class="price">${price:.2f}</div>
  <div class="rating">{stars} <span>{rating}/5 ({rating_count:,} ratings)</span></div>
  <div class="desc">{description}</div>
</div>"""


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _data_uri(image_path: str | Path) -> str:
    p = Path(image_path)
    if not p.is_absolute():
        # JSON stores paths like "images/<file>.png", relative to the data/ dir.
        p = DATA_DIR / p
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _stars(rating: float) -> str:
    full = int(round(rating))
    return "★" * full + "☆" * (5 - full)


def build_html(left: dict, right: dict) -> str:
    """left/right dicts: img (path), name, price, rating, rating_count, description."""
    def card(d):
        fields = {k: v for k, v in d.items() if k != "img"}
        return CARD.format(img=_data_uri(d["img"]), stars=_stars(d["rating"]), **fields)
    return CARD_TEMPLATE.format(cards=card(left) + card(right))


def render_html_to_png(html: str, out_png: str | Path,
                       width: int = 800, height: int = 560) -> Path:
    """Rasterise HTML to PNG via Playwright (headless Chromium)."""
    from playwright.sync_api import sync_playwright

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height},
                                device_scale_factor=2)
        page.set_content(html, wait_until="networkidle")
        page.screenshot(path=str(out_png))
        browser.close()
    return out_png


def render_pair_screenshot(left: dict, right: dict, out_png: str | Path) -> Path:
    return render_html_to_png(build_html(left, right), out_png)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    if a.demo:
        from products import load_pairs
        pair = load_pairs()[0]
        a_, b_ = pair.products()
        left = {"img": a_.image_optimized, "name": a_.name, "price": pair.base_price,
                "rating": pair.rating, "rating_count": pair.rating_count,
                "description": a_.neutral_description}
        right = {"img": b_.image_standard, "name": b_.name, "price": pair.base_price,
                 "rating": pair.rating, "rating_count": pair.rating_count,
                 "description": b_.neutral_description}
        out = Path(__file__).resolve().parent.parent / "data" / "results" / "render_demo.png"
        print("Rendering demo (needs generated images + playwright)...")
        print("Wrote", render_pair_screenshot(left, right, out))
    else:
        print("Use --demo, or import render_pair_screenshot() from experiment_multimodal.")
