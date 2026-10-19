"""Product pair loading and structural quality-control checks.

A "pair" is two near-identical products (headphones or coffee machines) matched on base price,
rating, and rating count. Descriptions are neutral by construction; this module
verifies the structural matching that the price-sweep design relies on (identical
price/rating, matched description length, identical specs) so that the only
manipulated variables are the nudge and the price gap.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRODUCTS_DIR = Path(__file__).resolve().parent.parent / "data" / "products"
DATA_PATH = PRODUCTS_DIR / "headphones_pairs.json"

# Named product categories -> their pair files. `--category <name>` in the runners
# resolves through here; an explicit path is also accepted.
CATEGORY_FILES = {
    "headphones": DATA_PATH,
    "coffee": PRODUCTS_DIR / "coffee_machines_pairs.json",
}


def resolve_category(category: str | Path) -> Path:
    """Map a category name (or a path) to a pairs file."""
    if isinstance(category, str) and category in CATEGORY_FILES:
        return CATEGORY_FILES[category]
    p = Path(category)
    if p.exists():
        return p
    raise KeyError(f"unknown category '{category}'. Known: {list(CATEGORY_FILES)} "
                   f"(or pass an existing file path).")


@dataclass
class Product:
    id: str
    name: str
    brand: str
    neutral_description: str
    specs: dict[str, Any]
    image_standard: str = ""
    image_optimized: str = ""

    @property
    def word_count(self) -> int:
        return len(re.findall(r"\b\w+\b", self.neutral_description))


@dataclass
class Pair:
    pair_id: str
    subcategory: str
    base_price: float
    rating: float
    rating_count: int
    product_a: Product
    product_b: Product
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def products(self) -> tuple[Product, Product]:
        return self.product_a, self.product_b


@dataclass
class QCResult:
    pair_id: str
    passed: bool
    issues: list[str]

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        body = "" if self.passed else "\n  - " + "\n  - ".join(self.issues)
        return f"[{status}] {self.pair_id}{body}"


def _product_from_dict(d: dict[str, Any]) -> Product:
    return Product(
        id=d["id"],
        name=d["name"],
        brand=d.get("brand", ""),
        neutral_description=d["neutral_description"],
        specs=d.get("specs", {}),
        image_standard=d.get("image_standard", ""),
        image_optimized=d.get("image_optimized", ""),
    )


def load_pairs(path: Path | str = DATA_PATH) -> list[Pair]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs: list[Pair] = []
    for p in data["pairs"]:
        pairs.append(
            Pair(
                pair_id=p["pair_id"],
                subcategory=p.get("subcategory", ""),
                base_price=float(p["base_price"]),
                rating=float(p["rating"]),
                rating_count=int(p["rating_count"]),
                product_a=_product_from_dict(p["product_a"]),
                product_b=_product_from_dict(p["product_b"]),
                raw=p,
            )
        )
    return pairs


def load_nudges(path: Path | str = DATA_PATH) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("nudges", {})


def structural_qc(pair: Pair, max_word_delta: float = 0.15) -> QCResult:
    """Verify the structural matching the design relies on.

    Checks: identical specs, matched description length (within +/- max_word_delta),
    and that the description repeats price/rating identically (it should repeat
    neither, but if one does, both must).
    """
    issues: list[str] = []
    a, b = pair.product_a, pair.product_b

    if a.specs != b.specs:
        diff_keys = sorted(set(a.specs) ^ set(b.specs)) or [
            k for k in a.specs if a.specs.get(k) != b.specs.get(k)
        ]
        issues.append(f"specs differ on: {diff_keys}")

    wa, wb = a.word_count, b.word_count
    if max(wa, wb) > 0:
        delta = abs(wa - wb) / max(wa, wb)
        if delta > max_word_delta:
            issues.append(
                f"description length mismatch: {wa} vs {wb} words "
                f"({delta:.0%} > {max_word_delta:.0%})"
            )

    # Neither description should embed price or rating numerals (those live in the
    # card UI / structured fields). If one does, it creates an unmatched cue.
    for prod in (a, b):
        if re.search(rf"\b{int(pair.rating)}\b\.\d", prod.neutral_description):
            issues.append(f"{prod.id}: description embeds the rating value")
        if re.search(r"\$\s?\d", prod.neutral_description):
            issues.append(f"{prod.id}: description embeds a price")

    return QCResult(pair_id=pair.pair_id, passed=not issues, issues=issues)


def run_all_qc(path: Path | str = DATA_PATH) -> list[QCResult]:
    return [structural_qc(p) for p in load_pairs(path)]


if __name__ == "__main__":
    import sys
    category = sys.argv[1] if len(sys.argv) > 1 else "headphones"
    path = resolve_category(category)
    print(f"Loading pairs from: {path}")
    results = run_all_qc(path)
    for r in results:
        print(r)
    ok = sum(r.passed for r in results)
    print(f"\n{ok}/{len(results)} pairs passed structural QC.")
