"""Checkpointing for expensive measurement runs (screening/confirmation).

Every fully measured candidate is persisted IMMEDIATELY. A restart with the same
run-id loads finished results (0 API) and only measures the missing candidates. This makes
an abort (Ctrl-C, API failure or interruption) safe at any time.

The cache key binds model + n + seed + grid + candidate text; if any of these changes,
the candidate is re-measured (no silent incorrect reuse).
"""
from __future__ import annotations
import hashlib, json, os


def key(model_name, n, seed, grid, text) -> str:
    s = json.dumps([str(model_name), int(n), int(seed),
                    [round(float(g), 6) for g in grid], text], sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def path_for(out_md_path) -> str:
    return str(out_md_path).replace(".md", ".ckpt.json")


def load(path) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save(path, data) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)          # atomic
