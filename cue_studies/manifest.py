"""Run manifest for reproducibility.

Every run writes manifest_<run>.json with what is needed to reproduce it:
model version + region, UTC time, git commit, seeds, sampling, prompt info,
pool hash, cost. Hosted LLMs are NOT fully deterministic; the manifest states this explicitly.
"""
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path: str) -> str:
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except Exception:
        return "unknown"


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _sdk_versions() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("openai", "boto3", "botocore", "numpy", "pandas"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "n/a"
    return v


def resolve_model(alias: str) -> dict:
    """Resolved model ID/version for the alias (Bedrock: exact model_id; OpenAI: alias +
    a note that it is a rolling alias)."""
    try:
        from models import BEDROCK_MODELS
        if alias in BEDROCK_MODELS:
            return {"alias": alias, "model_id": BEDROCK_MODELS[alias]["model_id"],
                    "provider": BEDROCK_MODELS[alias].get("family", "bedrock")}
    except Exception:
        pass
    if alias == "mock":
        return {"alias": "mock", "model_id": "mock", "provider": "mock"}
    return {"alias": alias, "model_id": alias, "provider": "openai",
            "note": "rolling alias; snapshot not pinned (see README)"}


def build(run_id: str, model_aliases: list[str], cfg: dict, pool, guard=None,
          extra: dict | None = None) -> dict:
    m = {
        "run_id": run_id,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "models": [resolve_model(a) for a in model_aliases],
        "region": os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or "n/a",
        "sdk_versions": _sdk_versions(),
        "sampling": {
            "note": "Hosted LLMs are NOT fully deterministic, even with fixed seed/temperature.",
            **{k: cfg.get(k) for k in ("seed", "temperature", "prompt_variant", "label_mode") if k in cfg},
        },
        "pool_sha256": _sha256_obj(pool),
        "config": cfg,
        "cost": ({"spent_usd": round(guard.spent, 4), "calls": guard.calls,
                  "cap_usd": guard.max} if guard is not None else None),
    }
    if extra:
        m.update(extra)
    return m


def write(path: str, manifest: dict) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return path
