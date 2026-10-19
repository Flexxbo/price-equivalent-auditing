"""Model wrappers for the shopping-agent decision call.

Three backends:
  - OpenAIModel:  OpenAI chat models such as GPT-4o (text or multimodal).
  - BedrockModel: Amazon Bedrock models via the Converse API (text only).
  - MockModel:    deterministic-ish stub with a configurable nudge/price response,
                  so the full pipeline can be dry-run with no API key and no cost.

All expose:
    choose(prompt, image_path=None, parse_mode="left_right") -> Decision(raw_text, choice)
where choice is one of the labels for the given mode, or None (unparseable):
  - parse_mode="left_right" -> "Left" / "Right"
  - parse_mode="ab"         -> "A" / "B"
"""
from __future__ import annotations

import base64
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def _load_env_file() -> None:
    """Load the project's .env and let it OVERRIDE any pre-existing system env vars.

    Without override=True, a stale/expired OPENAI_API_KEY in the shell environment
    would take precedence over the project .env. We always
    want the key from .env (repository root) to win.
    """
    try:
        from dotenv import load_dotenv
    except Exception:  # dotenv is optional at import time
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=True)


_load_env_file()

# Label sets per parse mode. The FIRST entry is the "primary" slot (shown first /
# on the left); analysis.py uses that convention to measure position bias.
LABELS = {
    "left_right": ("Left", "Right"),
    "ab": ("A", "B"),
}


@dataclass
class Decision:
    raw_text: str
    choice: str | None  # one of LABELS[mode], or None


def parse_choice(text: str, mode: str = "left_right") -> str | None:
    """Parse a choice for the given label mode. Honors the *first* label mentioned.

    left_right: case-insensitive word match on 'left' / 'right'.
    ab:         case-sensitive word match on 'A' / 'B' (so 'a' inside words is ignored).
    """
    if not text:
        return None
    if mode == "ab":
        specs = [("A", r"\bA\b"), ("B", r"\bB\b")]
        flags = 0
    else:
        specs = [("Left", r"\bleft\b"), ("Right", r"\bright\b")]
        flags = re.IGNORECASE
    best_label, best_pos = None, None
    for label, pat in specs:
        m = re.search(pat, text, flags)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos, best_label = m.start(), label
    return best_label


def _is_permanent_error(e) -> bool:
    """True for non-transient API errors (quota/auth/billing) where retrying is pointless."""
    s = str(e).lower()
    return any(t in s for t in (
        "insufficient_quota", "exceeded your current quota", "invalid_api_key",
        "incorrect api key", "401", "billing",
        # Bedrock permanent errors (won't fix by retrying):
        "accessdenied", "unrecognizedclient", "validationexception",
        "is not authorized", "could not be found", "don't have access",
        "invalid model identifier",
    ))


class BaseModel:
    name: str = "base"

    def choose(self, prompt: str, image_path: str | None = None,
               parse_mode: str = "left_right") -> Decision:
        raise NotImplementedError


class OpenAIModel(BaseModel):
    def __init__(
        self,
        name: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 4,
        system_prompt: str | None = None,
        budget=None,
    ):
        from openai import OpenAI  # imported lazily so MockModel works without the dep

        self.name = name or os.getenv("PRIMARY_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_retries = max_retries
        self.budget = budget  # optional BudgetGuard; enforced after every call
        self.system_prompt = system_prompt or (
            "You are a shopping assistant helping a customer choose a product."
        )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Add it to .env (repository root) "
                "(copy .env.example to .env)."
            )
        # Pass the key explicitly so the .env value is always used, never a stale
        # system environment variable the OpenAI client might otherwise pick up.
        self.client = OpenAI(api_key=api_key)
        # Cumulative token usage across all calls (for cost estimation).
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        # Serialises usage-counter updates when several worker threads share this
        # model instance (--concurrency). The OpenAI client itself is thread-safe.
        self._acct_lock = threading.Lock()

    def _track(self, resp) -> None:
        u = getattr(resp, "usage", None)
        with self._acct_lock:
            if u is not None:
                self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            self.usage["calls"] += 1

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _build_messages(self, prompt: str, image_path: str | None) -> list[dict]:
        if image_path:
            b64 = self._encode_image(image_path)
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        else:
            user_content = prompt
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _is_newer(self) -> bool:
        """Reasoning-style OpenAI models use max_completion_tokens and no temperature
        (not used in the paper)."""
        n = self.name.lower()
        return n.startswith(("gpt-5", "o1", "o3", "o4"))

    def _create(self, messages, max_out: int, temperature: float):
        kw = {"model": self.name, "messages": messages}
        if self._is_newer():
            kw["max_completion_tokens"] = max_out    # reasoning models: generous token budget
        else:
            kw["max_tokens"] = max_out
            kw["temperature"] = temperature          # older models: temperature allowed
        return self.client.chat.completions.create(**kw)

    def choose(self, prompt: str, image_path: str | None = None,
               parse_mode: str = "left_right") -> Decision:
        messages = self._build_messages(prompt, image_path)
        max_out = 2048 if self._is_newer() else 8    # reasoning needs room, else empty answer
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._create(messages, max_out, self.temperature)
                self._track(resp)
                if self.budget is not None:
                    self.budget.add_chat(getattr(resp, "usage", None), self.name)
                text = (resp.choices[0].message.content or "").strip()
                return Decision(raw_text=text, choice=parse_choice(text, parse_mode))
            except Exception as e:
                last_err = e
                if _is_permanent_error(e):
                    break
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} retries: {last_err}")

    # Free-form completion used by the stimulus-audit judge and the cue_studies/ generator.
    def complete(self, prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
        max_out = max(max_tokens, 2048) if self._is_newer() else max_tokens
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._create([{"role": "user", "content": prompt}], max_out, temperature)
                self._track(resp)
                if self.budget is not None:
                    self.budget.add_chat(getattr(resp, "usage", None), self.name)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if _is_permanent_error(e):
                    break
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenAI call failed after {self.max_retries} retries: {last_err}")


class MockModel(BaseModel):
    """No-cost stub. Picks the nudged side with a probability that erodes as the
    nudged product's price premium grows.

    The experiment passes structured info the mock reads from the prompt:
      <<NUDGE_SIDE:Left>>  the label carried by the nudged product
      <<PRICE_GAP:0.10>>   the current price premium
      <<LABELS:Left,Right>> the two labels in use (so the mock works for A/B too)
    Real models never see these (the experiment strips them).
    """

    def __init__(self, name: str = "mock", nudge_pull: float = 0.72,
                 price_sensitivity: float = 8.0, seed: int | None = None, budget=None):
        self.name = name
        self.nudge_pull = nudge_pull
        self.price_sensitivity = price_sensitivity
        self.rng = random.Random(seed)
        self.budget = budget  # ignored (mock is free); accepted for a uniform interface

    def choose(self, prompt: str, image_path: str | None = None,
               parse_mode: str = "left_right") -> Decision:
        side_m = re.search(r"<<NUDGE_SIDE:([^>]+)>>", prompt)
        gap_m = re.search(r"<<PRICE_GAP:([-0-9.]+)>>", prompt)
        lab_m = re.search(r"<<LABELS:([^>]+)>>", prompt)
        labels = lab_m.group(1).split(",") if lab_m else list(LABELS.get(parse_mode, LABELS["left_right"]))
        nudge_side = side_m.group(1) if side_m else None
        price_gap = float(gap_m.group(1)) if gap_m else 0.0

        if nudge_side is None:
            choice = self.rng.choice(labels)
            return Decision(raw_text=f"(mock) {choice}", choice=choice)

        other = labels[1] if nudge_side == labels[0] else labels[0]
        import math
        logit = math.log(self.nudge_pull / (1 - self.nudge_pull))
        logit -= self.price_sensitivity * price_gap
        p_nudged = 1 / (1 + math.exp(-logit))
        pick_nudged = self.rng.random() < p_nudged
        choice = nudge_side if pick_nudged else other
        return Decision(raw_text=f"(mock) {choice}", choice=choice)


# --- Amazon Bedrock model registry (capability axis: strong + weak per family) ---
# Verify model ids and region for your account, e.g. with
#   aws bedrock list-foundation-models --query "modelSummaries[].modelId"
# Newer models often need a region-prefixed inference-profile id (e.g. "us.anthropic...").
# The alias (dict key) is the pricing key in budget.py.
BEDROCK_MODELS = {  # us-east-1 inference-profile ids
    "claude-sonnet": {"model_id": "us.anthropic.claude-sonnet-4-6",                 "tier": "strong", "family": "anthropic"},
    "claude-haiku":  {"model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",    "tier": "weak",   "family": "anthropic"},
    "nova-pro":      {"model_id": "us.amazon.nova-pro-v1:0",                         "tier": "strong", "family": "amazon"},
    "nova-micro":    {"model_id": "us.amazon.nova-micro-v1:0",                       "tier": "weak",   "family": "amazon"},
    "llama-70b":     {"model_id": "us.meta.llama3-3-70b-instruct-v1:0",             "tier": "strong", "family": "meta"},
    "llama-8b":      {"model_id": "us.meta.llama3-1-8b-instruct-v1:0",              "tier": "weak",   "family": "meta"},
}


class BedrockModel(BaseModel):
    """Amazon Bedrock via the unified Converse API. Same interface as OpenAIModel,
    so the experiment/analysis pipeline runs unchanged. `name` is the short alias
    (and the pricing key in budget.py); the real Bedrock id is `model_id`."""

    def __init__(self, name: str, temperature: float = 0.7, max_retries: int = 8,
                 system_prompt: str | None = None, budget=None):
        import boto3  # lazy
        from botocore.config import Config

        if name not in BEDROCK_MODELS:
            raise KeyError(f"unknown bedrock alias '{name}'. Known: {list(BEDROCK_MODELS)}")
        self.name = name
        self.model_id = BEDROCK_MODELS[name]["model_id"]
        self.temperature = temperature
        self.max_retries = max_retries
        self.system_prompt = system_prompt or (
            "You are a shopping assistant helping a customer choose a product.")
        self.budget = budget
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        region = os.getenv("BEDROCK_REGION") or os.getenv("AWS_REGION") or "us-east-1"
        # Adaptive retries: botocore self-throttles the client to stay under the
        # account's Bedrock rate quota, which fixes sustained ThrottlingExceptions
        # far better than app-level backoff alone. Longer timeouts for slow models.
        _cfg = Config(retries={"max_attempts": 10, "mode": "adaptive"},
                      read_timeout=90, connect_timeout=10)
        self.client = boto3.client("bedrock-runtime", region_name=region, config=_cfg)
        # Serialises usage-counter updates under --concurrency. boto3 low-level
        # clients are thread-safe for concurrent calls.
        self._acct_lock = threading.Lock()

    def _track(self, usage: dict) -> None:
        with self._acct_lock:
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0)
            self.usage["calls"] += 1

    def choose(self, prompt: str, image_path: str | None = None,
               parse_mode: str = "left_right") -> Decision:
        if image_path is not None:
            raise NotImplementedError("BedrockModel is text-only (no image input).")
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    system=[{"text": self.system_prompt}],
                    # maxTokens 24 so chatty models (e.g. Llama) still emit Left/Right
                    # after a short preamble. Nova Pro is given 128 because it deliberates
                    # before answering (its runs reported in the paper used 24).
                    inferenceConfig={"maxTokens": 128 if self.name == "nova-pro" else 24,
                                     "temperature": self.temperature},
                )
                u = resp.get("usage", {}) or {}
                norm = {"prompt_tokens": u.get("inputTokens", 0) or 0,
                        "completion_tokens": u.get("outputTokens", 0) or 0}
                self._track(norm)
                if self.budget is not None:
                    self.budget.add_chat(norm, self.name)
                text = resp["output"]["message"]["content"][0]["text"].strip()
                return Decision(raw_text=text, choice=parse_choice(text, parse_mode))
            except Exception as e:
                last_err = e
                if _is_permanent_error(e):
                    break
                throttled = any(t in str(e).lower() for t in ("throttl", "too many", "rate exceeded"))
                base = min(2 ** attempt, 30) + (8 if throttled else 0)
                time.sleep(base + random.uniform(0, 1.5))  # backoff + jitter
        raise RuntimeError(f"Bedrock call failed after {self.max_retries} retries: {last_err}")

    def complete(self, prompt: str, max_tokens: int = 400, temperature: float = 0.7) -> str:
        """Free-form text generation (used by the cue_studies/ generator). Mirrors
        `choose` but returns raw text and allows more tokens. Additive; does not
        affect the 2AFC pipeline."""
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    system=[{"text": "You are a precise product copywriter."}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
                )
                u = resp.get("usage", {}) or {}
                norm = {"prompt_tokens": u.get("inputTokens", 0) or 0,
                        "completion_tokens": u.get("outputTokens", 0) or 0}
                self._track(norm)
                if self.budget is not None:
                    self.budget.add_chat(norm, self.name)
                return resp["output"]["message"]["content"][0]["text"].strip()
            except Exception as e:
                last_err = e
                if _is_permanent_error(e):
                    break
                time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1.5))
        raise RuntimeError(f"Bedrock complete failed after {self.max_retries} retries: {last_err}")


def get_model(name: str = "gpt-4o", **kwargs) -> BaseModel:
    if name == "mock":
        return MockModel(**kwargs)
    if name in BEDROCK_MODELS:
        return BedrockModel(name=name, **kwargs)
    return OpenAIModel(name=name, **kwargs)
