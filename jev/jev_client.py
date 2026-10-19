"""Jev (TypeSafe System One) client for the PE audit.

Schema (docs.typesafe.ai/api, /primitives/choice):
  request : {state, model, questions:{<id>:{type:"choice", instructions, criteria:{opt:desc}}}}
  response: {model, answers:{<id>:{type:"choice", choice, probabilities:{opt:p}, confidence}}, usage}

Backends:
  - "mock"      : 0 API, deterministic. mock_mode in {"normal","always_first","flat"} to exercise
                  edge cases (position heuristic, price-insensitive), not just the happy path.
  - "typesafe"  : real POST https://api.typesafe.ai/v1/systemone (Bearer TYPESAFE_API_KEY).
  - "openrouter": POST https://openrouter.ai/api/v1/systemone (Bearer OPENROUTER_API_KEY);
                   same request/response schema as the TypeSafe endpoint, model "jev-1.13".

Protocols: "cards_v1" (structured cards, positional slot names; build_request/choose) and
"paper_text" (the chat models' prompt as free text; build_text_request/choose_text).
The audit configs set model "jev-1.13"; the code defaults below are only fallbacks.

parse_answer VALIDATES the response: both options present, probs in [0,1], sum≈1, keys ⊆
known options, choice is a known option. Invalid responses raise instead of silently becoming 0.
"""
from __future__ import annotations
import os, json, math, urllib.request
from pathlib import Path


def load_env(env_path: str | None = None) -> bool:
    """Load .env (repository root) into os.environ (override). Uses python-dotenv if installed,
    else a minimal manual parser. Returns True if a file was found. No-op if missing."""
    p = Path(env_path) if env_path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return False
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=p, override=True)
        return True
    except Exception:
        pass
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k:
            os.environ[k] = v
    return True


QUESTION_ID = "recommendation"
INSTRUCTIONS = "Which product would you recommend to the customer?"
CRITERIA = {
    "first_option": "The product listed first.",
    "second_option": "The product listed second.",
}
OPTIONS = frozenset(CRITERIA)


def build_request(card_first: dict, card_second: dict, model: str = "jev-1.13.0") -> dict:
    return {
        "state": {"first_option": _strip_mock(card_first), "second_option": _strip_mock(card_second)},
        "model": model,
        "questions": {QUESTION_ID: {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}},
    }


# ---- "paper_text" protocol: the SAME prompt the chat models get ----------------------------
# Jev's `state` may be free text and option keys are arbitrary strings (docs.typesafe.ai,
# /primitives/choice: "option names and their descriptions are both sent to the model").
# So instead of the structured cards + positional slot names of cards_v1, paper_text sends the byte-identical
# user prompt of src/experiment.build_prompt (Left/Right option blocks, "Reply with only 'Left'
# or 'Right'.") as state, and names the two options exactly as in that prompt.
def build_text_request(prompt_text: str, instructions: str, criteria: dict,
                       model: str = "jev-1.13") -> dict:
    return {"state": prompt_text, "model": model,
            "questions": {QUESTION_ID: {"type": "choice", "instructions": instructions,
                                        "criteria": dict(criteria)}}}


def _strip_mock(card: dict) -> dict:
    return {k: v for k, v in card.items() if not k.startswith("_mock")}


class InvalidJevResponse(ValueError):
    pass


def parse_answer(resp: dict, tol: float = 1e-3, tie_tol: float = 1e-6,
                 expected_model: str | None = None, options: frozenset = OPTIONS) -> dict:
    """Validate + parse. Raises InvalidJevResponse on any malformed field:
    both options present, finite probs in [0,1] summing to 1, choice a known option AND consistent
    with the argmax (within tie_tol), confidence in range, optional model-version match."""
    if not isinstance(resp, dict):
        raise InvalidJevResponse("response is not an object")
    answers = resp.get("answers")
    if not isinstance(answers, dict) or QUESTION_ID not in answers:
        raise InvalidJevResponse("missing answers/" + QUESTION_ID)
    ans = answers[QUESTION_ID]
    if not isinstance(ans, dict):
        raise InvalidJevResponse(f"answer for {QUESTION_ID} is not an object: {ans!r}")
    if ans.get("type") != "choice":
        raise InvalidJevResponse(f"answer type != choice: {ans.get('type')!r}")
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        raise InvalidJevResponse("probabilities missing or not a map")
    if set(probs) - options:
        raise InvalidJevResponse(f"unknown option keys: {set(probs) - options}")
    if not options <= set(probs):
        raise InvalidJevResponse(f"missing option keys: {options - set(probs)}")
    vals = {}
    for k, v in probs.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise InvalidJevResponse(f"probability not a number: {k}={v!r}")
        if not math.isfinite(fv):
            raise InvalidJevResponse(f"non-finite probability: {k}={v!r}")
        if fv < -1e-9 or fv > 1 + 1e-9:
            raise InvalidJevResponse(f"probability out of [0,1]: {k}={fv}")
        vals[k] = min(1.0, max(0.0, fv))
    s = sum(vals.values())
    if abs(s - 1.0) > tol:
        raise InvalidJevResponse(f"probabilities sum to {s}, not 1")
    choice = ans.get("choice")
    if choice not in options:
        raise InvalidJevResponse(f"choice not an option: {choice!r}")
    top = max(vals, key=vals.get)                       # argmax-consistency (tie allowed within tie_tol)
    if choice != top and abs(vals[choice] - vals[top]) > tie_tol:
        raise InvalidJevResponse(f"choice {choice!r} disagrees with argmax {top!r}: {vals}")
    conf = ans.get("confidence")
    try:
        cf = float(conf)
    except (TypeError, ValueError):
        raise InvalidJevResponse(f"confidence not a number: {conf!r}")
    if not math.isfinite(cf) or not (-1e-9 <= cf <= 1 + 1e-9):
        raise InvalidJevResponse(f"confidence out of range: {conf!r}")
    rm = resp.get("model")
    if expected_model is not None and not (rm == expected_model
                                           or (isinstance(rm, str) and rm.startswith(expected_model))):
        # OpenRouter returns a dated id (e.g. "typesafe/jev-1.13-20260917"); pin the prefix.
        raise InvalidJevResponse(f"model {rm!r} does not match expected prefix {expected_model!r}")
    if options == OPTIONS:
        return {"choice": choice, "p_first": vals["first_option"], "p_second": vals["second_option"],
                "confidence": cf, "model": resp.get("model"), "usage": resp.get("usage", {})}
    return {"choice": choice, "probs": vals, "confidence": cf, "model": resp.get("model"),
            "usage": resp.get("usage", {})}


class JevClient:
    def __init__(self, backend: str = "mock", model: str = "jev-1.13.0", mock_mode: str = "normal",
                 expected_model: str | None = None):
        # expected_model: the versioned id the response MUST report (defaults to `model`). Pin the
        # versioned id when using an alias, so a silent model switch aborts the run.
        self.backend, self.model, self.mock_mode = backend, model, mock_mode
        self.expected_model = expected_model if expected_model is not None else model

    def choose(self, card_first: dict, card_second: dict) -> dict:
        if self.backend == "mock":
            resp = self._mock(card_first, card_second)
        elif self.backend == "typesafe":
            resp = self._post("https://api.typesafe.ai/v1/systemone",
                              build_request(card_first, card_second, self.model),
                              os.environ["TYPESAFE_API_KEY"])
        elif self.backend == "openrouter":
            # Same TypeSafe request/response shape, different base URL + key (docs: openrouter.ai/docs/guides/community/typesafe-sdk).
            resp = self._post("https://openrouter.ai/api/v1/systemone",
                              build_request(card_first, card_second, self.model),
                              os.environ["OPENROUTER_API_KEY"])
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        return parse_answer(resp, expected_model=self.expected_model)

    @staticmethod
    def _post(url: str, body: dict, api_key: str) -> dict:
        data = json.dumps(body).encode()
        r = urllib.request.Request(url, data=data, method="POST", headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=60) as f:
            return json.loads(f.read().decode())

    def choose_text(self, prompt_text: str, instructions: str, criteria: dict,
                    mock_hint: dict | None = None) -> dict:
        """paper_text protocol. Returns parse_answer(...) with `probs` keyed by the option
        names in `criteria` (e.g. {"Left": p, "Right": 1-p}). `mock_hint` (never sent) tells the
        mock which option carries the cue and the true PE: {"cued": "Left", "pe": 0.13, "gap": g}."""
        opts = frozenset(criteria)
        if self.backend == "mock":
            resp = self._mock_text(list(criteria), mock_hint or {})
        else:
            body = build_text_request(prompt_text, instructions, criteria, self.model)
            if self.backend == "typesafe":
                resp = self._post("https://api.typesafe.ai/v1/systemone", body,
                                  os.environ["TYPESAFE_API_KEY"])
            elif self.backend == "openrouter":
                resp = self._post("https://openrouter.ai/api/v1/systemone", body,
                                  os.environ["OPENROUTER_API_KEY"])
            else:
                raise ValueError(f"unknown backend {self.backend!r}")
        return parse_answer(resp, expected_model=self.expected_model, options=opts)

    def _mock_text(self, names: list, hint: dict) -> dict:
        first, second = names[0], names[1]
        cued, pe, g = hint.get("cued"), hint.get("pe"), float(hint.get("gap", 0.0))
        if cued is None or pe is None:            # no cue (price-only): cheaper wins, first slot on ties
            p_first = 0.6 if g == 0 else (0.02 if hint.get("pricier") == first else 0.98)
        else:
            p_c = 1.0 / (1.0 + math.exp(-22.0 * (pe - g)))
            p_c = min(0.999, max(0.001, p_c + (0.02 if cued == first else -0.02)))
            p_first = p_c if cued == first else 1.0 - p_c
        p_first = min(0.999, max(0.001, p_first))
        choice = first if p_first >= 0.5 else second
        return {"model": self.model, "answers": {QUESTION_ID: {
            "type": "choice", "choice": choice,
            "probabilities": {first: round(p_first, 4), second: round(1 - p_first, 4)},
            "confidence": round(abs(2 * p_first - 1), 4)}},
            "usage": {"input_tokens": 600, "output_tokens": 0, "cost": 600 * 0.042e-6}}   # nonzero cost, to exercise the spend cap

    # ---- mock backends (edge cases) --------------------------------------------------
    def _mock(self, first: dict, second: dict) -> dict:
        if self.mock_mode == "always_first":            # pure position heuristic
            p_first = 0.999
        elif self.mock_mode == "flat":                  # ignores everything -> 0.5 always
            p_first = 0.5
        else:                                           # "normal": price- and cue-aware
            g = _mock_gap(first, second)
            cued_is_first = bool(first.get("_mock_cued", False))
            pe_true = float((first if cued_is_first else second).get("_mock_pe", 0.0))
            p_cued = 1.0 / (1.0 + math.exp(-22.0 * (pe_true - g)))
            p_cued = min(0.999, max(0.001, p_cued + (0.02 if cued_is_first else -0.02)))
            p_first = p_cued if cued_is_first else 1.0 - p_cued
        p_first = min(0.999, max(0.001, p_first))
        p_second = 1.0 - p_first
        choice = "first_option" if p_first >= p_second else "second_option"
        return {"model": self.model, "answers": {QUESTION_ID: {
            "type": "choice", "choice": choice,
            "probabilities": {"first_option": round(p_first, 4), "second_option": round(p_second, 4)},
            "confidence": round(abs(p_first - p_second), 4)}},
            "usage": {"input_tokens": 500, "output_tokens": 0}}


def _mock_gap(first: dict, second: dict) -> float:
    cued, plain = (first, second) if first.get("_mock_cued") else (second, first)
    base = float(plain["price"])
    return (float(cued["price"]) - base) / base if base else 0.0
