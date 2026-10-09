"""Decisions API client (same as experiment 01's jevroute/decide.py, env vars prefixed SCREENJEV_).

Client for the OpenAI-compatible Decisions API on Vercel AI Gateway.

    POST https://ai-gateway.vercel.sh/v1/decisions
    {model, input, questions: [{type: predicate|choice|score, name, instructions, choices|levels}]}

Any decision model on the gateway works (typesafe-ai/jev, openai/gpt-6-luna, ...). OpenAI models
("openai/<model>") go straight to api.openai.com instead when OPENAI_API_KEY is set: same
Decisions API, same request and response shapes. Jev ("typesafe-ai/jev") can instead go through
OpenRouter's Decisions API (POST /api/alpha/decisions, TypeSafe's native request shape) when
SCREENJEV_JEV_VIA=openrouter; requests are translated so callers always use the OpenAI shape.
Set SCREENJEV_MOCK=1 to answer locally with a lexical heuristic: for plumbing tests only,
results produced that way are tagged mock and must never be reported.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

GATEWAY_BASE_URL = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
OPENAI_BASE_URL = "https://api.openai.com/v1"
MAX_CHOICES = 255
# List input price per 1M tokens, for responses that don't report cost (OpenAI direct).
# Decision calls bill input only. Check https://openai.com/api/pricing before quoting.
INPUT_PRICE_PER_MTOK = {"gpt-6-luna": 0.10}


OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_JEV = "typesafe/jev-1.13"


def backend(model: str) -> tuple[str, str, str]:
    """(base_url, api key env var, model name to send) for a decision model slug."""
    if model.startswith("openai/") and os.getenv("OPENAI_API_KEY") and not os.getenv("SCREENJEV_FORCE_GATEWAY"):
        return OPENAI_BASE_URL, "OPENAI_API_KEY", model.split("/", 1)[1]
    if model == "typesafe-ai/jev" and os.getenv("SCREENJEV_JEV_VIA") == "openrouter":
        return OPENROUTER_DECISIONS_URL, "OPENROUTER_API_KEY", OPENROUTER_JEV
    allow = [m for m in os.getenv("SCREENJEV_GATEWAY_ALLOW", "").split(",") if m]
    if allow and model not in allow:  # e.g. a key the owner scoped to one model
        raise DecisionError(f"{model} is not in SCREENJEV_GATEWAY_ALLOW ({', '.join(allow)}); refusing to use the gateway key")
    return GATEWAY_BASE_URL, "AI_GATEWAY_API_KEY", model


def has_key(model: str) -> bool:
    try:
        return bool(os.getenv(backend(model)[1]))
    except DecisionError:
        return False


class TokenRateLimiter:
    """Keeps estimated input tokens per rolling minute under a cap (OpenAI enforces TPM per model).

    Retries alone can't keep up once a run saturates the limit; pacing requests up front can.
    """

    def __init__(self, tpm: int):
        self.tpm = tpm
        self.window: deque[tuple[float, int]] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                while self.window and now - self.window[0][0] > 60:
                    self.window.popleft()
                if sum(t for _, t in self.window) + tokens <= self.tpm or not self.window:
                    self.window.append((now, tokens))
                    return
                await asyncio.sleep(60 - (now - self.window[0][0]) + 0.05)


_LIMITERS: dict[str, TokenRateLimiter] = {}


def _limiter(base_url: str) -> TokenRateLimiter | None:
    default = "1500000" if base_url == OPENAI_BASE_URL else "0"  # 75% of gpt-6-luna's 2M TPM
    tpm = int(os.getenv("SCREENJEV_TPM", default))
    if tpm <= 0:
        return None
    return _LIMITERS.setdefault(base_url, TokenRateLimiter(tpm))


def estimate_tokens(input: str, questions: list[dict[str, Any]]) -> int:
    # ~5 chars/token matches OpenAI's reported usage on these requests (JSON choice lists)
    return int((len(input) + len(json.dumps(questions))) / 5) + 50


@dataclass
class Answer:
    type: str
    name: str | None
    choice: Any = None
    confidence: float | None = None
    probability: float | None = None  # predicate
    probabilities: dict[Any, float] = field(default_factory=dict)  # choice / score
    score: float | None = None

    def ranked(self) -> list[tuple[Any, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)


@dataclass
class Decision:
    model: str
    answers: dict[str, Answer]
    latency_s: float
    input_tokens: int
    cost_usd: float
    mock: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


class DecisionError(RuntimeError):
    pass


def is_mock() -> bool:
    return os.getenv("SCREENJEV_MOCK", "").lower() in ("1", "true", "yes")


class DecisionClient:
    def __init__(self, model: str = "typesafe-ai/jev", api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, max_retries: int = 10):  # 429s back off and retry
        self.model = model
        self.mock = is_mock()
        self._client = None
        self.limiter = None
        self.via = "mock" if self.mock else None
        if not self.mock:
            url, key_var, self.wire_model = backend(model)
            self.via = url
            self.limiter = _limiter(url)
            key = api_key or os.getenv(key_var)
            if not key:
                raise DecisionError(f"{key_var} is not set (needed for {model}; SCREENJEV_MOCK=1 for a dry run)")
            if url == OPENROUTER_DECISIONS_URL:
                self._client = _NativeClient(url, key, timeout, max_retries)
            else:
                self._client = AsyncOpenAI(api_key=key, base_url=base_url or url, timeout=timeout,
                                           max_retries=max_retries)

    async def decide(self, input: str, questions: list[dict[str, Any]]) -> Decision:
        for q in questions:
            if q["type"] == "choice" and not 1 <= len(q["choices"]) <= MAX_CHOICES:
                raise DecisionError(f"choice question {q.get('name')!r} has {len(q['choices'])} options (1..255)")
        if self.mock:
            return _mock_decide(self.model, input, questions)

        if self.limiter:
            await self.limiter.acquire(estimate_tokens(input, questions))
        t0 = time.perf_counter()
        if isinstance(self._client, _NativeClient):
            body = await self._client.decide(self.wire_model, input, questions)
        else:
            resp = await self._client.decisions.with_raw_response.create(model=self.wire_model, input=input,
                                                                        questions=questions)
            body = resp.http_response.json()
        latency = time.perf_counter() - t0
        if "error" in body:
            raise DecisionError(body["error"].get("message", str(body["error"])))
        d = _parse(body, latency)
        if not d.cost_usd and self.wire_model in INPUT_PRICE_PER_MTOK:
            d.cost_usd = d.input_tokens * INPUT_PRICE_PER_MTOK[self.wire_model] / 1e6
        return d

    async def aclose(self) -> None:
        if self._client:
            await self._client.close()


class _NativeClient:
    """TypeSafe-native Decisions endpoint (OpenRouter /api/alpha/decisions), spoken in the OpenAI shape.

    OpenAI shape            native shape
    input                   state
    [{name, type, ...}]     {name: {type, instructions, criteria}}
    predicate               noul                (answer: noul -> probability)
    choices [{value, d}]    criteria {value: d} (answer probabilities: {value: p})
    levels [{label}]        criteria [label]
    """

    RETRY = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, url: str, key: str, timeout: float, max_retries: int):
        import httpx
        self.url = url
        self.max_retries = max_retries
        self.http = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {key}"})

    @staticmethod
    def to_native(questions: list[dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for i, q in enumerate(questions):
            name = q.get("name") or f"q{i}"
            if q["type"] == "choice":
                out[name] = {"type": "choice", "instructions": q["instructions"],
                             "criteria": {c["value"]: c.get("description") or c["value"] for c in q["choices"]}}
            elif q["type"] == "predicate":
                out[name] = {"type": "noul", "instructions": q["instructions"]}
            elif q["type"] == "score":
                out[name] = {"type": "score", "instructions": q["instructions"],
                             "criteria": [l["label"] for l in q["levels"]]}
            else:
                raise DecisionError(f"unsupported question type {q['type']!r}")
        return out

    @staticmethod
    def from_native(body: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, Any]:
        answers = []
        for i, q in enumerate(questions):
            name = q.get("name") or f"q{i}"
            a = (body.get("answers") or {}).get(name) or {"type": "refusal"}
            if a["type"] == "noul":
                answers.append({"type": "predicate", "name": name, "probability": a.get("noul")})
            elif a["type"] in ("choice", "score"):
                probs = [{"value": k, "probability": v} for k, v in (a.get("probabilities") or {}).items()]
                answers.append({"type": a["type"], "name": name, "choice": a.get("choice"), "score": a.get("score"),
                                "confidence": a.get("confidence"), "probabilities": probs})
            else:
                answers.append({"type": "refusal", "name": name})
        usage = body.get("usage") or {}
        return {"model": body.get("model"), "answers": answers, "usage": usage, "provider": body.get("provider"),
                "provider_metadata": {"gateway": {"cost": usage.get("cost")}}}

    async def decide(self, model: str, input: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {"model": model, "state": input, "questions": self.to_native(questions)}
        for attempt in range(self.max_retries + 1):
            r = await self.http.post(self.url, json=payload)
            if r.status_code in self.RETRY and attempt < self.max_retries:
                wait = float(r.headers.get("retry-after") or 0) or min(2 ** attempt, 30)
                await asyncio.sleep(wait)
                continue
            body = r.json()
            if r.status_code >= 400 or "error" in body:
                err = body.get("error", body)
                raise DecisionError(f"HTTP {r.status_code}: {err.get('message', err) if isinstance(err, dict) else err}")
            return self.from_native(body, questions)
        raise DecisionError("unreachable")

    async def close(self):
        await self.http.aclose()


def _parse(body: dict[str, Any], latency: float) -> Decision:
    answers: dict[str, Answer] = {}
    for i, a in enumerate(body.get("answers", [])):
        name = a.get("name") or f"q{i}"
        probs = {p.get("value"): float(p.get("probability", 0.0)) for p in a.get("probabilities") or []}
        answers[name] = Answer(
            type=a.get("type"), name=name, choice=a.get("choice"), confidence=a.get("confidence"),
            probability=a.get("probability"), probabilities=probs, score=a.get("score"),
        )
    usage = body.get("usage") or {}
    meta = (body.get("provider_metadata") or body.get("providerMetadata") or {}).get("gateway", {})
    try:
        cost = float(meta.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    return Decision(
        model=body.get("model", ""), answers=answers, latency_s=latency,
        input_tokens=int(usage.get("input_tokens") or usage.get("inputTokens") or 0), cost_usd=cost, raw=body,
    )


# --- mock backend -------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(text.lower()):
        if len(w) > 2:
            out.add(w[:6])  # crude stemming so "repositories" ~ "repository"
    return out


def _mock_decide(model: str, input: str, questions: list[dict[str, Any]]) -> Decision:
    t0 = time.perf_counter()
    state = _words(input)
    answers = {}
    for i, q in enumerate(questions):
        name = q.get("name") or f"q{i}"
        if q["type"] == "choice":
            scores = []
            for c in q["choices"]:
                words = _words(f"{c['value']} {c.get('description', '')}")
                scores.append(len(state & words) / math.sqrt(len(words) + 1))
            z = [math.exp(4 * s) for s in scores]
            total = sum(z)
            probs = {c["value"]: zi / total for c, zi in zip(q["choices"], z)}
            best = max(probs, key=probs.get)
            answers[name] = Answer("choice", name, choice=best, confidence=probs[best], probabilities=probs)
        elif q["type"] == "predicate":
            overlap = len(state & _words(q["instructions"]))
            answers[name] = Answer("predicate", name, probability=1 - math.exp(-0.7 * overlap))
        else:
            raise DecisionError(f"mock backend does not support {q['type']!r}")
    n_tokens = len(input) // 4 + sum(len(str(q)) // 4 for q in questions)
    return Decision(model=f"mock:{model}", answers=answers, latency_s=time.perf_counter() - t0,
                    input_tokens=n_tokens, cost_usd=0.0, mock=True)
