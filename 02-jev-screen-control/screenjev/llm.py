"""Chat LLM client (OpenAI SDK, Chat Completions) for the planner and the LLM baselines.

Model routing:
  "gpt-6-luna" or "openai/gpt-6-luna"  -> api.openai.com with OPENAI_API_KEY
  any other "provider/model"            -> Vercel AI Gateway with AI_GATEWAY_API_KEY
                                           (or OpenRouter with OPENROUTER_API_KEY if SCREENJEV_LLM_VIA=openrouter)
SCREENJEV_MOCK=1 swaps in a scripted stand-in (plumbing tests only; never reported).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .decide import is_mock

DEFAULT_PLANNER = os.getenv("SCREENJEV_PLANNER_MODEL", "gpt-6-luna")
# $ per 1M tokens (input, output) for cost accounting; unknown models report tokens only.
# Override with SCREENJEV_LLM_PRICE="in,out". Check the provider's pricing page before quoting.
PRICES: dict[str, tuple[float, float]] = {}


@dataclass
class LLMReply:
    text: str
    data: dict[str, Any]
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    mock: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def route(model: str) -> tuple[str | None, str, str]:
    """(base_url, key env var, model name to send)."""
    if "/" not in model or model.startswith("openai/"):
        if os.getenv("OPENAI_API_KEY"):
            return None, "OPENAI_API_KEY", model.split("/", 1)[-1]
        model = model if "/" in model else f"openai/{model}"
    if os.getenv("SCREENJEV_LLM_VIA") == "openrouter":
        return "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", model
    return os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1"), "AI_GATEWAY_API_KEY", model


def image_part(img: Image.Image, max_side: int = 1600) -> dict[str, Any]:
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


class LLM:
    def __init__(self, model: str = DEFAULT_PLANNER, timeout: float = 120.0, max_retries: int = 6):
        self.model = model
        self.mock = is_mock()
        self.scripted = None  # mock: callable(messages) -> dict
        if not self.mock:
            from openai import AsyncOpenAI
            base, key_var, self.wire_model = route(model)
            key = os.getenv(key_var)
            if not key:
                raise RuntimeError(f"{key_var} is not set (needed for {model}; SCREENJEV_MOCK=1 for a dry run)")
            self.client = AsyncOpenAI(api_key=key, base_url=base, timeout=timeout, max_retries=max_retries)
        price = os.getenv("SCREENJEV_LLM_PRICE")
        self.price = tuple(float(x) for x in price.split(",")) if price else PRICES.get(model.split("/")[-1])

    async def json(self, system: str, content: list[dict[str, Any]] | str) -> LLMReply:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        if self.mock:
            data = self.scripted(messages) if self.scripted else {}
            return LLMReply(json.dumps(data), data, 0.0, model=f"mock:{self.model}", mock=True)
        t0 = time.perf_counter()
        resp = await self.client.chat.completions.create(model=self.wire_model, messages=messages,
                                                         response_format={"type": "json_object"})
        latency = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        u = resp.usage
        tin, tout = (u.prompt_tokens, u.completion_tokens) if u else (0, 0)
        cost = (tin * self.price[0] + tout * self.price[1]) / 1e6 if self.price else 0.0
        return LLMReply(text, parse_json(text), latency, tin, tout, cost, resp.model or self.model)

    async def aclose(self):
        if not self.mock:
            await self.client.close()
