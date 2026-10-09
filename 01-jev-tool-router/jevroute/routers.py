"""Routers rank a catalog's tools for the agent's current state.

Every router returns a full ranking from one call, so recall@1/3/5/10 all come from the
same request. `select()` turns a ranking into the tool list the agent actually gets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .catalog import Catalog
from .decide import GATEWAY_BASE_URL, MAX_CHOICES, DecisionClient, is_mock

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

NEXT_STEP = ("An AI agent is working on the task in the input, which may include its progress so far. "
             "Which ONE tool should the agent call next to make progress?")
WHOLE_TASK = ("An AI agent must complete the task in the input. "
              "Which tool is the most essential one for completing it?")


@dataclass
class Ranking:
    order: list[str]  # tool names, best first; tools the router never scored are appended at the end
    scores: dict[str, float]
    confidence: float | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0
    router_tokens: int = 0
    calls: int = 0
    mock: bool = False
    debug: dict[str, Any] = field(default_factory=dict)


def select(ranking: Ranking, k: int, adaptive_mass: float | None = None) -> list[str]:
    """Top-k, or with adaptive_mass the smallest prefix whose probability mass reaches it (capped at k).

    Adaptive selection is where calibration pays off: a confident router hands the agent
    one tool, an unsure one hands it up to k.
    """
    if adaptive_mass is None:
        return ranking.order[:k]
    picked, mass = [], 0.0
    for name in ranking.order[:k]:
        picked.append(name)
        mass += ranking.scores.get(name, 0.0)
        if mass >= adaptive_mass:
            break
    return picked


class Router(ABC):
    name = "router"

    @abstractmethod
    async def rank(self, state: str, catalog: Catalog, intent: str = "next") -> Ranking: ...

    async def aclose(self) -> None:
        pass


def _complete(order: list[str], catalog: Catalog) -> list[str]:
    seen = set(order)
    return order + [t.name for t in catalog.tools if t.name not in seen]


# --- baselines ----------------------------------------------------------------------------

class AllTools(Router):
    """No routing: the agent sees every tool (what Claude Code does today)."""
    name = "all"

    async def rank(self, state, catalog, intent="next"):
        return Ranking(order=[t.name for t in catalog.tools], scores={t.name: 1.0 for t in catalog.tools})


_SPLIT = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _tok(text: str) -> list[str]:
    return [w.lower() for w in _SPLIT.findall(text.replace("_", " ").replace("-", " "))]


class BM25(Router):
    """Lexical retrieval over tool name + server + description. Free and instant."""
    name = "bm25"

    def __init__(self):
        self._index: dict[int, BM25Okapi] = {}

    def _bm25(self, catalog: Catalog) -> BM25Okapi:
        key = id(catalog)
        if key not in self._index:
            docs = [_tok(f"{t.name} {t.server} {t.description}") for t in catalog.tools]
            self._index[key] = BM25Okapi(docs)
        return self._index[key]

    async def rank(self, state, catalog, intent="next"):
        t0 = time.perf_counter()
        scores = self._bm25(catalog).get_scores(_tok(state))
        order = [catalog.tools[i].name for i in np.argsort(-scores)]
        top = float(scores.max()) if len(scores) else 0.0
        norm = {catalog.tools[i].name: float(s) / top if top > 0 else 0.0 for i, s in enumerate(scores)}
        return Ranking(order=order, scores=norm, latency_s=time.perf_counter() - t0)


class Embedding(Router):
    """Dense retrieval: cosine similarity between the state and each tool's text.

    The usual "tool RAG" answer to large tool sets, so the baseline Jev has to beat.
    """

    def __init__(self, model: str = "openai/text-embedding-3-small", price_per_mtok: float = 0.02):
        self.model = model
        self.name = f"embed:{model.split('/')[-1]}"
        self.price = price_per_mtok
        self.mock = is_mock()
        self._client = None
        if not self.mock:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.environ["AI_GATEWAY_API_KEY"], base_url=GATEWAY_BASE_URL)
        self._mats: dict[str, np.ndarray] = {}

    async def _embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        if self.mock:
            return np.stack([_hash_vec(t) for t in texts]), sum(len(t) // 4 for t in texts)
        vecs, tokens = [], 0
        for i in range(0, len(texts), 128):
            r = await self._client.embeddings.create(model=self.model, input=texts[i:i + 128])
            vecs += [d.embedding for d in r.data]
            tokens += r.usage.total_tokens if r.usage else 0
        m = np.asarray(vecs, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True), tokens

    async def _matrix(self, catalog: Catalog) -> np.ndarray:
        texts = [f"{t.name} ({t.server}): {t.short(500)}" for t in catalog.tools]
        h = hashlib.sha1(("\n".join(texts) + self.model + str(self.mock)).encode()).hexdigest()[:16]
        if h in self._mats:
            return self._mats[h]
        path = CACHE_DIR / f"emb_{h}.npy"
        if path.exists():
            m = np.load(path)
        else:
            m, _ = await self._embed(texts)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(path, m)
        self._mats[h] = m
        return m

    async def rank(self, state, catalog, intent="next"):
        m = await self._matrix(catalog)
        t0 = time.perf_counter()
        q, tokens = await self._embed([state[-8000:]])
        sims = m @ q[0]
        order = [catalog.tools[i].name for i in np.argsort(-sims)]
        return Ranking(order=order, scores={catalog.tools[i].name: float(s) for i, s in enumerate(sims)},
                       latency_s=time.perf_counter() - t0, cost_usd=tokens * self.price / 1e6,
                       router_tokens=tokens, calls=1, mock=self.mock)

    async def aclose(self):
        if self._client:
            await self._client.close()


def _hash_vec(text: str, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in _tok(text):
        v[int(hashlib.md5(w[:6].encode()).hexdigest(), 16) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


class LLMRouter(Router):
    """A chat LLM reads the whole catalog and names the best tools, as JSON.

    This is "ask a small fast LLM to pick", i.e. the obvious alternative to a decision model.
    """

    def __init__(self, model: str = "anthropic/claude-haiku-4.5", top: int = 10):
        self.model = model
        self.name = f"llm:{model.split('/')[-1]}"
        self.top = top
        self.mock = is_mock()
        if not self.mock:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.environ["AI_GATEWAY_API_KEY"], base_url=GATEWAY_BASE_URL)

    async def rank(self, state, catalog, intent="next"):
        if self.mock:
            r = await BM25().rank(state, catalog, intent)
            r.mock = True
            return r
        listing = "\n".join(f"- {t.name} [{t.server}]: {t.short(200)}" for t in catalog.tools)
        ask = NEXT_STEP if intent == "next" else WHOLE_TASK
        prompt = (f"{ask}\nReturn JSON only: {{\"tools\": [up to {self.top} tool names, best first]}}\n\n"
                  f"TOOLS:\n{listing}\n\nINPUT:\n{state}")
        t0 = time.perf_counter()
        raw = await self._client.chat.completions.with_raw_response.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=400, temperature=0)
        latency = time.perf_counter() - t0
        resp = raw.parse()
        body = raw.http_response.json()
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.S)
        names = []
        if m:
            try:
                names = [n for n in json.loads(m.group(0)).get("tools", []) if n in catalog]
            except json.JSONDecodeError:
                pass
        scores = {n: 1.0 / (i + 1) for i, n in enumerate(names)}
        cost = _gateway_cost(body)
        return Ranking(order=_complete(names, catalog), scores=scores, latency_s=latency, cost_usd=cost,
                       router_tokens=resp.usage.prompt_tokens if resp.usage else 0, calls=1,
                       debug={"raw": text[:500]})

    async def aclose(self):
        if not self.mock:
            await self._client.close()


def _gateway_cost(body: dict[str, Any]) -> float:
    meta = (body.get("provider_metadata") or body.get("providerMetadata") or {}).get("gateway", {})
    try:
        return float(meta.get("cost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


# --- the decision-model router (Jev) ------------------------------------------------------

class DecisionRouter(Router):
    """Routes with a decision model through the gateway's /v1/decisions endpoint.

    strategy:
      flat       one `choice` question over every tool (catalogs up to 255 tools)
      two_stage  one `predicate` per server ("will the agent need this server?") in a single
                 request, then one `choice` over the tools of the servers that pass
      prefilter  BM25 shortlist of `prefilter_k` tools, then one `choice` over the shortlist
      auto       flat when the catalog fits in one choice question, otherwise two_stage
    """

    def __init__(self, model: str = "typesafe-ai/jev", strategy: str = "auto", max_servers: int = 3,
                 server_threshold: float = 0.3, prefilter_k: int = 60, desc_chars: int = 200):
        self.model = model
        self.strategy = strategy
        self.max_servers = max_servers
        self.server_threshold = server_threshold
        self.prefilter_k = prefilter_k
        self.desc_chars = desc_chars
        short = model.split("/")[-1]
        self.name = short if strategy == "auto" else f"{short}:{strategy}"
        self.client = DecisionClient(model=model)
        self._bm25 = BM25()

    def _choice(self, catalog: Catalog, names: list[str], intent: str) -> dict[str, Any]:
        return {
            "type": "choice", "name": "tool", "instructions": NEXT_STEP if intent == "next" else WHOLE_TASK,
            "choices": [{"value": n, "description": f"[{catalog.by_name[n].server}] {catalog.by_name[n].short(self.desc_chars)}"}
                        for n in names],
        }

    async def rank(self, state, catalog, intent="next"):
        strategy = self.strategy
        if strategy == "auto":
            strategy = "flat" if len(catalog) <= MAX_CHOICES else "two_stage"
        if strategy == "flat" and len(catalog) > MAX_CHOICES:
            strategy = "two_stage"

        latency = cost = 0.0
        tokens = calls = 0
        debug: dict[str, Any] = {"strategy": strategy}
        mock = False

        if strategy == "flat":
            candidates = [t.name for t in catalog.tools]
        elif strategy == "prefilter":
            r = await self._bm25.rank(state, catalog, intent)
            candidates = r.order[: min(self.prefilter_k, MAX_CHOICES)]
        else:
            servers = list(catalog.servers)
            questions = [{
                "type": "predicate", "name": f"s{i}",
                "instructions": (f"Will the agent need any tool from the '{s}' MCP server "
                                 f"({catalog.server_summary(s)}) to complete this task?"),
            } for i, s in enumerate(servers)]
            d = await self.client.decide(state, questions)
            latency += d.latency_s; cost += d.cost_usd; tokens += d.input_tokens; calls += 1; mock |= d.mock
            p_server = {s: (d.answers.get(f"s{i}").probability or 0.0) if d.answers.get(f"s{i}") else 0.0
                        for i, s in enumerate(servers)}
            ranked = sorted(servers, key=p_server.get, reverse=True)
            chosen = [s for s in ranked if p_server[s] >= self.server_threshold][: self.max_servers] or ranked[:2]
            debug["servers"] = {s: round(p_server[s], 3) for s in ranked[:8]}
            debug["chosen_servers"] = chosen
            candidates = [t.name for s in chosen for t in catalog.servers[s]]
            if len(candidates) > MAX_CHOICES:
                sub = catalog.subset(candidates)
                candidates = (await self._bm25.rank(state, sub, intent)).order[:MAX_CHOICES]

        d = await self.client.decide(state, [self._choice(catalog, candidates, intent)])
        latency += d.latency_s; cost += d.cost_usd; tokens += d.input_tokens; calls += 1; mock |= d.mock
        ans = d.answers.get("tool")
        if ans is None:  # refusal: fall back to candidate order
            scores, confidence = {}, None
            order = candidates
        else:
            scores = {str(k): v for k, v in ans.probabilities.items()}
            if ans.choice is not None and ans.choice not in scores:
                scores[str(ans.choice)] = ans.confidence or 1.0
            order = [n for n, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if n in catalog]
            order += [n for n in candidates if n not in scores]
            confidence = ans.confidence if ans.confidence is not None else (max(scores.values()) if scores else None)
        return Ranking(order=_complete(order, catalog), scores=scores, confidence=confidence, latency_s=latency,
                       cost_usd=cost, router_tokens=tokens, calls=calls, mock=mock, debug=debug)

    async def aclose(self):
        await self.client.aclose()


# --- registry -----------------------------------------------------------------------------

def make_router(spec: str) -> Router:
    """Router from a short spec string, as used by the CLI, proxy profiles and dashboard.

    all | bm25 | embed[:model] | llm[:model] | jev[:strategy] | decision:<model>[:strategy]
    """
    head, _, rest = spec.partition(":")
    if head == "all":
        return AllTools()
    if head == "bm25":
        return BM25()
    if head == "embed":
        return Embedding(rest or "openai/text-embedding-3-small")
    if head == "llm":
        return LLMRouter(rest or "anthropic/claude-haiku-4.5")
    if head == "jev":
        return DecisionRouter("typesafe-ai/jev", strategy=rest or "auto")
    if head == "decision":  # decision:<model slug>[:strategy]
        model, strategy = rest, "auto"
        if rest.rsplit(":", 1)[-1] in ("flat", "two_stage", "prefilter", "auto"):
            model, strategy = rest.rsplit(":", 1)
        return DecisionRouter(model, strategy=strategy)
    raise ValueError(f"unknown router spec {spec!r}")


async def close_all(routers: list[Router]) -> None:
    await asyncio.gather(*(r.aclose() for r in routers), return_exceptions=True)
