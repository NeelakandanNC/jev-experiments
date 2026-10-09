"""OpenAI-compatible chat-completions proxy that routes tools before the model sees them.

Point any agent harness that speaks chat completions at this proxy. The routing profile
rides in the model name, so one proxy serves every experimental condition at once:

    model = "[<run tag>|]<profile>::<upstream model>"
    profile = full | <router spec>@k<k>[~<mass>]
    e.g.  full::anthropic/claude-sonnet-4.5          every tool, passed through untouched
          jev@k3::anthropic/claude-sonnet-4.5        Jev picks 3 tools per turn
          jev@k5~0.9::anthropic/claude-sonnet-4.5    Jev picks until 90% probability mass, max 5
          bm25@k5::anthropic/claude-sonnet-4.5       lexical baseline

Requests are forwarded to AI Gateway (UPSTREAM_BASE_URL) and every call is appended to a
JSONL log with tool tokens before/after routing, router latency/cost, and model usage.

    python -m jevroute.proxy --port 8787 [--catalog data/catalogs/atlas.json]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .catalog import Catalog
from .policy import RoutePolicy, dumps
from .routers import Router, make_router

ROOT = Path(__file__).resolve().parent.parent
PROFILE_RE = re.compile(r"^(?P<spec>[^@]+)@k(?P<k>\d+)(?:~(?P<mass>0?\.\d+|1(?:\.0)?))?$")


def parse_model(model: str) -> tuple[str, str, str]:
    """"[tag|]profile::upstream" -> (tag, profile, upstream). Bare names like "gpt-4.1" (what benchmark
    judges send) get the gateway's provider prefix."""
    tag, profile, upstream = "", "full", model
    if "::" in model:
        profile, upstream = model.split("::", 1)
    if "|" in profile:
        tag, profile = profile.split("|", 1)
    if "/" not in upstream:
        upstream = f"{os.getenv('JEVROUTE_BARE_PREFIX', 'openai')}/{upstream}"
    return tag, profile, upstream


class Proxy:
    def __init__(self, upstream_base: str, upstream_key: str, log_path: Path, known: Catalog | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.upstream_base = upstream_base.rstrip("/")
        self.upstream_key = upstream_key
        self.log_path = log_path
        self.known = known
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30), transport=transport)
        self.routers: dict[str, Router] = {}
        self.catalogs: dict[str, Catalog] = {}
        self.lock = asyncio.Lock()
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def policy(self, profile: str) -> RoutePolicy | None:
        if profile == "full":
            return None
        m = PROFILE_RE.match(profile)
        if not m:
            raise ValueError(f"bad routing profile {profile!r}; expected e.g. jev@k3 or bm25@k5~0.9")
        spec = m["spec"]
        if spec not in self.routers:
            self.routers[spec] = make_router(spec)
        return RoutePolicy(self.routers[spec], k=int(m["k"]), adaptive_mass=float(m["mass"]) if m["mass"] else None)

    def catalog(self, tools: list[dict[str, Any]]) -> Catalog:
        key = hashlib.sha1("|".join(sorted(t.get("function", t)["name"] for t in tools)).encode()).hexdigest()
        if key not in self.catalogs:
            self.catalogs[key] = Catalog.from_openai_tools(tools, known=self.known)
        return self.catalogs[key]

    async def log(self, rec: dict[str, Any]) -> None:
        async with self.lock:
            with self.log_path.open("a") as f:
                f.write(dumps(rec) + "\n")

    async def chat(self, body: dict[str, Any], headers: dict[str, str]) -> Response:
        tag, profile, upstream_model = parse_model(body.get("model", ""))
        body = {**body, "model": upstream_model}
        tools = body.get("tools") or []
        rec: dict[str, Any] = {"ts": time.time(), "run": tag or headers.get("x-jevroute-run", ""), "profile": profile,
                               "model": upstream_model, "n_tools_in": len(tools), "turn": _turn(body)}
        try:
            policy = self.policy(profile)
        except ValueError as e:
            return JSONResponse({"error": {"message": str(e), "type": "invalid_request_error"}}, status_code=400)

        if tools:
            cat = self.catalog(tools)
            rec["tool_tokens_in"] = cat.tokens()
            if policy is not None:
                chosen, ranking, _ = await policy.choose(body.get("messages", []), cat)
                by_name = {t.get("function", t)["name"]: t for t in tools}
                body["tools"] = [by_name[n] for n in chosen if n in by_name]  # best first
                if not body["tools"]:
                    body.pop("tools")
                    body.pop("tool_choice", None)
                rec.update(tools_out=chosen, router=policy.router.name, router_latency_s=ranking.latency_s,
                           router_cost_usd=ranking.cost_usd, router_tokens=ranking.router_tokens,
                           router_calls=ranking.calls, router_confidence=ranking.confidence, mock=ranking.mock,
                           router_debug=ranking.debug)
            rec["n_tools_out"] = len(body.get("tools") or [])
            rec["tool_tokens_out"] = cat.tokens(t.get("function", t)["name"] for t in body.get("tools") or [])

        url = f"{self.upstream_base}/chat/completions"
        auth = {"Authorization": f"Bearer {self.upstream_key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        if body.get("stream"):
            return await self._stream(url, auth, body, rec, t0)
        r = await self.http.post(url, headers=auth, json=body)
        rec["upstream_latency_s"] = time.perf_counter() - t0
        rec["status"] = r.status_code
        try:
            out = r.json()
        except json.JSONDecodeError:
            await self.log(rec)
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
        usage = out.get("usage") or {}
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        rec["upstream_cost_usd"] = _cost(out)
        calls = [tc.get("function", {}).get("name") for c in out.get("choices", [])
                 for tc in (c.get("message") or {}).get("tool_calls") or []]
        rec["called"] = calls
        offered = {t.get("function", t)["name"] for t in body.get("tools") or []}
        rec["called_unoffered"] = [c for c in calls if c not in offered]
        if r.status_code >= 400:
            rec["error"] = str(out)[:500]
        await self.log(rec)
        return JSONResponse(out, status_code=r.status_code)

    async def _stream(self, url, auth, body, rec, t0) -> StreamingResponse:
        req = self.http.build_request("POST", url, headers=auth, json=body)
        r = await self.http.send(req, stream=True)

        async def gen():
            try:
                async for chunk in r.aiter_raw():
                    yield chunk
            finally:
                await r.aclose()
                rec["upstream_latency_s"] = time.perf_counter() - t0
                rec["status"] = r.status_code
                rec["streamed"] = True
                await self.log(rec)

        return StreamingResponse(gen(), status_code=r.status_code, media_type=r.headers.get("content-type"))

    async def passthrough(self, method: str, path: str, body: bytes) -> Response:
        r = await self.http.request(method, f"{self.upstream_base}/{path}", content=body or None,
                                    headers={"Authorization": f"Bearer {self.upstream_key}",
                                             "Content-Type": "application/json"})
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    async def aclose(self):
        await self.http.aclose()
        await asyncio.gather(*(r.aclose() for r in self.routers.values()), return_exceptions=True)


def _turn(body: dict[str, Any]) -> int:
    return sum(1 for m in body.get("messages", []) if m.get("role") == "assistant")


def _cost(body: dict[str, Any]) -> float | None:
    meta = (body.get("provider_metadata") or body.get("providerMetadata") or {}).get("gateway") or {}
    try:
        return float(meta["cost"]) if "cost" in meta else None
    except (TypeError, ValueError):
        return None


def create_app(proxy: Proxy) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        yield
        await proxy.aclose()

    app = FastAPI(title="jevroute proxy", lifespan=lifespan)
    app.state.proxy = proxy

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        return await proxy.chat(body, {k.lower(): v for k, v in request.headers.items()})

    @app.get("/health")
    async def health():
        return {"ok": True, "log": str(proxy.log_path), "upstream": proxy.upstream_base}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def other(path: str, request: Request):
        return await proxy.passthrough(request.method, path, await request.body())

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.getenv("JEVROUTE_PROXY_PORT", 8787)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--catalog", help="known catalog JSON, used to map tool names to servers")
    ap.add_argument("--log", default=str(ROOT / "results" / "proxy" / f"calls-{time.strftime('%Y%m%d')}.jsonl"))
    args = ap.parse_args()

    key = os.getenv("UPSTREAM_API_KEY") or os.getenv("AI_GATEWAY_API_KEY")
    if not key:
        raise SystemExit("set AI_GATEWAY_API_KEY")
    known = Catalog.load(args.catalog) if args.catalog else None
    proxy = Proxy(os.getenv("UPSTREAM_BASE_URL", "https://ai-gateway.vercel.sh/v1"), key, Path(args.log), known)
    import uvicorn
    uvicorn.run(create_app(proxy), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
