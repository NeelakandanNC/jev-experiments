"""Build data/catalogs/<benchmark>.json: every tool each benchmark can expose, with schemas.

Tool names follow each harness's own convention so router output matches what the model sees:
    atlas     server_tool    (Atlas sandbox, fastmcp prefixing)
    mcpbench  Server:tool    (MCP-Bench connector)
    universe  server__tool   (MCP-Universe function-call agent)

    python -m bench.catalogs atlas     --local        (launch the pinned servers, list_tools only)
    python -m bench.catalogs atlas     [--sandbox http://localhost:1984]   (or ask Scale's Docker sandbox)
    python -m bench.catalogs mcpbench  [--info vendor/mcp-bench/mcp_servers_info.json]
    python -m bench.catalogs universe  [--python vendor/MCP-Universe/.venv/bin/python]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import httpx

from jevroute.catalog import Catalog, Tool

from .datasets import DATA, VENDOR, catalog_path, universe_tasks

DATA_DIR = DATA / "catalogs"


def build_atlas(sandbox: str) -> Catalog:
    template = json.loads((VENDOR / "mcp-atlas" / "services" / "agent-environment" / "src" / "agent_environment"
                           / "mcp_server_template.json").read_text())["mcpServers"]
    servers = sorted(template, key=len, reverse=True)
    r = httpx.post(f"{sandbox.rstrip('/')}/list-tools", timeout=600)
    r.raise_for_status()
    tools = []
    for t in r.json():
        server = next((s for s in servers if t["name"].startswith(s + "_")), t["name"].split("_", 1)[0])
        tools.append(Tool(t["name"], server, t.get("description") or "", t.get("inputSchema") or {}))
    return Catalog(tools, name="atlas")


async def build_atlas_local(timeout: float = 240, concurrency: int = 6) -> Catalog:
    """List every Atlas server's tools by launching its pinned package locally (list_tools only,
    no tool is ever called). Same tool set as the sandbox, without Docker. Names get the
    sandbox's `server_` prefix."""
    import re
    import tempfile

    template = json.loads((VENDOR / "mcp-atlas" / "services" / "agent-environment" / "src" / "agent_environment"
                           / "mcp_server_template.json").read_text())["mcpServers"]
    scratch = tempfile.mkdtemp(prefix="atlas-list-")
    sem = asyncio.Semaphore(concurrency)

    def fill(text: str) -> str:
        text = text.replace("/data", scratch)
        return re.sub(r"\$\{[A-Z0-9_]+\}", "dummy", text)

    async def one(name: str, cfg: dict) -> list[Tool]:
        args = [fill(a) for a in cfg.get("args", [])]
        if cfg["command"] == "npx" and "-y" not in args:
            args = ["-y", *args]
        if cfg["command"] == "uvx":  # the sandbox image predates mcp 2.x, which renamed FastMCP
            args = ["--with", "mcp<2", "--with", "pydantic<2.12", *args]
        env = {k: fill(v) if v else "dummy" for k, v in (cfg.get("env") or {}).items()}
        env = {k: (scratch if "DIR" in k or "PATH" in k else v) for k, v in env.items()}
        async with sem:
            try:
                got = await asyncio.wait_for(_list_stdio(name, {"command": cfg["command"], "args": args, "env": env},
                                                         "python3", timeout), timeout + 30)
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {str(e)[:120]}")
                return []
        tools = [Tool(f"{name}_{t.name.split('__', 1)[1]}", name, t.description, t.input_schema) for t in got]
        print(f"  {name}: {len(tools)} tools")
        return tools

    results = await asyncio.gather(*(one(n, c) for n, c in template.items()))
    tools = [t for r in results for t in r]
    for t in tools:  # servers echo their allowed dir; show the sandbox's /data, not the local scratch path
        t.description = t.description.replace(scratch, "/data")
        t.input_schema = json.loads(json.dumps(t.input_schema).replace(scratch, "/data"))
    official = DATA_DIR / "atlas_tool_names_official.txt"  # Scale's published list of the 307 tools
    if official.exists():
        allowed = set(re.findall(r"^- (\S+)", official.read_text(), re.M))
        dropped = [t.name for t in tools if t.name not in allowed]
        tools = [t for t in tools if t.name in allowed]
        print(f"  kept {len(tools)} of {len(allowed)} official tools; dropped {len(dropped)} not in the official list")
    return Catalog(tools, name="atlas")


def build_mcpbench(info: Path) -> Catalog:
    d = json.loads(info.read_text())
    tools = []
    for server, s in d["servers"].items():
        name = s.get("name", server)
        for t in s.get("tools", {}).values():
            tools.append(Tool(f"{name}:{t['name']}", name, t.get("description") or "", t.get("input_schema") or {}))
    return Catalog(tools, name="mcpbench")


async def _list_stdio(name: str, cfg: dict, python: str, timeout: float) -> list[Tool]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    cmd = cfg["command"]
    if cmd in ("python", "python3"):
        cmd = python
    env = {**os.environ, **{k: v or "dummy" for k, v in (cfg.get("env") or {}).items()}}
    params = StdioServerParameters(command=cmd, args=cfg.get("args", []), env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await asyncio.wait_for(session.initialize(), timeout)
            res = await asyncio.wait_for(session.list_tools(), timeout)
    def schema(t):
        return getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}

    return [Tool(f"{name}__{t.name}", name, t.description or "", schema(t)) for t in res.tools]


async def build_universe(python: str, timeout: float = 120) -> Catalog:
    server_list = json.loads((VENDOR / "MCP-Universe" / "mcpuniverse" / "mcp" / "configs" / "server_list.json").read_text())
    needed = sorted({s["name"] for t in universe_tasks() for s in t.get("mcp_servers", [])})
    tools: list[Tool] = []
    for name in needed:
        cfg = server_list[name].get("stdio") or server_list[name]
        try:
            got = await _list_stdio(name, cfg, python, timeout)
            print(f"  {name}: {len(got)} tools")
            tools += got
        except Exception as e:
            print(f"  {name}: FAILED {type(e).__name__}: {e}")
    return Catalog(tools, name="universe")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["atlas", "mcpbench", "universe"])
    ap.add_argument("--sandbox", default="http://localhost:1984")
    ap.add_argument("--local", action="store_true", help="atlas: launch the pinned servers locally instead of the sandbox")
    ap.add_argument("--info", default=str(VENDOR / "mcp-bench" / "mcp_servers_info.json"))
    ap.add_argument("--python", default=str(VENDOR / "MCP-Universe" / ".venv" / "bin" / "python"))
    args = ap.parse_args()
    if args.benchmark == "atlas":
        cat = asyncio.run(build_atlas_local()) if args.local else build_atlas(args.sandbox)
    elif args.benchmark == "mcpbench":
        cat = build_mcpbench(Path(args.info))
    else:
        cat = asyncio.run(build_universe(args.python))
    path = catalog_path(args.benchmark)
    cat.save(path)
    print(f"{args.benchmark}: {len(cat)} tools across {len(cat.servers)} servers, "
          f"{cat.tokens():,} schema tokens -> {path}")


if __name__ == "__main__":
    main()
