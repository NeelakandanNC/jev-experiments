"""Tool catalog: every tool an agent could be shown, grouped by MCP server."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import tiktoken


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


@dataclass
class Tool:
    name: str  # name exactly as the agent sees it, e.g. "github_search_code"
    server: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    @classmethod
    def from_openai(cls, d: dict[str, Any], server: str = "default") -> "Tool":
        fn = d.get("function", d)
        return cls(
            name=fn["name"],
            server=server,
            description=fn.get("description") or "",
            input_schema=fn.get("parameters") or {},
        )

    @property
    def schema_tokens(self) -> int:
        # What this tool costs in the agent's context when its definition is loaded.
        return count_tokens(json.dumps(self.to_openai(), separators=(",", ":")))

    def short(self, limit: int = 240) -> str:
        desc = " ".join(self.description.split())
        return desc if len(desc) <= limit else desc[: limit - 1] + "…"


class Catalog:
    def __init__(self, tools: Iterable[Tool], name: str = "catalog"):
        self.name = name
        self.tools: list[Tool] = list(tools)
        self.by_name: dict[str, Tool] = {t.name: t for t in self.tools}
        self.servers: dict[str, list[Tool]] = {}
        for t in self.tools:
            self.servers.setdefault(t.server, []).append(t)
        self._token_cache: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, name: str) -> bool:
        return name in self.by_name

    def subset(self, names: Iterable[str], name: str | None = None) -> "Catalog":
        keep = [self.by_name[n] for n in names if n in self.by_name]
        return Catalog(keep, name=name or f"{self.name}[subset]")

    def tokens(self, names: Iterable[str] | None = None) -> int:
        names = list(self.by_name) if names is None else names
        total = 0
        for n in names:
            if n not in self._token_cache:
                t = self.by_name.get(n)
                self._token_cache[n] = t.schema_tokens if t else 0
            total += self._token_cache[n]
        return total

    def server_summary(self, server: str, limit: int = 400) -> str:
        tools = self.servers[server]
        names = ", ".join(_strip_prefix(t.name, server) for t in tools)
        text = f"{len(tools)} tools: {names}"
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tools": [
                {"name": t.name, "server": t.server, "description": t.description, "input_schema": t.input_schema}
                for t in self.tools
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        d = json.loads(Path(path).read_text())
        return cls((Tool(**t) for t in d["tools"]), name=d.get("name", Path(path).stem))

    @classmethod
    def from_openai_tools(cls, tools: list[dict[str, Any]], known: "Catalog | None" = None) -> "Catalog":
        """Build a catalog from a chat-completions `tools` array.

        Server membership comes from `known` when the tool is in it, otherwise from the
        `server_tool` naming convention used by MCP-Atlas and most MCP clients.
        """
        out = []
        known_servers = sorted(known.servers, key=len, reverse=True) if known else []
        for d in tools:
            fn = d.get("function", d)
            name = fn["name"]
            if known and name in known.by_name:
                server = known.by_name[name].server
            else:
                server = next((s for s in known_servers if name.startswith(s + "_")), None) or _guess_server(name)
            out.append(Tool.from_openai(d, server=server))
        return cls(out, name="request")


def _strip_prefix(name: str, server: str) -> str:
    for sep in ("_", "__", ":", "."):
        if name.startswith(server + sep):
            return name[len(server) + len(sep):]
    return name


def _guess_server(name: str) -> str:
    for sep in ("__", ":", "."):
        if sep in name:
            return name.split(sep, 1)[0]
    return name.split("_", 1)[0] if "_" in name else "default"
