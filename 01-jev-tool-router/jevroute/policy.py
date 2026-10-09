"""Per-turn routing policy: conversation -> router state -> the tool subset the agent gets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .catalog import Catalog
from .routers import Ranking, Router, select


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def state_from_messages(messages: list[dict[str, Any]], max_chars: int = 6000,
                        result_chars: int = 300) -> tuple[str, list[str]]:
    """Compress an OpenAI-format conversation into the text the router decides on.

    Keeps the task (first user message) whole, then the most recent steps, newest last.
    Returns (state, tools already called).
    """
    users = [m for m in messages if m.get("role") == "user"]
    task = _text(users[0].get("content")) if users else ""
    steps: list[str] = []
    used: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                used.append(fn.get("name", ""))
                steps.append(f"called {fn.get('name')}({_text(fn.get('arguments'))[:200]})")
        elif role == "tool":
            steps.append(f"  -> {' '.join(_text(m.get('content')).split())[:result_chars]}")
        elif role == "user" and m is not users[0]:
            steps.append(f"user: {_text(m.get('content'))[:500]}")

    state = f"TASK:\n{task}"
    if steps:
        budget = max_chars - len(state) - 40
        tail: list[str] = []
        for s in reversed(steps):
            if budget - len(s) < 0:
                tail.append("…")
                break
            tail.append(s)
            budget -= len(s) + 1
        state += "\n\nPROGRESS SO FAR:\n" + "\n".join(reversed(tail))
    return state, [u for u in dict.fromkeys(used) if u]


@dataclass
class RoutePolicy:
    router: Router
    k: int = 3
    adaptive_mass: float | None = None  # e.g. 0.9 -> fewer tools when the router is confident
    sticky: bool = True  # keep tools the agent already called available
    always: list[str] = field(default_factory=list)

    async def choose(self, messages: list[dict[str, Any]], catalog: Catalog) -> tuple[list[str], Ranking, str]:
        state, used = state_from_messages(messages)
        ranking = await self.router.rank(state, catalog, intent="next")
        picked = select(ranking, self.k, self.adaptive_mass)
        extra = (used if self.sticky else []) + self.always
        chosen = list(dict.fromkeys(picked + [n for n in extra if n in catalog]))
        return chosen, ranking, state


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))
