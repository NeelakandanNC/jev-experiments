"""Benchmarks as routing cases: (state the router sees, gold tools it should surface)."""

from __future__ import annotations

import glob
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from jevroute.catalog import Catalog
from jevroute.policy import state_from_messages

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VENDOR = ROOT / "vendor"
BENCHMARKS = ("atlas", "mcpbench", "universe")


@dataclass
class Case:
    id: str
    task_id: str
    state: str
    gold: list[str]
    mode: str  # step (gold = tool called next) | task (gold = tools the task needs) | server (gold = servers)
    meta: dict[str, Any] = field(default_factory=dict)


def catalog_path(benchmark: str) -> Path:
    return DATA / "catalogs" / f"{benchmark}.json"


def load_catalog(benchmark: str) -> Catalog:
    p = catalog_path(benchmark)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: build it with `python -m bench.catalogs {benchmark}`")
    return Catalog.load(p)


def load_cases(benchmark: str, catalog: Catalog, limit_tasks: int | None = None, seed: int = 0) -> tuple[list[Case], dict]:
    fn = {"atlas": _atlas, "mcpbench": _mcpbench, "universe": _universe}[benchmark]
    cases, stats = fn(catalog)
    task_ids = sorted({c.task_id for c in cases})
    if limit_tasks and limit_tasks < len(task_ids):
        keep = set(random.Random(seed).sample(task_ids, limit_tasks))
        cases = [c for c in cases if c.task_id in keep]
    stats.update(tasks=len({c.task_id for c in cases}), cases=len(cases))
    return cases, stats


# --- MCP-Atlas ----------------------------------------------------------------------------

def atlas_tasks() -> list[dict[str, Any]]:
    """The 500 public tasks, cached from the HuggingFace datasets-server API."""
    path = DATA / "tasks" / "atlas.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows, offset = [], 0
    with httpx.Client(timeout=60) as http:
        while True:
            r = http.get("https://datasets-server.huggingface.co/rows",
                         params={"dataset": "ScaleAI/MCP-Atlas", "config": "default", "split": "train",
                                 "offset": offset, "length": 100})
            r.raise_for_status()
            batch = [x["row"] for x in r.json()["rows"]]
            rows += batch
            offset += len(batch)
            if len(batch) < 100:
                break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


def _json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def _atlas(catalog: Catalog) -> tuple[list[Case], dict]:
    cases, missing, steps = [], 0, 0
    for row in atlas_tasks():
        traj = _json(row["TRAJECTORY"]) or []
        messages: list[dict[str, Any]] = [{"role": "user", "content": row["PROMPT"]}]
        step = 0
        for msg in traj:
            calls = msg.get("tool_calls") if msg.get("role") == "assistant" else None
            if calls:
                gold = list(dict.fromkeys(tc["function"]["name"] for tc in calls))
                steps += 1
                if all(g in catalog for g in gold):
                    state, _ = state_from_messages(messages)
                    cases.append(Case(f"{row['TASK']}#{step}", row["TASK"], state, gold, "step",
                                      {"enabled_tools": _json(row["ENABLED_TOOLS"])}))
                else:
                    missing += 1
                step += 1
            messages.append(msg)
    return cases, {"benchmark": "atlas", "gold_steps": steps, "skipped_gold_not_in_catalog": missing}


# --- MCP-Bench ----------------------------------------------------------------------------

def mcpbench_tasks() -> list[dict[str, Any]]:
    out = []
    for f in ("single", "multi_2server", "multi_3server"):
        d = json.loads((VENDOR / "mcp-bench" / "tasks" / f"mcpbench_tasks_{f}_runner_format.json").read_text())
        for group in d["server_tasks"]:
            servers = group.get("servers") or [group["server_name"]]
            for t in group["tasks"]:
                out.append({**t, "servers": servers, "split": f})
    return out


def _mcpbench(catalog: Catalog) -> tuple[list[Case], dict]:
    cases, no_gold = [], 0
    for t in mcpbench_tasks():
        text = f"{t.get('dependency_analysis', '')}\n{t.get('task_description', '')}"
        gold = []
        for server in t["servers"]:
            for tool in catalog.servers.get(server, []):
                bare = tool.name.split(":", 1)[-1]
                generic = "_" not in bare and len(bare) < 8  # "search", "fetch", "think" also occur as prose
                pattern = rf"[`'\"]{re.escape(bare)}[`'\"(]" if generic else rf"(?<![\w-]){re.escape(bare)}(?![\w-])"
                if f"{server}:{bare}" in text or re.search(pattern, text):
                    gold.append(tool.name)
        if not gold:
            no_gold += 1
            continue
        cases.append(Case(t["task_id"], t["task_id"], t["fuzzy_description"], gold, "task",
                          {"servers": t["servers"], "split": t["split"]}))
    return cases, {"benchmark": "mcpbench", "skipped_no_gold": no_gold}


# --- MCP-Universe -------------------------------------------------------------------------

def universe_tasks() -> list[dict[str, Any]]:
    base = VENDOR / "MCP-Universe" / "mcpuniverse" / "benchmark" / "configs" / "mcpuniverse"
    out = []
    for f in sorted(glob.glob(str(base / "*" / "*.json"))):
        d = json.loads(Path(f).read_text())
        d["task_id"] = f"{Path(f).parent.name}/{Path(f).stem}"
        d["domain"] = Path(f).parent.name
        out.append(d)
    return out


def _universe(catalog: Catalog) -> tuple[list[Case], dict]:
    cases, missing = [], 0
    for t in universe_tasks():
        servers = [s["name"] for s in t.get("mcp_servers", [])]
        if not servers or any(s not in catalog.servers for s in servers):
            missing += 1
            continue
        state = t["question"]
        if t.get("output_format"):
            state += f"\n\nOutput format: {json.dumps(t['output_format'])}"
        cases.append(Case(t["task_id"], t["task_id"], state, servers, "server", {"domain": t["domain"]}))
    return cases, {"benchmark": "universe", "skipped_server_not_in_catalog": missing}
