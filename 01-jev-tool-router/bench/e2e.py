"""End-to-end eval: run each benchmark's own harness + scorer, with and without routing.

    python -m bench.e2e atlas    --model anthropic/claude-sonnet-4.5 --profiles full,jev@k5 --num-tasks 20
    python -m bench.e2e mcpbench --model anthropic/claude-sonnet-4.5 --profiles full,jev@k8 --num-tasks 20
    python -m bench.e2e universe --model anthropic/claude-sonnet-4.5 --profiles full,jev@k5 --domains financial_analysis

Agent traffic for atlas/universe goes through the jevroute proxy (started here, one log per run);
mcpbench routes inside its planner via bench/mcpbench_entry.py. Each condition writes to
results/e2e/<benchmark>/<run>/<profile>/ and the run's summary.json compares them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from .datasets import ROOT, VENDOR, atlas_tasks, catalog_path, mcpbench_tasks, universe_tasks

RESULTS = ROOT / "results" / "e2e"
GATEWAY = "https://ai-gateway.vercel.sh"
PY = sys.executable


def load_env() -> dict[str, str]:
    """Process env plus 01-jev-tool-router/.env (gitignored), which holds the run's keys."""
    env = dict(os.environ)
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return env


def log(msg: str, f=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if f:
        f.write(line + "\n")
        f.flush()


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_port(port: int, seconds: float, what: str):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if port_open(port):
            return
        time.sleep(1)
    raise RuntimeError(f"{what} did not come up on port {port} within {seconds:.0f}s")


def run(cmd: list[str], cwd: Path, env: dict, logf, check: bool = True) -> int:
    log(f"$ {' '.join(cmd)}  (cwd={cwd.relative_to(ROOT) if cwd.is_relative_to(ROOT) else cwd})", logf)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        logf.write(line)
        logf.flush()
        sys.stdout.write(line)
    p.wait()
    if check and p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd[:3])} ...")
    return p.returncode


class Background:
    """Child processes (proxy, harness) that live for the duration of the run."""

    def __init__(self):
        self.procs: list[subprocess.Popen] = []

    def start(self, cmd, cwd, env, logpath: Path):
        f = logpath.open("a")
        self.procs.append(subprocess.Popen(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT))

    def stop(self):
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


def start_proxy(bg: Background, env: dict, out: Path, port: int, catalog: Path | None) -> str:
    if port_open(port):
        raise RuntimeError(f"port {port} is busy; stop the other proxy or pass --proxy-port")
    cmd = [PY, "-m", "jevroute.proxy", "--port", str(port), "--log", str(out / "proxy.jsonl")]
    if catalog and catalog.exists():
        cmd += ["--catalog", str(catalog)]
    bg.start(cmd, ROOT, env, out / "proxy.log")
    wait_port(port, 30, "jevroute proxy")
    return f"http://127.0.0.1:{port}"


def proxy_stats(path: Path, tag: str) -> dict[str, Any]:
    recs = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    recs = [r for r in recs if r.get("run") == tag]
    if not recs:
        return {"llm_calls": 0}

    def tot(k):
        return sum(r.get(k) or 0 for r in recs)

    return {
        "llm_calls": len(recs),
        "prompt_tokens": tot("prompt_tokens"),
        "completion_tokens": tot("completion_tokens"),
        "tool_tokens_offered": tot("tool_tokens_out"),
        "tool_tokens_full": tot("tool_tokens_in"),
        "mean_tools_offered": tot("n_tools_out") / len(recs),
        "mean_tools_available": tot("n_tools_in") / len(recs),
        "router_cost_usd": tot("router_cost_usd"),
        "router_latency_s": tot("router_latency_s"),
        "model_cost_usd": tot("upstream_cost_usd"),
        "model_latency_s": tot("upstream_latency_s"),
        "errors": sum(1 for r in recs if (r.get("status") or 200) >= 400),
        "called_unoffered": tot_list(recs, "called_unoffered"),
        "mock": any(r.get("mock") for r in recs),
    }


def tot_list(recs, k):
    return sum(len(r.get(k) or []) for r in recs)


def tag_for(run_id: str, profile: str) -> str:
    return f"{run_id}/{profile}"


# --- MCP-Atlas ----------------------------------------------------------------------------

def e2e_atlas(args, env, out: Path, logf) -> dict:
    atlas = VENDOR / "mcp-atlas"
    sandbox = env.get("MCP_SANDBOX_URL", "http://localhost:1984")
    try:
        enabled = httpx.get(f"{sandbox}/enabled-servers", timeout=10).json()
    except httpx.HTTPError:
        raise RuntimeError(f"MCP-Atlas sandbox not reachable at {sandbox}. Start it: cd vendor/mcp-atlas && "
                           "docker run --rm -p 1984:1984 --env-file .env ghcr.io/scaleapi/mcp-atlas:1.2.7")
    log(f"sandbox servers: {json.dumps(enabled)[:300]}", logf)

    rows = atlas_tasks()
    rng = random.Random(args.seed)
    rows = rng.sample(rows, min(args.num_tasks, len(rows))) if args.num_tasks else rows
    all_tools = None
    if args.tools == "all":
        cat = json.loads(catalog_path("atlas").read_text())
        all_tools = [t["name"] for t in cat["tools"]]
    tasks_csv = out / "tasks.csv"
    with tasks_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["TASK", "PROMPT", "ENABLED_TOOLS", "GTFA_CLAIMS"])
        w.writeheader()
        for r in rows:
            w.writerow({"TASK": r["TASK"], "PROMPT": r["PROMPT"], "GTFA_CLAIMS": r["GTFA_CLAIMS"],
                        "ENABLED_TOOLS": json.dumps(all_tools) if all_tools else r["ENABLED_TOOLS"]})
    log(f"{len(rows)} tasks, tool menu = {args.tools} ({len(all_tools) if all_tools else 'curated per task'})", logf)

    bg = Background()
    try:
        proxy = start_proxy(bg, env, out, args.proxy_port, catalog_path("atlas"))
        harness_env = {**env, "LLM_BASE_URL": proxy, "LLM_API_KEY": "via-jevroute-proxy", "PORT": "3001",
                       "MCP_SANDBOX_URL": sandbox}
        if not port_open(3001):
            bg.start(["npm", "run", "dev"], atlas / "services" / "agent-harness", harness_env, out / "harness.log")
            wait_port(3001, 120, "MCP-Atlas agent harness")
        else:
            log("WARNING: something already listens on 3001; assuming it is a harness pointed at this proxy", logf)
        apy = str(atlas / ".venv" / "bin" / "python")
        conditions = {}
        for profile in args.profiles:
            d = out / profile.replace("/", "_")
            d.mkdir(exist_ok=True)
            tag = tag_for(out.name, profile)
            run([apy, "run_eval.py", "--input", str(tasks_csv), "--model", f"{tag}|{profile}::{args.model}",
                 "--output", str(d / "outputs.csv"), "--concurrency", str(args.concurrency),
                 "--skip-health-check"], atlas, {**env, "HARNESS_URL": "http://localhost:3001"}, logf)
            score_env = {**env, "EVAL_LLM_BASE_URL": GATEWAY, "EVAL_LLM_API_KEY": env["AI_GATEWAY_API_KEY"],
                         "EVAL_LLM_MODEL": args.judge or "google/gemini-3.1-pro-preview"}
            run([apy, "services/scoring/score_claims.py", "--groundtruth-file", str(tasks_csv),
                 "--model-file", str(d / "outputs.csv"), "--model-name", profile.replace("/", "_"),
                 "--output-dir", str(d / "score")], atlas, score_env, logf)
            stats = sorted((d / "score").glob("coverage_stats_*.json"))
            score = json.loads(stats[-1].read_text()) if stats else {}
            conditions[profile] = {"score": score, **proxy_stats(out / "proxy.jsonl", tag)}
            log(f"== {profile}: {json.dumps(score)[:300]}", logf)
        return conditions
    finally:
        bg.stop()


# --- MCP-Bench ----------------------------------------------------------------------------

def e2e_mcpbench(args, env, out: Path, logf) -> dict:
    mb = VENDOR / "mcp-bench"
    tasks = mcpbench_tasks()
    if args.split:
        tasks = [t for t in tasks if t["split"] in args.split.split(",")]
    rng = random.Random(args.seed)
    picked = {t["task_id"] for t in (rng.sample(tasks, min(args.num_tasks, len(tasks))) if args.num_tasks else tasks)}
    # rewrite the runner-format task files keeping only the sampled tasks
    files = []
    for f in ("single", "multi_2server", "multi_3server"):
        d = json.loads((mb / "tasks" / f"mcpbench_tasks_{f}_runner_format.json").read_text())
        groups = []
        for g in d["server_tasks"]:
            keep = [t for t in g["tasks"] if t["task_id"] in picked]
            if keep:
                groups.append({**g, "tasks": keep})
        if groups:
            p = out / f"tasks_{f}.json"
            p.write_text(json.dumps({**d, "server_tasks": groups, "total_tasks": sum(len(g["tasks"]) for g in groups)}))
            files.append(str(p))
    log(f"{len(picked)} tasks; distraction servers = {args.distraction_count}", logf)
    mpy = str(mb / ".venv" / "bin" / "python")
    conditions = {}
    for profile in args.profiles:
        d = out / profile.replace("/", "_")
        d.mkdir(exist_ok=True)
        penv = {**env, "JEVROUTE_PROFILE": profile, "JEVROUTE_LOG": str(d / "routing.jsonl"),
                "PYTHONPATH": f"{ROOT}:{mb}"}
        if args.judge:
            penv["JEVROUTE_JUDGE"] = args.judge
        run([mpy, str(ROOT / "bench" / "mcpbench_entry.py"), "--models", args.model, "--tasks-file", ",".join(files),
             "--distraction-count", str(args.distraction_count), "--output", str(d / "results.json")], mb, penv, logf)
        res = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else {}
        conditions[profile] = {"score": _mcpbench_scores(res), **_routing_log_stats(d / "routing.jsonl")}
        log(f"== {profile}: {json.dumps(conditions[profile]['score'])[:300]}", logf)
    return conditions


def _mcpbench_scores(res: dict) -> dict:
    """Pull the headline numbers out of MCP-Bench's results JSON, whatever level they sit at."""
    found: dict[str, Any] = {}

    def walk(x, depth=0):
        if depth > 6:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, (int, float)) and any(s in k for s in (
                        "score", "rate", "compliance", "tokens", "accuracy", "fulfillment", "grounding")):
                    found.setdefault(k, v)
                else:
                    walk(v, depth + 1)

    walk(res.get("summary") or res.get("aggregated_metrics") or res)
    return found


def _routing_log_stats(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    if not recs:
        return {"planning_rounds_routed": 0}
    return {"planning_rounds_routed": len(recs),
            "mean_tools_offered": sum(r["n_tools_out"] for r in recs) / len(recs),
            "mean_tools_available": sum(r["n_tools_in"] for r in recs) / len(recs),
            "tool_tokens_offered": sum(r["tool_tokens_out"] for r in recs),
            "tool_tokens_full": sum(r["tool_tokens_in"] for r in recs),
            "router_cost_usd": sum(r["router_cost_usd"] for r in recs),
            "router_latency_s": sum(r["router_latency_s"] for r in recs),
            "mock": any(r.get("mock") for r in recs)}


# --- MCP-Universe -------------------------------------------------------------------------

def e2e_universe(args, env, out: Path, logf) -> dict:
    uni = VENDOR / "MCP-Universe"
    cfg_root = uni / "mcpuniverse" / "benchmark" / "configs"
    tasks = universe_tasks()
    if args.domains:
        tasks = [t for t in tasks if t["domain"] in args.domains.split(",")]
    rng = random.Random(args.seed)
    tasks = rng.sample(tasks, min(args.num_tasks, len(tasks))) if args.num_tasks else tasks
    pool = sorted({s["name"] for t in tasks for s in t.get("mcp_servers", [])} | set(filter(None, args.pool.split(","))))
    log(f"{len(tasks)} tasks; every task's agent gets the full server pool: {pool}", logf)

    rel = f"jevroute/{out.name}"
    tdir = cfg_root / rel / "tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    names = []
    for t in tasks:
        src = cfg_root / "mcpuniverse" / f"{t['task_id']}.json"
        d = json.loads(src.read_text())
        d["use_specified_server"] = False  # the agent sees the pool, not just the task's servers
        p = tdir / f"{t['task_id'].replace('/', '__')}.json"
        p.write_text(json.dumps(d, indent=1))
        names.append(f"{rel}/tasks/{p.name}")

    bg = Background()
    try:
        proxy = start_proxy(bg, env, out, args.proxy_port, catalog_path("universe"))
        upy = str(uni / ".venv" / "bin" / "python")
        conditions = {}
        for profile in args.profiles:
            d = out / profile.replace("/", "_")
            d.mkdir(exist_ok=True)
            tag = tag_for(out.name, profile)
            yaml_rel = f"{rel}/{profile.replace('/', '_').replace('@', '_').replace('~', '_')}.yaml"
            (cfg_root / yaml_rel).write_text(_universe_yaml(f"{tag}|{profile}::{args.model}", f"{proxy}/v1", pool,
                                                             names, args.max_iterations))
            uenv = {**env, "OPENAI_BASE_URL": f"{proxy}/v1", "OPENAI_API_KEY": "via-jevroute-proxy"}
            run([upy, str(ROOT / "bench" / "universe_entry.py"), yaml_rel, str(d / "results.json")], uni, uenv, logf)
            res = json.loads((d / "results.json").read_text()) if (d / "results.json").exists() else {"tasks": {}}
            tr = res["tasks"]
            score = {"tasks": len(tr), "task_success_rate": sum(t["passed"] for t in tr.values()) / max(len(tr), 1),
                     "eval_pass_rate": sum(t["evals_passed"] for t in tr.values()) / max(
                         sum(t["evals_total"] for t in tr.values()), 1)}
            conditions[profile] = {"score": score, **proxy_stats(out / "proxy.jsonl", tag)}
            log(f"== {profile}: {json.dumps(score)}", logf)
        return conditions
    finally:
        bg.stop()


def _universe_yaml(model: str, base_url: str, servers: list[str], tasks: list[str], max_iter: int) -> str:
    server_lines = "\n".join(f"      - name: {s}" for s in servers)
    task_lines = "\n".join(f"    - {t}" for t in tasks)
    return f"""kind: llm
spec:
  name: llm-1
  type: openai
  config:
    model_name: "{model}"
    base_url: "{base_url}"
    timeout: 600
    max_completion_tokens: 8000

---
kind: agent
spec:
  name: fc-agent
  type: function_call
  config:
    llm: llm-1
    instruction: You are an agent with access to many MCP tools. Use the tools you need to complete the task.
    max_iterations: {max_iter}
    servers:
{server_lines}

---
kind: benchmark
spec:
  description: jevroute end-to-end run
  agent: fc-agent
  tasks:
{task_lines}
"""


# --- driver -------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["atlas", "mcpbench", "universe"])
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.5", help="agent model (AI Gateway slug)")
    ap.add_argument("--profiles", default="full,jev@k5", help="comma-separated: full, jev@k5, bm25@k5, jev@k8~0.9 ...")
    ap.add_argument("--num-tasks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--judge", help="override the benchmark's judge model (gateway slug)")
    ap.add_argument("--proxy-port", type=int, default=8787)
    ap.add_argument("--tools", choices=["all", "curated"], default="all", help="atlas: tool menu per task")
    ap.add_argument("--distraction-count", type=int, default=27, help="mcpbench: extra servers (27 = all)")
    ap.add_argument("--split", help="mcpbench: single,multi_2server,multi_3server")
    ap.add_argument("--domains", help="universe: e.g. financial_analysis,location_navigation")
    ap.add_argument("--pool", default="", help="universe: extra servers given to every task")
    ap.add_argument("--max-iterations", type=int, default=20, help="universe agent step limit")
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)
    args.profiles = [p for p in args.profiles.split(",") if p]

    env = load_env()
    if not env.get("AI_GATEWAY_API_KEY") and not env.get("JEVROUTE_MOCK"):
        raise SystemExit("AI_GATEWAY_API_KEY missing (env or 01-jev-tool-router/.env)")
    run_id = args.run_id or time.strftime(f"{args.benchmark}-%Y%m%d-%H%M%S")
    out = RESULTS / args.benchmark / run_id
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items()}
    config.update(run_id=run_id, started=time.time())
    (out / "config.json").write_text(json.dumps(config, indent=1))
    with (out / "log.txt").open("a") as logf:
        log(f"e2e {args.benchmark} run {run_id}: model={args.model} profiles={args.profiles}", logf)
        fn = {"atlas": e2e_atlas, "mcpbench": e2e_mcpbench, "universe": e2e_universe}[args.benchmark]
        try:
            conditions = fn(args, env, out, logf)
        except Exception as e:
            log(f"FAILED: {type(e).__name__}: {e}", logf)
            (out / "summary.json").write_text(json.dumps({**config, "error": str(e)}, indent=1))
            raise SystemExit(1)
        summary = {**config, "finished": time.time(), "conditions": conditions}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"wrote {out / 'summary.json'}", logf)
    if args.benchmark == "universe":
        shutil.rmtree(VENDOR / "MCP-Universe" / "mcpuniverse" / "benchmark" / "configs" / "jevroute" / run_id,
                      ignore_errors=True)


if __name__ == "__main__":
    main()
