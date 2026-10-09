"""End-to-end MiniWoB++: does the agent finish the task?

    python -m bench.miniwob --variants yolo+jev,yolo+luna,yolo+llm,yolo+som,dom+jev --episodes 10

Variant = <detector>+<selector>:
  yolo+jev   our YOLO + OCR, Jev picks the element           (the system under test)
  yolo+luna  same, OpenAI's decision model gpt-6-luna picks
  yolo+llm   same, the chat LLM (gpt-6-luna) picks from the list (no decision model)
  yolo+som   the planner sees numbered boxes and picks the id itself (one LLM does everything)
  dom+jev    the page's DOM instead of YOLO (oracle detector: how much do detector misses cost?)
The planner (default gpt-6-luna) is the same in every variant. Seeds 0..episodes-1 per task, the
same seeds for every variant. Results append to results/miniwob/<run>/episodes.jsonl; rerunning
with --run <name> resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict

from screenjev.agent import Agent, AgentConfig
from screenjev.decide import is_mock
from screenjev.detect import Perception, make_detector
from screenjev.llm import DEFAULT_PLANNER, LLM
from screenjev.planner import Planner, mock_plan
from screenjev.selector import make_selector
from screenjev.trace import Trace

from .common import git_rev, load_env, parse_variant, read_jsonl, run_dir
from .miniwob_env import SUITE, MiniWoBEnv


async def run(a) -> None:
    out = run_dir("miniwob", a.run)
    tasks = a.tasks.split(",") if a.tasks else SUITE
    variants = a.variants.split(",")
    cfg = {"tasks": tasks, "variants": variants, "episodes": a.episodes, "seed0": a.seed0, "planner": a.planner,
           "max_steps": a.max_steps, "min_confidence": a.min_confidence, "scale": a.scale, "mock": is_mock(),
           "git": git_rev(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    path = out / "episodes.jsonl"
    done = {(r["variant"], r["task"], r["seed"]) for r in read_jsonl(path) if r.get("status") != "error" or not a.retry_errors}

    jobs: asyncio.Queue = asyncio.Queue()
    for v in variants:
        for t in tasks:
            for s in range(a.seed0, a.seed0 + a.episodes):
                if (v, t, s) not in done:
                    jobs.put_nowait((v, t, s))
    total = jobs.qsize()
    print(f"{total} episodes to run -> {out}", flush=True)

    perceptions: dict[str, Perception] = {}
    selectors: dict[str, object] = {}
    llm = LLM(a.planner)
    if llm.mock:
        llm.scripted = mock_plan
    lock = asyncio.Lock()
    finished = [0]

    def perception(det: str) -> Perception:
        if det not in perceptions:
            perceptions[det] = Perception(make_detector(det))
        return perceptions[det]

    async def worker():
        async with MiniWoBEnv(scale=a.scale) as env:
            while not jobs.empty():
                v, task, seed = jobs.get_nowait()
                det, sel_spec = parse_variant(v)
                if sel_spec not in selectors:
                    selectors[sel_spec] = make_selector(sel_spec)
                planner = Planner(llm, device="browser", som=sel_spec == "som")
                trace = Trace(out / "traces" / v / f"{task}_{seed}") if a.traces else None
                agent = Agent(env.device, perception(det), planner, selectors[sel_spec],
                              AgentConfig(max_steps=a.max_steps, min_confidence=a.min_confidence), trace)
                try:
                    utterance = await env.reset(task, seed)
                    res = await agent.run(utterance, env_done=lambda: _done(env))
                    done_, raw, reason = await env.result()
                except Exception as e:
                    res, done_, raw, reason = None, False, 0.0, f"harness error: {type(e).__name__}: {e}"
                row = {"variant": v, "task": task, "seed": seed, "success": bool(done_ and raw > 0), "raw_reward": raw,
                       "env_done": done_, "reason": reason}
                if res is not None:
                    row.update({"utterance": res.task, "status": res.status, "n_steps": len(res.steps),
                                "planner_calls": res.planner_calls, "planner_tokens": res.planner_tokens,
                                "planner_cost_usd": res.planner_cost_usd, "select_calls": res.select_calls,
                                "select_cost_usd": res.select_cost_usd, "select_latency_s": res.select_latency_s,
                                "wall_s": res.wall_s, "error": res.error,
                                "not_found": sum(1 for s in res.steps if s.outcome.startswith("NOT DONE")),
                                "steps": [{"step": s.step, "selection": s.selection, "outcome": s.outcome,
                                           "n_elements": s.n_elements} for s in res.steps]})
                else:
                    row["status"] = "error"
                async with lock:
                    with path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    finished[0] += 1
                    if finished[0] % 10 == 0 or finished[0] == total:
                        print(f"  {finished[0]}/{total}", flush=True)

    await asyncio.gather(*(worker() for _ in range(a.concurrency)))
    for s in selectors.values():
        await s.aclose()
    await llm.aclose()
    summary = summarize(read_jsonl(path))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["overall"], indent=1))


async def _done(env: MiniWoBEnv) -> bool:
    return (await env.result())[0]


def summarize(rows: list[dict]) -> dict:
    by_v: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)
    overall, per_task = {}, {}
    for v, rs in by_v.items():
        n = len(rs)
        sel_calls = sum(r.get("select_calls", 0) for r in rs)
        overall[v] = {
            "episodes": n,
            "success": sum(r["success"] for r in rs) / n,
            "task_macro_success": None,
            "mean_steps": sum(r.get("n_steps", 0) for r in rs) / n,
            "not_found_per_episode": sum(r.get("not_found", 0) for r in rs) / n,
            "errors": sum(r.get("status") == "error" for r in rs),
            "planner_tokens_per_episode": sum(r.get("planner_tokens", 0) for r in rs) / n,
            "select_calls": sel_calls,
            "select_cost_per_1k": 1000 * sum(r.get("select_cost_usd", 0) for r in rs) / max(1, sel_calls),
            "select_latency_mean_s": sum(r.get("select_latency_s", 0) for r in rs) / max(1, sel_calls),
            "wall_mean_s": sum(r.get("wall_s", 0) for r in rs) / n,
        }
        pt = defaultdict(list)
        for r in rs:
            pt[r["task"]].append(r["success"])
        per_task[v] = {t: sum(x) / len(x) for t, x in sorted(pt.items())}
        overall[v]["task_macro_success"] = sum(per_task[v].values()) / len(per_task[v])
    return {"overall": overall, "per_task": per_task}


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="yolo+jev,yolo+luna,yolo+llm,yolo+som,dom+jev")
    ap.add_argument("--tasks", help="comma-separated (default: the 25-task suite)")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--planner", default=DEFAULT_PLANNER)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--run", help="results/miniwob/<run> (resumes if it exists)")
    ap.add_argument("--traces", action="store_true", help="save annotated screenshots per step")
    ap.add_argument("--retry-errors", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
