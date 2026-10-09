"""Routing eval: does the router put the right tools in front of the agent?

    python -m bench.routing --benchmark atlas --routers bm25,embed,jev --ks 1,3,5,10 --limit-tasks 50

Only needs AI_GATEWAY_API_KEY: no MCP servers run. Writes results/routing/<run>/
{config,progress,summary}.json and cases.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

from jevroute.metrics import summarize
from jevroute.routers import DecisionRouter, Router, close_all, make_router, select

from .datasets import BENCHMARKS, ROOT, Case, load_cases, load_catalog

RESULTS = ROOT / "results" / "routing"


def server_order(order: list[str], catalog) -> list[str]:
    """Server of each ranked tool, duplicates kept, so order[:k] = servers covered by the top-k tools."""
    return [catalog.by_name[n].server for n in order if n in catalog]


async def run_case(router: Router, case: Case, catalog, ks: list[int], adaptive: float | None) -> dict:
    row = {"router": router.name, "case": case.id, "task": case.task_id, "gold": case.gold, "mode": case.mode}
    try:
        r = await router.rank(case.state, catalog, intent="next" if case.mode == "step" else "task")
    except Exception as e:  # keep going; errors are counted in the summary
        return {**row, "error": f"{type(e).__name__}: {e}"}
    order = server_order(r.order, catalog) if case.mode == "server" else r.order
    tool_tokens = {str(k): catalog.tokens(r.order[:k]) for k in ks}
    row.update(order=order[:25], tokens_at=tool_tokens, full_tokens=catalog.tokens(), latency_s=r.latency_s,
               cost_usd=r.cost_usd, router_tokens=r.router_tokens, confidence=r.confidence, mock=r.mock,
               debug=r.debug)
    if adaptive and isinstance(router, DecisionRouter) and case.mode != "server":
        picked = select(r, max(ks), adaptive)
        row.update(adaptive=picked, adaptive_tokens=catalog.tokens(picked))
    return row


async def retry_errors(args) -> Path:
    """Re-run only the decisions that errored in an existing run, then rewrite its summary."""
    out = RESULTS / args.retry_errors
    config = json.loads((out / "config.json").read_text())
    catalog = load_catalog(config["benchmark"])
    cases, _ = load_cases(config["benchmark"], catalog, config.get("limit_tasks"), config.get("seed", 0))
    by_id = {c.id: c for c in cases}
    rows = [json.loads(l) for l in (out / "cases.jsonl").read_text().splitlines()]
    todo = [(i, r) for i, r in enumerate(rows) if r.get("error")]
    print(f"[{out.name}] retrying {len(todo)} errored decisions", flush=True)
    routers = {}
    for spec in config["routers"]:
        r = make_router(spec)
        routers[r.name] = r
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i, row):
        async with sem:
            rows[i] = await run_case(routers[row["router"]], by_id[row["case"]], catalog, config["ks"],
                                     config.get("adaptive"))

    try:
        await asyncio.gather(*(one(i, r) for i, r in todo))
    finally:
        await close_all(list(routers.values()))
    (out / "cases.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    mode = cases[0].mode if cases else "step"
    by_router: dict[str, list[dict]] = {}
    for r in rows:
        by_router.setdefault(r["router"], []).append(r)
    summary = {**config, "finished": time.time(),
               "routers": {n: summarize(rs, config["ks"], mode) for n, rs in by_router.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for n, s in summary["routers"].items():
        print(f"  = {n}: errors={s.get('errors')} hit@5={s.get('hit@5', float('nan')):.3f}", flush=True)
    return out


async def main_async(args) -> Path:
    catalog = load_catalog(args.benchmark)
    cases, stats = load_cases(args.benchmark, catalog, args.limit_tasks, args.seed)
    ks = [int(k) for k in args.ks.split(",")]
    run_id = args.run_id or time.strftime(f"{args.benchmark}-%Y%m%d-%H%M%S")
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    specs = [s for s in args.routers.split(",") if s]
    config = {"run_id": run_id, "benchmark": args.benchmark, "routers": specs, "ks": ks, "adaptive": args.adaptive,
              "limit_tasks": args.limit_tasks, "seed": args.seed, "catalog_tools": len(catalog),
              "catalog_servers": len(catalog.servers), "catalog_tokens": catalog.tokens(), **stats,
              "started": time.time()}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(f"[{run_id}] {len(cases)} cases from {stats['tasks']} tasks; catalog {len(catalog)} tools / "
          f"{len(catalog.servers)} servers / {catalog.tokens():,} tokens", flush=True)

    routers = [make_router(s) for s in specs]
    config["routes"] = {r.name: getattr(getattr(r, "client", None), "via", None) or "local" for r in routers}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    total = len(cases) * len(routers)
    done = 0
    sem = asyncio.Semaphore(args.concurrency)
    rows_by_router: dict[str, list[dict]] = {r.name: [] for r in routers}
    progress = out / "progress.json"

    async def one(router, case):
        nonlocal done
        async with sem:
            row = await run_case(router, case, catalog, ks, args.adaptive)
        rows_by_router[router.name].append(row)
        with (out / "cases.jsonl").open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        done += 1
        if done % 10 == 0 or done == total:
            progress.write_text(json.dumps({"done": done, "total": total, "updated": time.time()}))
            if row.get("error"):
                print(f"  ! {row['router']} {row['case']}: {row['error'][:200]}", flush=True)
            print(f"  {done}/{total}", flush=True)

    try:
        for router in routers:  # routers run one after another so latencies don't interfere
            t0 = time.time()
            await asyncio.gather(*(one(router, c) for c in cases))
            s = summarize(rows_by_router[router.name], ks, cases[0].mode if cases else "step")
            headline = f"hit@{ks[-2] if len(ks) > 1 else ks[0]}"
            print(f"  = {router.name}: {headline}={s.get(headline, float('nan')):.3f} "
                  f"p50={s.get('latency_p50_ms', float('nan')):.0f}ms errors={s.get('errors')} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        await close_all(routers)

    summary = {**config, "finished": time.time(),
               "routers": {name: summarize(rows, ks, cases[0].mode if cases else "step")
                           for name, rows in rows_by_router.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    progress.write_text(json.dumps({"done": total, "total": total, "updated": time.time(), "finished": True}))
    print(f"[{run_id}] wrote {out / 'summary.json'}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=BENCHMARKS)
    ap.add_argument("--retry-errors", metavar="RUN_ID", help="re-run only the errored decisions of a run")
    ap.add_argument("--routers", default="jev,decision:openai/gpt-6-luna")
    ap.add_argument("--ks", default="1,5,10")
    ap.add_argument("--adaptive", type=float, default=0.9, help="probability mass for adaptive-k (0 disables)")
    ap.add_argument("--limit-tasks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--run-id")
    args = ap.parse_args(argv)
    args.adaptive = args.adaptive or None
    if not args.benchmark and not args.retry_errors:
        ap.error("--benchmark or --retry-errors is required")
    try:
        asyncio.run(retry_errors(args) if args.retry_errors else main_async(args))
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
