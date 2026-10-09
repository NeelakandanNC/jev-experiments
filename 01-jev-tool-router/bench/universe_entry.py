"""Run one MCP-Universe benchmark YAML and dump per-task evaluation results as JSON.

Runs inside MCP-Universe's venv (cwd = vendor/MCP-Universe):
    .venv/bin/python ../../bench/universe_entry.py <config relative to benchmark/configs> <out.json>
"""

import asyncio
import json
import sys
import time


async def main(config: str, out: str):
    from mcpuniverse.benchmark.runner import BenchmarkRunner
    from mcpuniverse.tracer.collectors import MemoryCollector

    t0 = time.time()
    runner = BenchmarkRunner(config)
    results = await runner.run(trace_collector=MemoryCollector())
    tasks = {}
    for res in results:
        for name, r in res.task_results.items():
            evals = r.get("evaluation_results") or []
            tasks[name] = {
                "passed": bool(evals) and all(e.passed for e in evals),
                "evals_passed": sum(1 for e in evals if e.passed),
                "evals_total": len(evals),
                "reasons": [getattr(e, "reason", "") for e in evals if not e.passed][:3],
            }
    with open(out, "w") as f:
        json.dump({"config": config, "seconds": time.time() - t0, "tasks": tasks}, f, indent=1, default=str)
    n = len(tasks)
    print(f"universe: {sum(t['passed'] for t in tasks.values())}/{n} tasks passed", flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
