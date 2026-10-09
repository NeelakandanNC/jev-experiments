"""Run MCP-Bench's own runner with Jev routing patched into its planner.

MCP-Bench pastes every tool's description and schema into the planning prompt as text, so a
chat-completions proxy can't filter them. Instead this wrapper (run with MCP-Bench's venv,
cwd = vendor/mcp-bench) patches TaskExecutor._plan_next_actions to plan against the routed
subset each round, and wires the agent models and the o4-mini judge to AI Gateway.

    JEVROUTE_PROFILE=jev@k8  .venv/bin/python ../../bench/mcpbench_entry.py --models anthropic/claude-sonnet-4.5 ...
    JEVROUTE_PROFILE=full    ... (no routing; same wiring, the baseline)

Everything after the script name is passed to MCP-Bench's runner unchanged.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.getcwd())

from openai import AsyncOpenAI  # noqa: E402

from jevroute.catalog import Catalog, Tool  # noqa: E402
from jevroute.policy import RoutePolicy, dumps  # noqa: E402
from jevroute.proxy import PROFILE_RE  # noqa: E402
from jevroute.routers import make_router, select  # noqa: E402

GATEWAY = os.getenv("AI_GATEWAY_BASE_URL", "https://ai-gateway.vercel.sh/v1")
KEY = os.environ.get("AI_GATEWAY_API_KEY", "")
PROFILE = os.getenv("JEVROUTE_PROFILE", "full")
LOG = Path(os.getenv("JEVROUTE_LOG", ROOT / "results" / "e2e" / "mcpbench" / f"routing-{PROFILE}.jsonl"))
JUDGE = os.getenv("JEVROUTE_JUDGE", "openai/o4-mini")


def patch_models():
    """Expose gateway models (any slug passed to --models) as openai_compatible configs."""
    from llm import factory

    orig = factory.LLMFactory.get_model_configs

    def get_model_configs():
        configs = orig() if os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") else {}
        wanted = []
        if "--models" in sys.argv:
            i = sys.argv.index("--models") + 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                wanted += sys.argv[i].split(",")
                i += 1
        for slug in wanted:
            if "/" in slug:
                configs[slug] = factory.ModelConfig(name=slug, provider_type="openai_compatible", api_key=KEY,
                                                    base_url=GATEWAY, model_name=slug)
        return configs

    factory.LLMFactory.get_model_configs = staticmethod(get_model_configs)


def patch_judge():
    from benchmark import runner
    from llm.provider import LLMProvider

    class GatewayAzure(AsyncOpenAI):  # the runner builds AsyncAzureOpenAI(...) for the judge
        def __init__(self, *a, **kw):
            super().__init__(api_key=KEY, base_url=GATEWAY)

    class JudgeProvider(LLMProvider):
        def __init__(self, client, deployment_name, provider_type):
            if deployment_name == "o4-mini" and provider_type == "azure":
                deployment_name, provider_type = JUDGE, "openai_compatible"
            super().__init__(client, deployment_name, provider_type)

    runner.AsyncAzureOpenAI = GatewayAzure
    runner.LLMProvider = JudgeProvider


def patch_routing():
    if PROFILE == "full":
        return
    m = PROFILE_RE.match(PROFILE)
    if not m:
        raise SystemExit(f"bad JEVROUTE_PROFILE {PROFILE!r}")
    from agent.executor import TaskExecutor

    router = make_router(m["spec"])
    policy = RoutePolicy(router, k=int(m["k"]), adaptive_mass=float(m["mass"]) if m["mass"] else None)
    orig = TaskExecutor._plan_next_actions
    LOG.parent.mkdir(parents=True, exist_ok=True)

    async def routed(self, task, round_num):
        full = self.all_tools
        catalog = Catalog([Tool(n, i.get("server", "?"), i.get("description") or "", i.get("input_schema") or {})
                           for n, i in full.items()], name="mcpbench-live")
        progress = self._build_execution_summary()
        state = f"TASK:\n{task}\n\nPROGRESS SO FAR:\n{progress[-4000:]}" if self.execution_results else f"TASK:\n{task}"
        ranking = await router.rank(state, catalog, intent="next")
        picked = select(ranking, policy.k, policy.adaptive_mass)
        used = [r.get("tool") for r in self.execution_results if r.get("tool") in full]
        chosen = list(dict.fromkeys(picked + used))
        self.all_tools = {n: full[n] for n in chosen}
        try:
            return await orig(self, task, round_num)
        finally:
            self.all_tools = full
            with LOG.open("a") as f:
                f.write(dumps({"ts": time.time(), "profile": PROFILE, "round": round_num, "task": task[:120],
                               "n_tools_in": len(full), "n_tools_out": len(chosen),
                               "tool_tokens_in": catalog.tokens(), "tool_tokens_out": catalog.tokens(chosen),
                               "tools_out": chosen, "router_latency_s": ranking.latency_s,
                               "router_cost_usd": ranking.cost_usd, "router_confidence": ranking.confidence,
                               "mock": ranking.mock}) + "\n")

    TaskExecutor._plan_next_actions = routed


def main():
    if not KEY:
        raise SystemExit("set AI_GATEWAY_API_KEY")
    patch_models()
    patch_judge()
    patch_routing()
    from benchmark.runner import main as bench_main
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    asyncio.run(bench_main())


if __name__ == "__main__":
    main()
