# 01 · Jev as a tool router for MCP agents

A coding agent with 10 MCP servers connected carries 250–300 tool definitions (40–50k tokens) in its context on every turn, and has to pick the right tool out of all of them. Here a **decision model picks the 5 or 10 tools the agent needs next**, and the agent only sees those.

```
 task + progress so far ──▶ decision model ──▶ top-5 / top-10 tools ──▶ agent (Claude etc.) calls one
                           (one /v1/decisions call,                     with ~1–2k tokens of tools
                            a probability for every tool)               instead of ~45k
```

Both decision models are called the same way, through the OpenAI-compatible **Decisions API on Vercel AI Gateway** (`POST https://ai-gateway.vercel.sh/v1/decisions`):

| | top 5 tools | top 10 tools |
|---|---|---|
| **Jev** (`typesafe-ai/jev`, trained with RLCD) | ✓ | ✓ |
| **OpenAI decisions** (`openai/gpt-6-luna-decisions`) | ✓ | ✓ |

That's 4 variations × 3 benchmarks.

## Results

_Pending the first real run._ Each benchmark's numbers, charts (`bench/report.py`) and raw per-decision logs land in `results/routing/<run>/`.

## Benchmarks and what "right" means

| Benchmark | Tasks | Catalog the router chooses from | Gold label per decision |
|---|---|---|---|
| [MCP-Atlas](https://github.com/scaleapi/mcp-atlas) (Scale AI) | 495 of 500 public tasks, 2,060 decisions | 246 tools / 33 servers | the tool the reference trajectory called **at that step** |
| [MCP-Bench](https://github.com/Accenture/mcp-bench) (Accenture) | 103 of 104 tasks | 257 tools / 28 servers | every `Server:tool` the task's dependency plan names; the score is the **share of them in the shortlist** |
| [MCP-Universe](https://github.com/SalesforceAIResearch/MCP-Universe) (Salesforce) | 199 of 232 tasks | 88 tools / 10 servers | the servers the task requires (tasks name servers, not tools) |

The router always sees the **whole catalog**, not the benchmark's per-task subset. That's the realistic "everything connected" setup this idea is about.

Known gaps, reported rather than hidden:
- **MCP-Atlas:** the official list has 307 tools. MongoDB, Slack and Twelve Data (61 tools) won't start without real credentials, so they're missing. The 361 gold steps that need them are skipped and counted in `config.json`.
- **MCP-Universe:** the 33 GitHub tasks are skipped until the GitHub server (Docker) is available.
- **Tool definitions:** these come from each server's `list_tools`, at the pinned versions in `pins/` and the benchmarks' own configs.

## Run it

```bash
./setup.sh                                  # venv + benchmark repos at pinned commits
cp .env.example .env                        # add AI_GATEWAY_API_KEY
.venv/bin/python -m web.server              # http://127.0.0.1:8790: paste key, press Run per benchmark
```

Or from the CLI (same defaults as the dashboard):

```bash
.venv/bin/python -m bench.routing --benchmark atlas --routers jev,decision:openai/gpt-6-luna-decisions --ks 1,5,10
.venv/bin/python -m bench.report results/routing/<run>       # PNG charts + REPORT.md
```

Extra baselines (CLI only): `--routers bm25,embed,llm:anthropic/claude-haiku-4.5`.

## How the router works

`jevroute/routers.py` → `DecisionRouter`:
- **Catalog of 255 tools or fewer** (every catalog here): one `choice` question whose choices are the tools (`value` = tool name, `description` = server + one-line description). The probabilities rank the tools, and top-k is the shortlist.
- **More than 255 tools** (the API's `choice` limit): two stages. First a `predicate` per server ("will the agent need this server?"), all in one request, then a `choice` over the tools of the servers that pass.
- **Per turn:** the router sees the task plus a compressed log of the tool calls so far (`jevroute/policy.py`), and tools already used stay available.

## End-to-end (next step: compare against published leaderboards)

The routing eval checks that the right tools reach the agent. The end-to-end runs check whether the agent then scores as well with 5–10 tools as with all of them, using each benchmark's own harness and judge:

```bash
.venv/bin/python -m bench.e2e atlas    --model anthropic/claude-sonnet-4.5 --profiles full,jev@k5,jev@k10 --num-tasks 20
.venv/bin/python -m bench.e2e mcpbench --model anthropic/claude-sonnet-4.5 --profiles full,jev@k5,jev@k10
.venv/bin/python -m bench.e2e universe --model anthropic/claude-sonnet-4.5 --profiles full,jev@k5 --domains financial_analysis
```

How each benchmark is hooked:
- **MCP-Atlas and MCP-Universe:** the agent's traffic goes through `jevroute/proxy.py`, an OpenAI-compatible proxy that trims the request's `tools` array. The condition rides in the model name (`jev@k5::anthropic/claude-sonnet-4.5`), so no benchmark code changes.
- **MCP-Bench:** its tools are pasted into the planning prompt as text, so `bench/mcpbench_entry.py` patches its planner instead. Its judge stays `o4-mini`, via the Gateway.

These need each benchmark's server keys (see `.env.example`), Docker for MCP-Atlas, and `scripts/setup_*.sh`.

## Layout

```
jevroute/   catalog.py  decide.py (Decisions API client)  routers.py  policy.py  proxy.py  metrics.py
bench/      datasets.py (benchmarks → routing cases)  routing.py  e2e.py  report.py  catalogs.py
            mcpbench_entry.py  universe_entry.py
web/        server.py + static/index.html   (minimal dashboard)
data/catalogs/   tool catalogs for the three benchmarks (committed)
results/    routing/<run>/{config,summary}.json, cases.jsonl, charts/
tests/      .venv/bin/python -m pytest
```
