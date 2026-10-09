# jev-experiments

Experiments with [Jev](https://openrouter.ai/~typesafe/jev-latest), TypeSafe AI's decision model trained with **RLCD** (Reinforcement Learning for Calibrated Decisions).

Jev is called through a Decisions API ([OpenRouter](https://openrouter.ai/docs/guides/community/jev) or [Vercel AI Gateway](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-decisions)). It doesn't generate text. You give it an `input` and typed questions (`choice`, `score`, `predicate`), and it returns decisions with calibrated probabilities. Each folder here tests one idea against a baseline and publishes reproducible benchmarks.

## Experiments

| # | Idea | Headline result |
|---|------|-----------------|
| [01](01-jev-tool-router) | **Jev as a tool router for MCP agents**: Jev picks the 5–10 tools an agent needs instead of loading all ~250. MCP-Atlas, MCP-Bench and MCP-Universe, vs OpenAI's decision model | Jev's top-10 contains the right tool 80% of the time on MCP-Atlas vs 64% for OpenAI's decision model, with far better calibration (ECE 0.11 vs 0.33) at ~half the cost; ~96% less tool context |

## Running

Each experiment folder has its own README with setup, required keys, and the one-command benchmark.
