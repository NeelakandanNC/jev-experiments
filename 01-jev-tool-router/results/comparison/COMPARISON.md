| Benchmark | Score | Model | top 5 | top 10 | Tool tokens (all → top 10) | Top-1 right | Mean confidence | ECE | p50 latency | $ / 1k decisions |
|---|---|---|---|---|---|---|---|---|---|---|
| MCP-Atlas | right tool in shortlist | Jev | 70.8% | 80.4% | 43,297 → 1,880 | 39.8% | 50.9% | 0.111 | 619 ms | $0.45 |
|  |  | OpenAI gpt-6-luna | 57.7% | 64.2% | 43,297 → 1,703 | 29.6% | 62.1% | 0.325 | 1252 ms | $0.83 |
| MCP-Bench | share of needed tools in shortlist | Jev | — | 74.5% | 50,256 → 1,944 | 85.4% | 62.5% | 0.230 | 1198 ms | $0.19 |
|  |  | OpenAI gpt-6-luna | — | 72.4% | 50,256 → 2,550 | 84.8% | 55.5% | 0.330 | 897 ms | $0.73 |
| MCP-Universe | needed server in shortlist | Jev | 100.0% | 100.0% | 26,465 → 1,734 | 92.5% | 82.7% | 0.099 | 607 ms | $0.18 |
|  |  | OpenAI gpt-6-luna | 96.0% | 97.0% | 26,465 → 1,592 | 82.4% | 66.6% | 0.158 | 245 ms | $0.33 |

Access: Jev: OpenRouter Decisions API; OpenAI gpt-6-luna: OpenAI Decisions API.

![](accuracy.png)
![](calibration.png)
