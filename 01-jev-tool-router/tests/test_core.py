import json
import os

import httpx
import pytest

os.environ["JEVROUTE_MOCK"] = "1"

from jevroute.catalog import Catalog, Tool  # noqa: E402
from jevroute.decide import _parse  # noqa: E402
from jevroute.metrics import ece, hit_at_k, mrr, recall_at_k, summarize  # noqa: E402
from jevroute.policy import RoutePolicy, state_from_messages  # noqa: E402
from jevroute.proxy import Proxy, create_app, parse_model  # noqa: E402
from jevroute.routers import BM25, DecisionRouter, Ranking, make_router, select  # noqa: E402


def make_catalog(n_extra: int = 0) -> Catalog:
    tools = [
        Tool("github_search_repositories", "github", "Search GitHub repositories by query"),
        Tool("github_get_issue", "github", "Get an issue from a GitHub repository"),
        Tool("whois_whois_domain", "whois", "Look up WHOIS domain registration records"),
        Tool("weather_get_forecast", "weather", "Get the weather forecast for a city"),
        Tool("calculator_calculate", "calculator", "Evaluate a math expression"),
    ]
    tools += [Tool(f"filler{i}_noop", f"filler{i}", f"Unrelated filler tool number {i}") for i in range(n_extra)]
    return Catalog(tools, name="test")


def test_catalog_tokens_and_openai_roundtrip():
    cat = make_catalog()
    assert cat.tokens() > 0
    assert cat.tokens(["calculator_calculate"]) < cat.tokens()
    back = Catalog.from_openai_tools([t.to_openai() for t in cat.tools], known=cat)
    assert [t.server for t in back.tools] == [t.server for t in cat.tools]


def test_server_guess_without_known_catalog():
    tools = [Tool("mcp-code-executor_execute_code", "x").to_openai(), Tool("github__get_issue", "x").to_openai(),
             Tool("Wikipedia:search", "x").to_openai()]
    servers = [t.server for t in Catalog.from_openai_tools(tools).tools]
    assert servers == ["mcp-code-executor", "github", "Wikipedia"]


def test_parse_openai_decisions_response():
    body = {
        "model": "typesafe-ai/jev",
        "answers": [{"type": "choice", "name": "tool", "choice": "b", "confidence": 0.8,
                     "probabilities": [{"value": "a", "probability": 0.2}, {"value": "b", "probability": 0.8}]},
                    {"type": "predicate", "name": "s0", "probability": 0.95}],
        "usage": {"input_tokens": 96, "output_tokens": 0, "total_tokens": 96},
        "provider_metadata": {"gateway": {"cost": "0.0000096"}},
    }
    d = _parse(body, 0.1)
    assert d.answers["tool"].ranked()[0] == ("b", 0.8)
    assert d.answers["s0"].probability == 0.95
    assert d.input_tokens == 96 and d.cost_usd == pytest.approx(9.6e-6)


async def test_decision_router_flat_and_two_stage():
    cat = make_catalog()
    for strategy in ("flat", "two_stage", "prefilter"):
        r = DecisionRouter(strategy=strategy, server_threshold=0.1)
        rank = await r.rank("when was the domain assaultcube.net registered? use whois", cat)
        assert rank.order[0] == "whois_whois_domain", strategy
        assert set(rank.order) == set(cat.by_name), "ranking must cover the whole catalog"
        assert rank.mock


async def test_two_stage_kicks_in_above_255_tools():
    cat = make_catalog(n_extra=300)
    rank = await DecisionRouter().rank("weather forecast for Paris", cat)
    assert rank.debug["strategy"] == "two_stage"
    assert rank.calls == 2
    assert rank.order[0] == "weather_get_forecast"


async def test_bm25_ranks_lexical_match_first():
    rank = await BM25().rank("search repositories on github", make_catalog())
    assert rank.order[0] == "github_search_repositories"


def test_select_topk_and_adaptive():
    r = Ranking(order=["a", "b", "c", "d"], scores={"a": 0.6, "b": 0.35, "c": 0.04, "d": 0.01})
    assert select(r, 3) == ["a", "b", "c"]
    assert select(r, 3, adaptive_mass=0.9) == ["a", "b"]
    assert select(r, 1, adaptive_mass=0.99) == ["a"]


def test_state_from_messages_keeps_task_and_used_tools():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Find the repo age"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "1", "type": "function", "function": {"name": "github_search_repositories", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "assaultcube/AC created 2013"}]
    state, used = state_from_messages(msgs)
    assert state.startswith("TASK:\nFind the repo age")
    assert "created 2013" in state
    assert used == ["github_search_repositories"]


async def test_policy_is_sticky():
    cat = make_catalog()
    msgs = [{"role": "user", "content": "weather forecast in Paris"},
            {"role": "assistant", "tool_calls": [{"function": {"name": "github_get_issue", "arguments": "{}"}}]}]
    chosen, _, _ = await RoutePolicy(BM25(), k=1).choose(msgs, cat)
    assert chosen[0] == "weather_get_forecast" and "github_get_issue" in chosen


def test_metrics():
    assert hit_at_k(["a", "b"], {"b"}, 2) and not hit_at_k(["a", "b"], {"b"}, 1)
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert mrr(["a", "b"], {"b"}) == 0.5
    e, table = ece([0.9, 0.9, 0.1, 0.1], [True, True, False, False])
    assert e == pytest.approx(0.1) and len(table) == 2
    rows = [{"order": ["a", "b"], "gold": ["a"], "tokens_at": {"1": 10}, "full_tokens": 100, "latency_s": 0.01,
             "cost_usd": 0.0, "router_tokens": 5, "confidence": 0.7}]
    s = summarize(rows, [1], "step")
    assert s["hit@1"] == 1.0 and s["top1_accuracy"] == 1.0


def test_make_router_specs():
    assert make_router("jev").name == "jev"
    assert make_router("jev:two_stage").strategy == "two_stage"
    r = make_router("decision:openai/gpt-6-luna:flat")
    assert r.model == "openai/gpt-6-luna" and r.strategy == "flat"
    assert parse_model("jev@k3::anthropic/claude-sonnet-4.5") == ("", "jev@k3", "anthropic/claude-sonnet-4.5")
    assert parse_model("run7|bm25@k5~0.9::x/y") == ("run7", "bm25@k5~0.9", "x/y")
    assert parse_model("openai/gpt-4o") == ("", "full", "openai/gpt-4o")
    assert parse_model("gpt-4.1") == ("", "full", "openai/gpt-4.1")


async def test_proxy_filters_tools_and_logs(tmp_path):
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["body"] = body
        name = body["tools"][0]["function"]["name"] if body.get("tools") else None
        msg = {"role": "assistant", "content": None,
               "tool_calls": [{"id": "x", "type": "function", "function": {"name": name, "arguments": "{}"}}]}
        return httpx.Response(200, json={"choices": [{"message": msg}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": 5}})

    cat = make_catalog(n_extra=20)
    proxy = Proxy("https://gateway.test/v1", "k", tmp_path / "log.jsonl", transport=httpx.MockTransport(upstream))
    app = create_app(proxy)
    req = {"model": "t1|jev@k2::anthropic/claude-sonnet-4.5", "tools": [t.to_openai() for t in cat.tools],
           "messages": [{"role": "user", "content": "what is the weather forecast for Paris"}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as c:
        r = await c.post("/v1/chat/completions", json=req)
        assert r.status_code == 200
        assert seen["body"]["model"] == "anthropic/claude-sonnet-4.5"
        assert len(seen["body"]["tools"]) == 2
        assert seen["body"]["tools"][0]["function"]["name"] == "weather_get_forecast"

        r = await c.post("/v1/chat/completions", json={**req, "model": "full::anthropic/claude-sonnet-4.5"})
        assert len(seen["body"]["tools"]) == len(cat)

        r = await c.post("/v1/chat/completions", json={**req, "model": "nonsense@::x"})
        assert r.status_code == 400

    recs = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert recs[0]["run"] == "t1" and recs[0]["n_tools_in"] == 25 and recs[0]["n_tools_out"] == 2
    assert recs[0]["tool_tokens_out"] < recs[0]["tool_tokens_in"]
    assert recs[0]["called"] == ["weather_get_forecast"] and recs[0]["called_unoffered"] == []
    assert recs[1]["profile"] == "full" and recs[1]["n_tools_out"] == 25


async def test_server_mode_scores_servers_covered_by_top_k_tools():
    from bench.datasets import Case
    from bench.routing import run_case

    cat = make_catalog()
    case = Case("t#0", "t", "weather forecast for Paris", ["weather"], "server")
    row = await run_case(BM25(), case, cat, ks=[1, 3], adaptive=None)
    assert row["order"][0] == "weather"  # server of the top-ranked tool
    assert len(row["order"]) == len(cat)  # one entry per ranked tool, duplicates kept
    assert row["tokens_at"]["1"] == cat.tokens(["weather_get_forecast"])


def test_decision_spec_without_strategy_keeps_full_slug():
    r = make_router("decision:openai/gpt-6-luna")
    assert r.model == "openai/gpt-6-luna" and r.strategy == "auto"


def test_native_adapter_roundtrip():
    from jevroute.decide import _NativeClient

    qs = [{"type": "choice", "name": "tool", "instructions": "next?",
           "choices": [{"value": "a", "description": "tool a"}, {"value": "b"}]},
          {"type": "predicate", "name": "s0", "instructions": "need server?"},
          {"type": "score", "name": "u", "instructions": "urgency", "levels": [{"label": "low"}, {"label": "high"}]}]
    native = _NativeClient.to_native(qs)
    assert native["tool"]["criteria"] == {"a": "tool a", "b": "b"}
    assert native["s0"] == {"type": "noul", "instructions": "need server?"}
    assert native["u"]["criteria"] == ["low", "high"]
    body = {"model": "typesafe/jev-1.13-20260917", "provider": "TypeSafe",
            "answers": {"tool": {"type": "choice", "choice": "b", "confidence": 0.7, "probabilities": {"a": 0.2, "b": 0.8}},
                        "s0": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 4.2e-6}}
    d = _parse(_NativeClient.from_native(body, qs), 0.1)
    assert d.answers["tool"].ranked()[0] == ("b", 0.8)
    assert d.answers["s0"].probability == 0.9
    assert d.answers["u"].type == "refusal"
    assert d.cost_usd == pytest.approx(4.2e-6) and d.input_tokens == 100
