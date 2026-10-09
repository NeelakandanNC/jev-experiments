"""Charts (PNG, sized for X posts) and a markdown table from a run's summary.json.

    python -m bench.report results/routing/<run>      -> <run>/charts/*.png + <run>/REPORT.md
    python -m bench.report results/e2e/<bench>/<run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1c1917", "#78716c", "#e7e5e4"
JEV = ["#4f46e5", "#818cf8", "#3730a3", "#a5b4fc"]
OTHERS = ["#a8a29e", "#57534e", "#d6d3d1", "#0f766e", "#b45309", "#7c3aed"]
NAMES = {"atlas": "MCP-Atlas", "mcpbench": "MCP-Bench", "universe": "MCP-Universe"}


def _style(ax, title: str, subtitle: str = ""):
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=22)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _colors(names):
    out, i, j = [], 0, 0
    for n in names:
        if n.startswith("jev"):
            out.append(JEV[j % len(JEV)])
            j += 1
        else:
            out.append(OTHERS[i % len(OTHERS)])
            i += 1
    return out


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def routing_charts(s: dict, out: Path) -> list[str]:
    rs = s["routers"]
    names = list(rs)
    ks = s["ks"]
    bench = NAMES.get(s["benchmark"], s["benchmark"])
    sub = f"{s['cases']} routing decisions from {s['tasks']} tasks · {s['catalog_tools']} tools / {s['catalog_servers']} servers"
    mock = any(v.get("mock") for v in rs.values())
    if mock:
        sub += " · MOCK DATA, NOT REAL RESULTS"
    files = []

    # 1. hit@k grouped bars
    fig, ax = plt.subplots(figsize=(10, 5.6))
    width = 0.8 / len(names)
    for i, (n, c) in enumerate(zip(names, _colors(names))):
        vals = [rs[n].get(f"hit@{k}", 0) * 100 for k in ks]
        xs = [j + i * width for j in range(len(ks))]
        bars = ax.bar(xs, vals, width * 0.92, label=n, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(ks))], [f"top-{k}" for k in ks])
    ax.set_ylim(0, 108)
    ax.set_ylabel("right tool offered (%)", color=MUTED)
    ax.legend(frameon=False, fontsize=9, ncol=min(len(names), 5), loc="upper center", bbox_to_anchor=(0.5, -0.08))
    _style(ax, f"{bench}: is the right tool in the agent's shortlist?", sub)
    _save(fig, out / "hit_at_k.png")
    files.append("hit_at_k.png")

    # 2. context tokens: full catalog vs what each router hands the agent
    k = ks[min(2, len(ks) - 1)]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["all tools"] + [f"{n} (top-{k})" for n in names]
    vals = [s["catalog_tokens"]] + [rs[n].get(f"tokens@{k}", 0) for n in names]
    bars = ax.barh(labels[::-1], vals[::-1], color=(["#d6d3d1"] + _colors(names))[::-1])
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:,.0f}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("tool-definition tokens in the agent's context, per turn", color=MUTED)
    ax.grid(axis="x", color=GRID)
    _style(ax, f"{bench}: tool context the agent has to read", sub)
    ax.grid(axis="y", visible=False)
    _save(fig, out / "context_tokens.png")
    files.append("context_tokens.png")

    # 3. reliability diagram for routers that report confidence
    cal = {n: v for n, v in rs.items() if v.get("reliability")}
    if cal:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.plot([0, 1], [0, 1], ls="--", color=GRID, lw=1.2, label="perfect calibration")
        for (n, v), c in zip(cal.items(), _colors(list(cal))):
            xs = [b["confidence"] for b in v["reliability"]]
            ys = [b["accuracy"] for b in v["reliability"]]
            sz = [20 + 4 * b["n"] for b in v["reliability"]]
            ax.plot(xs, ys, color=c, lw=1.5)
            ax.scatter(xs, ys, s=sz, color=c, alpha=0.8, label=f"{n} (ECE {v['ece']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("router's confidence in its top pick", color=MUTED)
        ax.set_ylabel("how often the top pick was right", color=MUTED)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        _style(ax, f"{bench}: does confidence mean anything?", "dot size = number of decisions")
        _save(fig, out / "calibration.png")
        files.append("calibration.png")

    # 4. latency and cost per routing call
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    cols = _colors(names)
    lat = [rs[n].get("latency_p50_ms", 0) for n in names]
    cost = [rs[n].get("cost_per_call_usd", 0) * 1000 for n in names]
    a1.bar(names, lat, color=cols)
    a2.bar(names, cost, color=cols)
    for a, vals, fmt in ((a1, lat, "{:.0f} ms"), (a2, cost, "${:.4f}")):
        for i, v in enumerate(vals):
            a.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=MUTED)
        a.tick_params(axis="x", rotation=25)
    _style(a1, "Router latency (p50)", "")
    _style(a2, "Router cost per 1,000 calls", "")
    fig.suptitle(f"{bench}: what routing costs", x=0.06, ha="left", fontsize=15, fontweight="bold", color=INK, y=1.03)
    _save(fig, out / "latency_cost.png")
    files.append("latency_cost.png")
    return files


def routing_table(s: dict) -> str:
    ks = s["ks"]
    k_tok = ks[min(2, len(ks) - 1)]
    head = ["router"] + [f"hit@{k}" for k in ks] + ["MRR", f"tool tokens@{k_tok}", "p50 latency", "$/1k calls", "ECE"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n, v in s["routers"].items():
        row = [f"`{n}`"] + [f"{v.get(f'hit@{k}', 0) * 100:.1f}%" for k in ks]
        row += [f"{v.get('mrr', 0):.3f}", f"{v.get(f'tokens@{k_tok}', 0):,.0f}",
                f"{v.get('latency_p50_ms', 0):.0f} ms", f"${v.get('cost_per_call_usd', 0) * 1000:.4f}",
                f"{v['ece']:.3f}" if "ece" in v else "—"]
        lines.append("| " + " | ".join(row) + " |")
    lines.append(f"\nFull catalog: {s['catalog_tools']} tools, {s['catalog_tokens']:,} tokens of tool definitions. "
                 f"{s['cases']} routing decisions from {s['tasks']} tasks.")
    return "\n".join(lines)


def e2e_charts(s: dict, out: Path) -> list[str]:
    cs = s.get("conditions") or {}
    if not cs:
        return []
    names = list(cs)
    score_key = next((k for k in ("pass_rate_0.75", "pass_rate", "task_success_rate", "overall_score")
                      if all(k in (cs[n].get("score") or {}) for n in names)), None)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    cols = _colors(names)
    if score_key:
        vals = [cs[n]["score"][score_key] for n in names]
        a1.bar(names, vals, color=cols)
        for i, v in enumerate(vals):
            a1.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    _style(a1, score_key or "score", "benchmark's own scorer")
    toks = [cs[n].get("prompt_tokens") or cs[n].get("tool_tokens_offered") or 0 for n in names]
    a2.bar(names, toks, color=cols)
    for i, v in enumerate(toks):
        a2.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, color=MUTED)
    _style(a2, "Agent input tokens", "summed over the run")
    bench = NAMES.get(s.get("benchmark", ""), s.get("benchmark", ""))
    fig.suptitle(f"{bench} end-to-end · {s.get('model')}", x=0.06, ha="left", fontsize=15, fontweight="bold",
                 color=INK, y=1.03)
    _save(fig, out / "e2e.png")
    return ["e2e.png"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    args = ap.parse_args()
    d = Path(args.run_dir)
    s = json.loads((d / "summary.json").read_text())
    print(f"report for {d}")
    if "routers" in s:
        files = routing_charts(s, d / "charts")
        md = routing_table(s)
    else:
        files = e2e_charts(s, d / "charts")
        md = "```json\n" + json.dumps(s.get("conditions"), indent=1)[:4000] + "\n```"
    (d / "REPORT.md").write_text(f"# {s.get('run_id')}\n\n{md}\n\n" + "\n".join(f"![]({'charts/' + f})" for f in files) + "\n")
    print(f"  {d / 'REPORT.md'}")


if __name__ == "__main__":
    main()
