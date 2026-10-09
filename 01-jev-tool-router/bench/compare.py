"""Side-by-side comparison of the decision models across the three benchmarks.

    python -m bench.compare            -> results/comparison/{comparison.json,COMPARISON.md,*.png}

Takes each model's latest finished routing run per benchmark (the same merge the dashboard shows).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .datasets import ROOT  # noqa: E402

RESULTS = ROOT / "results" / "routing"
OUT = ROOT / "results" / "comparison"
BENCH = {"atlas": ("MCP-Atlas", [5, 10]), "mcpbench": ("MCP-Bench", [10]), "universe": ("MCP-Universe", [5, 10])}
LABEL = {"jev": "Jev", "gpt-6-luna": "OpenAI gpt-6-luna"}
COLOR = {"jev": "#4f46e5", "gpt-6-luna": "#a8a29e"}
VIA = {"openrouter.ai": "OpenRouter Decisions API", "api.openai.com": "OpenAI Decisions API",
       "ai-gateway.vercel.sh": "Vercel AI Gateway Decisions API"}
INK, MUTED, GRID = "#1c1917", "#78716c", "#e7e5e4"


def latest(bench: str) -> dict:
    merged: dict = {"routers": {}}
    for p in sorted(RESULTS.glob(f"{bench}-*/summary.json"), key=lambda p: p.stat().st_mtime):
        s = json.loads(p.read_text())
        routes = json.loads((p.parent / "config.json").read_text()).get("routes", {})
        for name, v in s.get("routers", {}).items():
            if v.get("cases") and not v.get("errors") and not v.get("mock"):
                via = routes.get(name) or ("https://api.openai.com/v1" if "luna" in name else "")
                merged["routers"][name] = {**v, "run_id": s["run_id"],
                                           "via": next((t for h, t in VIA.items() if h in via), via)}
                merged.update({k: s[k] for k in ("catalog_tokens", "catalog_tools", "catalog_servers", "cases", "tasks")})
    return merged


def metric(mode: str) -> tuple[str, str]:
    return {"step": ("hit", "right tool in shortlist"), "task": ("recall", "share of needed tools in shortlist"),
            "server": ("hit", "needed server in shortlist")}[mode]


def table(data: dict) -> str:
    lines = ["| Benchmark | Score | Model | " + " | ".join(["top 5", "top 10"]) +
             " | Tool tokens (all → top 10) | Top-1 right | Mean confidence | ECE | p50 latency | $ / 1k decisions |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, (name, ks) in BENCH.items():
        d = data[b]
        for i, (m, v) in enumerate(sorted(d["routers"].items(), key=lambda kv: kv[0] != "jev")):
            key, what = metric(v["mode"])
            cells = [f"{v[f'{key}@{k}'] * 100:.1f}%" if k in ks else "—" for k in (5, 10)]
            lines.append(
                f"| {name if i == 0 else ''} | {what if i == 0 else ''} | {LABEL.get(m, m)} | {' | '.join(cells)} | "
                f"{d['catalog_tokens']:,} → {v['tokens@10']:,.0f} | {v['top1_accuracy'] * 100:.1f}% | "
                f"{v['mean_confidence'] * 100:.1f}% | {v['ece']:.3f} | {v['latency_p50_ms']:.0f} ms | "
                f"${v['cost_per_call_usd'] * 1000:.2f} |")
    return "\n".join(lines)


def _style(ax, title, sub=""):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=20)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


def charts(data: dict):
    # 1. accuracy: grouped bars per benchmark x k
    groups = [(b, k) for b, (_, ks) in BENCH.items() for k in ks]
    models = ["jev", "gpt-6-luna"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    w = 0.38
    for j, m in enumerate(models):
        vals = []
        for b, k in groups:
            v = data[b]["routers"].get(m)
            key = metric(v["mode"])[0] if v else "hit"
            vals.append(v[f"{key}@{k}"] * 100 if v else 0)
        xs = [i + (j - 0.5) * w for i in range(len(groups))]
        bars = ax.bar(xs, vals, w * 0.94, color=COLOR[m], label=LABEL[m])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.0f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(groups)), [f"{BENCH[b][0]}\ntop {k}" for b, k in groups])
    ax.set_ylim(0, 110)
    ax.set_ylabel("%", color=MUTED)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    _style(ax, "Picking 5-10 MCP tools for the agent: Jev vs OpenAI",
           "Atlas: exact tool the reference agent called · MCP-Bench: share of needed tools · Universe: needed server")
    fig.savefig(OUT / "accuracy.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2. calibration: one panel per benchmark
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, (b, (name, _)) in zip(axes, BENCH.items()):
        ax.plot([0, 1], [0, 1], ls="--", color=GRID)
        for m in models:
            v = data[b]["routers"].get(m)
            if not v or not v.get("reliability"):
                continue
            xs = [r["confidence"] for r in v["reliability"]]
            ys = [r["accuracy"] for r in v["reliability"]]
            ax.plot(xs, ys, color=COLOR[m], lw=1.5)
            ax.scatter(xs, ys, s=[15 + 3 * r["n"] ** 0.8 for r in v["reliability"]], color=COLOR[m], alpha=0.85,
                       label=f"{LABEL[m]} · ECE {v['ece']:.2f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence in top pick", color=MUTED)
        ax.legend(frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=8.5, loc="lower right")
        _style(ax, name)
    axes[0].set_ylabel("top pick right", color=MUTED)
    fig.suptitle("Does the router's confidence mean anything? (diagonal = perfectly calibrated)", x=0.02, ha="left",
                 fontsize=14, fontweight="bold", color=INK, y=1.04)
    fig.savefig(OUT / "calibration.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {b: latest(b) for b in BENCH}
    (OUT / "comparison.json").write_text(json.dumps(data, indent=1, default=str))
    md = table(data)
    vias = sorted({f"{LABEL.get(m, m)}: {v['via']}" for d in data.values() for m, v in d["routers"].items()})
    (OUT / "COMPARISON.md").write_text(f"{md}\n\nAccess: {'; '.join(vias)}.\n\n![](accuracy.png)\n![](calibration.png)\n")
    charts(data)
    print(md)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
