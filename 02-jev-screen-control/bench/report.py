"""Charts + markdown tables from benchmark runs.

    python -m bench.report                                   # latest screenspot + miniwob runs
    python -m bench.report --screenspot <run> --miniwob <run>

Writes results/report/{REPORT.md, *.png}. The README's results section is pasted from REPORT.md.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import RESULTS, ROOT, pct, read_jsonl, reliability  # noqa: E402
from .miniwob import summarize as mw_summarize  # noqa: E402
from .screenspot import summarize as ss_summarize  # noqa: E402

OUT = RESULTS / "report"
COLORS = {"jev": "#7c3aed", "luna": "#0ea5e9", "llm": "#f59e0b", "som": "#64748b", "llm-coords": "#ef4444"}
LABEL = {"jev": "Jev", "luna": "OpenAI gpt-6-luna (decision)", "llm": "gpt-6-luna (chat, picks id)",
         "som": "gpt-6-luna Set-of-Mark", "llm-coords": "gpt-6-luna (x, y directly)"}


def color(v: str) -> str:
    return COLORS.get(v.split("+")[-1], "#94a3b8")


def nice(v: str) -> str:
    det, _, sel = v.rpartition("+")
    name = LABEL.get(sel, sel)
    return f"{name} · {det}" if det else name


def latest(kind: str) -> Path | None:
    runs = sorted((RESULTS / kind).glob("*/"), key=lambda p: p.stat().st_mtime) if (RESULTS / kind).exists() else []
    runs = [r for r in runs if not json.loads((r / "config.json").read_text()).get("mock")] if runs else []
    return runs[-1] if runs else None


def screenspot_section(run: Path) -> str:
    rows = read_jsonl(run / "cases.jsonl")
    s = ss_summarize(rows)
    ov, bd = s["overall"], s["by_domain_type"]
    variants = sorted(ov, key=lambda v: -ov[v]["accuracy"])
    cells = sorted({k for v in bd.values() for k in v})
    md = [f"### ScreenSpot-v2 grounding (`{run.relative_to(ROOT)}`)\n",
          "| Variant | n | Accuracy | " + " | ".join(cells) + " | Detector ceiling | Mean conf. | ECE | Acc. @ top-50% conf. | p50 latency | $ / 1k |",
          "|---|---|---|" + "---|" * len(cells) + "---|---|---|---|---|---|"]
    for v in variants:
        o = ov[v]
        f = lambda x, p=True: "—" if x is None else (pct(x) if p else f"{x:.3f}")
        md.append(f"| {nice(v)} | {o['n']} | **{pct(o['accuracy'])}** | " + " | ".join(pct(bd[v].get(c, float('nan'))) for c in cells)
                  + f" | {f(o['reachable'])} | {f(o['mean_confidence'])} | {f(o['ece'], False)} | {f(o['acc_at_50pct_coverage'])}"
                  + f" | {'—' if o['latency_p50_s'] is None else f'{1000 * o['latency_p50_s']:.0f} ms'} | ${o['cost_per_1k']:.2f} |")

    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        ys = [bd[v].get(c, 0) * 100 for c in cells]
        ax.bar([j + i * width for j in range(len(cells))], ys, width, label=nice(v), color=color(v),
               alpha=1.0 if v.startswith("yolo") or "+" not in v else 0.55)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(cells))], cells, rotation=0, fontsize=8)
    ax.set_ylabel("click accuracy (%)")
    ax.set_title("ScreenSpot-v2: is the click inside the target?")
    ax.legend(fontsize=7, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "screenspot_accuracy.png", dpi=150)
    plt.close(fig)

    by_v = defaultdict(list)
    for r in rows:
        if r.get("confidence") is not None:
            by_v[r["variant"]].append((r["confidence"], r["hit"]))
    if by_v:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
        a1.plot([0, 1], [0, 1], ls="--", c="#999", lw=1)
        for v, ch in sorted(by_v.items()):
            pts = reliability([c for c, _ in ch], [h for _, h in ch])
            a1.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=nice(v), color=color(v))
            srt = sorted(ch, key=lambda t: -t[0])
            covs = [k / len(srt) for k in range(1, len(srt) + 1)]
            accs, hits = [], 0
            for k, (_, h) in enumerate(srt, 1):
                hits += h
                accs.append(hits / k)
            a2.plot(covs, accs, label=nice(v), color=color(v))
        a1.set(xlabel="confidence of the pick", ylabel="share of picks that hit", title="Calibration")
        a2.set(xlabel="share of instructions acted on (most confident first)", ylabel="accuracy on those",
               title="Abstain when unsure")
        for a in (a1, a2):
            a.spines[["top", "right"]].set_visible(False)
        a1.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "screenspot_calibration.png", dpi=150)
        plt.close(fig)
    md.append("\n![ScreenSpot accuracy](screenspot_accuracy.png)\n\n![Calibration](screenspot_calibration.png)\n")
    return "\n".join(md)


def miniwob_section(run: Path) -> str:
    rows = read_jsonl(run / "episodes.jsonl")
    s = mw_summarize(rows)
    ov, pt = s["overall"], s["per_task"]
    variants = sorted(ov, key=lambda v: -ov[v]["success"])
    md = [f"### MiniWoB++ end-to-end (`{run.relative_to(ROOT)}`)\n",
          "| Variant | Episodes | Success | Task-macro success | Steps / episode | Not-found / episode | Grounding $ / 1k calls | Grounding latency | Planner tokens / episode |",
          "|---|---|---|---|---|---|---|---|---|"]
    for v in variants:
        o = ov[v]
        md.append(f"| {nice(v)} | {o['episodes']} | **{pct(o['success'])}** | {pct(o['task_macro_success'])} | {o['mean_steps']:.1f}"
                  f" | {o['not_found_per_episode']:.2f} | ${o['select_cost_per_1k']:.2f} | {1000 * o['select_latency_mean_s']:.0f} ms"
                  f" | {o['planner_tokens_per_episode']:.0f} |")
    tasks = sorted({t for v in pt.values() for t in v})
    md += ["", "<details><summary>Per task</summary>\n", "| Task | " + " | ".join(nice(v) for v in variants) + " |",
           "|---|" + "---|" * len(variants)]
    for t in tasks:
        md.append(f"| {t} | " + " | ".join(pct(pt[v].get(t, float('nan'))) for v in variants) + " |")
    md.append("\n</details>\n")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh([nice(v) for v in variants][::-1], [ov[v]["success"] * 100 for v in variants][::-1],
            color=[color(v) for v in variants][::-1])
    for i, v in enumerate(variants[::-1]):
        ax.text(ov[v]["success"] * 100 + 1, i, pct(ov[v]["success"]), va="center", fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("episodes solved (%)")
    ax.set_title(f"MiniWoB++ ({len(tasks)} tasks)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "miniwob_success.png", dpi=150)
    plt.close(fig)
    md.append("![MiniWoB success](miniwob_success.png)\n")
    return "\n".join(md)


def detector_section() -> str:
    p = ROOT / "models" / "screenjev-yolo.json"
    if not p.exists():
        return ""
    m = json.loads(p.read_text())
    md = ["### Detector (`models/screenjev-yolo.pt`)\n",
          f"YOLO11n fine-tuned from COCO for {m['epochs']} epochs at {m['imgsz']} px on {m['device']} ({m['train_hours']} h"
          + (f"; {' + '.join(str(st['epochs']) for st in m['stages'])} epochs: " + "; then ".join(st['note'] for st in m['stages']) if m.get('stages') else "") + "). "
          f"Train set: {m['dataset']['train']['images']} screenshots.\n",
          "| Eval image size | Split | mAP50 | mAP50-95 | Precision | Recall |", "|---|---|---|---|---|---|"]
    for size, res in m["metrics"].items():
        for split, r in res.items():
            md.append(f"| {size.replace('imgsz', '')} | {split} | {pct(r['mAP50'])} | {pct(r['mAP50-95'])} | {pct(r['precision'])} | {pct(r['recall'])} |")
    curve = ROOT / "models" / "screenjev-yolo-training.csv"
    if curve.exists():
        import csv
        rows = list(csv.DictReader(curve.open()))
        ep = [int(r["epoch"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        for key, lab, c in (("metrics/mAP50(B)", "mAP50", "#7c3aed"), ("metrics/mAP50-95(B)", "mAP50-95", "#0ea5e9"),
                            ("metrics/recall(B)", "recall", "#f59e0b")):
            ax.plot(ep, [100 * float(r[key]) for r in rows], marker="o", ms=3, label=lab, color=c)
        if m.get("stages") and len(m["stages"]) > 1:
            ax.axvline(m["stages"][0]["epochs"] + 0.5, color="#999", ls="--", lw=1)
            ax.text(m["stages"][0]["epochs"] + 0.7, 8, "relabelled MiniWoB links", fontsize=7, color="#666")
        ax.set(xlabel="epoch", ylabel="%", title="Detector training (val split, 640 px)", ylim=(0, 100))
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / "detector_training.png", dpi=150)
        plt.close(fig)
        md.append("\n![Detector training](detector_training.png)\n")
    first = next(iter(m["metrics"].values()))["val"]["per_class_mAP50-95"]
    md += ["", "Per-class mAP50-95 (val): " + ", ".join(f"{k} {pct(v)}" for k, v in first.items()), ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenspot")
    ap.add_argument("--miniwob")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ["## Results\n", detector_section()]
    ss = RESULTS / "screenspot" / a.screenspot if a.screenspot else latest("screenspot")
    mw = RESULTS / "miniwob" / a.miniwob if a.miniwob else latest("miniwob")
    if ss:
        parts.append(screenspot_section(ss))
    if mw:
        parts.append(miniwob_section(mw))
    (OUT / "REPORT.md").write_text("\n\n".join(p for p in parts if p))
    print(f"wrote {OUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
