"""Like-for-like ScreenSpot comparison: every variant scored on the same instructions.

    python -m bench.same_cases --run openai-full --cases-from yolo+luna,yolo+luna@desc,omniparser+luna,omniparser+luna@desc

The case set is the instructions all --cases-from variants answered (API failures excluded). Writes
results/report/SAME_CASES.md and prints the table.
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict

from .common import RESULTS, ece, pct, read_jsonl
from .report import nice
from .screenspot import _api_error, selective

ORDER = ["llm-coords", "yolo+onejev-vis@desc", "omniparser+onejev@desc", "yolo+onejev@desc", "omniparser+onejev", "yolo+onejev",
         "omniparser+prism@desc", "omniparser+luna@desc", "yolo+prism@desc", "yolo+llm@desc",
         "yolo+luna@desc", "omniparser+prism", "yolo+prism", "yolo+llm", "omniparser+luna", "yolo+luna"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="openai-full")
    ap.add_argument("--cases-from", default="yolo+luna,yolo+luna@desc,omniparser+luna,omniparser+luna@desc")
    a = ap.parse_args()
    rows = read_jsonl(RESULTS / "screenspot" / a.run / "cases.jsonl")
    cases = None
    for v in a.cases_from.split(","):
        ok = {r["idx"] for r in rows if r["variant"] == v and not _api_error(r.get("error") or "")}
        cases = ok if cases is None else cases & ok
    by_v = defaultdict(dict)
    for r in rows:
        if r["idx"] in cases and not _api_error(r.get("error") or ""):
            by_v[r["variant"]][r["idx"]] = r  # last row per (variant, idx)
    variants = [v for v in ORDER if v in by_v] + sorted(v for v in by_v if v not in ORDER)

    lines = [f"Same {len(cases)} ScreenSpot-v2 instructions for every variant (those all of {a.cases_from} answered).\n",
             "| Variant | n | Accuracy | Text targets | Icon targets | ECE | Acc. on most-confident half | Pick latency (p50) |",
             "|---|---|---|---|---|---|---|---|"]
    for v in variants:
        rs = list(by_v[v].values())
        hit = lambda xs: sum(r["hit"] for r in xs) / len(xs) if xs else float("nan")
        text = [r for r in rs if r["type"] == "text"]
        icon = [r for r in rs if r["type"] == "icon"]
        ch = [(r["confidence"], r["hit"]) for r in rs if r.get("confidence") is not None]
        e = ece([c for c, _ in ch], [h for _, h in ch]) if ch else float("nan")
        half = selective(ch).get(0.5, float("nan")) if ch else float("nan")
        lat = [r["latency_s"] for r in rs if r.get("latency_s")]
        lines.append(f"| {nice(v)} | {len(rs)} | **{pct(hit(rs))}** | {pct(hit(text))} | {pct(hit(icon))} | "
                     f"{'—' if e != e else f'{e:.3f}'} | {pct(half)} | {f'{st.median(lat):.2f} s' if lat else '—'} |")
    md = "\n".join(lines) + "\n"
    out = RESULTS / "report" / "SAME_CASES.md"
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
