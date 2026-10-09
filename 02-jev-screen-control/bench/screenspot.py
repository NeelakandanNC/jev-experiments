"""ScreenSpot-v2 grounding: given a screenshot and an instruction ("close the dialog"), click the right element.

    python -m bench.screenspot --variants yolo+jev,yolo+luna,yolo+llm,omniparser+jev,omniparser+luna,llm-coords

1,272 instructions over mobile (iOS/Android), desktop (macOS/Windows) and web screenshots, split
into text targets and icon targets. A prediction is a hit when the click point falls inside the
target's box (the standard ScreenSpot metric).

Variants (<detector>+<selector>):
  yolo+jev / yolo+luna   our YOLO + OCR elements, a decision model picks one (choice question)
  yolo+llm               same elements, the chat LLM (gpt-6-luna) picks one, self-reported confidence
  omniparser+*           Microsoft OmniParser v2's YOLO instead of ours (downloads from Hugging Face)
  llm-coords             no detector: the vision LLM answers with x, y directly
Each detector runs once per screenshot; all its selectors see the same element list. Also recorded:
whether any detected element's center lies in the target box (the detector's ceiling).
Data: Hugging Face `HongxinLi/ScreenSpot_v2` (the parquet copy GUI-Actor evaluates on).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict

from screenjev.decide import is_mock
from screenjev.detect import Perception, make_detector
from screenjev.llm import DEFAULT_PLANNER, LLM, image_part
from screenjev.planner import Step
from screenjev.selector import make_selector

from .common import ece, git_rev, load_env, parse_variant, read_jsonl, run_dir

DATASET = "HongxinLi/ScreenSpot_v2"
DOMAIN = {"windows": "desktop", "macos": "desktop", "ios": "mobile", "android": "mobile", "tool": "web",
          "shop": "web", "gitlab": "web", "forum": "web"}

COORD_SYSTEM = """You locate UI elements. Given a screenshot and an instruction, return the point to click to carry
it out, in pixel coordinates of the image you see (origin top-left), and your confidence (0 to 1) that the
point lands on the right element. Reply with JSON only: {"x": <int>, "y": <int>, "confidence": <0..1>}"""


def load_samples(limit: int | None, domains: list[str] | None):
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="test")
    per: dict[str, int] = defaultdict(int)
    for i, ex in enumerate(ds):
        dom = DOMAIN.get(ex["data_source"], ex["data_source"])
        if domains and dom not in domains:
            continue
        if limit and per[dom] >= limit:
            continue
        per[dom] += 1
        img = ex["image"].convert("RGB")
        W, H = img.size
        x1, y1, x2, y2 = [float(v) for v in ex["bbox"]]
        if max(x1, y1, x2, y2) <= 1.0:  # normalized
            x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
        yield {"idx": i, "file": ex["file_name"], "domain": dom, "source": ex["data_source"], "type": ex["data_type"],
               "instruction": ex["instruction"], "gt": [x1, y1, x2, y2], "size": [W, H]}, img


def inside(pt, box) -> bool:
    return pt is not None and box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


async def run(a) -> None:
    out = run_dir("screenspot", a.run)
    variants = a.variants.split(",")
    cfg = {"variants": variants, "limit_per_domain": a.limit, "domains": a.domains, "llm": a.llm, "mock": is_mock(),
           "dataset": DATASET, "git": git_rev(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    path = out / "cases.jsonl"
    done = {(r["variant"], r["idx"]) for r in read_jsonl(path)}

    by_det: dict[str, list[str]] = defaultdict(list)
    for v in variants:
        det, sel = parse_variant(v)
        by_det[det].append(sel)
    perceptions = {d: Perception(make_detector(d)) for d in by_det if d != "none"}
    selectors = {s: make_selector(s) for d, ss in by_det.items() if d != "none" for s in ss}
    coord_llm = LLM(a.llm) if "llm-coords" in variants else None
    if coord_llm and coord_llm.mock:
        coord_llm.scripted = lambda m: {"x": 10, "y": 10, "confidence": 0.5}
    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    n = [0]

    async def write(row):
        async with lock:
            with path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    async def one_select(det, sel_spec, sample, screen, t_det):
        v = f"{det}+{sel_spec}"
        step = Step("click", target=sample["instruction"])
        async with sem:
            try:
                s = await selectors[sel_spec].select(sample["instruction"], step, screen)
                err = s.error
            except Exception as e:
                s, err = None, f"{type(e).__name__}: {e}"
        pt = s.element.point_for(sample["instruction"]) if s and s.element else None
        await write({**sample, "variant": v, "n_elements": len(screen.elements), "t_detect_s": t_det,
                     "reachable": any(inside(e.center, sample["gt"]) for e in screen.elements),
                     "reachable_widget": any(inside(e.center, sample["gt"]) for e in screen.elements if e.cls != "text"),
                     "pred": pt, "pred_element": s.element.to_json() if s and s.element else None,
                     "hit": inside(pt, sample["gt"]), "confidence": s.confidence if s else None,
                     "ranked": s.to_json()["ranked"] if s else [], "latency_s": s.latency_s if s else None,
                     "cost_usd": s.cost_usd if s else 0.0, "tokens": s.tokens if s else 0, "error": err})

    async def coords(sample, img):
        async with sem:
            try:
                part = image_part(img)
                W, H = img.size
                shown = max(W, H)
                k = min(1.0, 1600 / shown)
                r = await coord_llm.json(COORD_SYSTEM, [{"type": "text", "text": f"Image size: {round(W * k)}x{round(H * k)}.\n"
                                                                              f"Instruction: {sample['instruction']}"}, part])
                pt = (float(r.data["x"]) / k, float(r.data["y"]) / k)
                conf = float(r.data.get("confidence")) if r.data.get("confidence") is not None else None
                row = {"pred": pt, "confidence": conf, "latency_s": r.latency_s, "cost_usd": r.cost_usd,
                       "tokens": r.input_tokens + r.output_tokens, "error": ""}
            except Exception as e:
                row = {"pred": None, "confidence": None, "latency_s": None, "cost_usd": 0.0, "tokens": 0,
                       "error": f"{type(e).__name__}: {e}"}
        await write({**sample, "variant": "llm-coords", **row, "hit": inside(row["pred"], sample["gt"])})

    pending: set[asyncio.Task] = set()
    for sample, img in load_samples(a.limit, a.domains.split(",") if a.domains else None):
        for det, sels in by_det.items():
            todo = [s for s in sels if (f"{det}+{s}" if det != "none" else s, sample["idx"]) not in done]
            if not todo:
                continue
            if det == "none":
                pending.add(asyncio.create_task(coords(sample, img)))
                continue
            t0 = time.perf_counter()
            screen = await perceptions[det].screen(img)
            t_det = time.perf_counter() - t0
            for s in todo:
                pending.add(asyncio.create_task(one_select(det, s, sample, screen, t_det)))
        n[0] += 1
        if n[0] % 25 == 0:
            print(f"  {n[0]} screenshots", flush=True)
        if len(pending) > 4 * a.concurrency:  # backpressure: don't run detection far ahead of the API
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    if pending:
        await asyncio.wait(pending)
    for s in selectors.values():
        await s.aclose()
    if coord_llm:
        await coord_llm.aclose()
    summary = summarize(read_jsonl(path))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["overall"], indent=1))


def summarize(rows: list[dict]) -> dict:
    by_v: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)
    overall, breakdown = {}, {}
    for v, rs in sorted(by_v.items()):
        conf = [(r["confidence"], r["hit"]) for r in rs if r.get("confidence") is not None]
        sel = selective(conf)
        overall[v] = {
            "n": len(rs), "accuracy": sum(r["hit"] for r in rs) / len(rs),
            "reachable": (sum(r["reachable"] for r in rs) / len(rs)) if "reachable" in rs[0] else None,
            "ece": ece([c for c, _ in conf], [h for _, h in conf]) if conf else None,
            "mean_confidence": sum(c for c, _ in conf) / len(conf) if conf else None,
            "acc_at_50pct_coverage": sel.get(0.5), "acc_at_80pct_coverage": sel.get(0.8),
            "errors": sum(bool(r.get("error")) for r in rs),
            "cost_per_1k": 1000 * sum(r.get("cost_usd") or 0 for r in rs) / len(rs),
            "latency_p50_s": _median([r["latency_s"] for r in rs if r.get("latency_s") is not None]),
        }
        cells = defaultdict(list)
        for r in rs:
            cells[f"{r['domain']}/{r['type']}"].append(r["hit"])
        breakdown[v] = {k: sum(x) / len(x) for k, x in sorted(cells.items())}
    return {"overall": overall, "by_domain_type": breakdown}


def selective(conf_hit: list[tuple[float, bool]]) -> dict[float, float]:
    """Accuracy on the most-confident X% of cases (what you get by abstaining on the rest)."""
    if not conf_hit:
        return {}
    s = sorted(conf_hit, key=lambda t: -t[0])
    return {cov: sum(h for _, h in s[: max(1, int(cov * len(s)))]) / max(1, int(cov * len(s))) for cov in (0.5, 0.8)}


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def main():
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="yolo+jev,yolo+luna,yolo+llm,omniparser+jev,omniparser+luna,llm-coords")
    ap.add_argument("--limit", type=int, help="max screenshots per domain (mobile / desktop / web)")
    ap.add_argument("--domains", help="comma-separated subset of mobile,desktop,web")
    ap.add_argument("--llm", default=DEFAULT_PLANNER, help="model for llm-coords")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--run", help="results/screenspot/<run> (resumes if it exists)")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
