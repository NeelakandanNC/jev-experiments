"""Builds the YOLO dataset: screenshots auto-labelled from the DOM (screenjev/detect/label.js).

    python -m bench.make_dataset --synth 3000 --miniwob-seeds 12

Sources
  synth     random UI pages from bench/synth.py (exact labels via data-ui)
  miniwob   MiniWoB++ pages, excluding every task in the evaluation suite (and its variants),
            so the detector never sees the eval tasks' layouts
Splits: train / val by page seed. A separate `test_miniwob_suite` split renders the eval-suite
tasks (seeds 10_000+, which the benchmark doesn't use) to measure detection on them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import tempfile
from pathlib import Path

from screenjev.devices.browser import BrowserDevice, launch
from screenjev.types import CLASSES

from . import synth
from .miniwob_env import SUITE, MiniWoBEnv, detector_train_tasks

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yolo"
CID = {c: i for i, c in enumerate(CLASSES)}


def write_sample(split: str, name: str, img, labels: list[dict]) -> int:
    (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    W, H = img.size
    lines = []
    for lab in labels:
        if lab["cls"] not in CID:
            continue
        x1, y1, x2, y2 = lab["box"]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(f"{CID[lab['cls']]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    img.save(OUT / "images" / split / f"{name}.jpg", quality=93)
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
    return len(lines)


async def render_synth(seeds: list[int], split: str, workers: int, stats: dict) -> None:
    from playwright.async_api import async_playwright
    tmp = Path(tempfile.mkdtemp(prefix="synth-"))
    queue: asyncio.Queue[int] = asyncio.Queue()
    for s in seeds:
        queue.put_nowait(s)

    async with async_playwright() as pw:
        browser = await launch(pw)

        async def worker():
            while not queue.empty():
                seed = queue.get_nowait()
                page_html, opt = synth.make_page(seed)
                path = tmp / f"{seed}.html"
                path.write_text(page_html)
                ctx = await browser.new_context(viewport={"width": opt["width"], "height": opt["height"]},
                                                device_scale_factor=opt["scale"], is_mobile=opt["mobile"],
                                                has_touch=opt["mobile"])
                try:
                    page = await ctx.new_page()
                    await page.goto(path.as_uri(), wait_until="load", timeout=30000)
                    if opt["scroll"]:
                        await page.evaluate(f"window.scrollTo(0, {opt['scroll']})")
                    await page.wait_for_timeout(50)
                    dev = BrowserDevice(page, opt["scale"])
                    labels = await dev.dom_elements()
                    img = await dev.screenshot()
                    n = write_sample(split, f"synth_{seed}", img, labels)
                    stats["synth"] += 1
                    stats["boxes"] += n
                except Exception as e:  # a broken random page shouldn't kill the run
                    stats["errors"].append(f"synth {seed}: {e}")
                finally:
                    await ctx.close()
                if stats["synth"] % 200 == 0:
                    print(f"  synth {stats['synth']} pages, {stats['boxes']} boxes", flush=True)

        await asyncio.gather(*(worker() for _ in range(workers)))
        await browser.close()
    shutil.rmtree(tmp, ignore_errors=True)


async def render_miniwob(tasks: list[str], seeds: list[int], split: str, stats: dict, workers: int = 4) -> None:
    """Each (task, seed) is rendered at 3x, and about half of them again at 2x. Randomness is per sample,
    so the output doesn't depend on the number of workers."""
    async def worker(scale: float, queue: asyncio.Queue):
        async with MiniWoBEnv(scale=scale) as env:
            while not queue.empty():
                task, seed = queue.get_nowait()
                rng = random.Random(f"{task}/{seed}/{scale}")
                try:
                    await env.reset(task, seed)
                    # random interactions, so opened menus / expanded sections / dialogs get labelled too
                    for _ in range(rng.choice([0, 0, 1, 2])):
                        els = await env.device.dom_elements()
                        clickable = [e for e in els if e["cls"] not in ("scrollbar", "text_input")]
                        if not clickable:
                            break
                        x1, y1, x2, y2 = rng.choice(clickable)["box"]
                        await env.device.click((x1 + x2) / 2, (y1 + y2) / 2)
                        await asyncio.sleep(0.1)
                        if (await env.result())[0]:  # the click ended the episode: START cover is up
                            await env.reset(task, seed)
                            break
                    labels = await env.device.dom_elements()
                    img = await env.device.screenshot()
                    stats["boxes"] += write_sample(split, f"mw_{task}_{seed}_{int(scale)}x", img, labels)
                    stats["miniwob"] += 1
                except Exception as e:
                    stats["errors"].append(f"miniwob {task} {seed}: {e}")
                if stats["miniwob"] % 200 == 0:
                    print(f"  miniwob {stats['miniwob']} pages", flush=True)

    for scale in (3.0, 2.0):
        queue: asyncio.Queue = asyncio.Queue()
        for t in tasks:
            for sd in seeds:
                if scale == 3.0 or random.Random(f"{t}/{sd}/2x").random() < 0.5:
                    queue.put_nowait((t, sd))
        await asyncio.gather(*(worker(scale, queue) for _ in range(workers)))


async def main_async(a) -> None:
    global OUT
    OUT = Path(a.out).resolve()
    if a.clean and OUT.exists():
        shutil.rmtree(OUT)
    stats = {"synth": 0, "miniwob": 0, "boxes": 0, "errors": []}
    n_val = max(1, int(a.synth * a.val_frac))
    if a.only == "miniwob":  # relabel MiniWoB pages only (e.g. after a label.js change); synthetic pages stay
        for f in list(OUT.glob("images/*/mw_*")) + list(OUT.glob("labels/*/mw_*")):
            f.unlink()
        for f in OUT.glob("labels/*.cache"):
            f.unlink()
    else:
        await render_synth(list(range(a.synth - n_val)), "train", a.workers, stats)
        await render_synth(list(range(1_000_000, 1_000_000 + n_val)), "val", a.workers, stats)
    mw_tasks = detector_train_tasks()
    seeds = list(range(a.miniwob_seeds))
    n_mw_val = max(1, int(len(seeds) * a.val_frac))
    await render_miniwob(mw_tasks, seeds[n_mw_val:], "train", stats, a.workers)
    await render_miniwob(mw_tasks, seeds[:n_mw_val], "val", stats, a.workers)
    await render_miniwob(SUITE, list(range(10_000, 10_000 + a.suite_seeds)), "test_miniwob_suite", stats, a.workers)

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test_miniwob_suite\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    counts = {}
    for split in ("train", "val", "test_miniwob_suite"):
        per = {c: 0 for c in CLASSES}
        files = list((OUT / "labels" / split).glob("*.txt"))
        for f in files:
            for line in f.read_text().splitlines():
                per[CLASSES[int(line.split()[0])]] += 1
        counts[split] = {"images": len(files), "boxes": per}
    meta = {"args": vars(a), "miniwob_train_tasks": mw_tasks, "miniwob_eval_suite_excluded": SUITE,
            "counts": counts, "errors": stats["errors"][:50], "n_errors": len(stats["errors"])}
    (OUT / "dataset.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(counts, indent=1))
    print(f"{len(stats['errors'])} render errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=int, default=3000)
    ap.add_argument("--miniwob-seeds", type=int, default=12)
    ap.add_argument("--suite-seeds", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", choices=["miniwob"], help="re-render only these pages")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
