"""Run the agent on a real screen.

    python -m screenjev --device browser --url https://example.com --task "Open the pricing page"
    python -m screenjev --device desktop --task "Open Settings and turn on dark mode" --selector jev
    python -m screenjev --device android --serial emulator-5554 --task "Turn on Wi-Fi"

Planner LLM: --planner (default gpt-6-luna). Grounding: --selector jev | luna | llm | som.
Detector: --detector yolo (ours) | omniparser | dom (browser only, oracle). Traces go to runs/<time>/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from .agent import Agent, AgentConfig
from .detect import Perception, make_detector
from .devices import make_device
from .llm import DEFAULT_PLANNER, LLM
from .planner import Planner, mock_plan
from .selector import make_selector
from .trace import Trace


def load_env(path: Path) -> None:
    import os
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


async def main_async(a) -> int:
    kw = {}
    if a.device == "browser":
        kw = {"url": a.url, "width": a.width, "height": a.height, "scale": a.scale, "headless": not a.show,
              "mobile": a.mobile}
    elif a.device == "android":
        kw = {"serial": a.serial}
    device = await make_device(a.device, **kw)
    llm = LLM(a.planner)
    if llm.mock:
        llm.scripted = mock_plan
    planner = Planner(llm, device=a.device, som=a.selector == "som")
    selector = make_selector(a.selector)
    perception = Perception(make_detector(a.detector))
    out = Path(a.trace or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    agent = Agent(device, perception, planner, selector,
                  AgentConfig(max_steps=a.max_steps, min_confidence=a.min_confidence), Trace(out))
    try:
        res = await agent.run(a.task)
    finally:
        await selector.aclose()
        await llm.aclose()
        await device.close()
    for s in res.steps:
        print(f"{s.i + 1:>2}. {s.step.get('action'):<12} {s.step.get('target', '')[:50]:<50} -> {s.outcome[:90]}")
    print(json.dumps({k: v for k, v in res.to_json().items() if k != "steps"}, indent=1))
    print(f"trace: {out / 'index.html'}")
    return 0 if res.status == "done" else 1


def main():
    load_env(Path(__file__).resolve().parent.parent / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--device", default="browser", choices=["browser", "desktop", "android"])
    ap.add_argument("--url")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--mobile", action="store_true", help="browser: phone viewport + touch")
    ap.add_argument("--show", action="store_true", help="browser: headed window")
    ap.add_argument("--serial", help="android: adb serial")
    ap.add_argument("--planner", default=DEFAULT_PLANNER)
    ap.add_argument("--selector", default="jev")
    ap.add_argument("--detector", default="yolo")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    ap.add_argument("--trace")
    raise SystemExit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
