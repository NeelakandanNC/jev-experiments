"""MiniWoB++ tasks driven through Playwright (instead of the package's Selenium env).

The task area is 160x210 CSS px; screenshots are taken of just that area at `scale`x so text and
widgets are big enough for the detector and OCR. Episodes get no time limit (LLM calls take
seconds); success = the task's raw reward > 0.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import miniwob

from screenjev.devices.browser import BrowserDevice, launch

HTML_DIR = Path(miniwob.__file__).resolve().parent / "html" / "miniwob"

# Evaluated tasks: solvable with click / type / select / scroll / keys (no drag, no canvas reasoning).
SUITE = ["click-button", "click-button-sequence", "click-checkboxes", "click-checkboxes-soft", "click-collapsible",
         "click-collapsible-2", "click-dialog", "click-dialog-2", "click-link", "click-option", "click-scroll-list",
         "click-tab", "click-tab-2", "click-test-2", "click-widget", "enter-password", "enter-text",
         "enter-text-dynamic", "focus-text-2", "login-user", "choose-list", "navigate-tree", "search-engine",
         "social-media", "multi-layouts"]


def all_tasks() -> list[str]:
    return sorted(p.stem for p in HTML_DIR.glob("*.html"))


def detector_train_tasks() -> list[str]:
    """Tasks the detector may train on: everything except the eval suite and its variants (click-tab-2-hard, ...)."""
    return [t for t in all_tasks() if not any(t == s or t.startswith(s + "-") for s in SUITE)
            and not t.startswith("drag-")]


class MiniWoBEnv:
    def __init__(self, scale: float = 3.0, headless: bool = True):
        self.scale = scale
        self.headless = headless
        self.task = None

    async def __aenter__(self):
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        self._browser = await launch(self._pw, self.headless)
        self._ctx = await self._browser.new_context(viewport={"width": 400, "height": 300}, device_scale_factor=self.scale)
        self.page = await self._ctx.new_page()
        self.device = BrowserDevice(self.page, self.scale, clip={"x": 0, "y": 0, "width": 160, "height": 210})
        return self

    async def __aexit__(self, *exc):
        await self._browser.close()
        await self._pw.stop()

    async def reset(self, task: str, seed: int) -> str:
        """Load the task, start an episode with this seed, return the instruction."""
        if task != self.task:
            await self.page.goto((HTML_DIR / f"{task}.html").as_uri())
            await self.page.wait_for_function("typeof core !== 'undefined' && typeof genProblem !== 'undefined'")
            self.task = task
        await self.page.evaluate("""(seed) => {
            Math.seedrandom(seed);
            core.EPISODE_MAX_TIME = 1e9;
            core.startEpisodeReal();
        }""", str(seed))
        for _ in range(40):
            if await self.page.evaluate("WOB_TASK_READY"):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.15)
        return await self.page.evaluate("core.getUtterance()")

    async def result(self) -> tuple[bool, float, str | None]:
        """(done, raw reward, reason)."""
        done, raw, reason = await self.page.evaluate("[WOB_DONE_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON]")
        return bool(done), float(raw), reason
