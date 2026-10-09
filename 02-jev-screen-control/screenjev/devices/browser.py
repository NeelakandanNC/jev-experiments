"""Chromium through Playwright. Also the MiniWoB++ and dataset-rendering backend."""

from __future__ import annotations

import glob
import io
import os
from pathlib import Path

from PIL import Image

from .base import Device

LABEL_JS = (Path(__file__).resolve().parent.parent / "detect" / "label.js").read_text()

_KEYS = {"ctrl": "Control", "control": "Control", "alt": "Alt", "shift": "Shift", "cmd": "Meta", "meta": "Meta",
         "enter": "Enter", "return": "Enter", "tab": "Tab", "esc": "Escape", "escape": "Escape",
         "backspace": "Backspace", "delete": "Delete", "space": "Space", "up": "ArrowUp", "down": "ArrowDown",
         "left": "ArrowLeft", "right": "ArrowRight", "pageup": "PageUp", "pagedown": "PageDown", "home": "Home",
         "end": "End"}


def pw_key(key: str) -> str:
    parts = [p.strip() for p in key.replace("-", "+").split("+") if p.strip()]
    return "+".join(_KEYS.get(p.lower(), p.upper() if len(p) == 1 else p) for p in parts)


def chromium_path() -> str | None:
    """Playwright's bundled browser if present, else CHROMIUM_PATH or a preinstalled /opt/pw-browsers build."""
    if os.getenv("CHROMIUM_PATH"):
        return os.environ["CHROMIUM_PATH"]
    found = sorted(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome"))
    return found[-1] if found else None


async def launch(playwright, headless: bool = True):
    # Playwright hides scrollbars in headless mode; they're one of the things the detector finds.
    kw = {"headless": headless, "ignore_default_args": ["--hide-scrollbars"]}
    try:
        return await playwright.chromium.launch(**kw)
    except Exception:
        path = chromium_path()
        if not path:
            raise
        return await playwright.chromium.launch(executable_path=path, **kw)


class BrowserDevice(Device):
    """A Playwright page. `scale` = device pixel ratio; `clip` = CSS-pixel region to screenshot."""

    name = "browser"

    def __init__(self, page, scale: float = 1.0, clip: dict | None = None):
        self.page = page
        self.scale = scale
        self.clip = clip

    @classmethod
    async def open(cls, url: str | None = None, width: int = 1280, height: int = 800, scale: float = 1.0,
                   headless: bool = True, mobile: bool = False) -> "BrowserDevice":
        from playwright.async_api import async_playwright
        pw = await async_playwright().start()
        browser = await launch(pw, headless)
        ctx = await browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale,
                                        is_mobile=mobile, has_touch=mobile)
        page = await ctx.new_page()
        if url:
            await page.goto(url)
        dev = cls(page, scale)
        dev._owned = (pw, browser)
        return dev

    def to_css(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = (self.clip["x"], self.clip["y"]) if self.clip else (0, 0)
        return x / self.scale + ox, y / self.scale + oy

    def to_image(self, box: list[float]) -> tuple[float, float, float, float]:
        ox, oy = (self.clip["x"], self.clip["y"]) if self.clip else (0, 0)
        x1, y1, x2, y2 = box
        return ((x1 - ox) * self.scale, (y1 - oy) * self.scale, (x2 - ox) * self.scale, (y2 - oy) * self.scale)

    async def screenshot(self) -> Image.Image:
        png = await self.page.screenshot(clip=self.clip, type="png")
        return Image.open(io.BytesIO(png)).convert("RGB")

    async def click(self, x, y):
        await self.page.mouse.click(*self.to_css(x, y))
        await self._close_native_popup()

    async def double_click(self, x, y):
        await self.page.mouse.dblclick(*self.to_css(x, y))
        await self._close_native_popup()

    async def _close_native_popup(self):
        # A clicked <select> opens a native popup that isn't in screenshots and blocks them in headless
        # Chromium. Close it; the select stays focused, and the `select` action picks options by text.
        try:
            if await self.page.evaluate("document.activeElement && document.activeElement.tagName === 'SELECT'"):
                await self.page.keyboard.press("Escape")
                self.note = ("that is a dropdown list whose options open off-screen: use the select action "
                             "with this target and the option's text")
        except Exception:
            pass

    async def type_text(self, text):
        await self.page.keyboard.type(text)

    async def key(self, key):
        await self.page.keyboard.press(pw_key(key))

    async def scroll(self, x, y, dy):
        await self.page.mouse.move(*self.to_css(x, y))
        await self.page.mouse.wheel(0, dy / self.scale)

    async def back(self):
        await self.page.go_back()

    async def select_option(self, x, y, option):
        # Native <select> popups aren't part of the screenshot, so the browser picks the option by text.
        cx, cy = self.to_css(x, y)
        return await self.page.evaluate("""([x, y, want]) => {
            let el = document.elementFromPoint(x, y);
            while (el && el.tagName !== 'SELECT') el = el.parentElement;
            if (!el) return false;
            const norm = s => s.replace(/\\s+/g, ' ').trim().toLowerCase();
            const w = norm(want);
            const opt = [...el.options].find(o => norm(o.text) === w) || [...el.options].find(o => norm(o.text).includes(w));
            if (!opt) return false;
            el.value = opt.value;
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
            return true;
        }""", [cx, cy, option])

    async def dom_elements(self) -> list[dict]:
        """Visible interactive elements from the DOM (label.js), boxes converted to screenshot pixels."""
        raw = await self.page.evaluate(LABEL_JS)
        out = []
        for r in raw:
            x1, y1, x2, y2 = self.to_image(r["box"])
            if self.clip:  # keep what's inside the screenshot
                w, h = self.clip["width"] * self.scale, self.clip["height"] * self.scale
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
            out.append({**r, "box": (x1, y1, x2, y2)})
        return out

    async def close(self):
        owned = getattr(self, "_owned", None)
        if owned:
            pw, browser = owned
            await browser.close()
            await pw.stop()
