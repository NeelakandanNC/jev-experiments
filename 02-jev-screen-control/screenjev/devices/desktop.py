"""The local desktop: mss screenshots, pyautogui input. Needs a display (and accessibility permission on macOS)."""

from __future__ import annotations

import asyncio

from PIL import Image

from .base import Device


class DesktopDevice(Device):
    name = "desktop"

    def __init__(self, monitor: int = 1):
        import mss
        import pyautogui
        pyautogui.FAILSAFE = True  # slam the mouse into a corner to abort
        self.pg = pyautogui
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]
        # HiDPI: screenshots are physical pixels, pyautogui works in logical points
        self.scale = self.monitor["width"] / pyautogui.size().width if monitor == 1 else 1.0

    def _pt(self, x, y):
        return self.monitor["left"] + x / self.scale, self.monitor["top"] + y / self.scale

    async def screenshot(self) -> Image.Image:
        shot = await asyncio.to_thread(self.sct.grab, self.monitor)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    async def click(self, x, y, modifiers: str = ""):
        mods = [m.strip().lower() for m in modifiers.replace("-", "+").split("+") if m.strip()]
        mods = ["command" if m in ("cmd", "meta") else m for m in mods]

        def _click():
            for m in mods:
                self.pg.keyDown(m)
            try:
                self.pg.click(*self._pt(x, y))
            finally:
                for m in reversed(mods):
                    self.pg.keyUp(m)
        await asyncio.to_thread(_click)

    async def double_click(self, x, y):
        await asyncio.to_thread(self.pg.doubleClick, *self._pt(x, y))

    async def type_text(self, text):
        await asyncio.to_thread(self.pg.write, text, interval=0.01)

    async def key(self, key):
        keys = [k.strip().lower() for k in key.replace("-", "+").split("+")]
        keys = ["command" if k in ("cmd", "meta") else "enter" if k == "return" else k for k in keys]
        await asyncio.to_thread(self.pg.hotkey, *keys)

    async def scroll(self, x, y, dy):
        await asyncio.to_thread(self.pg.moveTo, *self._pt(x, y))
        await asyncio.to_thread(self.pg.scroll, -int(dy / 40) or (-1 if dy > 0 else 1))
