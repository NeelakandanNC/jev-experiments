"""An Android phone or emulator over adb (USB debugging on, `adb devices` lists it)."""

from __future__ import annotations

import asyncio
import io

from PIL import Image

from .base import Device

KEYCODES = {"enter": 66, "return": 66, "back": 4, "home": 3, "tab": 61, "backspace": 67, "del": 67,
            "delete": 112, "escape": 111, "esc": 111, "menu": 82, "search": 84, "up": 19, "down": 20,
            "left": 21, "right": 22, "app_switch": 187}


class AndroidDevice(Device):
    name = "android"

    def __init__(self, serial: str | None = None, adb: str = "adb"):
        self.base = [adb] + (["-s", serial] if serial else [])

    async def _adb(self, *args: str) -> bytes:
        proc = await asyncio.create_subprocess_exec(*self.base, *args, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"adb {' '.join(args)}: {err.decode(errors='replace').strip()}")
        return out

    async def screenshot(self) -> Image.Image:
        return Image.open(io.BytesIO(await self._adb("exec-out", "screencap", "-p"))).convert("RGB")

    async def click(self, x, y, modifiers: str = ""):  # no modifier keys on touch screens
        await self._adb("shell", "input", "tap", str(int(x)), str(int(y)))

    async def type_text(self, text):
        # `input text` needs spaces as %s and shell metacharacters escaped
        esc = "".join("%s" if c == " " else "\\" + c if c in "()<>|;&*~\"'`$\\?#!" else c for c in text)
        await self._adb("shell", "input", "text", esc)

    async def key(self, key):
        k = key.lower()
        if k in ("ctrl+a",):  # select-all: move to end, then delete back is done by clear_field
            return await self._adb("shell", "input", "keycombination", "113", "29")
        code = KEYCODES.get(k)
        if code is None:
            raise ValueError(f"no Android keycode for {key!r}")
        await self._adb("shell", "input", "keyevent", str(code))

    async def clear_field(self):
        await self._adb("shell", "input", "keycombination", "113", "29")  # ctrl+a (Android 12+)
        await self._adb("shell", "input", "keyevent", "67")

    async def scroll(self, x, y, dy):
        y2 = max(1, y - dy)
        await self._adb("shell", "input", "swipe", str(int(x)), str(int(y)), str(int(x)), str(int(y2)), "300")

    async def back(self):
        await self._adb("shell", "input", "keyevent", "4")
