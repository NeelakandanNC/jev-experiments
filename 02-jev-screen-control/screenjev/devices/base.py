"""A Device is a screen the agent can see and touch. Coordinates are always screenshot pixels."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class Device(ABC):
    name = "device"
    note = ""  # set by an action when the screen won't show what happened (read and cleared by the agent)

    @abstractmethod
    async def screenshot(self) -> Image.Image: ...

    @abstractmethod
    async def click(self, x: float, y: float) -> None: ...

    async def double_click(self, x: float, y: float) -> None:
        await self.click(x, y)
        await self.click(x, y)

    @abstractmethod
    async def type_text(self, text: str) -> None: ...

    @abstractmethod
    async def key(self, key: str) -> None:
        """Enter, Tab, Escape, Backspace, ctrl+a, ..."""

    @abstractmethod
    async def scroll(self, x: float, y: float, dy: float) -> None:
        """Scroll the region under (x, y) by dy screenshot pixels (positive = down)."""

    async def back(self) -> None:
        await self.key("alt+Left")

    async def clear_field(self) -> None:
        await self.key("ctrl+a")
        await self.key("Backspace")

    async def select_option(self, x: float, y: float, option: str) -> bool:
        """Pick `option` in the dropdown at (x, y) if the platform renders it off-screen. False = not handled."""
        return False

    async def close(self) -> None:
        pass
