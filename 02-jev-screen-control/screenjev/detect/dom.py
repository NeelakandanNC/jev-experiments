"""Oracle detector for the browser: the page's own DOM says where every widget is (label.js).

Used for the detector ablation (how much of the error is the detector's?) and to label training
data. Only available on BrowserDevice; YOLO is the detector that works on any screen.
"""

from __future__ import annotations

from ..types import Element


class DomDetector:
    name = "dom"
    needs_device = True

    async def boxes_from(self, device) -> list[Element]:
        if not hasattr(device, "dom_elements"):
            raise TypeError("the dom detector needs a browser device")
        return [Element("", r["cls"], tuple(r["box"]), text=r.get("text", ""), source="dom")
                for r in await device.dom_elements()]
