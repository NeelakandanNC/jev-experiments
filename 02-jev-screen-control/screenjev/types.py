"""Shared types: what a detector returns and what the agent acts on."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Element classes the YOLO detector is trained on (order = YOLO class ids; append only).
CLASSES = [
    "button",      # text button (incl. submit / FAB with text)
    "link",        # hyperlink text
    "text_input",  # text field, search box, textarea, password, date
    "checkbox",
    "radio",
    "toggle",      # switch
    "dropdown",    # <select>, combobox
    "slider",
    "tab",         # tab / segmented control / bottom-nav item
    "icon",        # icon-only clickable that isn't one of the named ones below
    "back",        # back / previous arrow
    "close",       # close / dismiss (x)
    "menu",        # hamburger / kebab / more
    "search",      # magnifier icon button
    "scrollbar",   # scrollbar track of a scrollable region
]
# Not detected by YOLO: OCR text that isn't inside any detected element becomes a "text" element.
TEXT = "text"


@dataclass
class Element:
    id: str                                   # "e0", "e1", ... in reading order
    cls: str                                  # one of CLASSES or TEXT
    box: tuple[float, float, float, float]    # x1, y1, x2, y2 in screenshot pixels
    text: str = ""                            # OCR text inside the element
    label: str = ""                           # nearby text naming it (field label, checkbox label)
    score: float = 1.0                        # detector confidence
    source: str = "yolo"                      # yolo | ocr | dom | omniparser
    parts: list[tuple[str, tuple[float, float, float, float]]] = field(default_factory=list)  # OCR words (text runs)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.box
        return x1 <= x <= x2 and y1 <= y <= y2

    def point_for(self, target: str) -> tuple[float, float]:
        """Where to click. For a run of OCR text, the words the target quotes (`the link "in."`), if present."""
        if self.parts and target:
            norm = lambda t: re.sub(r"[^a-z0-9]+", "", t.lower())
            for q in re.findall(r'"([^"]+)"|\'([^\']+)\'', target):
                want = norm(q[0] or q[1])
                if not want:
                    continue
                words = [norm(w) for w, _ in self.parts]
                for i in range(len(words)):
                    acc = ""
                    for j in range(i, len(words)):
                        acc += words[j]
                        if acc == want:
                            boxes = [b for _, b in self.parts[i:j + 1]]
                            return ((min(b[0] for b in boxes) + max(b[2] for b in boxes)) / 2,
                                    (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2)
                        if not want.startswith(acc):
                            break
        return self.center

    def describe(self, width: int, height: int) -> str:
        """One line the decision model reads: kind, text, label, where it is."""
        cx, cy = self.center
        parts = [self.cls.replace("_", " ")]
        if self.text:
            parts.append(f'"{_clip(self.text, 80)}"')
        if self.label and self.label != self.text:
            parts.append(f'labeled "{_clip(self.label, 60)}"')
        parts.append(f"at {where(cx / width, cy / height)} ({100 * cx / width:.0f}%, {100 * cy / height:.0f}%)")
        return " ".join(parts)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "cls": self.cls, "box": [round(v, 1) for v in self.box], "text": self.text,
                "label": self.label, "score": round(self.score, 3), "source": self.source}


@dataclass
class Screen:
    width: int
    height: int
    elements: list[Element] = field(default_factory=list)

    def by_id(self, eid: str) -> Element | None:
        return next((e for e in self.elements if e.id == eid), None)


def where(fx: float, fy: float) -> str:
    v = "top" if fy < 1 / 3 else "bottom" if fy > 2 / 3 else "middle"
    h = "left" if fx < 1 / 3 else "right" if fx > 2 / 3 else "center"
    return "center" if (v, h) == ("middle", "center") else f"{v}-{h}"


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def reading_order(elements: list[Element], row_tol: float = 8.0) -> list[Element]:
    """Sort top-to-bottom, then left-to-right within a row, and renumber ids e0, e1, ..."""
    rows: list[list[Element]] = []
    for e in sorted(elements, key=lambda e: e.center[1]):
        if rows and abs(rows[-1][0].center[1] - e.center[1]) <= row_tol:
            rows[-1].append(e)
        else:
            rows.append([e])
    out = [e for row in rows for e in sorted(row, key=lambda e: e.box[0])]
    for i, e in enumerate(out):
        e.id = f"e{i}"
    return out
