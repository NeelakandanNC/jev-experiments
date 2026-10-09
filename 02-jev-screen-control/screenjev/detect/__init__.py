"""Perception: detector boxes + OCR words -> a Screen of described elements.

    boxes from the detector (YOLO / OmniParser / DOM) -> each takes the OCR words inside it as its text
    OCR words outside every box -> grouped into "text" elements (also clickable targets)
    fields, checkboxes, sliders, ... -> labelled with the nearest text next to / above them
"""

from __future__ import annotations

import asyncio

from PIL import Image

from ..types import TEXT, Element, Screen, reading_order
from .ocr import Ocr, Word

LABELLED = {"text_input", "dropdown", "checkbox", "radio", "toggle", "slider"}
ICONISH = {"icon", "back", "close", "menu", "search", "interactable", "button", "link", "tab"}  # "near" text if textless
MAX_ELEMENTS = 250  # the Decisions API takes up to 255 choices per question


def make_detector(kind: str, **kw):
    if kind == "yolo":
        from .yolo import YoloDetector
        return YoloDetector(**kw)
    if kind == "omniparser":
        from .yolo import OmniParserDetector
        return OmniParserDetector(**kw)
    if kind == "dom":
        from .dom import DomDetector
        return DomDetector()
    if kind == "none":
        return None
    raise ValueError(f"unknown detector {kind!r} (yolo | omniparser | dom | none)")


class Perception:
    """Detector + OCR. The CPU-bound parts run in a worker thread, one image at a time (shared lock),
    so concurrent episodes keep their LLM / decision calls overlapping without fighting over cores."""

    _lock: asyncio.Lock | None = None

    def __init__(self, detector, ocr: Ocr | None | bool = True):
        self.detector = detector
        self.ocr = Ocr() if ocr is True else (ocr or None)

    def _cpu(self, image: Image.Image, detect: bool) -> tuple[list[Element], list[Word]]:
        boxes = self.detector.boxes(image) if detect else []
        return boxes, (self.ocr.words(image) if self.ocr else [])

    async def screen(self, image: Image.Image, device=None) -> Screen:
        boxes: list[Element] = []
        detect = self.detector is not None and not getattr(self.detector, "needs_device", False)
        if self.detector is not None and not detect:
            boxes = await self.detector.boxes_from(device)
        if Perception._lock is None:
            Perception._lock = asyncio.Lock()
        async with Perception._lock:
            found, words = await asyncio.to_thread(self._cpu, image, detect)
        return merge(boxes + found, words, *image.size)


def merge(boxes: list[Element], words: list[Word], width: int, height: int) -> Screen:
    # smallest box first, so a word in a button inside a card goes to the button
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].area)
    owner: dict[int, int] = {}
    for wi, w in enumerate(words):
        cx, cy = w.center
        for bi in order:
            b = boxes[bi]
            if b.cls != "scrollbar" and b.contains(cx, cy):
                owner[wi] = bi
                break
    for bi, b in enumerate(boxes):
        ws = sorted((words[wi] for wi, o in owner.items() if o == bi), key=lambda w: (w.line, w.box[0]))
        b.parts = [(w.text, w.box) for w in ws]  # lets a click land on the quoted word inside a wide box
        if not b.text:  # the DOM detector already knows its text
            b.text = " ".join(w.text for w in ws)

    texts = _group([w for wi, w in enumerate(words) if wi not in owner])
    for b in boxes:
        if b.cls in LABELLED and not b.label:
            b.label = _label_for(b, texts)
        elif b.cls in ICONISH and not b.text and not b.label:
            b.label = _near_text(b, texts)
    elements = boxes + texts
    if len(elements) > MAX_ELEMENTS:  # keep every detected widget, drop the smallest text runs
        texts = sorted(texts, key=lambda e: -e.area)[: max(0, MAX_ELEMENTS - len(boxes))]
        elements = (boxes + texts)[:MAX_ELEMENTS]
    return Screen(width, height, reading_order(elements))


def _group(words: list[Word]) -> list[Element]:
    """Unclaimed words -> runs of adjacent words on the same OCR line."""
    out: list[Element] = []
    by_line: dict[int, list[Word]] = {}
    for w in words:
        by_line.setdefault(w.line, []).append(w)
    for ws in by_line.values():
        ws.sort(key=lambda w: w.box[0])
        run = [ws[0]]
        for w in ws[1:]:
            h = run[-1].box[3] - run[-1].box[1]
            if w.box[0] - run[-1].box[2] <= 1.2 * h:
                run.append(w)
            else:
                out.append(_text_el(run))
                run = [w]
        out.append(_text_el(run))
    return out


def _text_el(run: list[Word]) -> Element:
    box = (min(w.box[0] for w in run), min(w.box[1] for w in run), max(w.box[2] for w in run), max(w.box[3] for w in run))
    return Element("", TEXT, box, text=" ".join(w.text for w in run), score=min(w.score for w in run), source="ocr",
                   parts=[(w.text, w.box) for w in run])


def _label_for(b: Element, texts: list[Element]) -> str:
    """Checkbox-like: text to the right on the same row. Fields: text to the left on the row, or just above.
    Distances scale with the text's line height (a tall field shouldn't reach far-away text)."""
    x1, y1, x2, y2 = b.box
    h = max(8.0, y2 - y1)
    best, best_d = None, 1e9
    for t in texts:
        tx1, ty1, tx2, ty2 = t.box
        th = max(6.0, ty2 - ty1)
        same_row = min(y2, ty2) - max(y1, ty1) > 0.3 * min(h, th)
        d = 1e9
        if b.cls in ("checkbox", "radio", "toggle"):
            if same_row and -4 <= tx1 - x2 <= 10 * th:
                d = tx1 - x2
            elif same_row and 0 <= x1 - tx2 <= 6 * th:  # label on the left (settings rows)
                d = (x1 - tx2) * 1.5
        else:
            if same_row and 0 <= x1 - tx2 <= 12 * th:
                d = x1 - tx2
            if 0 <= y1 - ty2 <= 1.5 * th and min(x2, tx2) - max(x1, tx1) > -th:
                d = min(d, (y1 - ty2) * 1.2 + abs(tx1 - x1) * 0.2)
        if d < best_d:
            best, best_d = t, d
    return best.text if best else ""


def _near_text(b: Element, texts: list[Element]) -> str:
    """For a textless icon: the closest text run (app-grid caption below, label beside), if it's close."""
    x1, y1, x2, y2 = b.box
    size = max(x2 - x1, y2 - y1)
    best, best_d = None, 1.2 * size + 12
    for t in texts:
        tx1, ty1, tx2, ty2 = t.box
        dx = max(0.0, tx1 - x2, x1 - tx2)
        dy = max(0.0, ty1 - y2, y1 - ty2)
        d = (dx * dx + dy * dy) ** 0.5
        if d < best_d:
            best, best_d = t, d
    return best.text if best else ""
