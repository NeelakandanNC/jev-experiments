"""OCR with RapidOCR (PP-OCRv4 ONNX models ship inside the wheel, CPU only). Returns word boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Word:
    text: str
    box: tuple[float, float, float, float]
    line: int      # OCR line it came from
    score: float

    @property
    def center(self):
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


class Ocr:
    def __init__(self, min_score: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.min_score = min_score

    def words(self, image: Image.Image) -> list[Word]:
        """Word boxes. The bundled (Chinese/English) recognizer often drops the spaces between English
        words ("Signin"), so word boundaries come from blank pixel columns in each line crop, and
        characters are assigned to those segments by their CTC positions."""
        rgb = np.asarray(image.convert("RGB"))
        result, _ = self.engine(rgb[:, :, ::-1], use_cls=False, return_word_box=True)
        gray = rgb.mean(axis=2)
        out: list[Word] = []
        for li, r in enumerate(result or []):
            quad, text, score = r[0], r[1], float(r[2])
            chars = r[3] if len(r) > 3 else None
            if score < self.min_score or not text.strip():
                continue
            xs, ys = [p[0] for p in quad], [p[1] for p in quad]
            box = (min(xs), min(ys), max(xs), max(ys))
            words = _gap_words(gray, text, box, chars, li, score)
            out += words if words else split_words(text, box, li, score)
        return out


def _segments(gray: np.ndarray, box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """x-ranges of ink in a text line, split at gaps wider than ~a quarter em."""
    H, W = gray.shape
    x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(W, box[2])), int(min(H, box[3]))
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[1] < 4:
        return [(box[0], box[2])]
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    mask = np.abs(crop - np.median(border)) > 40
    mask = mask[[_longest_run(row) < 0.5 * len(row) for row in mask]]  # drop underlines / rules
    if mask.size == 0:
        return [(box[0], box[2])]
    ink = mask.any(axis=0)
    gap = max(2, int(0.2 * (y2 - y1)))
    segs, start, blank = [], None, 0
    for i, on in enumerate(ink):
        if on:
            if start is None:
                start = i
            elif blank >= gap:
                segs.append((start, i - blank))
                start = i
            blank = 0
        elif start is not None:
            blank += 1
    if start is not None:
        segs.append((start, len(ink) - blank))
    return [(x1 + a, x1 + b) for a, b in segs if b > a] or [(box[0], box[2])]


def _longest_run(row: np.ndarray) -> int:
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _gap_words(gray, text, box, chars, line, score) -> list[Word]:
    if not chars:
        return []
    centers = [(min(p[0] for p in q) + max(p[0] for p in q)) / 2 for q in chars]
    if len(centers) == len(text):
        pairs = list(zip(text, centers))
    elif len(centers) == len(text.replace(" ", "")):
        it = iter(centers)
        pairs = [(c, next(it) if c != " " else None) for c in text]
    else:
        return []
    segs = _segments(gray, box)

    def seg_of(x):
        return min(range(len(segs)), key=lambda i: 0 if segs[i][0] <= x <= segs[i][1] else min(abs(x - segs[i][0]), abs(x - segs[i][1])))

    words, cur, cur_seg = [], "", None
    for c, x in pairs:
        if c == " " or x is None:
            if cur:
                words.append((cur, cur_seg))
            cur, cur_seg = "", None
            continue
        si = seg_of(x)
        if cur and si != cur_seg:
            words.append((cur, cur_seg))
            cur = ""
        cur, cur_seg = cur + c, si
    if cur:
        words.append((cur, cur_seg))
    # merge words sharing a segment's box (a space inside one ink run): give each its own share
    out = []
    for w, si in words:
        sx1, sx2 = segs[si]
        out.append(Word(w, (sx1, box[1], sx2, box[3]), line, score))
    by_seg: dict[tuple, list[Word]] = {}
    for w in out:
        by_seg.setdefault(w.box, []).append(w)
    for ws in by_seg.values():
        if len(ws) > 1:
            x1, _, x2, _ = ws[0].box
            n = sum(len(w.text) + 1 for w in ws) - 1
            i = 0
            for w in ws:
                w.box = (x1 + (x2 - x1) * i / n, w.box[1], x1 + (x2 - x1) * (i + len(w.text)) / n, w.box[3])
                i += len(w.text) + 1
    return out


def split_words(text: str, box: tuple[float, float, float, float], line: int, score: float) -> list[Word]:
    x1, y1, x2, y2 = box
    n = len(text)
    if n == 0:
        return []
    per = (x2 - x1) / n
    words, i = [], 0
    for tok in text.split(" "):
        if tok:
            words.append(Word(tok, (x1 + i * per, y1, x1 + (i + len(tok)) * per, y2), line, score))
        i += len(tok) + 1
    return words
