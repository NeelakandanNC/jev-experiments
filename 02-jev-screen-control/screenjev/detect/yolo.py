"""YOLO UI-element detectors (Ultralytics): ours (typed classes) and Microsoft OmniParser v2 (one class)."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from ..types import Element

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS = ROOT / "models" / "screenjev-yolo.pt"


class YoloDetector:
    name = "yolo"

    def __init__(self, weights: str | Path | None = None, conf: float = 0.25, imgsz: int | str = "auto",
                 device: str = "cpu"):
        from ultralytics import YOLO
        weights = Path(weights or os.getenv("SCREENJEV_YOLO_WEIGHTS") or DEFAULT_WEIGHTS)
        if not weights.exists():
            raise FileNotFoundError(f"{weights} not found: train it with `python -m bench.train_detector` "
                                    f"or use --detector omniparser / dom")
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.names = self.model.names

    def size_for(self, image: Image.Image) -> int:
        """auto: 640 for small screens (phone-sized, MiniWoB), 960 for desktop-sized ones. Upscaling a small
        screenshot past its own size hurts (held-out MiniWoB mAP50 70% at 640 vs 65% at 960), while big
        screenshots gain (val mAP50 92.5% -> 94.3%)."""
        if self.imgsz != "auto":
            return int(self.imgsz)
        return 640 if max(image.size) <= 900 else 960

    def boxes(self, image: Image.Image) -> list[Element]:
        r = self.model.predict(image, imgsz=self.size_for(image), conf=self.conf, agnostic_nms=True, iou=0.5,
                               device=self.device, verbose=False, max_det=300)[0]
        out = []
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            out.append(Element("", self.names[int(c)], (x1, y1, x2, y2), score=float(s), source=self.name))
        return out


class OmniParserDetector(YoloDetector):
    """OmniParser v2 icon_detect (YOLOv8, trained on interactable regions). Weights from Hugging Face,
    downloaded on first use (AGPL, like Ultralytics). Its single class becomes "interactable"."""

    name = "omniparser"

    def __init__(self, weights: str | Path | None = None, conf: float = 0.05, imgsz: int = 1280, device: str = "cpu"):
        if weights is None:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download("microsoft/OmniParser-v2.0", "icon_detect/model.pt")
        super().__init__(weights, conf, imgsz, device)

    def boxes(self, image):
        out = super().boxes(image)
        for e in out:
            e.cls = "interactable"
        return out
