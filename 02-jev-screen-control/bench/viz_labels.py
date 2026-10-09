"""Draw YOLO labels on dataset images to eyeball label quality: python -m bench.viz_labels <split> <n> <outdir>"""

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from screenjev.types import CLASSES

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
          "#008080", "#9a6324", "#800000", "#808000", "#000075", "#808080"]


def draw(img_path: Path, label_path: Path) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    for line in label_path.read_text().splitlines():
        c, cx, cy, w, h = line.split()
        c = int(c)
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=COLORS[c % len(COLORS)], width=2)
        d.text((cx - w / 2 + 2, cy - h / 2 - 10), CLASSES[c], fill=COLORS[c % len(COLORS)])
    return img


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "data" / "yolo"
    split, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted((root / "images" / split).glob("*.jpg"))
    for p in random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 0).sample(imgs, min(n, len(imgs))):
        draw(p, root / "labels" / split / f"{p.stem}.txt").save(out / f"{p.stem}.png")
