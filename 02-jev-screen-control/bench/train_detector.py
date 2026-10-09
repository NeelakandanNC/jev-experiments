"""Train the UI-element YOLO on data/yolo (bench/make_dataset.py), on CPU if that's all there is.

    python -m bench.train_detector --epochs 30 --imgsz 640
    python -m bench.train_detector --eval-only models/screenjev-yolo.pt     # mAP on val + eval-suite pages

Starts from COCO-pretrained YOLO11n. The best checkpoint is copied to models/screenjev-yolo.pt and
its metrics to models/screenjev-yolo.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "yolo" / "data.yaml"
MODELS = ROOT / "models"


def evaluate(weights: Path, imgsz: int, device: str) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out = {}
    for split in ("val", "test"):
        m = model.val(data=str(DATA), split=split, imgsz=imgsz, device=device, verbose=False, plots=False,
                      project=str(ROOT / "data" / "runs"), name=f"val_{split}_{imgsz}", exist_ok=True)
        per = {model.names[int(c)]: round(float(m.box.maps[int(c)]), 4) for c in m.box.ap_class_index}
        out["miniwob_suite" if split == "test" else split] = {
            "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4), "per_class_mAP50-95": per}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "data" / "weights" / "yolo11n.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="use this share of the training images")
    ap.add_argument("--name", default="screenjev")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--eval-only")
    ap.add_argument("--eval-imgsz", default="640,960")
    a = ap.parse_args()

    if a.eval_only:
        res = {f"imgsz{s}": evaluate(Path(a.eval_only), int(s), a.device) for s in a.eval_imgsz.split(",")}
        print(json.dumps(res, indent=1))
        return

    from ultralytics import YOLO
    project = ROOT / "data" / "runs"
    last = project / a.name / "weights" / "last.pt"
    t0 = time.time()
    if a.resume and last.exists():
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(a.model).train(
            data=str(DATA), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, workers=a.workers,
            project=str(project), name=a.name, exist_ok=True, fraction=a.fraction, cache="ram", plots=True,
            # screenshots: no flips (text / arrows have direction), mild scale, no mosaic at the end
            fliplr=0.0, flipud=0.0, degrees=0.0, scale=0.3, translate=0.05, mosaic=1.0, close_mosaic=5,
            hsv_h=0.1, hsv_s=0.5, hsv_v=0.3, patience=50)
    hours = (time.time() - t0) / 3600
    best = project / a.name / "weights" / "best.pt"
    MODELS.mkdir(exist_ok=True)
    shutil.copy(best, MODELS / "screenjev-yolo.pt")
    res = {f"imgsz{s}": evaluate(best, int(s), a.device) for s in a.eval_imgsz.split(",")}
    meta = {"base": Path(a.model).name, "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
            "device": a.device, "train_hours": round(hours, 2),
            "dataset": json.loads((ROOT / "data" / "yolo" / "dataset.json").read_text())["counts"], "metrics": res}
    (MODELS / "screenjev-yolo.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
