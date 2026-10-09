"""Shared benchmark plumbing: .env loading, variants, result files, calibration metrics."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load_env() -> None:
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())


def parse_variant(v: str) -> tuple[str, str]:
    """'yolo+jev' -> ('yolo', 'jev'); 'llm-coords' -> ('none', 'llm-coords')."""
    if "+" not in v:
        return "none", v
    det, sel = v.split("+", 1)
    return det, sel


def run_dir(kind: str, name: str | None) -> Path:
    d = RESULTS / kind / (name or time.strftime("%Y%m%d-%H%M%S"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    if not conf:
        return float("nan")
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum(abs(c[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(c))


def reliability(conf: list[float], correct: list[bool], bins: int = 10) -> list[tuple[float, float, int]]:
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return [(float(c[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum())) for b in range(bins) if (idx == b).any()]


def pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f}%"
