"""Routing metrics: recall@k, MRR, context tokens, latency, cost, calibration."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def hit_at_k(order: list[str], gold: set[str], k: int) -> bool:
    """At least one gold tool is in the top k (enough to take the next step)."""
    return bool(gold & set(order[:k]))


def recall_at_k(order: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold tools in the top k (all needed for a whole task)."""
    return len(gold & set(order[:k])) / len(gold) if gold else 0.0


def mrr(order: list[str], gold: set[str]) -> float:
    for i, n in enumerate(order):
        if n in gold:
            return 1.0 / (i + 1)
    return 0.0


def ece(confidences: Iterable[float], correct: Iterable[bool], bins: int = 10) -> tuple[float, list[dict[str, Any]]]:
    """Expected calibration error of top-1 confidence, plus the reliability-diagram table."""
    c = np.asarray(list(confidences), dtype=float)
    y = np.asarray(list(correct), dtype=float)
    if len(c) == 0:
        return float("nan"), []
    edges = np.linspace(0, 1, bins + 1)
    total, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
        if not m.any():
            continue
        conf, acc = c[m].mean(), y[m].mean()
        total += m.sum() / len(c) * abs(conf - acc)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "confidence": float(conf), "accuracy": float(acc)})
    return float(total), table


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summarize(rows: list[dict[str, Any]], ks: list[int], mode: str) -> dict[str, Any]:
    """Aggregate per-case rows of one router into the numbers that go in the results table.

    mode "step": gold is the tool(s) actually called at that step -> hit@k is the headline.
    mode "task": gold is every tool the task needs -> recall@k is the headline.
    """
    ok = [r for r in rows if not r.get("error")]
    out: dict[str, Any] = {"cases": len(rows), "errors": len(rows) - len(ok)}
    if not ok:
        return out
    for k in ks:
        out[f"hit@{k}"] = float(np.mean([hit_at_k(r["order"], set(r["gold"]), k) for r in ok]))
        out[f"recall@{k}"] = float(np.mean([recall_at_k(r["order"], set(r["gold"]), k) for r in ok]))
        out[f"tokens@{k}"] = float(np.mean([r["tokens_at"][str(k)] for r in ok]))
    out["mrr"] = float(np.mean([mrr(r["order"], set(r["gold"])) for r in ok]))
    out["full_catalog_tokens"] = float(np.mean([r["full_tokens"] for r in ok]))
    lat = [r["latency_s"] for r in ok]
    out["latency_p50_ms"] = pct(lat, 50) * 1000
    out["latency_p95_ms"] = pct(lat, 95) * 1000
    out["cost_per_call_usd"] = float(np.mean([r["cost_usd"] for r in ok]))
    out["router_tokens_mean"] = float(np.mean([r["router_tokens"] for r in ok]))
    if "adaptive" in ok[0]:
        out["adaptive_size_mean"] = float(np.mean([len(r["adaptive"]) for r in ok]))
        out["adaptive_hit"] = float(np.mean([bool(set(r["adaptive"]) & set(r["gold"])) for r in ok]))
        out["adaptive_tokens"] = float(np.mean([r["adaptive_tokens"] for r in ok]))
    conf = [(r["confidence"], r["order"][0] in set(r["gold"])) for r in ok if r.get("confidence") is not None]
    if conf:
        e, table = ece([c for c, _ in conf], [y for _, y in conf])
        out["ece"] = e
        out["reliability"] = table
        out["top1_accuracy"] = float(np.mean([y for _, y in conf]))
        out["mean_confidence"] = float(np.mean([c for c, _ in conf]))
    out["mode"] = mode
    out["mock"] = any(r.get("mock") for r in ok)
    return out
