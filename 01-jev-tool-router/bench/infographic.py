"""One-image summary of the results, as SVG (vector) + PNG for X.

    python -m bench.infographic      -> results/comparison/jev-vs-openai.{svg,png}

Every number is read from results/comparison/comparison.json (run bench.compare first).
Colors: Jev = categorical slot 1 (#2a78d6); OpenAI = de-emphasized gray (#8a8983, >= 3:1 on
the surface); text in text tokens. Both series are direct-labeled and in the legend.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from .datasets import ROOT

OUT = ROOT / "results" / "comparison"
W, H = 1600, 900
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7974", "#e6e5e0"
JEV, OAI = "#2a78d6", "#8a8983"
FONT = "Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"


def t(x, y, text, size=16, fill=INK, weight=400, anchor="start", extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" {extra}>{escape(str(text))}</text>')


def bar(x, y, w, h, fill, r=4):
    """Horizontal bar: square at the baseline, rounded data-end."""
    w = max(w, r)
    return (f'<path d="M{x},{y} h{w - r} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r} a{r},{r} 0 0 1 -{r},{r} h-{w - r} z" '
            f'fill="{fill}"/>')


def build(data: dict) -> str:
    A, B, U = data["atlas"]["routers"], data["mcpbench"]["routers"], data["universe"]["routers"]
    jev, oai = "jev", "gpt-6-luna"
    rows = [  # (benchmark, detail, metric key, k)
        ("MCP-Atlas", "top 5 · exact tool", A, "hit", 5),
        ("MCP-Atlas", "top 10 · exact tool", A, "hit", 10),
        ("MCP-Bench", "top 10 · needed tools", B, "recall", 10),
        ("MCP-Universe", "top 5 · needed server", U, "hit", 5),
        ("MCP-Universe", "top 10 · needed server", U, "hit", 10),
    ]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="{FONT}">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']

    # header
    s.append(t(64, 84, "Let a decision model pick your agent's MCP tools", 42, INK, 700))
    s.append(t(64, 122, "Jev vs OpenAI's decision model, each handing the agent its 5 or 10 best tools out of "
                        "88–257, on three public MCP benchmarks", 20, INK2))
    lx = 64
    for color, label in ((JEV, "Jev (TypeSafe, RLCD)"), (OAI, "OpenAI gpt-6-luna")):
        s.append(f'<rect x="{lx}" y="150" width="16" height="16" rx="3" fill="{color}"/>')
        s.append(t(lx + 24, 164, label, 16, INK2, 500))
        lx += 250

    # left: grouped horizontal bars
    x0, xmax, top, group, bh, gap = 320, 920, 256, 82, 26, 5
    s.append(t(64, 228, "Is the right tool in the shortlist?", 22, INK, 700))
    s.append(t(64 + 362, 228, "% of routing decisions", 16, MUTED))
    plot_bottom = top + group * len(rows) - 18
    for p in (0, 25, 50, 75, 100):
        x = x0 + (xmax - x0) * p / 100
        s.append(f'<line x1="{x}" y1="{top - 8}" x2="{x}" y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>')
        s.append(t(x, plot_bottom + 22, f"{p}%", 13, MUTED, anchor="middle"))
    for i, (name, detail, R, key, k) in enumerate(rows):
        y = top + i * group
        s.append(t(64, y + 22, name, 18, INK, 700))
        s.append(t(64, y + 44, detail, 14, INK2))
        for j, (m, color) in enumerate(((jev, JEV), (oai, OAI))):
            v = R[m][f"{key}@{k}"]
            by = y + j * (bh + gap)
            w = (xmax - x0) * v
            s.append(bar(x0, by, w, bh, color))
            s.append(t(x0 + w + 10, by + 19, f"{v * 100:.1f}%", 16, INK if j == 0 else INK2, 700 if j == 0 else 500))

    # right: calibration on MCP-Atlas
    px, py, ps = 1100, 300, 362
    s.append(t(1030, 228, "Does its confidence mean anything?", 22, INK, 700))
    jb, ob = A[jev]["reliability"][-1], A[oai]["reliability"][-1]
    for i, (m, color, b) in enumerate(((jev, JEV, jb), (oai, OAI, ob))):
        ky = 254 + i * 22
        name = "Jev" if m == jev else "OpenAI"
        s.append(f'<circle cx="1036" cy="{ky - 5}" r="5.5" fill="{color}"/>')
        s.append(t(1050, ky, f"{name}: {b['confidence'] * 100:.0f}% sure → right {b['accuracy'] * 100:.0f}% of the time",
                   15, INK, 600))
        s.append(t(1050 + 392, ky, f"ECE {A[m]['ece']:.2f}", 14, INK2, anchor="start"))
    s.append(f'<rect x="{px}" y="{py}" width="{ps}" height="{ps}" fill="none" stroke="{GRID}"/>')
    for p in (0.25, 0.5, 0.75):
        s.append(f'<line x1="{px + ps * p}" y1="{py}" x2="{px + ps * p}" y2="{py + ps}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<line x1="{px}" y1="{py + ps * (1 - p)}" x2="{px + ps}" y2="{py + ps * (1 - p)}" stroke="{GRID}" stroke-width="1"/>')
    for p in (0, 0.5, 1):
        s.append(t(px + ps * p, py + ps + 20, f"{int(p * 100)}%", 13, MUTED, anchor="middle"))
        s.append(t(px - 8, py + ps * (1 - p) + 4, f"{int(p * 100)}%", 13, MUTED, anchor="end"))
    s.append(f'<line x1="{px}" y1="{py + ps}" x2="{px + ps}" y2="{py}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="6 6"/>')
    s.append(t(px + 150, py + ps - 166, "perfect calibration", 12, MUTED,
               extra=f'transform="rotate(-45 {px + 150} {py + ps - 166})"'))
    s.append(t(px + ps / 2, py + ps + 42, "confidence in its top pick · MCP-Atlas", 14, INK2, anchor="middle"))
    s.append(t(px - 44, py + ps / 2, "top pick right", 14, INK2, anchor="middle",
               extra=f'transform="rotate(-90 {px - 44} {py + ps / 2})"'))

    def xy(c, a):
        return px + ps * c, py + ps * (1 - a)

    for m, color in ((oai, OAI), (jev, JEV)):
        pts = [xy(b["confidence"], b["accuracy"]) for b in A[m]["reliability"]]
        s.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" '
                 f'stroke-width="2.5" stroke-linejoin="round"/>')
        for x, y in pts:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>')
        ex, ey = pts[-1]
        b = A[m]["reliability"][-1]
        s.append(t(px + ps + 10, ey + 5, f"{b['accuracy'] * 100:.0f}%", 15, INK, 700))

    # stat strip
    atlas = data["atlas"]
    tok_full, tok_k = atlas["catalog_tokens"], A[jev]["tokens@10"]
    costs = [d["routers"][oai]["cost_per_call_usd"] / d["routers"][jev]["cost_per_call_usd"] for d in data.values()]
    tiles = [
        (f"−{(1 - tok_k / tok_full) * 100:.0f}%", "tool context per turn",
         f"{tok_full:,} → {tok_k:,.0f} tokens on MCP-Atlas (top 10)"),
        (f"{A[oai]['ece'] / A[jev]['ece']:.0f}× lower", "calibration error",
         f"ECE {A[jev]['ece']:.2f} vs {A[oai]['ece']:.2f} on MCP-Atlas"),
        (f"{min(costs):.1f}–{max(costs):.1f}× cheaper", "per decision",
         f"${A[jev]['cost_per_call_usd'] * 1000:.2f} vs ${A[oai]['cost_per_call_usd'] * 1000:.2f} per 1k on MCP-Atlas"),
    ]
    ty, tw, th = 726, (W - 128 - 2 * 24) / 3, 108
    for i, (big, what, detail) in enumerate(tiles):
        tx = 64 + i * (tw + 24)
        s.append(f'<rect x="{tx:.1f}" y="{ty}" width="{tw:.1f}" height="{th}" rx="10" fill="#f3f2ee"/>')
        s.append(t(tx + 22, ty + 44, big, 34, INK, 700))
        s.append(t(tx + 22, ty + 71, what, 16, INK2, 600))
        s.append(t(tx + 22, ty + 93, detail, 14, MUTED))

    # footer
    n = {b: (d["cases"], d["catalog_tools"]) for b, d in data.items()}
    s.append(t(64, 866, f"MCP-Atlas: {n['atlas'][0]:,} step decisions over {n['atlas'][1]} tools, scored on the exact "
                        f"tool the reference agent called · MCP-Bench: {n['mcpbench'][0]} tasks, {n['mcpbench'][1]} tools, "
                        f"share of needed tools (max 90%) · MCP-Universe: {n['universe'][0]} tasks, {n['universe'][1]} tools, "
                        "needed server", 13, MUTED))
    s.append(t(64, 886, "Routing accuracy, not end-to-end agent scores · Jev via OpenRouter (typesafe/jev-1.13), "
                        "gpt-6-luna via OpenAI's Decisions API · jev-experiments / 01-jev-tool-router", 13, MUTED))
    s.append("</svg>")
    return "\n".join(s)


def to_png(svg: Path, png: Path) -> bool:
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome).exists() and not shutil.which("google-chrome"):
        return False
    html = svg.with_suffix(".render.html")
    html.write_text(f'<html><body style="margin:0">{svg.read_text()}</body></html>')
    subprocess.run([chrome if Path(chrome).exists() else "google-chrome", "--headless=new", "--disable-gpu",
                    "--hide-scrollbars", f"--window-size={W},{H}", "--force-device-scale-factor=2",
                    f"--screenshot={png}", html.as_uri()], check=True, capture_output=True)
    html.unlink()
    return png.exists()


def main():
    data = json.loads((OUT / "comparison.json").read_text())
    svg = OUT / "jev-vs-openai.svg"
    svg.write_text(build(data))
    print(f"wrote {svg}")
    png = OUT / "jev-vs-openai.png"
    if to_png(svg, png):
        print(f"wrote {png} ({W * 2}x{H * 2})")


if __name__ == "__main__":
    main()
