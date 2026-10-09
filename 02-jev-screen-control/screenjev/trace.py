"""Per-run trace: annotated screenshot per step + trace.json + a static index.html to scroll through."""

from __future__ import annotations

import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .types import Screen


class Trace:
    def __init__(self, out_dir: str | Path):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []

    def add(self, image: Image.Image, screen: Screen, step, sel, rec) -> None:
        img = image.convert("RGB").copy()
        d = ImageDraw.Draw(img)
        for e in screen.elements:
            d.rectangle(e.box, outline="#22c55e" if e.cls == "text" else "#f97316", width=1)
        if sel and sel.element is not None:
            d.rectangle(sel.element.box, outline="#2563eb", width=4)
            x, y = sel.element.center
            d.ellipse([x - 6, y - 6, x + 6, y + 6], outline="#2563eb", width=3)
        name = f"step_{rec.i:02d}.png"
        img.save(self.dir / name)
        self.rows.append({"img": name, "step": rec.step, "selection": rec.selection, "outcome": rec.outcome,
                          "elements": [e.to_json() for e in screen.elements]})

    def finish(self, result) -> None:
        (self.dir / "trace.json").write_text(json.dumps({"result": result.to_json(), "steps": self.rows}, indent=1))
        cards = []
        for r in self.rows:
            s = r["step"]
            sel = r["selection"] or {}
            ranked = ", ".join(f"{k} {v:.2f}" for k, v in sel.get("ranked", [])) if sel else ""
            cards.append(f"""<section><img src="{r['img']}"><div><h3>{html.escape(s.get('action', ''))}
              {html.escape(s.get('target', ''))}</h3><p><i>{html.escape(s.get('thought', ''))}</i></p>
              <p>pick: <b>{html.escape(str(sel.get('element')))}</b> {html.escape(ranked)}</p>
              <p>&rarr; {html.escape(r['outcome'])}</p></div></section>""")
        page = f"""<!doctype html><meta charset="utf-8"><title>screenjev trace</title>
<style>body{{font:14px system-ui;margin:16px;background:#fff;color:#111}} section{{display:flex;gap:16px;margin:16px 0;
border-top:1px solid #ddd;padding-top:16px;flex-wrap:wrap}} img{{max-width:min(720px,100%);border:1px solid #ccc}}
h3{{margin:0 0 6px}} @media (prefers-color-scheme: dark){{body{{background:#111;color:#eee}}}}</style>
<h2>{html.escape(result.task)}</h2><p>status: <b>{result.status}</b> {html.escape(result.error)} · {len(self.rows)} steps ·
{result.wall_s:.1f}s</p>{''.join(cards)}"""
        (self.dir / "index.html").write_text(page)
