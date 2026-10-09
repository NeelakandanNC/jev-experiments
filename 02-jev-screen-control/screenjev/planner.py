"""Top-level planning: an LLM looks at the screen and says what to do next, in words.

It never gives coordinates or element ids. It names the action and describes the target
("the blue 'Submit' button under the form"); finding that element among the detector's boxes
is the decision model's job (selector.py).

The `som` variant is the baseline where the LLM does both: it sees numbered boxes
(Set-of-Mark) and picks the element id itself.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from PIL import Image, ImageDraw

from .llm import LLM, LLMReply, image_part
from .types import Screen

ACTIONS = ("click", "double_click", "type", "select", "scroll", "key", "back", "wait", "done", "fail")
TARGETED = ("click", "double_click", "type", "select")

SYSTEM = """You operate a {device} screen to complete the user's task. Each turn you get a screenshot and the
history of what was done so far, and you reply with the single next action as a JSON object.

You never give coordinates. Describe the target element so that it can be found in a list of
elements detected on the screen: its visible text, its kind (button, link, text field, checkbox,
radio, toggle, dropdown, tab, slider, icon, back arrow, close X, menu icon, search icon,
scrollbar, plain text), and roughly where it is.

Actions (JSON keys in brackets):
  click         [target]                     click an element (buttons, links, checkboxes, tabs, text, ...)
  double_click  [target]
  type          [target, text, submit]       focus a text field, replace its content with `text`; submit=true presses Enter after
  select        [target, option]             choose `option` in a dropdown
  scroll        [direction, target?]         direction "up" or "down"; target = the scrollable area (omit for the whole screen)
  key           [key]                        a key or chord: Enter, Tab, Escape, Backspace, ctrl+a
  back          []                           go back (browser / Android back)
  wait          []                           let the screen settle
  done          []                           the task is complete
  fail          [reason]                     the task cannot be done

Rules:
- Only target elements visible in the current screenshot. If what you need isn't visible, scroll or navigate first.
- One action per reply. Check the screenshot to see whether the previous action worked before repeating it.
- If the history says a target could not be found, describe it differently or change approach.
- Reply with JSON only: {{"thought": "<one or two sentences>", "action": "...", "target": "...", "text": "...",
  "submit": false, "option": "...", "direction": "...", "key": "...", "reason": "..."}} (omit keys you don't need).{extra}"""

SOM_EXTRA = """
- The screenshot has numbered boxes, and the element list gives each box's id. For click / double_click /
  type / select, also return "element": "<id>" naming the box to act on (or "none" if no box fits)."""


@dataclass
class Step:
    action: str
    target: str = ""
    text: str = ""
    submit: bool = False
    option: str = ""
    direction: str = ""
    key: str = ""
    reason: str = ""
    thought: str = ""
    element: str = ""  # som only

    def describe(self) -> str:
        a = self.action
        if a == "type":
            return f'type "{self.text}" into {self.target}' + (" and press Enter" if self.submit else "")
        if a == "select":
            return f'select "{self.option}" in {self.target}'
        if a == "scroll":
            return f"scroll {self.direction or 'down'}" + (f" in {self.target}" if self.target else "")
        if a == "key":
            return f"press {self.key}"
        if a in TARGETED:
            return f"{a.replace('_', ' ')} {self.target}"
        return a

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in ("", False)}


def parse_step(data: dict[str, Any]) -> Step:
    action = str(data.get("action", "")).strip().lower().replace(" ", "_")
    if action not in ACTIONS:
        action = "fail" if not action else ("click" if "click" in action else "wait")
    s = Step(action=action)
    for k in ("target", "text", "option", "direction", "key", "reason", "thought", "element"):
        v = data.get(k)
        if v is not None:
            setattr(s, k, str(v))
    s.submit = bool(data.get("submit")) and str(data.get("submit")).lower() not in ("false", "0")
    return s


@dataclass
class HistoryItem:
    step: Step
    outcome: str  # what happened: "clicked button 'Submit' (e4, confidence 0.93)" / "target not found"

    def line(self, i: int) -> str:
        return f"{i + 1}. {self.step.describe()} -> {self.outcome}"


@dataclass
class Planner:
    llm: LLM
    device: str = "browser"
    som: bool = False
    max_history: int = 15
    calls: list[LLMReply] = field(default_factory=list)

    async def next(self, task: str, image: Image.Image, history: list[HistoryItem], screen: Screen | None = None) -> Step:
        system = SYSTEM.format(device=self.device, extra=SOM_EXTRA if self.som else "")
        hist = "\n".join(h.line(i) for i, h in enumerate(history))[-6000:] if history else "(nothing yet)"
        text = f"Task: {task}\n\nHistory:\n{hist}\n\nWhat is the next action?"
        img = image
        if self.som and screen is not None:
            img = draw_marks(image, screen)
            listing = "\n".join(f"{e.id}: {e.describe(screen.width, screen.height)}" for e in screen.elements)
            text += f"\n\nElements (id: description):\n{listing}"
        reply = await self.llm.json(system, [{"type": "text", "text": text}, image_part(img)])
        self.calls.append(reply)
        step = parse_step(reply.data)
        if not reply.data:
            step = Step("wait", reason="planner returned no JSON")
        return step


def draw_marks(image: Image.Image, screen: Screen) -> Image.Image:
    """Set-of-Mark: numbered boxes on the screenshot (for the som baseline and for traces)."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    for e in screen.elements:
        color = "#16a34a" if e.cls == "text" else "#dc2626"
        d.rectangle(e.box, outline=color, width=2)
        tag = e.id[1:]
        x, y = e.box[0], max(0, e.box[1] - 11)
        d.rectangle([x, y, x + 7 * len(tag) + 4, y + 11], fill=color)
        d.text((x + 2, y), tag, fill="white")
    return img


def mock_plan(messages: list[dict]) -> dict:
    """Scripted stand-in for tests: click whatever the task quotes, then stop."""
    text = next(p["text"] for p in messages[1]["content"] if p.get("type") == "text")
    task = text.split("\n", 1)[0]
    if "(nothing yet)" not in text:
        return {"action": "done", "thought": "mock"}
    m = re.search(r'"([^"]+)"', task)
    return {"action": "click", "target": m.group(1) if m else task, "thought": "mock"}
