"""The control loop: screenshot -> detect -> plan (LLM) -> select (decision model) -> act."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .detect import Perception
from .devices.base import Device
from .planner import TARGETED, HistoryItem, Planner, Step
from .selector import Selection
from .trace import Trace
from .types import Screen


@dataclass
class AgentConfig:
    max_steps: int = 15
    min_confidence: float = 0.0   # don't act on picks below this; tell the planner instead
    settle_s: float = 0.4         # wait after each action before the next screenshot
    scroll_frac: float = 0.6      # scroll by this fraction of the screen height


@dataclass
class StepRecord:
    i: int
    step: dict[str, Any]
    selection: dict[str, Any] | None
    outcome: str
    acted: bool
    n_elements: int
    t_perceive: float
    t_plan: float
    t_select: float


@dataclass
class RunResult:
    task: str
    status: str                    # done | fail | max_steps | env_done | error
    steps: list[StepRecord] = field(default_factory=list)
    planner_calls: int = 0
    planner_tokens: int = 0
    planner_cost_usd: float = 0.0
    select_calls: int = 0
    select_cost_usd: float = 0.0
    select_latency_s: float = 0.0
    wall_s: float = 0.0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["steps"] = [s.__dict__ for s in self.steps]
        return d


class Agent:
    def __init__(self, device: Device, perception: Perception, planner: Planner, selector, config: AgentConfig | None = None,
                 trace: Trace | None = None):
        self.device, self.perception, self.planner, self.selector = device, perception, planner, selector
        self.cfg = config or AgentConfig()
        self.trace = trace

    async def run(self, task: str, env_done: Callable[[], Awaitable[bool]] | None = None) -> RunResult:
        res = RunResult(task, "max_steps")
        history: list[HistoryItem] = []
        t_start = time.perf_counter()
        n_calls0 = len(self.planner.calls)
        try:
            for i in range(self.cfg.max_steps):
                t0 = time.perf_counter()
                image = await self.device.screenshot()
                screen = await self.perception.screen(image, self.device)
                t1 = time.perf_counter()
                step = await self.planner.next(task, image, history, screen if self.planner.som else None)
                t2 = time.perf_counter()
                sel = None
                if step.action in TARGETED or (step.action == "scroll" and step.target):
                    sel = await self.selector.select(task, step, screen)
                    res.select_calls += 1
                    res.select_cost_usd += sel.cost_usd
                    res.select_latency_s += sel.latency_s
                t3 = time.perf_counter()

                if step.action in ("done", "fail"):
                    res.status = step.action
                    outcome, acted = step.reason or step.action, False
                else:
                    self.device.note = ""
                    outcome, acted = await self.act(step, sel, screen)
                    if self.device.note:
                        outcome += f" ({self.device.note})"
                rec = StepRecord(i, step.to_json(), sel.to_json() if sel else None, outcome, acted,
                                 len(screen.elements), t1 - t0, t2 - t1, t3 - t2)
                res.steps.append(rec)
                history.append(HistoryItem(step, outcome))
                if self.trace:
                    self.trace.add(image, screen, step, sel, rec)
                if res.status in ("done", "fail"):
                    break
                await asyncio.sleep(self.cfg.settle_s)
                if env_done and await env_done():
                    res.status = "env_done"
                    break
        except Exception as e:  # report, don't crash a benchmark run
            res.status, res.error = "error", f"{type(e).__name__}: {e}"
        calls = self.planner.calls[n_calls0:]
        res.planner_calls = len(calls)
        res.planner_tokens = sum(c.input_tokens + c.output_tokens for c in calls)
        res.planner_cost_usd = sum(c.cost_usd for c in calls)
        res.wall_s = time.perf_counter() - t_start
        if self.trace:
            self.trace.finish(res)
        return res

    async def act(self, step: Step, sel: Selection | None, screen: Screen) -> tuple[str, bool]:
        d, a = self.device, step.action
        if a in TARGETED:
            el = sel.element if sel else None
            if el is None or (sel.confidence is not None and sel.confidence < self.cfg.min_confidence):
                why = sel.error if sel and sel.error else "no detected element matched"
                guess = ""
                if sel and sel.element is not None:
                    guess = f"; best guess {sel.element.describe(screen.width, screen.height)} at p={sel.confidence:.2f}"
                return f"NOT DONE: target not found ({why}{guess})", False
            x, y = el.point_for(step.target)
            what = f"{el.describe(screen.width, screen.height)} [{el.id}" + (f", p={sel.confidence:.2f}]" if sel.confidence is not None else "]")
            if a == "click":
                await d.click(x, y)
                return f"clicked {what}", True
            if a == "double_click":
                await d.double_click(x, y)
                return f"double-clicked {what}", True
            if a == "type":
                await d.click(x, y)
                await d.clear_field()
                await d.type_text(step.text)
                if step.submit:
                    await d.key("Enter")
                return f'typed "{step.text}" into {what}' + (" and pressed Enter" if step.submit else ""), True
            if a == "select":
                if await d.select_option(x, y, step.option):
                    return f'selected "{step.option}" in {what}', True
                await d.click(x, y)
                return f"opened {what}; now click the option", True
        if a == "scroll":
            dy = self.cfg.scroll_frac * screen.height * (-1 if step.direction.lower() == "up" else 1)
            el = sel.element if sel else None
            x, y = el.center if el else (screen.width / 2, screen.height / 2)
            await d.scroll(x, y, dy)
            return f"scrolled {step.direction or 'down'}" + (f" over {el.id}" if el else ""), True
        if a == "key":
            await d.key(step.key or "Enter")
            return f"pressed {step.key or 'Enter'}", True
        if a == "back":
            await d.back()
            return "went back", True
        if a == "wait":
            await asyncio.sleep(1.0)
            return "waited", True
        return f"unknown action {a!r}", False
