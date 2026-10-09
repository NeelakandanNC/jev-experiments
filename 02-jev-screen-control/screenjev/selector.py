"""Grounding: given the planner's step and the detected elements, pick the element to act on.

  DecisionSelector  one Decisions API call: a `choice` question whose options are the elements
                    (+ "none"), answered with a probability per element. Jev or gpt-6-luna.
  LLMSelector       baseline: a chat LLM reads the same element list and names one, with a
                    self-reported confidence.
  SomSelector       baseline: the planner already picked an element id itself (Set-of-Mark).

The probability of the pick is the selector's confidence. Below `min_confidence` the agent
doesn't act and tells the planner the target wasn't found, which is where calibration pays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .decide import DecisionClient
from .llm import LLM, parse_json
from .planner import Step
from .types import Element, Screen

NONE = "none"
NONE_DESC = "None of these: the target is not on the screen (it needs scrolling, navigation, or a different screen)"

INSTRUCTIONS = ("An agent operating a computer or phone screen wants to perform the action in the input. "
                "Which detected on-screen element is the one to act on? The options list each element's kind, "
                "its visible text, its label, and where it is on the screen.")


@dataclass
class Selection:
    element: Element | None
    confidence: float | None          # probability (decision models) / self-reported (LLM) of the pick
    ranked: list[tuple[str, float]] = field(default_factory=list)  # top candidates (id, p)
    latency_s: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    model: str = ""
    mock: bool = False
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {"element": self.element.id if self.element else None, "confidence": self.confidence,
                "ranked": [[k, round(v, 4)] for k, v in self.ranked[:5]], "latency_s": round(self.latency_s, 3),
                "cost_usd": self.cost_usd, "tokens": self.tokens, "model": self.model, "error": self.error}


def step_input(task: str, step: Step, screen: Screen) -> str:
    lines = [f"Task: {task}", f"Action to perform now: {step.describe()}"]
    if step.target:
        lines.append(f"Target description: {step.target}")
    lines.append(f"Screen: {screen.width}x{screen.height} px, {len(screen.elements)} detected elements.")
    return "\n".join(lines)


class DecisionSelector:
    def __init__(self, model: str = "typesafe-ai/jev"):
        self.model = model
        self.name = "jev" if model == "typesafe-ai/jev" else model.split("/")[-1]
        self.client = DecisionClient(model)

    async def select(self, task: str, step: Step, screen: Screen, allow_none: bool = True) -> Selection:
        """allow_none=False when the target is known to be on screen (grounding benchmarks)."""
        if not screen.elements:
            return Selection(None, None, model=self.model, error="no elements detected")
        choices = [{"value": e.id, "description": e.describe(screen.width, screen.height)} for e in screen.elements]
        if allow_none or len(choices) < 2:  # the API needs at least 2 options
            choices.append({"value": NONE, "description": NONE_DESC})
        q = [{"type": "choice", "name": "target", "instructions": INSTRUCTIONS, "choices": choices}]
        for attempt in range(2):  # an empty answer / refusal is rare; one retry
            d = await self.client.decide(step_input(task, step, screen), q)
            a = d.answers.get("target")
            ranked = a.ranked() if a else []
            if ranked:
                break
        if not ranked:
            return Selection(None, None, latency_s=d.latency_s, cost_usd=d.cost_usd, tokens=d.input_tokens,
                             model=d.model or self.model, mock=d.mock, error="refusal / empty answer")
        best, p = ranked[0]
        return Selection(screen.by_id(best) if best != NONE else None, p, ranked, d.latency_s, d.cost_usd,
                         d.input_tokens, d.model or self.model, d.mock)

    async def aclose(self):
        await self.client.aclose()


LLM_SYSTEM = """You ground UI actions. Given an action and the list of elements detected on the screen, return the
id of the element to act on, or "none" if no element fits. Also give your confidence (0 to 1) that the
chosen id is correct. Reply with JSON only: {"element": "<id or none>", "confidence": <0..1>}"""


class LLMSelector:
    def __init__(self, model: str = "gpt-6-luna"):
        self.model = model
        self.name = f"llm:{model.split('/')[-1]}"
        self.llm = LLM(model)
        if self.llm.mock:
            self.llm.scripted = _mock_llm_select

    async def select(self, task: str, step: Step, screen: Screen, allow_none: bool = True) -> Selection:
        listing = "\n".join(f"{e.id}: {e.describe(screen.width, screen.height)}" for e in screen.elements)
        system = LLM_SYSTEM if allow_none else LLM_SYSTEM.replace(', or "none" if no element fits', " (the target is on the screen, so always pick one)")
        r = await self.llm.json(system, f"{step_input(task, step, screen)}\n\nElements:\n{listing}")
        eid = str(r.data.get("element", "")).strip()
        try:
            conf = min(1.0, max(0.0, float(r.data.get("confidence"))))
        except (TypeError, ValueError):
            conf = None
        el = screen.by_id(eid)
        return Selection(el, conf, [(eid, conf or 0.0)], r.latency_s, r.cost_usd, r.input_tokens + r.output_tokens,
                         r.model, r.mock, "" if el or eid == NONE else f"unknown id {eid!r}")

    async def aclose(self):
        await self.llm.aclose()


class SomSelector:
    """The planner picked `step.element` itself; nothing to call."""
    name = "som"

    async def select(self, task: str, step: Step, screen: Screen, allow_none: bool = True) -> Selection:
        el = screen.by_id(step.element.strip())
        return Selection(el, None, [(step.element, 1.0)], model="planner")

    async def aclose(self):
        pass


def make_selector(spec: str):
    """jev | luna | decision:<model> | llm[:<model>] | som"""
    if spec == "jev":
        return DecisionSelector("typesafe-ai/jev")
    if spec == "luna":
        return DecisionSelector("openai/gpt-6-luna")
    if spec.startswith("decision:"):
        return DecisionSelector(spec.split(":", 1)[1])
    if spec == "llm" or spec.startswith("llm:"):
        return LLMSelector(spec.split(":", 1)[1] if ":" in spec else "gpt-6-luna")
    if spec == "som":
        return SomSelector()
    raise ValueError(f"unknown selector {spec!r} (jev | luna | decision:<model> | llm[:<model>] | som)")


def _mock_llm_select(messages):
    from .decide import _words
    text = messages[1]["content"]
    target = next((l.split(":", 1)[1] for l in text.splitlines() if l.startswith("Target description:")), "")
    best, score = NONE, 0.0
    for line in text.split("Elements:\n", 1)[-1].splitlines():
        eid, _, desc = line.partition(": ")
        s = len(_words(target) & _words(desc))
        if s > score:
            best, score = eid, s
    return {"element": best, "confidence": 0.9 if score else 0.1}


__all__ = ["Selection", "DecisionSelector", "LLMSelector", "SomSelector", "make_selector", "parse_json"]
