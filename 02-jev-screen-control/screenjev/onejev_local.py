"""OneJev (OmniJev/OneJev1.6-0.8B) run in-process with Hugging Face transformers, on CPU if need be.

OmniJev serves OneJev with Flide, whose kernels are Triton (GPU only). This client reproduces Flide's OneJev
format (flide_onejev, "onejev-labels-v2" prompts, debias=1 as `flide serve` defaults to) without its kernels:

  prompt   flide_onejev.prompt.render_question + build_messages, the tokenizer's chat template with
           enable_thinking=False and add_generation_prompt=True: the exact text Flide builds
  readout  one causal forward pass; the output-head logits of the answer-slot letters (A, B, ..., AA, ...) at the
           last token; softmax at the checkpoint's calibration temperature (no calibration.json -> 1.0)

The checkpoint has causal attention and the slot_lm readout (no readout.json, no modeling_overrides), which is
what this reproduces. Choices with more options than slots (256) are not supported here; ours have at most 251.
Requests are answered one at a time (a lock), each in a worker thread.
"""

from __future__ import annotations

import asyncio
import math
import sys
import threading
import time
from typing import Any

# transformers' Qwen3.5 picks flash-linear-attention / causal-conv1d kernels when those packages import (Flide
# installs FLA), and those are Triton: GPU only. A failed import makes transformers use its pure-PyTorch versions of
# the same gated-delta-rule and conv ops. Must run before transformers' qwen3_5 modeling module is imported.
for _pkg in ("fla", "causal_conv1d"):
    if _pkg not in sys.modules:
        sys.modules[_pkg] = None  # type: ignore[assignment]

MODEL_ID = "OmniJev/OneJev1.6-0.8B"


class OneJevLocal:
    _instances: dict[str, "OneJevLocal"] = {}

    @classmethod
    def get(cls, model_id: str = MODEL_ID) -> "OneJevLocal":
        if model_id not in cls._instances:
            cls._instances[model_id] = cls(model_id)
        return cls._instances[model_id]

    def __init__(self, model_id: str = MODEL_ID, threads: int | None = None):
        import torch
        import transformers
        from huggingface_hub import snapshot_download

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.dir = snapshot_download(model_id)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.dir)
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(self.dir, dtype=torch.float32)
        self.model.eval()
        from flide_onejev.prompt import LETTERS, SLOT_LABELS

        def one(label):
            enc = self.tokenizer.encode(label, add_special_tokens=False)
            return enc[0] if len(enc) == 1 and self.tokenizer.decode(enc) == label else None

        ids = [one(l) for l in LETTERS]
        extra = [one(l) for l in SLOT_LABELS[len(LETTERS):]]
        self.slot_ids = ids + extra if all(i is not None for i in extra) else ids
        self.lock = threading.Lock()

    def _prompt(self, state: Any, qid: str, criteria: dict[str, str], instructions: str) -> tuple[str, list[str]]:
        from flide.api.schema import ChoiceQuestion
        from flide_onejev.prompt import build_messages, render_question

        r = render_question(qid, ChoiceQuestion(type="choice", instructions=instructions, criteria=criteria))
        text = self.tokenizer.apply_chat_template(build_messages(state, r.suffix), tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        return text, r.labels

    def choice(self, state: Any, criteria: dict[str, str], instructions: str) -> tuple[dict[str, float], int]:
        """Probability per option (keyed as in `criteria`) and the prompt's token count."""
        if len(criteria) > len(self.slot_ids):
            raise ValueError(f"{len(criteria)} options exceed {len(self.slot_ids)} slots")
        text, labels = self._prompt(state, "q", criteria, instructions)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        torch = self.torch
        with self.lock, torch.inference_mode():
            out = self.model(input_ids=torch.tensor([ids]), logits_to_keep=1)
            logits = out.logits[0, -1, self.slot_ids[: len(labels)]].float().tolist()
        m = max(logits)
        w = [math.exp(x - m) for x in logits]
        s = sum(w)
        return {lab: x / s for lab, x in zip(labels, w)}, len(ids)


    # Screenshots are capped at this many pixels before the processor's own resize (a full 2880x1800 screen is
    # ~5,000 vision tokens, minutes per request on CPU). Flide itself would use the processor's 16.7M-pixel cap.
    MAX_IMAGE_PIXELS = 1024 * 1024

    def choice_with_image(self, state: str, image, criteria: dict[str, str], instructions: str) -> tuple[dict[str, float], int]:
        """As `choice`, with one screenshot. `state` must contain the placeholder <image:1> where the image goes
        (OneJev's convention); rendered with flide_onejev.media.render_prompt, as Flide renders media requests."""
        import transformers
        from flide.api.schema import ChoiceQuestion
        from flide_onejev.media import render_prompt
        from flide_onejev.prompt import build_messages, render_question

        if getattr(self, "processor", None) is None:
            self.processor = transformers.AutoProcessor.from_pretrained(self.dir)
        img = image.convert("RGB")
        if img.width * img.height > self.MAX_IMAGE_PIXELS:
            k = (self.MAX_IMAGE_PIXELS / (img.width * img.height)) ** 0.5
            img = img.resize((max(32, int(img.width * k)), max(32, int(img.height * k))))
        r = render_question("q", ChoiceQuestion(type="choice", instructions=instructions, criteria=criteria))
        text = render_prompt(self.processor, {"enable_thinking": False}, build_messages(state, r.suffix),
                             [{"type": "image", "image": img}])
        inputs = self.processor(text=[text], images=[img], return_tensors="pt")
        torch = self.torch
        with self.lock, torch.inference_mode():
            out = self.model(**inputs, logits_to_keep=1)
            logits = out.logits[0, -1, self.slot_ids[: len(r.labels)]].float().tolist()
        m = max(logits)
        w = [math.exp(x - m) for x in logits]
        z = sum(w)
        return {lab: x / z for lab, x in zip(r.labels, w)}, int(inputs["input_ids"].shape[1])


class OneJevLocalClient:
    """Speaks the native decision shape (like OpenRouter / Sarvam), so DecisionClient can treat it the same way."""

    def __init__(self, model_id: str = MODEL_ID):
        self.engine = OneJevLocal.get(model_id)

    async def decide(self, model: str, input: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
        answers, tokens = [], 0
        for i, q in enumerate(questions):
            name = q.get("name") or f"q{i}"
            if q["type"] != "choice":
                raise ValueError("the local OneJev client answers choice questions only")
            criteria = {c["value"]: c.get("description") or c["value"] for c in q["choices"]}
            probs, n = await asyncio.to_thread(self.engine.choice, input, criteria, q.get("instructions") or "")
            tokens += n
            best = max(probs, key=probs.get)
            answers.append({"type": "choice", "name": name, "choice": best, "confidence": probs[best],
                            "probabilities": [{"value": k, "probability": v} for k, v in probs.items()]})
        return {"model": MODEL_ID, "answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0},
                "provider_metadata": {"gateway": {"cost": 0.0}}}

    async def close(self):
        pass


def _selftest():
    eng = OneJevLocal.get()
    t0 = time.perf_counter()
    probs, n = eng.choice("Task: log in. Action to perform now: click the Login button.",
                          {"e0": 'button "Cancel" at bottom-left', "e1": 'button "Login" at bottom-right',
                           "none": "None of these"}, "Which element should be clicked?")
    print({k: round(v, 4) for k, v in probs.items()}, n, "tokens", f"{time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    _selftest()
