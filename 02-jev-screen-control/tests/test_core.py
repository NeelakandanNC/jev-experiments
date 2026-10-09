"""Offline tests (no keys, no network): perception merge, planner parsing, selectors in mock mode,
the browser device + DOM labeller, and the MiniWoB harness."""

import asyncio
import os

import pytest

os.environ["SCREENJEV_MOCK"] = "1"

from screenjev.detect import merge  # noqa: E402
from screenjev.detect.ocr import Word, split_words  # noqa: E402
from screenjev.planner import Step, parse_step  # noqa: E402
from screenjev.selector import DecisionSelector, LLMSelector  # noqa: E402
from screenjev.types import Element, Screen, reading_order, where  # noqa: E402
from bench.common import ece  # noqa: E402
from bench.screenspot import selective  # noqa: E402


def test_split_words_proportional():
    ws = split_words("Sign in now", (0, 0, 110, 10), 0, 0.9)
    assert [w.text for w in ws] == ["Sign", "in", "now"]
    assert ws[0].box[0] == 0 and abs(ws[2].box[2] - 110) < 1e-6


def test_merge_assigns_words_labels_and_text():
    boxes = [Element("", "button", (100, 100, 200, 140)), Element("", "text_input", (120, 20, 300, 50)),
             Element("", "checkbox", (10, 200, 24, 214))]
    words = (split_words("Submit", (120, 110, 180, 130), 0, 0.9) + split_words("Name:", (40, 25, 100, 45), 1, 0.9)
             + split_words("Remember me", (30, 200, 140, 214), 2, 0.9) + split_words("Welcome back", (10, 300, 200, 320), 3, 0.9))
    s = merge(boxes, words, 400, 400)
    by = {e.cls: e for e in s.elements}
    assert by["button"].text == "Submit"
    assert by["text_input"].label == "Name:"
    assert by["checkbox"].label == "Remember me"
    assert by["text"].text in ("Welcome back", "Name:", "Remember me")
    assert [e.id for e in s.elements] == [f"e{i}" for i in range(len(s.elements))]


def test_reading_order_and_where():
    els = reading_order([Element("", "button", (300, 10, 340, 30)), Element("", "button", (10, 12, 50, 30)),
                         Element("", "link", (10, 200, 50, 220))])
    assert [e.box[0] for e in els] == [10, 300, 10]
    assert where(0.1, 0.1) == "top-left" and where(0.5, 0.5) == "center" and where(0.9, 0.9) == "bottom-right"


def test_parse_step():
    s = parse_step({"action": "Type", "target": "email field", "text": "a@b.c", "submit": "true"})
    assert s.action == "type" and s.submit and "a@b.c" in s.describe()
    assert parse_step({"action": "explode"}).action == "wait"
    assert parse_step({}).action == "fail"
    assert parse_step({"action": "click", "submit": "false"}).submit is False


def _screen():
    els = reading_order([Element("", "button", (10, 10, 60, 30), text="Cancel"),
                         Element("", "button", (80, 10, 130, 30), text="Submit"),
                         Element("", "close", (300, 5, 320, 25))])
    return Screen(400, 300, els)


async def test_decision_selector_mock():
    sel = DecisionSelector("typesafe-ai/jev")
    s = await sel.select("Click submit", Step("click", target='the "Submit" button'), _screen())
    best_real = next(k for k, _ in s.ranked if k != "none")
    assert _screen().by_id(best_real).text == "Submit" and 0 < s.confidence <= 1
    assert sum(p for _, p in s.ranked) == pytest.approx(1.0)


async def test_llm_selector_mock():
    sel = LLMSelector()
    s = await sel.select("Click cancel", Step("click", target="Cancel button"), _screen())
    assert s.element is not None and s.element.text == "Cancel"


def test_ece_and_selective():
    assert ece([1.0, 1.0], [True, True]) == pytest.approx(0.0)
    assert ece([0.9, 0.9], [False, False]) == pytest.approx(0.9)
    sel = selective([(0.9, True), (0.8, True), (0.2, False), (0.1, False)])
    assert sel[0.5] == 1.0


def test_synth_page_is_deterministic():
    from bench.synth import make_page
    a, oa = make_page(7)
    b, ob = make_page(7)
    assert a == b and oa == ob and 'data-ui="' in a


@pytest.mark.slow
async def test_browser_dom_labels_and_miniwob():
    from bench.miniwob_env import MiniWoBEnv
    async with MiniWoBEnv(scale=2.0) as env:
        utter = await env.reset("click-button", 3)
        assert "button" in utter.lower()
        els = await env.device.dom_elements()
        buttons = [e for e in els if e["cls"] == "button"]
        assert buttons
        want = utter.split('"')[1]
        target = next(e for e in buttons if e["text"] == want)
        x1, y1, x2, y2 = target["box"]
        await env.device.click((x1 + x2) / 2, (y1 + y2) / 2)
        await asyncio.sleep(0.2)
        done, raw, _ = await env.result()
        assert done and raw > 0


def test_point_for_clicks_the_quoted_word_in_a_text_run():
    from screenjev.detect import _text_el
    run = split_words("Lobortis in. Enim risus", (0, 0, 230, 10), 0, 0.9)
    el = _text_el(run)
    assert el.center[0] == pytest.approx(115)
    assert el.point_for('click "Lobortis"')[0] == pytest.approx(40)  # "Lobortis" spans x 0..80
    assert el.point_for('the link "Enim risus"')[0] == pytest.approx((130 + 230) / 2)
    assert el.point_for("no quotes here") == el.center
    assert el.point_for("the “Lobortis” link")[0] == pytest.approx(40)  # curly quotes, as GPT writes them
