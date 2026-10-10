# 02 · Jev as the hands of a screen agent

An agent that controls any screen (browser, desktop or Android phone) from pixels alone, with the work split three ways:

```
                 ┌──────────── what to do ────────────┐   ┌──────── where exactly ─────────┐
 screenshot ──▶  LLM planner (gpt-6-luna)            ──▶  decision model (Jev)          ──▶  click / type / scroll
     │           "type 'alice' into the Username       picks one of the detected          on the device
     │            field"  (words, never coordinates)    elements, with a probability
     └──────▶  YOLO UI detector + OCR ─── 40-150 typed elements ───────▲
               button · link · text_input · checkbox · radio · toggle · dropdown · slider ·
               tab · icon · back · close · menu · search · scrollbar  (+ OCR text runs)
```

- **The LLM plans.** Each turn it sees the screenshot and the history, and names the next action and its target in plain words. It never outputs coordinates.
- **YOLO sees.** A YOLO11n trained here on auto-labelled screenshots finds the interactive elements and their kind. OCR reads their text and attaches nearby labels to fields and checkboxes.
- **Jev picks.** One Decisions API call: a `choice` question whose options are the detected elements (plus "none of these"). The returned probabilities give the pick and a calibrated confidence. If the confidence is low, the agent can tell the planner "not found" instead of clicking the wrong thing.

Compared against the same pipeline with **OpenAI's decision model** (`gpt-6-luna` on OpenAI's Decisions API), against the **chat LLM picking the element itself** (from the same list, or Set-of-Mark style on numbered boxes), and against the **LLM clicking raw coordinates** with no detector.

## Results

### Detector: done (trained here, on CPU)

YOLO11n, fine-tuned from COCO on 4,156 auto-labelled screenshots for 23 epochs: 15, then 8 more after fixing a labelling gap. That took 4.6 h on 4 CPU cores. The weights are `models/screenjev-yolo.pt` (5 MB); metrics are in `models/screenjev-yolo.json`.

| Split | Images | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| Val: synthetic pages + MiniWoB training tasks (640 px) | 368 | **94.2%** | 77.7% | 91.9% | 89.3% |
| Val at 960 px | 368 | **95.4%** | 79.7% | 94.0% | 91.4% |
| **Held-out MiniWoB eval tasks** (25 tasks the detector never saw, 640 px) | 146 | **71.5%** | 52.3% | 90.5% | 60.3% |

![Detector training](results/report/detector_training.png)

Per class (val, mAP50-95): button 90%, text_input 90%, tab 87%, back 86%, toggle 85%, dropdown 84%, icon 79%, slider 76%, menu 75%, scrollbar 73%, search 72%, link 71%, checkbox 70%, close 69%, radio 61%.

On held-out MiniWoB tasks, YOLO + OCR elements (red = detected widget, green = OCR text run):

![Perception on held-out MiniWoB tasks](results/report/perception_miniwob.png)

What it gets right, and where it falls short:
- **Fields, buttons, checkboxes, tabs:** found on unseen task layouts, with labels attached ("text input labeled Username", "checkbox labeled CXjt"). This takes ~0.35 s per screenshot on CPU (YOLO + OCR).
- **Held-out vs val:** the gap (72% vs 94% mAP50) is mostly MiniWoB's own widget styles that only occur in eval tasks. Inline `<span>` links are 33% mAP50-95. Native `<select>` reads as a text input, which still works because `select` picks options by text at that point. Tiny glyph icons that are styled only on hover (social-media's reply / retweet / more) aren't found.
- **Inline links:** when a link isn't boxed, it is still clickable. It sits inside an OCR text run, and the agent clicks the quoted word (`the link "in."`), not the run's center.
- **Image size:** `auto` uses 640 px for phone and MiniWoB-sized screenshots (upscaling them to 960 costs 7 points of mAP50) and 960 px for desktop-sized ones (+1.2 points mAP50 on val).
- **Training length:** the curve is still rising at epoch 23. The GPU notebook (40 epochs at 960 px) should do better on small elements.

### MiniWoB++ end-to-end: gpt-6-luna plans, the detector finds, a picker clicks

25 tasks × 10 seeds per variant in a real Chromium, 1,000 episodes. The planner is gpt-6-luna in every variant. Success means the task's own reward is positive. Run: `results/miniwob/openai-v1`.

| Who picks the element | Boxes from | Solved | Steps / episode | Picker latency |
|---|---|---|---|---|
| gpt-6-luna **decision model** (Decisions API) | our YOLO + OCR | **92.4%** | 2.8 | **0.20 s** |
| gpt-6-luna chat, names an element id | our YOLO + OCR | 92.8% | 2.7 | 1.45 s |
| gpt-6-luna decision model | the page's DOM (perfect boxes) | 92.0% | 2.8 | 0.20 s |
| the planner itself, on numbered boxes (Set-of-Mark) | our YOLO + OCR | 90.0% | 2.9 | in the planner call |

![MiniWoB success](results/report/miniwob_success.png)

- **The split works end to end.** The planner describes each step in words, YOLO finds the elements, and the decision model picks one. That solves 92% of episodes, the same as letting a chat LLM pick and 2 points better than one LLM doing everything (Set-of-Mark). The pick takes 0.2 s instead of 1.45 s. The pick is one part of a step: a full step, including the planner's screenshot call, takes ~5–6 s.
- **Our detector costs nothing here.** The DOM's perfect boxes do no better (92.0% vs 92.4%).
- **23 of the 25 tasks** are at 90–100% with either picker on YOLO boxes ([per task](results/report/REPORT.md)). Set-of-Mark drops on click-tab-2 (60%) and click-collapsible-2 (70%). Two tasks fail for every variant:
  - **social-media is 0% for every variant, DOM boxes included.** Its reply / retweet / like / ⋯ icons are 14 px CSS images that only look clickable on hover, and neither the detector nor the DOM labeller finds them.
  - **click-scroll-list is 50%.** In multi-select lists, clicks used to replace the selection, and a click on a named option landed in the middle of the list box. Both are fixed since this run (9/10 in a 10-episode re-check, not in the table).

### ScreenSpot-v2 grounding: can the picker hit the target in one shot?

ScreenSpot-v2 has 1,272 instructions ("close this window", "view battery usage") on iOS, Android, macOS, Windows and web screenshots. A hit means the click lands inside the target box. Run: `results/screenspot/openai-full`.

The OpenAI account ran out of credits during this run. 2,310 of the 8,904 calls failed with `429 no credits remaining`, mostly the web screenshots, which come last. Those rows are **left out, not counted as misses**, leaving ~950 answered instructions per variant: ~450 mobile, ~305 desktop, ~200 web. `python -m bench.screenspot --run openai-full --retry-errors` re-runs only those rows.

| Variant | Accuracy | Text targets | Icon targets | Detector ceiling | ECE | Acc. on most-confident half | p50 latency |
|---|---|---|---|---|---|---|---|
| gpt-6-luna clicks x, y (no detector) | **96.7%** | 96–99% | 89–97% | — | **0.008** | 99.4% | 1.8 s |
| OmniParser + description → gpt-6-luna decision | 80.8% | 93–96% | 55–65% | 96% | 0.038 | 95.7% | 0.6 s |
| our YOLO + description → gpt-6-luna chat | 77.5% | 94–98% | 46–62% | 83% | 0.099 | 95.7% | 1.4 s |
| our YOLO + description → gpt-6-luna decision | 74.3% | 94–97% | 39–48% | 83% | 0.090 | 94.3% | **0.18 s** |
| our YOLO → gpt-6-luna chat | 72.7% | 94–95% | 37–49% | 83% | 0.131 | 93.7% | 1.9 s |
| OmniParser → gpt-6-luna decision | 68.4% | 86–91% | 40–41% | 96% | 0.049 | 92.6% | 0.34 s |
| our YOLO → gpt-6-luna decision | 67.0% | 92–93% | 28–38% | 83% | 0.069 | 92.6% | 0.35 s |

"+ description" (the `@desc` variants) is how the agent works: gpt-6-luna looks at the screenshot and describes the target in words (no coordinates), then the picker grounds that description. Without it, the picker only gets the raw instruction. Per-platform numbers, ECE and tokens are in [REPORT.md](results/report/REPORT.md).

![ScreenSpot accuracy](results/report/screenspot_accuracy.png)

![Calibration](results/report/screenspot_calibration.png)

What it says:
- **gpt-6-luna is a strong grounding model on its own.** Clicking coordinates directly, it hits 96.7%, and its confidence is almost perfectly calibrated (ECE 0.008).
- **On text targets, our YOLO pipeline matches it** at 92–98%. That's why MiniWoB, which is mostly text widgets, is at 92%.
- **Icons are the gap.** The picker reads elements as text ("icon next to 'Downloads' at top-right"), so unlabelled glyphs (⋯, ⚙, ↗) are guesswork. Two things help:
  - The description step: +7 points for YOLO and +12 for OmniParser.
  - A detector that covers more icons: OmniParser's ceiling is 96% vs our 83%. Ours is trained only on synthetic and MiniWoB pages.
- **Calibration makes abstaining useful.** Acting only on the most confident half of picks gives 93–96% accuracy for every detector pipeline. The decision model's confidence is better calibrated than the chat model's self-reported confidence (ECE 0.04–0.09 vs 0.10–0.13), so a confidence threshold (`--min-confidence`) is meaningful.
- **Speed: the latency column is the pick alone, not the pipeline.** Medians per screenshot, on a 4-core CPU with no GPU:

  | Stage | Time |
  |---|---|
  | YOLO detection (ours) / OmniParser | 0.07 s / 0.69 s |
  | OCR (RapidOCR on CPU) | 1.49 s |
  | Description (gpt-6-luna reads the screenshot) | 2.14 s |
  | Pick: decision model / chat model | 0.18 s / 1.40 s |

  End to end, our YOLO + description + decision takes ~3.9 s, ~1.7 s without the description, against 1.8 s for gpt-6-luna clicking x, y. So for one-shot grounding the pipeline isn't faster, and OCR is the slow part. In the agent, the planner already reads the screenshot every step, so the description comes free. The pipeline then adds detection + OCR (~0.35 s on MiniWoB-sized screens) and the pick (0.17 s with the decision model vs 1.26 s with the chat model).
- **Cost:** a decision call reads ~900–1,100 tokens of element list. A description adds a ~1,800-token screenshot call, and an x, y answer is a ~1,650-token screenshot call. In the agent, the planner already sees the screenshot every step, so the description comes free and only the 0.2 s pick is added.

### Next: Jev

Jev hasn't run yet; it needs an OpenRouter key, `openrouter.ai` in the network allowlist, and `SCREENJEV_JEV_VIA=openrouter`. The runs above are built to take it as extra variants. ScreenSpot reuses the saved gpt-6-luna descriptions, so Jev grounds exactly the same text:

```bash
python -m bench.screenspot --run openai-full --variants yolo+jev,yolo+jev@desc,omniparser+jev,omniparser+jev@desc
python -m bench.miniwob    --run openai-v1  --variants yolo+jev,dom+jev --episodes 10
python -m bench.report --screenspot openai-full --miniwob openai-v1
```

## The detector

No UI-detection dataset was needed: every training screenshot is labelled by the page's own DOM.

- **`bench/synth.py`** generates random UI pages: web layouts (navbars, forms, tables, cards, dialogs, scrolling lists), phone-app screens (app bar with back/search/menu, list rows, bottom tabs, FAB) and desktop-app windows (title bar, menu bar, toolbars). They're styled with Bootstrap, Bulma, Pico or random custom CSS, light and dark, with icons from Bootstrap Icons, Lucide and Material Design Icons, at desktop, tablet and phone viewports. Every interactive element carries `data-ui="<class>"`, so labels are exact. Icons are labelled `back` / `close` / `menu` / `search` by which glyph was drawn.
- **MiniWoB++ pages** add the benchmark's widget style. They are labelled by `screenjev/detect/label.js` heuristics (tag, role, type, aria-label), with random clicks first so opened menus and expanded sections get labelled too. **Every task in the evaluation suite, and its variants, is excluded**, so the detector never sees the eval tasks' layouts.
- `label.js` also finds classic scrollbars (Playwright hides them in headless mode by default; we turn that off) and skips elements hidden behind overlays.

```bash
./scripts/fetch_assets.sh                       # icons, CSS frameworks, fonts (from the npm registry)
python -m bench.make_dataset --synth 3000       # ~4.2k train / 365 val screenshots, ~56k boxes
python -m bench.viz_labels train 20 /tmp/viz    # eyeball the labels
python -m bench.train_detector --device 0 --epochs 40 --imgsz 960 --cache disk   # GPU
python -m bench.train_detector --epochs 30 --imgsz 640                            # CPU (hours)
```

On a GPU in one click: [`notebooks/train_detector_colab.ipynb`](notebooks/train_detector_colab.ipynb) ([open in Colab](https://colab.research.google.com/github/NeelakandanNC/jev-experiments/blob/ccr-79ef4f1c-ef404d/02-jev-screen-control/notebooks/train_detector_colab.ipynb)). It builds the same dataset (same seeds), trains on the T4 and pushes or downloads `models/screenjev-yolo.pt`.

OmniParser v2's detector (Microsoft, YOLOv8, one "interactable" class) plugs in with `--detector omniparser`. The DOM itself is available as an oracle detector, `--detector dom` (browser only), to measure how much the detector's misses cost.

**OCR** is RapidOCR (PP-OCRv4 ONNX, bundled in the wheel, CPU). Its recognizer often drops spaces between English words, so word boundaries come from blank pixel columns in each line, ignoring underlines, plus per-character CTC positions. Words inside a detected box become its text; other words become `text` elements, which are clickable too. Fields, checkboxes, toggles and sliders get the nearest label beside or above them.

## Benchmarks

**ScreenSpot-v2** (grounding, 1,272 instructions on mobile, desktop and web screenshots): is the click inside the target box? This isolates "detector + picker" from planning. Each detector runs once per screenshot, and every picker sees the same element list. Also reported: the share of targets that have any detected element centred inside them (the detector's ceiling), and calibration (ECE, plus accuracy when acting only on the most confident half).

```bash
python -m bench.screenspot --variants yolo+jev,yolo+luna,yolo+llm,omniparser+jev,omniparser+luna,llm-coords
```

**MiniWoB++** (end-to-end, 25 tasks × 10 seeds, a real Chromium): does the agent finish the task? Tasks need only clicks, typing, selection, scrolling and keys (click-button, click-checkboxes, click-tab-2, click-collapsible-2, enter-text-dynamic, login-user, search-engine, social-media, navigate-tree, choose-list, …; see `bench/miniwob_env.py`). The task area is screenshotted at 3× so the 10 px text is readable. Episodes have no time limit, and success means a positive raw reward. The planner is identical across variants.

```bash
python -m bench.miniwob --variants yolo+jev,yolo+luna,yolo+llm,yolo+som,dom+jev --episodes 10
```

| Variant | Planner | Elements | Who picks the element |
|---|---|---|---|
| `yolo+jev` | gpt-6-luna | our YOLO + OCR | **Jev** (Decisions API, `choice`) |
| `yolo+luna` | gpt-6-luna | our YOLO + OCR | OpenAI decision model gpt-6-luna |
| `yolo+llm` | gpt-6-luna | our YOLO + OCR | gpt-6-luna (chat) reads the list, names an id + confidence |
| `yolo+som` | gpt-6-luna | our YOLO + OCR | the planner itself, on numbered boxes (Set-of-Mark), in the same call |
| `dom+jev` | gpt-6-luna | the page's DOM (oracle) | Jev |
| `llm-coords` (ScreenSpot) | — | none | gpt-6-luna outputs x, y |

`python -m bench.report` turns the latest runs into `results/report/REPORT.md` and charts.

## Use it on a real screen

```bash
./setup.sh                      # venv + Chromium (./setup.sh --desktop adds mss + pyautogui)
cp .env.example .env            # OPENAI_API_KEY; OPENROUTER_API_KEY + SCREENJEV_JEV_VIA=openrouter for Jev

.venv/bin/python -m screenjev --device browser --url https://news.ycombinator.com --task "Open the comments of the top story" --show
.venv/bin/python -m screenjev --device desktop --task "Open System Settings and turn on dark mode"
.venv/bin/python -m screenjev --device android --serial emulator-5554 --task "Turn on airplane mode"
```

Options: `--planner` (any OpenAI model, or `provider/model` via the gateway), `--selector jev|luna|llm|som|decision:<model>` (default `jev`; with only an OpenAI key use `luna`), `--detector yolo|omniparser|dom`, `--min-confidence 0.3` (below it, report "not found" to the planner instead of clicking). Every run writes `runs/<time>/index.html`: each step's screenshot with the detected boxes, the pick, the top candidates' probabilities and the outcome.

Desktop control moves your real mouse (slam it into a screen corner to abort, pyautogui's failsafe). Android needs `adb` with USB debugging on.

## Layout

```
screenjev/  devices/{browser,desktop,android}.py   one async Device interface, screenshot-pixel coordinates
            detect/{yolo,ocr,dom}.py, label.js     detectors, OCR, DOM labeller; __init__.py merges them into elements
            planner.py  selector.py  agent.py      LLM plan → decision-model pick → act
            decide.py (Decisions API client, from experiment 01)  llm.py (OpenAI SDK chat)  trace.py
bench/      synth.py  make_dataset.py  train_detector.py  viz_labels.py     detector data + training
            miniwob_env.py  miniwob.py  screenspot.py  report.py            benchmarks
models/     screenjev-yolo.pt (+ .json metrics)    the trained detector (committed, 5 MB)
results/    miniwob/<run>/  screenspot/<run>/  report/
tests/      .venv/bin/python -m pytest   (offline: SCREENJEV_MOCK=1 stand-ins, never reported)
```

Licences: the detector is trained with Ultralytics (AGPL-3.0), and so are its weights. OmniParser's weights are AGPL too.
