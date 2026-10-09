#!/usr/bin/env bash
# One-time setup: venv, Chromium for Playwright, and (for retraining only) the synthetic-UI assets.
set -euo pipefail
cd "$(dirname "$0")"
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
.venv/bin/python -m playwright install chromium || echo "playwright install failed: set CHROMIUM_PATH to a Chromium binary"
[ "${1:-}" = "--desktop" ] && uv pip install -q --python .venv/bin/python mss pyautogui
[ "${1:-}" = "--train" ] && ./scripts/fetch_assets.sh
echo "Ready. The trained detector ships in models/screenjev-yolo.pt."
