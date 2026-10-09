#!/usr/bin/env bash
# Pushes the training checkpoint to the branch every N epochs, so a lost machine loses at most N epochs.
# Resume elsewhere: python -m bench.make_dataset (same seeds -> same data), then
#   python -m bench.train_detector --resume-from models/checkpoints/screenjev-last.pt
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=data/runs/${1:-screenjev}; EVERY=${2:-3}; BRANCH=$(git rev-parse --abbrev-ref HEAD)
last_pushed=0
push() {
  mkdir -p models/checkpoints
  cp "$RUN/weights/last.pt" models/checkpoints/screenjev-last.pt
  cp "$RUN/results.csv" models/checkpoints/results.csv 2>/dev/null
  git add models/checkpoints && git commit -q -m "02-jev-screen-control: detector training checkpoint (epoch $1)" -- models/checkpoints || return 0
  for i in 1 2 3 4; do git push -q origin "$BRANCH" && return 0; sleep $((2 ** i)); done
}
while true; do
  done_epochs=$(( $(wc -l < "$RUN/results.csv" 2>/dev/null || echo 1) - 1 ))
  if [ "$done_epochs" -ge $((last_pushed + EVERY)) ] && [ -f "$RUN/weights/last.pt" ]; then
    push "$done_epochs" && last_pushed=$done_epochs && echo "$(date -u +%H:%M) pushed epoch $done_epochs"
  fi
  if ! pgrep -f "python -m bench.train_detector" > /dev/null; then
    [ "$done_epochs" -gt "$last_pushed" ] && [ -f "$RUN/weights/last.pt" ] && push "$done_epochs" && echo "pushed final epoch $done_epochs"
    exit 0
  fi
  sleep 120
done
