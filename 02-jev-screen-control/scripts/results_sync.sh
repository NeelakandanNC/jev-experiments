#!/usr/bin/env bash
# Commits + pushes benchmark result files every N minutes while a benchmark runs, so a lost machine loses
# at most N minutes of results. Runs resume from them: rerun the same command with the same --run name.
set -uo pipefail
cd "$(dirname "$0")/.."
EVERY=${1:-15}; BRANCH=$(git rev-parse --abbrev-ref HEAD)
sync() {
  git add results/screenspot results/miniwob 2>/dev/null
  git diff --cached --quiet && return 0
  git commit -q -m "02-jev-screen-control: benchmark results in progress ($(date -u +%H:%M) UTC)" -- results || return 0
  for i in 1 2 3 4; do git push -q origin "$BRANCH" && echo "$(date -u +%H:%M) pushed" && return 0; sleep $((2 ** i)); done
}
while pgrep -f "python -m bench\.(screenspot|miniwob)" > /dev/null; do
  sleep $((EVERY * 60))
  sync
done
sync
echo "benchmarks finished; final results pushed"
