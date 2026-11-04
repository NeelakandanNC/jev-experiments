#!/usr/bin/env bash
# Keeps the OneJev ScreenSpot run going: restarts it (it resumes from cases.jsonl) and the results sync if the
# machine was restored without them. Prints progress. Usage: scripts/ensure_onejev.sh [sample]
cd "$(dirname "$0")/.."
SAMPLE=${1:-100}
VARIANTS=yolo+onejev,yolo+onejev@desc,omniparser+onejev,omniparser+onejev@desc,yolo+onejev-vis@desc
CASES=yolo+luna,yolo+luna@desc,omniparser+luna,omniparser+luna@desc
done_rows=$(grep -c onejev results/screenspot/openai-full/cases.jsonl 2>/dev/null || echo 0)
target=$((SAMPLE * 5))
if [ "$done_rows" -ge "$target" ]; then echo "done: $done_rows/$target"; exit 0; fi
if ! pgrep -f "python -m bench\.screenspot" > /dev/null; then
  setsid nohup .venv/bin/python -m bench.screenspot --run openai-full --variants "$VARIANTS" --same-cases-as "$CASES" \
    --sample "$SAMPLE" --concurrency 2 >> data/logs/screenspot_onejev.log 2>&1 < /dev/null & disown
  echo "restarted the run"
fi
pgrep -f "scripts/results_sync\.sh" > /dev/null || { setsid nohup ./scripts/results_sync.sh 10 > data/logs/results_sync.log 2>&1 < /dev/null & disown; echo "restarted results sync"; }
echo "progress: $done_rows/$target rows"
