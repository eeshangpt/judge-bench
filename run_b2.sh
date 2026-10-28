#!/usr/bin/env bash
# B2 prediction runner (#146): inference only, no metrics. Resumable. Qwen3-4B (primary) then MiniCheck.
cd "$(dirname "$0")"
for m in Qwen/Qwen3-4B-Instruct-2507 minicheck; do
  echo "=== $m $(date -Iseconds)"
  .venv/bin/python -u judge.py run --model "$m" --sets faithbench --splits samples --out preds
done
echo "=== done $(date -Iseconds)"
