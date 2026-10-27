#!/usr/bin/env bash
# B input prep (#146): prediction only, no metrics. Resumable.
cd "$(dirname "$0")"
for m in minicheck Qwen/Qwen3-4B-Instruct-2507 microsoft/Phi-4-mini-instruct; do
  echo "=== $m $(date -Is)"
  .venv/bin/python -u judge.py run --model "$m" --sets ragtruthgen --splits samples --out preds
done
echo "=== done $(date -Is)"
