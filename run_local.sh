#!/usr/bin/env bash
# Full local inference: every model on every set + calib splits. Resumable (judge.py skips done ids,
# prefetch.py resumes partial downloads). Exit 0 only if all outputs are complete.
cd "$(dirname "$0")"
mkdir -p out preds
PY=.venv/bin/python
MODELS="minicheck Qwen/Qwen3-4B-Instruct-2507 microsoft/Phi-4-mini-instruct"  # cut 2026-10-05: ~12 s/item on 1660 Ti => ~4.5 h per causal model; Qwen2.5-3B and SmolLM3 deferred
SETS="ragtruth,wice,xsum,cnn,expertqa"
{
  echo "=== run_local start $(date -Is)"
  for i in 1 2 3; do
    $PY prefetch.py && break
    echo "prefetch attempt $i incomplete"
  done
  for m in $MODELS; do
    echo "=== $m $(date -Is)"
    $PY -u judge.py run --model "$m" --sets "$SETS" --splits samples,calib --out preds || echo "run failed for $m"
  done
  echo "=== check $(date -Is)"
} >> out/infer.log 2>&1
$PY judge.py check --models "$(echo $MODELS | tr ' ' ',')" --sets "$SETS" --out preds >> out/infer.log 2>&1
rc=$?
tail -n 3 out/infer.log
exit $rc
