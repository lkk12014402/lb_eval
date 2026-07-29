#!/usr/bin/env bash
# XPU evaluate wrapper — env-driven, re-runnable by the agent fix-loop.
# Mirrors phases/evaluate.sh's entry contract but for the standalone XPU evaluator.
set -euo pipefail

XPU_DIR="${XPU_DIR:-/workspace/xpu}"
MODEL_PATH="${1:-${QUANT_DIR:?QUANT_DIR or arg required}}"
EVAL_TASKS="${EVAL_TASKS:-piqa,mmlu,hellaswag}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
NUM_XPU="${NUM_XPU:-1}"
EVAL_DIR="${EVAL_DIR:?EVAL_DIR required}"

echo "=== XPU Phase: Evaluate ==="
echo "  model=$MODEL_PATH tasks=$EVAL_TASKS batch=$EVAL_BATCH_SIZE num_xpu=$NUM_XPU"
exec python3 "${XPU_DIR}/xpu_evaluate.py" \
    --model-path "$MODEL_PATH" \
    --tasks "$EVAL_TASKS" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --num-gpus "$NUM_XPU" \
    --output-dir "$EVAL_DIR"
