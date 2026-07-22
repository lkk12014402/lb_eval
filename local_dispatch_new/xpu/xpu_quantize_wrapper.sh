#!/usr/bin/env bash
# XPU quantize wrapper — env-driven, re-runnable by the agent fix-loop.
# Mirrors phases/quantize_wrapper.sh but for the standalone XPU quantizer.
set -euo pipefail

XPU_DIR="${XPU_DIR:-/workspace/xpu}"
MODEL_ID="${MODEL_ID:?MODEL_ID required}"
SCHEME="${SCHEME:-W4A16}"
ITERS="${ITERS:-0}"
EXPORT_FORMAT="${EXPORT_FORMAT:-auto_round}"
QUANT_DIR="${QUANT_DIR:?QUANT_DIR required}"
NUM_XPU="${NUM_XPU:-1}"
MODEL_FREE="${MODEL_FREE:-false}"

ARGS=(--model "$MODEL_ID" --scheme "$SCHEME" --iters "$ITERS"
      --export_format "$EXPORT_FORMAT" --output_dir "$QUANT_DIR" --num_gpus "$NUM_XPU")
[ "$MODEL_FREE" = "true" ] && ARGS+=(--model_free)

echo "=== XPU Phase: Quantize ==="
echo "  model=$MODEL_ID scheme=$SCHEME iters=$ITERS export=$EXPORT_FORMAT model_free=$MODEL_FREE num_xpu=$NUM_XPU"
exec python3 "${XPU_DIR}/xpu_quantize.py" "${ARGS[@]}"
