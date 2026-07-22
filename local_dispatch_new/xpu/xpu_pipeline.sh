#!/usr/bin/env bash
# xpu_pipeline.sh — in-container XPU pipeline (quant → eval → uploads).
#
# Independent of lb_eval's CUDA auto.sh. Runs inside the XPU container with the
# lb_eval checkout mounted at /workspace/lb_eval and this xpu/ dir alongside.
#
# Inputs (env):
#   REQ_REL              request json path relative to the repo
#   LOCAL_RESULTS_DATASET / LOCAL_RUN_ID   dataset upload target + run id
#   HF_TOKENS / HF_UPLOAD_ORGS / ...       model upload settings (from config.env)
#   ZE_AFFINITY_MASK     reserved XPU cards (set by the docker run)
set -uo pipefail

REPO="/workspace/lb_eval"
XPU_DIR="/workspace/xpu"
cd "$REPO"

# Load committed non-secret config (HF orgs, eval tasks, etc.) + injected secrets.
if [ -f auto_quant/config.env ]; then
    set -a
    # shellcheck disable=SC1091
    source auto_quant/config.env
    set +a
fi

# ── Minimal log helpers (agent_fix_loop.sh expects these; auto.sh normally defines them) ──
if [ -t 1 ]; then C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[1m'; N='\033[0m';
else C=''; G=''; Y=''; R=''; B=''; N=''; fi
log_info()  { echo -e "${C}[xpu]${N} $*"; }
log_ok()    { echo -e "${G}[xpu]${N} $*"; }
log_warn()  { echo -e "${Y}[xpu]${N} $*"; }
log_error() { echo -e "${R}[xpu]${N} $*"; }
log_step()  { echo -e "\n${B}${C}═══════ $* ═══════${N}\n"; }

REQ_JSON="$REPO/${REQ_REL}"
if [ ! -f "$REQ_JSON" ]; then
    echo "[xpu-pipeline] ERROR: request json not found: $REQ_JSON" >&2
    exit 2
fi

# ── Parse request fields ─────────────────────────────────────────────────────
read_field() { python3 -c "import json,sys;print(json.load(open('$REQ_JSON')).get('$1','') or '')"; }
MODEL_ID="$(read_field model)"
SCHEME="$(read_field quant_scheme)"; [ -z "$SCHEME" ] && SCHEME="$(read_field scheme)"
METHOD="$(read_field method)"; [ -z "$METHOD" ] && METHOD="RTN"
EXPORT_FORMAT="$(read_field export_format)"; [ -z "$EXPORT_FORMAT" ] && EXPORT_FORMAT="auto_round"
VISIBLE="$(read_field cuda_visible_devices)"   # reused field: reserved card list

# Normalise scheme label (strip "INT4 (W4A16)" style to W4A16).
SCHEME="$(python3 -c "
s='''$SCHEME'''.strip()
for c in ('W4A16','MXFP4','NVFP4','W8A16','MXFP8'):
    if c in s.replace(' ','').upper(): print(c); break
else: print(s or 'W4A16')
")"

case "$METHOD" in
    RTN|"")        ITERS=0;   METHOD_SUFFIX="RTN";       MODEL_FREE=false ;;
    TUNING)        ITERS=200; METHOD_SUFFIX="Tuning";    MODEL_FREE=false ;;
    MODEL_FREE)    ITERS=0;   METHOD_SUFFIX="ModelFree"; MODEL_FREE=true ;;
    *)             ITERS=0;   METHOD_SUFFIX="$METHOD";   MODEL_FREE=false ;;
esac

NUM_XPU=1
if [ -n "$VISIBLE" ]; then
    NUM_XPU=$(awk -F',' '{print NF}' <<< "$VISIBLE")
fi

MODEL_SHORT="${MODEL_ID##*/}"
ARTIFACT="${MODEL_SHORT}-AutoRound-${SCHEME}-${METHOD_SUFFIX}"
RUN_OUTPUT_DIR="${XPU_DIR}/output/runs/${ARTIFACT}"
QUANT_DIR="${RUN_OUTPUT_DIR}/quantized_model"
EVAL_DIR="${RUN_OUTPUT_DIR}/lm_eval_results"
LOG_DIR="${RUN_OUTPUT_DIR}/logs"
mkdir -p "$RUN_OUTPUT_DIR" "$LOG_DIR"
cp "$REQ_JSON" "$RUN_OUTPUT_DIR/request.json" 2>/dev/null || true

echo "=== XPU pipeline ==="
echo "  model=$MODEL_ID  scheme=$SCHEME  method=$METHOD  export=$EXPORT_FORMAT"
echo "  artifact=$ARTIFACT  num_xpu=$NUM_XPU  ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-<unset>}  ONEAPI_DEVICE_SELECTOR=${ONEAPI_DEVICE_SELECTOR:-<unset>}"

pipeline_status="Finished"
FAILED_STEP=""

# ── Phase 1: install AutoRound XPU deps for the requested refs (best-effort) ──
# The image already has AutoRound; only override if the request pins a ref.
AUTO_ROUND_REF="$(read_field auto_round_ref)"
if [ -n "$AUTO_ROUND_REF" ] && [ "$AUTO_ROUND_REF" != "latest" ]; then
    echo "[xpu-pipeline] Overriding auto-round → $AUTO_ROUND_REF"
    if echo "$AUTO_ROUND_REF" | grep -qE '^[0-9]'; then
        pip install --no-cache-dir "auto-round==${AUTO_ROUND_REF}" 2>&1 | tail -3 || true
    else
        pip install --no-cache-dir "auto-round @ git+https://github.com/intel/auto-round.git@${AUTO_ROUND_REF}" 2>&1 | tail -3 || true
    fi
fi

# ── Agent fix-loop setup (REUSE lb_eval's device-agnostic loop; override XPU bits) ──
# Export everything the loop + wrappers read.
export MODEL_ID SCHEME METHOD ITERS EXPORT_FORMAT MODEL_FREE NUM_XPU
export QUANT_DIR EVAL_DIR RUN_OUTPUT_DIR XPU_DIR
export EVAL_TASKS="${EVAL_TASKS:-piqa,mmlu,hellaswag}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
export LESSONS_DIR="${LESSONS_DIR:-$REPO/auto_quant/lessons}"
export MAX_FIX_ATTEMPTS="${MAX_FIX_ATTEMPTS:-5}"
export REQUIRE_CUDA=false          # disable the CUDA regression guard (this is XPU)
export CLEANUP_STALE_GPU="${CLEANUP_STALE_GPU:-false}"

# Source the loop, then the XPU overrides (last-defined wins → replaces device funcs).
FIXLOOP="$REPO/auto_quant/phases/agent_fix_loop.sh"
USE_FIXLOOP=false
if [ -f "$FIXLOOP" ] && command -v openclaw >/dev/null 2>&1; then
    # shellcheck disable=SC1090
    if source "$FIXLOOP" 2>/dev/null && source "${XPU_DIR}/xpu_fixloop_overrides.sh" 2>/dev/null; then
        USE_FIXLOOP=true
        log_info "Agent fix-loop enabled (XPU overrides active, REQUIRE_CUDA=false)"
    fi
fi
[ "$USE_FIXLOOP" = "true" ] || log_warn "Agent fix-loop unavailable — running phases deterministically"

run_phase() {  # <name> <wrapper> [args...]
    local name="$1" wrapper="$2"; shift 2
    if [ "$USE_FIXLOOP" = "true" ]; then
        agent_fix_loop "$name" "$wrapper" "$@"
    else
        bash "$wrapper" "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"; return "${PIPESTATUS[0]}"
    fi
}

# ── Phase 2: quantize (with agent fix-loop) ──────────────────────────────────
if ! run_phase "quantize" "${XPU_DIR}/xpu_quantize_wrapper.sh"; then
    pipeline_status="Failed"; FAILED_STEP="quantize"
fi

# ── Phase 3: evaluate (only if quant succeeded, with agent fix-loop) ─────────
if [ "$pipeline_status" = "Finished" ]; then
    if ! run_phase "evaluate" "${XPU_DIR}/xpu_evaluate_wrapper.sh" "$QUANT_DIR"; then
        pipeline_status="Failed"; FAILED_STEP="evaluate"
    fi
fi

# ── Report (best-effort, reuse lb_eval generator) ────────────────────────────
python3 "$REPO/auto_quant/phases/generate_report.py" "$RUN_OUTPUT_DIR" 2>&1 | tail -5 || true

# ── Phase 4a: upload quantized model to HF model repo (success only) ─────────
if [ "$pipeline_status" = "Finished" ] && [ -d "$QUANT_DIR" ]; then
    echo "[xpu-pipeline] Uploading quantized model to Hugging Face..."
    python3 "$REPO/auto_quant/upload_model_hf.py" \
        "$QUANT_DIR" "$ARTIFACT" \
        --tokens "${HF_TOKENS:-}" \
        --orgs "${HF_UPLOAD_ORGS:-}" \
        --account-ids "${HF_ACCOUNT_IDS:-}" \
        --summary-json "${RUN_OUTPUT_DIR}/quant_summary.json" \
        --accuracy-json "${RUN_OUTPUT_DIR}/accuracy.json" \
        --usage-file "${HF_USAGE_FILE:-}" \
        --capacity-gb "${HF_ACCOUNT_CAPACITY_GB:-1000}" \
        --shared-ledger-enabled "${HF_SHARED_LEDGER_ENABLED:-false}" \
        --shared-ledger-repo "${HF_SHARED_LEDGER_REPO:-}" \
        --shared-ledger-token "${HF_SHARED_LEDGER_TOKEN:-}" \
        --shared-ledger-branch "${HF_SHARED_LEDGER_BRANCH:-main}" \
        2>&1 | tee "${LOG_DIR}/upload_hf.log" || echo "[xpu-pipeline] WARNING: model upload failed"
fi

# ── Bundle this run's lessons into the run dir so they reach the dataset ─────
if ls "${LESSONS_DIR}"/*.jsonl >/dev/null 2>&1; then
    mkdir -p "$RUN_OUTPUT_DIR/lessons"
    cp -f "${LESSONS_DIR}"/*.jsonl "$RUN_OUTPUT_DIR/lessons/" 2>/dev/null || true
fi

# ── Phase 4b: upload results to the independent HF dataset ────────────────────
echo "[xpu-pipeline] Uploading results to dataset ${LOCAL_RESULTS_DATASET}..."
pipeline_rc=0; [ "$pipeline_status" = "Finished" ] || pipeline_rc=1
dataset_rc=0
python3 "$REPO/_local_upload_results.py" \
    "$RUN_OUTPUT_DIR" \
    --dataset "${LOCAL_RESULTS_DATASET}" \
    --run-id "${LOCAL_RUN_ID}" \
    --pipeline-rc "$pipeline_rc" || dataset_rc=$?

echo "=== XPU pipeline done: status=$pipeline_status failed_step=${FAILED_STEP:-none} ==="
[ "$pipeline_status" = "Finished" ] || exit 1
exit "$dataset_rc"
