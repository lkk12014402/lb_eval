#!/usr/bin/env bash
# xpu_container_bootstrap.sh — runs INSIDE the XPU container before the pipeline.
#
# Mirrors the CUDA container bootstrap: provision OpenClaw (+ optional MiniMax key),
# inject secrets/proxy into config.env, then hand off to the XPU pipeline. Uses only
# runtime env vars so it can be staged as a static file.
set -uo pipefail
cd /workspace/lb_eval

if [ "${DO_OPENCLAW:-1}" = "1" ]; then
    echo "== xpu-container: Set Up OpenClaw =="
    cp -r openclaw_config/ /root/.openclaw/ 2>/dev/null || true
    if [ -f .azure-pipelines/scripts/sync_minimax_key.py ]; then
        python3 .azure-pipelines/scripts/sync_minimax_key.py \
            --token="${MINIMAX_API_KEY:-}" \
            --path=/root/.openclaw/agents/main/agent/auth-profiles.json || true
    fi
fi

echo "== xpu-container: Update Config (inject secrets into config.env) =="
SETS=()
[ -n "${HF_TOKENS:-}" ]              && SETS+=(--set "HF_TOKENS=${HF_TOKENS}")
[ -n "${MINIMAX_API_KEY:-}" ]        && SETS+=(--set "MINIMAX_API_KEY=${MINIMAX_API_KEY}")
[ -n "${GIT_TOKEN:-}" ]              && SETS+=(--set "GIT_TOKEN=${GIT_TOKEN}")
[ -n "${LB_STORAGE_BLOB_TOKEN:-}" ]  && SETS+=(--set "LB_STORAGE_BLOB_TOKEN=${LB_STORAGE_BLOB_TOKEN}")
if [ "${#SETS[@]}" -gt 0 ] && [ -f .azure-pipelines/scripts/update_config_env.py ]; then
    python3 .azure-pipelines/scripts/update_config_env.py \
        --output /workspace/lb_eval/auto_quant/config.env "${SETS[@]}"
fi

echo "== xpu-container: run XPU pipeline =="
exec bash /workspace/xpu/xpu_pipeline.sh
