# Local-dispatch XPU runtime image.
#
# Base: Intel's B60/B70-validated vLLM XPU image (torch 2.11+xpu, vLLM 0.21,
# oneAPI + compute-runtime that actually enumerates 8 Arc Pro B60 cards on this
# host — the upstream vllm/vllm-openai-xpu:nightly reports device_count()==0 here).
#
# On top of the base we add: AutoRound (XPU quant), lm-eval, Node.js + OpenClaw +
# GitHub Copilot CLI — matching the CUDA agent image so the agent fix-loop and
# recipe/report generation behave identically. OpenClaw config is injected at
# RUNTIME from lb_eval/openclaw_config (same as the GPU path), not baked in.
ARG XPU_BASE_IMAGE=intel/llm-scaler-vllm:0.21.0-b1
FROM ${XPU_BASE_IMAGE}

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TQDM_MININTERVAL=120 \
    DEVICE=xpu \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# The base image bakes an Intel-internal proxy into ENV; clear it so the B60 host's
# direct network access is used (overridable at build/run time).
ENV http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

ARG http_proxy=""
ARG https_proxy=""

# ── System deps (git for clone, jq for JSON, node prereqs) ───────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git jq curl ca-certificates xz-utils numactl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── AutoRound (XPU) + eval deps into the base venv (/opt/venv) ───────────────
# torch is already installed with XPU support in the base image; never reinstall
# it (a plain `pip install torch` would pull the CUDA build). Use --no-deps-safe
# installs that respect the existing torch.
ARG AUTO_ROUND_REF=latest
RUN python -c "import torch; print('base torch', torch.__version__)" && \
    if [ "${AUTO_ROUND_REF}" = "latest" ] || [ -z "${AUTO_ROUND_REF}" ]; then \
        pip install --no-cache-dir "auto-round"; \
    elif echo "${AUTO_ROUND_REF}" | grep -qE '^[0-9]'; then \
        pip install --no-cache-dir "auto-round==${AUTO_ROUND_REF}"; \
    else \
        pip install --no-cache-dir "auto-round @ git+https://github.com/intel/auto-round.git@${AUTO_ROUND_REF}"; \
    fi

RUN pip install --no-cache-dir \
        "lm-eval>=0.4.10" \
        transformers accelerate datasets \
        loguru sentencepiece protobuf hf_transfer

# ── Node.js v22.20 + OpenClaw + GitHub Copilot CLI (agent backends) ──────────
# openclaw>=2026.6.8 requires Node >= 22.19; keep this pinned at/above that floor.
RUN curl -fsSL https://nodejs.org/dist/v22.20.0/node-v22.20.0-linux-x64.tar.xz \
        | tar -xJ --strip-components=1 -C /usr/local
#RUN npm install -g openclaw@2026.3.24 && npm cache clean --force
RUN npm install -g openclaw@2026.6.8 && npm cache clean --force
RUN npm install -g @github/copilot && npm cache clean --force

# ── Verify the XPU stack is intact ──────────────────────────────────────────
RUN python -c "import torch, auto_round, lm_eval, transformers; \
print('torch', torch.__version__); \
print('auto_round', auto_round.__version__); \
print('lm_eval', lm_eval.__version__)"

WORKDIR /workspace
ENTRYPOINT ["/bin/bash"]
