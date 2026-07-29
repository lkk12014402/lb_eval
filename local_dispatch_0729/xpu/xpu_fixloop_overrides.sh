#!/usr/bin/env bash
# xpu_fixloop_overrides.sh — XPU device overrides for the reused agent fix-loop.
#
# Sourced AFTER agent_fix_loop.sh so these definitions win (bash: last-defined
# function wins). Replaces the CUDA-coupled cleanup helper with an XPU equivalent.
# The fix PROMPT is NOT overridden here anymore: agent_fix_loop.sh's build_fix_prompt
# is device-aware via AGENT_DEVICE_KIND (the XPU pipeline exports it =xpu), so the
# structured diagnosis contract (COMPONENT/ERROR_CLASS/ROOT_CAUSE_HYPOTHESIS/FIX_TIER)
# is byte-for-byte identical to the GPU path — only the device wording differs.
# The CUDA regression guard is disabled via REQUIRE_CUDA=false (set by the caller).
#
# This keeps lb_eval's production scripts UNMODIFIED (zero GPU-path risk).

# ── Override: free XPU memory before a re-run (was nvidia-smi based) ──────────
cleanup_stale_gpu_procs() {
    # Best-effort: XPU has no reliable per-process VRAM API like nvidia-smi.
    # xpu-smi may exist; if so, log utilisation, but do not kill (shared host).
    if command -v xpu-smi >/dev/null 2>&1; then
        xpu-smi stats 2>/dev/null | head -5 || true
    fi
    # Give any prior torch-xpu workers a moment to release device handles.
    sleep 2
    return 0
}

# ── XPU device availability check (parity with the CUDA guard) ───────────────
# The generic loop's CUDA guard is disabled (REQUIRE_CUDA=false). Provide an XPU
# check the loop can call if REQUIRE_XPU=true. The fix prompt itself is now the
# canonical device-aware build_fix_prompt (AGENT_DEVICE_KIND=xpu) from
# agent_fix_loop.sh — we no longer override it here, so the structured protocol
# (COMPONENT/ERROR_CLASS/ROOT_CAUSE_HYPOTHESIS/FIX_TIER) stays identical to GPU.
xpu_is_available() {
    python3 -c "import torch,sys; sys.exit(0 if hasattr(torch,'xpu') and torch.xpu.is_available() else 1)" 2>/dev/null
}

