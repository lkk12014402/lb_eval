#!/usr/bin/env bash
# xpu_fixloop_overrides.sh — XPU device overrides for the reused agent fix-loop.
#
# Sourced AFTER lb_eval's phases/agent_fix_loop.sh so these definitions win
# (bash: last-defined function wins). Replaces the two CUDA-coupled functions
# with XPU equivalents. The CUDA regression guard is disabled via REQUIRE_CUDA=false
# (set by the caller), and an XPU guard is added inside build_fix_prompt wording.
#
# This keeps lb_eval's agent_fix_loop.sh UNMODIFIED (zero GPU-path risk) while
# giving the XPU pipeline the same fix → retry → lesson behaviour.

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

# ── Override: XPU regression guard (parity with the CUDA guard) ──────────────
# The generic loop's CUDA guard is disabled (REQUIRE_CUDA=false). Provide an XPU
# check the loop can call if REQUIRE_XPU=true; the loop doesn't call this directly,
# but build_fix_prompt below instructs the agent to keep XPU working.
xpu_is_available() {
    python3 -c "import torch,sys; sys.exit(0 if hasattr(torch,'xpu') and torch.xpu.is_available() else 1)" 2>/dev/null
}

# ── Override: build_fix_prompt with XPU wording (was CUDA-specific) ───────────
build_fix_prompt() {
    local phase="$1"
    local error_context="$2"
    local lessons="$3"
    local attempt="$4"
    local prior_block="${5:-}"

    local lessons_section=""
    if [ -n "${lessons}" ]; then
        lessons_section="
## Lessons from previous runs (you MAY use or ignore — verify relevance):
${lessons}
"
    fi
    local prior_section=""
    if [ -n "${prior_block}" ]; then
        prior_section="
## L1 automated triage (a pattern-based guess — you MAY be right to override it):
${prior_block}
"
    fi

    cat <<PROMPT
You are debugging a failed Intel **XPU** (Intel Arc Pro B60, oneAPI/Level-Zero) LLM
quantization/evaluation phase. Fix the ROOT CAUSE so the phase passes on XPU.

## Failed phase: ${phase}
## Attempt: ${attempt}

## Error output (tail):
\`\`\`
${error_context}
\`\`\`
${prior_section}${lessons_section}
## Respond with this protocol (fill every field):
VERDICT: <FIXABLE or UNFIXABLE>
ROOT_CAUSE: <1-2 lines — the actual cause, not the symptom>
FIX_TIER: <1=env/deps, 2=model custom code, 3=our script>
FIX_PLAN: <3 lines max — what you will change and why it fixes the ROOT CAUSE>
SMOKE_TEST: <ONE fast command (NOT the full phase) that proves the fix works>

## Rules:
- If VERDICT is UNFIXABLE: print the block and STOP (no wasted retries).
- Prefer the LOWEST FIX_TIER. Patching source is a last resort.
- After applying the fix, RUN your SMOKE_TEST and show its output before finishing.
- **XPU IS REQUIRED.** This box has Intel XPU (no NVIDIA CUDA). The re-run MUST run on XPU.
  - Use \`device_map="xpu"\` / \`device="xpu:0"\`; never \`cpu\` or \`cuda\`.
  - Select devices with \`ZE_AFFINITY_MASK\` / \`ONEAPI_DEVICE_SELECTOR=level_zero:...\`,
    NOT \`CUDA_VISIBLE_DEVICES\`.
  - **Do NOT reinstall/downgrade torch** — the image ships \`torch==2.11.0+xpu\`. A plain
    \`pip install torch\` pulls the CUDA build and BREAKS XPU. If you must install packages,
    use \`--no-deps\` or a torch constraint so torch-xpu stays intact.
  - After any install, verify: \`python3 -c "import torch; assert torch.xpu.is_available()"\`.
- vLLM on XPU: CUDA graphs are unsupported → keep \`enforce_eager=True\`.
- This is attempt ${attempt}. Earlier attempts are in your session history — do NOT repeat a
  fix that already failed; try a different hypothesis.

## Patching model custom code:
If the traceback shows files under \`~/.cache/huggingface/modules/transformers_modules/\`,
that is the MODEL'S downloaded code — YOU CAN AND SHOULD EDIT IT (dtype/device/regex/import fixes).
Common: replace \`.float()\` with \`.to(other.dtype)\`; add \`device=hidden_states.device\`;
for \`.cuda()\` / \`device='cuda'\` in model code on this XPU box, change to \`.to("xpu")\`.

## Constraints:
- Do NOT modify evaluation tasks or the expected output format.
- Keep fixes minimal and targeted.
- Working directory: ${RUN_OUTPUT_DIR}
- Model: ${MODEL_ID}
PROMPT
}
