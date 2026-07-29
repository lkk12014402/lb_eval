#!/bin/bash
# report_agent.sh — agent-generated run report (copilot + MiniMax).
#
# Replaces the fragile deterministic generate_report.py output with a report the
# agent writes from the actual run history (phase logs, per-attempt structured
# diagnosis, tier-escalation trail, lessons, quant/accuracy JSON). Falls back to the
# deterministic generator when the agent is unavailable or produces nothing usable.
#
# Public entry point:
#   generate_run_report_agent <run_dir> <out_md>   → 0 on success (out_md written)
#
# Config:
#   REPORT_AGENT           1 (default) to try the agent; 0 to force deterministic only
#   REPORT_AGENT_TIMEOUT   seconds for the copilot call (default 600)
#   MINIMAX_API_KEY        required (copilot BYOK, no GitHub auth)
#   LB_EVAL_DIR            lb_eval checkout (for the deterministic fallback)

[[ -n "${_REPORT_AGENT_SOURCED:-}" ]] && return 0
_REPORT_AGENT_SOURCED=1

command -v log_info >/dev/null 2>&1 || log_info() { echo "[report] $*"; }
command -v log_warn >/dev/null 2>&1 || log_warn() { echo "[report] $*"; }

# Bring in the copilot BYOK helpers (_copilot_setup_byok, _copilot_gen_uuid). Guarded
# source so it is a no-op if the fix-loop already sourced agent_backends.sh.
_REPORT_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! declare -f _copilot_setup_byok >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "${_REPORT_AGENT_DIR}/agent_backends.sh" 2>/dev/null || true
fi

# ── _collect_run_history <run_dir> → compact text blob for the prompt ─────────
# Pulls the request, quant/accuracy JSON, per-attempt structured diagnosis (the
# COMPONENT/ERROR_CLASS/ROOT_CAUSE_HYPOTHESIS/FIX_TIER/FIX_PLAN/VERDICT blocks the
# agents emit), retry pass/fail, and the phase-log tails — bounded so the prompt
# stays small even on long runs.
_collect_run_history() {
    local run_dir="$1"
    RUN_DIR="${run_dir}" python3 <<'PYEOF' 2>/dev/null
import os, json, glob, re

run = os.environ["RUN_DIR"]
out = []

def add(s=""):
    out.append(s)

def read(path, limit=4000):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-limit:]
    except OSError:
        return ""

def jload(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except Exception:
        return None

# 1) Request / config
for name in ("request.json",):
    for p in glob.glob(os.path.join(run, "**", name), recursive=True):
        d = jload(p)
        if d:
            add("## REQUEST"); add(json.dumps(d, ensure_ascii=False, indent=2)); add()
            break

# 2) Quant summary + accuracy
for label, name in (("QUANT_SUMMARY", "quant_summary.json"),
                    ("ACCURACY", "accuracy.json")):
    p = os.path.join(run, name)
    d = jload(p)
    if d is not None:
        add(f"## {label}"); add(json.dumps(d, ensure_ascii=False, indent=2)); add()

# 3) Per-phase agent-fix attempts: structured diagnosis + retry outcome
LABELS = ["COMPONENT", "ERROR_CLASS", "ROOT_CAUSE_HYPOTHESIS", "EVIDENCE_RESULT",
          "VERDICT", "UNFIXABLE_REASON", "FIX_TIER", "FIX_PLAN", "SMOKE_TEST"]

def struct_block(text):
    """Pull the labeled fields from an attempt log (last occurrence of each)."""
    found = {}
    others = "|".join(LABELS)
    for name in LABELS:
        rest = "|".join(l for l in LABELS if l != name)
        for m in re.finditer(
            rf'^[\s>*\-]*{name}\s*:\s*(.*?)(?=^[\s>*\-]*(?:{rest})\s*:|\Z)',
            text, re.MULTILINE | re.DOTALL):
            val = re.sub(r'`', '', m.group(1))
            val = re.sub(r'\*+', '', val)
            val = re.sub(r'\s+', ' ', val).strip()
            if val and not (val.startswith('<') and val.endswith('>')):
                found[name] = val[:400]
    return found

fixes_root = os.path.join(run, "logs", "agent_fixes")
if os.path.isdir(fixes_root):
    for phase in sorted(os.listdir(fixes_root)):
        pdir = os.path.join(fixes_root, phase)
        if not os.path.isdir(pdir):
            continue
        add(f"## AGENT FIX TRAIL — phase: {phase}")
        attempts = sorted(glob.glob(os.path.join(pdir, "attempt_*.log")),
                          key=lambda p: int(re.sub(r'\D', '', os.path.basename(p)) or 0))
        for a in attempts:
            n = re.sub(r'\D', '', os.path.basename(a)) or "?"
            fields = struct_block(read(a, 20000))
            add(f"### Attempt {n}")
            if fields:
                for k in LABELS:
                    if k in fields:
                        add(f"- {k}: {fields[k]}")
            else:
                add("- (no structured diagnosis captured)")
            retry = os.path.join(pdir, f"retry_{n}.log")
            if os.path.exists(retry):
                rt = read(retry, 1500)
                ok = "FAILED" not in rt and "Traceback" not in rt and "Error" not in rt
                add(f"- Re-run: {'PASSED' if ok else 'still failing'}")
            else:
                add("- Re-run: not executed")
        add()

# 4) Phase log tails (final state)
logs_dir = os.path.join(run, "logs")
if os.path.isdir(logs_dir):
    for lg in sorted(glob.glob(os.path.join(logs_dir, "*.log"))):
        base = os.path.basename(lg)
        add(f"## PHASE LOG TAIL — {base}")
        add(read(lg, 2500)); add()

# 5) Lessons for this run
for p in sorted(glob.glob(os.path.join(run, "lessons", "*.jsonl"))):
    add(f"## LESSONS — {os.path.basename(p)}")
    for line in read(p, 4000).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            add(f"- [{d.get('status','?')}] {d.get('phase','?')}: {d.get('solution','')[:200]}")
        except Exception:
            pass
    add()

print("\n".join(out))
PYEOF
}

# ── _report_template → the markdown skeleton the agent must fill ──────────────
_report_template() {
    cat <<'TMPL'
# Pipeline Report: <MODEL_ID>

**Generated:** <UTC timestamp>

## Status
| Field | Value |
|-------|-------|
| Overall | <Succeeded ✅ / Quant Failed ❌ / Eval Failed ❌> |
| Model | `<model id>` |
| Architecture | <arch / model_type / Dense or MoE, else "N/A"> |
| Scheme | `<scheme>` |
| Method | <RTN/Tuning/ModelFree> (iters=<n or N/A>) |
| Export Format | <auto_round/...> |
| Quant Duration | <e.g. 3m12s or N/A> |

## Root Cause Analysis
<If any phase failed and was worked on: 2-6 sentences. State the ACTUAL root cause the
agents converged on (component + file:line + why), NOT the surface symptom. If the run
succeeded with no fixes, write "No failures — clean run.">

## Fix Timeline (tier escalation)
<A chronological table of every agent fix attempt. One row per attempt.>
| Attempt | Tier (backend+model) | Error class | Verdict | Fix (1 line) | Re-run |
|--------:|----------------------|-------------|---------|--------------|--------|
| 1 | openclaw+MiniMax | ... | FIXABLE | ... | still failing |
| ... | ... | ... | ... | ... | ... |
<Reflect the REAL tier used per attempt from the history (openclaw+MiniMax → copilot+MiniMax
→ copilot+Opus4.8). If a fix finally worked, the last row's Re-run = PASSED.>

## Evaluation Results
<Accuracy table if accuracy.json present, else "No evaluation results available.">

## Phase Execution
| Phase | Result |
|-------|--------|
| setup_env | <✅ / ⏭️ skipped / ❌> |
| quantize | <✅ passed / ✅ fixed on attempt N / ❌ failed after N attempts / ⏭️ skipped> |
| evaluate | <same convention> |

## Lessons Learned
<Bullet the run-local lessons (status + one-line takeaway). Omit the section if none.>

---
*Report generated by report_agent (copilot+MiniMax) from run history.*
TMPL
}

# ── generate_run_report_agent <run_dir> <out_md> ─────────────────────────────
generate_run_report_agent() {
    local run_dir="$1"
    local out_md="$2"

    if [[ "${REPORT_AGENT:-1}" != "1" ]]; then
        return 1   # disabled → caller uses the deterministic generator
    fi
    if ! command -v copilot >/dev/null 2>&1; then
        log_warn "report_agent: copilot CLI not found — using deterministic generator"
        return 1
    fi
    if [[ -z "${MINIMAX_API_KEY:-}${COPILOT_PROVIDER_API_KEY:-}" ]]; then
        log_warn "report_agent: no MiniMax key — using deterministic generator"
        return 1
    fi

    # Activate copilot + MiniMax BYOK (no GitHub auth needed).
    export COPILOT_MINIMAX=1
    export COPILOT_MODEL="${MINIMAX_MODEL:-MiniMax-M3}"
    _copilot_setup_byok || { log_warn "report_agent: BYOK setup failed"; return 1; }

    local history template prompt
    history="$(_collect_run_history "${run_dir}")"
    if [[ -z "${history}" ]]; then
        log_warn "report_agent: no run history collected — using deterministic generator"
        return 1
    fi
    template="$(_report_template)"

    prompt="You are generating the FINAL pipeline run report for a model quantization run.

Write a COMPLETE, accurate markdown report to this exact path:
  ${out_md}

Use the TEMPLATE below (keep its section order and headers). Fill EVERY placeholder from
the RUN HISTORY — never leave a raw <...> placeholder. Be precise and factual: derive the
overall status, the root cause, and the per-attempt tier/verdict/fix strictly from the
history. Do not invent numbers. If a value is genuinely absent, write 'N/A'.

Key rules:
- The 'Fix Timeline' MUST have one row per attempt found in the history, showing the REAL
  tier used (tier 0 = openclaw+MiniMax, tier 1 = copilot+MiniMax, tier 2 = copilot+Opus4.8).
- 'Root Cause Analysis' must state the actual converged root cause (component + file:line +
  why), not the surface error message.
- Overall status: 'Succeeded ✅' only if the failing phase was ultimately fixed and re-ran
  clean; otherwise the appropriate Failed state.
- Write ONLY the report file; do not print the report to stdout. When done, print the single
  line: REPORT_WRITTEN

===== TEMPLATE =====
${template}

===== RUN HISTORY =====
${history}
"

    local timeout="${REPORT_AGENT_TIMEOUT:-600}"
    local sess; sess="$(_copilot_gen_uuid 2>/dev/null || echo report-$$)"
    local logdir="${run_dir}/logs"; mkdir -p "${logdir}"
    local agent_log="${logdir}/report_agent.log"

    log_info "report_agent: generating report with copilot+MiniMax (timeout=${timeout}s)..."
    rm -f "${out_md}.agenttmp" 2>/dev/null || true

    timeout "${timeout}" copilot -p "${prompt}" \
        --allow-all-tools \
        --no-color \
        --session-id "${sess}" \
        --add-dir "${run_dir}" \
        --add-dir "/tmp" \
        --log-level error \
        >"${agent_log}" 2>&1 || {
        local rc=$?
        [ $rc -eq 124 ] && log_warn "report_agent: timed out after ${timeout}s"
    }

    # Validate the agent actually wrote a real report (non-trivial + has our headers).
    if [[ -s "${out_md}" ]] \
       && grep -q "Pipeline Report" "${out_md}" 2>/dev/null \
       && grep -qi "Fix Timeline\|Root Cause\|Phase Execution" "${out_md}" 2>/dev/null; then
        local n; n=$(wc -l < "${out_md}" 2>/dev/null || echo 0)
        log_info "report_agent: wrote ${out_md} (${n} lines)"
        return 0
    fi

    log_warn "report_agent: agent did not produce a valid report — using deterministic generator"
    return 1
}
