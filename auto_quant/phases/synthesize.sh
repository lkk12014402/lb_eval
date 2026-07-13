#!/bin/bash
# synthesize.sh — Agent-synthesized run narrative (G4 tried-solutions + G6 report).
#
# Sourceable library. Provides:
#   synthesize_run_summary <run_output_dir> <phase_status_desc>
#     → writes <run_dir>/resolution_summary.md (readable narrative of the run:
#       what happened, what was tried each fix attempt & why it failed, the root
#       cause, and the final resolution or why it's unresolved).
#
# Reuses the pluggable agent (run_agent_fix) at the CHEAP tier (e.g. MiniMax) —
# synthesis is a summarization task, not a hard fix, so we never spend Opus on it.
# Fully fault-tolerant: if no backend/inputs are available it logs a warning and
# returns 0 (never breaks the pipeline).

[[ -n "${_SYNTHESIZE_SOURCED:-}" ]] && return 0
_SYNTHESIZE_SOURCED=1


# _collect_attempt_context <run_dir> — dump attempt logs + lessons + diagnosis into
# a compact context string (bounded) for the synthesis prompt.
_collect_attempt_context() {
    local run_dir="$1"
    local fixes_dir="${run_dir}/logs/agent_fixes"
    local out=""

    if [[ -d "${fixes_dir}" ]]; then
        local phase_dir attempt
        for phase_dir in "${fixes_dir}"/*/; do
            [[ -d "${phase_dir}" ]] || continue
            out+=$'\n=== PHASE: '"$(basename "${phase_dir}")"$' ==='$'\n'
            for attempt in "${phase_dir}"/attempt_*.log; do
                [[ -f "${attempt}" ]] || continue
                out+=$'\n--- '"$(basename "${attempt}" .log)"$' ---'$'\n'
                # Keep the structured diagnosis lines + a tail; cap size per attempt.
                out+="$(grep -aiE 'COMPONENT:|ERROR_CLASS:|ROOT_CAUSE|EVIDENCE|VERDICT:|FIX_TIER:|FIX_PLAN:|SMOKE_TEST:|SOLUTION:' "${attempt}" 2>/dev/null | head -30)"
                out+=$'\n… (tail) …\n'
                out+="$(tail -15 "${attempt}" 2>/dev/null)"
                out+=$'\n'
            done
        done
    fi

    # Failure diagnosis JSONs (if any).
    local diag
    for diag in "${run_dir}"/failure_diagnosis_*.json; do
        [[ -f "${diag}" ]] || continue
        out+=$'\n=== DIAGNOSIS: '"$(basename "${diag}")"$' ==='$'\n'
        out+="$(head -c 2000 "${diag}" 2>/dev/null)"$'\n'
    done

    # Hard cap the whole context so the prompt stays reasonable.
    printf '%s' "${out}" | head -c 12000
}


# synthesize_run_summary <run_dir> <status_desc>
synthesize_run_summary() {
    local run_dir="$1"
    local status_desc="${2:-unknown}"
    local out_md="${run_dir}/resolution_summary.md"

    # Skip cleanly if synthesis is disabled or no agent dispatcher is available.
    if [[ "${SYNTHESIZE_SUMMARY:-true}" != "true" ]]; then
        return 0
    fi
    if ! declare -f run_agent_fix >/dev/null 2>&1; then
        log_warn "synthesize: run_agent_fix not available; skipping narrative"
        return 0
    fi

    local quant_summary="${run_dir}/quant_summary.json"
    local accuracy="${run_dir}/accuracy.json"
    local qs_head acc_head context
    qs_head="$( [[ -f "${quant_summary}" ]] && head -c 2500 "${quant_summary}" || echo '{}' )"
    acc_head="$( [[ -f "${accuracy}" ]] && head -c 1500 "${accuracy}" || echo '{}' )"
    context="$(_collect_attempt_context "${run_dir}")"

    # If there were no fix attempts and the run succeeded cleanly, a short summary
    # is still useful but doesn't need the agent — skip to keep it cheap.
    if [[ -z "${context}" && "${status_desc}" == *Finished* ]]; then
        log_info "synthesize: clean run, no fix attempts — skipping agent narrative"
        return 0
    fi

    local prompt
    prompt="$(cat <<EOF
You are writing a concise, high-signal SUMMARY of one automated quantization run.
Be factual and specific; do NOT invent details not present in the inputs. Output
GitHub-flavored Markdown only (no preamble). Use these exact sections:

## Executive summary
2-3 sentences: what model/scheme/method, and the final outcome (${status_desc}).

## What was tried
A bullet per fix attempt IN ORDER: the error class, the hypothesis/root cause, the
fix tier + action, and why it failed (or that it succeeded). If there were no fix
attempts, write "No agent fixes were needed."

## Root cause
The single most likely underlying cause, traced to a component (our code / a
dependency / the model's own code / data / environment). One short paragraph.

## Resolution
What finally worked (the concrete fix), OR — if unresolved — why, and the most
promising next step. One short paragraph.

## Recipe note
One sentence a future run could reuse (e.g. "for <arch>, exclude <layers> / use
<scheme>+<layer_config>"), or "n/a".

── RUN FACTS (quant_summary.json) ──
${qs_head}

── ACCURACY (accuracy.json) ──
${acc_head}

── FIX ATTEMPTS / DIAGNOSIS CONTEXT ──
${context:-<none>}
EOF
)"

    # Always synthesize on the CHEAP tier — this is summarization, not a hard fix.
    if declare -f agent_tier_activate >/dev/null 2>&1; then
        agent_tier_activate 0 >/dev/null 2>&1 || true
    fi

    log_info "Synthesizing run resolution summary (cheap tier)..."
    local synth_log="${run_dir}/logs/synthesis.log"
    mkdir -p "$(dirname "${synth_log}")"
    # Reuse the agent dispatcher; its stdout (the narrative) is teed to synth_log.
    AGENT_TIMEOUT="${SYNTH_TIMEOUT:-240}" run_agent_fix "${prompt}" "${synth_log}" "synth_$$_$(date +%s)" || true

    # Extract the markdown (from the first '## ' header onward) into the summary file.
    if [[ -s "${synth_log}" ]] && grep -qaE '^##[[:space:]]' "${synth_log}"; then
        sed -n '/^##[[:space:]]/,$p' "${synth_log}" > "${out_md}"
        log_ok "Wrote resolution summary: ${out_md}"
    else
        log_warn "synthesize: agent produced no usable markdown; skipping ${out_md}"
        return 0
    fi
    return 0
}
