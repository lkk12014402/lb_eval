#!/bin/bash
# agent_backends.sh — Pluggable agent backends for the fix loop.
#
# Decouples "WHICH agent runs the fix" from "the fix-loop orchestration" so the
# OpenClaw and Copilot backends can be optimized independently.
#
# Public entry point (called by agent_fix_loop.sh):
#   run_agent_fix <prompt> <log_file> <session_id>
#     → dispatches to the backend selected by ${AGENT_BACKEND} (default: openclaw).
#
# Each backend MUST:
#   * run the agent non-interactively with the given prompt,
#   * tee the agent's textual output (which contains the labeled diagnosis fields
#     ROOT_CAUSE:/VERDICT:/FIX_TIER:/… that the prompt asks for) into <log_file>,
#   * be self-contained so downstream parsing (extract_agent_analysis, VERDICT
#     grep, extract_agent_field) works identically regardless of backend.
#
# Selection & config (from config.env / environment):
#   AGENT_BACKEND         openclaw | copilot        (default: openclaw)
#   AGENT_TIMEOUT         per-attempt timeout seconds (default: 600)
#   # OpenClaw:
#   OPENCLAW_SESSIONS_DIR path to session jsonl dir
#   # Copilot (settings.json in the image provides defaults; these OVERRIDE):
#   COPILOT_GITHUB_TOKEN  headless auth token (env-var auth, like MINIMAX key)
#   COPILOT_MODEL         override ~/.copilot/settings.json "model" (optional)
#   COPILOT_EFFORT        override reasoning effort (optional)
#   COPILOT_ADD_DIRS      space-separated extra dirs the agent may access
#                         (default: the pipeline SCRIPT_DIR)

# Guard against double-source
[[ -n "${_AGENT_BACKENDS_SOURCED:-}" ]] && return 0
_AGENT_BACKENDS_SOURCED=1


# ═══════════════════════════════════════════════════════════════════
# Escalation ladder — cheap→strong model tiers.
#
# AGENT_TIERS is a space-separated list of tier NAMES tried in order. The fix
# loop starts at tier 0 (cheapest) and escalates to the next tier when the
# current one stalls (drift) or the agent emits `VERDICT: ESCALATE`.
#
# Default ladder: MiniMax-M3 (cheap) → Opus 4.8 (strong), both via Copilot.
#   AGENT_TIERS="minimax opus"
#
# Built-in tier names (agent_tier_activate maps name → backend + model env):
#   minimax   → copilot backend, BYOK MiniMax-M3 (reuses MINIMAX_API_KEY)
#   opus      → copilot backend, Opus 4.8 (COPILOT_MODEL=claude-opus-4.8)
#   sonnet    → copilot backend, Claude Sonnet 4.5
#   openclaw  → openclaw backend (its own config/model)
#   copilot   → copilot backend, whatever COPILOT_* is already configured
# Unknown names fall back to: copilot backend with COPILOT_MODEL=<name>.
# ═══════════════════════════════════════════════════════════════════
AGENT_TIERS="${AGENT_TIERS:-minimax opus}"

agent_tiers_count() {
    # shellcheck disable=SC2086
    set -- ${AGENT_TIERS}
    echo $#
}

# agent_tier_name <idx> → prints the tier name at 0-based index
agent_tier_name() {
    local idx="$1" i=0 t
    for t in ${AGENT_TIERS}; do
        [ "$i" -eq "$idx" ] && { echo "$t"; return 0; }
        i=$((i + 1))
    done
    return 1
}

# agent_tier_activate <idx> — set AGENT_BACKEND + model env for that tier.
# Toggles Copilot BYOK (MiniMax) vs GitHub (Opus/Sonnet) mode as needed.
agent_tier_activate() {
    local idx="$1"
    local name
    name=$(agent_tier_name "$idx") || return 1
    case "${name,,}" in
        minimax|minimax-m3|minimax_m3)
            export AGENT_BACKEND=copilot
            export COPILOT_MINIMAX=1
            export COPILOT_MODEL="${MINIMAX_MODEL:-MiniMax-M3}"
            unset COPILOT_EFFORT
            ;;
        opus|opus-4.8|claude-opus-4.8)
            export AGENT_BACKEND=copilot
            # Opus is served via GitHub Copilot (or a non-MiniMax provider), so
            # turn OFF the MiniMax BYOK shortcut for this tier.
            unset COPILOT_MINIMAX COPILOT_PROVIDER_BASE_URL
            export COPILOT_MODEL="${OPUS_MODEL:-claude-opus-4.8}"
            export COPILOT_EFFORT="${OPUS_EFFORT:-high}"
            ;;
        sonnet|sonnet-4.5|claude-sonnet-4.5)
            export AGENT_BACKEND=copilot
            unset COPILOT_MINIMAX COPILOT_PROVIDER_BASE_URL
            export COPILOT_MODEL="claude-sonnet-4.5"
            ;;
        openclaw)
            export AGENT_BACKEND=openclaw
            ;;
        copilot)
            export AGENT_BACKEND=copilot
            ;;
        *)
            export AGENT_BACKEND=copilot
            export COPILOT_MODEL="${name}"
            ;;
    esac
    log_info "Agent tier ${idx} activated: '${name}' (backend=${AGENT_BACKEND}, model=${COPILOT_MODEL:-<default>})"
}


# ═══════════════════════════════════════════════════════════════════
# _agent_progress_reporter — background pinger; prints elapsed + log size
#   Usage: _agent_progress_reporter <watch_file>  → echoes the reporter PID
# ═══════════════════════════════════════════════════════════════════
_agent_progress_reporter() {
    local watch_file="$1"
    (
        local _start=$SECONDS
        while true; do
            sleep 30
            local elapsed=$(( SECONDS - _start ))
            local lines=0
            [[ -f "${watch_file}" ]] && lines=$(wc -l < "${watch_file}" 2>/dev/null || echo 0)
            log_info "  [agent running ${elapsed}s] output: ${lines} lines"
        done
    ) &
    echo $!
}


# ═══════════════════════════════════════════════════════════════════
# run_openclaw_fix — OpenClaw backend
# ═══════════════════════════════════════════════════════════════════
run_openclaw_fix() {
    local prompt="$1"
    local log_file="$2"
    local session_id_arg="${3:-}"

    if ! command -v openclaw >/dev/null 2>&1; then
        log_warn "openclaw not found, skipping agent fix"
        echo "openclaw not available" > "${log_file}"
        return 1
    fi

    local timeout="${AGENT_TIMEOUT:-600}"
    local session_id="${session_id_arg:-fix_${phase_name:-unknown}_$$_$(date +%s)}"
    local sessions_dir="${OPENCLAW_SESSIONS_DIR:-/root/.openclaw/agents/main/sessions}"
    local session_file="${sessions_dir}/${session_id}.jsonl"

    log_info "Calling openclaw agent (session=${session_id}, timeout=${timeout}s)..."
    log_info "  Session file: ${session_file}"

    local _progress_pid
    _progress_pid=$(_agent_progress_reporter "${session_file}")

    timeout "${timeout}" openclaw agent --local \
        --session-id "${session_id}" \
        --message "${prompt}" \
        --timeout "${timeout}" \
        2>&1 | tee "${log_file}" || {
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo "[TIMEOUT] Agent exceeded ${timeout}s" >> "${log_file}"
            log_warn "Agent timed out after ${timeout}s"
        fi
    }

    if [[ -n "${_progress_pid}" ]]; then
        kill "${_progress_pid}" 2>/dev/null || true
        wait "${_progress_pid}" 2>/dev/null || true
    fi

    if [[ -f "${session_file}" ]]; then
        local msg_count tool_count
        msg_count=$(grep -c '"type":"message"\|"type": "message"' "${session_file}" 2>/dev/null || echo 0)
        tool_count=$(grep -c '"tool_use"\|"tool_call"' "${session_file}" 2>/dev/null || echo 0)
        log_info "  Agent session complete: ${msg_count} messages, ${tool_count} tool calls"
    fi

    return 0
}


# _copilot_gen_uuid — a v4-ish UUID for a deterministic copilot session id.
_copilot_gen_uuid() {
    if [[ -r /proc/sys/kernel/random/uuid ]]; then
        cat /proc/sys/kernel/random/uuid
    elif command -v uuidgen >/dev/null 2>&1; then
        uuidgen
    else
        python3 -c "import uuid; print(uuid.uuid4())"
    fi
}


# ═══════════════════════════════════════════════════════════════════
# _copilot_setup_byok — activate Copilot BYOK (custom model provider).
#
# Lets Copilot CLI drive a NON-GitHub model (e.g. MiniMax-M3) via its own API.
# BYOK bypasses GitHub model routing entirely: no GitHub token, no api.github.com.
# Reuses MINIMAX_API_KEY (the same key OpenClaw uses) as the provider key.
#
# Config (config.env / env):
#   COPILOT_PROVIDER_BASE_URL   provider endpoint (activates BYOK).
#                               Default when COPILOT_MINIMAX=1: MiniMax anthropic URL.
#   COPILOT_PROVIDER_TYPE       openai | azure | anthropic (default anthropic here)
#   COPILOT_PROVIDER_API_KEY    provider key (defaults to MINIMAX_API_KEY)
#   COPILOT_MODEL               wire model name (default MiniMax-M3)
#   COPILOT_PROVIDER_MODEL_ID   well-known base model for agent profile (default claude-sonnet-4)
# ═══════════════════════════════════════════════════════════════════
_copilot_setup_byok() {
    # Convenience: COPILOT_MINIMAX=1 fills MiniMax defaults without extra config.
    if [[ "${COPILOT_MINIMAX:-}" == "1" && -z "${COPILOT_PROVIDER_BASE_URL:-}" ]]; then
        export COPILOT_PROVIDER_BASE_URL="https://api.minimaxi.com/anthropic"
    fi
    export COPILOT_PROVIDER_TYPE="${COPILOT_PROVIDER_TYPE:-anthropic}"
    export COPILOT_PROVIDER_API_KEY="${COPILOT_PROVIDER_API_KEY:-${MINIMAX_API_KEY:-}}"
    export COPILOT_MODEL="${COPILOT_MODEL:-MiniMax-M3}"
    export COPILOT_PROVIDER_MODEL_ID="${COPILOT_PROVIDER_MODEL_ID:-claude-sonnet-4}"
    export COPILOT_PROVIDER_WIRE_MODEL="${COPILOT_PROVIDER_WIRE_MODEL:-${COPILOT_MODEL}}"
    export COPILOT_PROVIDER_MAX_PROMPT_TOKENS="${COPILOT_PROVIDER_MAX_PROMPT_TOKENS:-200000}"
    export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS="${COPILOT_PROVIDER_MAX_OUTPUT_TOKENS:-32768}"

    if [[ -z "${COPILOT_PROVIDER_API_KEY}" ]]; then
        log_warn "Copilot BYOK: no provider key (set COPILOT_PROVIDER_API_KEY or MINIMAX_API_KEY)"
        return 1
    fi
    log_info "Copilot BYOK: ${COPILOT_PROVIDER_TYPE} @ ${COPILOT_PROVIDER_BASE_URL} model=${COPILOT_PROVIDER_WIRE_MODEL} (no GitHub auth)"
    return 0
}


# ═══════════════════════════════════════════════════════════════════
# run_copilot_fix — GitHub Copilot CLI backend (headless)
#
# Two auth modes, auto-detected:
#   BYOK   — COPILOT_PROVIDER_BASE_URL (or COPILOT_MINIMAX=1) set → custom provider
#            (e.g. MiniMax-M3). No GitHub token needed; reuses MINIMAX_API_KEY.
#   GitHub — otherwise, env-var COPILOT_GITHUB_TOKEN (like OpenClaw's MINIMAX key).
# Model/effort default to ~/.copilot/settings.json; overridden by COPILOT_MODEL/EFFORT.
# ═══════════════════════════════════════════════════════════════════
run_copilot_fix() {
    local prompt="$1"
    local log_file="$2"
    local session_id_arg="${3:-}"   # accepted for interface parity (unused: prompt carries prior context)

    if ! command -v copilot >/dev/null 2>&1; then
        log_warn "copilot not found, skipping agent fix"
        echo "copilot not available" > "${log_file}"
        return 1
    fi

    # Auth mode auto-detect:
    #   BYOK  — COPILOT_PROVIDER_BASE_URL set (custom model provider, e.g. MiniMax).
    #           No GitHub auth required; reuses MINIMAX_API_KEY as the provider key.
    #   GitHub — otherwise, requires COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN.
    local auth_mode="github"
    if [[ -n "${COPILOT_PROVIDER_BASE_URL:-}" || -n "${COPILOT_MINIMAX:-}" ]]; then
        auth_mode="byok"
        _copilot_setup_byok || { echo "copilot BYOK setup failed" > "${log_file}"; return 1; }
    elif [[ -z "${COPILOT_GITHUB_TOKEN:-}${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]]; then
        log_warn "No Copilot auth (set COPILOT_PROVIDER_BASE_URL for BYOK, or COPILOT_GITHUB_TOKEN); skipping agent fix"
        echo "copilot auth not set" > "${log_file}"
        return 1
    fi

    local timeout="${AGENT_TIMEOUT:-600}"
    local copilot_log_dir="${RUN_OUTPUT_DIR:-/tmp}/copilot_logs"
    mkdir -p "${copilot_log_dir}"

    # Deterministic session UUID so we can locate this run's rich session transcript
    # (~/.copilot/session-state/<id>/events.jsonl) afterwards for archival/upload.
    local cp_session_id="${session_id_arg:-}"
    if [[ ! "${cp_session_id}" =~ ^[0-9a-fA-F-]{36}$ ]]; then
        cp_session_id="$(_copilot_gen_uuid)"
    fi

    # Assemble optional flags (settings.json is the baseline; these override).
    local -a extra=()
    # In BYOK mode the model comes from COPILOT_MODEL env (set by _copilot_setup_byok);
    # only add --model when explicitly overriding in GitHub mode.
    if [[ "${auth_mode}" == "github" && -n "${COPILOT_MODEL:-}" ]]; then
        extra+=(--model "${COPILOT_MODEL}")
    fi
    [[ -n "${COPILOT_EFFORT:-}" ]] && extra+=(--effort "${COPILOT_EFFORT}")
    # Grant filesystem access to the pipeline dir(s) so the agent can edit code.
    local add_dirs="${COPILOT_ADD_DIRS:-${SCRIPT_DIR:-$PWD}}"
    local d
    for d in ${add_dirs}; do
        [[ -n "${d}" ]] && extra+=(--add-dir "${d}")
    done

    log_info "Calling copilot agent (model=${COPILOT_MODEL:-<settings.json>}, session=${cp_session_id}, timeout=${timeout}s)..."
    log_info "  Copilot logs: ${copilot_log_dir}"

    # Native readable transcript export (session_* so upload picks it up).
    local cp_share_md="${RUN_OUTPUT_DIR}/session_copilot_${phase_name:-fix}_${cp_session_id}.md"

    local _progress_pid
    _progress_pid=$(_agent_progress_reporter "${log_file}")

    timeout "${timeout}" copilot -p "${prompt}" \
        --allow-all-tools \
        --no-color \
        --session-id "${cp_session_id}" \
        --share "${cp_share_md}" \
        --log-dir "${copilot_log_dir}" \
        --log-level error \
        "${extra[@]}" \
        2>&1 | tee "${log_file}" || {
        local rc=$?
        if [ $rc -eq 124 ]; then
            echo "[TIMEOUT] Agent exceeded ${timeout}s" >> "${log_file}"
            log_warn "Agent timed out after ${timeout}s"
        fi
    }

    if [[ -n "${_progress_pid}" ]]; then
        kill "${_progress_pid}" 2>/dev/null || true
        wait "${_progress_pid}" 2>/dev/null || true
    fi

    # Archive the RICH session transcript (events.jsonl, like OpenClaw's jsonl) so it
    # is uploaded with the run. Named session_* to match the upload glob.
    local cp_state_dir="${COPILOT_SESSION_STATE_DIR:-${HOME}/.copilot/session-state}/${cp_session_id}"
    if [[ -f "${cp_state_dir}/events.jsonl" ]]; then
        cp "${cp_state_dir}/events.jsonl" \
           "${RUN_OUTPUT_DIR}/session_copilot_${phase_name:-fix}_${cp_session_id}.jsonl" 2>/dev/null || true
        log_info "  Archived copilot session events.jsonl (${cp_session_id})"
    fi

    if [[ -f "${log_file}" ]]; then
        local lines
        lines=$(wc -l < "${log_file}" 2>/dev/null || echo 0)
        log_info "  Copilot session complete: ${lines} output lines"
    fi

    return 0
}


# ═══════════════════════════════════════════════════════════════════
# run_agent_fix — backend dispatcher (the fix loop calls THIS, not a backend)
# ═══════════════════════════════════════════════════════════════════
run_agent_fix() {
    local backend="${AGENT_BACKEND:-openclaw}"
    case "${backend,,}" in
        copilot)
            run_copilot_fix "$@"
            ;;
        openclaw|"")
            run_openclaw_fix "$@"
            ;;
        *)
            log_warn "Unknown AGENT_BACKEND='${backend}', falling back to openclaw"
            run_openclaw_fix "$@"
            ;;
    esac
}


# ═══════════════════════════════════════════════════════════════════
# agent_backend_setup — one-time per-run setup for the selected backend.
#   Called by auto.sh before the pipeline runs. Keeps backend-specific
#   provisioning out of the orchestrator.
# ═══════════════════════════════════════════════════════════════════
agent_backend_setup() {
    local backend="${AGENT_BACKEND:-openclaw}"
    case "${backend,,}" in
        copilot)
            # Install the baked-in settings.json baseline (model/effort/contextTier/
            # allowedUrls) unless one already exists. CLI flags (--model/--effort in
            # run_copilot_fix) override these at call time. Token is env-var based.
            mkdir -p /root/.copilot
            local tmpl="${SCRIPT_DIR:-$(dirname "${BASH_SOURCE[0]}")/..}/copilot_config/settings.json"
            if [[ ! -f /root/.copilot/settings.json && -f "${tmpl}" ]]; then
                cp "${tmpl}" /root/.copilot/settings.json
                log_info "Copilot backend: installed settings.json baseline from ${tmpl}"
            fi
            # Auth check: BYOK (provider URL / MiniMax) OR a GitHub token.
            if [[ -n "${COPILOT_PROVIDER_BASE_URL:-}" || "${COPILOT_MINIMAX:-}" == "1" ]]; then
                if [[ -z "${COPILOT_PROVIDER_API_KEY:-}${MINIMAX_API_KEY:-}" ]]; then
                    log_warn "Copilot BYOK selected but no provider key (COPILOT_PROVIDER_API_KEY/MINIMAX_API_KEY) — fixes will be skipped."
                else
                    log_info "Copilot backend: BYOK mode (custom provider, no GitHub auth)."
                fi
            elif [[ -z "${COPILOT_GITHUB_TOKEN:-}${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]]; then
                log_warn "Copilot backend selected but no auth (COPILOT_PROVIDER_BASE_URL for BYOK, or COPILOT_GITHUB_TOKEN) — fixes will be skipped."
            fi
            ;;
        *)
            : # openclaw provisioning is handled by the pipeline template (cp openclaw_config)
            ;;
    esac
}
