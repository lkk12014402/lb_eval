#!/bin/bash
# patch_capture.sh — capture agent fixes as categorized patches + surface prior
# changes to the strongest (opus) tier so it refines rather than restarts.
#
# The agents edit files IN PLACE (auto_round source, HF model custom code, etc.). This
# module snapshots those editable areas before the fix loop, lets us (a) show the opus
# tier exactly what earlier tiers already changed, and (b) emit categorized .patch files
# into the run dir so they travel to the HF dataset. auto_round is the primary target
# but every touched area is captured and classified.
#
# Public API:
#   patch_snapshot                 # BEFORE the fix loop — snapshot editable areas
#   patch_prior_diff               # combined diff of current vs pristine (for the prompt)
#   patch_capture_all <run_dir>    # AFTER the loop — write <run_dir>/patches/*.patch + manifest
#   patch_maybe_pr <run_dir>       # token-gated PR for the auto_round patch (opt-in)
#
# Config:
#   PATCH_CAPTURE_ENABLED   1 (default) to snapshot + capture; 0 to disable
#   AUTOROUND_REPO / AUTOROUND_PR_TOKEN / AUTOROUND_PR_BASE   token-gated PR (opt-in)
#   HF_HOME                 HF cache root (default ~/.cache/huggingface)
#   PATCH_SNAPSHOT_ROOT     where pristine copies live (default /tmp)

[[ -n "${_PATCH_CAPTURE_SOURCED:-}" ]] && return 0
_PATCH_CAPTURE_SOURCED=1

command -v log_info >/dev/null 2>&1 || log_info() { echo "[patch] $*"; }
command -v log_warn >/dev/null 2>&1 || log_warn() { echo "[patch] $*"; }

patch_capture_enabled() { [[ "${PATCH_CAPTURE_ENABLED:-1}" == "1" ]]; }

# Locate the installed auto_round package dir (empty if not importable).
_patch_autoround_dir() {
    python3 - <<'PY' 2>/dev/null
import importlib.util, os
spec = importlib.util.find_spec("auto_round")
if spec and spec.submodule_search_locations:
    print(os.path.realpath(list(spec.submodule_search_locations)[0]))
PY
}

# The HF model custom-code cache (transformers_modules/<Org>/<Model>/<hash>/*.py).
_patch_hfmodules_dir() {
    local hf="${HF_HOME:-${HOME:-/root}/.cache/huggingface}/modules/transformers_modules"
    [[ -d "${hf}" ]] && echo "${hf}"
}

# ── The editable AREAS we track: "label|category|path" ───────────────────────
# category feeds the upload manifest (auto_round is primary).
_patch_areas() {
    local ar hf
    ar="$(_patch_autoround_dir)"
    [[ -n "${ar}" && -d "${ar}" ]] && echo "auto_round|auto_round|${ar}"
    hf="$(_patch_hfmodules_dir)"
    [[ -n "${hf}" ]] && echo "model_code|model_code|${hf}"
}

# patch_snapshot — pristine copy of every editable area (before any fix runs).
patch_snapshot() {
    patch_capture_enabled || return 0
    local root="${PATCH_SNAPSHOT_ROOT:-/tmp}/patch_pristine_$$"
    rm -rf "${root}"; mkdir -p "${root}"
    export PATCH_SNAPSHOT_DIR="${root}"
    local line label cat path
    : > "${root}/.areas"
    while IFS= read -r line; do
        [[ -z "${line}" ]] && continue
        label="${line%%|*}"; path="${line##*|}"; cat="$(echo "${line}" | cut -d'|' -f2)"
        if cp -a "${path}" "${root}/${label}" 2>/dev/null; then
            echo "${label}|${cat}|${path}" >> "${root}/.areas"
        fi
    done < <(_patch_areas)
    if [[ -s "${root}/.areas" ]]; then
        log_info "patch: snapshotted $(wc -l < "${root}/.areas") editable area(s) for diff/capture"
    else
        log_info "patch: no editable areas found to snapshot (auto_round/model-code not present yet)"
    fi
}

# _patch_area_diff <label> <category> <path> — unified diff (a/<label> b/<label>) or "".
_patch_area_diff() {
    local label="$1" cat="$2" path="$3"
    local snap="${PATCH_SNAPSHOT_DIR:-}/${label}"
    [[ -z "${PATCH_SNAPSHOT_DIR:-}" || ! -d "${snap}" || ! -d "${path}" ]] && return 0
    ( cd "$(dirname "${snap}")" && diff -ruN "${label}" "${path}" 2>/dev/null ) \
        | sed -e "s#^--- ${label}#--- a/${label}#" \
              -e "s#^+++ ${path}#+++ b/${label}#"
}

# patch_prior_diff — combined diff of ALL areas (current vs pristine). Used to show the
# opus tier what earlier tiers already changed so it refines instead of restarting.
# Bounded so it can't blow up the prompt.
patch_prior_diff() {
    patch_capture_enabled || return 0
    [[ -z "${PATCH_SNAPSHOT_DIR:-}" || ! -f "${PATCH_SNAPSHOT_DIR}/.areas" ]] && return 0
    local line label cat path d out=""
    while IFS='|' read -r label cat path; do
        [[ -z "${label}" ]] && continue
        d="$(_patch_area_diff "${label}" "${cat}" "${path}")"
        [[ -n "${d}" ]] && out+="### ${label} (${cat})
${d}
"
    done < "${PATCH_SNAPSHOT_DIR}/.areas"
    # Cap at ~500 lines to protect the prompt.
    printf '%s' "${out}" | head -500
}

# patch_capture_all <run_dir> — write categorized patches into <run_dir>/patches/ and a
# manifest.json. Returns 0 if at least one non-empty patch was produced.
patch_capture_all() {
    patch_capture_enabled || return 1
    local run_dir="$1"
    [[ -z "${PATCH_SNAPSHOT_DIR:-}" || ! -f "${PATCH_SNAPSHOT_DIR}/.areas" ]] && return 1
    local pdir="${run_dir}/patches"; mkdir -p "${pdir}"
    local line label cat path d wrote=0
    local manifest="[]"
    while IFS='|' read -r label cat path; do
        [[ -z "${label}" ]] && continue
        d="$(_patch_area_diff "${label}" "${cat}" "${path}")"
        if [[ -n "${d}" ]]; then
            printf '%s\n' "${d}" > "${pdir}/${label}.patch"
            local lines; lines=$(wc -l < "${pdir}/${label}.patch")
            log_info "patch: captured ${label} (${cat}) → patches/${label}.patch (${lines} lines)"
            manifest=$(CAT="${cat}" LABEL="${label}" PFILE="${label}.patch" LINES="${lines}" \
                       SRC="${path}" M="${manifest}" python3 -c '
import json, os
m = json.loads(os.environ["M"])
m.append({"label": os.environ["LABEL"], "category": os.environ["CAT"],
          "patch_file": os.environ["PFILE"], "lines": int(os.environ["LINES"]),
          "source_path": os.environ["SRC"]})
print(json.dumps(m))')
            wrote=1
        fi
    done < "${PATCH_SNAPSHOT_DIR}/.areas"

    if [[ "${wrote}" == "1" ]]; then
        # Attach the opus root-cause analysis (if the fix loop recorded one) so the
        # dataset shows WHY each patch exists + the component classification. Also
        # record whether the patch actually RESOLVED the run (did the phase pass after
        # the fix), so downstream can tell a verified fix from an unverified attempt.
        local analysis="${OPUS_ANALYSIS_JSON:-}"
        local run_status="${PATCH_RUN_STATUS:-unknown}"       # Finished | Failed | unknown
        local failed_step="${PATCH_FAILED_STEP:-}"
        local resolved="false"
        [[ "${run_status}" == "Finished" ]] && resolved="true"
        MANIFEST="${manifest}" ANALYSIS="${analysis}" OUT="${pdir}/manifest.json" \
        RESOLVED="${resolved}" RUN_STATUS="${run_status}" FAILED_STEP="${failed_step}" python3 -c '
import json, os
patches = json.loads(os.environ["MANIFEST"])
resolved = os.environ["RESOLVED"] == "true"
status = os.environ["RUN_STATUS"]
failed = os.environ.get("FAILED_STEP", "")
if resolved:
    outcome = "resolved — phase passed after the fix (patch verified)"
elif status == "Failed":
    outcome = f"unresolved — still failing at {failed} (patch did NOT fix the run)" if failed \
              else "unresolved — run still failing (patch did NOT fix it)"
else:
    outcome = "unknown — run outcome not reported"
# Stamp the verified status onto every patch too, so each .patch is self-describing.
for p in patches:
    p["resolved"] = resolved
doc = {"resolved": resolved, "run_status": status, "outcome": outcome,
       "patches": patches, "primary": "auto_round"}
if failed:
    doc["failed_step"] = failed
a = os.environ.get("ANALYSIS", "").strip()
if a:
    try: doc["opus_analysis"] = json.loads(a)
    except Exception: doc["opus_analysis_raw"] = a[:2000]
open(os.environ["OUT"], "w").write(json.dumps(doc, ensure_ascii=False, indent=2))'
        log_info "patch: wrote patches/manifest.json (resolved=${resolved}, status=${run_status})"
        return 0
    fi
    rmdir "${pdir}" 2>/dev/null || true
    return 1
}

# patch_maybe_pr <run_dir> — token-gated PR for the auto_round patch (opt-in). Keeps the
# patch in the run dir (uploaded with the run) when no token/repo is configured.
patch_maybe_pr() {
    patch_capture_enabled || return 0
    local run_dir="$1"
    local patch="${run_dir}/patches/auto_round.patch"
    [[ -s "${patch}" ]] || return 0

    # Only PR a VERIFIED auto_round patch (the run passed after the fix) — don't propose
    # an unverified/broken change upstream. Override with AUTOROUND_PR_ALLOW_UNVERIFIED=1.
    if [[ "${PATCH_RUN_STATUS:-}" != "Finished" && "${AUTOROUND_PR_ALLOW_UNVERIFIED:-0}" != "1" ]]; then
        log_warn "patch: auto_round patch is UNVERIFIED (run did not pass) — PR skipped. Set AUTOROUND_PR_ALLOW_UNVERIFIED=1 to PR anyway."
        return 0
    fi

    if [[ -z "${AUTOROUND_PR_TOKEN:-}" || -z "${AUTOROUND_REPO:-}" ]]; then
        log_warn "patch: auto_round patch saved to ${patch}; PR submission skipped (set AUTOROUND_REPO + AUTOROUND_PR_TOKEN to enable)."
        return 0
    fi
    local base="${AUTOROUND_PR_BASE:-main}"
    local branch="lb-autoround-fix-$(date +%Y%m%d-%H%M%S)"
    local work="/tmp/autoround_pr_$$"; rm -rf "${work}"
    log_info "patch: opening PR against ${AUTOROUND_REPO} (base=${base}, branch=${branch})"
    if git clone --depth 1 -b "${base}" \
        "https://x-access-token:${AUTOROUND_PR_TOKEN}@github.com/${AUTOROUND_REPO}.git" \
        "${work}" >/dev/null 2>&1; then
        ( cd "${work}" \
          && git checkout -b "${branch}" \
          && git apply -p1 "${patch}" \
          && git add -A \
          && git -c user.email="lb-agent@local" -c user.name="lb-agent" \
                 commit -m "auto_round fix from local_dispatch opus tier" \
          && git push origin "${branch}" ) >/dev/null 2>&1 \
        && log_info "patch: pushed ${branch} — open PR at https://github.com/${AUTOROUND_REPO}/pull/new/${branch}" \
        || log_warn "patch: PR push failed (patch retained at ${patch})"
    else
        log_warn "patch: clone of ${AUTOROUND_REPO} failed (patch retained at ${patch})"
    fi
    rm -rf "${work}"
}
