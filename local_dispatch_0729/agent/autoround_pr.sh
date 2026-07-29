#!/bin/bash
# autoround_pr.sh — optional: capture agent fixes that landed in the auto_round
# package as a standalone patch, and (when a push token is configured) open a PR.
#
# This is OFF by default and is fully self-contained to local_dispatch — it never
# affects the normal non-local-dispatch flow. Enable per run with:
#   AUTOROUND_PR_ENABLED=1
#
# Config (env):
#   AUTOROUND_PR_ENABLED   1 to enable snapshot+patch capture (default: 0/off)
#   AUTOROUND_REPO         target git repo for the PR (e.g. intel/auto-round or a fork)
#   AUTOROUND_PR_TOKEN     GitHub token with push access (fill in later)
#   AUTOROUND_PR_BASE      base branch for the PR (default: main)
#
# Flow (driven by the pipeline):
#   autoround_snapshot         # BEFORE the fix loop runs (records a pristine copy)
#   autoround_capture_patch    # AFTER the fix loop (diffs → <run_dir>/autoround_fix.patch)
#   autoround_maybe_pr         # opens a PR if token+repo set, else just keeps the patch

[[ -n "${_AUTOROUND_PR_SOURCED:-}" ]] && return 0
_AUTOROUND_PR_SOURCED=1

# Minimal log shims if the caller hasn't defined them.
command -v log_info >/dev/null 2>&1 || log_info() { echo "[INFO] $*"; }
command -v log_warn >/dev/null 2>&1 || log_warn() { echo "[WARN] $*"; }

autoround_enabled() { [[ "${AUTOROUND_PR_ENABLED:-0}" == "1" ]]; }

# Locate the installed auto_round package directory (empty if not importable).
autoround_pkg_dir() {
    python3 - <<'PY' 2>/dev/null
import importlib.util, os
spec = importlib.util.find_spec("auto_round")
if spec and spec.submodule_search_locations:
    print(os.path.realpath(list(spec.submodule_search_locations)[0]))
PY
}

# autoround_snapshot — record a pristine copy of auto_round before any fix touches it.
# Stores the path in $AUTOROUND_SNAPSHOT_DIR for autoround_capture_patch to diff against.
autoround_snapshot() {
    autoround_enabled || return 0
    local pkg; pkg="$(autoround_pkg_dir)"
    if [[ -z "${pkg}" || ! -d "${pkg}" ]]; then
        log_warn "autoround_pr: auto_round package not found — patch capture disabled for this run"
        export AUTOROUND_SNAPSHOT_DIR=""
        return 0
    fi
    local snap="${AUTOROUND_SNAPSHOT_ROOT:-/tmp}/autoround_pristine_$$"
    rm -rf "${snap}"
    if cp -a "${pkg}" "${snap}" 2>/dev/null; then
        export AUTOROUND_SNAPSHOT_DIR="${snap}"
        export AUTOROUND_PKG_DIR="${pkg}"
        log_info "autoround_pr: snapshotted pristine auto_round (${pkg})"
    else
        export AUTOROUND_SNAPSHOT_DIR=""
        log_warn "autoround_pr: failed to snapshot auto_round — patch capture disabled"
    fi
}

# autoround_capture_patch <run_dir> — diff the (possibly patched) package against the
# pristine snapshot into <run_dir>/autoround_fix.patch. Returns 0 only if a non-empty
# patch was produced (i.e. the agent actually modified auto_round).
autoround_capture_patch() {
    autoround_enabled || return 1
    local run_dir="$1"
    local snap="${AUTOROUND_SNAPSHOT_DIR:-}"
    local pkg="${AUTOROUND_PKG_DIR:-$(autoround_pkg_dir)}"
    [[ -z "${snap}" || ! -d "${snap}" || -z "${pkg}" || ! -d "${pkg}" ]] && return 1

    mkdir -p "${run_dir}"
    local patch="${run_dir}/autoround_fix.patch"
    # diff -ruN yields a portable unified patch even though site-packages isn't a git repo.
    # Prefix paths as a/auto_round and b/auto_round so `git apply -p1` works downstream.
    ( cd "$(dirname "${snap}")" && \
      diff -ruN "$(basename "${snap}")" "${pkg}" 2>/dev/null ) \
        | sed -e "s#^--- $(basename "${snap}")#--- a/auto_round#" \
              -e "s#^+++ ${pkg}#+++ b/auto_round#" \
        > "${patch}" || true

    if [[ -s "${patch}" ]]; then
        export AUTOROUND_PATCH_FILE="${patch}"
        log_info "autoround_pr: captured auto_round changes → ${patch} ($(wc -l < "${patch}") lines)"
        return 0
    fi
    rm -f "${patch}"
    return 1
}

# autoround_maybe_pr <run_dir> — if a patch was captured and a push token+repo are
# configured, open a PR. Otherwise keep the patch in the run dir (uploaded with the run)
# and log that PR submission was skipped. PR push wiring is intentionally token-gated.
autoround_maybe_pr() {
    autoround_enabled || return 0
    local run_dir="$1"
    local patch="${AUTOROUND_PATCH_FILE:-${run_dir}/autoround_fix.patch}"
    [[ -s "${patch}" ]] || return 0

    if [[ -z "${AUTOROUND_PR_TOKEN:-}" || -z "${AUTOROUND_REPO:-}" ]]; then
        log_warn "autoround_pr: patch saved to ${patch}, but PR submission is skipped (set AUTOROUND_REPO + AUTOROUND_PR_TOKEN to enable)."
        return 0
    fi

    # --- Token-gated PR submission (opt-in). Kept minimal until a real token/repo is provided. ---
    local base="${AUTOROUND_PR_BASE:-main}"
    local branch="lb-autoround-fix-$(date +%Y%m%d-%H%M%S)"
    local work="/tmp/autoround_pr_$$"
    rm -rf "${work}"
    log_info "autoround_pr: opening PR against ${AUTOROUND_REPO} (base=${base}, branch=${branch})"
    if git clone --depth 1 -b "${base}" \
        "https://x-access-token:${AUTOROUND_PR_TOKEN}@github.com/${AUTOROUND_REPO}.git" \
        "${work}" >/dev/null 2>&1; then
        ( cd "${work}" \
          && git checkout -b "${branch}" \
          && git apply -p1 "${patch}" \
          && git add -A \
          && git -c user.email="lb-agent@local" -c user.name="lb-agent" \
                 commit -m "auto_round fix from local_dispatch agent (opus tier)" \
          && git push origin "${branch}" ) >/dev/null 2>&1 \
        && log_info "autoround_pr: pushed branch ${branch} — open the PR at https://github.com/${AUTOROUND_REPO}/pull/new/${branch}" \
        || log_warn "autoround_pr: PR push failed (patch retained at ${patch})"
    else
        log_warn "autoround_pr: clone of ${AUTOROUND_REPO} failed (patch retained at ${patch})"
    fi
    rm -rf "${work}"
}
