# Pluggable Agent Fix-Loop Backends (OpenClaw / Copilot)

**Status:** Implemented.
**Scope:** `lb_eval/auto_quant` — the agent-assisted fix loop that repairs failed pipeline phases.

The fix loop can now run its repair agent via **either OpenClaw or the GitHub Copilot CLI**,
selected at runtime by `AGENT_BACKEND`. The two backends are **decoupled** behind a single
`run_agent_fix` abstraction so they can be optimized independently, and all downstream diagnosis
parsing (VERDICT / ROOT_CAUSE / lessons) is unchanged.

---

## 1. Why

The fix-loop previously hard-coded OpenClaw (`run_openclaw_fix`) at the call site. To evaluate /
use Copilot as the repair agent — and to keep both improvable in isolation — the "which agent
runs" decision was extracted into a pluggable backend layer.

Design goals:
- **Decoupled:** the orchestrator (`agent_fix_loop`) depends only on `run_agent_fix`.
- **Drop-in:** both backends tee the agent's textual output (which contains the labeled fields the
  prompt asks for) into the same `attempt_N.log`, so `extract_agent_analysis`, the `VERDICT:` grep,
  and `extract_agent_field` work identically for either backend.
- **Auth parity with OpenClaw:** OpenClaw uses `MINIMAX_API_KEY`; Copilot uses an env-var token
  (`COPILOT_GITHUB_TOKEN`) injected the same way.
- **settings.json baseline + CLI override:** Copilot model/effort default to a baked-in
  `~/.copilot/settings.json`; `COPILOT_MODEL` / `COPILOT_EFFORT` override per run.

---

## 2. Architecture

```
agent_fix_loop.sh  ── calls ──►  run_agent_fix(prompt, log, session_id)   [abstraction]
                                       │  reads ${AGENT_BACKEND:-openclaw}
                        ┌──────────────┴──────────────┐
                run_openclaw_fix                 run_copilot_fix
                (openclaw agent --local)         (copilot -p --allow-all-tools)
                auth: MINIMAX_API_KEY            auth: COPILOT_GITHUB_TOKEN
                        └──────────────┬──────────────┘
                          both tee agent text → attempt_N.log
                                       │
              downstream (unchanged): extract_agent_analysis / VERDICT grep /
                                      extract_agent_field / save_lesson
```

All of this lives in **`phases/agent_backends.sh`** (new), sourced by `agent_fix_loop.sh`.

---

## 3. Files changed

| File | Change |
|---|---|
| `phases/agent_backends.sh` | **NEW.** `run_agent_fix` (dispatcher), `run_openclaw_fix` (moved here), `run_copilot_fix` (new), `agent_backend_setup` (per-run provisioning), `_agent_progress_reporter` (shared). |
| `phases/agent_fix_loop.sh` | Removed inline `run_openclaw_fix`; `source agent_backends.sh`; call site now `run_agent_fix`. |
| `config.env` | Added `AGENT_BACKEND` + `COPILOT_GITHUB_TOKEN` / `COPILOT_MODEL` / `COPILOT_EFFORT` / `COPILOT_ADD_DIRS`. |
| `copilot_config/settings.json` | **NEW.** Clean baseline (model/effort/contextTier + container-relevant allowedUrls). |
| `.azure-pipelines/docker/agent.dockerfile` | Added `npm install -g @github/copilot`. |
| `auto.sh` | Calls `agent_backend_setup` and logs the active backend before the pipeline runs. |

---

## 4. Configuration (`config.env`)

```sh
# Which agent repairs failed phases:
#   openclaw (default) | copilot
AGENT_BACKEND=openclaw

# ── Copilot backend (only when AGENT_BACKEND=copilot) ──
# Headless token (fine-grained PAT v2 with "Copilot Requests", or Copilot/gh OAuth token).
COPILOT_GITHUB_TOKEN=
# Optional overrides of ~/.copilot/settings.json (leave empty to use the baseline):
COPILOT_MODEL=
COPILOT_EFFORT=
# Extra dirs the agent may read/write (default: the scripts dir):
COPILOT_ADD_DIRS=
```

In the Azure pipeline these are injected via `update_config_env.py --set` exactly like
`MINIMAX_API_KEY` / `GIT_TOKEN` today.

---

## 5. Copilot backend details

### 5.1 Auth — two modes, auto-detected

**Mode A — BYOK (custom model provider, e.g. MiniMax-M3) — recommended.**
Copilot CLI can drive a non-GitHub model via its own provider ("Bring Your Own Key"). This
**bypasses GitHub model routing entirely: no GitHub token, no `api.github.com` access** — and reuses
the same MiniMax key/endpoint OpenClaw already uses, so both backends can run the *same* model.

Activate by setting `COPILOT_PROVIDER_BASE_URL` (or the shortcut `COPILOT_MINIMAX=1`).
`_copilot_setup_byok` then exports the provider env:
```
COPILOT_PROVIDER_BASE_URL=https://api.minimaxi.com/anthropic
COPILOT_PROVIDER_TYPE=anthropic
COPILOT_PROVIDER_API_KEY=$MINIMAX_API_KEY        # reused
COPILOT_MODEL=MiniMax-M3
COPILOT_PROVIDER_MODEL_ID=claude-sonnet-4        # well-known profile → good tool/prompt/token defaults
COPILOT_PROVIDER_WIRE_MODEL=MiniMax-M3           # name sent to the provider
COPILOT_PROVIDER_MAX_PROMPT_TOKENS=200000
COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=32768
```
Verified working headless (connectivity + shell tools + file edit) via `test_copilot_minimax.sh`.

**Mode B — GitHub token.** Used only when no BYOK provider is configured. Copilot CLI checks
`COPILOT_GITHUB_TOKEN` → `GH_TOKEN` → `GITHUB_TOKEN`. Supported: **fine-grained PAT (v2) with
"Copilot Requests"**, or Copilot/gh OAuth tokens. **Classic `ghp_` PATs are not supported.**

`run_copilot_fix` picks the mode automatically: BYOK if `COPILOT_PROVIDER_BASE_URL`/`COPILOT_MINIMAX`
is set, else GitHub token (and skips with a warning if neither is present).

### 5.2 settings.json baseline + CLI override
- **Baseline:** `copilot_config/settings.json` is installed to `~/.copilot/settings.json` by
  `agent_backend_setup` (only if not already present — never clobbers an existing file). It carries
  `model`, `effortLevel`, `contextTier`, and a **cleaned** `allowedUrls` (personal/internal URLs
  removed; only container-relevant hosts kept).
- **Override:** in GitHub mode `--model`/`--effort` are passed only when `COPILOT_MODEL`/
  `COPILOT_EFFORT` are set. In BYOK mode the model comes from `COPILOT_MODEL` (provider env), so
  `--model` is not re-passed.

### 5.3 The invocation
```sh
timeout "${AGENT_TIMEOUT:-600}" copilot -p "${prompt}" \
    --allow-all-tools \                # required for non-interactive (no permission prompts)
    --no-color \
    --log-dir "${RUN_OUTPUT_DIR}/copilot_logs" \
    --log-level error \
    [--model "$COPILOT_MODEL"] [--effort "$COPILOT_EFFORT"] \
    [--add-dir <dir> ...] \
    2>&1 | tee "${attempt_log}"
```
- `--allow-all-tools` is mandatory for headless (otherwise it blocks on confirmation).
- No built-in `--timeout`; wrapped with `timeout` (same as OpenClaw). Exit 124 = timed out.
- Output is teed to `attempt_N.log` so the labeled diagnosis fields are parsed downstream as usual.
- The reusable `error_analysis` methodology can be provided to Copilot via `AGENTS.md` /
  `.github/copilot-instructions.md` (auto-loaded), same content as the OpenClaw skill.

---

## 6. Usage

**Keep OpenClaw (default):** nothing to do.

**Switch to Copilot + MiniMax (BYOK — recommended, no GitHub auth):**
```sh
# config.env
AGENT_BACKEND=copilot
COPILOT_MINIMAX=1                    # MiniMax-M3 via the MiniMax anthropic endpoint
MINIMAX_API_KEY=<your MiniMax key>   # already present for OpenClaw
# optional: COPILOT_MODEL=MiniMax-M2.7, COPILOT_PROVIDER_MODEL_ID=claude-opus-4
```

**Switch to Copilot + GitHub-hosted models:**
```sh
AGENT_BACKEND=copilot
COPILOT_GITHUB_TOKEN=<fine-grained-PAT-or-OAuth-token>
COPILOT_MODEL=claude-opus-4.8
COPILOT_EFFORT=high
```
Then run `auto.sh` as usual. On start it logs the active backend and (for copilot) the auth mode,
and installs the settings.json baseline.

---

## 7. Extending with another backend

1. Add `run_<name>_fix(prompt, log_file, session_id)` in `phases/agent_backends.sh` — it must tee
   the agent's text (with the prompt's labeled fields) into `log_file`.
2. Add a `case` arm in `run_agent_fix` (and any provisioning in `agent_backend_setup`).
3. Add its config keys to `config.env`.
No change to `agent_fix_loop.sh` or the downstream parsers is required.

---

## 8. Validation performed

- `bash -n` on `agent_backends.sh`, `agent_fix_loop.sh`, `auto.sh`; `settings.json` is valid JSON.
- Dispatcher routing unit-tested: unset→openclaw, `copilot`, `COPILOT` (case-insensitive),
  bogus→fallback-to-openclaw.
- `agent_backend_setup` warns when `AGENT_BACKEND=copilot` but no token is set, and installs the
  settings.json baseline without clobbering an existing file.

---

## 9. Notes & follow-ups

- **Verify the npm package name** for the Copilot CLI in your build environment
  (`@github/copilot` used here) and pin a version like the openclaw line.
- End-to-end Copilot runs require network access to `api.github.com` (the dev sandbox blocks it,
  which surfaces as `network fetch failed`, not a token error).
- Copilot cross-attempt memory (`--resume`) is not wired; each attempt is stateless, but the fix
  loop already seeds prior-attempt context into the prompt, so this is not required. Could be added
  later for parity with OpenClaw's `--session-id` continuity.
- See `../../COPILOT_HEADLESS_VERIFY.md` (repo root) for the standalone auth-verification commands.
