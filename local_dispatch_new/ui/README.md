# Local Dispatch UI

A simple Gradio app mirroring the leaderboard, for the **local GPU dispatch** path.
Displays results/queue/GPU data and lets you submit models to run via `local_dispatch`.

## Run

```bash
cd local_dispatch
source run.sh          # or export the secrets yourself (see below)
python3 ui/app.py      # opens on http://<host>:7899
```

Required environment (same as `run.sh`):

- `HF_TOKENS`, `MINIMAX_API_KEY`, `GIT_TOKEN` — injected into the run's config.env
- `LOCAL_SSH_PASS` — SSH password for the reserved machines
- `LOCAL_HTTP_PROXY`, `LOCAL_HTTPS_PROXY` — proxy for the remote Intel hosts

Optional:

- `LOCAL_DISPATCH_UI_PORT` — UI port (default `7899`)
- `LOCAL_RESULTS_DATASET` — results dataset (default `lvkaokao/lb_local`).
- `LOCAL_RESULTS_HF_TOKEN` — optional dedicated dataset token; otherwise `HF_TOKENS`
  is reused.
- `LOCAL_RESULTS_CACHE` — lightweight UI cache (default `ui/data/lb_local`).
- `LB_EVAL_UI_REPO` — optional local-path override; when set, the UI reads that
  local `results/` + `status/` tree and disables dataset synchronization.
- `LOCAL_DISPATCH_DB` — SQLite job store path (default `ui/jobs.db`)

## Tabs

| Tab | What it shows |
|-----|---------------|
| 🏆 Leaderboard | Finished/failed local runs from `lvkaokao/lb_local` dataset aggregates (model, scheme, status, avg acc, HF repo). |
| 📋 Queue | Local request status from the dataset's `status/**/*.json`. |
| 🎛️ GPU Machines | **Live** availability from the reservation API. |
| 📨 My Dispatches | Jobs submitted here — **auto-refreshing** status table + **live log tail**. |
| ➕ Submit | Submit a model (auto or manual machine) → runs `reserve_and_login.py` in the background. |

## Live monitoring of multiple jobs

- Each submitted dispatch runs in its own background thread with its own log file,
  so **many jobs run concurrently**.
- **My Dispatches** auto-refreshes every 2s (toggle with "Auto-refresh"):
  - the jobs table shows all jobs' live status at once;
  - the log panel streams the selected job's log (near real-time), or switch
    **Log view → "All running (combined)"** to watch every running job's tail together.

## How submit maps to the CLI

The Submit tab builds and runs:

```bash
python3 reserve_and_login.py --model <M> --scheme <S> --method <ME> \
    --user <U> --hours <H> [--server <SRV> [--gpus <G>]]
```

Auto mode omits `--server` (picks the best available GPU). Manual mode passes
`--server` (+ optional `--gpus`). Progress/status is tracked in the job store and
surfaced live in the UI.
