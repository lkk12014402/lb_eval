# local_dispatch — Local GPU dispatch path

An alternative to the Azure/AWS cloud path: run the auto-quant pipeline on
**local / on-prem GPU machines** reserved through the GPU reservation system
(`gpu_reserve.py`, live at the API in that file's `BASE_URL`).

- **Step 1** (`reserve_and_login.py`): select → reserve → SSH login → **then automatically builds & runs `auto.sh` on the host** (step 2). This is the single entry point.
- **Step 2** (`remote_exec.py`): the build+run engine that step 1 calls. Can also be run
  standalone against an already-reserved host. `git clone` lb_eval → `docker build` →
  `docker run auto.sh` (Azure-style, no agent registration).

## Modules

| File | Purpose |
|------|---------|
| `gpu_catalog.py` | Real local GPU inventory: model → VRAM, max cards, **quality rank** (prefer better GPUs). Keep in sync with the reservation API's `model` strings. |
| `vram_estimator.py` | **Download-free** VRAM estimate. Exact param count via `get_safetensors_metadata` (headers only), falls back to config dims / name regex. Sizes both the quantize phase (layerwise) and eval phase (whole model), takes the max. Standalone — no leaderboard import. |
| `reserve_and_login.py` | Main flow: estimate → pull live inventory → pick best-fitting machine (prefer highest-quality GPU, fewest cards) → reserve (immediate or scheduled) → verify SSH. |
| `remote_exec.py` | **Step 2** — on the reserved host: `git clone` lb_eval → `docker build` the Azure `agent.dockerfile` (layer-cached) → `docker run` `auto.sh` with the reserved GPUs pinned via the request's `cuda_visible_devices`. Reproduces the Azure pipeline but skips the self-hosted-agent registration. |

**See [`PARAMETERS.md`](./PARAMETERS.md) for every parameter's allowed values and the live `--server` machine list.**

## Usage

```bash
# Dry-run: estimate + select only, no reservation
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --scheme W4A16 --user alice --dry-run

# Reserve immediately for 4h and verify SSH login
export LOCAL_SSH_PASS='...'          # unified password for local machines
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --user alice --hours 4

# Scheduled start (CST), machine-readable output
python3 reserve_and_login.py --model meta-llama/Llama-2-70b-hf --user alice \
    --start 14:30 --hours 4 --json

# Manual: pick the machine (auto-sizes cards, avoids busy cards)
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --user alice --server H20

# Manual: pick exact machine AND cards (forces those cards; warns on overlap)
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --user alice --server 118.195 --gpus 2,3
```

### Step 2 — build & run on the reserved host

```bash
# Dry-run: print the exact remote script + request.json (review before running)
python3 remote_exec.py --host 118.195.144.97 --gpus 2,3 \
    --model Qwen/Qwen2.5-7B --scheme W4A16 --dry-run

# Real run: clone → docker build → docker run auto.sh, streaming logs back
export LOCAL_SSH_PASS='...'
python3 remote_exec.py --host 118.195.144.97 --gpus 2,3 \
    --model Qwen/Qwen2.5-7B --scheme W4A16 --method RTN
```

- `--gpus` here are the **physical** reserved card ids (from step 1). The container
  is launched with `--gpus "device=<ids>"`; inside it the cards re-index to `0..N-1`,
  and the request's `cuda_visible_devices` is set accordingly so `auto.sh` pins both
  the quantize and eval phases to exactly those cards.
- lb_eval is cloned from `GIT_REPO`/`GIT_BRANCH` (defaults match `auto_quant/config.env`).
- The HF cache is mounted from a persistent host dir (`--hf-cache`, default `~/hf_cache`)
  so re-runs don't re-download.
- Pass `--request-json <path>` to reuse an existing request instead of generating one.

### Azure parity: OpenClaw setup + secrets in config.env (important)

The committed `config.env` ships with **empty** secrets (`HF_TOKENS=`, `MINIMAX_API_KEY=`,
`GIT_TOKEN=`), and Azure fills them at runtime. `auto.sh` does `source config.env`, so
**passing secrets via `-e` alone is not enough** — the empty values in config.env would
override them. The dispatch therefore reproduces Azure's two pre-`auto.sh` steps **inside
the container** (see `_container_bootstrap.sh`):

1. **Set Up OpenClaw** — `cp -r openclaw_config/ /root/.openclaw/` + `sync_minimax_key.py`
   (required because `AGENT_BACKEND=openclaw`; `auto.sh` deliberately does not do this).
   Skip with `--no-openclaw-setup` if you switch backends.
2. **Update Config** — `update_config_env.py` writes the secrets (and the **real** proxy,
   opposite of Azure which clears it) **into** config.env before `auto.sh` sources it.

Provide the secrets as environment variables on the dispatching machine (staged to the
host as a `chmod 600` env-file, injected via `--env-file`, and scrubbed after the run):

```bash
export HF_TOKENS='hf_...'            # or HUGGINGFACE_TOKEN / HF_TOKEN
export MINIMAX_API_KEY='...'         # or MINIMAX_KEY (also synced into OpenClaw auth)
export GIT_TOKEN='...'
export LB_STORAGE_BLOB_TOKEN='...'   # optional
export LOCAL_SSH_PASS='...'
export LOCAL_HTTP_PROXY='http://proxy.ims.intel.com:911'
export LOCAL_HTTPS_PROXY='http://proxy.ims.intel.com:911'

python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --scheme W4A16 --user kaokao \
    --server L20x8-smc-1 --gpus 0,1,2,3
```

### Manual card-conflict policy

指定 `--server` = "我知道我在做什么，直接约并让我 SSH"。**指定机器后完全不因冲突中断**：

- **显式 `--gpus 0,1`** — 严格用这几张卡，只 warn 谁占着，不拦截。
- **`auto:N` / 省略** — 优先选空闲卡；**不够也不报错**，回退到低编号卡（warn 重叠）。
- **预约被后端拒绝**（如时段冲突）也**不中断**，记录 `reservation_error` 后**照常 SSH**。

（自动模式——不带 `--server`——仍会在无机器可容纳或预约失败时报错终止。）

### Environment

- `LOCAL_SSH_PASS` — SSH password (unified across local machines). Required for login verify.
- `LOCAL_SSH_USER` — SSH user (default `root`).
- `LOCAL_SSH_PORT` — SSH port (default `22`).
- `HF_TOKEN` — optional, for gated/private models.

## Design decisions (confirmed with user)

1. **Standalone VRAM estimation** — no cross-repo import of the leaderboard.
2. **Prefer better GPUs** — pick the highest-quality machine that fits (opposite of the
   leaderboard's cheapest-fit), then fewest cards.
3. **Unified SSH** — user + `LOCAL_SSH_PASS`, same for all machines.
4. **Immediate and scheduled** start both supported (`--start HH:MM`).

## Notes

- `duration_hours` is capped at 1–4 by the reservation API.
- Multi-GPU aggregate VRAM uses a 0.85 efficiency factor (parameter duplication / comm buffers).
- If the model size can't be determined, the tool errors out rather than under-sizing.
