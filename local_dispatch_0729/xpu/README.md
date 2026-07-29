# Intel B60 (XPU) execution

Independent XPU runner for the local-dispatch pipeline. Unlike the CUDA path
(`../remote_exec.py` + lb_eval `auto.sh`, which assert `torch.cuda`), this runs a
standalone AutoRound-XPU quantize + vLLM-XPU eval flow, but reuses the same
OpenClaw/Copilot provisioning, model upload, report generation and independent
HF dataset result upload.

## Connectivity (3 hops)

```
tensorflow@10.23.167.71            (LOCAL_JUMP_SSH_PASS)
  -> guest@146.152.205.45          (jump host key ~/.ssh/id_rsa_4096)
     -> sdp@192.168.11.2           (B60_SSH_PASS)
```

The dispatcher opens a jump-host-local `-L` forward to the B60 through the guest
hop, then a Paramiko `direct-tcpip` channel over that forward (so SFTP + streamed
execution work end to end).

## Runtime

- Base image: `intel/llm-scaler-vllm:0.21.0-b1` — the Intel B60/B70-validated
  stack (torch `2.11.0+xpu`, vLLM `0.21`). The upstream `vllm/vllm-openai-xpu:nightly`
  reports `torch.xpu.device_count()==0` on this host, so it is **not** used.
- Derived image `xpu-openclaw:local` (see `Dockerfile`) adds AutoRound, lm-eval,
  Node.js + OpenClaw + GitHub Copilot CLI — matching the CUDA agent image.
- Device isolation via `ZE_AFFINITY_MASK`; container runs with
  `--device /dev/dri --ipc=host --privileged`.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Derived XPU image (base + AutoRound/lm-eval/OpenClaw/Copilot). |
| `xpu_quantize.py` | Standalone AutoRound XPU quantize (`device_map="xpu"`), writes `quant_summary.json`. |
| `xpu_evaluate.py` | lm-eval + vLLM XPU (`enforce_eager`), writes `accuracy.json`. |
| `xpu_pipeline.sh` | In-container orchestration: quantize → eval → report → model upload → dataset upload. |
| `xpu_container_bootstrap.sh` | OpenClaw setup + secrets→config.env, then runs the pipeline. |
| `xpu_dispatch.py` | Host-side dispatcher: 3-hop tunnel, stage files, `docker build`, `docker run`. |

## Usage

Via the single entry point (auto-routes B60 to the XPU dispatcher):

```bash
export LOCAL_JUMP_SSH_PASS='...'
export B60_SSH_PASS='...'
export HF_TOKENS='hf_...'

python3 ../reserve_and_login.py \
  --model Qwen/Qwen3-1.7B --scheme W4A16 --method RTN --user kaokao \
  --server b60-xpu --gpus 0
```

Or drive the dispatcher directly (supports `--dry-run` to preview the remote script):

```bash
python3 xpu_dispatch.py --model Qwen/Qwen3-1.7B --scheme W4A16 --gpus 0
python3 xpu_dispatch.py --model Qwen/Qwen3-1.7B --scheme W4A16 --gpus 0 --dry-run
```

## Validated

- Image builds on B60 (`BUILD_RC=0`); inside the container `torch.xpu.is_available()`
  is True with `ZE_AFFINITY_MASK` masking to the reserved card(s).
- Real quantization verified: `Qwen/Qwen3-0.6B` W4A16 → success in ~116s on one B60,
  correct `quant_summary.json` (status=success).

## Notes

- `--base-image` / `XPU_BASE_IMAGE` env overrides the base (e.g. a self-built
  `docker/Dockerfile.xpu` from vLLM main).
- Multi-card: pass `--gpus 0,1`; `ZE_AFFINITY_MASK` and vLLM `tensor_parallel_size`
  follow the reserved card count.
- Scope: quantize + eval + uploads. The agent fix-loop reuses OpenClaw/Copilot, but
  the deterministic XPU pipeline does not call lb_eval's CUDA `auto.sh`.

## Multi-card (tensor parallel)

Reserve multiple cards with `--gpus`; the runner sets both device selectors on the
container and derives `tensor_parallel_size` from the card count (per `eval_xpu.sh`):

```bash
python3 xpu_dispatch.py --model Qwen/Qwen3-30B-A3B --scheme MXFP4 --method RTN --gpus 0,1,2,3
```

- `ZE_AFFINITY_MASK=<ids>` and `ONEAPI_DEVICE_SELECTOR=level_zero:<ids>` are set to the
  reserved physical cards (matching the validated recipe, which sets both).
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` is set for multi-proc TP.
- vLLM eval `model_args` mirror the recipe: `tensor_parallel_size=N`, `dtype=bfloat16`,
  `enforce_eager=True`, `enable_prefix_caching=False`, plus batching knobs.

Give concurrent tasks on the same B60 **non-overlapping** cards (e.g. task A `--gpus 0,1`,
task B `--gpus 2,3`) — dirs/containers/secrets are isolated per run, but physical cards are not.

### vLLM eval tuning (env overrides)

`xpu_evaluate.py` honours these env vars (defaults match `eval_xpu.sh`):

| Env | Default | Meaning |
|-----|---------|---------|
| `VLLM_MAX_MODEL_LEN` | 8192 | max sequence length |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | 32768 | scheduler token budget |
| `VLLM_MAX_NUM_SEQS` | 128 | max concurrent sequences |
| `VLLM_MAX_GEN_TOKS` | 2048 | max generated tokens |
| `VLLM_GPU_MEM_UTIL` | 0.9 | XPU memory fraction |
| `VLLM_DTYPE` | bfloat16 | compute dtype |

## Agent fix-loop parity (plan B)

The XPU pipeline **reuses lb_eval's `phases/agent_fix_loop.sh` unchanged** (zero risk to
the GPU path) and layers XPU device behaviour on top via `xpu_fixloop_overrides.sh`:

- `xpu_pipeline.sh` sources the loop, then sources the overrides (last-defined bash
  function wins), and sets `REQUIRE_CUDA=false`.
- Overridden: `cleanup_stale_gpu_procs` (xpu-smi, no nvidia-smi) and `build_fix_prompt`
  (XPU wording — `device_map="xpu"`, `ZE_AFFINITY_MASK`, don't reinstall torch).
- `quantize` and `evaluate` run through `agent_fix_loop` via env-driven wrappers
  (`xpu_quantize_wrapper.sh`, `xpu_evaluate_wrapper.sh`), so a phase failure triggers
  OpenClaw analysis → fix → re-run → lesson, exactly like GPU.
- Report/recipe use the same `auto_quant/phases/generate_report.py`.
- Lessons stay local (never auto-pushed to git); results still go only to the dataset.

If `openclaw` or the loop file is unavailable, the pipeline falls back to deterministic
phase execution (no fix-loop) so a run still completes.

**Retired**: the earlier standalone `xpu_pipeline.sh` phase blocks are replaced by the
fix-loop wrappers. `xpu_quantize.py` / `xpu_evaluate.py` remain as the deterministic
phase implementations invoked by the wrappers.
