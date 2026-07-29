# 运行方式：三种 method × GPU / XPU

`local_dispatch` 统一入口 `reserve_and_login.py`。三种量化 method 在 GPU（NVIDIA）
与 XPU（Intel B60）上用法一致，只改 `--method` 与 `--server`。

## method 一览

| `--method` | iters | 说明 | 支持 scheme |
|-----------|-------|------|-------------|
| `RTN`（默认） | 0 | 快速取整量化，无校准 | 全部 |
| `TUNING` | 200 | AutoRound 符号梯度调优，精度更好、更慢 | 全部 |
| `MODEL_FREE` | 0 | 逐 shard 权重量化，不加载全模型（大模型省显存） | W4A16 / MXFP4 / MXFP8 |

一致性保证（GPU 与 XPU 相同）：

1. **method 归一化**：显式 `MODEL_FREE` 不会被 `iters=0` 覆盖成 RTN。
2. **export_format 自动纠正**：`MXFP4/MXFP8 + MODEL_FREE` → 强制 `llm_compressor`
   （评估走 vLLM）；无需手动传 `--export-format`。
3. **命名**：后缀 `RTN` / `Tuning` / `ModelFree`，产物目录
   `<model>-AutoRound-<scheme>-<suffix>`。
4. **评估后端**：`auto_round` 导出 → GPU 用 hf / XPU 用 vLLM-XPU；
   `llm_compressor` 导出 → 两边都 vLLM。

---

## GPU（NVIDIA：B200 / 4090 / L20 / H20 …）

预约池机器不带 `--server` 走自动选卡，或用 `--server 4090D` / `H20` 等；
multi-hop B200 用 `--server sc09-b200` / `changwa1-b200`。

```bash
# RTN（默认）
python3 reserve_and_login.py --model Qwen/Qwen3-8B --scheme W4A16 --method RTN \
  --user kaokao --server sc09-b200 --gpus 0

# TUNING（iters=200 调优）
python3 reserve_and_login.py --model Qwen/Qwen3-8B --scheme W4A16 --method TUNING \
  --user kaokao --server sc09-b200 --gpus 0

# MODEL_FREE（W4A16）
python3 reserve_and_login.py --model Qwen/Qwen3-30B-A3B --scheme W4A16 --method MODEL_FREE \
  --user kaokao --server sc09-b200 --gpus 0

# MODEL_FREE（MXFP4，自动切 llm_compressor + vLLM 评估）
python3 reserve_and_login.py --model Qwen/Qwen3-30B-A3B --scheme MXFP4 --method MODEL_FREE \
  --user kaokao --server sc09-b200 --gpus 0,1
```

自动选卡（不指定机器）：

```bash
python3 reserve_and_login.py --model Qwen/Qwen3-8B --scheme W4A16 --method RTN --user kaokao
```

---

## XPU（Intel B60）

同一入口，`--server b60-xpu`；也可直接用 `xpu/xpu_dispatch.py`（支持 `--dry-run`）。

```bash
# RTN
python3 reserve_and_login.py --model Qwen/Qwen3-8B --scheme W4A16 --method RTN \
  --user kaokao --server b60-xpu --gpus 0

# TUNING
python3 reserve_and_login.py --model Qwen/Qwen3-8B --scheme W4A16 --method TUNING \
  --user kaokao --server b60-xpu --gpus 0

# MODEL_FREE（W4A16）
python3 reserve_and_login.py --model Qwen/Qwen3-30B-A3B --scheme W4A16 --method MODEL_FREE \
  --user kaokao --server b60-xpu --gpus 0

# MODEL_FREE 多卡（MXFP4，tensor_parallel_size 按卡数）
python3 reserve_and_login.py --model Qwen/Qwen3-30B-A3B --scheme MXFP4 --method MODEL_FREE \
  --user kaokao --server b60-xpu --gpus 0,1,2,3

# 直接派发器 + 预览
python3 xpu/xpu_dispatch.py --model Qwen/Qwen3-30B-A3B --scheme MXFP4 --method MODEL_FREE --gpus 0,1 --dry-run
```

---

## 需要的环境变量

```bash
# 结果 / 模型上传
export HF_TOKENS='hf_...'
export LOCAL_RESULTS_DATASET='lvkaokao/lb_local'   # 可选，默认即此
export LOCAL_RESULTS_HF_TOKEN='hf_...'             # 可选，缺省复用 HF_TOKENS
export MINIMAX_API_KEY='...'                       # agent fix-loop（OpenClaw/Copilot）

# GPU 预约池机器
export LOCAL_SSH_PASS='...'
export LOCAL_HTTP_PROXY='http://proxy.ims.intel.com:911'
export LOCAL_HTTPS_PROXY='http://proxy.ims.intel.com:911'

# multi-hop B200 / B60
export LOCAL_JUMP_SSH_PASS='...'   # tensorflow@10.23.167.71
export SC09_SSH_PASS='...'         # hshen@172.26.46.180
export B60_SSH_PASS='...'          # sdp@192.168.11.2
```

---

## 输出

两条路径都产出相同结构，回传到独立 dataset `lvkaokao/lb_local`：

```text
results/<org>/<artifact>/run_<run_id>/   日志、accuracy、request、run_report、session
results/<org>/<artifact>/results_<run_id>.json
status/<org>/<request_filename>.json
```

量化权重另上传到 HF model repo。多 task 目录/容器/secrets/结果全隔离。
