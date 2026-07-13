# `reserve_and_login.py` — 参数参考

`python3 reserve_and_login.py [参数...]`

> 机器/GPU 清单是**实时**从预约系统拉取的，下表为 2026-07-13 的快照，可能变化。
> 随时用 `python3 gpu_reserve.py list` 查看最新的机器与空闲情况。

---

## 参数总览

| 参数 | 必填 | 默认 | 可选值 / 说明 |
|------|------|------|---------------|
| `--model` | ✅ | — | HuggingFace 模型 id，如 `Qwen/Qwen2.5-7B`、`meta-llama/Llama-2-70b-hf` |
| `--scheme` | | `W4A16` | 量化方案：`W4A16` / `MXFP4` / `NVFP4`（4-bit），`W8A16` / `MXFP8`（8-bit）。仅用于显存估算 |
| `--user` | ✅ | — | 预约系统用户名 / 工号 |
| `--hours` | | `4` | 预约时长，**只能是 `1` / `2` / `3` / `4`**（后端上限 4 小时） |
| `--start` | | 立即 | 定时开始 `HH:MM`（当天，CST/+08:00），如 `14:30`；不填则立即开始 |
| `--server` | | 自动选 | 手动指定机器，见下方 [「--server 可选值」](#--server-可选值)。覆盖自动选卡 |
| `--gpus` | | — | 手动指定卡号：`0,1,2` 或 `auto:N`。**必须配合 `--server`** |
| `--ssh-user` | | `root` | SSH 登录用户（或环境变量 `LOCAL_SSH_USER`） |
| `--ssh-port` | | `22` | SSH 端口（或环境变量 `LOCAL_SSH_PORT`） |
| `--revision` | | `main` | 模型分支 / commit |
| `--dry-run` | | 关 | 只估算 + 选机，不真正预约 |
| `--no-ssh` | | 关 | 预约但跳过 SSH 登录验证 |
| `--json` | | 关 | 只输出机器可读的 JSON |

环境变量：
- `LOCAL_SSH_PASS` — SSH 密码（本地机器统一密码），实际登录验证时必填。
- `LOCAL_SSH_USER` / `LOCAL_SSH_PORT` — 覆盖 SSH 用户 / 端口默认值。
- `HF_TOKEN` — 私有 / 受限模型可选。

---

## `--server` 可选值

`--server` 的值是**关键字**，会去匹配机器的 `name` 或 `host`。
⚠️ **若关键字匹配到多台机器会报错**，需要用更具体的关键字（如 host IP）区分。

| 机器 name | host（IP） | GPU 型号 | 卡数 | 单卡显存 | 推荐 `--server` 值 |
|-----------|-----------|----------|------|----------|--------------------|
| 4090D (24GB) x8 | `10.239.23.90` | 4090D | 8 | 24 GB | `4090D` 或 `10.239.23.90` |
| L20x8-xFusion-1 | `10.112.228.229` | L20 | 8 | 48 GB | `xFusion` 或 `10.112.228.229` |
| L20x8-smc-1 | `10.239.167.17` | L20 | 8 | 48 GB | `smc` 或 `10.239.167.17` |
| 5090D (32GB) x8 | `10.239.11.53` | 5090D | 8 | 32 GB | `5090D` 或 `10.239.11.53` |
| H20 x8（云主机） | `118.195.144.97` | H20 | 8 | 96 GB | `118.195.144.97` ⚠️ |
| H20 x8（云主机） | `1.13.254.120` | H20 | 8 | 96 GB | `1.13.254.120` ⚠️ |
| RTX 6000D (84GB) x8 | `10.239.23.130` | RTX 6000D | 8 | 84 GB | `10.239.23.130`（当前被 Leng,Qiuyu 预留） |

⚠️ 标注的行：
- `L20` 有 **2 台** → 用 `L20` 会歧义报错。用 name 唯一片段 `xFusion` / `smc`，或 host IP 区分。
- `H20` 有 **2 台**（`118.195.144.97`、`1.13.254.120`）→ 用 `H20` 会歧义报错，请用 host IP。
- `4090D` / `5090D` 各只有 1 台，可直接用型号名。

> 关键字匹配 `name` 或 `host` 的**子串**（大小写不敏感），只要唯一即可。
> 例：`xFusion`、`smc`、`228.229`、`167.17` 都能唯一命中某一台 L20。

---

## `--gpus` 可选值

只在配合 `--server` 时有效：

| 形式 | 含义 |
|------|------|
| `auto:N` | 自动挑 N 块卡：优先空闲；**不够也不报错**，回退到低编号卡并 warn |
| `0,1,2` | 显式指定 device_id。**不拦截时段冲突**，只 warn 谁占着；只有卡号不存在才报错 |
| 省略 | 等价于 `auto:N`，N = 按模型大小算出的最少卡数 |

> 指定 `--server` 后一律不因冲突中断：预约即使被后端拒绝也会记录 `reservation_error` 并**继续 SSH**。

每台机器的 device_id 均为 `0..7`（8 卡）。

---

## `--scheme` 与显存

`--scheme` 只影响**显存估算**（量化输出精度）：4-bit 方案显存约为 8-bit 的一半。
估算同时考虑量化阶段（逐层，占用小）和评估阶段（整模型 + KV cache），取较大者。

| 方案 | 输出 bit |
|------|----------|
| `W4A16` / `MXFP4` / `NVFP4` | 4 |
| `W8A16` / `MXFP8` | 8 |

---

## 示例

```bash
# 自动：估算 7B 显存并挑最好的空闲机器
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --user alice --dry-run

# 指定 5090D 机器，自动选卡数
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --user alice --server 5090D

# 指定某台 H20（用 IP 避免歧义）+ 具体卡 2,3
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --user alice \
    --server 118.195.144.97 --gpus 2,3

# 定时 14:30 开始，约 4 小时，输出 JSON
python3 reserve_and_login.py --model meta-llama/Llama-2-70b-hf --user alice \
    --start 14:30 --hours 4 --json

# 真正预约 + SSH 验证
export LOCAL_SSH_PASS='...'
python3 reserve_and_login.py --model Qwen/Qwen2.5-7B --user alice --server 5090D
```
