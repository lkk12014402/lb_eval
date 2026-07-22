# Local Dispatch UI — 环境变量与调度说明

本文说明 `local_dispatch/ui/app.py`（UI）依赖的**环境变量在哪配置**，以及 **app.py 如何调度**一次派发。

---

## 一、环境变量在哪配置

**原则：不写死在代码里。** 所有密钥/代理都在**启动 UI 之前**于 shell 里 `export`，
UI 进程继承它们，再原样传给每个派发子进程（`runner.py` 用 `env=os.environ.copy()`）。

推荐把它们写进 `local_dispatch/run.sh`，末尾启动 UI：

```bash
# local_dispatch/run.sh —— 配置密钥/代理后启动 UI
export HF_TOKENS='hf_xxx'
export MINIMAX_API_KEY='sk-xxx'
export GIT_TOKEN='ghp_xxx'
export LOCAL_SSH_PASS='intel@123'
export LOCAL_HTTP_PROXY='http://proxy.ims.intel.com:911'
export LOCAL_HTTPS_PROXY='http://proxy.ims.intel.com:911'
export LOCAL_RESERVE_USER='kaokao'      # UI Submit 里预约用户名的默认值

python3 ui/app.py                        # 用上述环境变量启动 UI
```

启动方式：

```bash
cd local_dispatch
source run.sh        # 或 bash run.sh
# UI 打开在 http://<host>:7899
```

### 变量用途

| 变量 | 用途 | 传递到哪里 |
|------|------|-----------|
| `HF_TOKENS` | HuggingFace 上传/下载 token | 容器内写入 `auto_quant/config.env` |
| `LOCAL_RESULTS_HF_TOKEN` | 可选的独立 dataset 写 token；未设置时复用 `HF_TOKENS` 第一个 token | `lvkaokao/lb_local` 结果上传 |
| `MINIMAX_API_KEY` | MiniMax agent key（OpenClaw/Copilot） | 容器内 config.env + OpenClaw auth |
| `GIT_TOKEN` | 结果回传 GitHub 的 token | 容器内 config.env |
| `LOCAL_SSH_PASS` | SSH 登录预约的机器（统一密码） | `reserve_and_login` / `remote_exec` |
| `LOCAL_HTTP_PROXY` / `LOCAL_HTTPS_PROXY` | 远程机器 git / docker build / 容器内 HF/pip 走代理 | 远程 shell + docker `--build-arg` + `-e` |
| `LOCAL_RESERVE_USER` | UI 预约用户名默认值（留空时用它） | UI 表单默认值 |

### 可选变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_DISPATCH_UI_PORT` | `7899` | UI 端口 |
| `LOCAL_DISPATCH_DB` | `ui/jobs.db` | 本地任务库 SQLite 路径 |
| `LB_EVAL_UI_REPO` | `../../lb_eval` | 读取 `results/`+`status/` 的 lb_eval 本地克隆路径（Leaderboard/Queue tab 用） |
| `LOCAL_SSH_USER` | `root` | SSH 登录用户（也作为预约用户名的次选默认） |
| `LOCAL_SSH_PORT` | `22` | SSH 端口 |
| `LOCAL_NO_PROXY` | `localhost,127.0.0.1,::1` | 远程 no_proxy 列表 |
| `GPU_RESERVE_BASE_URL` | `http://10.112.228.100:8688/` | 预约系统 API 地址（CLI 与 UI 的 GPU 数据都走它，改这一处即可换环境） |
| `LOCAL_RESULTS_DATASET` | `lvkaokao/lb_local` | 本地运行结果的 HF dataset repo |
| `LOCAL_RESULTS_CACHE` | `ui/data/lb_local` | UI 下载聚合 JSON/status 的轻量缓存 |
| `LOCAL_RESULTS_SYNC_INTERVAL` | `60` | UI 自动同步 dataset 的最短间隔（秒） |
| `LOCAL_JUMP_SSH_PASS` | — | 第一跳 `tensorflow@10.23.167.71` 的密码（未设时可回退到 `LOCAL_SSH_PASS`） |
| `SC09_SSH_PASS` | — | SC09 `hshen@172.26.46.180` 的目标密码 |
| `B60_SSH_PASS` | — | B60 `sdp@192.168.11.2` 的目标密码 |

### 环境变量的继承链

```
启动 shell (source run.sh)
   └─ python3 ui/app.py                     # 继承所有 export
        └─ runner: subprocess.Popen(env=os.environ.copy())   # 原样下传
             └─ reserve_and_login.py        # 读 LOCAL_SSH_PASS / LOCAL_*_PROXY
                  └─ remote_exec.py          # 组装远程脚本 + secrets env-file
                       └─ 远程机器 SSH + docker
                            └─ 容器 bootstrap: update_config_env 写入 config.env
                                 └─ auto.sh  # source config.env → 量化/评估/回传
```

---

## 二、app.py 如何调度

一次派发的完整链路：**UI → jobstore → 后台线程 → 子进程 → reserve_and_login.py**。

```
[Submit 按钮]  app.submit()
   ├─ jobstore.create_job()            # 写库，状态 Queued，记录 model/scheme/method/user/…
   └─ runner.start_dispatch(job.id)    # 起一个 daemon 线程（不阻塞 UI）
            │
            ▼
      runner._dispatch()               # 在后台线程里
        ├─ update_job → Running，写 log_path
        ├─ build_command(job)          # 拼 reserve_and_login.py 的 argv：
        │     python3 ../reserve_and_login.py \
        │         --model <M> --scheme <S> --method <ME> \
        │         --user <U> --hours <H> [--server <SRV> [--gpus <G>]]
        ├─ subprocess.Popen(cmd, cwd=local_dispatch, env=os.environ.copy())
        ├─ 逐行读子进程输出 → 写入 ui/logs/<job_id>.log
        │     并正则匹配 "Selected: …(host)… GPUs […]" 回填 host / reserved_gpus
        └─ 进程退出 → update_job → Finished（exit 0）/ Failed（非 0）(+exit_code)
```

再往下 `reserve_and_login.py`（单一入口）：
估算显存 → 预约机器 → SSH 登录 → 调 `remote_exec` 在远程 `docker build` + 跑 `auto.sh`。

### 调度特点

- **每个提交 = 一个独立线程 + 独立子进程 + 独立日志文件** → 天然并发，可同时跑多个请求。
- **UI 不阻塞**：Submit 后立即返回；`gr.Timer` 每 2 秒轮询 jobstore 刷新任务表和日志。
- **只有经 UI Submit 才写 `jobs.db`**：命令行直接跑 `reserve_and_login.py`（如旧 run.sh）
  不经过 jobstore，不会出现在 “My Dispatches” 里。

### jobs.db 更新时机

| 时机 | 动作 | 状态变化 |
|------|------|----------|
| 提交（Submit） | `create_job` | → `Queued` |
| 派发开始 | `update_job` | `Queued → Running`，写 `log_path` |
| 运行中 | `update_job` | 回填 `host` / `reserved_gpus`（从日志解析） |
| 运行结束 | `update_job` | `Running → Finished` / `Failed`（+`exit_code`） |

UI 每 2 秒读一次库，状态变化最多 2 秒内反映到界面。

---

## 三、实时日志与多任务监控

- 每个派发写自己的 `ui/logs/<job_id>.log`，多个任务并发互不干扰。
- **My Dispatches** 页每 2 秒自动刷新（可用 “Auto-refresh” 开关）：
  - 任务表同时显示所有任务的实时状态；
  - 日志面板可看**选中任务**的实时滚动，或切到 **“All running (combined)”** 同屏监控所有运行中任务的日志尾部。
- **看不到日志开头？** 用 **“Log position”** 选择：
  - `Tail (latest)` —— 看最新（默认，配合 Autoscroll 实时滚动到底部）；
  - `Head (start)` —— 看**开头**（自动关闭 autoscroll，不会被拽到底部）；
  - `Full` —— 看完整日志。
  - “Autoscroll to bottom” 开关可随时冻结滚动，方便往上翻读起始日志。

---

## 四、Leaderboard/Queue 数据源与刷新

默认数据源是 Hugging Face dataset **`lvkaokao/lb_local`**。`datasrc.py`
使用 `snapshot_download` 只同步：

- `results/*/*/results_*.json`
- `status/**/*.json`

到轻量缓存 `ui/data/lb_local`，不会下载大日志或量化权重。

若设置 **`LB_EVAL_UI_REPO`**，则完全切换为指定的本地路径，并停止网络同步。

- Leaderboard tab 读该目录下的 `results/**/results_*.json`；
- Queue tab 读 `status/**/*.json`。
- **纯本地文件读取**（`os.walk` + `json.load`），`datasrc.py` 里**没有任何 git pull/fetch/clone**。

**重启 app.py 会不会重新拉取/更新数据？**

- 每次启动、打开或刷新 Leaderboard/Queue 时会按同步间隔检查 dataset；
- 网络失败时保留并继续使用已有缓存；
- `LOCAL_RESULTS_SYNC_INTERVAL` 默认 60 秒，避免每次组件刷新都请求 Hub。

---

## 五、Model-Free 支持

**可以跑**，直接在 UI Submit 里 **Method 选 `MODEL_FREE`** 即可，其它照常填。

method 传递链（已验证一路贯通）：

```
UI Method (MODEL_FREE)
  → reserve_and_login.py --method MODEL_FREE
  → RemoteJob.method
  → request.json  "method": "MODEL_FREE"
  → auto.sh: case MODEL_FREE → MODEL_FREE=true, METHOD_SUFFIX=ModelFree
  → quantize.py: model_free=True 路由
```

正确性保障：

| 保障 | 位置 | 说明 |
|------|------|------|
| iters 不覆盖 method | `auto.sh` | 显式 MODEL_FREE 不会被 `iters=0` 归一化成 RTN |
| export_format 自动纠正 | `phases/quantize.py` | MXFP4/MXFP8 的 model-free **自动强制** `llm_compressor`（即使请求里是默认 `auto_round`），评估走 vLLM |
| 命名带 ModelFree | `remote_exec.py` / `upload_results_hf_dataset.py` | 文件名如 `..._MXFP4_4bit_mxfp4_MODEL_FREE.json`，结果目录 `...-AutoRound-MXFP4-ModelFree` |

实证：`test_6.log` 已有一次真实端到端成功 —— `Qwen3.5-35B-A3B / MXFP4 / MODEL_FREE → Finished`，
权重上传 + 结果回传均正常。

> W4A16 的 model-free 走 auto_round；MXFP4/MXFP8 的 model-free 自动切 llm_compressor。
> UI 里**无需手动管 export_format**。
