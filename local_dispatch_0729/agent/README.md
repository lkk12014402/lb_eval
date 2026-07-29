# local_dispatch agent fix backends (OpenClaw + Copilot, tier escalation)

Self-contained agent-fix layer for local_dispatch. It is a **drop-in overlay** —
the committed `lb_eval` scripts are never modified, so the normal (non-local-dispatch)
leaderboard flow is completely unaffected. Everything here is applied only inside the
per-run container of a local GPU/XPU dispatch.

## Files

| File | Role |
|------|------|
| `agent_backends.sh` | Pluggable backends: `run_openclaw_fix`, `run_copilot_fix` (BYOK + GitHub), `run_agent_fix` dispatcher, and the tier ladder (`agent_tier_activate` / `agent_tiers_count`). |
| `agent_fix_loop.sh` | Tier-aware fix loop (= production loop + escalation). Sources `agent_backends.sh` and calls `run_agent_fix`. |
| `copilot_config/settings.json` | Copilot CLI baseline (model/effort/contextTier/allowedUrls). CLI flags override it. |
| `autoround_pr.sh` | Optional (default OFF): capture agent fixes that land in the `auto_round` package as a patch, and open a PR when a push token/repo are configured. |

## Escalation ladder

`AGENT_TIERS` is a space-separated, ordered list of tiers. The loop starts at tier 0
(cheapest) and escalates when a tier stalls (drift), the agent prints
`VERDICT: UNFIXABLE` (with a stronger tier still available), or `VERDICT: ESCALATE`.

Default ladder (Opus is **off** to save budget):

```
AGENT_TIERS="openclaw minimax"
  tier 0  openclaw + MiniMax      (OpenClaw CLI, MINIMAX_API_KEY)
  tier 1  copilot  + MiniMax      (Copilot BYOK, reuses MINIMAX_API_KEY, no GitHub auth)
```

Enable the strong tier explicitly:

```
AGENT_TIERS="openclaw minimax opus"
  tier 2  copilot  + Opus 4.8     (needs COPILOT_GITHUB_TOKEN / GH_TOKEN)
```

### `AGENT_TIERS` = 空格分隔的有序列表，每个词就是一个 tier

按顺序尝试：第一个词是起始 tier，卡住后依次向右升级。例如
`AGENT_TIERS="openclaw minimax"` 就是**同时开启 tier0 和 tier1**（无 Opus）。

tier 名 → backend + 模型的映射（`agent_backends.sh` 的 `agent_tier_activate`）：

| 列表里的词 | backend | 模型 | 认证 |
|-----------|---------|------|------|
| `openclaw` | OpenClaw CLI | MiniMax | `MINIMAX_API_KEY` |
| `minimax` | **Copilot** (BYOK) | MiniMax-M3 | `MINIMAX_API_KEY`（无需 GitHub） |
| `opus` | Copilot | Opus 4.8 | `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |
| `sonnet` | Copilot | Sonnet 4.5 | `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` |

> ⚠️ 易混淆：tier 名 `minimax` 指的是 **copilot+MiniMax**，不是 openclaw。openclaw 那层
> 的名字就是 `openclaw`（它内部也用 MiniMax）。

常见组合：

```bash
AGENT_TIERS="openclaw minimax"        # tier0+tier1（默认）
AGENT_TIERS="openclaw minimax opus"   # tier0+tier1+tier2
AGENT_TIERS="minimax"                 # 只用 copilot+MiniMax 一层
AGENT_TIERS="openclaw"                # 只用 openclaw 一层（等于改动前的原行为）
```

## Config (env — forwarded by the dispatchers via the per-run env-file)

| Var | Meaning | Default |
|-----|---------|---------|
| `AGENT_TIERS` | tier ladder | `openclaw minimax` |
| `MINIMAX_API_KEY` | MiniMax key (openclaw tier 0 + copilot BYOK tier 1) | — |
| `COPILOT_GITHUB_TOKEN` / `GH_TOKEN` | GitHub auth for the Opus/Sonnet tier | — |
| `AUTOROUND_PR_ENABLED` | `1` to capture auto_round patches | `0` (off) |
| `AUTOROUND_REPO` | target repo for the auto_round PR | — |
| `AUTOROUND_PR_TOKEN` | push token for the PR | — |
| `AUTOROUND_PR_BASE` | PR base branch | `main` |

Copilot BYOK (tier 1) mirrors `test_copilot_minimax.sh`: it points Copilot at
`https://api.minimaxi.com/anthropic` (Anthropic-compatible), sends `MiniMax-M3`, and
maps onto a `claude-sonnet-4` agent profile — no `api.github.com` needed.

## 环境变量配置（在 `local_dispatch/run.sh` 里 export）

配置入口就是 **`run.sh`**：在启动 `reserve_and_login.py` 之前 `export`。数据流：

```
run.sh export  →  dispatcher._collect_secrets() 读 os.environ  →  写入 _secrets.env
               →  docker --env-file  →  容器内 fix-loop / autoround_pr 读取
```

GPU 与 XPU 两条路径都已接好（两个 dispatcher 的 `_collect_secrets` 都含这些键），
**只在容器内生效，不影响非 local_dispatch 的正常流程**。空值会被自动过滤（= 用默认）。

### 场景 1 · 默认梯子（openclaw+MiniMax → copilot+MiniMax）
无需新增变量，已有的这条即可（tier0 与 tier1 共用）：
```bash
export MINIMAX_API_KEY='sk-cp-...'
```
默认 `AGENT_TIERS="openclaw minimax"`，Opus 关闭。

### 场景 2 · 加 Opus4.8 兜底（copilot+Opus，需 GitHub token）
```bash
export AGENT_TIERS="openclaw minimax opus"
export COPILOT_GITHUB_TOKEN='ghp_...'      # 或 GH_TOKEN
```
只有这样，“几轮解决不了 → 最后走 copilot+Opus4.8” 才生效。

### 场景 3 · auto_round bug → 出 patch / 提 PR（可选，默认关）
```bash
export AUTOROUND_PR_ENABLED=1
export AUTOROUND_REPO='intel/auto-round'   # 目标仓库（或你的 fork）
export AUTOROUND_PR_TOKEN='ghp_有push权限'  # 不填则只存 patch，不提 PR
export AUTOROUND_PR_BASE='main'            # 可选，默认 main
```
`run.sh` 底部已内置一段注释好的模板，取消注释填值即可。

> ⚠️ `run.sh` 含真实 token，注意不要提交（建议与 `.env.multihop` 一样 gitignore）。

## How it is wired (no lb_eval edits)

- **XPU** (`xpu/xpu_pipeline.sh`): sources `agent/agent_fix_loop.sh` (tier-aware) then
  `xpu/xpu_fixloop_overrides.sh` (device funcs), calls `agent_backend_setup`.
- **GPU** (`remote_exec.py` `CONTAINER_BOOTSTRAP`): overlays `agent_backends.sh` +
  `agent_fix_loop.sh` + `settings.json` onto the freshly cloned repo so the stock
  `auto.sh` picks up the tier loop; installs the Copilot CLI at runtime (the committed
  GPU image ships OpenClaw only) and provisions the Copilot settings baseline.

Both paths default to `openclaw minimax`; export `AGENT_TIERS="openclaw minimax opus"`
(plus a GitHub token) to add the Opus tier.

## 升级流程图

完整的 tier 升级流程图、触发条件表、以及一个"量化 OOM"逐轮走位示例见 **[FLOW.md](FLOW.md)**。
简述：phase 失败 → tier0 openclaw+MiniMax 重试；卡住（drift 同类错误 ×2 / `VERDICT: UNFIXABLE`
/ `VERDICT: ESCALATE`）→ 升级 tier1 copilot+MiniMax → （若开启）tier2 copilot+Opus4.8；
有进展则重置 drift 留在本层；达到 `MAX_FIX_ATTEMPTS`(XPU=5) 或无更强 tier 则终止。

## auto_round → PR (optional, token-gated)

When `AUTOROUND_PR_ENABLED=1`, the pipeline snapshots the pristine `auto_round`
package before the fix loop, and afterwards diffs it. If the agent modified
`auto_round`, a `git apply -p1`-compatible `autoround_fix.patch` is written into the
run dir (and uploaded with the run). If `AUTOROUND_REPO` + `AUTOROUND_PR_TOKEN` are
set, a branch is pushed for a PR; otherwise the patch is retained and PR submission is
skipped with a clear log line (fill in the token later).
