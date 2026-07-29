# Agent fix 升级流程（local_dispatch）

## 回答你的问题

> 量化报错了，会先走 copilot + MiniMax 吗？迭代几轮解决不了，最后走 copilot + Opus4.8 吗？

**不是。默认第一步是 `openclaw + MiniMax`**，顺序如下：

```
tier 0   openclaw + MiniMax      ← 量化/评测报错，先由它修
   │  （卡住 / UNFIXABLE / ESCALATE）
   ▼
tier 1   copilot  + MiniMax      ← 升级到 Copilot（同一个 MiniMax key，BYOK）
   │  （卡住 / UNFIXABLE / ESCALATE）
   ▼
tier 2   copilot  + Opus 4.8     ← 仅当显式开启时才有这一层
```

- 默认 `AGENT_TIERS="openclaw minimax"` → **梯子止于 tier 1**（Opus 关闭，省预算）。
- 要用 Opus 兜底：`AGENT_TIERS="openclaw minimax opus"`（且需 `COPILOT_GITHUB_TOKEN`）。
  这时“几轮解决不了 → 最后走 copilot + Opus4.8”才成立。

## 升级触发条件（在同一 tier 内每次 attempt 后判断）

| 触发 | 说明 | 有更强 tier | 无更强 tier |
|------|------|------------|------------|
| `VERDICT: UNFIXABLE` | agent 判定本层修不了 | 升级 | 终止 |
| `VERDICT: ESCALATE` | agent 主动请求更强模型 | 升级 | 继续重试到上限 |
| **drift** | 同类错误连续 `DRIFT_THRESHOLD=2` 次没变 | 升级 | 终止 |
| 有进展 | 报错类别变了 / 走到更深阶段 | 重置 drift，**留在本层** | 同 |
| 达到 `MAX_FIX_ATTEMPTS`(XPU=5) | 总尝试次数用尽 | 终止 | 终止 |

> 升级时会 `drift_count=0`、`prev_eff_class=""`，让更强的模型从干净状态重新诊断。

## 完整流程图

```mermaid
flowchart TD
    A[phase 运行: quantize / evaluate] --> B{成功?}
    B -->|是| DONE([✅ 该 phase 通过])
    B -->|否| INIT[进入 fix-loop\ntier=0 openclaw+MiniMax\nattempt=1]

    INIT --> RUN[run_agent_fix\n当前 tier 的 backend+model]
    RUN --> RERUN{重跑 phase\n通过?}
    RERUN -->|是| DONE

    RERUN -->|否| V{检查 agent VERDICT}
    V -->|UNFIXABLE| SU{有更强 tier?}
    SU -->|有| ESC[升级 tier+1\n重置 drift]
    SU -->|无| STOP([❌ 终止: UNFIXABLE])

    V -->|ESCALATE| SE{有更强 tier?}
    SE -->|有| ESC
    SE -->|无| CONT

    V -->|FIXABLE / 无| D{同类错误?\n(drift)}
    D -->|变了/有进展| RESET[重置 drift\n留在本层]
    D -->|连续2次未变| SD{有更强 tier?}
    SD -->|有| ESC
    SD -->|无| STOP2([❌ 终止: drift])

    ESC --> CONT
    RESET --> CONT
    CONT{attempt < 上限?}
    CONT -->|是| INCR[attempt+1] --> RUN
    CONT -->|否| STOP3([❌ 终止: 达到 MAX_FIX_ATTEMPTS])
```

## 一个具体例子（量化 OOM，Opus 已开启）

```
tier0 openclaw+MiniMax  attempt1  OOM        → 修 batch/mem，重跑
tier0 openclaw+MiniMax  attempt2  还是 OOM   → 同类错误 streak=1
tier0 openclaw+MiniMax  attempt3  还是 OOM   → streak=2 ≥ drift阈值 → 升级
tier1 copilot+MiniMax   attempt4  OOM变成 dtype 错  → 有进展，drift重置
tier1 copilot+MiniMax   attempt5  dtype 错没变       → streak=1
                        （达到 MAX_FIX_ATTEMPTS=5 → 若仍失败则终止；
                         或 agent 主动 ESCALATE → 升到 tier2 Opus 兜底）
tier2 copilot+Opus4.8   ...       深度多文件推理修复 → 重跑通过 ✅
```

> 注：若 `AGENT_TIERS="openclaw minimax"`（默认，无 opus），上例在 tier1 用尽次数后即终止，
> 不会有 tier2。要 Opus 兜底须显式加 `opus` 并配置 GitHub token。
