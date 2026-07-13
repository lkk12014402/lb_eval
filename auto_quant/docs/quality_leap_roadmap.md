# Roadmap: A Qualitative Leap for auto_quant (Agent Escalation + Recipes + Synthesis)

Goal: make `lb_eval/auto_quant` quantization + evaluation dramatically better along six axes —
**success rate, root-cause quality, fix/patch quality, tried-solutions summary, quantization recipe,
and summary-doc quality** — by combining the pluggable agent backend (OpenClaw + Copilot) with a
**cost-aware model escalation ladder** (MiniMax-M3 first → Opus 4.8 on stubborn cases).

This is a design/roadmap doc. It builds on what already exists; it does not rebuild it.

---

## 1. What we already have (grounded in the code)

| Capability | Where | Notes |
|---|---|---|
| Multi-attempt fix loop w/ shared agent session memory | `phases/agent_fix_loop.sh` | attempts retry, agent remembers prior tries |
| L1 deterministic taxonomy → seeds prompt | `error_analysis/taxonomy.py`, `taxonomy_classify()` | pattern guess + hints, agent may override |
| Structured diagnosis fields | `build_fix_prompt`, `extract_agent_analysis` | ROOT_CAUSE/COMPONENT/EVIDENCE/FIX_TIER/VERDICT/ERROR_CLASS/SOLUTION |
| Lessons self-learning (save/load/promote/git) | `save_lesson`, `load_all_lessons`, `promote_lessons.py` | per-error JSONL, promoted to signatures |
| Drift detection, CUDA regression guard, smoke tests | `agent_fix_loop.sh` | stuck-detection, refuses CPU fallback |
| Post-hoc failure analysis + skill | `error_analysis/analyze_failures.py` + `error_analysis` skill | openclaw + Part 0/8/9 methodology |
| Deterministic run report | `phases/generate_report.py` → `run_report.md` | fills fields from quant_summary.json / accuracy.json |
| Pluggable agent backend + BYOK MiniMax | `phases/agent_backends.sh` | `AGENT_BACKEND`; MiniMax via BYOK, Opus via GitHub/BYOK |
| Download-free model structure analyzer | `src/submission/model_analysis.py` (leaderboard) | module schema + ignore/layer_config suggestions |

**Takeaway:** the scaffolding is strong. The leap comes from (a) a **model escalation ladder**,
(b) **recipes** (net-new), and (c) **agent-synthesized summaries** replacing template fills.

---

## 2. Core architectural idea: a cost-aware escalation ladder

Today: one backend, one model, N attempts. Proposed: the fix loop climbs a **tier ladder**, cheap→
expensive, only escalating when a tier gives up or stalls.

```
Tier 0  PRE-EMPT   deterministic: taxonomy + Part 9 catalog + analyzer recipe prior
                   → known signatures fixed WITHOUT any LLM call
Tier 1  CHEAP      copilot + MiniMax-M3   (most attempts; ~free relative to Opus)
Tier 2  STRONG     copilot + Opus 4.8     (only if Tier 1 exhausts/declares HARD)
                   (openclaw stays available as an alternative Tier-1 for A/B)
```

Escalation triggers (any):
- Tier 1 used its attempt budget without success.
- Drift detected (same ERROR_CLASS K times) → a stronger model may break the loop.
- Agent emits `VERDICT: ESCALATE` (new signal: "I can't, needs a stronger model").
- Error class is in a curated "hard" set (e.g. custom-model code patches, mixed-precision layout).

Why this is the right shape:
- **Copilot spans the whole ladder** (MiniMax via BYOK, Opus via GitHub/BYOK) → a single backend
  can do both tiers; OpenClaw becomes an optional comparison backend, not a requirement.
- Directly lifts **success rate** on hard cases while keeping **cost** bounded (Opus only when needed).
- Reuses the `AGENT_BACKEND` abstraction we built — this is `run_agent_fix` gaining a tier arg.

Implementation sketch: `AGENT_TIERS="copilot:MiniMax-M3 copilot:opus-4.8"` in config.env; the loop
iterates tiers, resetting the attempt budget per tier, carrying the lessons/session forward.

---

## 3. Goal-by-goal work items

### G1 — Quantization success rate
- **Tier-0 pre-emption:** before attempt 1, match the failure signature against the Part 9 catalog
  and the model's recipe prior (§G5). Apply the known fix deterministically (no LLM). Cheapest win.
- **Escalation ladder (§2):** MiniMax handles the long tail; Opus rescues the hard 5-10%.
- **Verify-before-rerun everywhere:** ensure every fix runs `run_smoke_test` (already exists) before
  the expensive full phase re-run — fail fast, save GPU-hours.
- **Analyzer as prior:** feed `model_analysis` output (module schema, is_moe, indexer, recommended
  ignore/layer_config) into the FIRST prompt so the agent starts from the right ignore/mixed-precision.

### G2 — Root-cause quality
- Make BOTH backends load the `error_analysis` SKILL (Part 0 reasoning + Part 9 catalog) as
  `AGENTS.md` / instructions — currently only openclaw does.
- Enforce **attribution stability** (Part 9.1): same signature → same category/verdict; the prompt
  should cite the catalog entry when it matches.
- On escalation to Opus, ask for a **deeper** root cause (mechanism, not just symptom) — Opus is worth
  it here.

### G3 — Fix/patch quality
- Keep the FIX_TIER ladder (config < pip < workaround < code-patch); require the agent to justify the
  tier and show a smoke-test command.
- Copilot's stronger tool use + Opus reasoning improves custom-model code patches (the §Part 8.x
  cache-path patching technique) — the highest-value, hardest fixes.

### G4 — Tried-solutions / bug summary (synthesis)
- **New: `summarize_attempts`** — at loop end (success OR give-up), an agent reads all
  `attempt_N.log` + lessons and produces a readable narrative: *what was tried each attempt, why it
  failed, what finally worked (or why UNFIXABLE)*. Cheap tier (MiniMax) is fine here.
- Persist as `failure_diagnosis.md` / `resolution_summary.md` next to the run; feed the distilled
  form back into the lessons store (closing the self-learning loop with quality, not just JSONL rows).

### G5 — Quantization recipe (NET-NEW; needs your template)
- **Definition:** on SUCCESS, emit a reusable **recipe** capturing everything needed to reproduce the
  good quantization of this model *family*: scheme, method (RTN/TUNING/model-free), `ignore_layers`,
  `layer_config` (mixed precision), `export_format`, key dep versions (auto-round/transformers),
  model-specific quirks (indexer, shared-experts, custom code patches applied), and the rationale.
- **Keyed by model family / architecture** so it becomes a **prior** for the next similar model
  (Tier-0 pre-emption). Store under `recipes/<arch>.json` + a human `recipes/<arch>.md`.
- **Generation:** agent-synthesized from `quant_summary.json` + applied fixes + analyzer output,
  rendered into YOUR TEMPLATE. → please share the template so I match it exactly.

### G6 — Summary-doc quality
- Keep `generate_report.py` for the **deterministic facts** (status, timings, accuracy table).
- **Add an agent-synthesized narrative layer** on top: executive summary, root-cause story, the
  recipe, and the tried-solutions summary. Default model MiniMax; use Opus for flagship/hard runs.
- Result: `run_report.md` = deterministic facts + high-quality AI narrative, not template fills.

---

## 4. Recommended combination strategy

**Primary: standardize on Copilot as the agent, with a MiniMax→Opus ladder.**
- One backend spans both tiers (BYOK MiniMax + Opus), simplest to operate and to reason about cost.
- Copilot is stronger at tool use / large-context, and can run Opus 4.8 for the hard tail.

**Keep OpenClaw as an optional Tier-1 / comparison backend.**
- Useful for A/B (same MiniMax model, different agent framework) and as a fallback.
- Zero extra cost to keep — the pluggable layer already supports it.

**Cost controls (config.env knobs):**
- `AGENT_TIERS`, per-tier `MAX_FIX_ATTEMPTS`, an overall `MAX_OPUS_ATTEMPTS`/budget cap, and a
  "hard-error set" that may skip straight to Opus. Opus is opt-in per trigger, never the default.

---

## 5. Prioritized roadmap

1. ✅ **Recipe system (G5)** — DONE. `generate_recipe.py` + arch-keyed `recipes/` store, wired into auto.sh.
2. ✅ **Escalation ladder (§2)** — DONE. `AGENT_TIERS="minimax opus"`; loop escalates on drift or
   `VERDICT: ESCALATE` (and tries a stronger tier before honoring UNFIXABLE). Tier activation toggles
   Copilot BYOK MiniMax ↔ Opus.
3. ✅ **Attempt/solution synthesis (G4) + report narrative (G6)** — DONE. `phases/synthesize.sh`
   (cheap-tier agent) writes `resolution_summary.md`; embedded into `run_report.md` (Summary & Root
   Cause) and the recipe summary (via `generate_recipe.py --narrative`).
4. ✅ **Tier-0 pre-emption (G1)** — DONE (proactive form). `quantize.py` consults the recipe store for
   `<arch>__<scheme>`; when the user didn't override, the FIRST attempt uses proven
   ignore_layers/layer_config (precedence: user > recipe prior > default). `RECIPES_DIR` exported by auto.sh.
5. **Verify-before-rerun + analyzer prior (G1/G3)** — smoke tests already exist; remaining: wire the
   download-free analyzer output into attempt-1 context. NEXT.

### Done in this pass — two clarifications
- **Copilot needs no skill.** The structured diagnosis fields + the debugging methodology now live
  INLINE in `build_fix_prompt` (self-contained), so both backends behave identically without a skill
  file. (OpenClaw can still load the `error_analysis` skill; it's now redundant with the prompt.)
- **Copilot session parsing.** Copilot's `-p` output is already human-readable and teed to
  `attempt_N.log` (so `extract_agent_analysis`/VERDICT/lessons work identically). auto.sh now also
  gathers those per-phase transcripts into `session_copilot_<phase>.md` so they upload alongside
  OpenClaw's `session_*.md` (OpenClaw's JSONL still gets formatted by `format_sessions.py`).

---

## 6. Open questions (need your input)

1. **Recipe template** — please share it; G5 + G6 render into it.
2. **Budget** — acceptable Opus usage per run (e.g. max 2 attempts / flagship-only)? Drives triggers.
3. **Backend stance** — OK to make Copilot the primary and keep OpenClaw as comparison-only? Or keep
   both first-class?
4. **Escalation trigger** — MiniMax attempt budget before Opus (e.g. 4), or only on `VERDICT: ESCALATE`
   + drift, or both?
5. **Recipe scope** — per exact model, or per architecture family (recommended, more reusable)?
