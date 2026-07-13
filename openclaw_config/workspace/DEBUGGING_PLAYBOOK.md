# Debugging Playbook — pointer

The framework-agnostic **expert debugging discipline** (attribute the bug's owning layer →
verify hypotheses against real state → reason about what data physically exists →
differential-debug against a working reference → confirm the fix takes effect → iterate one
root cause at a time) lives as the **single source of truth** in:

    skills/error_analysis/SKILL.md  →  "Part 0: How an Expert Actually Reasons"
                                        "Part 8: Deep Root-Cause Techniques"

These two Parts are **domain-agnostic** and apply to ANY error (services, DBs, frontends, CI,
not just quantization). Parts 1–9 of that skill apply the discipline to the auto-quant pipeline.

## Why a pointer and not a copy

- The error_analysis skill runs inside an automated fix loop; keeping the reasoning **inline**
  in its `SKILL.md` guarantees it's always in context. A separate content file would add a
  "did the agent remember to read it?" failure mode, and two copies would drift.

## If you add a NEW debugging skill for another domain (K8s, frontend, …)

Reuse Part 0 + Part 8 verbatim as that skill's reasoning core, then write your own
domain catalog (the equivalent of Parts 1–9). At that point, if duplication becomes real,
extract Part 0/8 into a shared file and have each skill reference it — not before (YAGNI).
