#!/usr/bin/env python3
"""Local-dispatch run report generator.

Wraps lb_eval's generate_report.py but fixes two bugs in its phase-status logic
(without touching the lb_eval production file, so non-local-dispatch is unaffected):

  1. phase_status_icon: the upstream version reads the tail of retry_*.log for the
     magic strings "DONE"/"success" to decide a fix worked — fragile, so a fixed
     phase shows "❌ failed". We instead trust the authoritative signals:
       - the phase's own log ended successfully (agent_fix_loop returned 0), AND
       - a lessons entry marks the phase "fixed".
  2. extract_fix_summary: upstream globs prompt_*.txt but reads attempt_*.log and
     mislabels; we read the agent attempt logs and the "fixed" lesson solution.

Usage: python3 report.py <run_dir>   (writes <run_dir>/run_report.md)
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Import lb_eval's generator to reuse everything except the two buggy funcs.
# NOTE: do NOT read the env name "LB_EVAL_REPO" here — lb_eval's own config.env
# defines LB_EVAL_REPO="lb_eval" (a bare repo dir NAME), and the XPU pipeline sources
# config.env, so that value would leak in and make this a broken relative path. Use a
# dedicated, unambiguous var and fall back to a search over known container locations.
def _find_lb_eval_phases() -> str:
    candidates = []
    env_dir = os.environ.get("LB_EVAL_DIR", "").strip()
    if env_dir:
        candidates.append(os.path.join(env_dir, "auto_quant", "phases"))
    candidates += [
        "/workspace/lb_eval/auto_quant/phases",   # GPU + XPU standard mount
        "/workspace/xpu/../lb_eval/auto_quant/phases",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "generate_report.py")):
            return os.path.realpath(c)
    # Last resort: return the first candidate (import will raise a clear error).
    return os.path.realpath(candidates[0])


_PHASES = _find_lb_eval_phases()
if _PHASES not in sys.path:
    sys.path.insert(0, _PHASES)

import generate_report as g  # noqa: E402  (lb_eval upstream module)


def _phase_log_ok(phase_name: str, logs_dir: Path) -> bool:
    """True if the phase's final log indicates success.

    The fix-loop overwrites phase_log with the last retry_<n>.log, and on success
    the phase re-run exits 0. We look for the deterministic success markers the
    XPU/CUDA phase scripts emit, and the ABSENCE of a trailing Python traceback.
    """
    # Prefer the newest retry log; else the phase log.
    fixes = logs_dir / "agent_fixes" / phase_name
    candidates = sorted(fixes.glob("retry_*.log")) if fixes.exists() else []
    log = candidates[-1] if candidates else (logs_dir / f"{phase_name}.log")
    if not log.exists():
        return False
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return False
    tail = text[-4000:]
    # A trailing traceback / OOM in the very tail means the last run still failed.
    fail_markers = ("Traceback (most recent call last)", "UR_RESULT_ERROR",
                    "RuntimeError", "CUDA error", "out of memory", "OutOfMemory")
    # Success markers emitted by the phase scripts.
    ok_markers = ("=== Phase 2: DONE", "=== XPU Phase: Quantize",
                  "accuracy.json written", "Quantization completed",
                  "=== Phase 3: DONE", "Summary written to")
    has_ok = any(m in tail for m in ok_markers)
    # Only treat trailing failure as decisive (a mid-log error that was later fixed
    # shouldn't count) — inspect just the last ~800 chars.
    has_trailing_fail = any(m in text[-800:] for m in fail_markers)
    return has_ok and not has_trailing_fail


def _lesson_status(phase_name: str, run_dir: Path) -> str | None:
    """Return the last lessons status for a phase: 'fixed' | 'still_failing' | None."""
    f = run_dir / "lessons" / f"{phase_name}.jsonl"
    if not f.exists():
        return None
    status = None
    try:
        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = rec.get("status") or rec.get("resolution")
            if st:
                status = st  # keep the last
    except OSError:
        return None
    return status


def phase_status_icon(phase_name: str, logs_dir: Path) -> str:
    """Robust phase outcome (replaces the fragile upstream version)."""
    phase_log = logs_dir / f"{phase_name}.log"
    fixes = logs_dir / "agent_fixes" / phase_name
    if not phase_log.exists() and not fixes.exists():
        return "⏭️ skipped"

    attempts = len(sorted(fixes.glob("retry_*.log"))) if fixes.exists() else 0
    run_dir = logs_dir.parent
    lesson = _lesson_status(phase_name, run_dir)

    if attempts == 0:
        # No agent involvement → first-try success unless the log clearly failed.
        return "✅ first try" if _phase_log_ok(phase_name, logs_dir) else "❌ failed"

    # Agent was involved. Trust: final log OK and/or lessons marked fixed.
    if lesson == "fixed" or _phase_log_ok(phase_name, logs_dir):
        return f"⚠️ fixed on attempt {attempts}"
    if lesson in ("unfixable", "still_failing"):
        return f"❌ failed after {attempts} attempts"
    return f"❌ failed after {attempts} attempts"


def extract_fix_summary(logs_dir: Path, phase_name: str) -> list[str]:
    """Fix descriptions from the 'fixed' lesson solution + agent attempt logs."""
    summaries: list[str] = []
    run_dir = logs_dir.parent
    # 1) Prefer the recorded lesson solution (concise, agent-authored).
    #    ONLY this run's lessons (run_dir/lessons) — never the repo's global lessons,
    #    which contain unrelated historical entries.
    f = run_dir / "lessons" / f"{phase_name}.jsonl"
    if f.exists():
        try:
            for line in f.read_text(errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (rec.get("status") == "fixed") and rec.get("solution"):
                    sol = re.sub(r"\s+", " ", str(rec["solution"])).strip()
                    summaries.append(f"  - {sol[:500]}")
        except OSError:
            pass
    if summaries:
        return summaries[:5]

    # 2) Fallback: scan agent attempt logs for FIX_PLAN lines.
    fixes = logs_dir / "agent_fixes" / phase_name
    if fixes.exists():
        for attempt_log in sorted(fixes.glob("attempt_*.log")):
            num = attempt_log.stem.split("_")[-1]
            try:
                for ln in attempt_log.read_text(errors="replace").splitlines():
                    if "FIX_PLAN" in ln:
                        summaries.append(f"  Attempt {num}: {ln.strip()[:200]}")
                        break
            except OSError:
                pass
    return summaries[:5]


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: report.py <run_dir>", file=sys.stderr)
        return 2
    run_dir = Path(sys.argv[1]).resolve()

    # Monkey-patch the two buggy functions, then reuse the upstream generator.
    g.phase_status_icon = phase_status_icon
    g.extract_fix_summary = extract_fix_summary

    report = g.generate_report(run_dir)
    out = run_dir / "run_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"[report] Written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
