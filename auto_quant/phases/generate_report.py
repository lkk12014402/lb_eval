#!/usr/bin/env python3
"""Generate a deterministic run_report.md summarizing the pipeline execution.

Reads structured outputs (quant_summary.json, accuracy.json, logs/) and
produces a human-readable markdown report. No LLM calls — 100% reliable.

Usage:
    python generate_report.py <run_output_dir> [--output run_report.md]

The script is fault-tolerant: missing files result in "N/A" sections, never crashes.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json_safe(path: Path) -> dict | None:
    """Load JSON file, return None on any error."""
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_duration(seconds) -> str:
    """Format seconds into human readable string."""
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return "N/A"
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60}s"
    else:
        return f"{s // 3600}h {(s % 3600) // 60}m"


def phase_status_icon(phase_name: str, logs_dir: Path) -> str:
    """Determine phase outcome from logs."""
    agent_fixes_dir = logs_dir / "agent_fixes" / phase_name

    # No agent fixes dir → check if phase log exists and is non-empty
    phase_log = logs_dir / f"{phase_name}.log"
    if not phase_log.exists():
        return "⏭️ skipped"

    if not agent_fixes_dir.exists():
        return "✅ first try"

    # Count retry attempts
    retries = sorted(agent_fixes_dir.glob("retry_*.log"))
    attempts = len(retries)

    if attempts == 0:
        return "✅ first try"

    # Check if last retry succeeded (the phase ultimately passed)
    # We check by looking at whether the next phase ran
    # Simpler: check for "fixed" lessons or just count attempts
    prompts = sorted(agent_fixes_dir.glob("prompt_*.log"))

    # Look for success indicator in the last retry
    if retries:
        last_retry = retries[-1]
        try:
            content = last_retry.read_text(errors="replace")[-200:]
            if "DONE" in content or "Phase" in content and "success" in content.lower():
                return f"⚠️ fixed on attempt {attempts}"
        except Exception:
            pass

    return f"❌ failed after {attempts} attempts"


def extract_fix_summary(logs_dir: Path, phase_name: str) -> list[str]:
    """Extract brief fix descriptions from agent fix logs."""
    agent_fixes_dir = logs_dir / "agent_fixes" / phase_name
    if not agent_fixes_dir.exists():
        return []

    summaries = []
    for prompt_file in sorted(agent_fixes_dir.glob("prompt_*.txt")):
        attempt_num = prompt_file.stem.split("_")[-1]
        # Try to find the FIX_PLAN from agent output
        attempt_log = agent_fixes_dir / f"attempt_{attempt_num}.log"
        if attempt_log.exists():
            try:
                content = attempt_log.read_text(errors="replace")
                # Look for FIX_PLAN section
                for line in content.splitlines():
                    if "FIX_PLAN" in line or "pip install" in line or "fix" in line.lower():
                        clean = line.strip()[:120]
                        if clean:
                            summaries.append(f"  Attempt {attempt_num}: {clean}")
                            break
            except Exception:
                pass

    return summaries[:5]  # Max 5 lines


def generate_report(run_dir: Path) -> str:
    """Generate the full markdown report."""
    lines = []

    # Load data
    quant_summary = load_json_safe(run_dir / "quant_summary.json")
    accuracy = load_json_safe(run_dir / "accuracy.json")
    request = load_json_safe(run_dir / "request.json")
    logs_dir = run_dir / "logs"

    # Derive basic info
    model_id = (quant_summary or {}).get("model_id") or \
               (request or {}).get("model") or \
               run_dir.name
    scheme = (quant_summary or {}).get("scheme") or \
             (request or {}).get("scheme", "W4A16")
    method = "RTN" if (quant_summary or {}).get("iters", 0) == 0 else "TUNING"
    ar_scheme = (quant_summary or {}).get("ar_scheme", scheme)
    duration = (quant_summary or {}).get("duration_seconds")
    arch = (quant_summary or {}).get("architecture", "N/A")
    model_type = (quant_summary or {}).get("model_type", "N/A")
    is_moe = (quant_summary or {}).get("is_moe", False)
    ignore_layers = (quant_summary or {}).get("ignore_layers", "lm_head")
    export_fmt = (quant_summary or {}).get("export_format", "auto_round")

    # Overall status
    acc_status = (accuracy or {}).get("status", "missing")
    qs_status = (quant_summary or {}).get("status", "missing")
    if qs_status == "success" and acc_status == "success":
        overall = "Finished ✅"
    elif qs_status == "failed":
        overall = "Quant Failed ❌"
    elif acc_status == "failed":
        overall = "Eval Failed ❌"
    else:
        overall = "Partial ⚠️"

    # Header
    lines.append(f"# Pipeline Report: {model_id}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # Status summary
    lines.append("## Status")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| Overall | {overall} |")
    lines.append(f"| Model | `{model_id}` |")
    lines.append(f"| Architecture | {arch} ({model_type}, {'MoE' if is_moe else 'Dense'}) |")
    lines.append(f"| Scheme | {scheme} → `{ar_scheme}` |")
    lines.append(f"| Method | {method} (iters={(quant_summary or {}).get('iters', 'N/A')}) |")
    lines.append(f"| Export Format | {export_fmt} |")
    lines.append(f"| Ignore Layers | `{ignore_layers}` |")
    lines.append(f"| Quant Duration | {format_duration(duration)} |")
    lines.append("")

    # Agent-synthesized narrative (resolution summary), if present — placed near the
    # top for high visibility. This is the G6 narrative layer on top of the facts.
    summary_md = run_dir / "resolution_summary.md"
    if summary_md.exists():
        try:
            text = summary_md.read_text().strip()
        except Exception:
            text = ""
        if text:
            lines.append("## Summary & Root Cause")
            lines.append("")
            lines.append(text)
            lines.append("")

    # Evaluation results
    lines.append("## Evaluation Results")
    lines.append("")
    if accuracy and isinstance(accuracy.get("tasks"), dict):
        tasks = accuracy["tasks"]
        # Separate top-level tasks from subtasks
        top_tasks = {k: v for k, v in tasks.items() if not k.startswith("mmlu_")}
        sub_tasks = {k: v for k, v in tasks.items() if k.startswith("mmlu_") and not k.startswith("mmlu_"*2)}

        lines.append("| Task | Accuracy |")
        lines.append("|------|----------|")
        for task_name, val in sorted(top_tasks.items()):
            acc_val = val if not isinstance(val, dict) else val.get("accuracy", val)
            try:
                lines.append(f"| {task_name} | {float(acc_val):.4f} |")
            except (TypeError, ValueError):
                lines.append(f"| {task_name} | {acc_val} |")

        # MMLU subcategory summary (if exists)
        mmlu_cats = {k: v for k, v in tasks.items()
                     if k.startswith("mmlu_") and k.count("_") == 1}
        if mmlu_cats:
            lines.append("")
            lines.append("<details><summary>MMLU Subcategories</summary>")
            lines.append("")
            lines.append("| Category | Accuracy |")
            lines.append("|----------|----------|")
            for cat, val in sorted(mmlu_cats.items()):
                acc_val = val if not isinstance(val, dict) else val.get("accuracy", val)
                try:
                    lines.append(f"| {cat} | {float(acc_val):.4f} |")
                except (TypeError, ValueError):
                    lines.append(f"| {cat} | {acc_val} |")
            lines.append("")
            lines.append("</details>")
    else:
        lines.append("*No evaluation results available.*")
    lines.append("")

    # Phase execution summary
    lines.append("## Phase Execution")
    lines.append("")
    phases = ["setup_env", "quantize", "evaluate"]
    lines.append("| Phase | Result |")
    lines.append("|-------|--------|")
    for phase in phases:
        status = phase_status_icon(phase, logs_dir) if logs_dir.exists() else "N/A"
        lines.append(f"| {phase} | {status} |")
    lines.append("")

    # Agent fix details (if any)
    if logs_dir.exists() and (logs_dir / "agent_fixes").exists():
        has_fixes = False
        fix_lines = []
        for phase in phases:
            fixes = extract_fix_summary(logs_dir, phase)
            if fixes:
                has_fixes = True
                fix_lines.append(f"### {phase}")
                fix_lines.append("")
                fix_lines.extend(fixes)
                fix_lines.append("")

        if has_fixes:
            lines.append("## Agent Fix Log")
            lines.append("")
            lines.extend(fix_lines)

    # Upload status
    lines.append("## Upload")
    lines.append("")
    hf_log = logs_dir / "upload_hf.log" if logs_dir.exists() else None
    gh_log = logs_dir / "upload_github.log" if logs_dir.exists() else None

    hf_status = "N/A"
    if hf_log and hf_log.exists():
        content = hf_log.read_text(errors="replace")
        if "Upload complete" in content or "repo_url" in content:
            # Try to extract repo URL
            for line in content.splitlines():
                if "huggingface.co" in line:
                    hf_status = f"✅ {line.strip()[:100]}"
                    break
            else:
                hf_status = "✅ uploaded"
        elif "ERROR" in content or "failed" in content.lower():
            hf_status = "❌ failed"
        else:
            hf_status = "⏭️ skipped"

    gh_status = "N/A"
    if gh_log and gh_log.exists():
        content = gh_log.read_text(errors="replace")
        if "git push" in content.lower() and "error" not in content.lower():
            gh_status = "✅ pushed"
        elif "ERROR" in content or "failed" in content.lower():
            gh_status = "❌ failed"
        else:
            gh_status = "⏭️ skipped"

    lines.append(f"- **HuggingFace Hub:** {hf_status}")
    lines.append(f"- **GitHub lb_eval:** {gh_status}")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Report generated by `generate_report.py` from `{run_dir.name}`*")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate pipeline run report")
    parser.add_argument("run_dir", help="Path to the run output directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (default: <run_dir>/run_report.md)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(run_dir)

    output_path = Path(args.output) if args.output else run_dir / "run_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"[report] Written to: {output_path}")


if __name__ == "__main__":
    main()
