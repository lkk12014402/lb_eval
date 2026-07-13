#!/usr/bin/env python3
"""Generate a quantization RECIPE deliverable for a fully-successful run.

A recipe is produced ONLY when quantization AND evaluation both succeeded. It
documents exactly how the model was quantized so the result is reproducible, and
is uploaded to GitHub alongside the results.

Follows the template in ``recipe.md``: a Recipe config table, an Accuracy Result
table, quantization/evaluation code sections, and a summary. Emits into the run dir:

  * ``<run_dir>/recipe.md``    — human-readable recipe deliverable
  * ``<run_dir>/recipe.json``  — machine-readable recipe metadata

Deterministic + fault-tolerant: missing inputs degrade to "N/A", never crash.
An optional agent-written narrative (summary/root-cause) can be layered on top by
passing --narrative <file>.

Usage:
    python generate_recipe.py <run_output_dir> [--narrative FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json_safe(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ── Scheme → recipe-row spec (mirrors recipe.md's Recipe table) ─────────────
# Each entry supplies the deterministic per-scheme rows. Unknown schemes fall
# back to sensible defaults derived from the name.
_SCHEME_SPEC: dict[str, dict] = {
    "MXFP4": {
        "data_type": "MXFP4", "element_format": "E2M1",
        "weight_scheme_line": "Weight – MXFP4, Activation – BF16",
        "block_size": "32", "scale_type": "FP8 E8M0",
        "scale_method": "1D block-wise scaling for MXFP4",
    },
    "MXFP8": {
        "data_type": "MXFP8", "element_format": "E4M3",
        "weight_scheme_line": "Weight – MXFP8, Activation – BF16",
        "block_size": "32", "scale_type": "FP8 E8M0",
        "scale_method": "1D block-wise scaling for MXFP8",
    },
    "NVFP4": {
        "data_type": "NVFP4", "element_format": "E2M1",
        "weight_scheme_line": "Weight – NVFP4, Activation – BF16",
        "block_size": "16", "scale_type": "FP8 E4M3",
        "scale_method": "2-level (per-block E4M3 + per-tensor FP32) for NVFP4",
    },
    "W4A16": {
        "data_type": "INT4 (W4A16)", "element_format": "INT4",
        "weight_scheme_line": "Weight – INT4, Activation – BF16",
        "block_size": "128", "scale_type": "BF16",
        "scale_method": "Group-wise (group_size=128) integer scaling",
    },
    "W8A16": {
        "data_type": "INT8 (W8A16)", "element_format": "INT8",
        "weight_scheme_line": "Weight – INT8, Activation – BF16",
        "block_size": "128", "scale_type": "BF16",
        "scale_method": "Group-wise (group_size=128) integer scaling",
    },
}


def _scheme_spec(scheme: str) -> dict:
    return _SCHEME_SPEC.get(scheme, {
        "data_type": scheme, "element_format": "N/A",
        "weight_scheme_line": f"Weight – {scheme}, Activation – BF16",
        "block_size": "N/A", "scale_type": "N/A", "scale_method": "N/A",
    })


def _quant_algo(method: str, iters, model_free: bool) -> str:
    if model_free:
        return "RTN (model-free, no calibration)"
    if str(method).upper() == "TUNING" or (isinstance(iters, int) and iters > 0):
        return f"AutoRound (sign-SGD, iters={iters})"
    return "RTN"


# ── Accuracy aggregation ────────────────────────────────────────────────────
_HEADLINE_TASKS = ["gsm8k", "mmlu", "piqa", "hellaswag"]


def _acc_of(val) -> float | None:
    if isinstance(val, dict):
        val = val.get("accuracy", val.get("acc"))
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def aggregate_accuracy(accuracy: dict | None) -> dict:
    """Return {task: acc} for headline tasks + 'avg'. Averages mmlu_* subtasks."""
    out: dict[str, float] = {}
    if not accuracy or not isinstance(accuracy.get("tasks"), dict):
        return out
    tasks = accuracy["tasks"]

    # mmlu: average all mmlu_* subtasks
    mmlu_vals = [a for k, v in tasks.items() if k.startswith("mmlu_")
                 and (a := _acc_of(v)) is not None]
    if "mmlu" in tasks and (a := _acc_of(tasks["mmlu"])) is not None:
        out["mmlu"] = a
    elif mmlu_vals:
        out["mmlu"] = sum(mmlu_vals) / len(mmlu_vals)

    for t in ("gsm8k", "piqa", "hellaswag"):
        for k, v in tasks.items():
            if k == t or k.startswith(t):
                a = _acc_of(v)
                if a is not None:
                    out[t] = a
                break

    present = [out[t] for t in _HEADLINE_TASKS if t in out]
    if present:
        out["avg"] = sum(present) / len(present)
    return out


# ── layer_config → mixed-precision notes ────────────────────────────────────
def _mixed_precision_notes(layer_config: str | None) -> list[str]:
    notes: list[str] = []
    if not layer_config:
        return notes
    lc = str(layer_config)
    # Parse relaxed JSON best-effort to name per-module overrides.
    try:
        from auto_round.utils.common import parse_layer_config_arg  # type: ignore
        parsed = parse_layer_config_arg(lc)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for mod, cfg in parsed.items():
            bits = cfg.get("bits") if isinstance(cfg, dict) else None
            dt = cfg.get("data_type") if isinstance(cfg, dict) else None
            desc = f"bits={bits}" if bits is not None else ""
            if dt:
                desc += (", " if desc else "") + f"data_type={dt}"
            notes.append(f"`{mod}`: {desc or lc}")
    else:
        notes.append(f"layer_config: `{lc}`")
    return notes


# ── Render ──────────────────────────────────────────────────────────────────
def build_recipe_md(quant_summary: dict, accuracy: dict | None,
                    narrative: str | None) -> str:
    qs = quant_summary or {}
    scheme = qs.get("scheme") or "N/A"
    spec = _scheme_spec(scheme)
    method = qs.get("method") or "RTN"
    iters = qs.get("iters", 0)
    model_free = bool(qs.get("model_free"))
    export_fmt = qs.get("export_format") or "auto_round"
    ignore_layers = qs.get("ignore_layers") or ""
    layer_config = qs.get("layer_config")
    model_id = qs.get("model_id") or "N/A"
    arch = qs.get("architecture") or qs.get("model_type") or "N/A"

    exclude_fmt = ", ".join(f"`{x.strip()}`" for x in ignore_layers.split(",") if x.strip()) or "—"

    L: list[str] = []
    L.append(f"# {scheme} Recipe — {model_id}")
    L.append("")
    L.append(f"*Architecture: `{arch}`  ·  Export: `{export_fmt}`  ·  "
             f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    L.append("")

    # ── Recipe table ──
    L.append("## Recipe")
    L.append("")
    L.append("| Config/Setting | Value |")
    L.append("|----------|----------|")
    L.append("| Quant approach | PTQ |")
    L.append(f"| Quant algo | {_quant_algo(method, iters, model_free)} |")
    L.append("| Quantized ops | Linear |")
    L.append(f"| Data type | {spec['data_type']} |")
    L.append(f"| Element format | {spec['element_format']} |")
    L.append(f"| Scheme | {spec['weight_scheme_line']} |")
    L.append("| Weight scheme | Static, block-wise |")
    L.append("| Activation scheme | Dynamic, block-wise |")
    L.append(f"| Block size | {spec['block_size']} |")
    L.append(f"| Scale type | {spec['scale_type']} |")
    L.append(f"| Scale method | {spec['scale_method']} |")
    L.append(f"| Method | {method} (iters={iters}, model_free={str(model_free).lower()}) |")
    L.append(f"| Export format | {export_fmt} |")
    L.append(f"| Exclude layers | {exclude_fmt} |")
    L.append("")
    L.append("---")
    L.append("")

    # ── Accuracy Result ──
    L.append("# Accuracy Result")
    L.append("")
    agg = aggregate_accuracy(accuracy)
    if agg:
        cols = [t for t in _HEADLINE_TASKS if t in agg]
        header = "| " + (qs.get("model_type") or "Model") + " | " + " | ".join(cols) + " | avg |"
        sep = "|" + "---|" * (len(cols) + 2)
        row = f"| {method}{' MF' if model_free else ''} ({scheme}) | " + \
              " | ".join(f"{agg[c]:.4f}" for c in cols) + \
              f" | {agg.get('avg', 0):.4f} |"
        L.append(header)
        L.append(sep)
        L.append(row)
    else:
        L.append("*No evaluation results available.*")
    L.append("")

    # ── mixed-precision notes ──
    mp = _mixed_precision_notes(layer_config)
    if mp:
        L.append("> **Note (mixed precision)**")
        L.append(">")
        for n in mp:
            L.append(f"> - {n}")
        L.append("")

    # ── code sections ──
    L.append("# quantization code")
    L.append("")
    L.append("```bash")
    L.append(_reproduce_command(qs))
    L.append("```")
    L.append("")
    L.append("# evaluation code")
    L.append("")
    L.append("```bash")
    L.append(f"# lm_eval backend: {'vllm' if export_fmt == 'llm_compressor' else 'hf'}")
    L.append("bash phases/evaluate.sh <quantized_model_dir>")
    L.append("```")
    L.append("")

    # ── summary ──
    L.append("# summary")
    L.append("")
    if narrative:
        L.append(narrative.strip())
    else:
        L.append(_auto_summary(qs, agg))
    L.append("")
    return "\n".join(L)


def _reproduce_command(qs: dict) -> str:
    parts = ["auto-round", qs.get("model_id", "<model>"),
             f"--scheme {qs.get('scheme', 'W4A16')}"]
    if qs.get("model_free"):
        parts.append("--model_free")
    if qs.get("ignore_layers"):
        parts.append(f"--ignore_layers {qs['ignore_layers']}")
    if qs.get("layer_config"):
        parts.append(f'--layer_config "{qs["layer_config"]}"')
    parts.append(f"--format {qs.get('export_format', 'auto_round')}")
    parts.append("--output_dir ./quantized_model")
    return " \\\n  ".join(parts)


def _auto_summary(qs: dict, agg: dict) -> str:
    scheme = qs.get("scheme", "N/A")
    ratio = ""
    lines = [
        f"Quantized **{qs.get('model_id', 'the model')}** "
        f"(`{qs.get('architecture', 'N/A')}`, {'MoE' if qs.get('is_moe') else 'dense'}) "
        f"to **{scheme}** via {_quant_algo(qs.get('method','RTN'), qs.get('iters',0), bool(qs.get('model_free')))}.",
    ]
    if agg.get("avg") is not None:
        lines.append(f"Average headline accuracy: **{agg['avg']:.4f}**"
                     f" ({', '.join(f'{k}={agg[k]:.3f}' for k in ('gsm8k','mmlu','piqa','hellaswag') if k in agg)}).")
    if qs.get("ignore_layers"):
        lines.append(f"Excluded layers: `{qs['ignore_layers']}`.")
    if qs.get("layer_config"):
        lines.append(f"Mixed precision: `{qs['layer_config']}`.")
    dur = qs.get("duration_seconds")
    if dur:
        lines.append(f"Quantization took {dur:.0f}s. Exported as `{qs.get('export_format','auto_round')}`.")
    return " ".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a quantization recipe deliverable for a successful run")
    ap.add_argument("run_dir")
    ap.add_argument("--narrative", default=None,
                    help="Optional agent-written summary markdown to embed in the summary section")
    ap.add_argument("--output", default="recipe.md")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    quant_summary = load_json_safe(run_dir / "quant_summary.json") or {}
    accuracy = load_json_safe(run_dir / "accuracy.json")

    if not quant_summary:
        print(f"[recipe] no quant_summary.json in {run_dir}; skipping", file=sys.stderr)
        return 0
    # A recipe is a deliverable for a FULLY successful model: quantization AND
    # evaluation both succeeded. Skip otherwise (auto.sh already gates on this, but
    # re-check so the generator is safe to run standalone).
    quant_ok = quant_summary.get("status") in ("success", None) or bool(quant_summary.get("output_files"))
    eval_ok = (accuracy or {}).get("status") == "success"
    if not quant_ok:
        print(f"[recipe] quantization not successful (status={quant_summary.get('status')}); skipping",
              file=sys.stderr)
        return 0
    if not eval_ok:
        print(f"[recipe] evaluation not successful (status={(accuracy or {}).get('status')}); "
              "skipping recipe (recipes are only for quant+eval-successful models)", file=sys.stderr)
        return 0

    narrative = None
    if args.narrative and Path(args.narrative).is_file():
        narrative = Path(args.narrative).read_text()

    recipe_md = build_recipe_md(quant_summary, accuracy, narrative)

    # Recipe deliverables written INTO the run dir → uploaded to GitHub with the results.
    out_path = run_dir / args.output
    out_path.write_text(recipe_md)
    print(f"[recipe] wrote {out_path}")

    agg = aggregate_accuracy(accuracy)
    recipe_json = {
        "model_id": quant_summary.get("model_id"),
        "architecture": quant_summary.get("architecture"),
        "model_type": quant_summary.get("model_type"),
        "is_moe": quant_summary.get("is_moe"),
        "scheme": quant_summary.get("scheme"),
        "method": quant_summary.get("method"),
        "iters": quant_summary.get("iters"),
        "model_free": bool(quant_summary.get("model_free")),
        "export_format": quant_summary.get("export_format"),
        "ignore_layers": quant_summary.get("ignore_layers"),
        "layer_config": quant_summary.get("layer_config"),
        "accuracy": agg,
        "duration_seconds": quant_summary.get("duration_seconds"),
        "hf_repo": quant_summary.get("hf_repo"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "recipe.json").write_text(json.dumps(recipe_json, indent=2))
    print(f"[recipe] wrote {run_dir / 'recipe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
