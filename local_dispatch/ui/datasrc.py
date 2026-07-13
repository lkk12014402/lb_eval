#!/usr/bin/env python3
"""Data source for the local-dispatch UI.

Two kinds of data, mirroring the leaderboard:
  1. Results / queue read from an lb_eval repo clone (the same repo local_dispatch
     pushes to): ``results/**/results_*.json`` aggregates + ``status/**/*.json`` queue.
  2. Live GPU machine availability from the reservation API (``gpu_reserve``).

All reads are best-effort — a missing repo just yields empty tables.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # local_dispatch/
_ROOT = os.path.dirname(_PARENT)          # new_commit/
for p in (_PARENT, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Where the lb_eval repo (with results/ + status/) lives locally. Point this at a
# clone of the repo local_dispatch pushes to.
LB_EVAL_REPO = os.environ.get(
    "LB_EVAL_UI_REPO",
    os.path.join(_ROOT, "lb_eval"),
)


def _mean_accuracy(accuracy: dict | None) -> float | None:
    if not isinstance(accuracy, dict):
        return None
    tasks = accuracy.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        return None
    vals = []
    for v in tasks.values():
        a = v.get("accuracy") if isinstance(v, dict) else v
        try:
            vals.append(float(a))
        except (TypeError, ValueError):
            pass
    return round(sum(vals) / len(vals), 4) if vals else None


def load_results(repo: str | None = None) -> list[dict]:
    """Scan results/**/results_*.json aggregates → display rows (finished/failed runs)."""
    repo = repo or LB_EVAL_REPO
    results_dir = os.path.join(repo, "results")
    rows: list[dict] = []
    if not os.path.isdir(results_dir):
        return rows
    for root, _, files in os.walk(results_dir):
        for f in files:
            if not (f.startswith("results_") and f.endswith(".json")):
                continue
            try:
                with open(os.path.join(root, f)) as fh:
                    d = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if "run_dir" not in d or "copied_files" not in d:
                continue  # only auto_pipeline aggregates
            qs = d.get("quant_summary") or {}
            acc = d.get("accuracy") or {}
            rows.append({
                "model": d.get("model_id", ""),
                "artifact": d.get("artifact_name", ""),
                "scheme": qs.get("scheme", ""),
                "status": d.get("status", ""),
                "avg_acc": _mean_accuracy(acc),
                "hf_repo": qs.get("hf_repo", ""),
                "generated_at": d.get("generated_at", ""),
                "run_dir": d.get("run_dir", ""),
            })
    rows.sort(key=lambda r: r.get("generated_at", ""), reverse=True)
    return rows


def load_queue(repo: str | None = None) -> list[dict]:
    """Scan status/**/*.json → queue rows (submitted requests + their status)."""
    repo = repo or LB_EVAL_REPO
    status_dir = os.path.join(repo, "status")
    rows: list[dict] = []
    if not os.path.isdir(status_dir):
        return rows
    for root, _, files in os.walk(status_dir):
        for f in files:
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(root, f)) as fh:
                    d = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            rows.append({
                "model": d.get("model", ""),
                "scheme": d.get("quant_scheme", d.get("scheme", "")),
                "method": d.get("method", ""),
                "status": d.get("status", ""),
                "submitted_by": d.get("submitted_by", ""),
                "file": f,
            })
    rows.sort(key=lambda r: r.get("status", ""))
    return rows


def load_gpu_machines() -> tuple[list[dict], str]:
    """Live machine/GPU availability from the reservation API. Returns (rows, error)."""
    try:
        import gpu_reserve as gr
        servers = gr.api("/api/servers")
    except SystemExit as e:
        return [], str(e)
    except Exception as e:  # noqa: BLE001
        return [], f"{type(e).__name__}: {e}"
    rows: list[dict] = []
    for s in servers:
        gpus = s.get("gpus", [])
        model = gpus[0]["model"] if gpus else "-"
        avail = sum(1 for g in gpus if g.get("status") == "available")
        rows.append({
            "server": s.get("name", ""),
            "host": s.get("host", ""),
            "gpu": model,
            "total": len(gpus),
            "available": avail,
            "busy": len(gpus) - avail,
        })
    return rows, ""
