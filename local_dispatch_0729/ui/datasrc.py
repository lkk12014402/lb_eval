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
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # local_dispatch/
_ROOT = os.path.dirname(_PARENT)          # new_commit/
for p in (_PARENT, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# Local-dispatch results live in an independent Hugging Face dataset. The UI
# downloads only aggregate/status JSON files into a small local cache. Set
# LB_EVAL_UI_REPO to bypass the dataset and read any local results/status tree.
RESULTS_DATASET = os.environ.get("LOCAL_RESULTS_DATASET", "lvkaokao/lb_local")
RESULTS_CACHE = os.environ.get(
    "LOCAL_RESULTS_CACHE",
    os.path.join(_HERE, "data", "lb_local"),
)
LB_EVAL_REPO = os.environ.get("LB_EVAL_UI_REPO", RESULTS_CACHE)
SYNC_INTERVAL = int(os.environ.get("LOCAL_RESULTS_SYNC_INTERVAL", "60"))
_last_sync = 0.0


def _hf_token() -> str:
    return (
        os.environ.get("LOCAL_RESULTS_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HF_TOKENS", "").split(",")[0].strip()
    )


def sync_results_dataset(force: bool = False) -> str:
    """Refresh the lightweight results/status cache from the HF dataset.

    Returns an empty string on success, otherwise a warning. Existing cache data
    remains usable when the network is unavailable.
    """
    global _last_sync
    if os.environ.get("LB_EVAL_UI_REPO"):
        return ""  # explicit local source: never touch the network
    now = time.time()
    if not force and now - _last_sync < SYNC_INTERVAL:
        return ""
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(RESULTS_CACHE, exist_ok=True)
        snapshot_download(
            repo_id=RESULTS_DATASET,
            repo_type="dataset",
            token=_hf_token() or None,
            local_dir=RESULTS_CACHE,
            allow_patterns=[
                "results/*/*/results_*.json",
                "status/**/*.json",
            ],
        )
        _last_sync = now
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"Dataset sync failed; using local cache: {type(exc).__name__}: {exc}"


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
    if repo is None:
        sync_results_dataset()
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
    if repo is None:
        sync_results_dataset()
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
    rows: list[dict] = []
    error_message = ""
    try:
        import gpu_reserve as gr
        servers = gr.api("/api/servers")
    except SystemExit as e:
        servers = []
        error_message = str(e)
    except Exception as e:  # noqa: BLE001
        servers = []
        error_message = f"{type(e).__name__}: {e}"
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
            "access": "reservation API / direct",
        })
    try:
        from machine_profiles import ui_rows
        rows.extend(ui_rows())
    except Exception as exc:  # noqa: BLE001
        profile_error = f"Static machine profiles failed: {type(exc).__name__}: {exc}"
        error_message = "; ".join(x for x in (error_message, profile_error) if x)
    return rows, error_message
