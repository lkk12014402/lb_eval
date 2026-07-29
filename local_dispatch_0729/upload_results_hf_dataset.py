#!/usr/bin/env python3
"""Upload one local-dispatch run to a Hugging Face dataset repository.

This is intentionally independent of lb_eval's GitHub uploader. It stores:

  results/<org>/<artifact>/run_<run_id>/...   run artifacts (no model weights)
  results/<org>/<artifact>/results_<run_id>.json
  status/<org>/<request_filename>.json

The quantized model is uploaded separately by lb_eval's upload_model_hf.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


TEXT_SUFFIXES = {
    ".json", ".jsonl", ".log", ".md", ".txt", ".sh", ".py", ".yaml", ".yml", ".patch", ".diff",
}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"),
    re.compile(r"(?i)(x-access-token:)[^@\s/]+"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "unknown"


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def sanitize_text(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + "***REDACTED***", text)
        else:
            text = pattern.sub("***REDACTED***", text)
    return text


def copy_artifacts(src: Path, dst: Path) -> list[str]:
    """Copy run artifacts, excluding quantized model weights and caches."""
    copied: list[str] = []
    excluded_dirs = {"quantized_model", "__pycache__", ".git", ".cache"}
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        for name in files:
            source = root_path / name
            rel = rel_root / name
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() in TEXT_SUFFIXES:
                try:
                    text = source.read_text(encoding="utf-8", errors="replace")
                    target.write_text(sanitize_text(text), encoding="utf-8")
                except OSError:
                    continue
            else:
                try:
                    shutil.copy2(source, target)
                except OSError:
                    continue
            copied.append(str(rel))
    return copied


def derive_status(quant: dict, accuracy: dict, pipeline_rc: int) -> str:
    if pipeline_rc != 0 or quant.get("status") == "failed":
        return "Failed"
    if accuracy.get("status") == "failed":
        return "Eval Failed"
    tasks = accuracy.get("tasks")
    if isinstance(tasks, dict):
        for value in tasks.values():
            acc = value.get("accuracy") if isinstance(value, dict) else value
            try:
                if acc is not None and float(acc) == 0.0:
                    return "Eval Failed"
            except (TypeError, ValueError):
                pass
    if quant.get("status") == "success" and accuracy.get("status") == "success":
        return "Finished"
    return "Partial"


def upload_with_retry(
    api: HfApi,
    staging: Path,
    repo_id: str,
    commit_message: str,
    retries: int = 5,
) -> None:
    for attempt in range(1, retries + 1):
        try:
            api.upload_folder(
                folder_path=str(staging),
                path_in_repo="",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def main() -> int:
    p = argparse.ArgumentParser(description="Upload local-dispatch results to a HF dataset")
    p.add_argument("run_dir")
    p.add_argument("--dataset", default=os.environ.get("LOCAL_RESULTS_DATASET", "lvkaokao/lb_local"))
    p.add_argument("--token", default=os.environ.get("LOCAL_RESULTS_HF_TOKEN", ""))
    p.add_argument("--run-id", default=os.environ.get("LOCAL_RUN_ID", ""))
    p.add_argument("--pipeline-rc", type=int, default=0)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"[local-dataset] run directory not found: {run_dir}")
        return 2

    token = args.token or os.environ.get("HF_TOKEN", "")
    if not token:
        token = (os.environ.get("HF_TOKENS", "").split(",")[0]).strip()
    if not token:
        print("[local-dataset] no token: set LOCAL_RESULTS_HF_TOKEN or HF_TOKENS")
        return 2

    request = load_json(run_dir / "request.json")
    quant = load_json(run_dir / "quant_summary.json")
    accuracy = load_json(run_dir / "accuracy.json")
    model_id = str(request.get("model") or quant.get("model_id") or "unknown/unknown")
    org = safe_name(model_id.split("/", 1)[0] if "/" in model_id else "local")
    artifact = safe_name(run_dir.name)
    run_id = safe_name(args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    request_filename = safe_name(str(request.get("request_filename") or f"{artifact}.json"))
    if not request_filename.endswith(".json"):
        request_filename += ".json"
    status = derive_status(quant, accuracy, args.pipeline_rc)

    with tempfile.TemporaryDirectory(prefix="lb-local-upload-") as temp:
        staging = Path(temp)
        result_root = staging / "results" / org / artifact
        staged_run = result_root / f"run_{run_id}"
        copied = copy_artifacts(run_dir, staged_run)

        aggregate = {
            "status": status,
            "pipeline": "local_dispatch",
            "model_id": model_id,
            "artifact_name": artifact,
            "request_filename": request_filename,
            "generated_at": utc_now(),
            "run_id": run_id,
            "run_dir": str(staged_run.relative_to(staging)),
            "quant_summary": quant or None,
            "accuracy": accuracy or None,
            "copied_files": copied,
        }
        result_root.mkdir(parents=True, exist_ok=True)
        (result_root / f"results_{run_id}.json").write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        status_data = dict(request)
        status_data["status"] = status
        status_data["request_filename"] = request_filename
        status_data["local_run_id"] = run_id
        status_path = staging / "status" / org / request_filename
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(status_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.dataset,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        upload_with_retry(
            api,
            staging,
            args.dataset,
            f"Add local dispatch results for {artifact} ({run_id})",
        )

    print(f"[local-dataset] uploaded: https://huggingface.co/datasets/{args.dataset}")
    print(f"[local-dataset] result path: results/{org}/{artifact}/run_{run_id}")
    print(f"[local-dataset] status: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
