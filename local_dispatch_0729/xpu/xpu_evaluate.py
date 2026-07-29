#!/usr/bin/env python3
"""Standalone XPU evaluation via lm-eval + vLLM XPU backend.

Writes accuracy.json in the same shape lb_eval's evaluate.sh produces, so the
shared dataset uploader and report generator work unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[xpu-eval] {msg}", flush=True)


def build_model_args(model_path: str, num_gpus: int, max_model_len: int,
                     gpu_mem_util: float = 0.9, dtype: str = "bfloat16") -> str:
    """Build lm-eval vLLM model_args for XPU.

    Mirrors the validated multi-XPU recipe (see eval_xpu.sh): tensor parallelism
    across the reserved cards, enforce_eager (XPU has no CUDA graphs), bfloat16,
    and batching knobs. Env overrides: VLLM_MAX_MODEL_LEN, VLLM_MAX_NUM_BATCHED_TOKENS,
    VLLM_MAX_NUM_SEQS, VLLM_MAX_GEN_TOKS, VLLM_GPU_MEM_UTIL, VLLM_DTYPE.
    """
    import os
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", max_model_len))
    max_num_batched = int(os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS", "32768"))
    max_num_seqs = int(os.environ.get("VLLM_MAX_NUM_SEQS", "128"))
    max_gen_toks = int(os.environ.get("VLLM_MAX_GEN_TOKS", "2048"))
    gpu_mem_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", gpu_mem_util))
    dtype = os.environ.get("VLLM_DTYPE", dtype)
    parts = [
        f"pretrained={model_path}",
        f"tensor_parallel_size={num_gpus}",
        f"max_model_len={max_model_len}",
        f"max_num_batched_tokens={max_num_batched}",
        f"max_num_seqs={max_num_seqs}",
        f"max_gen_toks={max_gen_toks}",
        f"dtype={dtype}",
        "trust_remote_code=True",
        "enforce_eager=True",       # XPU: CUDA graphs unsupported
        "add_bos_token=True",
        "enable_prefix_caching=False",
        f"gpu_memory_utilization={gpu_mem_util}",
    ]
    return ",".join(parts)


def parse_results(output_dir: Path, model_path: str, tasks: str, num_gpus: str) -> dict:
    results_files = sorted(output_dir.rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
    if not results_files:
        results_files = sorted(output_dir.rglob("results.json"), key=lambda p: p.stat().st_mtime)
    if not results_files:
        return {
            "status": "failed",
            "errors": ["No results JSON found in lm_eval output directory"],
            "model_path": model_path,
            "tasks": {},
        }
    with results_files[-1].open() as f:
        lm_results = json.load(f)
    task_scores = {}
    for name, data in lm_results.get("results", {}).items():
        if isinstance(data, dict):
            acc = data.get("acc,none") or data.get("acc_norm,none") or data.get("acc")
            if acc is not None:
                task_scores[name] = {"accuracy": round(float(acc), 6)}
    has_zero = any(v.get("accuracy", -1) == 0.0 for v in task_scores.values())
    accuracy = {
        "status": "failed" if has_zero or not task_scores else "success",
        "model_id": model_path.rsplit("/", 1)[-1] if "/" in model_path else model_path,
        "model_path": model_path,
        "eval_framework": "lm_eval (vllm-xpu)",
        "num_gpus": num_gpus,
        "eval_num_gpus": num_gpus,
        "hardware": "Intel Arc Pro B60",
        "tasks": task_scores,
        "lm_eval_output_dir": str(output_dir),
        "errors": [],
    }
    if has_zero:
        zero = [k for k, v in task_scores.items() if v.get("accuracy") == 0.0]
        accuracy["errors"] = [f"Zero accuracy on tasks: {zero}"]
    return accuracy


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone XPU evaluation (lm-eval + vLLM XPU)")
    p.add_argument("--model-path", required=True)
    p.add_argument("--tasks", default="piqa,mmlu,hellaswag")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_args = build_model_args(args.model_path, args.num_gpus, args.max_model_len)

    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--batch_size", str(args.batch_size),
        "--output_path", str(output_dir),
        "--log_samples",
        "--seed", "42",
    ]
    _log(f"Running: {' '.join(cmd)}")
    log_path = output_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
        rc = proc.returncode

    accuracy = parse_results(output_dir, args.model_path, args.tasks, str(args.num_gpus))
    if rc != 0 and accuracy["status"] != "success":
        accuracy.setdefault("errors", []).append(f"lm_eval exited {rc}")
    accuracy_path = output_dir.parent / "accuracy.json"
    with accuracy_path.open("w") as f:
        json.dump(accuracy, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _log(f"accuracy.json written to {accuracy_path} (status={accuracy['status']})")
    for task, data in accuracy.get("tasks", {}).items():
        _log(f"  {task}: {data.get('accuracy', 'N/A')}")
    return 0 if accuracy["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
