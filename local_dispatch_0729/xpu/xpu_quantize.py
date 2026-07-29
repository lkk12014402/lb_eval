#!/usr/bin/env python3
"""Standalone XPU quantization for Intel Arc Pro B60.

Independent of lb_eval's CUDA phases (which assert torch.cuda). Uses AutoRound
with device_map="xpu" and ZE_AFFINITY_MASK-based device selection. Writes the
same quant_summary.json shape the shared uploaders expect.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


# scheme → AutoRound scheme string (mirrors lb_eval phases/quantize.py table)
SCHEME_MAP = {
    "W4A16": "W4A16",
    "W8A16": "W8A16",
    "MXFP4": "MXFP4",
    "MXFP8": "MXFP8",
    "NVFP4": "NVFP4",
}
# For the auto_round export format, MX schemes use the RCEIL rounding variant.
SCHEME_MAP_AUTOROUND_EXPORT = {
    "MXFP4": "MXFP4",
    "MXFP8": "MXFP8",
}
DENSE_IGNORE_LAYERS = {
    "W4A16": "lm_head",
    "MXFP4": "lm_head,self_attn",
    "NVFP4": "lm_head,self_attn",
    "MXFP8": "lm_head",
    "W8A16": "lm_head",
}
MOE_IGNORE_LAYERS = {
    "W4A16": "lm_head,mlp.gate",
    "MXFP4": "lm_head,self_attn,mlp.gate",
    "NVFP4": "lm_head,self_attn,mlp.gate",
    "MXFP8": "lm_head,mlp.gate",
    "W8A16": "lm_head,mlp.gate",
}


def _log(msg: str) -> None:
    print(f"[xpu-quant] {msg}", flush=True)


def is_moe_config(cfg) -> bool:
    keys = ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size")
    return any(getattr(cfg, k, None) for k in keys)


def assert_xpu() -> int:
    import torch

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError(
            "XPU is not available inside the container. Ensure the image is the "
            "Intel XPU build and the run mounts /dev/dri with --privileged."
        )
    count = torch.xpu.device_count()
    _log(f"XPU available: {count} device(s)")
    return count


def run(args) -> int:
    import torch
    from auto_round import AutoRound
    from transformers import AutoConfig, AutoTokenizer

    n_xpu = assert_xpu()
    model_free = bool(args.model_free)

    export_format = args.export_format
    if model_free and args.scheme in ("MXFP4", "MXFP8") and export_format != "llm_compressor":
        _log(f"Model-free {args.scheme} requires llm_compressor export; overriding.")
        export_format = "llm_compressor"

    if export_format == "auto_round" and args.scheme in SCHEME_MAP_AUTOROUND_EXPORT:
        ar_scheme = SCHEME_MAP_AUTOROUND_EXPORT[args.scheme]
    else:
        ar_scheme = SCHEME_MAP.get(args.scheme, args.scheme)

    # Device map: single XPU → "xpu" (index chosen via ZE_AFFINITY_MASK); the
    # container is launched with exactly the reserved cards visible, re-indexed 0..N-1.
    device_map = "xpu"
    _log(f"Model: {args.model}")
    _log(f"Scheme: {args.scheme} → AutoRound '{ar_scheme}', export={export_format}")
    _log(f"Device map: {device_map} (visible XPU={n_xpu})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    moe = is_moe_config(cfg)
    arch_name = (getattr(cfg, "architectures", None) or ["unknown"])[0]
    model_type = getattr(cfg, "model_type", "unknown")
    _log(f"Architecture: {arch_name} (model_type={model_type}, moe={moe})")

    custom_ignore = (args.ignore_layers or "").strip()
    if custom_ignore:
        ignore_layers = custom_ignore
    else:
        table = MOE_IGNORE_LAYERS if moe else DENSE_IGNORE_LAYERS
        ignore_layers = table.get(args.scheme, "lm_head")
    _log(f"Ignore layers: {ignore_layers}")

    ar_kwargs = {
        "model": args.model,
        "tokenizer": tokenizer,
        "scheme": ar_scheme,
        "iters": args.iters,
        "low_gpu_mem_usage": True,
        "device_map": device_map,
    }
    if model_free:
        ar_kwargs["model_free"] = True
    if ignore_layers:
        ar_kwargs["ignore_layers"] = ignore_layers
    if args.iters > 0:
        ar_kwargs["seqlen"] = args.seqlen
        ar_kwargs["nsamples"] = args.nsamples

    os.makedirs(args.output_dir, exist_ok=True)
    autoround = AutoRound(**ar_kwargs)

    start = time.time()
    if model_free:
        _log(f"Model-free quantize + export ({export_format})...")
        autoround.quantize_and_save(output_dir=args.output_dir, format=export_format)
    else:
        _log("Quantizing...")
        autoround.quantize()
        _log(f"Saving ({export_format})...")
        autoround.save_quantized(output_dir=args.output_dir, format=export_format)
    duration = time.time() - start
    _log(f"Quantization completed in {duration:.1f}s")

    output_files = sorted(
        os.path.join(args.output_dir, f)
        for f in os.listdir(args.output_dir)
        if os.path.isfile(os.path.join(args.output_dir, f))
    )
    quantized_size_mb = None
    try:
        b = sum(os.path.getsize(p) for p in output_files if p.endswith((".safetensors", ".bin")))
        if b > 0:
            quantized_size_mb = round(b / (1024 * 1024), 1)
    except OSError:
        pass

    method = "MODEL_FREE" if model_free else ("RTN" if args.iters == 0 else "TUNING")
    summary = {
        "status": "success",
        "model_id": args.model,
        "architecture": arch_name,
        "model_type": model_type,
        "is_moe": moe,
        "scheme": args.scheme,
        "method": method,
        "ar_scheme": ar_scheme,
        "iters": args.iters,
        "export_format": export_format,
        "ignore_layers": ignore_layers,
        "model_free": model_free,
        "duration_seconds": round(duration, 1),
        "output_dir": args.output_dir,
        "device": "xpu",
        "device_map": "xpu",
        "num_gpus": str(args.num_gpus),
        "hardware": "Intel Arc Pro B60",
        "output_files": output_files,
        "quantized_size_mb": quantized_size_mb,
        "errors": [],
        "solutions": [],
    }
    summary_path = os.path.normpath(os.path.join(args.output_dir, "..", "quant_summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    _log(f"Summary written to {summary_path}")
    return 0


def _write_failed_summary(output_dir: str, model: str, scheme: str, err: str) -> None:
    summary_path = os.path.normpath(os.path.join(output_dir, "..", "quant_summary.json"))
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "status": "failed",
            "model_id": model,
            "scheme": scheme,
            "device": "xpu",
            "hardware": "Intel Arc Pro B60",
            "errors": [err],
        }, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Standalone XPU quantization with AutoRound")
    p.add_argument("--model", required=True)
    p.add_argument("--scheme", default="W4A16", choices=list(SCHEME_MAP.keys()))
    p.add_argument("--iters", type=int, default=0)
    p.add_argument("--export_format", default="auto_round", choices=["auto_round", "llm_compressor"])
    p.add_argument("--output_dir", default="./quantized_model")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--num_gpus", default="1")
    p.add_argument("--model_free", action="store_true")
    p.add_argument("--ignore_layers", default="")
    args = p.parse_args()
    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001
        _write_failed_summary(args.output_dir, args.model, args.scheme, f"{type(exc).__name__}: {exc}")
        _log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
