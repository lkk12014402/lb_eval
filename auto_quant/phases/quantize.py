#!/usr/bin/env python3
"""Phase 2: Deterministic quantization script.

Quantizes a model using auto-round with scheme-based configuration.
Architecture handling is fully automatic:
  - AutoModelForCausalLM + trust_remote_code handles model loading
  - AutoRound internally detects model type (llm/mllm/diffusion)
  - Block discovery is automatic (searches ModuleList in model tree)
  - MoE models recognized automatically (Mixtral, DeepSeek, Qwen MoE, etc.)

All parameters are controlled via CLI args (set by parent auto_v3.sh).

Usage:
    python quantize.py \
        --model <hf_model_id> \
        --scheme W4A16 \
        --iters 0 \
        --export_format auto_round \
        --output_dir ./quantized_model
"""

import argparse
import json
import logging
import os
import re
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ═══ Scheme → AutoRound scheme string mapping ═══
# AutoRound natively accepts these as the `scheme` parameter.
# It internally resolves bits, group_size, sym, data_type etc.
SCHEME_MAP = {
    "W4A16": "W4A16",
    "MXFP4": "MXFP4",
    "NVFP4": "NVFP4",
    "MXFP8": "MXFP8",
    "W8A16": "W8A16",
    "W4A16_ASYM": "W4A16_ASYM",
}

# Scheme with RCEIL suffix for auto_round export (better rounding for MX formats)
SCHEME_MAP_AUTOROUND_EXPORT = {
    "MXFP4": "MXFP4_RCEIL",
}

# ═══ Ignore layers strategy (from Qwen quantization recipes) ═══
# FP4 schemes (MXFP4/NVFP4) are aggressive — sensitive layers must stay in FP16.
# MoE models additionally need mlp.gate (router) protected.

# For MoE models (Mixtral, DeepSeek-V2/V3, Qwen-MoE, etc.)
MOE_IGNORE_LAYERS = {
    "W4A16": "lm_head",
    "MXFP4": "lm_head,mlp.gate,self_attn",
    "NVFP4": "lm_head,mlp.gate,self_attn",
    "MXFP8": "lm_head,mlp.gate",
    "W8A16": "lm_head",
}

# For dense models (Llama, Qwen, Gemma, Mistral, etc.)
DENSE_IGNORE_LAYERS = {
    "W4A16": "lm_head",
    "MXFP4": "lm_head,self_attn",
    "NVFP4": "lm_head,self_attn",
    "MXFP8": "lm_head",
    "W8A16": "lm_head",
}


def _parse_layer_config(raw: str):
    """Parse an auto-round relaxed-JSON layer_config string into a dict.

    Prefers auto-round's own ``parse_layer_config_arg`` (authoritative, matches
    the CLI behavior). Falls back to strict ``json.loads`` if unavailable.
    Raises ValueError on unparseable input so the pipeline fails loudly rather
    than silently ignoring a mixed-precision request.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        from auto_round.utils.common import parse_layer_config_arg
        return parse_layer_config_arg(raw)
    except ImportError:
        import json as _json
        try:
            return _json.loads(raw)
        except Exception as e:
            raise ValueError(f"Could not parse layer_config (strict JSON fallback): {e}") from e


def is_moe_model(model) -> bool:
    """Detect if model is a Mixture-of-Experts architecture."""
    model_type = getattr(model.config, "model_type", "")
    # Check config-level indicators
    if hasattr(model.config, "num_experts") or hasattr(model.config, "num_local_experts"):
        return True
    # Check known MoE model types
    moe_types = {"mixtral", "arctic", "dbrx", "jamba", "deepseek", "deepseek_v2",
                 "deepseek_v3", "qwen2_moe", "qwen3_moe", "phimoe", "grok"}
    if model_type in moe_types:
        return True
    # Check module names for MoE indicators
    for name, _ in model.named_modules():
        if "moe" in name.lower() or "gate" in name.lower().split(".")[-1:][0:1]:
            return True
    return False


def is_moe_config(config) -> bool:
    """Detect MoE from a HF config WITHOUT loading weights (used by model-free,
    where loading a possibly-huge model just to detect MoE would defeat the point)."""
    for attr in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
        if getattr(config, attr, None):
            return True
    model_type = (getattr(config, "model_type", "") or "").lower()
    moe_types = {"mixtral", "arctic", "dbrx", "jamba", "deepseek", "deepseek_v2",
                 "deepseek_v3", "deepseek_v4", "qwen2_moe", "qwen3_moe", "qwen3_5_moe",
                 "phimoe", "grok", "minimax", "minimax_m3", "longcat", "glm_moe"}
    if model_type in moe_types:
        return True
    arch = " ".join(getattr(config, "architectures", None) or []).lower()
    return "moe" in arch or "sparse" in arch


def resolve_device_map(requested, num_gpus, device_index):
    """Resolve the device_map passed to AutoRound so quantization actually runs on GPU.

    Why this exists: auto-round's own default is device_map=0 (GPU 0). Passing the
    transformers-style "auto" instead lets accelerate auto-dispatch the model, which —
    combined with low_gpu_mem_usage=True — frequently OFFLOADS small / W4A16 models to
    CPU. That makes quantization silently run on CPU even when a GPU is present.

    Rules (mirrors the documented CUDA device rules):
      - no CUDA            -> "cpu" (with a loud warning; caller asserts against this)
      - single GPU (<=1)   -> explicit int index (e.g. 0) so the model loads on cuda:N
      - multi-GPU (>1)     -> "auto" (accelerate shards across cards intentionally)
    An explicit non-"auto"/non-CPU request from the caller is always honored.
    """
    import torch

    try:
        n_gpus = int(num_gpus)
    except (TypeError, ValueError):
        n_gpus = 1
    try:
        dev_idx = int(device_index)
    except (TypeError, ValueError):
        dev_idx = 0

    if not torch.cuda.is_available():
        logger.warning("CUDA is NOT available — quantization would run on CPU (very slow).")
        return "cpu"

    # Honor an explicit, deliberate override (a specific device or a real device map),
    # but treat the default "auto" as "let us decide" so we can force GPU on single card.
    if requested and requested not in ("auto", "cpu", ""):
        return requested

    if n_gpus > 1:
        return "auto"
    return dev_idx


def assert_gpu_or_explain(resolved_device_map):
    """Fail LOUDLY if CUDA is present but quantization resolved to CPU.

    Prevents the silent CPU fallback: better to error and let the fix loop react than
    to spend an hour quantizing on CPU (or OOM the box).
    """
    import torch

    if not torch.cuda.is_available():
        return  # genuinely CPU-only environment; nothing to enforce

    major = None
    try:
        from auto_round.utils.device import get_major_device
        major = str(get_major_device(resolved_device_map))
    except Exception:
        # Fallback: infer from the resolved value itself
        major = "cpu" if str(resolved_device_map).lower() in ("cpu",) else "cuda"

    logger.info(f"Quantization compute device: {major} (device_map={resolved_device_map!r})")
    if major.startswith("cpu"):
        raise RuntimeError(
            f"CUDA is available but quantization resolved to CPU (device_map={resolved_device_map!r}). "
            "Refusing to run quantization on CPU. Ensure a GPU device_map (single-GPU index or 'auto' "
            "for multi-GPU) and that no fix installed a CPU-only torch or cleared CUDA_VISIBLE_DEVICES."
        )

    # Preflight free-VRAM check. A leftover process from a previous run / fix attempt can
    # keep holding GPU memory, starving this run. With low_gpu_mem_usage=True, auto-round
    # then SILENTLY offloads to CPU and quantization crawls for hours. Fail fast instead.
    try:
        if isinstance(resolved_device_map, int):
            idx = resolved_device_map
        else:
            idx = torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(idx)
        free_gb = free_b / (1024 ** 3)
        total_gb = total_b / (1024 ** 3)
        min_free = float(os.environ.get("MIN_FREE_VRAM_GB", "2"))
        logger.info(f"GPU{idx} free VRAM: {free_gb:.1f}GB / {total_gb:.1f}GB (min required: {min_free:.1f}GB)")
        if free_gb < min_free:
            raise RuntimeError(
                f"Only {free_gb:.1f}GB VRAM free on GPU{idx} (< {min_free:.1f}GB required). "
                "A previous or leftover process is likely still holding GPU memory, which would force "
                "this quantization to SILENTLY fall back to CPU. Free the GPU (kill stale processes / "
                "wait for VRAM to release) before retrying. Set MIN_FREE_VRAM_GB to tune this threshold."
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Could not read free VRAM (non-fatal): {e}")


def quantize(args):
    """Run quantization using AutoRound.

    Architecture coverage is handled by auto-round internally:
    - Standard LLM: Llama, Qwen, Mistral, Gemma, Phi, GPT-NeoX, etc.
    - MoE models: Mixtral, DeepSeek-V2/V3, Qwen-MoE, Arctic, etc.
    - MLLM: Qwen-VL, LLaVA, InternVL, etc. (detected via multimodal assets)
    - Custom architectures: any model with trust_remote_code=True

    Ignore layer strategy (from Qwen quantization recipes):
    - W4A16: only lm_head
    - MXFP4/NVFP4: lm_head + self_attn (FP4 too aggressive for attention)
    - MoE models: additionally mlp.gate (router precision is critical)
    """
    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    model_free = bool(getattr(args, "model_free", False))

    # Resolve export format. Model-free MXFP4/MXFP8 ONLY supports the
    # llm_compressor format (auto-round would otherwise silently fall back to the
    # regular calibration flow), so force it here.
    export_format = args.export_format
    if model_free and args.scheme in ("MXFP4", "MXFP8") and export_format != "llm_compressor":
        logger.warning(
            f"Model-free {args.scheme} only supports 'llm_compressor' export; "
            f"overriding '{export_format}' → 'llm_compressor'."
        )
        export_format = "llm_compressor"

    # Resolve scheme string (use RCEIL variant for auto_round export if applicable)
    if export_format == "auto_round" and args.scheme in SCHEME_MAP_AUTOROUND_EXPORT:
        ar_scheme = SCHEME_MAP_AUTOROUND_EXPORT[args.scheme]
    else:
        ar_scheme = SCHEME_MAP.get(args.scheme, args.scheme)

    iters = args.iters

    # Resolve the device_map so quantization runs on GPU (not silent CPU fallback).
    effective_device_map = resolve_device_map(args.device_map, args.num_gpus, args.device_index)
    assert_gpu_or_explain(effective_device_map)

    logger.info(f"Model: {args.model}")
    logger.info(f"Scheme: {args.scheme} → AutoRound scheme='{ar_scheme}'")
    logger.info(f"Iters: {iters} ({'RTN' if iters == 0 else 'TUNING'})")
    logger.info(f"Export format: {export_format}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device map: {args.device_map} → effective: {effective_device_map!r}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if model_free:
        # Model-free reads the checkpoint directly, shard by shard — do NOT load the
        # full model (it may be far larger than VRAM). Detect MoE from config only.
        logger.info("Loading config (model-free: no full-weight load)...")
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        moe = is_moe_config(cfg)
        arch_name = (getattr(cfg, "architectures", None) or ["unknown"])[0]
        model_type = getattr(cfg, "model_type", "unknown")
    else:
        # Load model — AutoModelForCausalLM handles all architectures via config.json
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=effective_device_map,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        arch_name = type(model).__name__
        model_type = getattr(model.config, "model_type", "unknown")
        moe = is_moe_model(model)
        del model
    logger.info(f"Architecture: {arch_name} (model_type={model_type}, moe={moe})")

    # Determine ignore layers based on scheme and model type (MoE vs dense).
    # A user-supplied --ignore_layers (whitelisted advanced submissions) OVERRIDES
    # the built-in table entirely; otherwise use the scheme/MoE default.
    custom_ignore = (getattr(args, "ignore_layers", "") or "").strip()
    if custom_ignore:
        ignore_layers = custom_ignore
        logger.info(f"Ignore layers (user override): {ignore_layers}")
    else:
        ignore_table = MOE_IGNORE_LAYERS if moe else DENSE_IGNORE_LAYERS
        ignore_layers = ignore_table.get(args.scheme, "lm_head")
        logger.info(f"Ignore layers (default): {ignore_layers}")

    # Optional mixed-precision layer_config (auto-round relaxed JSON).
    custom_layer_config = (getattr(args, "layer_config", "") or "").strip()
    parsed_layer_config = None
    if custom_layer_config:
        parsed_layer_config = _parse_layer_config(custom_layer_config)
        logger.info(f"Layer config (mixed precision): {parsed_layer_config}")

    # Build AutoRound — scheme-based API (auto-round >= 0.13)
    logger.info("Configuring AutoRound...")
    ar_kwargs = {
        "model": args.model,
        "tokenizer": tokenizer,
        "scheme": ar_scheme,
        "iters": iters,
        "low_gpu_mem_usage": True,
        "device_map": effective_device_map,
        # "enable_torch_compile": True,
        # "disable_opt_rtn": True,
    }

    # Model-free: weight-only RTN straight from the checkpoint (no calibration
    # forward). Routed inside AutoRound via is_model_free_route when model_free=True.
    # Only valid for weight-only schemes (W4A16/MXFP4/MXFP8) — gated upstream.
    if model_free:
        ar_kwargs["model_free"] = True
        logger.info("Model-free mode enabled (weight-only RTN, no calibration).")

    # Use ignore_layers to completely skip quantization for sensitive layers
    if ignore_layers:
        ar_kwargs["ignore_layers"] = ignore_layers

    # Mixed-precision per-module overrides
    if parsed_layer_config:
        ar_kwargs["layer_config"] = parsed_layer_config

    # Only pass seqlen/nsamples if tuning (iters > 0)
    if iters > 0:
        ar_kwargs["seqlen"] = args.seqlen
        ar_kwargs["nsamples"] = args.nsamples

    autoround = AutoRound(**ar_kwargs)

    # Execute quantization + export.
    os.makedirs(args.output_dir, exist_ok=True)
    start_time = time.time()
    if model_free:
        # Model-free MUST use the one-shot quantize_and_save entry point. Calling
        # .quantize() on a ModelFreeCompressor deliberately falls back to the
        # regular (calibration) compressor, which would defeat model-free.
        logger.info(f"Starting model-free quantization + export ({export_format})...")
        autoround.quantize_and_save(output_dir=args.output_dir, format=export_format)
    else:
        logger.info("Starting quantization...")
        autoround.quantize()
        logger.info(f"Saving quantized model ({export_format} format)...")
        autoround.save_quantized(
            output_dir=args.output_dir,
            format=export_format,
        )
    duration = time.time() - start_time
    logger.info(f"Quantization completed in {duration:.1f}s")

    # Collect output file list (for backward-compatibility with leaderboard)
    output_files = []
    if os.path.isdir(args.output_dir):
        output_files = sorted(
            os.path.join(args.output_dir, f)
            for f in os.listdir(args.output_dir)
            if os.path.isfile(os.path.join(args.output_dir, f))
        )

    # Compute model size info
    original_size_mb = None
    quantized_size_mb = None
    compression_ratio = None
    try:
        quantized_size_bytes = sum(
            os.path.getsize(p) for p in output_files if p.endswith((".safetensors", ".bin"))
        )
        if quantized_size_bytes > 0:
            quantized_size_mb = round(quantized_size_bytes / (1024 * 1024), 1)
            # Estimate original size from model config
            num_params = getattr(model.config, "num_parameters", None) or getattr(model, "num_parameters", lambda: None)()
            if num_params:
                original_size_mb = round(num_params * 2 / (1024 * 1024), 1)  # fp16 baseline
                compression_ratio = round(original_size_mb / quantized_size_mb, 2) if quantized_size_mb else None
    except Exception:
        pass

    # Derive method name (backward-compat: old pipeline always wrote "RTN" or "TUNING")
    method = "RTN" if iters == 0 else "TUNING"

    # Write summary
    summary = {
        "status": "success",
        "model_id": args.model,
        "architecture": arch_name,
        "model_type": model_type,
        "is_moe": moe,
        "scheme": args.scheme,
        "method": method,
        "ar_scheme": ar_scheme,
        "iters": iters,
        "export_format": export_format,
        "ignore_layers": ignore_layers,
        "model_free": model_free,
        "layer_config": custom_layer_config or None,
        "duration_seconds": round(duration, 1),
        "output_dir": args.output_dir,
        "device": str(effective_device_map),
        "device_map": str(effective_device_map),
        "num_gpus": str(args.num_gpus),
        "output_files": output_files,
        "original_size_mb": original_size_mb,
        "quantized_size_mb": quantized_size_mb,
        "compression_ratio": compression_ratio,
        "errors": [],
        "solutions": [],
    }
    summary_path = os.path.join(args.output_dir, "..", "quant_summary.json")
    summary_path = os.path.normpath(summary_path)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info(f"Summary written to {summary_path}")

    logger.info("=== Phase 2: DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic quantization with AutoRound")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--scheme", choices=list(SCHEME_MAP.keys()), default="W4A16",
                        help="Quantization scheme (mapped to AutoRound scheme string)")
    parser.add_argument("--iters", type=int, default=0,
                        help="Optimization iterations (0=RTN, 200=TUNING)")
    parser.add_argument("--export_format", choices=["auto_round", "llm_compressor"],
                        default="auto_round", help="Model export format")
    parser.add_argument("--output_dir", default="./quantized_model",
                        help="Output directory for quantized model")
    parser.add_argument("--device_map", default="auto",
                        help="Device map for model loading (default 'auto' → resolved to GPU index on single card)")
    parser.add_argument("--device_index", default="0",
                        help="GPU index to use on a single-GPU run (forces cuda:N instead of CPU offload)")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Calibration sequence length (only used when iters > 0)")
    parser.add_argument("--nsamples", type=int, default=128,
                        help="Number of calibration samples (only used when iters > 0)")
    parser.add_argument("--num_gpus", default="1",
                        help="Number of GPUs: 1 → single-GPU (forced cuda:index); >1 → device_map='auto' sharding")
    parser.add_argument("--model_free", action="store_true",
                        help="Use auto-round model-free (weight-only RTN, no calibration forward). "
                             "Only valid for weight-only schemes (W4A16/MXFP4/MXFP8).")
    parser.add_argument("--ignore_layers", default="",
                        help="Comma-separated module substrings to skip. When set, OVERRIDES the "
                             "built-in scheme/MoE ignore table. Empty = use built-in defaults.")
    parser.add_argument("--layer_config", default="",
                        help="auto-round layer_config for mixed precision, e.g. "
                             "'{block_sparse_moe.experts:{bits:4,data_type:mx_fp}}'. Empty = uniform scheme.")
    args = parser.parse_args()

    try:
        quantize(args)
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
