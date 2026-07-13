#!/usr/bin/env python3
"""Standalone VRAM estimator — download-free, no cross-repo dependency.

Reads a model's HuggingFace ``config.json`` (and safetensors index for an exact
parameter count when available) to estimate the GPU VRAM needed for the two
phases of the auto-quant pipeline:

  * quantize  — layerwise/blockwise; only a few layers on GPU at once
  * evaluate  — the whole quantized model must fit (+ KV cache / activations)

The pipeline needs whichever phase is larger. Formulas mirror the leaderboard's
(estimate_weight_memory_gb / estimate_quantization_memory_gb) but are reimplemented
here so this package has no dependency on the leaderboard repo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import AutoConfig


# Bits per weight for the pipeline's quantization schemes (output/quantized).
_SCHEME_BITS = {
    "W4A16": 4, "INT4 (W4A16)": 4, "MXFP4": 4, "NVFP4": 4,
    "W8A16": 8, "INT8 (W8A16)": 8, "MXFP8": 8,
}


@dataclass
class VRamEstimate:
    params_b: float          # parameter count in billions
    num_layers: int
    input_bits: int          # source model precision (e.g. 16 for bf16)
    output_bits: int         # quantized precision
    quant_vram_gb: float     # VRAM to quantize
    eval_vram_gb: float      # VRAM to evaluate the quantized model
    required_gb: float       # max of the two (what we must reserve for)
    source: str              # how params were derived


def _params_from_safetensors(model_id: str, revision: str, token: str | None) -> float | None:
    """Exact parameter count (in billions) from safetensors headers, no weights.

    ``get_safetensors_metadata`` reads only each shard's header (tensor shapes),
    never the weight bytes, so it's fast and download-free.
    """
    try:
        from huggingface_hub import get_safetensors_metadata
        meta = get_safetensors_metadata(model_id, revision=revision, token=token)
        pc = getattr(meta, "parameter_count", None)
        if pc is None:
            return None
        total = sum(pc.values()) if isinstance(pc, dict) else pc
        return float(total) / 1e9 if total else None
    except Exception:
        pass
    # Fallback: parse the safetensors index metadata directly.
    try:
        from huggingface_hub import hf_hub_download
        import json
        p = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision, token=token)
        with open(p) as f:
            meta = json.load(f).get("metadata", {})
        if meta.get("total_parameters"):
            return float(meta["total_parameters"]) / 1e9
    except Exception:
        pass
    return None


def _params_from_config(cfg) -> float | None:
    """Estimate params (billions) from config dims when no exact count is available."""
    for attr in ("num_parameters", "n_params"):
        v = getattr(cfg, attr, None)
        if v:
            return float(v) / 1e9
    # Rough transformer estimate: 12 * L * h^2 (attn+mlp) + embeddings.
    h = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)
    L = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
    vocab = getattr(cfg, "vocab_size", None)
    if h and L:
        core = 12 * L * h * h
        emb = (vocab * h) if vocab else 0
        # MoE: multiply expert FFN by number of experts if present.
        n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None) \
            or getattr(cfg, "n_routed_experts", None)
        if n_exp:
            # crude: FFN is ~2/3 of core; scale it by experts.
            core = int(core * (1 + 0.66 * (int(n_exp) - 1)))
        return (core + emb) / 1e9
    return None


def _params_from_name(model_id: str) -> float | None:
    """Fallback: parse '7B' / '13B' / '0.6B' from the model id."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_id.replace("-", " "))
    return float(m.group(1)) if m else None


def estimate_vram(
    model_id: str,
    scheme: str = "W4A16",
    revision: str = "main",
    token: str | None = None,
    quant_overhead: float = 1.5,
    eval_overhead: float = 2.2,
) -> VRamEstimate:
    """Estimate required VRAM (GB) for quantizing + evaluating *model_id*.

    Never raises for the estimation itself; if params can't be derived, returns a
    conservative zero estimate (caller should treat 0 as "unknown → pick a big GPU").
    """
    cfg = None
    num_layers = 0
    input_bits = 16
    try:
        cfg = AutoConfig.from_pretrained(model_id, revision=revision, token=token, trust_remote_code=True)
        num_layers = getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0) or 0
        dtype = str(getattr(cfg, "torch_dtype", "") or "").lower()
        input_bits = 32 if "float32" in dtype or "fp32" in dtype else 16
    except Exception:
        pass

    params_b = _params_from_safetensors(model_id, revision, token)
    source = "safetensors"
    if not params_b and cfg is not None:
        params_b = _params_from_config(cfg)
        source = "config-estimate"
    if not params_b:
        params_b = _params_from_name(model_id)
        source = "name-regex"
    params_b = params_b or 0.0

    output_bits = _SCHEME_BITS.get(scheme, 4)

    # Source model weights (loaded in input precision for quantization).
    model_weight_gb = params_b * (input_bits / 8.0)
    # Quantized model weights (for eval).
    quant_weight_gb = params_b * (output_bits / 8.0)

    # Quantize phase: layerwise → a couple of layers on GPU at a time.
    if num_layers > 0 and model_weight_gb > 0:
        layer_gb = model_weight_gb / num_layers
        quant_vram = round(2 * layer_gb * quant_overhead, 2)
    else:
        # No layer count: fall back to a fraction of the full model.
        quant_vram = round(model_weight_gb * 0.5 * quant_overhead, 2)

    # Eval phase: whole quantized model + KV cache/activations.
    eval_vram = round(quant_weight_gb * eval_overhead, 2)

    required = max(quant_vram, eval_vram)
    return VRamEstimate(
        params_b=round(params_b, 3),
        num_layers=num_layers,
        input_bits=input_bits,
        output_bits=output_bits,
        quant_vram_gb=quant_vram,
        eval_vram_gb=eval_vram,
        required_gb=required,
        source=source,
    )
