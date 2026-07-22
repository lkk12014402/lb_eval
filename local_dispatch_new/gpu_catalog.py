#!/usr/bin/env python3
"""Local GPU catalog — the real machines available via the reservation system.

Unlike the leaderboard's cloud catalog, these are the on-prem / cloud-host GPUs
that ``gpu_reserve.py`` (http://.../api/servers) can reserve. Each entry maps a
GPU ``model`` string (as reported by the reservation API) to its per-card VRAM,
the max cards addressable in one node, and a *quality rank* used to prefer better
hardware first (per design: "prefer the better local machines").

Keep this in sync with the reservation system's inventory. The ``model`` key must
match the ``model`` field returned by ``/api/servers`` (e.g. "4090D", "L20", "H20").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalGPUSpec:
    model: str          # matches the reservation API's gpu["model"]
    vram_gb: int        # usable VRAM per card
    max_cards: int      # max cards in a single node
    quality: int        # higher = better/preferred (design: prefer better GPUs)


# Sorted best-first by quality. Adjust as the reservation inventory changes.
#   H20      96GB  — datacenter, highest throughput/interconnect
#   RTX6000D 84GB  — pro/server card
#   L20      48GB  — inference card, decent VRAM
#   5090D    32GB  — consumer flagship
#   4090D    24GB  — consumer
LOCAL_GPU_CATALOG: list[LocalGPUSpec] = [
    LocalGPUSpec("B200",      192, 8, 60),
    LocalGPUSpec("H20",       96, 8, 50),
    LocalGPUSpec("RTX 6000D", 84, 8, 40),
    LocalGPUSpec("RTX6000D",  84, 8, 40),   # alias without space
    LocalGPUSpec("L20",       48, 8, 30),
    LocalGPUSpec("5090D",     32, 8, 20),
    LocalGPUSpec("4090D",     24, 8, 10),
]

# Fast lookup by model string.
_BY_MODEL: dict[str, LocalGPUSpec] = {g.model: g for g in LOCAL_GPU_CATALOG}

# When N>1 cards are used, tensor/pipeline parallelism can't use 100% of the
# aggregate VRAM (parameter duplication, comm buffers). Mirrors leaderboard's 0.85.
MULTI_GPU_EFFICIENCY = 0.85


def spec_for_model(model: str) -> LocalGPUSpec | None:
    """Return the catalog spec for a reservation-API model string, or None."""
    if not model:
        return None
    if model in _BY_MODEL:
        return _BY_MODEL[model]
    # tolerant match: strip spaces / case
    norm = model.replace(" ", "").lower()
    for g in LOCAL_GPU_CATALOG:
        if g.model.replace(" ", "").lower() == norm:
            return g
    return None


def effective_vram(spec: LocalGPUSpec, n_cards: int) -> float:
    """Aggregate usable VRAM across *n_cards* of this GPU type."""
    if n_cards <= 1:
        return float(spec.vram_gb)
    return n_cards * spec.vram_gb * MULTI_GPU_EFFICIENCY


def min_cards_to_fit(spec: LocalGPUSpec, need_gb: float) -> int | None:
    """Fewest cards of this GPU type whose effective VRAM ≥ need_gb, or None."""
    for n in range(1, spec.max_cards + 1):
        if effective_vram(spec, n) >= need_gb:
            return n
    return None
