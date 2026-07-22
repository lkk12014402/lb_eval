#!/usr/bin/env python3
"""Static machines that are not managed by the GPU reservation API.

These profiles describe connectivity only. They are currently marked
``login_only`` because the existing execution engine assumes a directly reachable
NVIDIA/CUDA Docker host. B60 additionally needs an XPU-specific execution backend.

Passwords are never stored here; profiles reference environment-variable names.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MachineProfile:
    name: str
    display_name: str
    hardware: str
    target: str
    route: str
    target_password_env: str = ""
    probe_command: str = "hostname"
    execution_mode: str = "login_only"
    notes: str = ""


JUMP_HOST = "10.23.167.71"
JUMP_USER = "tensorflow"
JUMP_PASSWORD_ENV = "LOCAL_JUMP_SSH_PASS"


MACHINE_PROFILES: dict[str, MachineProfile] = {
    "sc09-b200": MachineProfile(
        name="sc09-b200",
        display_name="SC09 B200",
        hardware="NVIDIA B200",
        target="hshen@172.26.46.180",
        route=f"{JUMP_USER}@{JUMP_HOST} -> hshen@172.26.46.180",
        target_password_env="SC09_SSH_PASS",
        probe_command="hostname; nvidia-smi -L",
        execution_mode="nvidia_cuda",
        notes="Jump password + independent target password.",
    ),
    "changwa1-b200": MachineProfile(
        name="changwa1-b200",
        display_name="changwa1 B200",
        hardware="NVIDIA B200",
        target="changwa1_b200",
        route=f"{JUMP_USER}@{JUMP_HOST} -> changwa1_b200",
        probe_command="hostname; nvidia-smi -L",
        execution_mode="nvidia_cuda",
        notes="Target alias/key are configured only on the tensorflow jump host.",
    ),
    "b60-xpu": MachineProfile(
        name="b60-xpu",
        display_name="Intel B60 XPU",
        hardware="Intel XPU B60",
        target="sdp@192.168.11.2",
        route=(
            f"{JUMP_USER}@{JUMP_HOST} -> guest@146.152.205.45 "
            "-> sdp@192.168.11.2"
        ),
        target_password_env="B60_SSH_PASS",
        probe_command=(
            "hostname; "
            "(xpu-smi discovery 2>/dev/null || "
            "sycl-ls 2>/dev/null || "
            "lspci | grep -Ei 'display|vga|intel')"
        ),
        execution_mode="xpu_execution",
        notes=(
            "The guest hop uses ~/.ssh/id_rsa_4096 on the tensorflow jump host. "
            "Full XPU quant+eval runs via xpu/xpu_dispatch.py."
        ),
    ),
}


def get_profile(name: str) -> MachineProfile:
    key = (name or "").strip().lower()
    if key in MACHINE_PROFILES:
        return MACHINE_PROFILES[key]
    matches = [
        p for p in MACHINE_PROFILES.values()
        if key and (
            key in p.name.lower()
            or key in p.display_name.lower()
            or key in p.target.lower()
        )
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous machine profile '{name}'")
    raise ValueError(
        f"Unknown machine profile '{name}'. Available: "
        + ", ".join(MACHINE_PROFILES)
    )


def ui_rows() -> list[dict]:
    return [
        {
            "server": p.display_name,
            "host": p.target,
            "gpu": p.hardware,
            "total": 8,
            "available": "-",
            "busy": "-",
            "access": (
                "multi-hop / CUDA"
                if p.execution_mode == "nvidia_cuda"
                else "multi-hop / XPU"
                if p.execution_mode == "xpu_execution"
                else "multi-hop / login-only"
            ),
        }
        for p in MACHINE_PROFILES.values()
    ]
