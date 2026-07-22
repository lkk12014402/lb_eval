#!/usr/bin/env python3
"""Verify machines reached through the tensorflow SSH jump host.

The first hop is handled by Paramiko. Subsequent hops run the jump host's own
OpenSSH client so aliases and private keys that exist only on that host continue
to work (notably ``changwa1_b200`` and B60's ``~/.ssh/id_rsa_4096``).
"""
from __future__ import annotations

import argparse
import os
import shlex
import socket
import sys
import time

import paramiko

from machine_profiles import (
    JUMP_HOST,
    JUMP_PASSWORD_ENV,
    JUMP_USER,
    MACHINE_PROFILES,
    MachineProfile,
    get_profile,
)


def nested_ssh_argv(profile: MachineProfile) -> list[str]:
    common = [
        "ssh",
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
    ]
    if profile.name == "changwa1-b200":
        common += ["-o", "BatchMode=yes", "changwa1_b200"]
    elif profile.name == "sc09-b200":
        common += ["hshen@172.26.46.180"]
    elif profile.name == "b60-xpu":
        common += [
            "-i", "~/.ssh/id_rsa_4096",
            "-J", "guest@146.152.205.45",
            "sdp@192.168.11.2",
        ]
    else:
        raise ValueError(f"No SSH route implementation for {profile.name}")
    return common


def _target_ssh_command(profile: MachineProfile) -> str:
    common = [*nested_ssh_argv(profile), profile.probe_command]
    # Preserve ~ expansion for the jump-host key path; quote all other args.
    return " ".join(
        arg if arg == "~/.ssh/id_rsa_4096" else shlex.quote(arg)
        for arg in common
    )


def _sanitize_output(text: str, *secrets: str) -> str:
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        sanitized_line = line
        stripped = line.strip()
        for secret in secrets:
            if not secret:
                continue
            # Nested OpenSSH may echo the supplied password as its own PTY line.
            # Avoid global replacement: short passwords can coincidentally occur
            # inside harmless output such as GPU UUIDs.
            if stripped == secret:
                ending = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
                sanitized_line = "***REDACTED***" + ending
                break
        lines.append(sanitized_line)
    return "".join(lines)


def verify_profile(name: str, timeout: int = 60) -> tuple[bool, str]:
    profile = get_profile(name)
    jump_password = os.environ.get(JUMP_PASSWORD_ENV) or os.environ.get("LOCAL_SSH_PASS")
    if not jump_password:
        return False, (
            f"Missing {JUMP_PASSWORD_ENV} (LOCAL_SSH_PASS may be used as fallback)"
        )
    target_password = (
        os.environ.get(profile.target_password_env)
        if profile.target_password_env else ""
    )
    if profile.target_password_env and not target_password:
        return False, f"Missing {profile.target_password_env}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=os.environ.get("LOCAL_JUMP_SSH_HOST", JUMP_HOST),
            port=int(os.environ.get("LOCAL_JUMP_SSH_PORT", "22")),
            username=os.environ.get("LOCAL_JUMP_SSH_USER", JUMP_USER),
            password=jump_password,
            timeout=20,
            auth_timeout=20,
            banner_timeout=20,
            allow_agent=False,
            look_for_keys=False,
        )
        transport = client.get_transport()
        if transport is None:
            return False, "Jump-host SSH transport was not created"
        channel = transport.open_session(timeout=20)
        channel.get_pty()
        channel.exec_command(_target_ssh_command(profile))

        output = bytearray()
        password_sends = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if channel.recv_ready():
                chunk = channel.recv(4096)
                output.extend(chunk)
                recent = output[-2048:].decode("utf-8", "replace").lower()
                if "password:" in recent and target_password and password_sends < 2:
                    channel.send(target_password + "\n")
                    password_sends += 1
            if channel.recv_stderr_ready():
                output.extend(channel.recv_stderr(4096))
            if channel.exit_status_ready() and not channel.recv_ready():
                rc = channel.recv_exit_status()
                text = _sanitize_output(
                    output.decode("utf-8", "replace"),
                    jump_password,
                    target_password,
                ).strip()
                return rc == 0, text or f"nested ssh exited {rc}"
            time.sleep(0.1)
        channel.close()
        return False, (
            f"Timed out after {timeout}s\n"
            + _sanitize_output(
                output.decode("utf-8", "replace"),
                jump_password,
                target_password,
            ).strip()
        )
    except (paramiko.SSHException, socket.error, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        client.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Verify configured multi-hop machines")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    verify = sub.add_parser("verify")
    verify.add_argument("--machine", required=True, choices=sorted(MACHINE_PROFILES))
    verify.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()

    if args.command == "list":
        for profile in MACHINE_PROFILES.values():
            print(
                f"{profile.name:<16} {profile.hardware:<16} "
                f"{profile.route} [{profile.execution_mode}]"
            )
        return 0

    ok, detail = verify_profile(args.machine, args.timeout)
    print(("OK" if ok else "FAILED") + f": {args.machine}")
    print(detail)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
