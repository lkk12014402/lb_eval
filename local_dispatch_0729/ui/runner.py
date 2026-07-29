#!/usr/bin/env python3
"""Background dispatch runner for the UI.

Launches ``reserve_and_login.py`` (the single entry point: reserve → login →
build → run auto.sh) as a subprocess, streams its output to a per-job log file,
and updates the job store's status as the run progresses.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time

from jobstore import (STATUS_FAILED, STATUS_FINISHED, STATUS_RUNNING,
                      get_job, update_job)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)          # local_dispatch/
LOG_DIR = os.path.join(_HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def build_command(job: dict) -> list[str]:
    """Build the reserve_and_login.py argv from a job row."""
    cmd = [
        "python3", os.path.join(_PARENT, "reserve_and_login.py"),
        "--model", job["model"],
        "--scheme", job["scheme"],
        "--method", job["method"],
        "--user", job["user"],
        "--hours", str(job["hours"] or 4),
    ]
    if job.get("server"):
        cmd += ["--server", job["server"]]
        if job.get("gpus"):
            cmd += ["--gpus", job["gpus"]]
    return cmd


def _dispatch(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    log_path = os.path.join(LOG_DIR, f"{job_id}.log")
    update_job(job_id, status=STATUS_RUNNING, log_path=log_path)

    cmd = build_command(job)
    host = ""
    reserved = ""
    run_id = ""
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=_PARENT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=os.environ.copy(),
        )
        for line in proc.stdout:  # type: ignore[union-attr]
            log.write(line)
            log.flush()
            # Opportunistically capture host / reserved GPUs / run info from output.
            m = re.search(r"Selected:.*\(([^)]+)\)\s+\S+\s+x\d+\s+GPUs\s+\[([^\]]*)\]", line)
            if m:
                host = m.group(1)
                reserved = m.group(2).replace(" ", "")
                update_job(job_id, host=host, reserved_gpus=reserved)
        proc.wait()
        rc = proc.returncode
        log.write(f"\n=== dispatch finished: exit_code={rc} ({'OK' if rc == 0 else 'FAILED'}) ===\n")
        log.flush()

    status = STATUS_FINISHED if rc == 0 else STATUS_FAILED
    update_job(job_id, status=status, exit_code=rc)


def start_dispatch(job_id: str) -> threading.Thread:
    """Kick off the dispatch in a daemon thread; returns the thread."""
    t = threading.Thread(target=_dispatch, args=(job_id,), daemon=True)
    t.start()
    return t


def read_log(job_id: str, position: str = "tail", n: int = 1000) -> str:
    """Return a job's log. position: 'tail' (latest n), 'head' (first n), 'full'."""
    job = get_job(job_id)
    if not job or not job.get("log_path") or not os.path.isfile(job["log_path"]):
        return "(no log yet)"
    with open(job["log_path"], encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total = len(lines)
    if position == "full":
        return "".join(lines)
    if position == "head":
        body = "".join(lines[:n])
        return body + (f"\n… (+{total - n} more lines below; switch to Full/Tail)\n" if total > n else "")
    # tail
    head_note = f"… (showing last {n} of {total} lines; switch to Head/Full for the start)\n" if total > n else ""
    return head_note + "".join(lines[-n:])


def read_logs_combined(job_ids: list[str], tail_each: int = 40) -> str:
    """Concatenate the tail of several jobs' logs — for monitoring many at once."""
    if not job_ids:
        return "(no running jobs)"
    blocks = []
    for jid in job_ids:
        job = get_job(jid)
        header = f"───── {jid}  [{(job or {}).get('status','?')}]  {(job or {}).get('model','')} ─────"
        blocks.append(header + "\n" + read_log(jid, tail=tail_each))
    return "\n\n".join(blocks)
