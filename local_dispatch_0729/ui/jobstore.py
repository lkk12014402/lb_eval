#!/usr/bin/env python3
"""Local dispatch job store — a tiny SQLite table tracking submitted dispatches.

Each row is one dispatch (reserve → build → run auto.sh on a reserved GPU host),
with its live status and the path to its captured log. Kept intentionally small;
the authoritative *result* data (accuracy, artifacts) lives in the lb_eval repo
that ``upload_results_github.py`` pushes to — this store only tracks in-flight and
recent local dispatches so the UI has something to show immediately. The
authoritative result artifacts are uploaded to the configured Hugging Face
dataset (default: ``lvkaokao/lb_local``).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass

DB_PATH = os.environ.get("LOCAL_DISPATCH_DB",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs.db"))

# Job lifecycle states.
STATUS_QUEUED = "Queued"
STATUS_RUNNING = "Running"
STATUS_FINISHED = "Finished"
STATUS_FAILED = "Failed"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id            TEXT PRIMARY KEY,
                model         TEXT NOT NULL,
                scheme        TEXT NOT NULL,
                method        TEXT NOT NULL,
                user          TEXT,
                server        TEXT,
                gpus          TEXT,
                hours         INTEGER,
                status        TEXT NOT NULL,
                host          TEXT,
                reserved_gpus TEXT,
                run_id        TEXT,
                log_path      TEXT,
                exit_code     INTEGER,
                submitted_at  REAL,
                updated_at    REAL,
                extra         TEXT
            )
            """
        )


@dataclass
class Job:
    id: str
    model: str
    scheme: str
    method: str
    user: str
    server: str
    gpus: str
    hours: int
    status: str = STATUS_QUEUED
    host: str = ""
    reserved_gpus: str = ""
    run_id: str = ""
    log_path: str = ""
    exit_code: int | None = None
    submitted_at: float = 0.0
    updated_at: float = 0.0
    extra: dict | None = None


def create_job(job: Job) -> None:
    now = time.time()
    job.submitted_at = job.submitted_at or now
    job.updated_at = now
    with _conn() as c:
        c.execute(
            """
            INSERT INTO jobs (id, model, scheme, method, user, server, gpus, hours,
                              status, host, reserved_gpus, run_id, log_path, exit_code,
                              submitted_at, updated_at, extra)
            VALUES (:id, :model, :scheme, :method, :user, :server, :gpus, :hours,
                    :status, :host, :reserved_gpus, :run_id, :log_path, :exit_code,
                    :submitted_at, :updated_at, :extra)
            """,
            {**job.__dict__, "extra": json.dumps(job.extra or {})},
        )


def update_job(job_id: str, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = time.time()
    if "extra" in fields and not isinstance(fields["extra"], str):
        fields["extra"] = json.dumps(fields["extra"])
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with _conn() as c:
        c.execute(f"UPDATE jobs SET {sets} WHERE id = :id", {**fields, "id": job_id})


def get_job(job_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def list_jobs(limit: int = 200) -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM jobs ORDER BY submitted_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


init_db()
