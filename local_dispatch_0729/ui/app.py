#!/usr/bin/env python3
"""Local Dispatch UI — a simple Gradio app mirroring the leaderboard, for the
local GPU dispatch path.

Tabs:
  * Leaderboard  — finished/failed runs read from the lb_eval repo (results/).
  * Queue        — submitted requests + status read from the repo (status/).
  * GPU Machines — live availability from the reservation API.
  * My Dispatches — local jobs submitted through this UI (live status + logs).
  * Submit       — submit a model to run via local_dispatch (reserve → build → run).

Run:  python3 app.py   (then open the printed URL)
Secrets (HF_TOKENS, MINIMAX_API_KEY, GIT_TOKEN, LOCAL_SSH_PASS, LOCAL_HTTP_PROXY…)
must be exported in the environment first (see ../run.sh).
"""
from __future__ import annotations

import os
import sys
import uuid

import gradio as gr
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import datasrc
import jobstore
import runner
from machine_profiles import MACHINE_PROFILES
from multi_hop_ssh import verify_profile

SCHEMES = ["W4A16", "MXFP4", "NVFP4", "W8A16", "MXFP8"]
METHODS = ["RTN", "TUNING", "MODEL_FREE"]


def default_user() -> str:
    """Default reservation user: env override, else SSH user, else system user."""
    import getpass
    for key in ("LOCAL_RESERVE_USER", "LOCAL_SSH_USER"):
        v = os.environ.get(key)
        if v:
            return v
    try:
        return getpass.getuser()
    except Exception:
        return "root"


# ─────────────────────────── data adapters ───────────────────────────
def results_df() -> pd.DataFrame:
    cols = ["#", "model", "scheme", "status", "avg_acc", "hf_repo", "generated_at"]
    rows = datasrc.load_results()
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)[["model", "scheme", "status", "avg_acc", "hf_repo", "generated_at"]]
    # Rank finished runs by avg accuracy (desc); rows without acc sink to the bottom.
    df = df.sort_values(by="avg_acc", ascending=False, na_position="last").reset_index(drop=True)
    df.insert(0, "#", range(1, len(df) + 1))
    return df


def queue_df() -> pd.DataFrame:
    rows = datasrc.load_queue()
    if not rows:
        return pd.DataFrame(columns=["model", "scheme", "method", "status", "submitted_by"])
    df = pd.DataFrame(rows)
    return df[["model", "scheme", "method", "status", "submitted_by"]]


def gpu_df() -> tuple[pd.DataFrame, str]:
    rows, err = datasrc.load_gpu_machines()
    if not rows:
        return pd.DataFrame(columns=[
            "server", "host", "gpu", "total", "available", "busy", "access"
        ]), err
    return pd.DataFrame(rows), ("" if not err else f"⚠️ {err}")


def jobs_df() -> pd.DataFrame:
    rows = jobstore.list_jobs()
    cols = ["id", "model", "scheme", "method", "status", "host", "reserved_gpus", "exit_code"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df:
            df[c] = None
    return df[cols]


def job_choices() -> list[str]:
    return [r["id"] for r in jobstore.list_jobs()]


# ─────────────────────────── submit action ───────────────────────────
def submit(model, scheme, method, user, machine_mode, server, gpus, hours):
    model = (model or "").strip()
    user = (user or "").strip() or default_user()   # fall back to a sensible default
    if not model or "/" not in model:
        return "❌ Model id required, e.g. Qwen/Qwen3-1.7B", jobs_df(), gr.update()

    manual = machine_mode == "Manual (pick machine)"
    job = jobstore.Job(
        id=uuid.uuid4().hex[:12],
        model=model, scheme=scheme, method=method, user=user,
        server=(server.strip() if manual else ""),
        gpus=(gpus.strip() if manual else ""),
        hours=int(hours),
        status=jobstore.STATUS_QUEUED,
    )
    if manual and not job.server:
        return "❌ Manual mode needs a --server value", jobs_df(), gr.update()

    jobstore.create_job(job)
    runner.start_dispatch(job.id)
    msg = (f"✅ Submitted dispatch `{job.id}` — {model} / {scheme} / {method}\n"
           f"Mode: {'manual ' + job.server + (' gpus=' + job.gpus if job.gpus else '') if manual else 'auto (best GPU)'}\n"
           f"Watch progress in **My Dispatches** → select `{job.id}` for live logs.")
    return msg, jobs_df(), gr.update(choices=job_choices(), value=job.id)


def refresh_log(job_id, position="tail"):
    if not job_id:
        return "(select a job)"
    return runner.read_log(job_id, position=_pos_key(position))


def running_job_ids() -> list[str]:
    return [r["id"] for r in jobstore.list_jobs()
            if r["status"] in (jobstore.STATUS_QUEUED, jobstore.STATUS_RUNNING)]


def running_summary() -> str:
    n = len(running_job_ids())
    total = len(jobstore.list_jobs())
    return f"**{n} running / {total} total** dispatches"


def _pos_key(label: str) -> str:
    return {"Tail (latest)": "tail", "Head (start)": "head", "Full": "full"}.get(label, "tail")


# ─────────────────────────── UI layout ───────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local Dispatch") as demo:
        gr.Markdown("# 🖥️ Local Dispatch — Quantization on local GPU machines\n"
                    "Submit models to quantize+evaluate on reserved local GPUs, and view results "
                    "(mirrors the leaderboard's data, backed by the same lb_eval repo).")

        with gr.Tab("🏆 Leaderboard"):
            gr.Markdown("Finished / failed local runs (from the configured HF dataset), "
                        "ranked by average accuracy.")
            lb = gr.Dataframe(results_df, interactive=False, wrap=True, show_row_numbers=False)
            gr.Button("🔄 Refresh").click(lambda: results_df(), outputs=lb)

        with gr.Tab("📋 Queue"):
            gr.Markdown("Local-dispatch requests and status (from the configured HF dataset).")
            q = gr.Dataframe(queue_df, interactive=False, wrap=True, show_row_numbers=False)
            gr.Button("🔄 Refresh").click(lambda: queue_df(), outputs=q)

        with gr.Tab("🎛️ GPU Machines"):
            gr.Markdown("Live availability from the reservation system.")
            gpu_init, gpu_err0 = gpu_df()
            gpu_msg = gr.Markdown(gpu_err0 or "")
            gpu_tbl = gr.Dataframe(gpu_init, interactive=False, wrap=True, show_row_numbers=False)

            def _refresh_gpu():
                df, err = gpu_df()
                return df, (err or "")
            gr.Button("🔄 Refresh").click(_refresh_gpu, outputs=[gpu_tbl, gpu_msg])

            gr.Markdown(
                "### Multi-hop SSH verification\n"
                "SC09/changwa1 B200 support full CUDA dispatch; "
                "Intel B60 supports full XPU dispatch (independent runner)."
            )
            with gr.Row():
                profile_name = gr.Dropdown(
                    choices=sorted(MACHINE_PROFILES),
                    value="sc09-b200",
                    label="Multi-hop machine",
                )
                verify_timeout = gr.Dropdown(
                    choices=[30, 60, 90, 120], value=60, label="Timeout (seconds)"
                )
            verify_output = gr.Textbox(
                label="SSH verification output", lines=8, interactive=False
            )

            def _verify_multihop(name, timeout):
                ok, detail = verify_profile(name, int(timeout))
                return ("✅ " if ok else "❌ ") + name + "\n" + detail

            gr.Button("🔐 Verify multi-hop SSH").click(
                _verify_multihop,
                inputs=[profile_name, verify_timeout],
                outputs=verify_output,
            )

        with gr.Tab("📨 My Dispatches"):
            gr.Markdown("Jobs submitted through this UI. **Auto-refreshes** — monitor "
                        "multiple concurrent dispatches live.")
            with gr.Row():
                run_summary = gr.Markdown(running_summary())
                auto_refresh = gr.Checkbox(value=True, label="Auto-refresh (live)")
            jt = gr.Dataframe(jobs_df, interactive=False, wrap=True, show_row_numbers=False)

            with gr.Row():
                job_sel = gr.Dropdown(choices=job_choices(), label="Job id (for single-job log)")
                view_mode = gr.Radio(["Selected job", "All running (combined)"],
                                     value="Selected job", label="Log view")
            with gr.Row():
                log_pos = gr.Radio(["Tail (latest)", "Head (start)", "Full"],
                                   value="Tail (latest)", label="Log position",
                                   info="Head/Full let you read the beginning; turn off autoscroll to keep it still.")
                log_autoscroll = gr.Checkbox(value=True, label="Autoscroll to bottom")
            log_box = gr.Textbox(label="Log", lines=26, max_lines=26,
                                 interactive=False, autoscroll=True)

            def _log_view(mode, job_id, position):
                if mode == "All running (combined)":
                    return runner.read_logs_combined(running_job_ids())
                return refresh_log(job_id, position)

            # Toggling autoscroll re-configures the textbox live.
            log_autoscroll.change(lambda v: gr.update(autoscroll=v),
                                  inputs=log_autoscroll, outputs=log_box)
            # Switching to Head/Full auto-disables autoscroll so the top stays visible.
            def _on_pos(position):
                keep = position == "Tail (latest)"
                return gr.update(value=keep), gr.update(autoscroll=keep)
            log_pos.change(_on_pos, inputs=log_pos, outputs=[log_autoscroll, log_box])

            # Manual refresh button (also refreshes the dropdown choices).
            gr.Button("🔄 Refresh now").click(
                lambda m, j, p: (jobs_df(), gr.update(choices=job_choices()),
                                 _log_view(m, j, p), running_summary()),
                inputs=[view_mode, job_sel, log_pos], outputs=[jt, job_sel, log_box, run_summary])

            # Live auto-refresh: ticks every 2s while the checkbox is on.
            live = gr.Timer(2.0)

            def _tick(enabled, mode, job_id, position):
                if not enabled:
                    return gr.update(), gr.update(), gr.update()
                return jobs_df(), _log_view(mode, job_id, position), running_summary()
            live.tick(_tick, inputs=[auto_refresh, view_mode, job_sel, log_pos],
                      outputs=[jt, log_box, run_summary])

        with gr.Tab("➕ Submit"):
            gr.Markdown("Submit a model to run on a local GPU machine via **local_dispatch**.")
            with gr.Row():
                model_in = gr.Textbox(label="Model id", placeholder="Qwen/Qwen3-1.7B")
                user_in = gr.Textbox(label="Reservation user", value=default_user(),
                                     placeholder="kaokao")
            with gr.Row():
                scheme_in = gr.Dropdown(SCHEMES, value="W4A16", label="Scheme")
                method_in = gr.Dropdown(METHODS, value="RTN", label="Method")
                hours_in = gr.Dropdown([1, 2, 3, 4], value=4, label="Hours")
            machine_mode = gr.Radio(
                ["Auto (best available GPU)", "Manual (pick machine)"],
                value="Auto (best available GPU)", label="Machine selection")
            with gr.Row():
                server_in = gr.Textbox(
                    label="--server (manual)",
                    placeholder="L20x8-smc-1 / sc09-b200 / changwa1-b200 / b60-xpu / 4090D",
                )
                gpus_in = gr.Textbox(label="--gpus (manual, optional)", placeholder="0,1,2 or auto:2")
            submit_btn = gr.Button("🚀 Submit dispatch", variant="primary")
            submit_out = gr.Markdown()
            submit_jobs = gr.Dataframe(jobs_df, interactive=False, wrap=True, visible=False)

            submit_btn.click(
                submit,
                inputs=[model_in, scheme_in, method_in, user_in, machine_mode, server_in, gpus_in, hours_in],
                outputs=[submit_out, submit_jobs, job_sel],
            )

    return demo


if __name__ == "__main__":
    port = int(os.environ.get("LOCAL_DISPATCH_UI_PORT", "7899"))
    build_ui().queue().launch(server_name="0.0.0.0", server_port=port,
                              theme=gr.themes.Soft(), show_error=True)
