#!/usr/bin/env python3
"""Local GPU dispatch — step 1: select → reserve → login.

Given a user-submitted model, this:

  1. Estimates the VRAM the auto-quant pipeline needs (download-free).
  2. Pulls the live machine inventory from the reservation API.
  3. Selects a machine that fits — *preferring the better GPUs* (design choice),
     using the fewest cards needed on that machine.
  4. Reserves the cards (immediate or scheduled start).
  5. Verifies SSH login (unified user + LOCAL_SSH_PASS).

Output: a JSON descriptor {host, gpu_ids, reservation_id, ssh_ok, ...} that the
next pipeline step (docker build / run auto.sh on the host) will consume.

Reuses the reservation API helpers from ``gpu_reserve.py`` (same directory's
parent). Standalone VRAM estimation — no leaderboard import.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

# Make sibling modules + parent's gpu_reserve importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gpu_reserve as gr                                  # noqa: E402
from gpu_catalog import spec_for_model, effective_vram, min_cards_to_fit  # noqa: E402
from vram_estimator import estimate_vram                  # noqa: E402


DEFAULT_SSH_USER = os.environ.get("LOCAL_SSH_USER", "root")
DEFAULT_SSH_PORT = int(os.environ.get("LOCAL_SSH_PORT", "22"))


@dataclass
class Candidate:
    server_id: str
    server_name: str
    host: str
    model: str
    quality: int
    n_cards: int
    gpu_ids: list[int]
    effective_vram_gb: float


def _free_gpu_ids(server, start_iso, hours) -> list[dict]:
    """GPUs on *server* with no schedule conflict in the requested window."""
    return [g for g in server["gpus"] if not gr.has_conflict(g, start_iso, hours)]


def select_machine(servers, need_gb: float, start_iso, hours) -> Candidate | None:
    """Pick the best-fitting machine.

    Preference order (design: prefer better GPUs):
      1. Highest GPU quality rank that can fit the model.
      2. Among equal quality, fewest cards needed.
      3. Tie-break: most free cards (headroom), then server name.
    """
    candidates: list[Candidate] = []
    for s in servers:
        free = _free_gpu_ids(s, start_iso, hours)
        if not free:
            continue
        # All GPUs on a node are the same model in this inventory; group by model anyway.
        by_model: dict[str, list[int]] = {}
        for g in free:
            by_model.setdefault(g["model"], []).append(g["device_id"])
        for model, ids in by_model.items():
            spec = spec_for_model(model)
            if spec is None:
                continue
            n_free = len(ids)
            need_cards = min_cards_to_fit(spec, need_gb)
            if need_cards is None or need_cards > n_free:
                continue
            chosen = sorted(ids)[:need_cards]
            candidates.append(Candidate(
                server_id=s["id"], server_name=s["name"], host=s["host"],
                model=model, quality=spec.quality, n_cards=need_cards,
                gpu_ids=chosen,
                effective_vram_gb=round(effective_vram(spec, need_cards), 1),
            ))
    if not candidates:
        return None
    # prefer: higher quality, then fewer cards, then more free headroom, then name.
    candidates.sort(key=lambda c: (-c.quality, c.n_cards, c.server_name))
    return candidates[0]


def select_manual(servers, server_kw: str, gpus_arg: str | None, start_iso, hours,
                  need_gb: float) -> tuple[Candidate, list[str]]:
    """Build a Candidate from user-specified --server (+ optional --gpus).

    Returns (candidate, conflict_warnings). Raises SystemExit on bad input.
    Does NOT enforce the VRAM fit (caller warns) — the user's choice wins.

    Card-conflict policy (manual = user specified --server → never raise on conflict):
      * explicit --gpus '0,1,2'  → trust the user: honor exactly, only warn on overlap.
      * auto:N (or omitted)      → prefer free cards, but if not enough, fall back to
        the lowest device_ids anyway (warn) — a specified machine always proceeds.
    """
    server = gr.find_server(servers, server_kw)
    gpu_by_id = {g["device_id"]: g for g in server["gpus"]}
    all_ids = sorted(gpu_by_id)
    model = server["gpus"][0]["model"] if server["gpus"] else "?"
    spec = spec_for_model(model)

    if not gpus_arg:
        n = min_cards_to_fit(spec, need_gb) if (spec and need_gb > 0) else 1
        gpus_arg = f"auto:{n or 1}"

    warnings: list[str] = []
    if gpus_arg.startswith("auto"):
        # auto-pick: prefer free cards; never raise — fall back to busy ones if needed.
        n = int(gpus_arg.split(":")[1]) if ":" in gpus_arg else 1
        free = [g["device_id"] for g in server["gpus"]
                if not gr.has_conflict(g, start_iso, hours)]
        if len(free) >= n:
            gpu_ids = sorted(free)[:n]
        else:
            gpu_ids = free + [i for i in all_ids if i not in free]
            gpu_ids = sorted(gpu_ids[:n]) if len(gpu_ids) >= n else all_ids
            if len(free) < n:
                warnings.append(
                    f"该时段仅 {len(free)} 块空闲，已强制选用 {gpu_ids}（部分与他人重叠）")
    else:
        # explicit list: honor it; validate existence only, warn on conflicts.
        ids = [int(x) for x in gpus_arg.split(",") if x.strip() != ""]
        missing = [i for i in ids if i not in gpu_by_id]
        if missing:
            raise SystemExit(f"GPU {missing} 不存在于 {server['name']}。"
                             f"可用编号: {all_ids}")
        for i in ids:
            slot = gr.has_conflict(gpu_by_id[i], start_iso, hours)
            if slot:
                warnings.append(
                    f"GPU {i} 该时段已被 {slot.get('username','?')} 预约 "
                    f"({slot['start_time'][11:16]}-{slot['end_time'][11:16]})，将与其重叠")
        gpu_ids = ids

    quality = spec.quality if spec else 0
    eff = effective_vram(spec, len(gpu_ids)) if spec else 0.0
    cand = Candidate(
        server_id=server["id"], server_name=server["name"], host=server["host"],
        model=model, quality=quality, n_cards=len(gpu_ids),
        gpu_ids=gpu_ids, effective_vram_gb=round(eff, 1),
    )
    return cand, warnings


def verify_ssh(host: str, user: str, password: str, port: int = 22, timeout: int = 15) -> tuple[bool, str]:
    """Attempt an SSH login and run a trivial command. Returns (ok, detail)."""
    try:
        import paramiko
    except ImportError:
        return False, "paramiko not installed"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, port=port, username=user, password=password,
                       timeout=timeout, allow_agent=False, look_for_keys=False)
        stdin, stdout, stderr = client.exec_command("nvidia-smi -L || echo NO_NVIDIA_SMI", timeout=timeout)
        out = stdout.read().decode("utf-8", "replace").strip()
        return True, out.splitlines()[0] if out else "connected"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    finally:
        try:
            client.close()
        except Exception:
            pass


def dispatch(model_id: str, scheme: str, user: str, hours: int, start: str | None,
             ssh_user: str, ssh_port: int, dry_run: bool, no_ssh: bool,
             revision: str = "main", hf_token: str | None = None,
             server: str | None = None, gpus: str | None = None) -> dict:
    est = estimate_vram(model_id, scheme=scheme, revision=revision, token=hf_token)
    need_gb = est.required_gb

    result: dict = {
        "model_id": model_id,
        "scheme": scheme,
        "estimate": asdict(est),
    }

    manual = server is not None
    if need_gb <= 0 and not manual:
        result["error"] = ("could not estimate model size (no params/config/name); "
                            "use --server/--gpus to reserve manually")
        return result

    start_iso = gr.parse_start(start)
    servers = gr.api("/api/servers")

    if manual:
        cand, conflict_warnings = select_manual(servers, server, gpus, start_iso, hours, need_gb)
        result["manual"] = True
        if conflict_warnings:
            result["conflict_warnings"] = conflict_warnings
        # Warn (don't block) if the manual choice looks too small.
        if need_gb > 0 and cand.effective_vram_gb and cand.effective_vram_gb < need_gb:
            result["fit_warning"] = (
                f"selected ~{cand.effective_vram_gb}GB usable < estimated need "
                f"{need_gb}GB — may OOM")
    else:
        cand = select_machine(servers, need_gb, start_iso, hours)
        if cand is None:
            result["error"] = (f"no machine can fit {need_gb:.1f} GB in the requested window "
                               f"(start={start_iso or 'now'}, {hours}h)")
            return result

    result["selected"] = asdict(cand)
    result["start_time"] = start_iso or "immediate"
    result["duration_hours"] = hours

    if dry_run:
        result["dry_run"] = True
        return result

    payload = {
        "username": user,
        "server_id": cand.server_id,
        "gpu_ids": cand.gpu_ids,
        "start_time": start_iso,
        "duration_hours": hours,
    }
    try:
        resp = gr.api("/api/reservations", method="POST", body=payload)
        result["reservation"] = resp
        result["reservation_id"] = (resp or {}).get("id") if isinstance(resp, dict) else None
    except SystemExit as e:
        # Manual mode: a specified machine always proceeds to SSH even if the
        # reservation is rejected (e.g. time conflict). Auto mode still aborts.
        if not manual:
            raise
        result["reservation"] = None
        result["reservation_id"] = None
        result["reservation_error"] = str(e)

    if no_ssh:
        result["ssh_ok"] = None
        return result

    password = os.environ.get("LOCAL_SSH_PASS")
    if not password:
        result["ssh_ok"] = False
        result["ssh_detail"] = "LOCAL_SSH_PASS not set"
        return result
    ok, detail = verify_ssh(cand.host, ssh_user, password, port=ssh_port)
    result["ssh_ok"] = ok
    result["ssh_detail"] = detail
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Local GPU dispatch: select → reserve → login")
    p.add_argument("--model", required=True, help="HuggingFace model id to quantize/evaluate")
    p.add_argument("--scheme", default="W4A16", help="quantization scheme (W4A16/MXFP4/MXFP8/...)")
    p.add_argument("--user", required=True, help="reservation username")
    p.add_argument("--hours", type=int, default=4, choices=[1, 2, 3, 4], help="reservation duration (1-4h)")
    p.add_argument("--start", default=None, help="scheduled start HH:MM (CST); default immediate")
    p.add_argument("--server", default=None,
                   help="manual: reserve on this machine (matches name or host, e.g. H20 / 118.195 / 4090D). "
                        "Overrides auto-selection.")
    p.add_argument("--gpus", default=None,
                   help="manual: GPU device_ids '0,1,2' or 'auto:N'. Requires --server; "
                        "defaults to auto:N sized to fit.")
    p.add_argument("--ssh-user", default=DEFAULT_SSH_USER, help="SSH login user (default: LOCAL_SSH_USER or root)")
    p.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT)
    p.add_argument("--revision", default="main", help="model revision")
    p.add_argument("--dry-run", action="store_true", help="select only; do not reserve")
    p.add_argument("--no-ssh", action="store_true", help="reserve but skip SSH verification")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    # ── Step 2: build & run on the reserved host (default ON) ──────────────────
    p.add_argument("--reserve-only", action="store_true",
                   help="stop after reserve + SSH login; do NOT build/run auto.sh on the host")
    p.add_argument("--method", default="RTN", help="quant method: RTN | TUNING | MODEL_FREE")
    p.add_argument("--export-format", default="auto_round", help="auto_round | llm_compressor")
    p.add_argument("--git-repo", default=None, help="lb_eval clone URL (default: config.env value)")
    p.add_argument("--git-branch", default=None, help="lb_eval branch (default: main)")
    p.add_argument("--image", default=None, help="docker image tag (default: cuda-openclaw:local)")
    p.add_argument("--remote-root", default=None, help="remote working dir (default: ~/lb_eval_dispatch)")
    p.add_argument("--hf-cache", default=None, help="remote HF cache dir to mount (default: ~/hf_cache)")
    p.add_argument("--request-json", default=None, help="use an existing request.json instead of generating one")
    p.add_argument("--http-proxy", default=os.environ.get("LOCAL_HTTP_PROXY", os.environ.get("HTTP_PROXY", "")),
                   help="proxy for the remote host (git/docker build/run). Default: $LOCAL_HTTP_PROXY / $HTTP_PROXY")
    p.add_argument("--https-proxy", default=os.environ.get("LOCAL_HTTPS_PROXY", os.environ.get("HTTPS_PROXY", "")),
                   help="https proxy for the remote host. Default: $LOCAL_HTTPS_PROXY / $HTTPS_PROXY")
    p.add_argument("--no-proxy", default=os.environ.get("LOCAL_NO_PROXY", "localhost,127.0.0.1,::1"),
                   help="no_proxy list for the remote host")
    p.add_argument("--run-dry", action="store_true",
                   help="step 2: print the remote script + request.json, do not execute")
    args = p.parse_args()
    if args.gpus and not args.server:
        p.error("--gpus requires --server")

    res = dispatch(
        model_id=args.model, scheme=args.scheme, user=args.user, hours=args.hours,
        start=args.start, ssh_user=args.ssh_user, ssh_port=args.ssh_port,
        dry_run=args.dry_run, no_ssh=args.no_ssh, revision=args.revision,
        hf_token=os.environ.get("HF_TOKEN"),
        server=args.server, gpus=args.gpus,
    )

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if not res.get("error") else 1

    est = res["estimate"]
    print(f"Model:    {res['model_id']}  ({est['params_b']}B, {est['num_layers']} layers, "
          f"src={est['input_bits']}-bit, out={est['output_bits']}-bit, via {est['source']})")
    print(f"Scheme:   {res['scheme']}")
    print(f"VRAM:     quant={est['quant_vram_gb']}GB  eval={est['eval_vram_gb']}GB  "
          f"→ need {est['required_gb']}GB")
    if res.get("error"):
        print(f"ERROR:    {res['error']}")
        return 1
    c = res["selected"]
    mode = "manual" if res.get("manual") else "auto"
    print(f"Selected: {c['server_name']}  ({c['host']})  {c['model']} x{c['n_cards']}  "
          f"GPUs {c['gpu_ids']}  (~{c['effective_vram_gb']}GB usable)  [{mode}]")
    if res.get("fit_warning"):
        print(f"WARNING:  {res['fit_warning']}")
    for w in res.get("conflict_warnings", []):
        print(f"WARNING:  {w}")
    print(f"Window:   start={res['start_time']}  {res['duration_hours']}h")
    if res.get("dry_run"):
        print("[dry-run] not reserved.")
        return 0
    if res.get("reservation_error"):
        print(f"Reserved: FAILED ({res['reservation_error']}) — 继续 SSH（manual 模式）")
    else:
        print(f"Reserved: id={res.get('reservation_id')}")
    if res.get("ssh_ok") is None:
        print("SSH:      skipped")
    elif res.get("ssh_ok"):
        print(f"SSH:      OK — {res.get('ssh_detail')}")
    else:
        print(f"SSH:      FAILED — {res.get('ssh_detail')}")
        return 2

    # ── Step 2: build & run auto.sh on the reserved host ──────────────────────
    if args.reserve_only:
        print("Run:      skipped (--reserve-only)")
        return 0
    if res.get("ssh_ok") is None:
        print("Run:      skipped (SSH not verified; use without --no-ssh to run)")
        return 0

    return _run_on_host(args, c)


def _run_on_host(args, cand: dict) -> int:
    """Hand off to step 2 (remote_exec) on the reserved machine."""
    import remote_exec

    req = None
    if args.request_json:
        with open(args.request_json) as f:
            req = json.load(f)

    overrides = {}
    if args.git_repo:    overrides["git_repo"] = args.git_repo
    if args.git_branch:  overrides["git_branch"] = args.git_branch
    if args.image:       overrides["image"] = args.image
    if args.remote_root: overrides["remote_root"] = args.remote_root
    if args.hf_cache:    overrides["hf_cache"] = args.hf_cache
    if args.http_proxy:  overrides["http_proxy"] = args.http_proxy
    if args.https_proxy: overrides["https_proxy"] = args.https_proxy
    if args.no_proxy:    overrides["no_proxy"] = args.no_proxy

    job = remote_exec.RemoteJob(
        host=cand["host"], gpu_ids=cand["gpu_ids"], model=args.model, scheme=args.scheme,
        ssh_user=args.ssh_user, ssh_port=args.ssh_port,
        method=args.method, export_format=args.export_format,
        request_json=req, **overrides,
    )

    print(f"\n══════ Step 2: build & run on {cand['host']} (GPUs {cand['gpu_ids']}) ══════")
    if args.run_dry:
        r = remote_exec.run_remote(job, password="", dry_run=True)
        print("# request.json:\n" + r["request_json_text"])
        print("\n# remote script:\n" + r["script"])
        return 0

    password = os.environ.get("LOCAL_SSH_PASS")
    if not password:
        print("ERROR: LOCAL_SSH_PASS not set", file=sys.stderr)
        return 1
    r = remote_exec.run_remote(job, password=password)
    print(f"\n=== run done: exit_code={r.get('exit_code')} ok={r.get('ok')} ===")
    return 0 if r.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
