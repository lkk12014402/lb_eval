#!/usr/bin/env python3
"""Standalone Intel B60 (XPU) dispatcher.

Independent of the CUDA execution path. Reaches the B60 through the tensorflow
jump host + guest key + sdp password (3 hops), stages the lb_eval checkout and
the xpu/ scripts, builds the XPU image, and runs the XPU pipeline in a container
with /dev/dri passthrough and ZE_AFFINITY_MASK device selection.

Reuses remote_exec's jump-host/tunnel/tar helpers so the CUDA code is untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent               # local_dispatch/
for p in (str(_PARENT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import remote_exec  # noqa: E402  (reuses jump/tunnel/tar/exec helpers)

# ── B60 route (matches machine_profiles b60-xpu) ─────────────────────────────
GUEST_HOP = "guest@146.152.205.45"
GUEST_KEY = "~/.ssh/id_rsa_4096"
B60_HOST = "192.168.11.2"
B60_USER = "sdp"
B60_PASS_ENV = "B60_SSH_PASS"

DEFAULT_GIT_REPO = "https://github.com/XuehaoSun/lb_eval.git"
DEFAULT_GIT_BRANCH = "main"
DEFAULT_IMAGE = "xpu-openclaw:local"
DEFAULT_BASE_IMAGE = "intel/llm-scaler-vllm:0.21.0-b1"
DEFAULT_REMOTE_ROOT = "~/lb_xpu_dispatch"
DEFAULT_RESULTS_DATASET = os.environ.get("LOCAL_RESULTS_DATASET", "lvkaokao/lb_local")

LOCAL_UPLOADER = _PARENT / "upload_results_hf_dataset.py"


def leaderboard_request_filename(model: str, scheme: str, method: str) -> str:
    return remote_exec.leaderboard_request_filename(model, scheme, method, False)


def build_request(model, scheme, method, export_format, gpu_ids, existing) -> tuple[str, dict]:
    if existing:
        req = dict(existing)
        fname = req.get("request_filename") or leaderboard_request_filename(model, scheme, method)
    else:
        req = {
            "model": model, "quant_scheme": scheme, "method": method,
            "export_format": export_format, "script": "auto_quant",
            "model_type": "quantization", "status": "Pending", "hardware": "Intel Arc Pro B60",
        }
        fname = leaderboard_request_filename(model, scheme, method)
    n = len(gpu_ids)
    # Container sees only the reserved cards, re-indexed 0..N-1 via ZE_AFFINITY_MASK.
    req["cuda_visible_devices"] = ",".join(str(i) for i in range(n)) if n else ""
    req["request_filename"] = fname
    return fname, req


def _repo_tar(files: dict) -> bytes:
    return remote_exec._tar_bytes(files)


def build_remote_script(run_id: str, req_rel: str, gpu_ids: list, image: str,
                        base_image: str, git_repo: str, git_branch: str,
                        remote_root: str, has_secrets: bool, results_dataset: str,
                        do_openclaw: bool) -> str:
    ze_mask = ",".join(str(g) for g in gpu_ids) if gpu_ids else "0"
    oneapi_sel = "level_zero:" + ze_mask   # match ZE_AFFINITY_MASK (validated recipe)
    env_file = '    --env-file "$WORK/_secrets.env" \\\n' if has_secrets else ""
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT=$(eval echo "{remote_root}")
RUN_ID="{run_id}"
WORK="$ROOT/runs/$RUN_ID"
REPO="$WORK/lb_eval"
IMAGE="{image}"
BASE_IMAGE="{base_image}"
BRANCH="{git_branch}"
GIT_REPO="{git_repo}"
REQ_REL="{req_rel}"
ZE_MASK="{ze_mask}"
ONEAPI_SEL="{oneapi_sel}"
CONTAINER="lb-xpu-$RUN_ID"
HOST_UID="$(id -u)"; HOST_GID="$(id -g)"

# B60 has direct network access; ensure no stale corporate proxy is inherited.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

mkdir -p "$WORK"

echo "== [1/4] clone lb_eval (isolated: $REPO) =="
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO" reset --hard "origin/$BRANCH"
else
    rm -rf "$REPO"; git clone --depth 1 -b "$BRANCH" "$GIT_REPO" "$REPO"
fi

echo "== [2/4] stage xpu scripts + request + bootstrap =="
mkdir -p "$WORK/xpu" "$(dirname "$REPO/$REQ_REL")"
cp "$WORK"/_xpu_quantize.py          "$WORK/xpu/xpu_quantize.py"
cp "$WORK"/_xpu_evaluate.py          "$WORK/xpu/xpu_evaluate.py"
cp "$WORK"/_xpu_pipeline.sh          "$WORK/xpu/xpu_pipeline.sh"
cp "$WORK"/_xpu_quantize_wrapper.sh  "$WORK/xpu/xpu_quantize_wrapper.sh"
cp "$WORK"/_xpu_evaluate_wrapper.sh  "$WORK/xpu/xpu_evaluate_wrapper.sh"
cp "$WORK"/_xpu_fixloop_overrides.sh "$WORK/xpu/xpu_fixloop_overrides.sh"
cp "$WORK"/_xpu_bootstrap.sh         "$REPO/_xpu_bootstrap.sh"
cp "$WORK"/_local_upload_results.py "$REPO/_local_upload_results.py"
cp "$WORK"/_request.json            "$REPO/$REQ_REL"
chmod +x "$REPO/_xpu_bootstrap.sh" "$WORK/xpu/xpu_pipeline.sh" "$WORK/xpu/xpu_quantize_wrapper.sh" "$WORK/xpu/xpu_evaluate_wrapper.sh"

echo "== [3/4] docker build XPU image (base=$BASE_IMAGE) =="
docker build --build-arg XPU_BASE_IMAGE="$BASE_IMAGE" \
    -f "$WORK/_xpu.Dockerfile" -t "$IMAGE" "$WORK"

echo "== [4/4] docker run XPU pipeline (ZE_AFFINITY_MASK=$ZE_MASK) =="
trap 'rm -f "$WORK/_secrets.env"' EXIT
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
rc=0
docker run --rm --name "$CONTAINER" \\
    --device /dev/dri:/dev/dri \\
    -v /dev/dri/by-path:/dev/dri/by-path \\
    --ipc=host --privileged \\
    -e ZE_AFFINITY_MASK="$ZE_MASK" \\
    -e ONEAPI_DEVICE_SELECTOR="$ONEAPI_SEL" \\
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \\
    -e REQ_REL="$REQ_REL" \\
    -e DO_OPENCLAW="{1 if do_openclaw else 0}" \\
    -e LOCAL_RESULTS_DATASET="{results_dataset}" \\
    -e LOCAL_RUN_ID="$RUN_ID" \\
{env_file}    -v "$REPO":/workspace/lb_eval \\
    -v "$WORK/xpu":/workspace/xpu \\
    -v "$HOME/.cache/huggingface":/root/.cache/huggingface \\
    --entrypoint bash \\
    "$IMAGE" \\
    /workspace/lb_eval/_xpu_bootstrap.sh || rc=$?

# Restore ownership of the bind-mounted checkout (container runs as root).
docker run --rm -v "$WORK":/work --entrypoint sh "$BASE_IMAGE" \\
    -c "chown -R $HOST_UID:$HOST_GID /work" >/dev/null 2>&1 || true

exit $rc
"""


def _collect_secrets() -> dict:
    src = {
        "HF_TOKENS": os.environ.get("HF_TOKENS") or os.environ.get("HUGGINGFACE_TOKEN")
                     or os.environ.get("HF_TOKEN") or "",
        "MINIMAX_API_KEY": os.environ.get("MINIMAX_API_KEY") or os.environ.get("MINIMAX_KEY") or "",
        "GIT_TOKEN": os.environ.get("GIT_TOKEN") or "",
        "LB_STORAGE_BLOB_TOKEN": os.environ.get("LB_STORAGE_BLOB_TOKEN") or "",
        "LOCAL_RESULTS_HF_TOKEN": os.environ.get("LOCAL_RESULTS_HF_TOKEN") or "",
    }
    return {k: v for k, v in src.items() if v}


def _staged_payload(run_id, req, script, secrets) -> dict:
    files = {
        "_dispatch_run.sh": (script.encode(), 0o700),
        "_request.json": (json.dumps(req, ensure_ascii=False, indent=2).encode(), 0o600),
        "_xpu.Dockerfile": ((_HERE / "Dockerfile").read_bytes(), 0o600),
        "_xpu_quantize.py": ((_HERE / "xpu_quantize.py").read_bytes(), 0o600),
        "_xpu_evaluate.py": ((_HERE / "xpu_evaluate.py").read_bytes(), 0o600),
        "_xpu_pipeline.sh": ((_HERE / "xpu_pipeline.sh").read_bytes(), 0o700),
        "_xpu_quantize_wrapper.sh": ((_HERE / "xpu_quantize_wrapper.sh").read_bytes(), 0o700),
        "_xpu_evaluate_wrapper.sh": ((_HERE / "xpu_evaluate_wrapper.sh").read_bytes(), 0o700),
        "_xpu_fixloop_overrides.sh": ((_HERE / "xpu_fixloop_overrides.sh").read_bytes(), 0o600),
        "_xpu_bootstrap.sh": ((_HERE / "xpu_container_bootstrap.sh").read_bytes(), 0o700),
        "_local_upload_results.py": (LOCAL_UPLOADER.read_bytes(), 0o600),
    }
    if secrets:
        files["_secrets.env"] = ("".join(f"{k}={v}\n" for k, v in secrets.items()).encode(), 0o600)
    return files


def _open_b60_tunnel(jump, timeout: int) -> tuple[int, str, str]:
    """Open a jump-host-local forward to the B60 via the guest hop. Returns (port, sock, log)."""
    suffix = uuid.uuid4().hex[:8]
    _, out, _ = jump.exec_command(
        "python3 -c 'import socket; s=socket.socket(); s.bind((\"127.0.0.1\",0)); "
        "print(s.getsockname()[1]); s.close()'", timeout=timeout)
    port = int(out.read().decode().strip())
    sock = f"/tmp/lb-b60-{suffix}.sock"
    log = f"/tmp/lb-b60-{suffix}.log"
    cmd = (f"nohup ssh -NT -M -S {sock} -o ExitOnForwardFailure=yes "
           f"-o StrictHostKeyChecking=accept-new -i {GUEST_KEY} "
           f"-L 127.0.0.1:{port}:{B60_HOST}:22 {GUEST_HOP} "
           f">{log} 2>&1 < /dev/null & echo $!")
    jump.exec_command(cmd, timeout=timeout)
    for _ in range(40):
        _, po, _ = jump.exec_command(
            f"python3 -c 'import socket; s=socket.socket(); s.settimeout(.3); "
            f"print(s.connect_ex((\"127.0.0.1\",{port}))); s.close()'", timeout=timeout)
        if po.read().decode().strip() == "0":
            return port, sock, log
        time.sleep(1)
    _, lo, _ = jump.exec_command(f"cat {log}", timeout=timeout)
    raise RuntimeError(f"B60 tunnel did not open: {lo.read().decode()}")


def _close_b60_tunnel(jump, sock: str, log: str) -> None:
    jump.exec_command(
        f"ssh -S {sock} -O exit {GUEST_HOP} >/dev/null 2>&1 || true; rm -f {sock} {log}")


def dispatch(model, scheme, method, export_format, gpu_ids, remote_root, image,
             base_image, git_repo, git_branch, results_dataset, request_json,
             no_openclaw, dry_run, timeout=30) -> dict:
    import paramiko

    run_id = (f"{model.split('/')[-1]}_{scheme}_{method}".replace(" ", "")
              + "_" + time.strftime("%Y%m%d-%H%M%S") + "_" + uuid.uuid4().hex[:6])
    run_id = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in run_id)
    fname, req = build_request(model, scheme, method, export_format, gpu_ids, request_json)
    req_rel = f"pending_requests/local/{fname}"
    secrets = _collect_secrets()
    script = build_remote_script(run_id, req_rel, gpu_ids, image, base_image,
                                 git_repo, git_branch, remote_root, bool(secrets),
                                 results_dataset, not no_openclaw)
    result = {"machine": "b60-xpu", "run_id": run_id, "request_rel": req_rel,
              "request": req, "gpu_ids": gpu_ids, "secrets_present": sorted(secrets)}

    if dry_run:
        result["dry_run"] = True
        result["script"] = script
        return result

    if not os.environ.get(B60_PASS_ENV):
        raise RuntimeError(f"Missing {B60_PASS_ENV}")

    jump = remote_exec._connect_jump(timeout)
    port = sock = log = None
    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ch = None
    try:
        port, sock, log = _open_b60_tunnel(jump, timeout)
        ch = jump.get_transport().open_channel(
            "direct-tcpip", ("127.0.0.1", port), ("127.0.0.1", 0), timeout=timeout)
        target.connect(hostname=B60_HOST, username=B60_USER,
                       password=os.environ[B60_PASS_ENV], sock=ch, timeout=timeout,
                       auth_timeout=timeout, banner_timeout=timeout,
                       allow_agent=False, look_for_keys=False)

        remote_root_abs = remote_exec._expand_remote(target, remote_root)
        work = f"{remote_root_abs}/runs/{run_id}"
        sftp = target.open_sftp()
        try:
            remote_exec._sftp_makedirs(sftp, work)
            for name, (content, mode) in _staged_payload(run_id, req, script, secrets).items():
                path = f"{work}/{name}"
                with sftp.open(path, "wb") as f:
                    f.write(content)
                sftp.chmod(path, mode)
        finally:
            sftp.close()

        exit_code = remote_exec._exec_stream(target, f"bash {shlex.quote(work)}/_dispatch_run.sh")
        result["exit_code"] = exit_code
        result["ok"] = exit_code == 0
        return result
    finally:
        target.close()
        if ch is not None:
            ch.close()
        if sock:
            _close_b60_tunnel(jump, sock, log)
        jump.close()


def main() -> int:
    p = argparse.ArgumentParser(description="Dispatch a quantization run to the Intel B60 (XPU)")
    p.add_argument("--model", required=True)
    p.add_argument("--scheme", default="W4A16")
    p.add_argument("--method", default="RTN")
    p.add_argument("--export-format", default="auto_round")
    p.add_argument("--gpus", default="0", help="reserved XPU card ids, e.g. 0 or 0,1")
    p.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    p.add_argument("--git-repo", default=DEFAULT_GIT_REPO)
    p.add_argument("--git-branch", default=DEFAULT_GIT_BRANCH)
    p.add_argument("--results-dataset", default=DEFAULT_RESULTS_DATASET)
    p.add_argument("--request-json", default=None)
    p.add_argument("--no-openclaw-setup", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    req = None
    if args.request_json:
        with open(args.request_json) as f:
            req = json.load(f)
    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    res = dispatch(
        model=args.model, scheme=args.scheme, method=args.method,
        export_format=args.export_format, gpu_ids=gpu_ids, remote_root=args.remote_root,
        image=args.image, base_image=args.base_image, git_repo=args.git_repo,
        git_branch=args.git_branch, results_dataset=args.results_dataset,
        request_json=req, no_openclaw=args.no_openclaw_setup, dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"# run_id: {res['run_id']}")
        print(f"# request_filename: {res['request']['request_filename']}")
        print(f"# secrets: {res['secrets_present']}")
        print(res["script"])
        return 0
    print(f"\n=== XPU run done: exit_code={res.get('exit_code')} ok={res.get('ok')} ===")
    return 0 if res.get("ok") else 3


if __name__ == "__main__":
    raise SystemExit(main())
