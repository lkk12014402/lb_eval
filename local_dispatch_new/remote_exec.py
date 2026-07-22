#!/usr/bin/env python3
"""Local GPU dispatch — step 2: build & run on the reserved host (Azure-style).

After ``reserve_and_login.py`` has reserved a machine and verified SSH, this
module reproduces the Azure pipeline's execution on that host:

  1. SSH in (user + LOCAL_SSH_PASS).
  2. ``git clone`` (or update) the lb_eval repo — same GIT_REPO/GIT_BRANCH the
     Azure pipeline checks out. Keys live in the repo's ``auto_quant/config.env``.
  3. ``docker build`` the agent image from ``.azure-pipelines/docker/agent.dockerfile``
     (identical deps to Azure; layer-cached so only the first build is slow).
  4. Write a ``request.json`` carrying the reserved GPUs as ``cuda_visible_devices``.
  5. ``docker run`` the image with those GPUs, overriding the entrypoint to run
     ``bash auto.sh <request.json>`` directly (skipping Azure-agent registration).

Unlike Azure's RunPod path, we do NOT register a self-hosted agent — we drive the
container directly over SSH. Logs stream back to the caller.

Design choices:
  * GPU isolation via ``--gpus '"device=<reserved ids>"'``; inside the container the
    cards re-index to 0..N-1, so request.json's ``cuda_visible_devices`` is "0,1,...".
  * HF cache mounted from a persistent host dir to avoid re-downloading across runs.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shlex
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

# Defaults mirror auto_quant/config.env so a bare clone "just works".
DEFAULT_GIT_REPO = "https://github.com/XuehaoSun/lb_eval.git"
DEFAULT_GIT_BRANCH = "main"
DEFAULT_IMAGE = "cuda-openclaw:local"
DEFAULT_SCRIPTS_PATH = "auto_quant"       # dir holding auto.sh + config.env
DEFAULT_REMOTE_ROOT = "~/lb_eval_dispatch"
DEFAULT_HF_CACHE = "~/hf_cache"
DEFAULT_RESULTS_DATASET = os.environ.get("LOCAL_RESULTS_DATASET", "lvkaokao/lb_local")
LOCAL_UPLOADER = Path(__file__).resolve().parent / "upload_results_hf_dataset.py"


@dataclass
class RemoteJob:
    host: str
    gpu_ids: list[int]
    model: str
    scheme: str
    ssh_user: str = "root"
    ssh_port: int = 22
    method: str = "RTN"
    export_format: str = "auto_round"
    private: bool = False
    git_repo: str = DEFAULT_GIT_REPO
    git_branch: str = DEFAULT_GIT_BRANCH
    image: str = DEFAULT_IMAGE
    scripts_path: str = DEFAULT_SCRIPTS_PATH
    remote_root: str = DEFAULT_REMOTE_ROOT
    hf_cache: str = DEFAULT_HF_CACHE
    run_id: str = ""                       # unique per-run work dir (set in run_remote)
    request_json: dict | None = None      # explicit request; else generated
    results_dataset: str = DEFAULT_RESULTS_DATASET
    machine_profile: str = ""
    # Proxy for the remote host (Intel machines need one for external access).
    http_proxy: str = ""
    https_proxy: str = ""
    no_proxy: str = "localhost,127.0.0.1,::1"
    # Secrets injected into config.env inside the container (Azure "Update Config").
    # Keys: HF_TOKENS, MINIMAX_API_KEY, GIT_TOKEN, LB_STORAGE_BLOB_TOKEN.
    secrets: dict | None = None
    openclaw_setup: bool = True     # replicate Azure "Set Up OpenClaw" in-container


# The three Azure pre-auto.sh steps, reproduced INSIDE the container. Uses only
# runtime env vars (from docker --env-file / -e) so it can be a static string:
#   REQ_REL         request json path relative to the repo (via -e)
#   MINIMAX_API_KEY/HF_TOKENS/GIT_TOKEN/LB_STORAGE_BLOB_TOKEN   secrets (via --env-file)
#   http_proxy/https_proxy   proxy (via -e)
#   DO_OPENCLAW     "1" to run the OpenClaw provisioning step
CONTAINER_BOOTSTRAP = r"""#!/usr/bin/env bash
set -uo pipefail
cd /workspace/lb_eval

if [ "${DO_OPENCLAW:-1}" = "1" ]; then
    echo "== container: Set Up OpenClaw (Azure parity) =="
    uv pip install openclaw==2026.3.20 || true
    uv pip install -U huggingface_hub || true
    cp -r openclaw_config/ /root/.openclaw/
    python .azure-pipelines/scripts/sync_minimax_key.py \
        --token="${MINIMAX_API_KEY:-}" \
        --path=/root/.openclaw/agents/main/agent/auth-profiles.json || true
fi

echo "== container: Update Config (inject secrets + proxy into config.env) =="
SETS=()
[ -n "${HF_TOKENS:-}" ]            && SETS+=(--set "HF_TOKENS=${HF_TOKENS}")
[ -n "${MINIMAX_API_KEY:-}" ]      && SETS+=(--set "MINIMAX_API_KEY=${MINIMAX_API_KEY}")
[ -n "${GIT_TOKEN:-}" ]            && SETS+=(--set "GIT_TOKEN=${GIT_TOKEN}")
[ -n "${LB_STORAGE_BLOB_TOKEN:-}" ] && SETS+=(--set "LB_STORAGE_BLOB_TOKEN=${LB_STORAGE_BLOB_TOKEN}")
# Local path needs a REAL proxy (opposite of Azure which clears it).
[ -n "${http_proxy:-}" ]  && SETS+=(--set "HTTP_PROXY=${http_proxy}")
[ -n "${https_proxy:-}" ] && SETS+=(--set "HTTPS_PROXY=${https_proxy}")
if [ "${#SETS[@]}" -gt 0 ]; then
    python .azure-pipelines/scripts/update_config_env.py \
        --output /workspace/lb_eval/auto_quant/config.env "${SETS[@]}"
fi

echo "== container: run auto.sh (local upload mode) =="
cd auto_quant

# The built-in failure analysis always requests GitHub/community publication.
# Temporarily disable that hook; after auto.sh we run it locally without push flags.
ANALYZE="error_analysis/analyze_failures.py"
ANALYZE_DISABLED="error_analysis/analyze_failures.py.local-disabled"
if [ -f "$ANALYZE" ]; then
    mv "$ANALYZE" "$ANALYZE_DISABLED"
fi

pipeline_rc=0
bash auto.sh /workspace/lb_eval/"${REQ_REL}" --skip-upload || pipeline_rc=$?

if [ -f "$ANALYZE_DISABLED" ]; then
    mv "$ANALYZE_DISABLED" "$ANALYZE"
fi

# Every dispatch uses a fresh clone, so the newest output/runs child is this run.
RUN_DIR="$(find output/runs -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
    2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "[local-dispatch] ERROR: could not locate RUN_OUTPUT_DIR"
    exit "${pipeline_rc:-2}"
fi

# auto.sh sources config.env only in its subprocess. Load it here for upload_model_hf.
set -a
source config.env
set +a

# Preserve the existing behaviour for quantized weights: successful runs still
# upload to a Hugging Face model repository.
if [ "$pipeline_rc" -eq 0 ] && [ -d "$RUN_DIR/quantized_model" ]; then
    echo "== container: upload quantized model to Hugging Face model repo =="
    HF_REPO_NAME="$(basename "$RUN_DIR")"
    python upload_model_hf.py \
        "$RUN_DIR/quantized_model" \
        "$HF_REPO_NAME" \
        --tokens "${HF_TOKENS:-}" \
        --orgs "${HF_UPLOAD_ORGS:-}" \
        --account-ids "${HF_ACCOUNT_IDS:-}" \
        --summary-json "$RUN_DIR/quant_summary.json" \
        --accuracy-json "$RUN_DIR/accuracy.json" \
        --usage-file "${HF_USAGE_FILE:-}" \
        --capacity-gb "${HF_ACCOUNT_CAPACITY_GB:-1000}" \
        --shared-ledger-enabled "${HF_SHARED_LEDGER_ENABLED:-false}" \
        --shared-ledger-repo "${HF_SHARED_LEDGER_REPO:-}" \
        --shared-ledger-token "${HF_SHARED_LEDGER_TOKEN:-}" \
        --shared-ledger-branch "${HF_SHARED_LEDGER_BRANCH:-main}" \
        2>&1 | tee "$RUN_DIR/logs/upload_hf.log" || \
        echo "[local-dispatch] WARNING: model upload failed"
fi

# Keep root-cause analysis as a local artifact, but do not publish to the online
# leaderboard GitHub repository or community.
if [ "$pipeline_rc" -ne 0 ] && [ -f "$ANALYZE" ]; then
    echo "== container: local failure analysis (no GitHub/community push) =="
    python "$ANALYZE" --run-dir "$RUN_DIR" --limit 1 \
        2>&1 | tee "$RUN_DIR/logs/error_analysis.log" || \
        echo "[local-dispatch] WARNING: local error analysis failed"
fi

# Bundle this run's lessons into the run dir so they travel to the dataset.
if ls auto_quant/lessons/*.jsonl >/dev/null 2>&1; then
    mkdir -p "$RUN_DIR/lessons"
    cp -f auto_quant/lessons/*.jsonl "$RUN_DIR/lessons/" 2>/dev/null || true
fi

echo "== container: upload results to HF dataset ${LOCAL_RESULTS_DATASET} =="
dataset_rc=0
python /workspace/lb_eval/_local_upload_results.py \
    "$RUN_DIR" \
    --dataset "${LOCAL_RESULTS_DATASET}" \
    --run-id "${LOCAL_RUN_ID}" \
    --pipeline-rc "$pipeline_rc" || dataset_rc=$?

# Quant/eval failure is primary. If it passed, surface dataset-upload failure.
if [ "$pipeline_rc" -ne 0 ]; then
    exit "$pipeline_rc"
fi
exit "$dataset_rc"
"""


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(" ", "_")


# scheme → (precision, weight_dtype) for the leaderboard request-filename convention.
_SCHEME_NAMING = {
    "W4A16": ("4bit", "int4"),
    "INT4 (W4A16)": ("4bit", "int4"),
    "MXFP4": ("4bit", "mxfp4"),
    "NVFP4": ("4bit", "nvfp4"),
    "W8A16": ("8bit", "int8"),
    "INT8 (W8A16)": ("8bit", "int8"),
    "MXFP8": ("8bit", "mxfp8"),
}


def leaderboard_request_filename(model: str, scheme: str, method: str, private: bool) -> str:
    """Build the request/status filename EXACTLY as the leaderboard names it, so the
    upload's status write-back (rglob under status/) matches.

    Format: ``<model_short>_quant_request_<Private>_<SCHEME>_<precision>_<weight_dtype>[_<METHOD>].json``
    RTN carries no method suffix; TUNING/MODEL_FREE do.
    """
    model_short = model.split("/", 1)[-1] if "/" in model else model
    scheme_key = scheme.strip()
    precision, wdtype = _SCHEME_NAMING.get(scheme_key, ("4bit", "int4"))
    scheme_tag = scheme_key.replace(" ", "").replace("(", "").replace(")", "")
    # normalise "INT4W4A16" → "W4A16"
    for canon in ("W4A16", "MXFP4", "NVFP4", "W8A16", "MXFP8"):
        if canon in scheme_tag:
            scheme_tag = canon
            break
    parts = [model_short, "quant_request", "True" if private else "False",
             scheme_tag, precision, wdtype]
    m = (method or "RTN").strip().upper()
    if m in ("TUNING", "MODEL_FREE", "MODELFREE"):
        parts.append("MODEL_FREE" if m in ("MODEL_FREE", "MODELFREE") else "TUNING")
    return "_".join(parts) + ".json"


def build_request_json(job: RemoteJob) -> tuple[str, dict]:
    """Return (filename, request_dict). GPUs are re-indexed to 0..N-1 for the container.

    The filename follows the leaderboard convention so status write-back can match
    (or create) the corresponding status/<org>/<filename> entry.
    """
    if job.request_json:
        req = dict(job.request_json)
        # Honour an explicit request's own filename if it carries one.
        fname = req.get("request_filename") or leaderboard_request_filename(
            job.model, job.scheme, job.method, job.private)
    else:
        req = {
            "model": job.model,
            "quant_scheme": job.scheme,
            "method": job.method,
            "export_format": job.export_format,
            "private": job.private,
            "script": "auto_quant",
            "model_type": "quantization",
            "status": "Pending",
        }
        fname = leaderboard_request_filename(job.model, job.scheme, job.method, job.private)
    n = len(job.gpu_ids)
    # Inside the container docker exposes only the reserved cards, re-indexed 0..N-1.
    req["cuda_visible_devices"] = ",".join(str(i) for i in range(n)) if n else ""
    req["request_filename"] = fname
    return fname, req


def _proxy_pairs(job: RemoteJob) -> list[tuple[str, str]]:
    """(key, value) proxy pairs (both lower/upper case) if a proxy is configured."""
    # Both multi-hop B200 hosts use direct GitHub/Hugging Face/Docker access.
    # Supplying the inherited Intel proxy is unnecessary (and breaks changwa1).
    if job.machine_profile in ("sc09-b200", "changwa1-b200"):
        return []
    pairs: list[tuple[str, str]] = []
    if job.http_proxy:
        pairs += [("http_proxy", job.http_proxy), ("HTTP_PROXY", job.http_proxy)]
    if job.https_proxy:
        pairs += [("https_proxy", job.https_proxy), ("HTTPS_PROXY", job.https_proxy)]
    if (job.http_proxy or job.https_proxy) and job.no_proxy:
        pairs += [("no_proxy", job.no_proxy), ("NO_PROXY", job.no_proxy)]
    return pairs


def build_remote_script(job: RemoteJob, req_rel_path: str, has_secrets: bool) -> str:
    """Bash executed on the remote host: clone/update → build → docker run auto.sh.

    All per-run state lives under an isolated ``$ROOT/runs/<run_id>`` directory so
    multiple dispatches to the SAME machine never share a git working tree, staged
    files, config.env, or output. Only the HF cache and the docker image tag are
    shared (both are safe to share).
    """
    device_list = ",".join(str(g) for g in job.gpu_ids)
    pairs = _proxy_pairs(job)
    # Layer 1: export for the remote shell (git clone).
    proxy_shell = "".join(f'export {k}="{v}"\n' for k, v in pairs)
    # Layer 2: docker build-args (build-time apt/npm/curl inside RUN steps).
    build_proxy = "".join(f'    --build-arg {k}="{v}" \\\n' for k, v in pairs)
    # Layer 3: docker run -e (container runtime: HF/pip/github).
    run_proxy = "".join(f'    -e {k}="{v}" \\\n' for k, v in pairs)
    # Secrets go via a locked-down env-file (keeps them out of `ps`/the main script).
    env_file_arg = '    --env-file "$WORK/_secrets.env" \\\n' if has_secrets else ""
    do_openclaw = "1" if job.openclaw_setup else "0"
    return f"""#!/usr/bin/env bash
set -euo pipefail

ROOT=$(eval echo "{job.remote_root}")
RUN_ID="{job.run_id}"
WORK="$ROOT/runs/$RUN_ID"          # isolated per-run working dir
REPO="$WORK/lb_eval"
BRANCH="{job.git_branch}"
GIT_REPO="{job.git_repo}"
IMAGE="{job.image}"
SCRIPTS_PATH="{job.scripts_path}"
HF_CACHE_ABS=$(eval echo "{job.hf_cache}")   # shared across runs (safe)
REQ_REL="{req_rel_path}"
DEVICE_LIST="{device_list}"
CONTAINER="lb-dispatch-$RUN_ID"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

# ── proxy (Intel remote hosts need this for external access) ──
{proxy_shell}
mkdir -p "$WORK" "$HF_CACHE_ABS"

echo "== [1/4] clone repo (isolated: $REPO) =="
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO" checkout "$BRANCH"
    git -C "$REPO" reset --hard "origin/$BRANCH"
else
    rm -rf "$REPO"
    git clone --depth 1 -b "$BRANCH" "$GIT_REPO" "$REPO"
fi
cd "$REPO"

echo "== [2/4] docker build (shared image tag, layer-cached) =="
docker build \\
{build_proxy}    -f .azure-pipelines/docker/agent.dockerfile -t "$IMAGE" .azure-pipelines/docker

echo "== [3/4] stage request.json + container bootstrap =="
mkdir -p "$(dirname "$REPO/$REQ_REL")"
cp "$WORK/_request.json" "$REPO/$REQ_REL"
cp "$WORK/_container_bootstrap.sh" "$REPO/_container_bootstrap.sh"
cp "$WORK/_local_upload_results.py" "$REPO/_local_upload_results.py"
chmod +x "$REPO/_container_bootstrap.sh"

echo "== [4/4] docker run: Set Up OpenClaw + Update Config + auto.sh (GPUs device=$DEVICE_LIST) =="
# scrub secrets on any exit (success, failure, or interrupt)
trap 'rm -f "$WORK/_secrets.env"' EXIT
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
rc=0
docker run --rm --name "$CONTAINER" \\
    --gpus '"device='"$DEVICE_LIST"'"' \\
    --shm-size=16g \\
{env_file_arg}{run_proxy}    -e REQ_REL="$REQ_REL" \\
    -e DO_OPENCLAW="{do_openclaw}" \\
    -e LOCAL_RESULTS_DATASET="{job.results_dataset}" \\
    -e LOCAL_RUN_ID="$RUN_ID" \\
    -v "$REPO":/workspace/lb_eval \\
    -v "$HF_CACHE_ABS":/root/.cache/huggingface \\
    --entrypoint bash \\
    "$IMAGE" \\
    /workspace/lb_eval/_container_bootstrap.sh || rc=$?

# The container runs as root and writes into the bind-mounted checkout. Restore
# ownership so the SSH user can inspect or clean completed per-run directories.
docker run --rm \\
    -v "$WORK":/work \\
    --entrypoint sh \\
    ubuntu:24.04 \\
    -c "chown -R $HOST_UID:$HOST_GID /work" >/dev/null 2>&1 || true

exit $rc
"""


def _make_run_id(job: RemoteJob) -> str:
    """Unique, filesystem-safe per-run id: <model>_<scheme>_<method>_<ts>_<rand>."""
    import time as _t
    import uuid as _u
    base = f"{job.model.split('/')[-1]}_{job.scheme}_{job.method}".replace(" ", "")
    base = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in base)
    return f"{base}_{_t.strftime('%Y%m%d-%H%M%S')}_{_u.uuid4().hex[:6]}"


def run_remote(job: RemoteJob, password: str, dry_run: bool = False,
               timeout: int = 30) -> dict:
    """Execute the remote build+run over SSH, streaming output. Returns a summary."""
    if not job.run_id:
        job.run_id = _make_run_id(job)
    fname, req = build_request_json(job)
    req_rel = f"pending_requests/local/{fname}"
    secrets = _collect_secrets(job)
    script = build_remote_script(job, req_rel, has_secrets=bool(secrets))

    result: dict = {
        "host": job.host, "gpu_ids": job.gpu_ids, "model": job.model,
        "run_id": job.run_id, "request_rel": req_rel, "request": req,
        "secrets_present": sorted(secrets.keys()),
    }

    if dry_run:
        result["dry_run"] = True
        result["script"] = script
        result["bootstrap"] = CONTAINER_BOOTSTRAP
        result["request_json_text"] = json.dumps(req, ensure_ascii=False, indent=2)
        return result

    if job.machine_profile:
        if job.machine_profile == "sc09-b200":
            return _run_sc09(job, req, script, secrets, result, timeout)
        if job.machine_profile == "changwa1-b200":
            return _run_changwa(job, req, script, secrets, result, timeout)
        raise RuntimeError(
            f"Machine profile {job.machine_profile} is not execution-enabled"
        )

    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=job.host, port=job.ssh_port, username=job.ssh_user,
                   password=password, timeout=timeout, allow_agent=False, look_for_keys=False)
    try:
        # Stage everything under the ISOLATED per-run work dir ($ROOT/runs/<run_id>)
        # so concurrent dispatches to the same machine never collide. Files go
        # OUTSIDE the repo subdir so a fresh git clone into <work>/lb_eval isn't blocked.
        remote_root = _expand_remote(client, job.remote_root)
        work = f"{remote_root}/runs/{job.run_id}"
        sftp = client.open_sftp()
        _sftp_makedirs(sftp, work)
        with sftp.open(f"{work}/_dispatch_run.sh", "w") as f:
            f.write(script)
        with sftp.open(f"{work}/_request.json", "w") as f:
            f.write(json.dumps(req, ensure_ascii=False, indent=2))
        with sftp.open(f"{work}/_container_bootstrap.sh", "w") as f:
            f.write(CONTAINER_BOOTSTRAP)
        with sftp.open(f"{work}/_local_upload_results.py", "w") as f:
            f.write(LOCAL_UPLOADER.read_text(encoding="utf-8"))
        if secrets:
            secrets_path = f"{work}/_secrets.env"
            with sftp.open(secrets_path, "w") as f:
                f.write("".join(f"{k}={v}\n" for k, v in secrets.items()))
            sftp.chmod(secrets_path, 0o600)
        sftp.close()

        exit_code = _exec_stream(client, f"bash {shlex.quote(work)}/_dispatch_run.sh")
        result["exit_code"] = exit_code
        result["ok"] = exit_code == 0
        return result
    finally:
        client.close()


def _staged_files(req: dict, script: str, secrets: dict) -> dict[str, tuple[bytes, int]]:
    files: dict[str, tuple[bytes, int]] = {
        "_dispatch_run.sh": (script.encode("utf-8"), 0o700),
        "_request.json": (
            json.dumps(req, ensure_ascii=False, indent=2).encode("utf-8"),
            0o600,
        ),
        "_container_bootstrap.sh": (CONTAINER_BOOTSTRAP.encode("utf-8"), 0o700),
        "_local_upload_results.py": (
            LOCAL_UPLOADER.read_bytes(),
            0o600,
        ),
    }
    if secrets:
        files["_secrets.env"] = (
            "".join(f"{k}={v}\n" for k, v in secrets.items()).encode("utf-8"),
            0o600,
        )
    return files


def _run_on_paramiko_client(
    client,
    job: RemoteJob,
    req: dict,
    script: str,
    secrets: dict,
    result: dict,
) -> dict:
    remote_root = _expand_remote(client, job.remote_root)
    work = f"{remote_root}/runs/{job.run_id}"
    files = _staged_files(req, script, secrets)
    sftp = client.open_sftp()
    try:
        _sftp_makedirs(sftp, work)
        for name, (content, mode) in files.items():
            path = f"{work}/{name}"
            with sftp.open(path, "wb") as f:
                f.write(content)
            sftp.chmod(path, mode)
    finally:
        sftp.close()
    exit_code = _exec_stream(client, f"bash {shlex.quote(work)}/_dispatch_run.sh")
    result["exit_code"] = exit_code
    result["ok"] = exit_code == 0
    return result


def _connect_jump(timeout: int):
    import paramiko
    from machine_profiles import JUMP_HOST, JUMP_PASSWORD_ENV, JUMP_USER

    password = os.environ.get(JUMP_PASSWORD_ENV) or os.environ.get("LOCAL_SSH_PASS")
    if not password:
        raise RuntimeError(f"Missing {JUMP_PASSWORD_ENV}")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=os.environ.get("LOCAL_JUMP_SSH_HOST", JUMP_HOST),
        port=int(os.environ.get("LOCAL_JUMP_SSH_PORT", "22")),
        username=os.environ.get("LOCAL_JUMP_SSH_USER", JUMP_USER),
        password=password,
        timeout=timeout,
        auth_timeout=timeout,
        banner_timeout=timeout,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def _run_sc09(
    job: RemoteJob,
    req: dict,
    script: str,
    secrets: dict,
    result: dict,
    timeout: int,
) -> dict:
    """Jump Paramiko -> direct-tcpip -> target Paramiko (password auth)."""
    import paramiko

    target_password = os.environ.get("SC09_SSH_PASS")
    if not target_password:
        raise RuntimeError("Missing SC09_SSH_PASS")
    jump = _connect_jump(timeout)
    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    channel = None
    try:
        transport = jump.get_transport()
        if transport is None:
            raise RuntimeError("Jump-host transport is unavailable")
        channel = transport.open_channel(
            "direct-tcpip",
            ("172.26.46.180", 22),
            ("127.0.0.1", 0),
            timeout=timeout,
        )
        target.connect(
            hostname="172.26.46.180",
            username="hshen",
            password=target_password,
            sock=channel,
            timeout=timeout,
            auth_timeout=timeout,
            banner_timeout=timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        return _run_on_paramiko_client(target, job, req, script, secrets, result)
    finally:
        target.close()
        if channel is not None:
            channel.close()
        jump.close()


def _target_work_assignment(job: RemoteJob) -> str:
    root = job.remote_root
    suffix = f"/runs/{job.run_id}"
    if root == "~":
        return f'WORK="$HOME{suffix}"'
    if root.startswith("~/"):
        return f'WORK="$HOME/{root[2:]}{suffix}"'
    return f"WORK={shlex.quote(root + suffix)}"


def _nested_command(profile_name: str, remote_command: str) -> str:
    from machine_profiles import get_profile
    from multi_hop_ssh import nested_ssh_argv

    profile = get_profile(profile_name)
    return " ".join(
        [*(shlex.quote(arg) for arg in nested_ssh_argv(profile)),
         shlex.quote(remote_command)]
    )


def _tar_bytes(files: dict[str, tuple[bytes, int]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, (content, mode) in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _exec_with_input(client, command: str, payload: bytes) -> tuple[int, str]:
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport is unavailable")
    chan = transport.open_session()
    chan.exec_command(command)
    chan.sendall(payload)
    chan.shutdown_write()
    output = bytearray()
    while True:
        if chan.recv_ready():
            output.extend(chan.recv(4096))
        if chan.recv_stderr_ready():
            output.extend(chan.recv_stderr(4096))
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        time.sleep(0.05)
    return chan.recv_exit_status(), output.decode("utf-8", "replace")


def _run_changwa(
    job: RemoteJob,
    req: dict,
    script: str,
    secrets: dict,
    result: dict,
    timeout: int,
) -> dict:
    """Use jump-host OpenSSH alias/key; stage files as a tar stream over SSH."""
    jump = _connect_jump(timeout)
    try:
        files = _staged_files(req, script, secrets)
        work_assignment = _target_work_assignment(job)
        stage_remote = (
            f'{work_assignment}; mkdir -p "$WORK"; tar -xzf - -C "$WORK"'
        )
        rc, output = _exec_with_input(
            jump,
            _nested_command(job.machine_profile, stage_remote),
            _tar_bytes(files),
        )
        if rc != 0:
            raise RuntimeError(f"Failed to stage files on changwa1-b200: {output.strip()}")
        run_remote_command = (
            f'{work_assignment}; bash "$WORK/_dispatch_run.sh"'
        )
        exit_code = _exec_stream(
            jump,
            _nested_command(job.machine_profile, run_remote_command),
        )
        result["exit_code"] = exit_code
        result["ok"] = exit_code == 0
        return result
    finally:
        jump.close()


def _collect_secrets(job: RemoteJob) -> dict:
    """Secrets to inject into config.env. From job.secrets, else the dispatcher env.

    Only non-empty values are included so we never blank an existing config.env key.
    """
    if job.secrets is not None:
        src = job.secrets
    else:
        src = {
            "HF_TOKENS": os.environ.get("HF_TOKENS") or os.environ.get("HUGGINGFACE_TOKEN")
                         or os.environ.get("HF_TOKEN") or "",
            "MINIMAX_API_KEY": os.environ.get("MINIMAX_API_KEY") or os.environ.get("MINIMAX_KEY") or "",
            "GIT_TOKEN": os.environ.get("GIT_TOKEN") or "",
            "LB_STORAGE_BLOB_TOKEN": os.environ.get("LB_STORAGE_BLOB_TOKEN") or "",
            "LOCAL_RESULTS_HF_TOKEN": os.environ.get("LOCAL_RESULTS_HF_TOKEN") or "",
        }
    return {k: v for k, v in src.items() if v}


def _expand_remote(client, path: str) -> str:
    if not path.startswith("~"):
        return path
    # NOTE: do NOT shlex.quote the tilde — quoting suppresses ~ expansion, and
    # sftp does not expand ~ either. Resolve $HOME explicitly instead.
    _, out, _ = client.exec_command('echo "$HOME"')
    home = out.read().decode().strip() or "/root"
    return home + path[1:]


def _sftp_makedirs(sftp, path: str) -> None:
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass


def _exec_stream(client, command: str) -> int:
    """Run a command, streaming stdout+stderr live. Returns the exit status."""
    transport = client.get_transport()
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command(command)
    while True:
        if chan.recv_ready():
            sys.stdout.write(chan.recv(4096).decode("utf-8", "replace"))
            sys.stdout.flush()
        if chan.exit_status_ready() and not chan.recv_ready():
            break
        time.sleep(0.1)
    # drain
    while chan.recv_ready():
        sys.stdout.write(chan.recv(4096).decode("utf-8", "replace"))
    sys.stdout.flush()
    return chan.recv_exit_status()


def main() -> int:
    p = argparse.ArgumentParser(description="Local GPU dispatch step 2: build & run on reserved host")
    p.add_argument("--host", required=True, help="reserved machine host/IP")
    p.add_argument("--gpus", required=True, help="reserved physical device ids, e.g. 0,1")
    p.add_argument("--model", required=True, help="HuggingFace model id")
    p.add_argument("--scheme", default="W4A16")
    p.add_argument("--method", default="RTN", help="RTN | TUNING | MODEL_FREE")
    p.add_argument("--export-format", default="auto_round", help="auto_round | llm_compressor")
    p.add_argument("--ssh-user", default=os.environ.get("LOCAL_SSH_USER", "root"))
    p.add_argument("--ssh-port", type=int, default=int(os.environ.get("LOCAL_SSH_PORT", "22")))
    p.add_argument("--git-repo", default=DEFAULT_GIT_REPO)
    p.add_argument("--git-branch", default=DEFAULT_GIT_BRANCH)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    p.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    p.add_argument("--results-dataset", default=DEFAULT_RESULTS_DATASET)
    p.add_argument("--machine-profile", default="")
    p.add_argument("--http-proxy", default=os.environ.get("LOCAL_HTTP_PROXY", os.environ.get("HTTP_PROXY", "")))
    p.add_argument("--https-proxy", default=os.environ.get("LOCAL_HTTPS_PROXY", os.environ.get("HTTPS_PROXY", "")))
    p.add_argument("--no-proxy", default=os.environ.get("LOCAL_NO_PROXY", "localhost,127.0.0.1,::1"))
    p.add_argument("--no-openclaw-setup", action="store_true",
                   help="skip the in-container OpenClaw provisioning (use if AGENT_BACKEND!=openclaw)")
    p.add_argument("--request-json", default=None, help="path to an existing request.json to use")
    p.add_argument("--dry-run", action="store_true", help="print the remote script + request.json, do not run")
    args = p.parse_args()

    req = None
    if args.request_json:
        with open(args.request_json) as f:
            req = json.load(f)

    job = RemoteJob(
        host=args.host, gpu_ids=[int(x) for x in args.gpus.split(",") if x.strip() != ""],
        model=args.model, scheme=args.scheme, ssh_user=args.ssh_user, ssh_port=args.ssh_port,
        method=args.method, export_format=args.export_format,
        git_repo=args.git_repo, git_branch=args.git_branch, image=args.image,
        remote_root=args.remote_root, hf_cache=args.hf_cache, request_json=req,
        results_dataset=args.results_dataset,
        machine_profile=args.machine_profile,
        http_proxy=args.http_proxy, https_proxy=args.https_proxy, no_proxy=args.no_proxy,
        openclaw_setup=not args.no_openclaw_setup,
    )

    if args.dry_run:
        res = run_remote(job, password="", dry_run=True)
        print("# ===== request.json (pending_requests/local/) =====")
        print(res["request_json_text"])
        print(f"# secrets to inject into config.env: {res['secrets_present'] or '(none — set HF_TOKENS/MINIMAX_API_KEY/GIT_TOKEN…)'}")
        print(f"\n# ===== container bootstrap (runs before auto.sh) =====")
        print(res["bootstrap"])
        print(f"\n# ===== remote script ({res['request_rel']}) =====")
        print(res["script"])
        return 0

    password = os.environ.get("LOCAL_SSH_PASS")
    if not password:
        print("ERROR: LOCAL_SSH_PASS not set", file=sys.stderr)
        return 1
    res = run_remote(job, password=password)
    print(f"\n=== done: exit_code={res.get('exit_code')} ok={res.get('ok')} ===")
    return 0 if res.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
