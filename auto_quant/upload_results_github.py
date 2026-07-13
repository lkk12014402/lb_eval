#!/usr/bin/env python3
"""
Upload auto_quant result artifacts into the lb_eval GitHub repository.

Artifacts copied from the runtime output directory:
- quant_summary.json / summary.json
- accuracy.json
- lm_eval_results/
- quantize.py
- evaluate.sh
- logs/
- session_*.jsonl
- session_*.md

Layout:
results/<org>/<artifact_name>/results_<timestamp>.json
results/<org>/<artifact_name>/run_<timestamp>/...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

# Patterns that match common secret/token formats
_SECRET_PATTERNS = [
    # HuggingFace tokens: hf_xxxx (20+ alphanumeric chars)
    re.compile(r'\bhf_[a-zA-Z0-9]{20,}\b'),
    # GitHub PATs: ghp_, gho_, ghs_, ghu_, github_pat_
    re.compile(r'\b(?:ghp_|gho_|ghs_|ghu_)[a-zA-Z0-9]{36,}\b'),
    re.compile(r'\bgithub_pat_[a-zA-Z0-9_]{22,}\b'),
    # Git URL-embedded credentials: https://x-access-token:TOKEN@github.com/...
    re.compile(r'(https?://)([^\s@]+)(@(?:github\.com|huggingface\.co))'),
    # --token <value> or token=<value> on CLI (e.g. huggingface-cli download --token xxx)
    re.compile(r'(--token\s+)[a-zA-Z0-9_.-]{20,}', re.IGNORECASE),
    re.compile(r'(token=)[a-zA-Z0-9_.-]{20,}'),
    # Generic Bearer tokens in headers
    re.compile(r'(Bearer\s+)[a-zA-Z0-9_.~+/=-]{20,}', re.IGNORECASE),
    # Azure DevOps PATs (52-char base64)
    re.compile(r'\b[a-z0-9]{52}\b(?=.*(?:azuredevops|_work|pipeline))', re.IGNORECASE),
    # Weights & Biases tokens
    re.compile(r'\bwandb_[a-zA-Z0-9]{20,}\b'),
]


def sanitize_secrets(text: str) -> str:
    """Replace known secret/token patterns with [REDACTED].

    For patterns with capturing groups (prefix), preserves the prefix and only
    redacts the secret portion. For patterns without groups, replaces the whole match.
    """
    for pattern in _SECRET_PATTERNS:
        def _redact(m: re.Match, _p=pattern) -> str:
            if m.lastindex and m.lastindex >= 3:
                # URL pattern: group(1)=scheme, group(2)=creds, group(3)=@host
                return m.group(1) + "[REDACTED]" + m.group(3)
            if m.lastindex and m.lastindex >= 1:
                # Prefix patterns (Bearer, --token, token=)
                return m.group(1) + "[REDACTED]"
            return "[REDACTED]"
        text = pattern.sub(_redact, text)
    return text


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_timestamp() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def looks_like_quantized_artifact(model_short: str) -> bool:
    lowered = model_short.lower()
    markers = (
        "-autoround-",
        "-gptq",
        "-awq",
        ".gguf",
        "-gguf",
        "llm-compressor",
    )
    return any(marker in lowered for marker in markers)


def run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=check,
    )


def clone_repo(repo_url: str, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", repo_url, str(target_dir)],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git clone failed")


def ensure_git_config(repo_dir: Path, user_name: str, user_email: str) -> None:
    run_git(["config", "user.name", user_name], repo_dir)
    run_git(["config", "user.email", user_email], repo_dir)


def build_auth_url(repo_url: str, token: str | None) -> str:
    if not token:
        return repo_url
    parts = urlsplit(repo_url)
    if parts.scheme != "https":
        return repo_url
    netloc = f"x-access-token:{token}@{parts.netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def copy_file(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def copy_file_sanitized(src: Path, dst: Path, copied: list[str]) -> None:
    """Copy a text file, stripping secrets from its content."""
    if not src.exists() or not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = src.read_text(encoding="utf-8", errors="replace")
        sanitized = sanitize_secrets(content)
        dst.write_text(sanitized, encoding="utf-8")
        if content != sanitized:
            print(f"[github-upload] Sanitized secrets in {src.name}")
        copied.append(str(dst))
    except Exception as exc:
        print(f"[github-upload] WARNING: failed to sanitize {src.name}, skipping: {exc}")


def copy_tree(src: Path, dst: Path, copied: list[str]) -> None:
    if not src.exists() or not src.is_dir():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    copied.append(str(dst))


def generate_failure_digest(run_dir: Path, quant_summary: dict | None, accuracy: dict | None) -> None:
    """Generate a concise failure_digest.txt summarizing why the pipeline failed.

    Extracts key error information from summary JSONs, diagnosis files, and exec logs
    into a ≤50 line human-readable digest.
    """
    status = derive_pipeline_status(quant_summary, accuracy)
    if status == "Finished":
        return

    lines: list[str] = []
    lines.append(f"=== FAILURE DIGEST ===")
    lines.append(f"Status: {status}")
    lines.append("")

    # Extract from diagnosis JSON if available
    for diag_file in sorted(run_dir.glob("failure_diagnosis_*.json")):
        try:
            with diag_file.open(encoding="utf-8") as f:
                diag = json.load(f)
            stage = diag_file.stem.replace("failure_diagnosis_", "")
            lines.append(f"--- Diagnosis ({stage}) ---")
            lines.append(f"Category: {diag.get('failure_category', 'unknown')}")
            lines.append(f"Root cause: {diag.get('root_cause', 'N/A')}")
            lines.append(f"Retryable: {diag.get('retryable', 'unknown')}")
            lines.append(f"Fix: {diag.get('suggested_fix', 'N/A')}")
            key_error = diag.get("key_error", "")
            if key_error:
                lines.append(f"Key error: {key_error[:300]}")
            lines.append("")
        except Exception:
            pass

    # Extract errors from summary JSONs
    for label, summary in [("Quantization", quant_summary), ("Evaluation", accuracy)]:
        if not isinstance(summary, dict):
            continue
        s_status = summary.get("status")
        if s_status != "failed":
            continue
        errors = summary.get("errors", [])
        if errors:
            lines.append(f"--- {label} errors ---")
            for err in errors[:5]:
                lines.append(f"  • {str(err)[:200]}")
            lines.append("")

    # Extract last error lines from exec logs
    logs_dir = run_dir / "logs"
    for log_name in ("quant_exec.log", "eval_exec.log"):
        log_path = logs_dir / log_name
        if not log_path.is_file():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            log_lines = text.splitlines()
            # Find last traceback or error lines
            error_lines = []
            in_traceback = False
            for line in log_lines[-100:]:
                if "Traceback" in line or "Error" in line or "Exception" in line:
                    in_traceback = True
                if in_traceback:
                    error_lines.append(line)
                    if len(error_lines) > 15:
                        break
            if error_lines:
                lines.append(f"--- {log_name} (last error) ---")
                lines.extend(error_lines[:10])
                lines.append("")
        except Exception:
            pass

    if len(lines) <= 3:
        lines.append("No detailed error information available.")
        lines.append("Check session logs for more context.")

    digest_path = run_dir / "failure_digest.txt"
    digest_path.write_text("\n".join(lines[:50]) + "\n", encoding="utf-8")
    print(f"[github-upload] Generated {digest_path.name} ({len(lines)} lines)")


def derive_pipeline_status(quant_summary: dict | None, accuracy: dict | None) -> str:
    """Derive overall pipeline status from quant_summary and accuracy results.

    Returns one of: "Finished", "Quant Failed", "Eval Failed", "Partial".
    """
    qs_status = (quant_summary or {}).get("status", "missing")
    acc_status = (accuracy or {}).get("status", "missing")

    if qs_status == "failed":
        return "Quant Failed"
    if acc_status == "failed":
        return "Eval Failed"

    # Check for any task with acc=0 (indicates evaluation failure)
    if isinstance(accuracy, dict):
        tasks = accuracy.get("tasks")
        if isinstance(tasks, dict):
            for task_name, task_val in tasks.items():
                acc_value = task_val if not isinstance(task_val, dict) else task_val.get("accuracy")
                try:
                    if acc_value is not None and float(acc_value) == 0.0:
                        return "Eval Failed"
                except (TypeError, ValueError):
                    pass

    if qs_status == "success" and acc_status == "success":
        return "Finished"
    if acc_status == "success":
        return "Finished"
    if qs_status == "success" and acc_status == "partial":
        return "Partial"
    return "Partial"


def write_back_status(repo_dir: Path, request_filename: str, new_status: str,
                      copied: list[str]) -> None:
    """Update the status field in the matching request file under status/."""
    if not request_filename:
        print("[github-upload] No --request-filename provided; skipping status write-back")
        return

    status_dir = repo_dir / "status"
    if not status_dir.is_dir():
        print(f"[github-upload] status/ directory not found at {status_dir}; skipping write-back")
        return

    # Search for the request file under status/
    matches = list(status_dir.rglob(request_filename))
    if not matches:
        print(f"[github-upload] Request file '{request_filename}' not found under {status_dir}; skipping write-back")
        return

    for match_path in matches:
        try:
            with match_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            old_status = data.get("status", "unknown")
            data["status"] = new_status
            with match_path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=4, ensure_ascii=False)
                handle.write("\n")
            copied.append(str(match_path))
            print(f"[github-upload] Status write-back: {match_path.relative_to(repo_dir)} "
                  f"({old_status} -> {new_status})")
        except Exception as exc:
            print(f"[github-upload] WARNING: failed to write-back status for {match_path}: {exc}")


def detect_artifact_name(model_id: str, scheme: str, quant_summary: dict | None, method: str = "") -> str:
    if quant_summary:
        hf_repo = quant_summary.get("hf_repo")
        if isinstance(hf_repo, str) and hf_repo.strip():
            return hf_repo.rstrip("/").rsplit("/", 1)[-1]

    model_short = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    if looks_like_quantized_artifact(model_short):
        return sanitize_name(model_short)
    # Build name matching HF_REPO_NAME format: {model}-AutoRound-{SCHEME}-{METHOD_SUFFIX}
    method_upper = (method or "RTN").strip().upper()
    method_suffix = "Tuning" if method_upper == "TUNING" else "RTN"
    return sanitize_name(f"{model_short}-AutoRound-{scheme}-{method_suffix}")


def resolve_repo_dir(repo_dir_arg: str, clone_dir_arg: str, repo_url: str, token: str | None) -> Path:
    repo_dir_raw = (repo_dir_arg or "").strip()
    clone_dir_raw = (clone_dir_arg or "").strip()
    default_clone_dir = Path(__file__).resolve().parent / "lb_eval"

    if repo_dir_raw:
        repo_dir = Path(repo_dir_raw).resolve()
    else:
        repo_dir = Path(clone_dir_raw).resolve() if clone_dir_raw else default_clone_dir.resolve()

    if (repo_dir / ".git").exists():
        return repo_dir

    if not repo_url:
        raise RuntimeError("GIT_REPO is required when the local git repo is missing")

    # Clean up stale directory (failed clone, non-git leftover, etc.)
    if repo_dir.exists():
        print(f"[github-upload] Removing stale non-git directory: {repo_dir}")
        shutil.rmtree(repo_dir)

    auth_url = build_auth_url(repo_url, token)
    print(f"[github-upload] Cloning repo into: {repo_dir}")
    clone_repo(auth_url, repo_dir)
    if auth_url != repo_url:
        run_git(["remote", "set-url", "origin", repo_url], repo_dir)
    return repo_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload result artifacts to the lb_eval GitHub repo")
    parser.add_argument("runtime_output_dir", help="Directory containing quant/eval runtime artifacts")
    parser.add_argument("model_id", help="Original model id, e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--pipeline", default="", help="Pipeline label: auto_quant or auto_eval")
    parser.add_argument("--scheme", default="W4A16", help="Quantization scheme label")
    parser.add_argument("--method", default=os.environ.get("METHOD", "RTN"), help="Quantization method: RTN or TUNING")
    parser.add_argument("--quant-num-gpus", default="", help="Quantization GPU count")
    parser.add_argument("--eval-num-gpus", default="", help="Evaluation GPU count")
    parser.add_argument(
        "--model-output-dir",
        default="",
        help="Optional model directory associated with this runtime output",
    )
    parser.add_argument(
        "--repo-dir",
        default=os.environ.get(
            "GIT_RESULTS_REPO_DIR",
            "",
        ),
        help="Local clone of the lb_eval git repo. If empty or missing, the repo is cloned automatically.",
    )
    parser.add_argument(
        "--clone-dir",
        default=os.environ.get("GIT_RESULTS_CLONE_DIR") or str(Path(__file__).resolve().parent / "lb_eval"),
        help="Clone destination used when --repo-dir is empty or missing",
    )
    parser.add_argument(
        "--git-repo",
        default=os.environ.get("GIT_REPO", ""),
        help="Remote git repo URL used for authenticated pull/push when token is provided",
    )
    parser.add_argument(
        "--git-token",
        default=os.environ.get("GIT_TOKEN", ""),
        help="GitHub token used for authenticated push",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("GIT_BRANCH", ""),
        help="Branch to update. Defaults to current branch.",
    )
    parser.add_argument(
        "--request-filename",
        default="",
        help="Original request JSON filename (e.g. Qwen3-0.6B_quant_request_False_W4A16_4bit_int4.json). "
             "Used to write back status and recorded in the aggregate JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare artifact files locally but do not create a commit or push",
    )
    parser.add_argument(
        "--git-user-name",
        default=os.environ.get("GIT_USER_NAME", "lb-eval-bot"),
        help="git config user.name for generated commits",
    )
    parser.add_argument(
        "--git-user-email",
        default=os.environ.get("GIT_USER_EMAIL", "lb-eval-bot@local"),
        help="git config user.email for generated commits",
    )
    args = parser.parse_args()

    runtime_output_dir = Path(args.runtime_output_dir).resolve()
    if not runtime_output_dir.is_dir():
        print(f"ERROR: runtime output directory not found: {runtime_output_dir}")
        return 1

    model_output_dir = None
    if args.model_output_dir.strip():
        model_output_dir = Path(args.model_output_dir).resolve()

    try:
        repo_dir = resolve_repo_dir(args.repo_dir, args.clone_dir, args.git_repo, args.git_token or None)
    except Exception as exc:
        print(f"ERROR: could not prepare git repo: {exc}")
        return 1

    quant_summary_path = runtime_output_dir / "quant_summary.json"
    summary_path = runtime_output_dir / "summary.json"
    accuracy_path = runtime_output_dir / "accuracy.json"
    quantize_script_path = runtime_output_dir / "quantize.py"
    legacy_quantize_script_path = runtime_output_dir / "quantize_script.py"
    evaluation_script_candidates = [
        (runtime_output_dir / "evaluate.sh", "evaluate.sh"),
        (runtime_output_dir / "eval.sh", "evaluate.sh"),
        (runtime_output_dir / "eval_script.sh", "evaluate.sh"),
        (runtime_output_dir / "evaluate_script.sh", "evaluate.sh"),
        (runtime_output_dir / "evaluate.py", "evaluate.py"),
        (runtime_output_dir / "eval.py", "evaluate.py"),
        (runtime_output_dir / "eval_script.py", "evaluate.py"),
        (runtime_output_dir / "evaluate_script.py", "evaluate.py"),
    ]
    lm_eval_results_dir = runtime_output_dir / "lm_eval_results"
    logs_dir = runtime_output_dir / "logs"
    quant_summary = load_json(quant_summary_path) or load_json(summary_path)
    accuracy = load_json(accuracy_path)

    artifact_name = detect_artifact_name(args.model_id, args.scheme, quant_summary, args.method)
    org = args.model_id.split("/", 1)[0] if "/" in args.model_id else "unknown"
    timestamp = file_timestamp()

    branch = args.branch
    if not branch:
        branch = run_git(["branch", "--show-current"], repo_dir).stdout.strip() or "main"

    print(f"[github-upload] repo dir: {repo_dir}")
    print(f"[github-upload] model id: {args.model_id}")
    print(f"[github-upload] artifact name: {artifact_name}")
    print(f"[github-upload] result org: {org}")
    print(f"[github-upload] branch: {branch}")
    ensure_git_config(repo_dir, args.git_user_name, args.git_user_email)
    print(f"[github-upload] git user: {args.git_user_name} <{args.git_user_email}>")
    if not args.dry_run:
        remote_url = args.git_repo or run_git(["remote", "get-url", "origin"], repo_dir).stdout.strip()
        auth_url = build_auth_url(remote_url, args.git_token or None)
        pull_target = auth_url if args.git_token else "origin"
        push_target = auth_url if args.git_token else "origin"

        pull_args = ["pull", "--rebase", pull_target, branch]
        pull_result = run_git(pull_args, repo_dir, check=False)
        if pull_result.returncode != 0:
            print(f"[github-upload] ERROR: git pull failed:\n{pull_result.stderr.strip()}")
            return 1

    model_result_dir = repo_dir / "results" / org / artifact_name
    run_dir = model_result_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    copy_file(quant_summary_path, run_dir / "quant_summary.json", copied)
    copy_file(summary_path, run_dir / "summary.json", copied)
    copy_file(accuracy_path, run_dir / "accuracy.json", copied)
    # Copy request JSON (original task input) for traceability
    request_json_path = runtime_output_dir / "request.json"
    copy_file(request_json_path, run_dir / "request.json", copied)
    # Copy run report if it exists (generated by generate_report.py)
    run_report_path = runtime_output_dir / "run_report.md"
    copy_file(run_report_path, run_dir / "run_report.md", copied)
    # Copy the recipe deliverable (quant+eval-successful models only) and the
    # agent-synthesized resolution summary if present.
    for deliverable in ("recipe.md", "recipe.json", "resolution_summary.md"):
        src = runtime_output_dir / deliverable
        if src.is_file():
            copy_file(src, run_dir / deliverable, copied)
    if quantize_script_path.is_file():
        copy_file(quantize_script_path, run_dir / "quantize.py", copied)
    elif legacy_quantize_script_path.is_file():
        copy_file(legacy_quantize_script_path, run_dir / "quantize.py", copied)
    for evaluation_script_path, target_name in evaluation_script_candidates:
        if evaluation_script_path.is_file():
            copy_file(evaluation_script_path, run_dir / target_name, copied)
            break
    copy_tree(lm_eval_results_dir, run_dir / "lm_eval_results", copied)
    copy_tree(logs_dir, run_dir / "logs", copied)

    # Ensure exec logs are preserved (they may be in logs/ already via copy_tree,
    # but also check for them directly in runtime_output_dir in case logs_dir differs)
    logs_target = run_dir / "logs"
    logs_target.mkdir(parents=True, exist_ok=True)
    for exec_log_name in ("quant_exec.log", "eval_exec.log", "auto.log"):
        exec_log_src = runtime_output_dir / exec_log_name
        exec_log_dst = logs_target / exec_log_name
        if exec_log_src.is_file() and not exec_log_dst.exists():
            copy_file(exec_log_src, exec_log_dst, copied)

    # Copy failure diagnosis files if present
    for diag_file in sorted(runtime_output_dir.glob("failure_diagnosis_*.json")):
        copy_file(diag_file, run_dir / diag_file.name, copied)

    for path in sorted(runtime_output_dir.glob("session_*.jsonl")):
        copy_file_sanitized(path, run_dir / path.name, copied)
    for path in sorted(runtime_output_dir.glob("session_*.md")):
        copy_file_sanitized(path, run_dir / path.name, copied)

    # Copy agent session artifacts from common locations (openclaw may write here)
    for session_subdir_name in (".openclaw", ".agent_sessions"):
        session_subdir = runtime_output_dir / session_subdir_name
        if session_subdir.is_dir():
            copy_tree(session_subdir, run_dir / session_subdir_name, copied)

    aggregate = {
        "status": derive_pipeline_status(quant_summary, accuracy),
        "pipeline": args.pipeline or ("auto_quant" if quant_summary else "auto_eval"),
        "model_id": args.model_id,
        "artifact_name": artifact_name,
        "request_filename": args.request_filename or None,
        "generated_at": utc_now(),
        "source_runtime_dir": str(runtime_output_dir),
        "source_model_dir": str(model_output_dir) if model_output_dir else None,
        "run_dir": str(run_dir.relative_to(repo_dir)),
        "quant_summary": quant_summary,
        "accuracy": accuracy,
        "copied_files": [str(Path(path).relative_to(repo_dir)) for path in copied],
    }
    if aggregate["pipeline"] == "auto_quant":
        aggregate["quant_num_gpus"] = args.quant_num_gpus or (
            str(quant_summary.get("quant_num_gpus") or quant_summary.get("num_gpus"))
            if isinstance(quant_summary, dict) and (quant_summary.get("quant_num_gpus") or quant_summary.get("num_gpus")) is not None
            else None
        )
        aggregate["eval_num_gpus"] = args.eval_num_gpus or (
            str(accuracy.get("eval_num_gpus") or accuracy.get("num_gpus"))
            if isinstance(accuracy, dict) and (accuracy.get("eval_num_gpus") or accuracy.get("num_gpus")) is not None
            else None
        )
    else:
        aggregate["num_gpus"] = args.eval_num_gpus or args.quant_num_gpus or (
            str(accuracy.get("num_gpus"))
            if isinstance(accuracy, dict) and accuracy.get("num_gpus") is not None
            else None
        )
    aggregate_path = model_result_dir / f"results_{timestamp}.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    copied.append(str(aggregate_path))

    # Generate failure digest for failed pipelines
    generate_failure_digest(run_dir, quant_summary, accuracy)

    # Write-back status to the original request file in status/
    pipeline_status = derive_pipeline_status(quant_summary, accuracy)
    print(f"[github-upload] Derived pipeline status: {pipeline_status}")
    write_back_status(repo_dir, args.request_filename, pipeline_status, copied)

    # ═══ Copy lessons into repo (same commit as results) ═══
    lessons_dir = Path(os.environ.get("LESSONS_DIR", ""))
    if lessons_dir.is_dir():
        repo_lessons_dir = repo_dir / "auto_quant" / "lessons"
        repo_lessons_dir.mkdir(parents=True, exist_ok=True)
        for jsonl_file in sorted(lessons_dir.glob("*.jsonl")):
            dst = repo_lessons_dir / jsonl_file.name
            shutil.copy2(jsonl_file, dst)
            copied.append(str(dst))
        print(f"[github-upload] Included lessons from {lessons_dir}")

    rel_paths = [str(Path(path).relative_to(repo_dir)) for path in copied]
    if not rel_paths:
        print("[github-upload] No artifacts found to upload.")
        return 0

    if args.dry_run:
        print("[github-upload] Dry run prepared artifacts:")
        for rel_path in rel_paths:
            print(f"  - {rel_path}")
        return 0

    # Use --force to override .gitignore (e.g. *.log files must be uploaded)
    run_git(["add", "--force", *rel_paths], repo_dir)
    diff_result = run_git(["diff", "--cached", "--quiet"], repo_dir, check=False)
    if diff_result.returncode == 0:
        print("[github-upload] No changes to commit.")
        return 0

    commit_message = (
        f"Add auto_quant artifacts for {artifact_name}\n\n"
        "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
    )
    commit_result = run_git(["commit", "-m", commit_message], repo_dir, check=False)
    if commit_result.returncode != 0:
        print(f"[github-upload] ERROR: git commit failed:\n{commit_result.stderr.strip()}")
        return 1

    # Push with retry: if another container pushed first, pull --rebase and retry.
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        push_args = ["push", push_target, f"HEAD:{branch}"]
        push_result = run_git(push_args, repo_dir, check=False)
        if push_result.returncode == 0:
            break

        stderr = push_result.stderr.strip()
        # Only retry on non-fast-forward / concurrent push conflicts
        is_conflict = any(hint in stderr.lower() for hint in [
            "non-fast-forward", "fetch first", "stale info",
            "failed to push", "cannot lock ref",
        ])
        if not is_conflict or attempt == max_retries:
            print(f"[github-upload] ERROR: git push failed (attempt {attempt}/{max_retries}):\n{stderr}")
            return 1

        wait = 2 ** attempt  # 2, 4, 8, 16, 32 seconds
        print(f"[github-upload] Push conflict (attempt {attempt}/{max_retries}), "
              f"retrying in {wait}s after pull --rebase ...")
        time.sleep(wait)

        rebase_result = run_git(["pull", "--rebase", pull_target, branch], repo_dir, check=False)
        if rebase_result.returncode != 0:
            # Rebase conflict — abort and retry with a fresh rebase
            run_git(["rebase", "--abort"], repo_dir, check=False)
            print(f"[github-upload] WARNING: rebase conflict, resetting and retrying ...")
            # Reset to remote state, re-apply our changes
            run_git(["fetch", pull_target, branch], repo_dir, check=False)
            run_git(["reset", "--hard", f"FETCH_HEAD"], repo_dir, check=False)
            # Re-stage and re-commit all our artifact files
            run_git(["add", *rel_paths], repo_dir, check=False)
            recommit = run_git(["diff", "--cached", "--quiet"], repo_dir, check=False)
            if recommit.returncode != 0:
                run_git(["commit", "-m", commit_message], repo_dir, check=False)

    print("[github-upload] Uploaded artifacts:")
    for rel_path in rel_paths:
        print(f"  - {rel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
