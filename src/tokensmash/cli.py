#!/usr/bin/env python3
"""
Run controlled Codex A/B token benchmarks.

Primary metric: provider-reported total session tokens per successful task.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


STATE_DIR = Path.home() / ".local" / "state" / "tokensmash"
AB_RUNS_DIR = STATE_DIR / "ab-runs"
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = Path(__file__).resolve().parent / "data" / "suites" / "generic" / "tool-comparison.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark tokensmashing tools by Codex session-token spend."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Validate and print the run plan.")
    plan.add_argument("--suite", default=str(DEFAULT_SUITE))
    plan.add_argument("--tasks", help="Comma-separated task ids to include.")
    plan.add_argument("--variants", help="Comma-separated variant ids to include.")
    plan.add_argument("--replicates", type=int, help="Override suite replicate count.")

    run = sub.add_parser("run", help="Run the suite. Dry-run unless --live is set.")
    run.add_argument("--suite", default=str(DEFAULT_SUITE))
    run.add_argument("--tasks", help="Comma-separated task ids to include.")
    run.add_argument("--variants", help="Comma-separated variant ids to include.")
    run.add_argument("--replicates", type=int, help="Override suite replicate count.")
    run.add_argument("--run-id", help="Stable run id; default is timestamped.")
    run.add_argument("--timeout", type=float, help="Per Codex run timeout seconds.")
    run.add_argument("--live", action="store_true", help="Actually spend model quota.")
    run.add_argument("--keep-scratch", action="store_true")

    table = sub.add_parser("table", help="Print baseline/tool/improvement table.")
    table.add_argument("results", help="Run directory or results.json.")

    report = sub.add_parser("report", help="Write a benchmark-card Markdown report.")
    report.add_argument("results", nargs="+", help="Run directories or results.json files.")
    report.add_argument("-o", "--output", required=True, help="Markdown output path.")

    sessions = sub.add_parser("sessions", help="Audit local Claude/Codex/Gemini session logs without copying raw logs.")
    sessions.add_argument("--agent", choices=["all", "codex", "claude", "gemini"], default="all")
    sessions.add_argument("--codex-root", default=str(Path.home() / ".codex" / "sessions"))
    sessions.add_argument("--claude-root", default=str(Path.home() / ".claude" / "projects"))
    sessions.add_argument("--gemini-root", default=str(Path.home() / ".gemini"))
    sessions.add_argument("--days", type=float, default=7.0)
    sessions.add_argument("--limit-files", type=int, default=500)
    sessions.add_argument("-o", "--output", help="Write sanitized JSON summary.")

    args = parser.parse_args()
    if args.cmd == "plan":
        suite = load_suite(Path(args.suite))
        selection = select_work(suite, args.tasks, args.variants, args.replicates)
        print_plan(suite, selection)
        return 0
    if args.cmd == "run":
        suite = load_suite(Path(args.suite))
        selection = select_work(suite, args.tasks, args.variants, args.replicates)
        if not args.live:
            print_plan(suite, selection)
            print()
            print("dry run only; add --live to spend model quota")
            return 0
        run_dir = run_suite(suite, selection, args)
        print(f"results: {run_dir / 'results.json'}")
        print_table(load_results(run_dir / "results.json"))
        return 0
    if args.cmd == "table":
        path = Path(args.results).expanduser()
        if path.is_dir():
            path = path / "results.json"
        print_table(load_results(path))
        return 0
    if args.cmd == "report":
        results = [load_results(resolve_results_path(Path(path))) for path in args.results]
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_report(results) + "\n")
        print(output)
        return 0
    if args.cmd == "sessions":
        summary = audit_session_logs(args)
        print_session_summary(summary)
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(f"wrote {output}")
        return 0
    return 2


def load_suite(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    suite = json.loads(path.read_text())
    suite["_suite_path"] = str(path)
    suite["_suite_dir"] = str(path.parent)
    require(isinstance(suite.get("tasks"), list) and suite["tasks"], "suite has no tasks")
    require(isinstance(suite.get("variants"), list) and suite["variants"], "suite has no variants")
    task_ids = [str(t.get("id")) for t in suite["tasks"]]
    variant_ids = [str(v.get("id")) for v in suite["variants"]]
    require(len(task_ids) == len(set(task_ids)), "duplicate task ids")
    require(len(variant_ids) == len(set(variant_ids)), "duplicate variant ids")
    baseline = suite.get("baseline")
    require(baseline in variant_ids, f"baseline variant not found: {baseline}")
    return suite


def select_work(
    suite: dict[str, Any],
    tasks_csv: str | None,
    variants_csv: str | None,
    replicate_override: int | None,
) -> dict[str, Any]:
    task_ids = set(csv_ids(tasks_csv))
    variant_ids = set(csv_ids(variants_csv))
    tasks = [t for t in suite["tasks"] if not task_ids or t["id"] in task_ids]
    variants = [v for v in suite["variants"] if not variant_ids or v["id"] in variant_ids]
    if variant_ids and suite["baseline"] not in {v["id"] for v in variants}:
        baseline_variant = next(v for v in suite["variants"] if v["id"] == suite["baseline"])
        variants = [baseline_variant, *variants]
    require(tasks, "no tasks selected")
    require(variants, "no variants selected")
    missing_tasks = task_ids - {t["id"] for t in tasks}
    missing_variants = variant_ids - {v["id"] for v in variants}
    require(not missing_tasks, f"unknown task ids: {', '.join(sorted(missing_tasks))}")
    require(not missing_variants, f"unknown variant ids: {', '.join(sorted(missing_variants))}")
    replicates = replicate_override or int(suite.get("replicates") or 1)
    require(replicates > 0, "replicates must be positive")
    return {"tasks": tasks, "variants": variants, "replicates": replicates}


def csv_ids(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def print_plan(suite: dict[str, Any], selection: dict[str, Any]) -> None:
    print(f"suite: {suite.get('name', Path(suite['_suite_path']).name)}")
    print(
        f"model: {render_env(str(suite.get('model', 'gpt-5.5')))} "
        f"reasoning={render_env(str(suite.get('reasoning', 'low')))} "
        f"sandbox={render_env(str(suite.get('sandbox', 'workspace-write')))}"
    )
    print(f"baseline: {suite['baseline']}")
    print(f"replicates: {selection['replicates']}")
    print()
    print("tasks")
    for task in selection["tasks"]:
        print(f"  {task['id']}: {render_env(str(task['repo']))} @ {render_env(str(task.get('base_ref', 'HEAD')))}")
    print()
    print("variants")
    enabled_count = 0
    for variant in selection["variants"]:
        status, why = variant_status(variant)
        if status == "enabled":
            enabled_count += 1
        print(f"  {variant['id']}: {status}{(' - ' + why) if why else ''}")
    print()
    total = len(selection["tasks"]) * enabled_count * selection["replicates"]
    print(f"live Codex runs if executed: {total}")


def variant_status(variant: dict[str, Any]) -> tuple[str, str]:
    if variant.get("enabled") is False:
        return "disabled", str(variant.get("disabled_reason") or "enabled=false")
    missing = missing_requirements(variant.get("requires") or {})
    if missing:
        return "skipped", "missing " + ", ".join(missing)
    return "enabled", ""


def missing_requirements(requires: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for name in requires.get("env", []) or []:
        if not os.environ.get(str(name)):
            missing.append(f"env:{name}")
    for name in requires.get("commands", []) or []:
        if shutil.which(str(name)) is None:
            missing.append(f"cmd:{name}")
    return missing


def run_suite(suite: dict[str, Any], selection: dict[str, Any], args: argparse.Namespace) -> Path:
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = AB_RUNS_DIR / run_id
    require(not run_dir.exists(), f"run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    results: dict[str, Any] = {
        "schema": 1,
        "kind": "tokensmash-ab",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite_path": suite["_suite_path"],
        "suite": scrub_suite(suite),
        "baseline": suite["baseline"],
        "model": render_env(str(suite.get("model", "gpt-5.5"))),
        "reasoning": render_env(str(suite.get("reasoning", "low"))),
        "replicates": selection["replicates"],
        "runs": [],
        "notes": [
            "primary metric is Codex final token_count total_token_usage.total_tokens",
            "only paired successful task runs are used for improvement percentages",
            "raw Codex stdout/stderr and last message are stored in the run directory for debugging",
        ],
    }
    (run_dir / "suite.json").write_text(json.dumps(scrub_suite(suite), indent=2, sort_keys=True) + "\n")
    try:
        for task in selection["tasks"]:
            for variant in selection["variants"]:
                status, why = variant_status(variant)
                if status != "enabled":
                    results["runs"].append(
                        {
                            "task_id": task["id"],
                            "variant_id": variant["id"],
                            "status": status,
                            "skip_reason": why,
                        }
                    )
                    continue
                for replicate in range(1, selection["replicates"] + 1):
                    result = run_one(suite, task, variant, replicate, run_dir, args)
                    results["runs"].append(result)
                    write_results(run_dir, results)
    finally:
        write_results(run_dir, results)
    return run_dir


def run_one(
    suite: dict[str, Any],
    task: dict[str, Any],
    variant: dict[str, Any],
    replicate: int,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_dir = run_dir / "cases" / task["id"] / variant["id"] / f"r{replicate}"
    repo_dir = case_dir / "repo"
    case_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        "task_id": task["id"],
        "variant_id": variant["id"],
        "replicate": replicate,
        "status": "running",
        "case_dir": str(case_dir),
    }
    started = time.time()
    try:
        prepare_repo(task, repo_dir)
        env = case_env(suite, task, variant, case_dir, repo_dir)
        run_setup_commands(variant, case_dir, repo_dir, env)
        prompt = build_prompt(suite, task, variant, case_dir, repo_dir)
        prompt_path = case_dir / "prompt.md"
        prompt_path.write_text(prompt)
        codex_result = run_codex(suite, variant, repo_dir, case_dir, prompt, args.timeout, env)
        result.update(codex_result)
        result["success_commands"] = run_success_commands(task, repo_dir, env)
        result["success"] = (
            result.get("codex_exit_code") == 0
            and result.get("token_total") is not None
            and all(cmd["exit_code"] == 0 for cmd in result["success_commands"])
        )
        result["status"] = "ok" if result["success"] else "failed"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        result["duration_ms"] = round((time.time() - started) * 1000)
        if not args.keep_scratch and repo_dir.exists():
            shutil.rmtree(repo_dir)
    return result


def prepare_repo(task: dict[str, Any], repo_dir: Path) -> None:
    source = Path(render_env(str(task["repo"]))).expanduser().resolve()
    require(source.exists(), f"repo does not exist: {source}")
    base_ref = render_env(str(task.get("base_ref") or "HEAD"))
    run_checked(["git", "clone", "--quiet", "--shared", str(source), str(repo_dir)], cwd=None)
    run_checked(["git", "checkout", "--quiet", base_ref], cwd=repo_dir)
    run_checked(["git", "clean", "-xfd", "--quiet"], cwd=repo_dir)


def case_env(
    suite: dict[str, Any],
    task: dict[str, Any],
    variant: dict[str, Any],
    case_dir: Path,
    repo_dir: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SUITE_DIR": suite["_suite_dir"],
            "TASK_ID": str(task["id"]),
            "VARIANT_ID": str(variant["id"]),
            "RUN_DIR": str(case_dir),
            "REPO_DIR": str(repo_dir),
        }
    )
    for source in (suite.get("env") or {}, task.get("env") or {}, variant.get("env") or {}):
        for key, value in source.items():
            env[str(key)] = render(str(value), case_dir, repo_dir, variant)
    prepare_variant_codex_home(variant, case_dir, repo_dir, env)
    return env


def prepare_variant_codex_home(
    variant: dict[str, Any],
    case_dir: Path,
    repo_dir: Path,
    env: dict[str, str],
) -> None:
    if not variant.get("codex_home"):
        return
    codex_home = case_dir / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    auth = Path.home() / ".codex" / "auth.json"
    require(auth.exists(), f"missing Codex auth file: {auth}")
    shutil.copy2(auth, codex_home / "auth.json")
    config = variant.get("codex_config")
    if config:
        (codex_home / "config.toml").write_text(render(str(config), case_dir, repo_dir, variant) + "\n")
    if variant.get("codex_hooks") == "context-mode":
        (codex_home / "hooks.json").write_text(json.dumps(context_mode_hooks_json(), indent=2) + "\n")
    env["CODEX_HOME"] = str(codex_home)


def context_mode_hooks_json() -> dict[str, Any]:
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "local_shell|shell|shell_command|exec_command|container.exec|Bash|Shell|grep_files",
                    "hooks": [{"type": "command", "command": "context-mode hook codex pretooluse"}],
                }
            ],
            "PostToolUse": [
                {"hooks": [{"type": "command", "command": "context-mode hook codex posttooluse"}]}
            ],
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "context-mode hook codex sessionstart"}]}
            ],
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "context-mode hook codex userpromptsubmit"}]}
            ],
            "Stop": [
                {"hooks": [{"type": "command", "command": "context-mode hook codex stop"}]}
            ],
        }
    }


def run_setup_commands(variant: dict[str, Any], case_dir: Path, repo_dir: Path, env: dict[str, str]) -> None:
    for command in variant.get("setup_commands", []) or []:
        rendered = render(str(command), case_dir, repo_dir, variant)
        run_checked(rendered, cwd=repo_dir, env=env, shell=True)


def build_prompt(
    suite: dict[str, Any],
    task: dict[str, Any],
    variant: dict[str, Any],
    case_dir: Path,
    repo_dir: Path,
) -> str:
    prefix = render(str(variant.get("prompt_prefix") or ""), case_dir, repo_dir, variant).strip()
    suffix = render(str(variant.get("prompt_suffix") or ""), case_dir, repo_dir, variant).strip()
    success = "\n".join(f"- `{render_env(str(cmd))}`" for cmd in task.get("success_commands", []) or [])
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(
        "\n".join(
            [
                "You are running inside a disposable benchmark checkout.",
                "Complete the task with the smallest correct code change.",
                "Do not skip verification unless the command is unavailable.",
                "",
                "Task:",
                render_env(str(task["prompt"])).strip(),
                "",
                "Verification commands expected by the harness:",
                success or "- no explicit command",
            ]
        )
    )
    if suffix:
        parts.append(suffix)
    return "\n\n".join(parts) + "\n"


def run_codex(
    suite: dict[str, Any],
    variant: dict[str, Any],
    repo_dir: Path,
    case_dir: Path,
    prompt: str,
    timeout_override: float | None,
    env: dict[str, str],
) -> dict[str, Any]:
    stdout_path = case_dir / "codex.stdout"
    stderr_path = case_dir / "codex.stderr"
    last_message_path = case_dir / "last-message.md"
    timeout = timeout_override or float(suite.get("timeout_seconds") or 1800)
    model = render_env(str(suite.get("model") or "gpt-5.5"))
    reasoning = render_env(str(suite.get("reasoning") or "low"))
    sandbox = render_env(str(suite.get("sandbox") or "workspace-write"))
    cmd = [
        "codex",
        "-a",
        "never",
        "-s",
        sandbox,
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning}"',
        "-C",
        str(repo_dir),
        "exec",
        "--color",
        "never",
        "-o",
        str(last_message_path),
        *[str(arg) for arg in variant.get("codex_args", []) or []],
        prompt,
    ]
    start_time = time.time()
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_dir,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout,
                check=False,
            )
            exit_code = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            exit_code = None
            timed_out = True
    session_id = parse_session_id(
        read_head(stdout_path)
        + "\n"
        + read_tail(stdout_path)
        + "\n"
        + read_head(stderr_path)
        + "\n"
        + read_tail(stderr_path)
    )
    session_path = find_codex_session(session_id, repo_dir, start_time, env)
    usage = extract_codex_usage(session_path) if session_path else {}
    return {
        "codex_command": shell_join_redacted(cmd),
        "codex_exit_code": exit_code,
        "codex_timed_out": timed_out,
        "codex_stdout": file_metrics(stdout_path),
        "codex_stderr": file_metrics(stderr_path),
        "last_message": file_metrics(last_message_path) if last_message_path.exists() else None,
        "session_id": session_id,
        "session_path": str(session_path) if session_path else None,
        "token_source": "codex:final_token_count" if usage else "missing",
        "token_total": usage.get("total_tokens"),
        "token_usage": usage,
    }


def run_success_commands(task: dict[str, Any], repo_dir: Path, env: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for command in task.get("success_commands", []) or []:
        rendered_command = render_env(str(command))
        started = time.time()
        proc = subprocess.run(
            rendered_command,
            cwd=repo_dir,
            env=env,
            shell=True,
            executable=os.environ.get("SHELL", "/bin/sh"),
            capture_output=True,
            check=False,
            timeout=float(task.get("success_timeout_seconds") or 300),
        )
        payload = proc.stdout + proc.stderr
        out.append(
            {
                "command": rendered_command,
                "exit_code": proc.returncode,
                "duration_ms": round((time.time() - started) * 1000),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return out


def render(template: str, case_dir: Path, repo_dir: Path, variant: dict[str, Any]) -> str:
    return render_env(template).format(
        run_dir=str(case_dir),
        repo_dir=str(repo_dir),
        variant_id=str(variant.get("id") or ""),
    )


def render_env(value: str) -> str:
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        fallback = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if fallback is not None:
            return fallback
        return match.group(0)

    return pattern.sub(replace, value)


def run_checked(
    command: list[str] | str,
    *,
    cwd: Path | None,
    env: dict[str, str] | None = None,
    shell: bool = False,
) -> None:
    proc = subprocess.run(command, cwd=cwd, env=env, shell=shell, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {command}\n"
            f"{(proc.stderr or proc.stdout).decode('utf-8', 'replace')[-1000:]}"
        )


def parse_session_id(text: str) -> str | None:
    patterns = [
        r"session id:\s*([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        r'"id"\s*:\s*"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def find_codex_session(
    session_id: str | None,
    repo_dir: Path,
    start_time: float,
    env: dict[str, str],
) -> Path | None:
    roots = session_roots(env)
    if session_id:
        matches = []
        for root in roots:
            matches.extend(root.glob(f"**/*{session_id}.jsonl"))
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
    candidates = []
    cutoff = start_time - 5
    for root in roots:
        for path in root.glob("**/*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    continue
            except OSError:
                continue
            if session_cwd(path) == str(repo_dir):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def session_roots(env: dict[str, str]) -> list[Path]:
    roots = []
    configured = env.get("CODEX_HOME")
    if configured:
        roots.append(Path(configured) / "sessions")
    roots.append(CODEX_SESSIONS_DIR)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            unique.append(root)
            seen.add(key)
    return unique


def session_cwd(path: Path) -> str | None:
    for obj in iter_jsonl(path):
        payload = obj.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "session_meta":
            return payload.get("cwd")
        if obj.get("type") == "session_meta" and isinstance(payload, dict):
            return payload.get("cwd")
    return None


def extract_codex_usage(path: Path | None) -> dict[str, int]:
    if not path:
        return {}
    usage: dict[str, Any] = {}
    for obj in iter_jsonl(path):
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "token_count":
            candidate = ((payload.get("info") or {}).get("total_token_usage") or {})
            if isinstance(candidate, dict):
                usage = candidate
    if not usage:
        return {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_output_tokens": int(usage.get("reasoning_output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def audit_session_logs(args: argparse.Namespace) -> dict[str, Any]:
    cutoff = time.time() - args.days * 86400
    agents: list[dict[str, Any]] = []
    if args.agent in {"all", "codex"}:
        agents.append(audit_agent_root("codex", Path(args.codex_root).expanduser(), cutoff, args.limit_files))
    if args.agent in {"all", "claude"}:
        agents.append(audit_agent_root("claude", Path(args.claude_root).expanduser(), cutoff, args.limit_files))
    if args.agent in {"all", "gemini"}:
        agents.append(audit_agent_root("gemini", Path(args.gemini_root).expanduser(), cutoff, args.limit_files))
    total_sessions = sum(agent["session_count"] for agent in agents)
    total_tokens = sum(agent["total_tokens"] for agent in agents)
    return {
        "schema": 1,
        "kind": "tokensmash-session-audit",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "days": args.days,
        "limit_files": args.limit_files,
        "agents": agents,
        "summary": {
            "agent_count": len(agents),
            "session_count": total_sessions,
            "total_tokens": total_tokens,
        },
        "privacy": [
            "raw session files are not copied",
            "raw prompts, messages, and tool outputs are not emitted",
            "source files are represented by hashes and aggregate counters only",
        ],
    }


def audit_agent_root(agent: str, root: Path, cutoff: float, limit_files: int) -> dict[str, Any]:
    files = recent_session_files(root, cutoff, limit_files)
    sessions = [audit_session_file(agent, path) for path in files]
    return {
        "agent": agent,
        "root": str(root),
        "root_exists": root.exists(),
        "file_count": len(files),
        "session_count": len(sessions),
        "total_tokens": sum(session["total_tokens"] for session in sessions),
        "tool_output_bytes": sum(session["tool_output_bytes"] for session in sessions),
        "tool_calls": sum(session["tool_calls"] for session in sessions),
        "token_confidence": token_confidence(agent),
        "sessions": sorted(sessions, key=lambda s: s["total_tokens"], reverse=True)[:20],
    }


def recent_session_files(root: Path, cutoff: float, limit: int) -> list[Path]:
    if not root.exists():
        return []
    patterns = ["*.jsonl", "*.json"]
    files: list[Path] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            try:
                if path.is_file() and path.stat().st_mtime >= cutoff:
                    files.append(path)
            except OSError:
                continue
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:limit]


def audit_session_file(agent: str, path: Path) -> dict[str, Any]:
    stats = {
        "source_file_sha256": sha_text(str(path)),
        "source_file_mtime": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
        "event_count": 0,
        "user_turns": 0,
        "assistant_messages": 0,
        "tool_calls": 0,
        "tool_output_bytes": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    objects = list(iter_session_objects(path))
    if agent == "codex":
        apply_codex_session_stats(stats, objects)
    elif agent == "claude":
        apply_claude_session_stats(stats, objects)
    else:
        apply_generic_session_stats(stats, objects)
    return stats


def iter_session_objects(path: Path):
    if path.suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    try:
        data = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        yield data


def apply_codex_session_stats(stats: dict[str, Any], objects: list[dict[str, Any]]) -> None:
    usage: dict[str, Any] = {}
    for obj in objects:
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        stats["event_count"] += 1
        typ = payload.get("type")
        if typ == "user_message":
            stats["user_turns"] += 1
        elif typ in {"agent_message", "message"}:
            stats["assistant_messages"] += 1
        elif typ in {"function_call", "custom_tool_call", "mcp_tool_call_begin", "web_search_call"}:
            stats["tool_calls"] += 1
        text = codex_output_text(payload)
        stats["tool_output_bytes"] += len(text.encode("utf-8", "replace"))
        if typ == "token_count":
            candidate = ((payload.get("info") or {}).get("total_token_usage") or {})
            if isinstance(candidate, dict):
                usage = candidate
    apply_usage(stats, usage)


def apply_claude_session_stats(stats: dict[str, Any], objects: list[dict[str, Any]]) -> None:
    for obj in objects:
        stats["event_count"] += 1
        msg = obj.get("message")
        if isinstance(msg, dict):
            role = msg.get("role")
            if role == "user":
                stats["user_turns"] += 1
            elif role == "assistant":
                stats["assistant_messages"] += 1
            usage = msg.get("usage")
            if isinstance(usage, dict):
                add_usage(stats, usage)
            content = msg.get("content")
            if isinstance(content, list):
                stats["tool_calls"] += sum(1 for item in content if isinstance(item, dict) and item.get("type") == "tool_use")
        if "toolUseResult" in obj:
            stats["tool_output_bytes"] += len(json.dumps(obj.get("toolUseResult"), separators=(",", ":")).encode("utf-8", "replace"))


def apply_generic_session_stats(stats: dict[str, Any], objects: list[dict[str, Any]]) -> None:
    for obj in objects:
        stats["event_count"] += 1
        text = json.dumps(obj, separators=(",", ":"))
        lowered = text.lower()
        if "tool" in lowered:
            stats["tool_calls"] += 1
        usage = find_usage_dict(obj)
        if usage:
            add_usage(stats, usage)


def find_usage_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if any(key in value for key in ("input_tokens", "promptTokenCount", "total_tokens", "totalTokenCount")):
            return value
        for child in value.values():
            found = find_usage_dict(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_usage_dict(child)
            if found:
                return found
    return None


def codex_output_text(payload: dict[str, Any]) -> str:
    typ = payload.get("type")
    if typ in {"function_call_output", "custom_tool_call_output"}:
        output = payload.get("output")
        return output if isinstance(output, str) else json.dumps(output, separators=(",", ":")) if output is not None else ""
    if typ == "mcp_tool_call_end":
        result = payload.get("result")
        return result if isinstance(result, str) else json.dumps(result, separators=(",", ":")) if result is not None else ""
    return ""


def apply_usage(stats: dict[str, Any], usage: dict[str, Any]) -> None:
    stats["input_tokens"] = int(usage.get("input_tokens") or usage.get("promptTokenCount") or stats.get("input_tokens") or 0)
    stats["cached_input_tokens"] = int(usage.get("cached_input_tokens") or stats.get("cached_input_tokens") or 0)
    stats["output_tokens"] = int(usage.get("output_tokens") or usage.get("candidatesTokenCount") or stats.get("output_tokens") or 0)
    stats["reasoning_output_tokens"] = int(usage.get("reasoning_output_tokens") or stats.get("reasoning_output_tokens") or 0)
    stats["total_tokens"] = int(
        usage.get("total_tokens")
        or usage.get("totalTokenCount")
        or (stats["input_tokens"] + stats["output_tokens"] + stats["reasoning_output_tokens"])
    )


def add_usage(stats: dict[str, Any], usage: dict[str, Any]) -> None:
    input_tokens = int(usage.get("input_tokens") or usage.get("promptTokenCount") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("candidatesTokenCount") or 0)
    total = int(usage.get("total_tokens") or usage.get("totalTokenCount") or (input_tokens + cache_creation + cache_read + output_tokens))
    stats["input_tokens"] += input_tokens
    stats["cached_input_tokens"] += cache_read
    stats["output_tokens"] += output_tokens
    stats["total_tokens"] += total


def token_confidence(agent: str) -> str:
    if agent == "codex":
        return "high when token_count events exist"
    if agent == "claude":
        return "medium-high when message usage fields exist"
    return "best-effort schema heuristics"


def print_session_summary(summary: dict[str, Any]) -> None:
    print("agent   files  sessions  tokens  tool_calls  tool_output_bytes  confidence")
    print("------  -----  --------  ------  ----------  -----------------  ----------")
    for agent in summary["agents"]:
        print(
            f"{agent['agent']:<6}  {agent['file_count']:<5}  {agent['session_count']:<8}  "
            f"{agent['total_tokens']:<6}  {agent['tool_calls']:<10}  "
            f"{agent['tool_output_bytes']:<17}  {agent['token_confidence']}"
        )


def iter_jsonl(path: Path):
    try:
        with path.open(errors="ignore") as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def file_metrics(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


def read_tail(path: Path, size: int = 12000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        try:
            handle.seek(max(0, path.stat().st_size - size))
        except OSError:
            pass
        return handle.read().decode("utf-8", "replace")


def read_head(path: Path, size: int = 12000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        return handle.read(size).decode("utf-8", "replace")


def write_results(run_dir: Path, results: dict[str, Any]) -> None:
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


def load_results(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    data = json.loads(resolved.read_text())
    data["_path"] = str(resolved)
    return data


def resolve_results_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_dir():
        return path / "results.json"
    return path


def print_table(results: dict[str, Any]) -> None:
    baseline_id = str(results["baseline"])
    rows = comparison_rows(results, baseline_id)
    headers = ["tool", "baseline token spend", "token spend with tool", "percent improvement"]
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def comparison_rows(results: dict[str, Any], baseline_id: str) -> list[list[str]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for run in results.get("runs", []):
        if not run.get("success"):
            continue
        key = (str(run.get("task_id")), int(run.get("replicate") or 1))
        grouped.setdefault(key, {})[str(run.get("variant_id"))] = run
    variants = sorted(
        {
            str(run.get("variant_id"))
            for run in results.get("runs", [])
            if run.get("variant_id") and run.get("status") not in {"disabled", "skipped"}
        }
    )
    rows: list[list[str]] = []
    for variant_id in variants:
        if variant_id == baseline_id:
            continue
        base_total = 0
        tool_total = 0
        pairs = 0
        for by_variant in grouped.values():
            base = by_variant.get(baseline_id)
            tool = by_variant.get(variant_id)
            if not base or not tool:
                continue
            base_tokens = base.get("token_total")
            tool_tokens = tool.get("token_total")
            if base_tokens is None or tool_tokens is None:
                continue
            base_total += int(base_tokens)
            tool_total += int(tool_tokens)
            pairs += 1
        if pairs == 0:
            rows.append([variant_id, "n/a", "n/a", "n/a"])
        else:
            rows.append([variant_id, str(base_total), str(tool_total), pct_improvement(base_total, tool_total)])
    return rows or [["no comparable tool rows", "n/a", "n/a", "n/a"]]


def build_report(results_list: list[dict[str, Any]]) -> str:
    rows = benchmark_rows(results_list)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# Tokensmash Agent Tool Benchmark Card",
        "",
        f"Generated: {now}",
        "",
        "## Summary",
        "",
        "Primary metric: Codex provider-reported `total_token_usage.total_tokens` per successful task run.",
        "Each row uses the paired baseline from the same result batch, task, and replicate.",
        "",
        markdown_table(
            [
                "tool",
                "baseline tokens",
                "tool tokens",
                "improvement",
                "model",
                "reasoning",
                "task",
                "oracle",
                "result",
            ],
            [
                [
                    row["tool"],
                    str(row["baseline_tokens"]),
                    str(row["tool_tokens"]),
                    pct_improvement(row["baseline_tokens"], row["tool_tokens"]),
                    row["model"],
                    row["reasoning"],
                    row["task_id"],
                    row["oracle"],
                    row["result"],
                ]
                for row in rows
            ],
        ),
        "",
        "## Token Breakdown",
        "",
        markdown_table(
            [
                "tool",
                "baseline input",
                "baseline output",
                "baseline reasoning",
                "tool input",
                "tool output",
                "tool reasoning",
                "duration",
            ],
            [
                [
                    row["tool"],
                    str(row["baseline_usage"].get("input_tokens", 0)),
                    str(row["baseline_usage"].get("output_tokens", 0)),
                    str(row["baseline_usage"].get("reasoning_output_tokens", 0)),
                    str(row["tool_usage"].get("input_tokens", 0)),
                    str(row["tool_usage"].get("output_tokens", 0)),
                    str(row["tool_usage"].get("reasoning_output_tokens", 0)),
                    format_duration(row["duration_ms"]),
                ]
                for row in rows
            ],
        ),
        "",
        "## Evaluation Card",
        "",
    ]
    for key, value in evaluation_card(results_list).items():
        lines.append(f"- **{key}:** {value}")
    lines.extend(
        [
            "",
            "## Result Files",
            "",
            *[f"- `{row['result_file']}`" for row in rows],
            "",
            "## Limitations",
            "",
            "- One task and one replicate per row; use directionally, not as a confidence interval.",
            "- Rows from different result batches have different paired baselines; compare each row to its own baseline.",
            "- This measures Codex CLI behavior only, not Claude/Gemini or non-coding research sessions.",
            "- Tool exposure is not always tool use; inspect session logs before making global defaults from a row.",
        ]
    )
    return "\n".join(lines)


def benchmark_rows(results_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for results in results_list:
        baseline_id = str(results["baseline"])
        tasks = {task["id"]: task for task in (results.get("suite") or {}).get("tasks", [])}
        variants = {variant["id"]: variant for variant in (results.get("suite") or {}).get("variants", [])}
        successful: dict[tuple[str, int, str], dict[str, Any]] = {}
        for run in results.get("runs", []):
            if run.get("success"):
                key = (str(run.get("task_id")), int(run.get("replicate") or 1), str(run.get("variant_id")))
                successful[key] = run
        for run in results.get("runs", []):
            if not run.get("success"):
                continue
            variant_id = str(run.get("variant_id"))
            if variant_id == baseline_id:
                continue
            task_id = str(run.get("task_id"))
            replicate = int(run.get("replicate") or 1)
            baseline = successful.get((task_id, replicate, baseline_id))
            if not baseline:
                continue
            task = tasks.get(task_id, {})
            rows.append(
                {
                    "tool": variant_id,
                    "tool_label": (variants.get(variant_id) or {}).get("label", variant_id),
                    "baseline_tokens": int(baseline.get("token_total") or 0),
                    "tool_tokens": int(run.get("token_total") or 0),
                    "baseline_usage": baseline.get("token_usage") or {},
                    "tool_usage": run.get("token_usage") or {},
                    "model": str(results.get("model") or ""),
                    "reasoning": str(results.get("reasoning") or ""),
                    "task_id": task_id,
                    "repo": str(task.get("repo") or ""),
                    "base_ref": str(task.get("base_ref") or ""),
                    "oracle": ", ".join(str(cmd) for cmd in task.get("success_commands", []) or []) or "none",
                    "result": "pass" if run.get("success") else "fail",
                    "duration_ms": int(run.get("duration_ms") or 0),
                    "result_file": str(resolve_results_path(Path(results.get("_path", ""))) if results.get("_path") else ""),
                }
            )
    order = {"context_mode": 0, "headroom": 1, "rtk": 2, "semmap": 3, "repomix": 4, "gitingest": 5}
    rows.sort(key=lambda row: (order.get(row["tool"], 999), row["tool"]))
    return rows


def evaluation_card(results_list: list[dict[str, Any]]) -> dict[str, str]:
    suite_names = sorted({str((r.get("suite") or {}).get("name") or "") for r in results_list})
    models = sorted({str(r.get("model") or "") for r in results_list})
    reasoning = sorted({str(r.get("reasoning") or "") for r in results_list})
    tasks = []
    repos = []
    base_refs = []
    oracles = []
    for results in results_list:
        for task in (results.get("suite") or {}).get("tasks", []):
            tasks.append(str(task.get("id") or ""))
            repos.append(str(task.get("repo") or ""))
            base_refs.append(str(task.get("base_ref") or ""))
            oracles.extend(str(cmd) for cmd in task.get("success_commands", []) or [])
    return {
        "suite": ", ".join(unique_nonempty(suite_names)),
        "agent": "Codex CLI",
        "model": ", ".join(unique_nonempty(models)),
        "reasoning effort": ", ".join(unique_nonempty(reasoning)),
        "tasks": ", ".join(unique_nonempty(tasks)),
        "repository": ", ".join(unique_nonempty(repos)),
        "base ref": ", ".join(unique_nonempty(base_refs)),
        "verification oracle": ", ".join(unique_nonempty(oracles)),
        "replicates": "1 per row",
        "sandbox": "workspace-write",
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        rows = [["n/a" for _ in headers]]
    escaped_rows = [[markdown_cell(cell) for cell in row] for row in rows]
    escaped_headers = [markdown_cell(header) for header in headers]
    widths = [
        max(len(escaped_headers[i]), *(len(row[i]) for row in escaped_rows))
        for i in range(len(headers))
    ]
    out = [
        "| " + " | ".join(escaped_headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in escaped_rows:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return "\n".join(out)


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def format_duration(ms: int) -> str:
    if ms <= 0:
        return "n/a"
    return f"{ms / 1000:.1f}s"


def pct_improvement(baseline: int, tool: int) -> str:
    if baseline <= 0:
        return "n/a"
    return f"{((baseline - tool) / baseline * 100):+.1f}%"


def scrub_suite(suite: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in suite.items() if not k.startswith("_")}


def shell_join_redacted(cmd: list[str]) -> str:
    return " ".join(sh_quote(part) for part in cmd[:-1] + ["<prompt>"])


def sh_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=+-]+", value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


if __name__ == "__main__":
    raise SystemExit(main())
