#!/usr/bin/env python3
"""
Run controlled agent A/B token benchmarks.

Primary metric: provider-reported total session tokens per successful task.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


STATE_DIR = Path.home() / ".local" / "state" / "tokensmash"
AB_RUNS_DIR = STATE_DIR / "ab-runs"
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = Path(__file__).resolve().parent / "data" / "suites" / "generic" / "tool-comparison.json"
GPT55_SHORT_CONTEXT_PRICES = {
    "fresh_input_per_m": 2.50,
    "cached_input_per_m": 0.25,
    "output_per_m": 15.00,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark tokensmashing tools by agent session-token spend."
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
    run.add_argument("--timeout", type=float, help="Per agent run timeout seconds.")
    run.add_argument("--live", action="store_true", help="Actually spend model quota.")
    run.add_argument("--keep-scratch", action="store_true")
    run.add_argument("--randomize-order", action="store_true", help="Shuffle task/variant execution order.")
    run.add_argument("--seed", type=int, default=0, help="Seed for --randomize-order.")

    table = sub.add_parser("table", help="Print baseline/tool/improvement table.")
    table.add_argument("results", help="Run directory or results.json.")

    aggregate = sub.add_parser("aggregate", help="Aggregate comparable tool rows across result files.")
    aggregate.add_argument("results", nargs="+", help="Run directories or results.json files.")
    aggregate.add_argument("--strict", action="store_true", help="Require methodology audit pass for both paired runs.")

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

    eval_sessions_parser = sub.add_parser(
        "eval-sessions",
        help="Evaluate token-saving tools against real local agent sessions.",
    )
    eval_sessions_parser.add_argument("--agent", choices=["codex"], default="codex")
    eval_sessions_parser.add_argument("--codex-root", default=str(CODEX_SESSIONS_DIR))
    eval_sessions_parser.add_argument("--repo-root", default=str(Path.cwd()), help="Repository root for artifact-tool evaluation.")
    eval_sessions_parser.add_argument("--sample-name", default="", help="Human label for this arbitrary session sample.")
    eval_sessions_parser.add_argument("--latest", type=int, default=5, help="Number of recent Codex sessions to evaluate.")
    eval_sessions_parser.add_argument("--session", action="append", default=[], help="Codex JSONL file, directory, glob, or @manifest path; repeatable.")
    eval_sessions_parser.add_argument(
        "--tools",
        default="rtk,context-mode,semmap,repomix,gitingest,headroom",
        help="Comma-separated tools to evaluate.",
    )
    eval_sessions_parser.add_argument("--semmap-bin", default=os.environ.get("SEMMAP_BIN", "semmap"))
    eval_sessions_parser.add_argument("--headroom-perf", help="Headroom perf raw JSON from `headroom perf --format json --raw`.")
    eval_sessions_parser.add_argument("--no-artifacts", action="store_true", help="Skip local SEMMAP/Repomix/Gitingest artifact generation.")
    eval_sessions_parser.add_argument("-o", "--output", help="Write sanitized JSON result.")

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
    if args.cmd == "aggregate":
        results = [load_results(resolve_results_path(Path(path))) for path in args.results]
        print_aggregate_table(results, strict=args.strict)
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
    if args.cmd == "eval-sessions":
        result = eval_sessions(args)
        print_eval_sessions_table(result)
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
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
    agent = suite_agent(suite)
    print(f"suite: {suite.get('name', Path(suite['_suite_path']).name)}")
    print(
        f"agent: {agent} "
        f"model: {suite_model(suite, agent)} "
        f"effort={suite_effort(suite, agent)} "
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
        status, why = variant_status(suite, variant)
        if status == "enabled":
            enabled_count += 1
        variant_agent_id = variant_agent(suite, variant)
        agent_suffix = "" if variant_agent_id == agent else f" [{variant_agent_id}]"
        print(f"  {variant['id']}{agent_suffix}: {status}{(' - ' + why) if why else ''}")
    print()
    total = len(selection["tasks"]) * enabled_count * selection["replicates"]
    print(f"live agent runs if executed: {total}")


def variant_status(suite: dict[str, Any], variant: dict[str, Any]) -> tuple[str, str]:
    if variant.get("enabled") is False:
        return "disabled", str(variant.get("disabled_reason") or "enabled=false")
    missing = missing_requirements(variant.get("requires") or {}, variant_agent(suite, variant))
    if missing:
        return "skipped", "missing " + ", ".join(missing)
    return "enabled", ""


def missing_requirements(requires: dict[str, Any], agent: str) -> list[str]:
    missing: list[str] = []
    agents = [str(value).lower() for value in requires.get("agents", []) or []]
    if agents and agent not in agents:
        missing.append(f"agent:{agent}")
    if agent in {"codex", "claude"} and shutil.which(agent) is None:
        missing.append(f"cmd:{agent}")
    for name in requires.get("env", []) or []:
        if not os.environ.get(str(name)):
            missing.append(f"env:{name}")
    for name in requires.get("commands", []) or []:
        command = render_env(str(name))
        if shutil.which(command) is None:
            missing.append(f"cmd:{command}")
    return missing


def suite_agent(suite: dict[str, Any]) -> str:
    return render_env(str(suite.get("agent") or "codex")).strip().lower()


def variant_agent(suite: dict[str, Any], variant: dict[str, Any]) -> str:
    return render_env(str(variant.get("agent") or suite.get("agent") or "codex")).strip().lower()


def suite_model(suite: dict[str, Any], agent: str) -> str:
    if agent == "claude":
        return render_env(str(suite.get("claude_model") or suite.get("model") or "sonnet"))
    return render_env(str(suite.get("model") or "gpt-5.5"))


def suite_effort(suite: dict[str, Any], agent: str) -> str:
    if agent == "claude":
        return render_env(str(suite.get("claude_effort") or suite.get("reasoning") or "low"))
    return render_env(str(suite.get("reasoning") or "low"))


def run_suite(suite: dict[str, Any], selection: dict[str, Any], args: argparse.Namespace) -> Path:
    run_id = args.run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = AB_RUNS_DIR / run_id
    require(not run_dir.exists(), f"run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    agent = suite_agent(suite)
    results: dict[str, Any] = {
        "schema": 1,
        "kind": "tokensmash-ab",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite_path": suite["_suite_path"],
        "suite": scrub_suite(suite),
        "baseline": suite["baseline"],
        "agent": agent,
        "model": suite_model(suite, agent),
        "reasoning": suite_effort(suite, agent),
        "replicates": selection["replicates"],
        "runs": [],
        "notes": [
            "primary metric is provider-reported total session tokens when available",
            "only paired successful task runs are used for improvement percentages",
            "raw agent stdout/stderr and last message are stored in the run directory for debugging",
        ],
    }
    (run_dir / "suite.json").write_text(json.dumps(scrub_suite(suite), indent=2, sort_keys=True) + "\n")
    cases: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for task in selection["tasks"]:
        for variant in selection["variants"]:
            for replicate in range(1, selection["replicates"] + 1):
                cases.append((task, variant, replicate))
    if getattr(args, "randomize_order", False):
        rng = random.Random(int(getattr(args, "seed", 0) or 0))
        rng.shuffle(cases)
        results["randomized_order"] = True
        results["random_seed"] = int(getattr(args, "seed", 0) or 0)
    else:
        results["randomized_order"] = False
    try:
        for run_order, (task, variant, replicate) in enumerate(cases, start=1):
            status, why = variant_status(suite, variant)
            if status != "enabled":
                results["runs"].append(
                    {
                        "task_id": task["id"],
                        "variant_id": variant["id"],
                        "agent": variant_agent(suite, variant),
                        "replicate": replicate,
                        "run_order": run_order,
                        "status": status,
                        "skip_reason": why,
                    }
                )
                continue
            result = run_one(suite, task, variant, replicate, run_dir, args)
            result["run_order"] = run_order
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
        "agent": variant_agent(suite, variant),
        "replicate": replicate,
        "status": "running",
        "case_dir": str(case_dir),
    }
    started = time.time()
    try:
        prepare_repo(task, repo_dir)
        agent = variant_agent(suite, variant)
        env = case_env(suite, task, variant, case_dir, repo_dir, agent)
        run_setup_commands(variant, case_dir, repo_dir, env)
        prompt = build_prompt(suite, task, variant, case_dir, repo_dir, agent)
        prompt_path = case_dir / "prompt.md"
        prompt_path.write_text(prompt)
        agent_result = run_agent(agent, suite, variant, repo_dir, case_dir, prompt, args.timeout, env)
        result.update(agent_result)
        result["mechanism_checks"] = run_mechanism_checks(variant, case_dir, repo_dir, result)
        result["success_commands"] = run_success_commands(task, repo_dir, env)
        result["success"] = (
            result.get("agent_exit_code") == 0
            and result.get("token_total") is not None
            and all(cmd["exit_code"] == 0 for cmd in result["success_commands"])
        )
        result["status"] = "ok" if result["success"] else "failed"
        result["methodology_audit"] = audit_run_methodology(result, variant, suite["baseline"])
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
    agent: str,
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
    if agent == "codex":
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
    if variant.get("isolated_home"):
        home = case_dir / "home"
        home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        codex_home = home / ".codex"
    else:
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
    for relative_path, content in (variant.get("codex_home_files") or {}).items():
        output = codex_home / str(relative_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render(str(content), case_dir, repo_dir, variant) + "\n")
    env["CODEX_HOME"] = str(codex_home)


def context_mode_hooks_json() -> dict[str, Any]:
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "local_shell|shell|shell_command|exec_command|container.exec|Bash|Shell|apply_patch|Edit|Write|grep_files|ctx_execute|ctx_execute_file|ctx_batch_execute|ctx_fetch_and_index|ctx_search|ctx_index|mcp__",
                    "hooks": [{"type": "command", "command": "context-mode hook codex pretooluse"}],
                }
            ],
            "PostToolUse": [
                {"hooks": [{"type": "command", "command": "context-mode hook codex posttooluse"}]}
            ],
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "context-mode hook codex sessionstart"}]}
            ],
            "PreCompact": [
                {"hooks": [{"type": "command", "command": "context-mode hook codex precompact"}]}
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
    agent: str,
) -> str:
    prefix = render(variant_prompt_text(variant, agent, "prompt_prefix"), case_dir, repo_dir, variant).strip()
    suffix = render(variant_prompt_text(variant, agent, "prompt_suffix"), case_dir, repo_dir, variant).strip()
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


def variant_prompt_text(variant: dict[str, Any], agent: str, key: str) -> str:
    specific = variant.get(f"{agent}_{key}")
    if specific is not None:
        return str(specific)
    return str(variant.get(key) or "")


def run_agent(
    agent: str,
    suite: dict[str, Any],
    variant: dict[str, Any],
    repo_dir: Path,
    case_dir: Path,
    prompt: str,
    timeout_override: float | None,
    env: dict[str, str],
) -> dict[str, Any]:
    if agent == "codex":
        return run_codex(suite, variant, repo_dir, case_dir, prompt, timeout_override, env)
    if agent == "claude":
        return run_claude(suite, variant, repo_dir, case_dir, prompt, timeout_override, env)
    raise RuntimeError(f"unsupported agent: {agent}")


def agent_command(agent: str, variant: dict[str, Any], args: list[str]) -> list[str]:
    wrapper = variant.get(f"{agent}_wrapper") or variant.get("agent_wrapper")
    if wrapper:
        if isinstance(wrapper, str):
            wrapper_args = [wrapper]
        else:
            wrapper_args = [str(part) for part in wrapper]
        return [*wrapper_args, "--", *args]
    return [agent, *args]


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
    model = suite_model(suite, "codex")
    reasoning = suite_effort(suite, "codex")
    sandbox = render_env(str(suite.get("sandbox") or "workspace-write"))
    codex_args = [
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
    cmd = agent_command("codex", variant, codex_args)
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
        "agent": "codex",
        "model": model,
        "reasoning": reasoning,
        "agent_command": shell_join_redacted(cmd),
        "agent_exit_code": exit_code,
        "agent_timed_out": timed_out,
        "agent_stdout": file_metrics(stdout_path),
        "agent_stderr": file_metrics(stderr_path),
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


def run_claude(
    suite: dict[str, Any],
    variant: dict[str, Any],
    repo_dir: Path,
    case_dir: Path,
    prompt: str,
    timeout_override: float | None,
    env: dict[str, str],
) -> dict[str, Any]:
    stdout_path = case_dir / "claude.stdout"
    stderr_path = case_dir / "claude.stderr"
    timeout = timeout_override or float(suite.get("timeout_seconds") or 1800)
    model = suite_model(suite, "claude")
    effort = suite_effort(suite, "claude")
    claude_args = [
        "-p",
        prompt,
        "--model",
        model,
        "--effort",
        effort,
        "--output-format",
        "json",
        "--permission-mode",
        "bypassPermissions",
    ]
    budget = (
        variant.get("claude_max_budget_usd")
        or suite.get("claude_max_budget_usd")
        or os.environ.get("TOKENSMASH_MAX_BUDGET_USD")
    )
    if budget is not None:
        budget = render_env(str(budget))
    if budget:
        claude_args.extend(["--max-budget-usd", str(budget)])
    mcp_config = variant.get("claude_mcp_config")
    if mcp_config:
        mcp_path = case_dir / "claude-mcp.json"
        mcp_path.write_text(render_json_config(mcp_config, case_dir, repo_dir, variant))
        claude_args.extend(["--mcp-config", str(mcp_path), "--strict-mcp-config"])
    settings = variant.get("claude_settings")
    if settings:
        settings_path = case_dir / "claude-settings.json"
        settings_path.write_text(render_json_config(settings, case_dir, repo_dir, variant))
        claude_args.extend(["--settings", str(settings_path)])
    claude_args.extend(str(arg) for arg in variant.get("claude_args", []) or [])
    cmd = agent_command("claude", variant, claude_args)
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
    payload, parse_error = parse_json_object(read_tail(stdout_path, size=200000))
    usage = extract_claude_usage(payload) if payload else {}
    session_id = payload.get("session_id") if payload else None
    session_path = find_claude_session(str(session_id) if session_id else None, repo_dir, start_time, env)
    return {
        "agent": "claude",
        "model": model,
        "reasoning": effort,
        "agent_command": shell_join_redacted(cmd),
        "agent_exit_code": exit_code,
        "agent_timed_out": timed_out,
        "agent_stdout": file_metrics(stdout_path),
        "agent_stderr": file_metrics(stderr_path),
        "claude_command": shell_join_redacted(cmd),
        "claude_exit_code": exit_code,
        "claude_timed_out": timed_out,
        "claude_stdout": file_metrics(stdout_path),
        "claude_stderr": file_metrics(stderr_path),
        "session_id": session_id,
        "session_path": str(session_path) if session_path else None,
        "token_source": "claude:stdout_usage" if usage else "missing",
        "token_total": usage.get("total_tokens"),
        "token_usage": usage,
        "total_cost_usd": payload.get("total_cost_usd") if payload else None,
        "result_json_error": parse_error,
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


def run_mechanism_checks(
    variant: dict[str, Any],
    case_dir: Path,
    repo_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    agent = str(result.get("agent") or "")
    checks = variant.get(f"{agent}_mechanism_checks") or variant.get("mechanism_checks") or []
    if not checks:
        return {"required": False, "ok": True, "checks": []}
    evidence = []
    for check in checks:
        if not isinstance(check, dict):
            evidence.append({"ok": False, "error": "mechanism check must be an object"})
            continue
        evidence.append(run_mechanism_check(check, case_dir, repo_dir, result))
    return {
        "required": True,
        "ok": bool(evidence) and all(item.get("ok") for item in evidence),
        "checks": evidence,
    }


def run_mechanism_check(
    check: dict[str, Any],
    case_dir: Path,
    repo_dir: Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    typ = str(check.get("type") or "")
    label = str(check.get("label") or typ)
    if typ == "file_min_bytes":
        path = resolve_check_path(check, case_dir, repo_dir, result)
        minimum = int(check.get("min_bytes") or 1)
        actual = path.stat().st_size if path.exists() else 0
        return {"label": label, "type": typ, "ok": actual >= minimum, "path": str(path), "bytes": actual}
    if typ == "file_regex":
        path = resolve_check_path(check, case_dir, repo_dir, result)
        pattern = str(check.get("pattern") or "")
        text = read_bounded(path, int(check.get("max_bytes") or 2_000_000))
        match_count = len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)) if pattern else 0
        minimum = int(check.get("min_matches") or 1)
        return {
            "label": label,
            "type": typ,
            "ok": match_count >= minimum,
            "path": str(path),
            "matches": match_count,
        }
    if typ == "session_regex":
        path_value = result.get("session_path")
        path = Path(str(path_value)).expanduser() if path_value else Path()
        pattern = str(check.get("pattern") or "")
        text = read_bounded(path, int(check.get("max_bytes") or 5_000_000)) if path_value else ""
        match_count = len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)) if pattern else 0
        minimum = int(check.get("min_matches") or 1)
        return {
            "label": label,
            "type": typ,
            "ok": bool(path_value) and match_count >= minimum,
            "path": str(path) if path_value else "",
            "matches": match_count,
        }
    if typ == "session_tool_regex":
        path_value = result.get("session_path")
        path = Path(str(path_value)).expanduser() if path_value else Path()
        pattern = str(check.get("pattern") or "")
        text = session_tool_text(path) if path_value else ""
        match_count = len(re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)) if pattern else 0
        minimum = int(check.get("min_matches") or 1)
        return {
            "label": label,
            "type": typ,
            "ok": bool(path_value) and match_count >= minimum,
            "path": str(path) if path_value else "",
            "matches": match_count,
        }
    if typ == "json_number_gte":
        path = resolve_check_path(check, case_dir, repo_dir, result)
        key_path = str(check.get("key") or "")
        minimum = float(check.get("min") or 0)
        value = json_number_at(path, key_path)
        return {
            "label": label,
            "type": typ,
            "ok": value is not None and value >= minimum,
            "path": str(path),
            "key": key_path,
            "value": value,
            "min": minimum,
        }
    return {"label": label, "type": typ, "ok": False, "error": f"unknown mechanism check type: {typ}"}


def audit_run_methodology(
    result: dict[str, Any],
    variant: dict[str, Any],
    baseline_id: str,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(label: str, ok: bool, detail: str = "") -> None:
        item: dict[str, Any] = {"label": label, "ok": bool(ok)}
        if detail:
            item["detail"] = detail
        checks.append(item)

    usage = result.get("token_usage")
    token_total = result.get("token_total")
    add("agent exited cleanly", result.get("agent_exit_code") == 0)
    add("agent did not time out", not bool(result.get("agent_timed_out")))
    add("verification oracle passed", all(cmd.get("exit_code") == 0 for cmd in result.get("success_commands") or []))
    add("token usage present", isinstance(usage, dict) and token_total is not None)
    if isinstance(usage, dict) and token_total is not None:
        total = int(usage.get("total_tokens") or 0)
        cached = int(usage.get("cached_input_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        add("token total matches usage", int(token_total) == total)
        add("cached input within input", 0 <= cached <= input_tokens)
        add("input plus output matches total", input_tokens + output_tokens == total)

    if str(result.get("variant_id")) != baseline_id:
        add("tool mechanism observed", mechanism_observed(result, baseline_id), mechanism_summary(result, baseline_id))

    case_dir_value = result.get("case_dir")
    session_path_value = result.get("session_path")
    if variant.get("isolated_home"):
        case_dir = Path(str(case_dir_value)).expanduser() if case_dir_value else None
        session_path = Path(str(session_path_value)).expanduser() if session_path_value else None
        add("session path present", session_path is not None and bool(str(session_path)))
        if case_dir and session_path:
            add(
                "session path isolated",
                path_is_relative_to(session_path, case_dir),
                f"{session_path}",
            )
        else:
            add("session path isolated", False)

    return {"ok": all(item["ok"] for item in checks), "checks": checks}


def session_tool_text(path: Path) -> str:
    chunks: list[str] = []
    for obj in iter_jsonl(path):
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
        if not isinstance(payload, dict):
            continue
        typ = str(payload.get("type") or obj.get("type") or "")
        if typ in {"function_call", "custom_tool_call", "mcp_tool_call_begin", "mcp_tool_call_end"}:
            chunks.append(json.dumps(payload, separators=(",", ":")))
            continue
        message = payload.get("message") if isinstance(payload.get("message"), dict) else payload.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                tool_items = [
                    item
                    for item in content
                    if isinstance(item, dict) and str(item.get("type") or "") in {"tool_use", "tool_result"}
                ]
                if tool_items:
                    chunks.append(json.dumps(tool_items, separators=(",", ":")))
    return "\n".join(chunks)


def resolve_check_path(check: dict[str, Any], case_dir: Path, repo_dir: Path, result: dict[str, Any]) -> Path:
    value = str(check.get("path") or "")
    if value == "{session_path}":
        session_path = result.get("session_path")
        return Path(str(session_path)).expanduser() if session_path else Path()
    rendered = render(value, case_dir, repo_dir, {"id": result.get("variant_id")})
    path = Path(rendered).expanduser()
    return path if path.is_absolute() else case_dir / path


def read_bounded(path: Path, max_bytes: int) -> str:
    if not path or not path.exists():
        return ""
    with path.open("rb") as handle:
        return handle.read(max_bytes).decode("utf-8", "replace")


def json_number_at(path: Path, key_path: str) -> float | None:
    if not path.exists():
        return None
    try:
        value: Any = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    for part in key_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, (int, float)):
        return float(value)
    return None


def render(template: str, case_dir: Path, repo_dir: Path, variant: dict[str, Any]) -> str:
    return render_env(template).format(
        run_dir=str(case_dir),
        repo_dir=str(repo_dir),
        variant_id=str(variant.get("id") or ""),
    )


def render_json_config(value: Any, case_dir: Path, repo_dir: Path, variant: dict[str, Any]) -> str:
    if isinstance(value, str):
        rendered = render(value, case_dir, repo_dir, variant)
        try:
            parsed = json.loads(rendered)
        except json.JSONDecodeError:
            return rendered + "\n"
        return json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    rendered = render_config_value(value, case_dir, repo_dir, variant)
    return json.dumps(rendered, indent=2, sort_keys=True) + "\n"


def render_config_value(value: Any, case_dir: Path, repo_dir: Path, variant: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return render(value, case_dir, repo_dir, variant)
    if isinstance(value, list):
        return [render_config_value(item, case_dir, repo_dir, variant) for item in value]
    if isinstance(value, dict):
        return {str(key): render_config_value(child, case_dir, repo_dir, variant) for key, child in value.items()}
    return value


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


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, str(exc)
        if isinstance(payload, dict):
            return payload, None
    return None, "no JSON object found in stdout"


def extract_claude_usage(payload: dict[str, Any]) -> dict[str, int]:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return {}
    input_tokens = int(raw.get("input_tokens") or 0)
    cache_creation = int(raw.get("cache_creation_input_tokens") or 0)
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    output_tokens = int(raw.get("output_tokens") or 0)
    total = input_tokens + cache_creation + cache_read + output_tokens
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cached_input_tokens": cache_read,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": 0,
        "total_tokens": total,
    }


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


def find_claude_session(
    session_id: str | None,
    repo_dir: Path,
    start_time: float,
    env: dict[str, str],
) -> Path | None:
    roots = claude_session_roots(env)
    if session_id:
        matches = []
        for root in roots:
            matches.extend(root.glob(f"**/{session_id}.jsonl"))
        if matches:
            return max(matches, key=lambda path: path.stat().st_mtime)
    encoded = claude_project_dir_name(repo_dir)
    candidates = []
    cutoff = start_time - 5
    for root in roots:
        project_dir = root / encoded
        search_roots = [project_dir] if project_dir.exists() else [root]
        for search_root in search_roots:
            for path in search_root.glob("**/*.jsonl"):
                try:
                    if path.stat().st_mtime < cutoff:
                        continue
                except OSError:
                    continue
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


def claude_session_roots(env: dict[str, str]) -> list[Path]:
    roots = []
    configured = env.get("CLAUDE_PROJECTS_DIR")
    if configured:
        roots.append(Path(configured))
    roots.append(CLAUDE_PROJECTS_DIR)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen and root.exists():
            unique.append(root)
            seen.add(key)
    return unique


def claude_project_dir_name(path: Path) -> str:
    return str(path.resolve()).replace("/", "-")


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


EVAL_TOOL_ORDER = ["rtk", "context-mode", "semmap", "repomix", "gitingest", "headroom"]
ARTIFACT_TOOLS = {"semmap", "repomix", "gitingest"}
RTK_SAVING_SUBCOMMANDS = {
    "read",
    "grep",
    "find",
    "ls",
    "diff",
    "git",
    "test",
    "cargo",
    "npm",
    "pnpm",
    "yarn",
    "pytest",
    "log",
    "json",
    "summary",
    "deps",
    "env",
    "gh",
    "docker",
    "kubectl",
}


def eval_sessions(args: argparse.Namespace) -> dict[str, Any]:
    tools = selected_eval_tools(args.tools)
    repo_root = Path(args.repo_root).expanduser().resolve()
    session_paths = selected_codex_eval_session_files(
        Path(args.codex_root).expanduser(),
        list(args.session or []),
        args.latest,
    )
    sessions, pressure_events, file_refs = extract_codex_eval_pressure(session_paths, repo_root)
    sample = sample_summary(sessions, args.sample_name)
    artifact_rows = [] if args.no_artifacts else eval_artifact_tools(tools, repo_root, file_refs, args.semmap_bin)
    headroom_row = eval_headroom_perf(Path(args.headroom_perf).expanduser()) if args.headroom_perf else None
    rows = []
    for tool in tools:
        if tool in ARTIFACT_TOOLS:
            row = next((item for item in artifact_rows if item["tool_id"] == tool), missing_eval_row(tool))
        elif tool == "headroom":
            row = headroom_row or missing_eval_row(
                tool,
                verdict="needs --headroom-perf evidence; no live canary is run by default",
            )
        else:
            row = eval_pressure_tool(tool, [event for event in pressure_events if event["tool_id"] == tool])
        rows.append(row)
    return {
        "schema": 1,
        "kind": "tokensmash-eval-sessions",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "agent": "codex",
        "repo_root": str(repo_root),
        "sample": sample,
        "session_count": len(sessions),
        "sessions": sessions,
        "rows": rows,
        "privacy": [
            "raw prompts, assistant messages, and tool outputs are not emitted",
            "session file paths are hashed",
            "commands are retained only to attribute mechanism use",
        ],
    }


def selected_eval_tools(value: str) -> list[str]:
    aliases = {"context_mode": "context-mode", "ctx": "context-mode"}
    requested = []
    for raw in (value or "").split(","):
        tool = aliases.get(raw.strip().lower(), raw.strip().lower())
        if tool and tool not in requested:
            requested.append(tool)
    allowed = set(EVAL_TOOL_ORDER)
    unknown = sorted(set(requested) - allowed)
    require(not unknown, f"unknown eval tool(s): {', '.join(unknown)}")
    return requested or list(EVAL_TOOL_ORDER)


def selected_codex_eval_session_files(root: Path, explicit: list[str], latest: int) -> list[Path]:
    selected: list[Path] = []
    seen: set[str] = set()
    for path in resolve_session_inputs(explicit):
        key = str(path.resolve())
        if key not in seen:
            selected.append(path)
            seen.add(key)
    if selected:
        return selected
    remaining = max(0, int(latest or 0) - len(selected))
    if remaining and root.exists():
        candidates = recent_session_files(root, cutoff=0, limit=max(remaining * 20, remaining))
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            selected.append(path)
            seen.add(key)
            if len(selected) >= int(latest or 0):
                break
    return selected


def resolve_session_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        value = str(raw).strip()
        if not value:
            continue
        if value.startswith("@"):
            paths.extend(resolve_session_manifest(Path(value[1:]).expanduser()))
            continue
        expanded = glob.glob(os.path.expanduser(value))
        if expanded:
            for match in expanded:
                paths.extend(resolve_session_path(Path(match).expanduser()))
            continue
        paths.extend(resolve_session_path(Path(value).expanduser()))
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths


def resolve_session_manifest(path: Path) -> list[Path]:
    try:
        values = [line.strip() for line in path.read_text().splitlines()]
    except OSError:
        return []
    entries = []
    for value in values:
        if not value or value.startswith("#"):
            continue
        entry = Path(value).expanduser()
        entries.append(str(entry if entry.is_absolute() else path.parent / entry))
    return resolve_session_inputs(entries)


def resolve_session_path(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.suffix in {".jsonl", ".json"}:
        return [path]
    if path.is_dir():
        return recent_session_files(path, cutoff=0, limit=1_000_000)
    return []


def extract_codex_eval_pressure(
    session_paths: list[Path],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    sessions = []
    events = []
    file_refs: set[str] = set()
    for path in session_paths:
        session, session_events, session_files = extract_one_codex_eval_session(path, repo_root)
        sessions.append(session)
        events.extend(session_events)
        file_refs.update(session_files)
    return sessions, events, file_refs


def extract_one_codex_eval_session(path: Path, repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    calls: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    file_refs: set[str] = set()
    token_usage: dict[str, Any] = {}
    event_count = 0
    tool_calls = 0
    tool_output_bytes = 0
    cwd = ""
    for obj in iter_jsonl(path):
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else obj
        if not isinstance(payload, dict):
            continue
        event_count += 1
        typ = str(payload.get("type") or obj.get("type") or "")
        cwd = cwd or str((obj.get("payload") or {}).get("cwd") or payload.get("cwd") or "")
        if typ == "turn_context" and isinstance(payload.get("cwd"), str):
            cwd = str(payload["cwd"])
        if typ == "token_count":
            candidate = ((payload.get("info") or {}).get("total_token_usage") or {})
            if isinstance(candidate, dict):
                token_usage = candidate
        if typ in {"function_call", "custom_tool_call", "mcp_tool_call_begin"}:
            tool_calls += 1
            call_id = str(payload.get("call_id") or "")
            if call_id:
                calls[call_id] = codex_call_info(payload)
                command = str(calls[call_id].get("command") or "")
                file_refs.update(repo_relative_paths_from_text(command, repo_root))
        if typ in {"function_call_output", "custom_tool_call_output", "mcp_tool_call_end"}:
            output = codex_output_text(payload)
            tool_output_bytes += len(output.encode("utf-8", "replace"))
            if not output:
                continue
            call_id = str(payload.get("call_id") or "")
            call = calls.get(call_id, {})
            tool_id = classify_codex_pressure_event(call, payload, output)
            if not tool_id:
                continue
            event = build_pressure_event(tool_id, call, payload, output, path)
            if event:
                events.append(event)
                file_refs.update(repo_relative_paths_from_text(str(call.get("command") or ""), repo_root))
                file_refs.update(repo_relative_paths_from_text(output[:20000], repo_root))
    usage = normalize_codex_token_usage(token_usage)
    stats = {
        "source_file_sha256": sha_text(str(path)),
        "source_file_mtime": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat(),
        "cwd": cwd,
        "event_count": event_count,
        "tool_calls": tool_calls,
        "tool_output_bytes": tool_output_bytes,
        "input_tokens": usage["input_tokens"],
        "cached_input_tokens": usage["cached_input_tokens"],
        "output_tokens": usage["output_tokens"],
        "reasoning_output_tokens": usage["reasoning_output_tokens"],
        "total_tokens": usage["total_tokens"],
        "non_cached_tokens": usage["non_cached_tokens"],
        "pressure_events": len(events),
    }
    return stats, events, file_refs


def sample_summary(sessions: list[dict[str, Any]], sample_name: str) -> dict[str, Any]:
    total = sum(int(session.get("total_tokens") or 0) for session in sessions)
    cached = sum(int(session.get("cached_input_tokens") or 0) for session in sessions)
    input_tokens = sum(int(session.get("input_tokens") or 0) for session in sessions)
    output = sum(int(session.get("output_tokens") or 0) for session in sessions)
    reasoning = sum(int(session.get("reasoning_output_tokens") or 0) for session in sessions)
    non_cached = sum(int(session.get("non_cached_tokens") or 0) for session in sessions)
    name = sample_name or ("supplied sessions" if sessions else "empty sample")
    return {
        "name": name,
        "session_count": len(sessions),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "reasoning_output_tokens": reasoning,
        "total_tokens": total,
        "non_cached_tokens": non_cached,
        "cached_share_percent": round(cached / total * 100, 1) if total else None,
    }


def codex_call_info(payload: dict[str, Any]) -> dict[str, Any]:
    invocation = payload.get("invocation") if isinstance(payload.get("invocation"), dict) else {}
    arguments = codex_call_arguments(payload)
    command = ""
    for key in ("cmd", "command", "shell_command"):
        value = arguments.get(key)
        if isinstance(value, str):
            command = value
            break
    return {
        "name": str(payload.get("name") or invocation.get("tool") or ""),
        "namespace": str(payload.get("namespace") or invocation.get("server") or ""),
        "arguments": arguments,
        "command": command,
    }


def codex_call_arguments(payload: dict[str, Any]) -> dict[str, Any]:
    invocation = payload.get("invocation") if isinstance(payload.get("invocation"), dict) else {}
    for value in (payload.get("arguments"), payload.get("input"), invocation.get("arguments")):
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


def classify_codex_pressure_event(call: dict[str, Any], payload: dict[str, Any], output: str) -> str | None:
    name = str(call.get("name") or "")
    namespace = str(call.get("namespace") or "")
    command = str(call.get("command") or "")
    invocation = payload.get("invocation") if isinstance(payload.get("invocation"), dict) else {}
    if not name and invocation:
        name = str(invocation.get("tool") or "")
        namespace = str(invocation.get("server") or "")
    if is_context_mode_call(name, namespace):
        return "context-mode"
    if is_rtk_saving_command(command):
        return "rtk"
    lowered = f"{name} {namespace} {command} {output[:400]}".lower()
    if "repomix" in lowered:
        return "repomix"
    if "gitingest" in lowered:
        return "gitingest"
    if "semmap" in lowered:
        return "semmap"
    if "headroom" in lowered:
        return "headroom"
    return None


def is_context_mode_call(name: str, namespace: str) -> bool:
    lowered = f"{namespace} {name}".lower()
    return "context-mode" in lowered or name.lower().startswith("ctx_")


def is_rtk_saving_command(command: str) -> bool:
    if not command:
        return False
    for match in re.finditer(r"(?:^|[;&|]\s*)rtk\s+([A-Za-z0-9:_-]+)", command):
        subcommand = match.group(1).split(":", 1)[0].lower()
        if subcommand in RTK_SAVING_SUBCOMMANDS:
            return True
    return False


def build_pressure_event(
    tool_id: str,
    call: dict[str, Any],
    payload: dict[str, Any],
    output: str,
    session_path: Path,
) -> dict[str, Any] | None:
    before = inferred_before_tokens(tool_id, output)
    after = approximate_text_tokens(output)
    if before is None:
        return None
    return {
        "tool_id": tool_id,
        "session_sha256": sha_text(str(session_path)),
        "call_id_sha256": sha_text(str(payload.get("call_id") or "")),
        "command": str(call.get("command") or ""),
        "name": str(call.get("name") or ""),
        "before_tokens": before,
        "after_tokens": after,
        "improvement_percent": pct_reduction(before, after),
    }


def inferred_before_tokens(tool_id: str, output: str) -> int | None:
    explicit = parse_original_token_count(output)
    if explicit:
        return explicit
    if tool_id == "context-mode":
        return parse_context_mode_before_tokens(output)
    return None


def parse_original_token_count(output: str) -> int | None:
    match = re.search(r"Original token count:\s*([\d,]+)", output)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_context_mode_before_tokens(output: str) -> int | None:
    matches = re.findall(r"\(([\d,]+)\s+lines,\s*([\d.]+)\s*KB\)", output, flags=re.IGNORECASE)
    if not matches:
        return None
    # Context-mode reports the raw source volume it indexed/searched. That is the closest
    # offline proxy for what a raw shell/file read would have pressured into context.
    kb_total = sum(float(kb) for _, kb in matches)
    return max(1, round(kb_total * 256))


def approximate_text_tokens(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


def normalize_codex_token_usage(usage: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(usage.get("input_tokens") or usage.get("promptTokenCount") or 0)
    cached_input_tokens = int(usage.get("cached_input_tokens") or usage.get("cache_read_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("candidatesTokenCount") or 0)
    reasoning_output_tokens = int(usage.get("reasoning_output_tokens") or 0)
    total_tokens = int(
        usage.get("total_tokens")
        or usage.get("totalTokenCount")
        or (input_tokens + cached_input_tokens + output_tokens + reasoning_output_tokens)
    )
    non_cached_tokens = max(0, total_tokens - cached_input_tokens)
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
        "total_tokens": total_tokens,
        "non_cached_tokens": non_cached_tokens,
    }


def repo_relative_paths_from_text(text: str, repo_root: Path) -> set[str]:
    if not text:
        return set()
    paths: set[str] = set()
    repo = str(repo_root)
    escaped_repo = re.escape(repo.rstrip("/"))
    patterns = [
        rf"{escaped_repo}/[A-Za-z0-9_@%+=:,./~ -]+\.[A-Za-z0-9]{{1,12}}",
        r"(?<![\w/-])(?:\.?/)?[A-Za-z0-9_@%+=:,./~-]+\.[A-Za-z0-9]{1,12}",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            raw = match.group(0).strip().strip("`'\"),:]")
            rel = repo_relative_path(raw, repo_root)
            if rel:
                paths.add(rel)
    return paths


def repo_relative_path(value: str, repo_root: Path) -> str | None:
    try:
        path = Path(value).expanduser()
    except (OSError, RuntimeError):
        return None
    if not path.is_absolute():
        path = repo_root / path
    try:
        resolved = path.resolve()
        rel = resolved.relative_to(repo_root)
    except (OSError, ValueError):
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    if any(part in {".git", ".venv", "node_modules", "dist", "__pycache__"} for part in rel.parts):
        return None
    return rel.as_posix()


def eval_pressure_tool(tool_id: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    before = sum(int(event["before_tokens"]) for event in events)
    after = sum(int(event["after_tokens"]) for event in events)
    row = base_eval_row(tool_id)
    row.update(
        {
            "mechanism_fired": "yes" if events else "no",
            "evaluated_on_real_session": "yes" if events else "no",
            "token_pressure_before": before if events else None,
            "token_spend_with_tool": after if events else None,
            "percent_improvement": pct_reduction(before, after) if events else None,
            "future_session_confidence_percent": pressure_confidence(tool_id, events, before, after),
            "sample_count": len(events),
            "verdict": pressure_verdict(tool_id, events, before, after),
        }
    )
    return row


def pressure_confidence(tool_id: str, events: list[dict[str, Any]], before: int, after: int) -> int:
    if not events:
        return 0
    if before <= 0 or after < 0:
        return 35
    if tool_id == "context-mode":
        if len(events) >= 10 and before >= 100000:
            return 94
        return 92 if len(events) >= 3 and before >= 50000 else 82
    if tool_id == "rtk":
        if len(events) >= 10 and before >= 50000:
            return 93
        return 90 if len(events) >= 3 and before >= 10000 else 80
    pct = pct_reduction(before, after)
    return 70 if pct >= 20 else 55


def pressure_verdict(tool_id: str, events: list[dict[str, Any]], before: int, after: int) -> str:
    if not events:
        return "not observed in selected sessions"
    pct = pct_reduction(before, after)
    if pct > 0:
        return f"observed {tool_id} savings on selected Codex sessions"
    if pct == 0:
        return f"observed {tool_id}, no measured token reduction"
    return f"observed {tool_id}, but selected events grew token pressure"


def eval_artifact_tools(
    tools: list[str],
    repo_root: Path,
    file_refs: set[str],
    semmap_bin: str,
) -> list[dict[str, Any]]:
    wanted = [tool for tool in tools if tool in ARTIFACT_TOOLS]
    if not wanted:
        return []
    from_sessions = bool(file_refs)
    working_set = focused_working_set(repo_root, file_refs)
    raw_tokens = raw_file_tokens(repo_root, working_set)
    rows = []
    tmp_root = Path.home() / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tokensmash-eval-", dir=str(tmp_root)) as tmp:
        temp_dir = Path(tmp)
        for tool in wanted:
            rows.append(eval_artifact_tool(tool, repo_root, working_set, raw_tokens, temp_dir, semmap_bin, from_sessions))
    return rows


def focused_working_set(repo_root: Path, file_refs: set[str]) -> list[str]:
    selected = [path for path in sorted(file_refs) if (repo_root / path).exists()]
    if selected:
        return selected[:40]
    fallbacks = ["src/tokensmash/cli.py", "README.md", "pyproject.toml"]
    return [path for path in fallbacks if (repo_root / path).exists()]


def raw_file_tokens(repo_root: Path, rel_paths: list[str]) -> int:
    total = 0
    for rel in rel_paths:
        path = repo_root / rel
        try:
            total += approximate_text_tokens(path.read_text(errors="ignore"))
        except OSError:
            continue
    return total


def eval_artifact_tool(
    tool_id: str,
    repo_root: Path,
    working_set: list[str],
    raw_tokens: int,
    temp_dir: Path,
    semmap_bin: str,
    from_sessions: bool,
) -> dict[str, Any]:
    row = base_eval_row(tool_id)
    if not working_set or raw_tokens <= 0:
        row["verdict"] = "no readable focused working set"
        return row
    output_path = temp_dir / f"{tool_id}.txt"
    semmap_cache = repo_root / ".semmap"
    semmap_cache_preexisting = semmap_cache.exists()
    try:
        proc = run_artifact_tool(tool_id, repo_root, working_set, output_path, semmap_bin)
    except subprocess.TimeoutExpired:
        proc = None
        row["verdict"] = f"{tool_id} timed out locally"
    if tool_id == "semmap" and not semmap_cache_preexisting and semmap_cache.exists():
        shutil.rmtree(semmap_cache, ignore_errors=True)
    if row["verdict"].endswith("timed out locally"):
        return row
    if proc is None:
        row["verdict"] = f"{tool_id} executable not found"
        return row
    if proc.returncode != 0 or not output_path.exists():
        row["verdict"] = f"{tool_id} failed locally"
        row["error_sha256"] = hashlib.sha256((proc.stderr or b"") + (proc.stdout or b"")).hexdigest()
        return row
    try:
        artifact_text = output_path.read_text(errors="ignore")
    except OSError:
        artifact_text = ""
    after = approximate_text_tokens(artifact_text)
    coverage = artifact_coverage(artifact_text, working_set)
    row.update(
        {
            "mechanism_fired": "yes",
            "evaluated_on_real_session": "yes" if from_sessions else "no",
            "token_pressure_before": raw_tokens,
            "token_spend_with_tool": after,
            "percent_improvement": pct_reduction(raw_tokens, after),
            "future_session_confidence_percent": artifact_confidence(tool_id, raw_tokens, after, coverage, len(working_set)),
            "sample_count": len(working_set),
            "coverage": f"{coverage}/{len(working_set)} files",
            "verdict": artifact_verdict(tool_id, raw_tokens, after, coverage, len(working_set)),
        }
    )
    return row


def run_artifact_tool(
    tool_id: str,
    repo_root: Path,
    working_set: list[str],
    output_path: Path,
    semmap_bin: str,
) -> subprocess.CompletedProcess[bytes] | None:
    if tool_id == "semmap":
        binary = resolve_semmap_bin(semmap_bin)
        if not binary:
            return None
        semmap_output = output_path.with_name("SEMMAP.md")
        return subprocess.run(
            [binary, "generate", "--root", str(repo_root), "--output", str(semmap_output), "--chat-output", str(output_path)],
            cwd=repo_root,
            capture_output=True,
            check=False,
            timeout=120,
        )
    if tool_id == "repomix":
        if not shutil.which("npx"):
            return None
        include = ",".join(working_set)
        return subprocess.run(
            [
                "npx",
                "--yes",
                "repomix@latest",
                str(repo_root),
                "--include",
                include,
                "--compress",
                "--style",
                "xml",
                "--output",
                str(output_path),
                "--quiet",
            ],
            cwd=repo_root,
            capture_output=True,
            check=False,
            timeout=120,
        )
    if tool_id == "gitingest":
        if not shutil.which("uvx"):
            return None
        include_args = [arg for rel in working_set for arg in ("-i", rel)]
        return subprocess.run(
            ["uvx", "gitingest", str(repo_root), *include_args, "-o", str(output_path)],
            cwd=repo_root,
            capture_output=True,
            check=False,
            timeout=120,
        )
    return None


def resolve_semmap_bin(configured: str) -> str | None:
    candidates = [
        configured,
        shutil.which("semmap") or "",
        str(Path.home() / ".tmp" / "semmap-eval" / "install" / "bin" / "semmap"),
        str(Path.home() / ".tmp" / "semmap-eval" / "target" / "release" / "semmap"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
        resolved = shutil.which(candidate) if candidate else None
        if resolved:
            return resolved
    return None


def artifact_coverage(text: str, working_set: list[str]) -> int:
    return sum(1 for path in working_set if path in text or Path(path).name in text)


def artifact_confidence(tool_id: str, before: int, after: int, coverage: int, total: int) -> int:
    if before <= 0 or total <= 0:
        return 0
    coverage_ratio = coverage / total
    pct = pct_reduction(before, after)
    if coverage_ratio < 0.8:
        return 45
    if tool_id == "repomix" and pct >= 20:
        return 90
    if tool_id == "semmap" and pct >= 50:
        return 88
    if tool_id == "gitingest":
        return 86 if pct <= 10 else 82
    return 70 if pct > 0 else 65


def artifact_verdict(tool_id: str, before: int, after: int, coverage: int, total: int) -> str:
    if total and coverage / total < 0.8:
        return f"{tool_id} output did not cover enough focused files"
    pct = pct_reduction(before, after)
    if pct > 0:
        return f"{tool_id} reduced focused working-set token pressure"
    if pct == 0:
        return f"{tool_id} matched raw working-set token pressure"
    return f"{tool_id} increased focused working-set token pressure"


def eval_headroom_perf(path: Path) -> dict[str, Any]:
    row = base_eval_row("headroom")
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        row["verdict"] = "headroom perf evidence unreadable"
        return row
    records = data if isinstance(data, list) else data.get("records") if isinstance(data, dict) else []
    before = 0
    after = 0
    seen: set[tuple[int, int, str]] = set()
    for item in records or []:
        if not isinstance(item, dict):
            continue
        b = first_int(item, ("tokens_before", "before_tokens", "before"))
        a = first_int(item, ("tokens_after", "after_tokens", "after"))
        key = (b, a, str(item.get("request_id") or item.get("id") or ""))
        if b <= 0 or a < 0 or key in seen:
            continue
        seen.add(key)
        before += b
        after += a
    if not seen:
        row["verdict"] = "headroom evidence had no before/after records"
        return row
    row.update(
        {
            "mechanism_fired": "yes",
            "evaluated_on_real_session": "no",
            "token_pressure_before": before,
            "token_spend_with_tool": after,
            "percent_improvement": pct_reduction(before, after),
            "future_session_confidence_percent": 62 if before < 100000 else 78,
            "sample_count": len(seen),
            "verdict": "headroom proxy evidence observed; needs real session A/B for default-enable confidence",
        }
    )
    return row


def first_int(item: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.replace(",", "").isdigit():
            return int(value.replace(",", ""))
    return 0


def base_eval_row(tool_id: str) -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "tool": display_tool_name(tool_id),
        "mechanism_fired": "no",
        "evaluated_on_real_session": "no",
        "token_pressure_before": None,
        "token_spend_with_tool": None,
        "cached_tokens_before": None,
        "cached_tokens_with_tool": None,
        "non_cached_tokens_before": None,
        "non_cached_tokens_with_tool": None,
        "percent_improvement": None,
        "non_cached_percent_improvement": None,
        "future_session_confidence_percent": 0,
        "sample_count": 0,
        "verdict": "not evaluated",
    }


def missing_eval_row(tool_id: str, verdict: str | None = None) -> dict[str, Any]:
    row = base_eval_row(tool_id)
    if verdict:
        row["verdict"] = verdict
    return row


def display_tool_name(tool_id: str) -> str:
    return {
        "rtk": "RTK",
        "context-mode": "context-mode",
        "semmap": "SEMMAP",
        "repomix": "Repomix",
        "gitingest": "Gitingest",
        "headroom": "Headroom",
    }.get(tool_id, tool_id)


def pct_reduction(before: int | float, after: int | float) -> float:
    if before <= 0:
        return 0.0
    return round((float(before) - float(after)) / float(before) * 100, 1)


def print_eval_sessions_table(result: dict[str, Any]) -> None:
    sample = result.get("sample") if isinstance(result.get("sample"), dict) else {}
    if sample:
        cached_share = sample.get("cached_share_percent")
        share = "n/a" if cached_share is None else f"{float(cached_share):.1f}%"
        print(
            "sample: "
            f"{sample.get('name') or 'supplied sessions'}; "
            f"sessions {int(sample.get('session_count') or 0):,}; "
            f"total {format_int_cell(sample.get('total_tokens'))}; "
            f"cached {format_int_cell(sample.get('cached_input_tokens'))}; "
            f"non-cached {format_int_cell(sample.get('non_cached_tokens'))}; "
            f"cached share {share}"
        )
        print()
    headers = [
        "tool",
        "mechanism fired?",
        "real session?",
        "baseline token spend",
        "token spend with tool",
        "percent improvement",
        "future-session confidence",
        "verdict",
    ]
    rows = []
    for row in result.get("rows", []):
        rows.append(
            [
                row["tool"],
                row["mechanism_fired"],
                row["evaluated_on_real_session"],
                format_int_cell(row["token_pressure_before"]),
                format_int_cell(row["token_spend_with_tool"]),
                format_percent_cell(row["percent_improvement"]),
                f"{row['future_session_confidence_percent']}%",
                row["verdict"],
            ]
        )
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def format_int_cell(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def format_percent_cell(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}%"


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


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def print_table(results: dict[str, Any]) -> None:
    baseline_id = str(results["baseline"])
    rows = comparison_rows(results, baseline_id)
    headers = [
        "tool",
        "baseline total",
        "tool total",
        "total improvement",
        "baseline non-cached",
        "tool non-cached",
        "non-cached improvement",
        "confidence",
        "mechanism",
    ]
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def print_aggregate_table(results_list: list[dict[str, Any]], strict: bool = False) -> None:
    headers = [
        "tool",
        "pairs",
        "baseline total",
        "tool total",
        "total improvement",
        "baseline non-cached",
        "tool non-cached",
        "non-cached improvement",
        "baseline cost",
        "tool cost",
        "cost improvement",
        "cost 95% ci",
        "positive non-cached",
        "mechanism",
        "methodology",
    ]
    rows = aggregate_rows(results_list, strict=strict)
    widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def aggregate_rows(results_list: list[dict[str, Any]], strict: bool = False) -> list[list[str]]:
    aggregate: dict[str, dict[str, Any]] = {}
    for results in results_list:
        baseline_id = str(results["baseline"])
        grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        for run in results.get("runs", []):
            if not run.get("success"):
                continue
            key = (str(run.get("task_id")), int(run.get("replicate") or 1))
            grouped.setdefault(key, {})[str(run.get("variant_id"))] = run
        for by_variant in grouped.values():
            baseline = by_variant.get(baseline_id)
            if not baseline:
                continue
            base_tokens = baseline.get("token_total")
            base_non_cached = run_non_cached_tokens(baseline)
            if base_tokens is None or base_non_cached is None:
                continue
            for variant_id, tool in by_variant.items():
                if variant_id == baseline_id:
                    continue
                bucket = aggregate.setdefault(
                    variant_id,
                    {
                        "pairs": 0,
                        "invalid": 0,
                        "base_total": 0,
                        "tool_total": 0,
                        "base_non_cached": 0,
                        "tool_non_cached": 0,
                        "base_cost": 0.0,
                        "tool_cost": 0.0,
                        "positive_non_cached": 0,
                        "cost_improvements": [],
                        "methodology_excluded": 0,
                    },
                )
                if not mechanism_observed(tool, baseline_id):
                    bucket["invalid"] += 1
                    continue
                if strict and not (methodology_passed(baseline) and methodology_passed(tool)):
                    bucket["methodology_excluded"] += 1
                    continue
                tool_tokens = tool.get("token_total")
                tool_non_cached = run_non_cached_tokens(tool)
                if tool_tokens is None or tool_non_cached is None:
                    bucket["invalid"] += 1
                    continue
                base_cost = run_api_cost(baseline)
                tool_cost = run_api_cost(tool)
                if base_cost is None or tool_cost is None:
                    bucket["invalid"] += 1
                    continue
                bucket["pairs"] += 1
                bucket["base_total"] += int(base_tokens)
                bucket["tool_total"] += int(tool_tokens)
                bucket["base_non_cached"] += int(base_non_cached)
                bucket["tool_non_cached"] += int(tool_non_cached)
                bucket["base_cost"] += base_cost
                bucket["tool_cost"] += tool_cost
                bucket["cost_improvements"].append((base_cost - tool_cost) / base_cost * 100 if base_cost else 0.0)
                if tool_non_cached < base_non_cached:
                    bucket["positive_non_cached"] += 1

    rows: list[list[str]] = []
    for variant_id, bucket in sorted(aggregate.items()):
        pairs = int(bucket["pairs"])
        invalid = int(bucket["invalid"])
        if pairs == 0:
            rows.append(
                [
                    variant_id,
                    "0",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    "0/0",
                    "not observed",
                    aggregate_methodology_cell(bucket),
                ]
            )
            continue
        mechanism = "observed" if invalid == 0 else f"observed ({invalid} excluded)"
        rows.append(
            [
                variant_id,
                str(pairs),
                format_int_cell(bucket["base_total"]),
                format_int_cell(bucket["tool_total"]),
                pct_improvement(bucket["base_total"], bucket["tool_total"]),
                format_int_cell(bucket["base_non_cached"]),
                format_int_cell(bucket["tool_non_cached"]),
                pct_improvement(bucket["base_non_cached"], bucket["tool_non_cached"]),
                format_cost_cell(bucket["base_cost"]),
                format_cost_cell(bucket["tool_cost"]),
                pct_improvement_float(bucket["base_cost"], bucket["tool_cost"]),
                ci_cell(bucket["cost_improvements"]),
                f"{bucket['positive_non_cached']}/{pairs}",
                mechanism,
                aggregate_methodology_cell(bucket),
            ]
        )
    return rows or [["no comparable tool rows", "0", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "0/0", "n/a", "n/a"]]


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
        base_non_cached_total = 0
        tool_non_cached_total = 0
        non_cached_pairs = 0
        measured_pairs = 0
        invalid_pairs = 0
        for by_variant in grouped.values():
            base = by_variant.get(baseline_id)
            tool = by_variant.get(variant_id)
            if not base or not tool:
                continue
            if not mechanism_observed(tool, baseline_id):
                invalid_pairs += 1
                continue
            base_tokens = base.get("token_total")
            tool_tokens = tool.get("token_total")
            if base_tokens is None or tool_tokens is None:
                continue
            base_total += int(base_tokens)
            tool_total += int(tool_tokens)
            base_non_cached = run_non_cached_tokens(base)
            tool_non_cached = run_non_cached_tokens(tool)
            if base_non_cached is not None and tool_non_cached is not None:
                base_non_cached_total += base_non_cached
                tool_non_cached_total += tool_non_cached
                non_cached_pairs += 1
            measured_pairs += 1
        if measured_pairs == 0 and invalid_pairs == 0:
            rows.append([variant_id, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "none", "not observed"])
        elif measured_pairs == 0:
            rows.append([variant_id, "not reported", "not reported", "not measured", "n/a", "n/a", "n/a", "none", "not observed"])
        elif invalid_pairs:
            rows.append(
                [
                    variant_id,
                    format_int_cell(base_total),
                    format_int_cell(tool_total),
                    pct_improvement(base_total, tool_total),
                    format_int_cell(base_non_cached_total) if non_cached_pairs else "n/a",
                    format_int_cell(tool_non_cached_total) if non_cached_pairs else "n/a",
                    pct_improvement(base_non_cached_total, tool_non_cached_total) if non_cached_pairs else "n/a",
                    actionable_confidence(base_total, tool_total, measured_pairs, invalid_pairs),
                    "partially observed",
                ]
            )
        else:
            rows.append(
                [
                    variant_id,
                    format_int_cell(base_total),
                    format_int_cell(tool_total),
                    pct_improvement(base_total, tool_total),
                    format_int_cell(base_non_cached_total) if non_cached_pairs else "n/a",
                    format_int_cell(tool_non_cached_total) if non_cached_pairs else "n/a",
                    pct_improvement(base_non_cached_total, tool_non_cached_total) if non_cached_pairs else "n/a",
                    actionable_confidence(base_total, tool_total, measured_pairs, invalid_pairs),
                    "observed",
                ]
            )
    return rows or [["no comparable tool rows", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "none", "n/a"]]


def run_non_cached_tokens(run: dict[str, Any]) -> int | None:
    usage = run.get("token_usage")
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    cached = usage.get("cached_input_tokens") or usage.get("cache_read_input_tokens") or 0
    if total is None:
        total = run.get("token_total")
    if total is None:
        return None
    return max(0, int(total) - int(cached or 0))


def run_api_cost(run: dict[str, Any]) -> float | None:
    usage = run.get("token_usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or usage.get("cache_read_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    fresh = max(0, input_tokens - cached)
    return (
        fresh / 1_000_000 * GPT55_SHORT_CONTEXT_PRICES["fresh_input_per_m"]
        + cached / 1_000_000 * GPT55_SHORT_CONTEXT_PRICES["cached_input_per_m"]
        + output / 1_000_000 * GPT55_SHORT_CONTEXT_PRICES["output_per_m"]
    )


def methodology_passed(run: dict[str, Any]) -> bool:
    audit = run.get("methodology_audit")
    if not isinstance(audit, dict):
        return False
    return audit.get("ok") is True


def aggregate_methodology_cell(bucket: dict[str, Any]) -> str:
    excluded = int(bucket.get("methodology_excluded") or 0)
    pairs = int(bucket.get("pairs") or 0)
    if excluded:
        return f"strict pass ({excluded} excluded)"
    if pairs:
        return "pass"
    return "n/a"


def mechanism_observed(run: dict[str, Any], baseline_id: str) -> bool:
    if str(run.get("variant_id")) == baseline_id:
        return True
    checks = run.get("mechanism_checks")
    return isinstance(checks, dict) and checks.get("required") is True and checks.get("ok") is True


def mechanism_summary(run: dict[str, Any], baseline_id: str) -> str:
    if str(run.get("variant_id")) == baseline_id:
        return "baseline"
    checks = run.get("mechanism_checks")
    if not isinstance(checks, dict) or not checks.get("required"):
        return "not checked"
    if checks.get("ok"):
        return "observed"
    failed = []
    for item in checks.get("checks") or []:
        if isinstance(item, dict) and not item.get("ok"):
            failed.append(str(item.get("label") or item.get("type") or "check"))
    return "not observed" + (f": {', '.join(failed)}" if failed else "")


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
        "Primary metric: paired session-token spend per successful task run, with total and non-cached tokens separated.",
        "Each row uses the paired baseline from the same result batch, task, and replicate.",
        "",
        markdown_table(
            [
                "tool",
                "how applied",
                "agent",
                "baseline total",
                "tool total",
                "total result",
                "baseline non-cached",
                "tool non-cached",
                "non-cached result",
                "confidence",
                "mechanism",
                "model",
                "reasoning",
                "task",
                "oracle",
                "oracle result",
            ],
            [
                [
                    row["tool"],
                    row["tool_label"],
                    row["agent"],
                    row["baseline_tokens_display"],
                    row["tool_tokens_display"],
                    row["token_result"],
                    row["baseline_non_cached_display"],
                    row["tool_non_cached_display"],
                    row["non_cached_result"],
                    row["confidence"],
                    row["mechanism"],
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
                "mechanism",
                "duration",
            ],
            [
                [
                    row["tool"],
                    usage_cell(row, "baseline_usage", "input_tokens"),
                    usage_cell(row, "baseline_usage", "output_tokens"),
                    usage_cell(row, "baseline_usage", "reasoning_output_tokens"),
                    usage_cell(row, "tool_usage", "input_tokens"),
                    usage_cell(row, "tool_usage", "output_tokens"),
                    usage_cell(row, "tool_usage", "reasoning_output_tokens"),
                    row["mechanism"],
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
            *[f"- `{path}`" for path in unique_nonempty(row["result_file"] for row in rows)],
            "",
            "## Limitations",
            "",
            "- One task and one replicate per row; use directionally, not as a confidence interval.",
            "- Rows from different result batches have different paired baselines; compare each row to its own baseline.",
            "- Agent integrations differ: hook/proxy/MCP rows are not interchangeable across Codex, Claude, Gemini, or non-coding research sessions.",
            "- A token result is reported only when the task oracle passed and the configured tool mechanism check was observed.",
            "- Confidence is about actionability of the percent change, not proof that the tool is universally better.",
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
            oracle_commands = run.get("success_commands") or baseline.get("success_commands") or []
            observed = mechanism_observed(run, baseline_id)
            baseline_tokens = int(baseline.get("token_total") or 0)
            tool_tokens = int(run.get("token_total") or 0)
            baseline_non_cached = run_non_cached_tokens(baseline)
            tool_non_cached = run_non_cached_tokens(run)
            confidence = actionable_confidence(baseline_tokens, tool_tokens, 1 if observed else 0, 0 if observed else 1)
            rows.append(
                {
                    "tool": variant_id,
                    "tool_label": (variants.get(variant_id) or {}).get("label", variant_id),
                    "baseline_tokens": baseline_tokens,
                    "tool_tokens": tool_tokens,
                    "baseline_tokens_display": str(baseline_tokens) if observed else "not reported",
                    "tool_tokens_display": str(tool_tokens) if observed else "not reported",
                    "token_result": pct_improvement(baseline_tokens, tool_tokens) if observed else "not measured",
                    "baseline_non_cached": baseline_non_cached,
                    "tool_non_cached": tool_non_cached,
                    "baseline_non_cached_display": format_int_cell(baseline_non_cached) if observed else "not reported",
                    "tool_non_cached_display": format_int_cell(tool_non_cached) if observed else "not reported",
                    "non_cached_result": pct_improvement(baseline_non_cached, tool_non_cached)
                    if observed and baseline_non_cached is not None and tool_non_cached is not None
                    else "n/a",
                    "confidence": confidence if observed else "none",
                    "mechanism": mechanism_summary(run, baseline_id),
                    "mechanism_ok": observed,
                    "baseline_usage": baseline.get("token_usage") or {},
                    "tool_usage": run.get("token_usage") or {},
                    "agent": str(run.get("agent") or results.get("agent") or "codex"),
                    "model": str(run.get("model") or results.get("model") or ""),
                    "reasoning": str(run.get("reasoning") or results.get("reasoning") or ""),
                    "task_id": task_id,
                    "repo": str(task.get("repo") or ""),
                    "base_ref": str(task.get("base_ref") or ""),
                    "oracle": format_oracle(oracle_commands, task),
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
    agents = sorted({str(r.get("agent") or "codex") for r in results_list})
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
        "agent": ", ".join(unique_nonempty(agents)),
        "model": ", ".join(unique_nonempty(models)),
        "reasoning effort": ", ".join(unique_nonempty(reasoning)),
        "tasks": ", ".join(unique_nonempty(tasks)),
        "repository": ", ".join(unique_nonempty(repos)),
        "base ref": ", ".join(unique_nonempty(base_refs)),
        "verification oracle": ", ".join(unique_nonempty(oracles)),
        "replicates": "1 per row",
        "sandbox": "workspace-write",
    }


def format_oracle(commands: Any, task: dict[str, Any]) -> str:
    if isinstance(commands, list) and commands:
        values = []
        for command in commands:
            if isinstance(command, dict):
                values.append(str(command.get("command") or ""))
            else:
                values.append(str(command))
        rendered = ", ".join(unique_nonempty(values))
        if rendered:
            return rendered
    return ", ".join(str(cmd) for cmd in task.get("success_commands", []) or []) or "none"


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


def usage_cell(row: dict[str, Any], usage_key: str, token_key: str) -> str:
    if not row.get("mechanism_ok"):
        return "not reported"
    usage = row.get(usage_key)
    return str(usage.get(token_key, 0)) if isinstance(usage, dict) else "0"


def actionable_confidence(baseline: int, tool: int, measured_pairs: int, invalid_pairs: int) -> str:
    if baseline <= 0 or tool <= 0 or measured_pairs <= 0:
        return "none"
    if invalid_pairs:
        return "low"
    delta = abs((baseline - tool) / baseline * 100)
    if measured_pairs < 3:
        return "low"
    if measured_pairs < 5:
        return "medium" if delta >= 10 else "low"
    if delta < 5:
        return "low"
    if delta < 10:
        return "medium"
    return "high"


def format_duration(ms: int) -> str:
    if ms <= 0:
        return "n/a"
    return f"{ms / 1000:.1f}s"


def pct_improvement(baseline: int, tool: int) -> str:
    if baseline <= 0:
        return "n/a"
    return f"{((baseline - tool) / baseline * 100):+.1f}%"


def pct_improvement_float(baseline: float, tool: float) -> str:
    if baseline <= 0:
        return "n/a"
    return f"{((baseline - tool) / baseline * 100):+.1f}%"


def format_cost_cell(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def ci_cell(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:+.1f}%"
    samples = bootstrap_means(values, iterations=2000, seed=17)
    lower = percentile(samples, 2.5)
    upper = percentile(samples, 97.5)
    return f"{lower:+.1f}%..{upper:+.1f}%"


def bootstrap_means(values: list[float], iterations: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    out = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        out.append(total / n)
    out.sort()
    return out


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * pct / 100
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


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
