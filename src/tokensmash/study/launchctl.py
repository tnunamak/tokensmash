"""Portable actuation for wrapper-class tools (contract §10).

Public API:
    resolve_launch(agent_cli, argv, study_dir=None) -> dict
    install_report(apply=False) -> dict

resolve_launch:
- NEVER raises; every failure path returns a plain-exec decision.
- Returns {"exec": [...], "env": {...merged...}, "arm": str|None,
           "resolution": dict|None, "error": str}  (error key absent on success)
- Caller is responsible for os.execvpe(*decision["exec"][0], decision["exec"], decision["env"])

install_report:
- check mode (apply=False): returns per-item presence + snippet.
- apply=True: idempotently edits the two JSON hook files; never touches shell rc.
- Path overrides: TOKENSMASH_CLAUDE_SETTINGS, TOKENSMASH_CODEX_HOOKS env vars.

Design note on PATH shims:
  shutil.which() is used for real-binary resolution. The project uses shell
  functions to shadow the agent commands, so which() finds the real binary
  at the original PATH location. PATH shims are not supported.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import tokensmash.schema as _schema
from tokensmash.study import assign as _assign

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tools"

# Map CLI names to the agent identifiers used in the registry's "agents" list.
_CLI_TO_AGENT = {
    "claude": "claude-code",
    "codex": "codex",
}

# Hook command snippets deployed today (must stay in sync with documentation).
_CLAUDE_HOOK_CMD = "tokensmash study link --agent claude-code 2>/dev/null"
_CODEX_HOOK_CMD = "tokensmash study link --agent codex 2>/dev/null"

# Default config paths (overridden by env vars in tests).
_DEFAULT_CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
_DEFAULT_CODEX_HOOKS = Path.home() / ".codex" / "hooks.json"


def _claude_settings_path() -> Path:
    override = os.environ.get("TOKENSMASH_CLAUDE_SETTINGS")
    return Path(override) if override else _DEFAULT_CLAUDE_SETTINGS


def _codex_hooks_path() -> Path:
    override = os.environ.get("TOKENSMASH_CODEX_HOOKS")
    return Path(override) if override else _DEFAULT_CODEX_HOOKS


def _load_registry(tool: str) -> dict:
    """Load data/tools/<tool>.json; raises FileNotFoundError if missing."""
    path = _DATA_DIR / f"{tool}.json"
    return json.loads(path.read_text())


def _build_exec_argv(template: list[str], agent_cli: str, argv: list[str]) -> list[str]:
    """Replace '<tool>' in template elements, then append '--' and caller argv.

    The registry on_command already ends with '--' so we append argv after it.
    For off_command (just ['<tool>']) we also append '--' + argv to be consistent,
    but off_command typically has no '--' sentinel — so we just append argv directly.

    Actually: per contract, argv is appended after the literal '--' element in
    on_command. off_command is just ['<tool>'] with argv appended without '--'.
    """
    result = [tok.replace("<tool>", agent_cli) for tok in template]
    # on_command ends with "--"; off_command doesn't. Append argv after the
    # existing "--" element (on_command) or directly (off_command).
    result.extend(argv)
    return result


def _plain_exec(agent_cli: str, argv: list[str], env: dict, error: str | None = None) -> dict:
    """Build a plain-exec decision (real binary, no wrapper)."""
    real = shutil.which(agent_cli)
    decision: dict[str, Any] = {
        "exec": [real, *argv] if real else None,
        "env": env,
        "arm": None,
        "resolution": None,
    }
    if not real:
        decision["exec"] = None
        decision["error"] = f"binary not found: {agent_cli}"
    if error:
        decision["error"] = error
    return decision


# ---------------------------------------------------------------------------
# Public: resolve_launch
# ---------------------------------------------------------------------------


def resolve_launch(
    agent_cli: str,
    argv: list[str],
    study_dir: Path | None = None,
) -> dict:
    """Resolve exec decision for agent_cli + argv.

    Returns a dict with keys:
        exec        list[str] | None  — argv0 + args to pass to os.execvpe
        env         dict              — full environment for the exec
        arm         "on"|"off"|None
        resolution  dict|None         — from arm_for_cwd
        error       str               — only present when something went wrong

    TOKENSMASH_LAUNCH_ACTIVE=1 is always set in the returned env.
    """
    env = {**os.environ, "TOKENSMASH_LAUNCH_ACTIVE": "1"}

    try:
        # Recursion guard: if already inside a tokensmash launch, pass through.
        if os.environ.get("TOKENSMASH_LAUNCH_ACTIVE"):
            return _plain_exec(agent_cli, argv, env)

        # Resolve the arm for the current working directory.
        try:
            resolution = _assign.arm_for_cwd(os.getcwd(), study_dir=study_dir)
        except Exception as exc:
            return _plain_exec(agent_cli, argv, env, error=f"arm_for_cwd failed: {exc}")

        if resolution is None:
            # No live study — plain exec.
            return _plain_exec(agent_cli, argv, env)

        arm = resolution["arm"]
        config = _assign.load_study_config(study_dir)
        tool = (config or {}).get("tool", "headroom")

        try:
            registry = _load_registry(tool)
        except Exception as exc:
            _assign.log_actuation({**resolution, "agent": _CLI_TO_AGENT.get(agent_cli, agent_cli)}, tool=tool, agent_command=agent_cli, study_dir=study_dir)
            return _plain_exec(agent_cli, argv, env, error=f"registry load failed: {exc}")

        # Merge registry env into our env copy (arm "on" and "off" both log;
        # env merge from registry applies regardless so the exec env is consistent).
        reg_env = registry.get("env") or {}
        env.update(reg_env)

        # Log actuation for both arms.
        _assign.log_actuation({**resolution, "agent": _CLI_TO_AGENT.get(agent_cli, agent_cli)}, tool=tool, agent_command=agent_cli, study_dir=study_dir)

        if arm == "on":
            cmd_template = registry.get("on_command", [])
            exec_argv = _build_exec_argv(cmd_template, agent_cli, argv)
            real = exec_argv[0] if exec_argv else None
            # Verify the wrapper binary exists; fall open if not.
            if real and not shutil.which(real):
                return {
                    "exec": None,
                    "env": env,
                    "arm": arm,
                    "resolution": resolution,
                    "error": f"on_command binary not found: {real}",
                }
            return {
                "exec": exec_argv,
                "env": env,
                "arm": arm,
                "resolution": resolution,
            }
        else:
            # arm == "off": plain exec of the real binary.
            cmd_template = registry.get("off_command", ["<tool>"])
            exec_argv = _build_exec_argv(cmd_template, agent_cli, argv)
            # exec_argv[0] should be the real agent binary.
            real = shutil.which(exec_argv[0]) if exec_argv else None
            if not real:
                return {
                    "exec": None,
                    "env": env,
                    "arm": arm,
                    "resolution": resolution,
                    "error": f"binary not found: {exec_argv[0] if exec_argv else agent_cli}",
                }
            # Replace argv[0] with the full resolved path.
            exec_argv[0] = real
            return {
                "exec": exec_argv,
                "env": env,
                "arm": arm,
                "resolution": resolution,
            }

    except Exception as exc:
        return _plain_exec(agent_cli, argv, env, error=f"resolve_launch internal error: {exc}")


# ---------------------------------------------------------------------------
# Public: install_report
# ---------------------------------------------------------------------------


def _hook_entry_for_command(cmd: str) -> dict:
    """Build a codex/claude SessionStart hook entry dict for cmd."""
    return {"hooks": [{"type": "command", "command": cmd}]}


def _check_claude_settings(path: Path, cmd: str) -> dict:
    """Return presence info for the claude SessionStart hook."""
    snippet = json.dumps(_hook_entry_for_command(cmd))
    if not path.exists():
        return {"present": False, "path": str(path), "snippet": snippet}
    try:
        data = json.loads(path.read_text())
        hooks_section = data.get("hooks", {})
        session_start = hooks_section.get("SessionStart", [])
        present = any(
            any(h.get("command") == cmd for h in entry.get("hooks", []))
            for entry in session_start
        )
        return {"present": present, "path": str(path), "snippet": snippet}
    except Exception as exc:
        return {"present": False, "path": str(path), "snippet": snippet, "error": str(exc)}


def _check_codex_hooks(path: Path, cmd: str) -> dict:
    """Return presence info for the codex SessionStart hook."""
    snippet = json.dumps(_hook_entry_for_command(cmd))
    if not path.exists():
        return {"present": False, "path": str(path), "snippet": snippet}
    try:
        data = json.loads(path.read_text())
        hooks = data.get("hooks", {})
        session_start = hooks.get("SessionStart", [])
        present = any(
            any(h.get("command") == cmd for h in entry.get("hooks", []))
            for entry in session_start
        )
        return {"present": present, "path": str(path), "snippet": snippet}
    except Exception as exc:
        return {"present": False, "path": str(path), "snippet": snippet, "error": str(exc)}


def _apply_claude_settings(path: Path, cmd: str) -> dict:
    """Idempotently insert the hook entry into claude settings.json.

    Returns a result dict with "ok": bool and optionally "error": str.
    Never edits the file if the command is already present.
    Returns error (without modifying) if the file exists but contains invalid JSON.
    """
    entry = _hook_entry_for_command(cmd)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            return {"ok": False, "error": f"malformed JSON at {path}: {exc}"}
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}

    hooks_section = data.setdefault("hooks", {})
    session_start = hooks_section.setdefault("SessionStart", [])

    # Dedupe by command string.
    existing_cmds = {
        h.get("command")
        for e in session_start
        for h in e.get("hooks", [])
    }
    if cmd in existing_cmds:
        return {"ok": True, "already_present": True}

    session_start.append(entry)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return {"ok": True, "already_present": False}


def _apply_codex_hooks(path: Path, cmd: str) -> dict:
    """Idempotently insert the hook entry into codex hooks.json.

    Returns a result dict with "ok": bool and optionally "error": str.
    Never edits the file if the command is already present.
    Returns error (without modifying) if the file exists but contains invalid JSON.
    """
    entry = _hook_entry_for_command(cmd)
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            return {"ok": False, "error": f"malformed JSON at {path}: {exc}"}
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}

    # codex hooks.json wraps everything in a top-level "hooks" key.
    hooks_section = data.setdefault("hooks", {})
    session_start = hooks_section.setdefault("SessionStart", [])

    # Dedupe by command string.
    existing_cmds = {
        h.get("command")
        for e in session_start
        for h in e.get("hooks", [])
    }
    if cmd in existing_cmds:
        return {"ok": True, "already_present": True}

    session_start.append(entry)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return {"ok": True, "already_present": False}


def install_report(apply: bool = False) -> dict:
    """Return install state for the two SessionStart hooks.

    Keys:
        claude  — {present, path, snippet, [error], [apply_result]}
        codex   — {present, path, snippet, [error], [apply_result]}
        shell   — {present: False, snippet: str}  (never applied; user adds manually)

    apply=True edits only the two JSON files; never touches shell rc files.
    """
    claude_path = _claude_settings_path()
    codex_path = _codex_hooks_path()

    claude_result = _check_claude_settings(claude_path, _CLAUDE_HOOK_CMD)
    codex_result = _check_codex_hooks(codex_path, _CODEX_HOOK_CMD)

    # Shell function snippet — informational only, never applied automatically.
    # Uses the portable `tokensmash launch` subcommand; deployments may
    # substitute a faster local shim (see the reference dotfiles).
    shell_snippet = (
        "# Add to ~/.bashrc / ~/.zshrc:\n"
        "claude() { tokensmash launch claude -- \"$@\"; }\n"
        "codex()  { tokensmash launch codex -- \"$@\"; }"
    )

    if apply:
        if "error" not in claude_result:
            ar = _apply_claude_settings(claude_path, _CLAUDE_HOOK_CMD)
            claude_result["apply_result"] = ar
            if ar.get("ok"):
                claude_result["present"] = True
        if "error" not in codex_result:
            ar = _apply_codex_hooks(codex_path, _CODEX_HOOK_CMD)
            codex_result["apply_result"] = ar
            if ar.get("ok"):
                codex_result["present"] = True

    return {
        "claude": claude_result,
        "codex": codex_result,
        "shell": {"present": False, "snippet": shell_snippet},
    }
