"""Normalized session schema and stable identifiers.

This module is the contract between session parsers, the store, opportunity
analysis, and the study machinery. See docs/CONTRACTS.md and
docs/study-architecture.md (D3). Stdlib only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Any

SESSION_SCHEMA = "tokensmash-session/1"
ASSIGNMENT_SCHEMA = "tokensmash-assignment/1"

STUDY_DIR = Path(os.environ.get("TOKENSMASH_STUDY_DIR", "")) if os.environ.get(
    "TOKENSMASH_STUDY_DIR"
) else Path.home() / ".local" / "state" / "tokensmash" / "study"

AGENTS = ("codex", "claude-code")
ARMS = ("on", "off")

# Canonical usage fields. Providers define input_tokens differently; parsers
# MUST translate into these and never pass provider names through:
#   codex:  fresh_input = input_tokens - cached_input_tokens
#           cache_read  = cached_input_tokens, cache_write = 0
#   claude: fresh_input = input_tokens (already excludes cache activity)
#           cache_read  = cache_read_input_tokens
#           cache_write = cache_creation_input_tokens
USAGE_KEYS = ("fresh_input", "cache_read", "cache_write", "output", "reasoning_output")

REQUIRED_RECORD_KEYS = (
    "schema",
    "agent",
    "session_id",
    "machine_id",
    "started_at",
    "model",
    "repo_id",
    "user_turns",
    "tool_calls",
    "compactions",
    "duration_ms",
    "usage",
    "provider_raw",
)


def empty_usage() -> dict[str, Any]:
    return {
        "fresh_input": 0,
        "cache_read": 0,
        "cache_write": 0,
        "output": 0,
        "reasoning_output": None,
    }


def validate_session_record(record: dict[str, Any]) -> list[str]:
    """Return a list of validation errors; empty list means valid."""
    errors: list[str] = []
    for key in REQUIRED_RECORD_KEYS:
        if key not in record:
            errors.append(f"missing key: {key}")
    if record.get("schema") != SESSION_SCHEMA:
        errors.append(f"schema must be {SESSION_SCHEMA}")
    if record.get("agent") not in AGENTS:
        errors.append(f"agent must be one of {AGENTS}")
    usage = record.get("usage")
    if not isinstance(usage, dict):
        errors.append("usage must be a dict")
    else:
        for key in USAGE_KEYS:
            if key not in usage:
                errors.append(f"usage missing key: {key}")
                continue
            value = usage[key]
            if key == "reasoning_output":
                if value is not None and (not isinstance(value, int) or value < 0):
                    errors.append("usage.reasoning_output must be None or int >= 0")
            elif not isinstance(value, int) or value < 0:
                errors.append(f"usage.{key} must be int >= 0")
    arm = record.get("arm")
    if arm is not None and arm not in ARMS:
        errors.append(f"arm must be one of {ARMS} or absent")
    return errors


def _secret_path() -> Path:
    return STUDY_DIR / "secret"


def load_or_create_secret() -> bytes:
    """Local HMAC key for stable, non-reversible identifiers.

    Never exported; scrubbed exports keep the hashes, which are useless
    without this key.
    """
    path = _secret_path()
    if path.exists():
        return bytes.fromhex(path.read_text().strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    path.write_text(key.hex() + "\n")
    path.chmod(0o600)
    return key


def stable_id(kind: str, value: str, key: bytes | None = None) -> str:
    """HMAC-keyed stable identifier, 16 hex chars, namespaced by kind."""
    key = key if key is not None else load_or_create_secret()
    digest = hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()
    return digest[:16]


def repo_identity(cwd: str | Path) -> str:
    """Canonical pre-hash identity for a working directory.

    Prefers the git remote URL (stable across clones/worktrees); falls back to
    the git toplevel path, then the cwd itself. Pure function of filesystem
    state; no network.
    """
    import subprocess

    cwd = Path(cwd)
    for args in (["git", "config", "--get", "remote.origin.url"], ["git", "rev-parse", "--show-toplevel"]):
        try:
            proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=5, check=False)
            value = proc.stdout.strip()
            if proc.returncode == 0 and value:
                return value
        except (OSError, subprocess.SubprocessError):
            pass
    return str(cwd)


def machine_id() -> str:
    identity = f"{os.uname().nodename}:{Path.home()}"
    return stable_id("machine", identity)


def dumps_record(record: dict[str, Any]) -> str:
    """Canonical single-line JSON for the append-only stores."""
    return json.dumps(record, separators=(",", ":"), sort_keys=True)
