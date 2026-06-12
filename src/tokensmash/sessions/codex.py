"""Codex session parser.

Parses Codex JSONL transcript files into normalized session records and
timelines per docs/CONTRACTS.md section 1. Stdlib only; no imports from
cli.py.

Real Codex JSONL format (discovered from ~/.codex/sessions/):
  Each line: {"timestamp": ISO-8601, "type": str, "payload": dict}

  Outer types observed:
    session_meta   — first line; payload has id, cwd, cli_version, model_provider
    turn_context   — payload has model (e.g. "gpt-5.5"), cwd, turn_id
    event_msg      — payload.type in: task_started, user_message, agent_message,
                     token_count, task_complete, patch_apply_end, …
    response_item  — payload.type in: function_call, function_call_output,
                     custom_tool_call, custom_tool_call_output, message, reasoning

  Token counting:
    event_msg / token_count → payload.info.total_token_usage = {
      input_tokens, cached_input_tokens, output_tokens,
      reasoning_output_tokens, total_tokens
    }
    These are CUMULATIVE per session. Per-request deltas = successive diffs.
    Negative deltas (session restart / compaction reset) → clamped to 0,
    noted in record["anomalies"].

  Canonical usage translation (per schema.py comment):
    fresh_input  = input_tokens - cached_input_tokens
    cache_read   = cached_input_tokens
    cache_write  = 0  (codex has no explicit cache-write billing)
    output       = output_tokens
    reasoning_output = reasoning_output_tokens (None when 0 and no reasoning)

  Tool calls:
    function_call / function_call_output — linked by call_id
    custom_tool_call / custom_tool_call_output — same structure

  Model: turn_context.payload.model  (session_meta only has model_provider)
  Session ID: session_meta.payload.id
  CWD: session_meta.payload.cwd (also in turn_context)
  CLI version: session_meta.payload.cli_version → agent_version
  Timestamps: outer "timestamp" field (ISO-8601 UTC)

  Compaction: no explicit compaction marker in observed data; count = 0.
  User turns: event_msg with payload.type == "user_message".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from tokensmash.schema import (
    SESSION_SCHEMA,
    empty_usage,
    machine_id,
    repo_identity,
    stable_id,
    validate_session_record,
)

# ---------------------------------------------------------------------------
# Tool categorisation helpers
# ---------------------------------------------------------------------------

# Shell-execution tool names
_SHELL_TOOLS = frozenset({"exec_command", "shell_command"})

# File-reading tool names
_FILE_READ_TOOLS = frozenset({"get_document", "view_image"})

# Web-fetch tool names
_WEB_TOOLS = frozenset({"searxng_web_search", "web_url_read"})

# Search tool names (grep/find/rg-style)
_SEARCH_TOOLS: frozenset[str] = frozenset()  # categorised by command inspection below

# Shell commands that indicate file-reading behaviour
_FILE_READ_CMD_RE = re.compile(
    r"^\s*(cat|head|tail|less|more|read|bat)\s", re.IGNORECASE
)
# Shell commands that indicate search behaviour
_SEARCH_CMD_RE = re.compile(
    r"^\s*(grep|rg|find|ag|ack|fd|fzf|locate)\s", re.IGNORECASE
)


_TARGET_MAX_LEN = 120


def _extract_target(name: str, args: dict[str, Any]) -> str | None:
    """Return a short target string for the tool call, or None.

    For codex all tools are shell-based (shell_command / exec_command).
    The target is the first 120 chars of the command string.
    Argument key is "cmd" for exec_command, "command" for shell_command.
    """
    for key in ("cmd", "command", "shell_command"):
        v = args.get(key)
        if isinstance(v, str) and v:
            return v[:_TARGET_MAX_LEN]
    return None


def _categorise_tool(name: str, args: dict[str, Any]) -> str:
    """Return one of shell|file_read|search|mcp|web|other."""
    lname = name.lower()

    # MCP-namespaced tools (mcp__server__tool) or known ctx_ tools
    if lname.startswith("mcp__") or lname.startswith("ctx_"):
        return "mcp"

    # Context-mode (compound MCP tools)
    if "context_mode" in lname or "context-mode" in lname:
        return "mcp"

    # Web tools
    if lname in _WEB_TOOLS or "web" in lname or "search" in lname and "url" in lname:
        return "web"

    # Browser tools (web-based)
    if lname.startswith("browser_") or lname.startswith("mcp__playwright__"):
        return "web"

    # File-read tools
    if lname in _FILE_READ_TOOLS:
        return "file_read"

    # Shell execution tools
    if lname in _SHELL_TOOLS or lname == "write_stdin":
        # Inspect the command string
        cmd = ""
        for key in ("cmd", "command", "shell_command"):
            v = args.get(key)
            if isinstance(v, str) and v:
                cmd = v
                break
        if cmd:
            if _FILE_READ_CMD_RE.match(cmd):
                return "file_read"
            if _SEARCH_CMD_RE.match(cmd):
                return "search"
        return "shell"

    # apply_patch / custom write tools → shell-adjacent
    if "patch" in lname or "write" in lname:
        return "shell"

    # Search-style tools
    if "search" in lname or "grep" in lname or "find" in lname:
        return "search"

    return "other"


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    """Yield (line_index, parsed_object, error_or_None) for each line."""
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield i, json.loads(raw), None
            except json.JSONDecodeError as exc:
                yield i, None, exc


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_session(path: Path) -> tuple[dict, list[dict]] | None:
    """Parse one Codex transcript JSONL into (session_record, timeline).

    Returns None if the file contains no usable usage data (no token_count
    events with non-zero totals).
    """
    # ── Pass 1: collect all events ─────────────────────────────────────────
    session_id: str | None = None
    cwd: str | None = None
    model: str | None = None
    agent_version: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    parse_errors: int = 0
    user_turns: int = 0
    compactions: int = 0
    anomalies: list[str] = []

    # Cumulative token_count snapshots in order
    cum_snapshots: list[dict[str, int]] = []

    # Tool call tracking: call_id → {name, args, category}
    pending_calls: dict[str, dict] = {}

    # Timeline events built during pass
    timeline: list[dict] = []

    # Request index counter: incremented each time we see a new token_count
    request_index: int = -1

    # Tool outputs buffered until we know their request_index
    # We assign them the request_index of the CURRENT token_count window
    # (the one that captures this tool's output).
    # Strategy: accumulate tool outputs between successive token_counts;
    # when the next token_count arrives we flush them with the new index.
    buffered_tool_outputs: list[dict] = []

    for _line_i, obj, err in _iter_jsonl(path):
        if err is not None:
            parse_errors += 1
            continue

        outer_type = obj.get("type", "")
        payload = obj.get("payload")
        ts = obj.get("timestamp", "")

        # Track first/last timestamp
        if ts:
            if started_at is None:
                started_at = ts
            ended_at = ts

        if not isinstance(payload, dict):
            continue

        pt = payload.get("type", "")

        # ── session_meta ────────────────────────────────────────────────────
        if outer_type == "session_meta":
            session_id = payload.get("id") or session_id
            cwd = payload.get("cwd") or cwd
            agent_version = payload.get("cli_version") or agent_version
            continue

        # ── turn_context ─────────────────────────────────────────────────────
        if outer_type == "turn_context":
            model = payload.get("model") or model
            if not cwd:
                cwd = payload.get("cwd")
            continue

        # ── event_msg ────────────────────────────────────────────────────────
        if outer_type == "event_msg":
            if pt == "user_message":
                user_turns += 1
            elif pt == "token_count":
                raw_usage = (payload.get("info") or {}).get("total_token_usage") or {}
                if isinstance(raw_usage, dict) and raw_usage:
                    snap = {
                        "input_tokens": int(raw_usage.get("input_tokens") or 0),
                        "cached_input_tokens": int(raw_usage.get("cached_input_tokens") or 0),
                        "output_tokens": int(raw_usage.get("output_tokens") or 0),
                        "reasoning_output_tokens": int(raw_usage.get("reasoning_output_tokens") or 0),
                        "total_tokens": int(raw_usage.get("total_tokens") or 0),
                    }
                    cum_snapshots.append(snap)
                    # Advance request index and flush buffered tool outputs
                    request_index += 1
                    for tev in buffered_tool_outputs:
                        tev["request_index"] = request_index
                        timeline.append(tev)
                    buffered_tool_outputs = []
                    # Append request timeline event (delta computed later)
                    timeline.append({"kind": "request", "index": request_index, "_snap_idx": len(cum_snapshots) - 1})
            continue

        # ── response_item ────────────────────────────────────────────────────
        if outer_type == "response_item":
            if pt in ("function_call", "custom_tool_call"):
                call_id = payload.get("call_id", "")
                name = payload.get("name", "")
                # Parse arguments
                raw_args = payload.get("arguments") or payload.get("input") or {}
                if isinstance(raw_args, str):
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}
                if not isinstance(raw_args, dict):
                    raw_args = {}
                category = _categorise_tool(name, raw_args)
                target = _extract_target(name, raw_args)
                if call_id:
                    pending_calls[call_id] = {"name": name, "category": category, "target": target}

            elif pt in ("function_call_output", "custom_tool_call_output"):
                call_id = payload.get("call_id", "")
                output = payload.get("output") or ""
                if not isinstance(output, str):
                    output = json.dumps(output)
                call_info = pending_calls.pop(call_id, {})
                tool_name = call_info.get("name", "")
                category = call_info.get("category", "other")
                target = call_info.get("target")
                tokens_est = max(1, len(output) // 4)
                tev = {
                    "kind": "tool_output",
                    "request_index": request_index,  # placeholder; may be updated
                    "category": category,
                    "tokens_est": tokens_est,
                    "tool_name": tool_name,
                }
                if target is not None:
                    tev["target"] = target
                # If we haven't seen a token_count yet, buffer it
                if request_index < 0:
                    buffered_tool_outputs.append(tev)
                else:
                    # Tool outputs appear BEFORE the token_count for their turn;
                    # buffer them until next token_count increments request_index
                    buffered_tool_outputs.append(tev)
            continue

    # ── Compute per-request deltas ─────────────────────────────────────────
    if not cum_snapshots:
        return None

    # Build delta list from cumulative snapshots
    def _snap_delta(prev: dict[str, int] | None, curr: dict[str, int]) -> dict[str, int]:
        if prev is None:
            # First snapshot: entire cumulative is the delta
            return dict(curr)
        delta: dict[str, int] = {}
        for k in curr:
            delta[k] = max(0, curr[k] - prev.get(k, 0))
        return delta

    # Build deltas in snapshot order
    snap_deltas: list[dict[str, int]] = []
    prev_snap: dict[str, int] | None = None
    for snap in cum_snapshots:
        delta = _snap_delta(prev_snap, snap)
        # Check for negative deltas (restart)
        if prev_snap is not None:
            for k in snap:
                raw_diff = snap[k] - prev_snap.get(k, 0)
                if raw_diff < 0:
                    anomalies.append(
                        f"negative delta for {k} at snapshot {len(snap_deltas)}: "
                        f"{prev_snap.get(k, 0)} → {snap[k]}, clamped to 0"
                    )
        snap_deltas.append(delta)
        prev_snap = snap

    # Patch timeline: replace _snap_idx placeholders with canonical usage
    final_timeline: list[dict] = []
    for ev in timeline:
        if ev.get("kind") == "request":
            snap_idx = ev.pop("_snap_idx")
            delta = snap_deltas[snap_idx]
            canonical = _delta_to_canonical(delta)
            final_timeline.append({"kind": "request", "index": ev["index"], "usage": canonical})
        else:
            final_timeline.append(ev)

    # ── Flush any remaining buffered tool outputs (edge case) ──────────────
    for tev in buffered_tool_outputs:
        if tev["request_index"] < 0:
            tev["request_index"] = max(0, request_index)
        final_timeline.append(tev)

    # Sort timeline: requests first by index, tool_outputs by request_index
    final_timeline.sort(key=lambda e: (e.get("index", e.get("request_index", 0)),
                                        0 if e["kind"] == "request" else 1))

    # ── Build session-level usage from final cumulative snapshot ───────────
    final_snap = cum_snapshots[-1]
    session_usage = _snap_to_canonical(final_snap)
    provider_raw = {
        "input_tokens": final_snap["input_tokens"],
        "cached_input_tokens": final_snap["cached_input_tokens"],
        "output_tokens": final_snap["output_tokens"],
        "reasoning_output_tokens": final_snap["reasoning_output_tokens"],
        "total_tokens": final_snap["total_tokens"],
    }

    # ── Count tool_calls ────────────────────────────────────────────────────
    tool_calls = sum(1 for ev in final_timeline if ev["kind"] == "tool_output")

    # ── Timestamps ─────────────────────────────────────────────────────────
    if started_at is None:
        started_at = ""
    if ended_at is None:
        ended_at = started_at

    duration_ms = _duration_ms(started_at, ended_at)

    # ── repo_id ─────────────────────────────────────────────────────────────
    if cwd:
        try:
            repo_id = stable_id("repo", repo_identity(cwd))
        except Exception:
            repo_id = stable_id("repo", cwd)
    else:
        repo_id = stable_id("repo", "unknown")

    # ── Fallback session_id (use filename stem if no internal id) ──────────
    if not session_id:
        session_id = path.stem

    record: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "agent": "codex",
        "session_id": session_id,
        # Resumed Codex sessions write NEW rollout files carrying the SAME
        # internal session id, so session_id does not identify a transcript.
        # transcript_id is the store identity; logical_session_id groups
        # rollouts of one logical session (and is the arm-join key).
        "transcript_id": stable_id("transcript", str(path.resolve())),
        "logical_session_id": session_id,
        "machine_id": machine_id(),
        "started_at": _normalise_ts(started_at),
        "ended_at": _normalise_ts(ended_at),
        "model": model or "unknown",
        "repo_id": repo_id,
        "user_turns": user_turns,
        "tool_calls": tool_calls,
        "compactions": compactions,
        "duration_ms": duration_ms,
        "usage": session_usage,
        "provider_raw": provider_raw,
        "transcript_path": str(path),
        "parse_errors": parse_errors,
    }
    if agent_version:
        record["agent_version"] = agent_version
    if anomalies:
        record["anomalies"] = anomalies
    if cwd:
        record["cwd"] = cwd

    errors = validate_session_record(record)
    if errors:
        # Record is malformed; still return it but note validation problems
        record["_validation_errors"] = errors

    return record, final_timeline


def _snap_to_canonical(snap: dict[str, int]) -> dict[str, Any]:
    """Convert a raw cumulative snapshot to canonical usage dict."""
    input_tok = snap.get("input_tokens", 0)
    cached = snap.get("cached_input_tokens", 0)
    output_tok = snap.get("output_tokens", 0)
    reasoning = snap.get("reasoning_output_tokens", 0)
    return {
        "fresh_input": max(0, input_tok - cached),
        "cache_read": cached,
        "cache_write": 0,
        "output": output_tok,
        "reasoning_output": reasoning if reasoning else None,
    }


def _delta_to_canonical(delta: dict[str, int]) -> dict[str, Any]:
    """Convert a delta snapshot to canonical usage dict."""
    input_tok = delta.get("input_tokens", 0)
    cached = delta.get("cached_input_tokens", 0)
    output_tok = delta.get("output_tokens", 0)
    reasoning = delta.get("reasoning_output_tokens", 0)
    return {
        "fresh_input": max(0, input_tok - cached),
        "cache_read": cached,
        "cache_write": 0,
        "output": output_tok,
        "reasoning_output": reasoning if reasoning else None,
    }


def _normalise_ts(ts: str) -> str:
    """Ensure timestamp ends with Z for UTC ISO-8601."""
    if not ts:
        return ts
    ts = ts.strip()
    if ts.endswith("Z"):
        return ts
    if ts.endswith("+00:00"):
        return ts[:-6] + "Z"
    return ts


def _duration_ms(started: str, ended: str) -> int:
    """Compute duration in milliseconds between two ISO-8601 timestamps."""
    try:
        from datetime import datetime, timezone

        def _parse(ts: str) -> datetime:
            ts = ts.strip()
            if ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts)

        return max(0, int((_parse(ended) - _parse(started)).total_seconds() * 1000))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# iter_session_files
# ---------------------------------------------------------------------------

def iter_session_files(root: Path) -> Iterator[Path]:
    """Yield Codex JSONL transcript files under root, newest last.

    Codex organises sessions under YYYY/MM/DD/ subdirectories; filenames
    are rollout-<ISO-date>-<uuid>.jsonl.
    """
    files: list[Path] = []
    if not root.is_dir():
        return
    for p in root.rglob("*.jsonl"):
        if p.is_file():
            files.append(p)
    # Sort by filename (which contains an ISO date prefix) → chronological order
    files.sort(key=lambda p: p.name)
    yield from files
