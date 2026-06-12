"""Claude Code session parser.

Parses Claude Code transcript JSONL files (under ~/.claude/projects/) into
normalized session records and per-request timelines per the tokensmash
session contract (docs/CONTRACTS.md section 1).

Real-format facts observed from Claude Code 2.1.x transcripts:
- Entry fields on every line: type, uuid, sessionId, timestamp, cwd, version,
  isSidechain, and for assistant/user: message (dict with role, content, usage).
- Entry types include: user, assistant, mode, permission-mode, attachment,
  file-history-snapshot, ai-title, last-prompt, queue-operation, and others
  (unknown types are silently skipped).
- assistant entries carry message.usage with keys:
    input_tokens, cache_creation_input_tokens, cache_read_input_tokens,
    output_tokens  (plus extra fields like server_tool_use, cache_creation,
    iterations, service_tier, inference_geo, speed — we ignore those).
- Streaming writes multiple entries with the SAME uuid (and same message.id).
  We dedupe by uuid: last entry in file order wins.
- user entries with content=[{type:"tool_result",...}] are tool responses,
  NOT genuine user turns.
- Compaction boundaries appear as user entries with isCompactSummary=true.
- isSidechain=true on entries means this is a subagent transcript; if all
  (or any) entries are sidechain, the session record gets "sidechain": true.
- tool_result content is either a string or [{type:"text",text:...},...].
  We look up the tool name via the preceding tool_use block (tool_use_id ->
  tool_name map built from assistant content blocks).
- model comes from message.model on assistant entries.
- cwd/version/sessionId live on every entry's top level.
Stdlib only; zero runtime deps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tokensmash.schema import (
    SESSION_SCHEMA,
    empty_usage,
    machine_id,
    repo_identity,
    stable_id,
    validate_session_record,
)

# ── Tool-name → timeline category mapping ────────────────────────────────────
# Contract: shell, file_read, search, mcp, web, other

_TOOL_CATEGORIES: dict[str, str] = {
    "Bash": "shell",
    "Read": "file_read",
    "Grep": "search",
    "Glob": "search",
    "WebFetch": "web",
    "WebSearch": "web",
}

_TARGET_MAX_LEN = 120

# Tools whose target comes from the "file_path" input key
_FILE_PATH_TOOLS = frozenset({"Read", "Edit", "Write"})
# Tools whose target comes from the "command" input key
_COMMAND_TOOLS = frozenset({"Bash"})
# Tools whose target comes from the "pattern" input key
_PATTERN_TOOLS = frozenset({"Grep", "Glob"})


def _categorize_tool(name: str) -> str:
    if name in _TOOL_CATEGORIES:
        return _TOOL_CATEGORIES[name]
    if name.startswith("mcp__"):
        return "mcp"
    return "other"


def _extract_target(name: str, inp: dict) -> str | None:
    """Return a short target string for the tool call, or None.

    - Read/Edit/Write: inp["file_path"]
    - Bash: first 120 chars of inp["command"]
    - Grep/Glob: inp["pattern"]
    """
    if name in _FILE_PATH_TOOLS:
        v = inp.get("file_path")
        if isinstance(v, str) and v:
            return v[:_TARGET_MAX_LEN]
    elif name in _COMMAND_TOOLS:
        v = inp.get("command")
        if isinstance(v, str) and v:
            return v[:_TARGET_MAX_LEN]
    elif name in _PATTERN_TOOLS:
        v = inp.get("pattern")
        if isinstance(v, str) and v:
            return v[:_TARGET_MAX_LEN]
    return None


# ── Text extraction from tool_result content blocks ──────────────────────────


def _extract_tool_result_text(block: dict) -> str:
    """Return concatenated text from a tool_result block (string or list form)."""
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for cb in content:
            if isinstance(cb, dict) and cb.get("type") == "text":
                text = cb.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


# ── Main parser ───────────────────────────────────────────────────────────────


def parse_session(path: Path) -> tuple[dict, list[dict]] | None:
    """Parse one Claude Code transcript JSONL into (session_record, timeline).

    Returns None if the file contains no usable usage data (i.e. no assistant
    entries with a usage block).

    session_record validates against schema.validate_session_record.
    timeline is an ordered list of "request" and "tool_output" events.
    """
    # ── Pass 1: read all lines, skip corrupt, collect raw entries in order ──
    raw_entries: list[dict] = []
    parse_errors = 0

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                raw_entries.append(obj)
            # non-dict JSON values are silently skipped
        except json.JSONDecodeError:
            parse_errors += 1

    if not raw_entries:
        return None

    # ── Pass 2: deduplicate by uuid (last entry wins) while preserving order ─
    # We want the deduped entries in file order (position of LAST occurrence).
    uuid_last_index: dict[str, int] = {}
    for i, entry in enumerate(raw_entries):
        uid = entry.get("uuid")
        if uid:
            uuid_last_index[uid] = i

    # Build deduped list preserving order of last occurrence
    seen_uuids: set[str] = set()
    deduped: list[dict] = []
    for i, entry in enumerate(raw_entries):
        uid = entry.get("uuid")
        if uid is None:
            # No uuid: always include (rare metadata entries)
            deduped.append(entry)
        elif uuid_last_index.get(uid) == i:
            # This is the last occurrence of this uuid
            if uid not in seen_uuids:
                seen_uuids.add(uid)
                deduped.append(entry)
        # else: earlier occurrence of a duplicated uuid; skip it

    # ── Extract session metadata from entries ─────────────────────────────────
    session_id: str | None = None
    cwd: str | None = None
    version: str | None = None
    model: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    is_sidechain = False

    for entry in deduped:
        if entry.get("isSidechain"):
            is_sidechain = True

        ts = entry.get("timestamp")
        if isinstance(ts, str):
            if started_at is None:
                started_at = ts
            ended_at = ts

        if session_id is None:
            sid = entry.get("sessionId")
            if isinstance(sid, str):
                session_id = sid

        if cwd is None:
            c = entry.get("cwd")
            if isinstance(c, str):
                cwd = c

        if version is None:
            v = entry.get("version")
            if isinstance(v, str):
                version = v

        # Extract model from assistant message
        if model is None and entry.get("type") == "assistant":
            msg = entry.get("message", {})
            if isinstance(msg, dict):
                m = msg.get("model")
                if isinstance(m, str):
                    model = m

    # ── Pass 3: build tool_use_id -> {name, input} map from assistant entries ──
    tool_use_map: dict[str, str] = {}  # tool_use_id -> tool_name
    tool_input_map: dict[str, dict] = {}  # tool_use_id -> input dict
    for entry in deduped:
        if entry.get("type") != "assistant":
            continue
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                bid = block.get("id")
                name = block.get("name", "")
                if isinstance(bid, str) and isinstance(name, str):
                    tool_use_map[bid] = name
                    inp = block.get("input")
                    if isinstance(inp, dict):
                        tool_input_map[bid] = inp

    # ── Pass 4: build timeline and accumulate usage ────────────────────────────
    timeline: list[dict] = []
    request_index = 0
    total_usage = empty_usage()
    has_usage = False
    user_turns = 0
    tool_calls = 0
    compactions = 0
    final_provider_raw: dict | None = None

    # For provider_raw we want the final assistant message's raw usage
    for entry in deduped:
        entry_type = entry.get("type")

        # ── Compaction boundary ──
        if entry.get("isCompactSummary"):
            compactions += 1
            continue

        # ── Assistant entry ──
        if entry_type == "assistant":
            msg = entry.get("message", {})
            if not isinstance(msg, dict):
                continue
            usage_raw = msg.get("usage")
            if not isinstance(usage_raw, dict):
                continue

            # Translate to canonical usage (Claude: input_tokens excludes cache)
            fresh_input = int(usage_raw.get("input_tokens", 0))
            cache_read = int(usage_raw.get("cache_read_input_tokens", 0))
            cache_write = int(usage_raw.get("cache_creation_input_tokens", 0))
            output = int(usage_raw.get("output_tokens", 0))

            canonical_usage = {
                "fresh_input": fresh_input,
                "cache_read": cache_read,
                "cache_write": cache_write,
                "output": output,
                "reasoning_output": None,
            }

            # Accumulate session totals
            total_usage["fresh_input"] += fresh_input
            total_usage["cache_read"] += cache_read
            total_usage["cache_write"] += cache_write
            total_usage["output"] += output
            has_usage = True
            final_provider_raw = dict(usage_raw)

            # Emit request timeline event
            timeline.append({
                "kind": "request",
                "index": request_index,
                "usage": canonical_usage,
            })

            # Count tool_use blocks in this assistant message
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_calls += 1

            request_index += 1
            continue

        # ── User entry ──
        if entry_type == "user":
            msg = entry.get("message", {})
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")

            # Determine if this is a genuine user turn or tool_result entry
            is_tool_result_only = False
            if isinstance(content, list):
                non_empty = [b for b in content if isinstance(b, dict)]
                if non_empty and all(b.get("type") == "tool_result" for b in non_empty):
                    is_tool_result_only = True

            if is_tool_result_only:
                # Extract tool_output timeline events
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    tool_use_id = block.get("tool_use_id", "")
                    tool_name = tool_use_map.get(tool_use_id, "")
                    category = _categorize_tool(tool_name)
                    text = _extract_tool_result_text(block)
                    tokens_est = max(1, len(text) // 4)
                    inp = tool_input_map.get(tool_use_id, {})
                    target = _extract_target(tool_name, inp)
                    tev: dict = {
                        "kind": "tool_output",
                        "request_index": max(0, request_index - 1),
                        "category": category,
                        "tokens_est": tokens_est,
                        "tool_name": tool_name,
                    }
                    if target is not None:
                        tev["target"] = target
                    timeline.append(tev)
            else:
                # Genuine user message (text or mixed content without tool_result)
                user_turns += 1

    # ── Return None if no usable usage data ───────────────────────────────────
    if not has_usage:
        return None

    # ── Normalize timestamps ───────────────────────────────────────────────────
    def _normalize_ts(ts: str | None) -> str:
        if ts is None:
            return datetime.now(timezone.utc).isoformat()
        # Already ISO-8601; ensure it ends with Z or +00:00
        if ts.endswith("Z"):
            return ts.replace("Z", "+00:00")
        if "+" not in ts[10:] and not ts.endswith("Z"):
            return ts + "+00:00"
        return ts

    started_at_norm = _normalize_ts(started_at)
    ended_at_norm = _normalize_ts(ended_at)

    # ── Compute duration ───────────────────────────────────────────────────────
    def _parse_ts(ts: str) -> float:
        """Parse ISO-8601 timestamp to epoch seconds."""
        try:
            ts_clean = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(ts_clean).timestamp()
        except (ValueError, AttributeError):
            return 0.0

    duration_ms = max(0, int((_parse_ts(ended_at or "") - _parse_ts(started_at or "")) * 1000))

    # ── Build session record ───────────────────────────────────────────────────
    effective_cwd = cwd or "/"
    repo_id = stable_id("repo", repo_identity(effective_cwd))

    record: dict = {
        "schema": SESSION_SCHEMA,
        "agent": "claude-code",
        "session_id": session_id or path.stem,
        # Subagent transcripts share the parent's sessionId, so session_id
        # does not identify a transcript. transcript_id is the store
        # identity; logical_session_id groups a session with its subagents
        # (and is the arm-join key, so subagents inherit the parent's arm).
        "transcript_id": stable_id("transcript", str(path.resolve())),
        "logical_session_id": session_id or path.stem,
        "machine_id": machine_id(),
        "started_at": started_at_norm,
        "ended_at": ended_at_norm,
        "model": model or "unknown",
        "agent_version": version,
        "repo_id": repo_id,
        "user_turns": user_turns,
        "tool_calls": tool_calls,
        "compactions": compactions,
        "duration_ms": duration_ms,
        "usage": total_usage,
        "provider_raw": final_provider_raw or {},
        "parse_errors": parse_errors,
    }

    if is_sidechain:
        record["sidechain"] = True

    # Validate (defensive — should always be clean)
    errors = validate_session_record(record)
    if errors:
        # Include error info but don't crash
        record["_validation_errors"] = errors

    return record, timeline


# ── Session file iterator ─────────────────────────────────────────────────────


def iter_session_files(root: Path) -> Iterator[Path]:
    """Yield Claude Code transcript JSONL files under root, newest last.

    Claude Code stores main-session transcripts at
    ~/.claude/projects/<encoded-dir>/*.jsonl, but subagent transcripts nest
    deeper (<encoded-dir>/<session>/subagents/agent-*.jsonl, and further for
    nested fan-out). Subagent spend is real spend, so recurse the whole tree.
    """
    files: list[tuple[float, Path]] = []
    try:
        for fpath in root.rglob("*.jsonl"):
            try:
                if fpath.is_file():
                    files.append((fpath.stat().st_mtime, fpath))
            except OSError:
                pass
    except OSError:
        return

    # Sort by mtime ascending so newest is last
    files.sort(key=lambda x: x[0])
    for _, fpath in files:
        yield fpath
