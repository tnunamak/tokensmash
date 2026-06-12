"""Offline realized-compression estimates for tokensmash.

Purpose
-------
Convert Layer-0 opportunity ceilings (100 % compression assumed) into realized
point estimates by running actual reducer tools over tool outputs recorded in
session transcripts.

Feasibility findings (investigated 2026-06-12)
-----------------------------------------------

rtk (``rtk pipe``)
    FEASIBLE.  ``rtk`` exposes ``rtk pipe [--filter <name>]`` which reads stdin,
    applies a named filter, and writes filtered output to stdout.  For replaying
    arbitrary shell/search/mcp outputs we use ``rtk pipe`` (no filter flag),
    which applies rtk's general heuristic log-deduplication pass.  The binary is
    found with ``shutil.which("rtk")``.  Invocation::

        echo "<recorded text>" | rtk pipe

    This maps to the ``shell`` category (rtk is a shell-output reducer).

headroom (``~/.local/share/uv/tools/headroom-ai/bin/python``)
    FEASIBLE via library-offline pattern from the legacy tokensmash script.
    The headroom Python distribution ships its own interpreter under
    ``~/.local/share/uv/tools/headroom-ai/bin/python``.  We spawn that
    interpreter with ``-c <inline code>`` to import ``headroom.compress.compress``
    and compress a single tool-message offline — no network, no proxy, no API
    key.  ``HEADROOM_TELEMETRY=off`` suppresses any outbound pings.  Headroom
    compresses the entire request wire payload, so its realized ratio is measured
    against the full ``(fresh_input + cache_read)`` token budget rather than a
    per-category subset.

repomix / packers
    OUT OF SCOPE.  repomix packs a live repository snapshot into a single context
    document.  Replaying it offline would require the exact repo state (branch,
    working tree, commit) that existed during the recorded session — that state
    is not captured in the transcript.  Attempting to use the current checkout
    would conflate present state with past state and produce meaningless ratios.
    Documented here; not implemented.

Methodology decisions
---------------------
* ``tokens_in`` / ``tokens_out`` are approximated as ``len(text) // 4`` (the
  same convention used everywhere else in tokensmash) — provider tokenization
  is not available offline.  The report caveat block states this explicitly.
* Realized ratio = ``tokens_out / tokens_in`` for each sampled tool output.
  Per-session ratio = weighted mean across samples.  Corpus ratio = weighted
  mean across sessions.
* The "realized estimate" in the report is:
    insertion ceiling USD × realized ratio
    + (capped reread ceiling − insertion ceiling) USD × realized ratio
  which simplifies to  ``with_rereads_ceiling_usd × realized ratio``.
* Trajectory effects (tool outputs that appear earlier compress more because
  they are re-read more often) are not captured by a single per-category ratio.
  The caveat block says so.
* Privacy: recorded tool outputs are piped to LOCAL tools only.  No text is
  written to disk; only byte-count aggregates are retained in memory.
* Tool failures (non-zero exit, timeout, binary missing) are counted in
  ``"failures"`` and never propagate as exceptions.

Stdlib only.  No external imports except the session parsers.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_HEADROOM_PY = (
    Path.home()
    / ".local"
    / "share"
    / "uv"
    / "tools"
    / "headroom-ai"
    / "bin"
    / "python"
)

# Per-subprocess timeout in seconds.
_DEFAULT_TIMEOUT = 30.0

# Categories that rtk can address via ``rtk pipe``.
_RTK_CATEGORIES = frozenset({"shell", "search", "mcp", "other"})

# headroom compresses all categories (wire-payload reducer).
_HEADROOM_CATEGORIES = frozenset({"shell", "file_read", "search", "mcp", "web", "other"})

# Minimum chars to bother submitting to a reducer.
_MIN_CHARS = 20


# ---------------------------------------------------------------------------
# Low-level reducer helpers (no external imports)
# ---------------------------------------------------------------------------

def _tokens_est(text: str) -> int:
    return max(1, len(text) // 4)


def _rtk_available() -> bool:
    return shutil.which("rtk") is not None


def _headroom_available() -> bool:
    return _HEADROOM_PY.exists()


_RTK_TWO_WORD_FILTERS = {
    ("cargo", "test"): "cargo-test",
    ("cargo", "build"): "cargo-build",
    ("git", "log"): "git-log",
    ("git", "status"): "git-status",
    ("git", "diff"): "git-diff",
    ("go", "test"): "go-test",
    ("npm", "test"): "npm-test",
}


def _rtk_filter_for(target: str | None) -> str | None:
    """Map a recorded shell command to an rtk pipe filter name.

    Bare ``rtk pipe`` is a PASSTHROUGH — rtk's compression is filter-specific
    (verified against rtk 0.x: 4000 bytes of pytest output → 4000 bytes bare,
    26 bytes with ``-f pytest``). Commands with no plausible filter return
    None and are scored as genuinely uncompressible by rtk (coverage matters:
    rtk only helps on commands it has filters for).
    """
    if not target:
        return None
    tokens = []
    for tok in target.split():
        # skip leading env assignments and an rtk wrapper prefix
        if "=" in tok and not tokens:
            continue
        if tok == "rtk" and not tokens:
            continue
        if tok.startswith("--") and tok in ("--ultra-compact", "--quiet") and not tokens:
            continue
        tokens.append(tok)
        if len(tokens) == 2:
            break
    if not tokens:
        return None
    if len(tokens) == 2 and (tokens[0], tokens[1]) in _RTK_TWO_WORD_FILTERS:
        return _RTK_TWO_WORD_FILTERS[(tokens[0], tokens[1])]
    head = tokens[0].rsplit("/", 1)[-1]
    return head or None


def _run_rtk_pipe(
    text: str, timeout: float = _DEFAULT_TIMEOUT, target: str | None = None
) -> dict | None:
    """Pipe *text* through ``rtk pipe -f <filter>`` chosen from the command.

    Returns {tokens_in, tokens_out, "unsupported": bool}. When no filter
    matches (or rtk rejects the filter name), the text is scored unchanged —
    rtk cannot compress output of commands it has no filter for, and that is
    real coverage data, not a failure. Returns None only on harness failure
    (binary missing, timeout).
    """
    if not _rtk_available():
        return None
    flt = _rtk_filter_for(target)
    if flt is None:
        return {"tokens_in": _tokens_est(text), "tokens_out": _tokens_est(text), "unsupported": True}
    try:
        proc = subprocess.run(
            ["rtk", "pipe", "-f", flt],
            input=text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None
    if proc.returncode != 0:
        # rtk rejected the filter name → no filter exists for this command;
        # rtk genuinely cannot compress it. Coverage data, not a failure.
        return {"tokens_in": _tokens_est(text), "tokens_out": _tokens_est(text), "unsupported": True}
    out = proc.stdout
    return {
        "tokens_in": _tokens_est(text),
        "tokens_out": _tokens_est(out),
    }


# Inline Python code executed inside the headroom interpreter.
_HEADROOM_INLINE = r"""
import json, sys
try:
    from headroom.compress import compress
    content = sys.stdin.read()
    result = compress(
        [{"role": "tool", "content": content}],
        model="claude-sonnet-4-5-20250929",
    )
    compressed = result.messages[0].get("content", content)
    if not isinstance(compressed, str):
        import json as _j
        compressed = _j.dumps(compressed, separators=(",", ":"))
    print(json.dumps({"ok": True, "out_len": len(compressed.encode("utf-8", "replace"))}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
"""


def _run_headroom(text: str, timeout: float = _DEFAULT_TIMEOUT) -> dict | None:
    """Compress *text* through headroom's offline library.

    Returns None on failure.
    """
    if not _headroom_available():
        return None
    env = {**os.environ, "HEADROOM_TELEMETRY": "off", "HEADROOM_TELEMETRY_WARN": "off"}
    try:
        proc = subprocess.run(
            [str(_HEADROOM_PY), "-c", _HEADROOM_INLINE],
            input=text,
            capture_output=True,
            text=True,
            timeout=max(timeout, 60.0),
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    try:
        import json
        parsed = json.loads(proc.stdout)
    except Exception:
        return None
    if not parsed.get("ok"):
        return None
    out_chars = parsed.get("out_len", 0)
    return {
        "tokens_in": _tokens_est(text),
        "tokens_out": max(1, out_chars // 4),
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

_TOOL_RUNNERS = {
    "rtk": _run_rtk_pipe,
    "headroom": _run_headroom,
}


def _reduce(
    tool: str, text: str, timeout: float = _DEFAULT_TIMEOUT, target: str | None = None
) -> dict | None:
    """Run *tool* on *text*; return {tokens_in, tokens_out} or None on failure."""
    runner = _TOOL_RUNNERS.get(tool)
    if runner is None:
        return None
    if len(text) < _MIN_CHARS:
        return None
    try:
        return runner(text, timeout, target=target)
    except TypeError:
        # Patched test runners may not accept target.
        return runner(text, timeout)


# ---------------------------------------------------------------------------
# Session parser import (lazy, tolerates missing)
# ---------------------------------------------------------------------------

def _iter_tool_outputs(path: Path) -> Iterator[dict]:
    """Yield tool_output timeline events from a session transcript.

    Uses the project's own parsers; returns nothing if they are unavailable.
    """
    try:
        agent = _detect_agent(path)
        if agent == "claude":
            from tokensmash.sessions.claude import parse_session
        elif agent == "codex":
            from tokensmash.sessions.codex import parse_session
        else:
            return
        result = parse_session(path)
        if result is None:
            return
        _record, timeline = result
        for event in timeline:
            if event.get("kind") == "tool_output":
                yield event
    except Exception:
        return


def _detect_agent(path: Path) -> str:
    """Guess agent from path structure, then from JSONL content.

    Checks well-known path components first (fast), then sniffs the first
    non-empty line of the file for Claude-specific fields (``sessionId``,
    ``isSidechain``) or Codex-specific fields (``type: response``,
    ``response_item``).
    """
    import json as _json

    parts = path.parts
    if ".claude" in parts:
        return "claude"
    if ".codex" in parts:
        return "codex"

    # Sniff first non-empty JSONL line for known agent signatures.
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                # Claude JSONL: has 'sessionId' and 'isSidechain' top-level keys.
                if "sessionId" in obj and "isSidechain" in obj:
                    return "claude"
                # Codex JSONL: has 'type' == 'response' or 'item' with response_item.
                typ = obj.get("type", "")
                if typ in ("response", "response_item", "function_call_output"):
                    return "codex"
                # Try a few more lines rather than giving up on the first.
    except OSError:
        pass
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def replay_session(
    path: Path,
    tools: list[str],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict | None:
    """Replay reducer tools over tool outputs in one transcript.

    For each tool in *tools*, pipe each relevant tool_output text through the
    reducer and accumulate tokens_in / tokens_out.  Returns a dict keyed by
    tool name, each value being::

        {
            "tokens_in":  int,   # total estimated tokens submitted
            "tokens_out": int,   # total estimated tokens after reduction
            "ratio":      float, # tokens_out / tokens_in  (1.0 = no saving)
            "samples":    int,   # number of successful reducer calls
            "failures":   int,   # number of failed reducer calls
        }

    Returns None if the file yields no tool_output events.
    """
    # Collect events WITH their real recorded text. Replaying synthetic text
    # is methodologically meaningless (reducers are content-sensitive), so
    # events whose text could not be re-extracted are skipped and counted.
    events = list(_iter_tool_outputs_with_text(path))
    if not events:
        return None

    acc: dict[str, dict] = {
        t: {"tokens_in": 0, "tokens_out": 0, "samples": 0, "failures": 0,
            "unsupported": 0, "no_text": 0}
        for t in tools
    }

    for event in events:
        cat = event.get("category", "other")
        text = event.get("_text", "")
        target = event.get("target")

        for tool in tools:
            # Check category eligibility
            if tool == "rtk" and cat not in _RTK_CATEGORIES:
                continue
            if not text:
                acc[tool]["no_text"] += 1
                continue
            # headroom addresses all categories
            result = _reduce(tool, text, timeout, target=target)
            if result is None:
                acc[tool]["failures"] += 1
            else:
                acc[tool]["tokens_in"] += result["tokens_in"]
                acc[tool]["tokens_out"] += result["tokens_out"]
                acc[tool]["samples"] += 1
                if result.get("unsupported"):
                    acc[tool]["unsupported"] += 1

    # Compute per-tool ratios
    out: dict[str, dict] = {}
    for t in tools:
        a = acc[t]
        tin = a["tokens_in"]
        tout = a["tokens_out"]
        ratio = (tout / tin) if tin > 0 else 1.0
        out[t] = {
            "tokens_in": tin,
            "tokens_out": tout,
            "ratio": ratio,
            "samples": a["samples"],
            "failures": a["failures"],
            "unsupported": a.get("unsupported", 0),
            "no_text": a.get("no_text", 0),
        }
    return out


# We need tool outputs with their actual text.  Patch _iter_tool_outputs to
# also carry the text so replay_session can use real data.

def _iter_tool_outputs_with_text(path: Path) -> Iterator[dict]:
    """Yield tool_output events augmented with '_text' field."""
    try:
        agent = _detect_agent(path)
        if agent == "claude":
            from tokensmash.sessions.claude import parse_session
        elif agent == "codex":
            from tokensmash.sessions.codex import parse_session
        else:
            return
        result = parse_session(path)
        if result is None:
            return
        _record, timeline = result
        # The parsers don't store text on timeline events (privacy).
        # We re-parse for text using the JSONL directly.
        texts = _extract_texts_from_jsonl(path, agent)
        tool_output_idx = 0
        for event in timeline:
            if event.get("kind") == "tool_output":
                e = dict(event)
                if tool_output_idx < len(texts):
                    e["_text"] = texts[tool_output_idx]
                    tool_output_idx += 1
                yield e
    except Exception:
        return


def _extract_texts_from_jsonl(path: Path, agent: str) -> list[str]:
    """Extract tool result texts from a JSONL file in order."""
    import json as _json

    texts: list[str] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                except Exception:
                    continue
                if agent == "claude":
                    texts.extend(_claude_texts(obj))
                elif agent == "codex":
                    text = _codex_text(obj)
                    if text:
                        texts.append(text)
    except OSError:
        pass
    return texts


def _claude_texts(obj: dict) -> list[str]:
    """Extract tool_result texts from a Claude JSONL entry."""
    out: list[str] = []
    msg = obj.get("message") or {}
    if not isinstance(msg, dict):
        return out
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                t = _extract_content_text(block.get("content", ""))
                if t:
                    out.append(t)
    # Also check toolUseResult (legacy path)
    tur = obj.get("toolUseResult")
    if tur:
        t = _extract_content_text(tur)
        if t:
            out.append(t)
    return out


def _codex_text(obj: dict) -> str:
    """Extract tool output text from a Codex JSONL entry.

    Real format: {"type": "response_item", "payload": {"type":
    "function_call_output", "call_id": ..., "output": "<str>"}}. The output
    string is often itself JSON ({"output": "...", "metadata": {...}}) for
    exec_command results — unwrap one level when that shape parses.
    """
    payload = obj.get("payload")
    item = payload if isinstance(payload, dict) else (obj.get("item") or obj)
    if not isinstance(item, dict):
        return ""
    if item.get("type") not in ("function_call_output", "custom_tool_call_output"):
        return ""
    output = item.get("output", "")
    if not isinstance(output, str):
        return ""
    if output.startswith("{"):
        try:
            inner = json.loads(output)
            if isinstance(inner, dict) and isinstance(inner.get("output"), str):
                return inner["output"]
        except (json.JSONDecodeError, ValueError):
            pass
    return output


def _extract_content_text(content) -> str:
    """Recursively extract text from a content block or string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            t = _extract_content_text(block)
            if t:
                parts.append(t)
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        # image or other — skip
        return ""
    return ""


# Override the lazy version with the text-carrying version
_iter_tool_outputs = _iter_tool_outputs_with_text


# ---------------------------------------------------------------------------
# Corpus replay
# ---------------------------------------------------------------------------

def replay_corpus(
    roots: "dict[str, Path]",
    tools: list[str],
    limit_sessions: int | None = None,
    seed: int = 17,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    """Replay reducer tools over a random sample of sessions across agents.

    Args:
        roots: mapping of agent name → transcript root directory.
        tools: list of tool names to replay (``"rtk"``, ``"headroom"``).
        limit_sessions: if given, randomly sample at most this many sessions
            (seeded with *seed*) from the full corpus.
        seed: RNG seed for reproducible sampling (default 17).
        timeout: per-subprocess timeout in seconds.

    Returns a dict::

        {
            "tools": {
                "<tool>": {
                    "tokens_in":  int,
                    "tokens_out": int,
                    "ratio":      float,
                    "samples":    int,
                    "failures":   int,
                }
            },
            "sessions_processed": int,
            "sessions_skipped":   int,
        }
    """
    from tokensmash.sessions.claude import iter_session_files as iter_claude
    from tokensmash.sessions.codex import iter_session_files as iter_codex

    # Contract root keys are "codex" / "claude-code"; tolerate bare "claude".
    _iters = {
        "claude": iter_claude,
        "claude-code": iter_claude,
        "codex": iter_codex,
    }

    # Collect all paths
    all_paths: list[Path] = []
    for agent, root in roots.items():
        it = _iters.get(agent)
        if it is None:
            continue
        try:
            all_paths.extend(it(root))
        except Exception:
            pass

    # Seeded sample
    rng = random.Random(seed)
    if limit_sessions is not None and limit_sessions < len(all_paths):
        all_paths = rng.sample(all_paths, limit_sessions)

    # Aggregate
    agg: dict[str, dict] = {
        t: {"tokens_in": 0, "tokens_out": 0, "samples": 0, "failures": 0,
            "unsupported": 0, "no_text": 0}
        for t in tools
    }
    sessions_processed = 0
    sessions_skipped = 0

    for path in all_paths:
        result = replay_session(path, tools, timeout=timeout)
        if result is None:
            sessions_skipped += 1
            continue
        sessions_processed += 1
        for t in tools:
            tr = result.get(t, {})
            for key in ("tokens_in", "tokens_out", "samples", "failures", "unsupported", "no_text"):
                agg[t][key] += tr.get(key, 0)

    # Compute corpus-level ratios
    tools_out: dict[str, dict] = {}
    for t in tools:
        a = agg[t]
        tin = a["tokens_in"]
        tout = a["tokens_out"]
        ratio = (tout / tin) if tin > 0 else 1.0
        tools_out[t] = {
            "tokens_in": tin,
            "tokens_out": tout,
            "ratio": ratio,
            "samples": a["samples"],
            "failures": a["failures"],
            "unsupported": a["unsupported"],
            "no_text": a["no_text"],
        }

    return {
        "tools": tools_out,
        "sessions_processed": sessions_processed,
        "sessions_skipped": sessions_skipped,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(result: dict) -> str:
    """Generate a markdown report joining realized ratios against ceiling table.

    *result* is the dict returned by :func:`replay_corpus` (or a single-session
    dict from :func:`replay_session` wrapped as ``{"tools": result, ...}``).

    The report reprints the Layer-0 ceilings table with an added
    "realized estimate" column equal to::

        with_rereads_ceiling_usd × realized_ratio

    plus the mandatory caveat block.
    """
    from tokensmash.opportunity import _TOOL_CATEGORIES  # type: ignore[attr-defined]

    # Normalise: accept either replay_corpus output or a plain per-tool dict
    if "tools" in result:
        tools_data = result["tools"]
        sessions_processed = result.get("sessions_processed", 0)
        sessions_skipped = result.get("sessions_skipped", 0)
    else:
        tools_data = result
        sessions_processed = 0
        sessions_skipped = 0

    all_tools = list(_TOOL_CATEGORIES.keys()) + ["headroom"]

    lines: list[str] = []
    lines.append("## Replay: Realized Compression Estimates\n")
    lines.append(
        f"Sessions processed: {sessions_processed}  |  "
        f"Sessions skipped (no data): {sessions_skipped}\n"
    )
    lines.append(
        "| Tool | Samples | Failures | Realized Ratio | Realized Est. (× ceiling) |"
    )
    lines.append(
        "|------|---------|----------|----------------|---------------------------|"
    )

    for tool in all_tools:
        td = tools_data.get(tool, {})
        samples = td.get("samples", 0)
        failures = td.get("failures", 0)
        ratio = td.get("ratio", None)
        tokens_in = td.get("tokens_in", 0)
        tokens_out = td.get("tokens_out", 0)

        if ratio is None or samples == 0:
            ratio_str = "n/a"
            est_str = "n/a"
        else:
            ratio_str = f"{ratio:.3f}"
            # Ceiling × ratio: expressed as a multiplier against the ceiling
            est_str = f"ceiling × {ratio:.3f}"

        lines.append(
            f"| {tool} | {samples} | {failures} | {ratio_str} | {est_str} |"
        )

    lines.append("")
    lines.append("### Notes on realized ratios")
    lines.append(
        "- **tokens_in / tokens_out**: Ratios are computed from sampled tool "
        "outputs replayed through local reducer tools.  `tokens_in` and "
        "`tokens_out` are both approximated as `len(bytes) // 4`; provider "
        "tokenization is not available offline."
    )
    lines.append(
        "- **Trajectory effects not captured**: A single per-category ratio is "
        "applied uniformly.  In practice, tool outputs that appear early in a "
        "session are re-read more times and benefit more from compression; this "
        "is not modelled here."
    )
    lines.append(
        "- **Ceiling × ratio**: The 'realized estimate' column multiplies the "
        "Layer-0 `with_rereads_usd` ceiling by the realized ratio.  The ceiling "
        "assumes 100 % compression; multiplying by the ratio scales it to the "
        "observed compression level."
    )
    lines.append(
        "- **headroom**: Headroom addresses the full wire payload.  Its realized "
        "ratio is measured across all tool output categories combined."
    )
    lines.append(
        "- **repomix**: Out of scope — offline replay requires the exact repo "
        "state at session time, which is not captured in the transcript."
    )
    lines.append(
        "- **Local tools only**: No transcript text is written to disk or sent "
        "to any remote service."
    )

    return "\n".join(lines) + "\n"
