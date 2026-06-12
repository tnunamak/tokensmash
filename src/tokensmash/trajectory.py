"""Trajectory / wasted-exploration analysis.

Implements §12 of docs/CONTRACTS.md.

Identifies *unused reads* — file_read tool outputs whose target path is never
referenced in any later tool output or write operation within the same session.
These represent reads that may not have been acted upon.

Normalization limits
--------------------
Targets are normalised with ``pathlib.PurePosixPath`` (``os.path.normpath``
on all platforms) and made absolute when the session record carries a ``cwd``.
Limits:

* Targets that are ``None`` or empty strings are excluded from both the "read"
  set and the "later-reference" set.
* If the session has no ``cwd``, relative paths are compared as-is after
  ``normpath``; two relative paths that resolve to the same absolute path via
  different cwds will *not* be matched (false positives in unused-read count).
* Windows-style paths in transcripts on a Linux host are not translated; the
  normalisation is purely lexical (``normpath``).
* Path comparison is case-sensitive (consistent with Linux filesystems).

The reported ``reads_unused`` count is an *upper bound* on wasted reads —
a file may have been read to orient the model even when it was never directly
edited or referenced in a subsequent tool call.

Stdlib only; no imports from cli.py, opportunity.py, or ingest.py.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Tool-name sets used in write-target detection
# ---------------------------------------------------------------------------

# Tool names that write/modify files: their targets are also "file was used"
_WRITE_TOOL_NAMES = frozenset({
    # Claude Code
    "Edit", "Write",
    # Codex / shell-based patching
    "apply_patch",
})


# ---------------------------------------------------------------------------
# Path normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_path(raw: str, cwd: str | None) -> str | None:
    """Return a normalised, absolute-when-possible path string.

    Returns None if *raw* is empty or non-string.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = raw.strip()
    # Make absolute if cwd is known and path is relative
    if cwd and not os.path.isabs(path):
        path = os.path.join(cwd, path)
    return os.path.normpath(path)


# ---------------------------------------------------------------------------
# Per-session analysis
# ---------------------------------------------------------------------------

def analyze_session(record: dict, timeline: list[dict]) -> dict[str, Any]:
    """Analyse one session's trajectory.

    Parameters
    ----------
    record:
        Session record dict (as returned by a parser).
    timeline:
        Ordered list of timeline events (as returned by a parser).

    Returns
    -------
    dict with keys:

    * ``reads``              — total file_read tool outputs observed
    * ``reads_unused``       — file_reads whose target never appears later
                               (upper bound on wasted reads)
    * ``tokens_unused``      — sum of tokens_est for unused reads
    * ``searches``           — total search tool outputs observed
    * ``searches_unfollowed`` — searches with no subsequent tool output
                               (possible dead-end searches)
    """
    cwd: str | None = record.get("cwd")

    # Separate tool_output events by position so we can do "later" checks
    tool_outputs: list[dict] = [e for e in timeline if e.get("kind") == "tool_output"]

    reads: int = 0
    reads_unused: int = 0
    tokens_unused: int = 0
    searches: int = 0
    searches_unfollowed: int = 0

    for i, ev in enumerate(tool_outputs):
        category = ev.get("category", "")
        raw_target = ev.get("target")
        tokens_est: int = ev.get("tokens_est", 0)

        # ── file_read analysis ───────────────────────────────────────────────
        if category == "file_read":
            reads += 1
            norm = _normalise_path(raw_target, cwd) if raw_target else None

            if norm is None:
                # No target — can't determine usage; count as unused (conservative)
                reads_unused += 1
                tokens_unused += tokens_est
            else:
                # Check all later tool outputs for a reference to this path
                used = False
                for later in tool_outputs[i + 1:]:
                    later_raw = later.get("target")
                    if later_raw:
                        later_norm = _normalise_path(later_raw, cwd)
                        if later_norm == norm:
                            used = True
                            break
                    # Also treat write-tool events without target as "using something"
                    # but we can't match them — skip.
                if not used:
                    reads_unused += 1
                    tokens_unused += tokens_est

        # ── search analysis ──────────────────────────────────────────────────
        elif category == "search":
            searches += 1
            # "Unfollowed" = no file_read among the remaining tool outputs of
            # the session: the search led to nothing being opened. A proxy for
            # dead-end exploration (the next search refining a query still
            # counts the earlier one as unfollowed only if no read EVER comes).
            if not any(
                later.get("category") == "file_read"
                for later in tool_outputs[i + 1 :]
            ):
                searches_unfollowed += 1

    return {
        "reads": reads,
        "reads_unused": reads_unused,
        "tokens_unused": tokens_unused,
        "searches": searches,
        "searches_unfollowed": searches_unfollowed,
    }


# ---------------------------------------------------------------------------
# Corpus analysis
# ---------------------------------------------------------------------------

def analyze_corpus(
    roots: dict[str, Path],
    limit_sessions: int | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Analyse a random sample of sessions from *roots*.

    Parameters
    ----------
    roots:
        Mapping of agent-name to session root directory.
        Keys typically "claude-code" and/or "codex".
    limit_sessions:
        If given, sample at most this many sessions (using *seed*).
        If None, all sessions are analysed.
    seed:
        Random seed for reproducible sampling (default: 17).

    Returns
    -------
    Aggregate dict — no paths, only counts and token totals:

    * ``sessions_sampled``   — number of sessions successfully parsed
    * ``sessions_skipped``   — sessions that failed to parse (no usage data)
    * ``reads``              — total file_read events
    * ``reads_unused``       — total unused reads (upper bound)
    * ``tokens_unused``      — total tokens_est for unused reads
    * ``reads_unused_pct``   — reads_unused / reads * 100 (or 0.0 if reads==0)
    * ``searches``           — total search events
    * ``searches_unfollowed``— total unfollowed searches
    """
    from tokensmash.sessions.codex import (
        iter_session_files as codex_iter,
        parse_session as codex_parse,
    )
    from tokensmash.sessions.claude import (
        iter_session_files as claude_iter,
        parse_session as claude_parse,
    )

    # Collect candidate paths
    candidates: list[tuple[str, Path]] = []
    for agent, root in roots.items():
        root = Path(root)
        if agent == "codex":
            for p in codex_iter(root):
                candidates.append(("codex", p))
        else:
            for p in claude_iter(root):
                candidates.append((agent, p))

    # Sample
    rng = random.Random(seed)
    if limit_sessions is not None and limit_sessions < len(candidates):
        candidates = rng.sample(candidates, limit_sessions)
    else:
        rng.shuffle(candidates)

    # Aggregate
    sessions_sampled = 0
    sessions_skipped = 0
    reads = 0
    reads_unused = 0
    tokens_unused = 0
    searches = 0
    searches_unfollowed = 0

    for agent, path in candidates:
        try:
            if agent == "codex":
                result = codex_parse(path)
            else:
                result = claude_parse(path)
        except Exception:
            sessions_skipped += 1
            continue

        if result is None:
            sessions_skipped += 1
            continue

        record, tl = result
        sessions_sampled += 1
        stats = analyze_session(record, tl)
        reads += stats["reads"]
        reads_unused += stats["reads_unused"]
        tokens_unused += stats["tokens_unused"]
        searches += stats["searches"]
        searches_unfollowed += stats["searches_unfollowed"]

    reads_unused_pct = (reads_unused / reads * 100) if reads > 0 else 0.0

    return {
        "sessions_sampled": sessions_sampled,
        "sessions_skipped": sessions_skipped,
        "reads": reads,
        "reads_unused": reads_unused,
        "tokens_unused": tokens_unused,
        "reads_unused_pct": reads_unused_pct,
        "searches": searches,
        "searches_unfollowed": searches_unfollowed,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report(result: dict) -> str:
    """Format an analyze_session or analyze_corpus result as a human-readable string.

    Works for both single-session dicts (no sessions_sampled key) and corpus
    dicts.
    """
    lines: list[str] = []

    is_corpus = "sessions_sampled" in result

    lines.append("=== Trajectory / Wasted-Exploration Report ===")
    lines.append("")

    if is_corpus:
        sampled = result.get("sessions_sampled", 0)
        skipped = result.get("sessions_skipped", 0)
        lines.append(f"Sessions analysed : {sampled}")
        lines.append(f"Sessions skipped  : {skipped}  (no usage data or parse error)")
        lines.append("")

    reads = result.get("reads", 0)
    reads_unused = result.get("reads_unused", 0)
    tokens_unused = result.get("tokens_unused", 0)
    reads_unused_pct = result.get("reads_unused_pct", (reads_unused / reads * 100) if reads > 0 else 0.0)
    searches = result.get("searches", 0)
    searches_unfollowed = result.get("searches_unfollowed", 0)
    searches_unfollowed_pct = (searches_unfollowed / searches * 100) if searches > 0 else 0.0

    lines.append("File reads")
    lines.append(f"  Total reads         : {reads}")
    lines.append(f"  Unused reads        : {reads_unused}"
                 f"  ({reads_unused_pct:.1f}%)")
    lines.append(f"  Tokens (unused)     : {tokens_unused}")
    lines.append("")
    lines.append("Searches")
    lines.append(f"  Total searches      : {searches}")
    lines.append(f"  Unfollowed searches : {searches_unfollowed}"
                 f"  ({searches_unfollowed_pct:.1f}%)")
    lines.append("")

    # ── Proxy caveat (required by contract §12) ──────────────────────────────
    lines.append("Caveats")
    lines.append(
        "  'Unused' reads are identified by checking whether the read target path"
    )
    lines.append(
        "  appears in any later tool call or write operation within the same session."
    )
    lines.append(
        "  This is a PROXY metric: a read may have informed the model's reasoning"
    )
    lines.append(
        "  even when the file was never subsequently referenced in a tool call."
    )
    lines.append(
        "  The reported count is therefore an UPPER BOUND on wasted reads."
    )
    lines.append(
        "  No lower bound on orientation-tool headroom is claimed."
    )
    lines.append(
        "  Path comparison is lexical (normpath); symlinks and relative paths"
    )
    lines.append(
        "  without a known cwd may produce false positives."
    )

    return "\n".join(lines)
