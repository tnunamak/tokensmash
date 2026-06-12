"""Study linkage: SessionStart hook handler.

See docs/CONTRACTS.md §4 and docs/study-architecture.md D5.

CRITICAL silence contract
--------------------------
link_from_hook MUST NEVER write to stdout or stderr.  Claude Code's
SessionStart hook captures stdout and injects it into the agent's context
window, which would contaminate the measurement.  All diagnostics go to
STUDY_DIR/errors.log instead.

CRITICAL exit-0 contract
-------------------------
link_from_hook ALWAYS returns 0.  A non-zero exit would abort the agent
session; we must never block the user's work due to instrumentation errors.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import tokensmash.schema as _schema
from tokensmash.study.assign import (
    arm_for,
    assignment_id,
    block_index,
    load_study_config,
)


def link_from_hook(stdin_text: str, agent: str, now: float | None = None) -> int:
    """Parse a SessionStart hook payload and append an assignment record.

    Parameters
    ----------
    stdin_text:
        Raw stdin from the hook (JSON with at minimum session_id and cwd).
    agent:
        Agent identifier, e.g. "claude-code" or "codex".
    now:
        Unix timestamp override for the current time (defaults to time.time()).
        Used in tests to produce deterministic block values.

    Returns
    -------
    int
        Always 0.
    """
    # Use schema.STUDY_DIR so monkeypatching in tests works transparently.
    study_dir = _schema.STUDY_DIR
    try:
        _link(stdin_text, agent, now=now, study_dir=study_dir)
    except Exception:
        _log_error(study_dir, traceback.format_exc())
    return 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _link(
    stdin_text: str,
    agent: str,
    now: float | None,
    study_dir: Path,
) -> None:
    """Core logic — may raise; caller catches and logs."""
    config = load_study_config(study_dir)
    if config is None:
        return  # No study configured; silent no-op.

    # Parse hook payload.
    hook = json.loads(stdin_text)  # raises on malformed JSON
    session_id = hook["session_id"]
    cwd = hook["cwd"]
    transcript_path = hook.get("transcript_path")
    source = hook.get("source", "new")

    # Compute identifiers.
    ts = now if now is not None else time.time()
    blk = block_index(ts)
    repo_id = _schema.stable_id("repo", _schema.repo_identity(cwd))
    seed = bytes.fromhex(config["seed"])
    arm = arm_for(seed, repo_id, blk)
    assign_id = assignment_id(repo_id, blk)

    # Build the assignment record.
    record: dict = {
        "schema": _schema.ASSIGNMENT_SCHEMA,
        "study_id": config["study_id"],
        "agent": agent,
        "session_id": session_id,
        "repo_id": repo_id,
        "block": blk,
        "arm": arm,
        "assignment_id": assign_id,
        "mode": config["mode"],
        "linked_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "source": source,
        "transcript_path": transcript_path,
    }
    if source == "resume":
        record["resumed"] = True

    # Append to assignments.jsonl — create parent dirs on first write.
    study_dir.mkdir(parents=True, exist_ok=True)
    assignments_path = study_dir / "assignments.jsonl"
    with assignments_path.open("a") as fh:
        fh.write(_schema.dumps_record(record) + "\n")


def _log_error(study_dir: Path, message: str) -> None:
    """Append a single-line error entry to errors.log.

    Must not raise; if logging itself fails, we swallow it to honour the
    unconditional exit-0 contract.
    """
    try:
        study_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        # Collapse to one line so each log entry is self-contained.
        one_line = message.replace("\n", " | ")
        with (study_dir / "errors.log").open("a") as fh:
            fh.write(f"{ts} {one_line}\n")
    except Exception:
        pass
