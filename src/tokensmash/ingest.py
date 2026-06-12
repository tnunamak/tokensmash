"""Session ingestion pipeline.

Walks codex and claude session roots, parses transcripts, attaches cost and
opportunity data, joins study assignment records, and upserts into the
STUDY_DIR/sessions.jsonl store.

Usage::

    from tokensmash.ingest import ingest
    stats = ingest({"codex": Path("~/.codex/sessions"), "claude-code": Path("~/.claude/projects")})

Stdlib only.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tokensmash.schema as schema
import tokensmash.pricing as pricing
from tokensmash.study.assign import (
    arm_for,
    assignment_id,
    block_index,
    load_study_config,
)
from tokensmash import store
from tokensmash import opportunity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_assignments(study_dir: Path) -> dict[tuple[str, str], dict]:
    """Load STUDY_DIR/assignments.jsonl keyed by (agent, session_id).

    Last-write-wins (same semantics as store.load_latest).  Returns {} if the
    file does not exist.
    """
    path = study_dir / "assignments.jsonl"
    result: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            agent = rec.get("agent")
            session_id = rec.get("session_id")
            if agent and session_id:
                result[(agent, session_id)] = rec
    return result


def _attach_study(
    record: dict,
    assignments: dict[tuple[str, str], dict],
    config: dict | None,
    study_dir: Path,
) -> dict:
    """Attach arm/assignment/study_id fields to *record* in place (returns it).

    Resolution order:
    1. Direct join on (agent, logical_session_id) in assignments.jsonl →
       arm_source="linked". The hook links the MAIN session; joining on the
       logical id makes subagent transcripts and resumed rollouts inherit the
       parent session's arm, which is the ITT-correct attribution.
    2. If a study config exists, recompute deterministically from (repo_id, started_at
       block) → arm_source="recomputed"
    3. No config: nothing attached.
    """
    agent = record.get("agent", "")
    session_id = record.get("logical_session_id") or record.get("session_id", "")
    key = (agent, session_id)

    if key in assignments:
        asgn = assignments[key]
        record["arm"] = asgn.get("arm")
        record["assignment_id"] = asgn.get("assignment_id")
        record["study_id"] = asgn.get("study_id")
        record["arm_source"] = "linked"
        return record

    if config is not None:
        started_at = record.get("started_at")
        if started_at:
            try:
                dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                ts = dt.timestamp()
            except (ValueError, AttributeError):
                ts = time.time()
        else:
            ts = time.time()
        blk = block_index(ts)
        repo_id = record.get("repo_id", "")
        seed = bytes.fromhex(config["seed"])
        record["arm"] = arm_for(seed, repo_id, blk)
        record["assignment_id"] = assignment_id(repo_id, blk)
        record["study_id"] = config["study_id"]
        record["arm_source"] = "recomputed"

    return record


def _attach_cost(record: dict) -> dict:
    """Attach cost fields to *record* in-place.

    Adds:
    - ``cost_api_usd`` + ``pricing_id``         (or null + "pricing": "unknown-model")
    - ``cost_codex_credits`` + ``credit_rate_id``  (codex only; skipped for claude)
    """
    agent = record.get("agent", "")
    model = record.get("model", "")
    usage = record.get("usage", {})

    usd_result = pricing.cost_usd(usage, agent, model)
    if usd_result is not None:
        record["cost_api_usd"], record["pricing_id"] = usd_result
    else:
        record["cost_api_usd"] = None
        record["pricing_id"] = None
        record.setdefault("notes", [])
        if isinstance(record["notes"], list):
            record["notes"].append("pricing: unknown-model")
        else:
            record["notes"] = ["pricing: unknown-model"]

    if agent == "codex":
        credits_result = pricing.codex_credits(usage, model)
        if credits_result is not None:
            record["cost_codex_credits"], record["credit_rate_id"] = credits_result
        else:
            record["cost_codex_credits"] = None
            record["credit_rate_id"] = None

    return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest(
    roots: dict[str, Path],
    since_days: float | None = None,
) -> dict:
    """Walk session roots, parse, enrich, and upsert into the session store.

    Parameters
    ----------
    roots:
        Mapping of agent name → root directory.  Recognized agents:
        ``"codex"`` and ``"claude-code"``.
    since_days:
        When set, skip files whose ``st_mtime`` is older than this many days
        (cheap mtime check before parsing).

    Returns
    -------
    dict
        ``{"scanned": n, "parsed": n, "added": n, "replaced": n,
           "skipped": n, "parse_errors": n, "unknown_model_sessions": n}``
    """
    from tokensmash.sessions import codex as codex_sessions
    from tokensmash.sessions import claude as claude_sessions

    _session_parsers = {
        "codex": (codex_sessions.parse_session, codex_sessions.iter_session_files),
        "claude-code": (claude_sessions.parse_session, claude_sessions.iter_session_files),
    }

    study_dir = schema.STUDY_DIR
    config = load_study_config(study_dir)
    assignments = _load_assignments(study_dir)
    exclude_repo_ids: set[str] = set(config.get("exclude_repo_ids", [])) if config else set()

    sessions_path = study_dir / "sessions.jsonl"

    # Cutoff timestamp for since_days filter
    cutoff_mtime: float | None = None
    if since_days is not None:
        cutoff_mtime = time.time() - since_days * 86400.0

    stats: dict[str, int] = {
        "scanned": 0,
        "parsed": 0,
        "added": 0,
        "replaced": 0,
        "skipped": 0,
        "parse_errors": 0,
        "unknown_model_sessions": 0,
    }

    enriched_records: list[dict] = []

    for agent, root in roots.items():
        if agent not in _session_parsers:
            continue
        parse_session_fn, iter_files_fn = _session_parsers[agent]

        for path in iter_files_fn(root):
            stats["scanned"] += 1

            # Cheap mtime filter — skip before parsing
            if cutoff_mtime is not None:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                if mtime < cutoff_mtime:
                    stats["skipped"] += 1
                    continue

            # Parse
            try:
                result = parse_session_fn(path)
            except Exception:
                stats["parse_errors"] += 1
                continue

            if result is None:
                stats["skipped"] += 1
                continue

            session_record, timeline = result
            stats["parsed"] += 1

            # Track parse_errors from inside the session record
            inner_errors = session_record.get("parse_errors", 0)
            if isinstance(inner_errors, int) and inner_errors > 0:
                stats["parse_errors"] += inner_errors

            # Unknown model tracking
            if session_record.get("model") in (None, "", "unknown"):
                stats["unknown_model_sessions"] += 1

            # Exclusion by repo_id
            repo_id = session_record.get("repo_id", "")
            if repo_id in exclude_repo_ids:
                session_record["excluded"] = "study-repo"

            # Attach cost
            usd_before = session_record.get("cost_api_usd")
            _attach_cost(session_record)
            if session_record.get("cost_api_usd") is None and usd_before is None:
                stats["unknown_model_sessions"] += 0  # already counted above if model unknown

            # Attach opportunity
            compactions = session_record.get("compactions", 0) or 0
            try:
                # Contract (docs/CONTRACTS.md §7): record["opportunity"] holds
                # the summarize() output itself — tool_ceilings() and report()
                # both read categories from that key, so it must be set BEFORE
                # tool_ceilings() is called and must not be wrapped.
                session_record["opportunity"] = opportunity.summarize(timeline, compactions)
                session_record["opportunity_ceilings"] = opportunity.tool_ceilings(session_record)
            except Exception as exc:
                session_record["opportunity_error"] = str(exc)

            # Attach study arm
            _attach_study(session_record, assignments, config, study_dir)

            _mark_block_boundary(session_record)

            enriched_records.append(session_record)

    # Merge parsed records over the existing store so cross-record exclusion
    # decisions (superseded codex rollouts) see the full picture even on
    # partial (since_days) ingests, then upsert. The store key is transcript
    # identity: one transcript file = one record. session_id is NOT unique —
    # claude subagents share the parent's sessionId and resumed codex
    # sessions write multiple rollout files with the same internal id.
    key = ("agent", "transcript_id")
    merged = store.load_latest(sessions_path, key=key)
    for record in enriched_records:
        merged[(record.get("agent"), record.get("transcript_id"))] = record
    _mark_superseded_codex_rollouts(list(merged.values()))
    if merged:
        added, replaced = store.upsert_many(sessions_path, list(merged.values()), key=key)
        stats["added"] += added
        stats["replaced"] += replaced

    return stats


def _record_total_tokens(record: dict) -> int:
    raw = record.get("provider_raw") or {}
    total = raw.get("total_tokens")
    if isinstance(total, int):
        return total
    usage = record.get("usage") or {}
    return sum(int(usage.get(k) or 0) for k in ("fresh_input", "cache_read", "cache_write", "output"))


def _mark_superseded_codex_rollouts(records: list[dict]) -> None:
    """Keep one rollout per logical codex session; exclude the rest.

    Codex token counters are cumulative ACROSS resumed rollout files of the
    same logical session, so summing rollouts would double-count and only the
    largest cumulative snapshot represents total spend. Keeper = max
    (total_tokens, transcript_id) — deterministic across runs. Excluded
    records keep their data for audit; analysis skips them via `excluded`.
    """
    groups: dict[str, list[dict]] = {}
    for record in records:
        if record.get("agent") != "codex":
            continue
        logical = record.get("logical_session_id")
        if logical:
            groups.setdefault(str(logical), []).append(record)
    for group in groups.values():
        if len(group) < 2:
            for record in group:
                if record.get("excluded") == "codex-superseded-rollout":
                    del record["excluded"]
            continue
        keeper = max(group, key=lambda r: (_record_total_tokens(r), str(r.get("transcript_id"))))
        for record in group:
            if record is keeper:
                if record.get("excluded") == "codex-superseded-rollout":
                    del record["excluded"]
            elif not record.get("excluded"):
                record["excluded"] = "codex-superseded-rollout"


def _mark_block_boundary(record: dict) -> None:
    """Set excluded="block-boundary" when a session spans 2h blocks.

    Pre-registered exclusion (PROTOCOL.md §4): a session crossing a block
    boundary has an ambiguous arm. Applied only when a lower-priority
    exclusion is not already set and both timestamps parse.
    """
    if record.get("excluded"):
        return
    started, ended = record.get("started_at"), record.get("ended_at")
    if not started or not ended:
        return
    try:
        start_ts = datetime.fromisoformat(str(started).replace("Z", "+00:00")).timestamp()
        end_ts = datetime.fromisoformat(str(ended).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return
    if block_index(start_ts) != block_index(end_ts):
        record["excluded"] = "block-boundary"
