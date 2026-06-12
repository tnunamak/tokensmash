"""Multi-user aggregation of scrubbed session exports.

Provides:
  merge(paths)   — load and deduplicate scrubbed JSONL exports
  report(records) — per-machine + combined human-readable report

Stdlib only.  Accepts scrubbed exports only (refuses records with transcript_path).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import tokensmash.schema as schema
from tokensmash.opportunity import tool_ceilings, _TOOL_CATEGORIES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEDUP_KEY = ("machine_id", "agent", "transcript_id")


def _has_transcript_path(record: dict, _path: str = "") -> bool:
    """Return True if *record* contains 'transcript_path' anywhere recursively."""
    for k, v in record.items():
        if k == "transcript_path":
            return True
        if isinstance(v, dict) and _has_transcript_path(v):
            return True
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and _has_transcript_path(item):
                    return True
    return False


def _dedup_key(record: dict) -> tuple:
    return tuple(record.get(k) for k in _DEDUP_KEY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge(paths: list[Path]) -> tuple[list[dict], dict]:
    """Load and deduplicate scrubbed JSONL exports.

    Each file in *paths* must be a scrubbed export (output of
    store.export_scrubbed).  Every record is validated via
    schema.validate_session_record.  Records with a ``transcript_path`` key
    anywhere (including nested) cause the entire file to be rejected with
    ValueError — that indicates an unscrubbed store was passed.

    Deduplication is on (machine_id, agent, transcript_id) with last-wins
    semantics.  The returned list is sorted by (machine_id, agent, started_at).

    Returns:
        (records, stats) where stats = {"conflicts": n, "invalid_skipped": n}.
    """
    merged: dict[tuple, dict] = {}
    conflicts = 0
    invalid_skipped = 0

    for path in paths:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_skipped += 1
                continue

            if not isinstance(record, dict):
                invalid_skipped += 1
                continue

            # Refuse entire file if transcript_path present anywhere.
            if _has_transcript_path(record):
                raise ValueError(
                    f"File '{path}' line {lineno} contains 'transcript_path'; "
                    "only scrubbed exports are accepted.  Run store.export_scrubbed "
                    "before passing to meta.merge."
                )

            # Validate schema.
            errors = schema.validate_session_record(record)
            if errors:
                invalid_skipped += 1
                continue

            key = _dedup_key(record)
            if key in merged:
                conflicts += 1
            merged[key] = record

    result = sorted(
        merged.values(),
        key=lambda r: (r.get("machine_id") or "", r.get("agent") or "", r.get("started_at") or ""),
    )
    return result, {"conflicts": conflicts, "invalid_skipped": invalid_skipped}


def report(records: list[dict]) -> str:
    """Generate a per-machine + combined human-readable report.

    Args:
        records: List of valid, deduplicated session records (e.g. from merge()).

    Returns:
        A markdown-formatted string with:
        - Header: N machines / N sessions
        - Per-machine block: sessions, total cost, cache-read share, top models
        - Combined section: aggregate stats + per-tool opportunity ceilings
        - Caveat line about self-selection bias
    """
    if not records:
        return (
            "## Multi-user aggregate report\n\n"
            "0 machines / 0 sessions\n\n"
            "_No records to report._\n\n"
            "> **Note:** Machines are self-selected and not a random sample; "
            "aggregate figures may not generalise.\n"
        )

    # Group by machine_id.
    by_machine: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        mid = rec.get("machine_id") or "unknown"
        by_machine[mid].append(rec)

    n_machines = len(by_machine)
    n_sessions = len(records)

    lines: list[str] = []
    lines.append("## Multi-user aggregate report")
    lines.append("")
    lines.append(f"**{n_machines} machine{'s' if n_machines != 1 else ''} / "
                 f"{n_sessions} session{'s' if n_sessions != 1 else ''}**")
    lines.append("")

    # Per-machine blocks.
    for mid in sorted(by_machine.keys()):
        recs = by_machine[mid]
        lines.append(f"### Machine `{mid}`")
        lines.append("")
        lines.append(_machine_block(recs))
        lines.append("")

    # Combined section.
    lines.append("### Combined")
    lines.append("")
    lines.append(_combined_block(records))
    lines.append("")
    lines.append(
        "> **Note:** Machines are self-selected and not a random sample; "
        "aggregate figures may not generalise."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------

def _sum_usage(records: list[dict]) -> dict[str, int]:
    """Sum canonical usage fields across records."""
    totals: dict[str, Any] = {k: 0 for k in schema.USAGE_KEYS if k != "reasoning_output"}
    totals["reasoning_output"] = None
    for rec in records:
        usage = rec.get("usage") or {}
        for k in schema.USAGE_KEYS:
            if k == "reasoning_output":
                v = usage.get(k)
                if isinstance(v, int):
                    totals[k] = (totals[k] or 0) + v
            else:
                totals[k] += usage.get(k) or 0
    return totals


def _cache_read_share(usage: dict) -> float | None:
    """Cache-read tokens as a fraction of total input tokens; None if zero input."""
    total_input = (usage.get("fresh_input") or 0) + (usage.get("cache_read") or 0)
    if total_input == 0:
        return None
    return (usage.get("cache_read") or 0) / total_input


def _top_models(records: list[dict], n: int = 3) -> list[tuple[str, int]]:
    """Return up to *n* (model, count) pairs sorted by count descending."""
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        m = rec.get("model") or "unknown"
        counts[m] += 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n]


def _total_cost(records: list[dict]) -> float:
    return sum(rec.get("cost_api_usd") or 0.0 for rec in records)


def _machine_block(records: list[dict]) -> str:
    """Render a summary block for a single machine."""
    n = len(records)
    cost = _total_cost(records)
    usage = _sum_usage(records)
    share = _cache_read_share(usage)
    models = _top_models(records)

    share_str = f"{share * 100:.1f}%" if share is not None else "n/a"
    model_str = ", ".join(f"{m} ({c})" for m, c in models) if models else "n/a"

    return (
        f"- Sessions: {n}\n"
        f"- Total cost (API-equivalent USD): ${cost:.4f}\n"
        f"- Cache-read share of input tokens: {share_str}\n"
        f"- Top models: {model_str}"
    )


def _combined_block(records: list[dict]) -> str:
    """Render combined stats + per-tool opportunity ceilings table."""
    n = len(records)
    cost = _total_cost(records)
    usage = _sum_usage(records)
    share = _cache_read_share(usage)
    models = _top_models(records, n=5)

    share_str = f"{share * 100:.1f}%" if share is not None else "n/a"
    model_str = ", ".join(f"{m} ({c})" for m, c in models) if models else "n/a"

    stat_lines = [
        f"- Sessions: {n}",
        f"- Total cost (API-equivalent USD): ${cost:.4f}",
        f"- Cache-read share of input tokens: {share_str}",
        f"- Top models: {model_str}",
    ]

    # Opportunity ceilings — aggregate using tool_ceilings per record.
    opp_table = _opportunity_table(records)

    return "\n".join(stat_lines) + "\n\n" + opp_table


def _opportunity_table(records: list[dict]) -> str:
    """Aggregate per-tool opportunity ceilings across all records."""
    tools = list(_TOOL_CATEGORIES.keys()) + ["headroom"]

    agg: dict[str, dict[str, float]] = {
        t: {"ins_usd": 0.0, "reread_usd": 0.0, "actual_usd": 0.0,
            "sessions": 0, "skipped": 0}
        for t in tools
    }

    for rec in records:
        ceilings = tool_ceilings(rec)
        actual_cost = rec.get("cost_api_usd") or 0.0
        if not ceilings:
            for t in tools:
                agg[t]["skipped"] += 1
            continue
        for t in tools:
            tc = ceilings.get(t, {})
            agg[t]["ins_usd"] += tc.get("insertion_only_usd", 0.0)
            agg[t]["reread_usd"] += tc.get("with_rereads_usd", 0.0)
            agg[t]["actual_usd"] += actual_cost
            agg[t]["sessions"] += 1

    header = (
        "| Tool         | Ins-only ceiling | +Rereads ceiling | "
        "Actual cost | Ceiling % actual | Sessions | Skipped |"
    )
    sep = (
        "|:-------------|----------------:|----------------:|"
        "-----------:|-----------------:|---------:|--------:|"
    )
    rows = [header, sep]
    for t in tools:
        d = agg[t]
        ins = d["ins_usd"]
        reread = d["reread_usd"]
        actual = d["actual_usd"]
        pct = (reread / actual * 100) if actual > 0 else 0.0
        row = (
            f"| {t:<12} | "
            f"${ins:>14.4f} | "
            f"${reread:>14.4f} | "
            f"${actual:>9.4f} | "
            f"{pct:>15.1f}% | "
            f"{d['sessions']:>8} | "
            f"{d['skipped']:>7} |"
        )
        rows.append(row)

    caveats = (
        "\n"
        "**Caveats**\n"
        "- Ceilings are deliberate 100%-compression upper bounds, not predictions.\n"
        "- Cache-read repricing assumes prefix stability across compactions; "
        "actual savings will be lower if caching breaks.\n"
        "- Subscription quota weighting for Claude is opaque; figures are "
        "API-equivalent USD only and do not reflect subscription value.\n"
    )

    return "\n".join(rows) + caveats
