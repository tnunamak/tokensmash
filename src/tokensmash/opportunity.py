"""Opportunity bounds for tokensmash — Layer 0 cost-ceiling analysis.

Design rationale (D9 in docs/study-architecture.md):
    Per-tool addressable spend is computed with formulas that deliberately
    over-estimate what a tool could save (100% compression assumed), so the
    results are defensible *upper bounds*, not predictions.

Upper-bound logic:
    A tool output inserted at request-turn t contributes tokens to the context
    for every subsequent API request until the next compaction boundary.  The
    first request after insertion pays fresh-input pricing; all subsequent
    requests in the same segment pay cache-read pricing (prefix caching
    assumption — generous in the tool's favour).

    insertion_only_usd  = insertion_tokens × fresh_input_rate
    with_rereads_usd    = insertion_only_usd
                        + reread_tokens × cache_read_rate

    where reread_tokens = Σ (tokens_est × rereads_until_next_compaction)
    and   rereads       = number of later request events in the same segment.

    Headroom is special: it compresses the entire wire payload, so its ceiling
    is the full session input bill (fresh_input + cache_read).  Both bounds
    are set to that value, and a "note" is added explaining the wire-payload
    semantics.

Compaction-boundary approximation:
    With C compactions, the request sequence is split into C+1 equal segments.
    Codex sessions always have compactions=0 (single segment), which is the
    generous direction consistent with upper bounds.

Stdlib only.  No external dependencies.
"""

from __future__ import annotations

import math
from typing import Any

from tokensmash.pricing import load_tables, resolve_model

# ---------------------------------------------------------------------------
# Category constants
# ---------------------------------------------------------------------------

CATEGORIES = ("shell", "file_read", "search", "mcp", "web", "other")

# Tool-to-category mappings used by tool_ceilings.
# Each entry maps a tool name to the list of categories it addresses.
_TOOL_CATEGORIES: dict[str, list[str]] = {
    "rtk": ["shell"],
    "context-mode": ["shell", "search", "mcp"],
    "repomix": ["file_read"],
    # headroom is handled specially (wire-payload bound)
}


def _compute_segment_boundaries(request_indices: list[int], compactions: int) -> list[int]:
    """Return the first request index of each segment boundary (exclusive end).

    With C compactions the request sequence is divided into C+1 equal segments.
    Returns a list of C boundary positions (exclusive upper bounds on request
    index within each segment), derived by evenly splitting the total number of
    request events.

    A segment boundary at position b means: request events with index >= b
    belong to the next segment (i.e. a compaction has occurred and reread
    counts reset).
    """
    if not request_indices or compactions <= 0:
        return []

    n = len(request_indices)
    # Approximate: split the *count* of requests evenly into compactions+1 parts.
    # Boundary positions are expressed as sequential ordinal positions (0-based).
    segments = compactions + 1
    boundaries: list[int] = []
    for seg in range(1, segments):
        # The ordinal index of the first request in the next segment.
        boundary_ordinal = math.ceil(n * seg / segments)
        if boundary_ordinal < n:
            boundaries.append(boundary_ordinal)
    return boundaries


def summarize(timeline: list[dict], compactions: int) -> dict:
    """Compute per-category token totals from an ordered session timeline.

    Args:
        timeline:    Ordered list of event dicts.  Two kinds are recognised:
                       {"kind": "request",     "index": int, "usage": {...}}
                       {"kind": "tool_output", "request_index": int,
                        "category": str, "tokens_est": int, "tool_name": str}
                     Any other kind is silently ignored.
        compactions: Number of compaction events in the session.  Codex
                     sessions always pass 0 (whole session = one segment),
                     which is the generous direction for upper bounds.

    Returns:
        Dict keyed by category with sub-dicts::

            {
              "shell":     {"insertion_tokens": int, "reread_tokens": int},
              "file_read": {...},
              ...
            }

        Only categories with at least one tool_output event are included.

    Upper-bound logic:
        rereads for a tool_output at request_index r =
            number of request events with index > r within the same segment.
        reread_tokens += tokens_est * rereads_count
    """
    # Collect all request events in order; track their sequential ordinals.
    request_ordinals: list[int] = []  # ordinal position -> request index value
    for event in timeline:
        if event.get("kind") == "request":
            idx = event.get("index", 0)
            request_ordinals.append(idx)

    # Compute segment boundary ordinals.
    boundaries = _compute_segment_boundaries(request_ordinals, compactions)
    # boundaries[i] = the ordinal position of the first request in segment i+1.

    # Build a lookup: request index value -> which segment it belongs to (0-based).
    request_index_to_segment: dict[int, int] = {}
    # Also build: request index value -> how many requests in its segment come after it.
    # We need: for a tool_output at request_index r, how many requests with index > r
    # are in the same segment?
    #
    # Strategy: build per-segment sorted lists of request index values.
    segments_requests: list[list[int]] = [[] for _ in range(compactions + 1)]
    for ordinal, req_idx in enumerate(request_ordinals):
        seg = 0
        for b in boundaries:
            if ordinal >= b:
                seg += 1
        request_index_to_segment[req_idx] = seg
        segments_requests[seg].append(req_idx)

    # For each segment, for each request index in that segment, count how many
    # requests with strictly higher index exist in the same segment.
    # We sort each segment's request list once.
    segment_sorted: list[list[int]] = [sorted(s) for s in segments_requests]

    def _rereads_in_segment(seg_idx: int, at_request_index: int) -> int:
        """Number of request events with index > at_request_index in segment."""
        lst = segment_sorted[seg_idx]
        # Binary search for insertion point just past at_request_index.
        lo, hi = 0, len(lst)
        while lo < hi:
            mid = (lo + hi) // 2
            if lst[mid] <= at_request_index:
                lo = mid + 1
            else:
                hi = mid
        return len(lst) - lo

    # Accumulate per-category totals.
    result: dict[str, dict[str, int]] = {}

    for event in timeline:
        if event.get("kind") != "tool_output":
            continue

        category = event.get("category", "other")
        if category not in CATEGORIES:
            category = "other"

        tokens_est = event.get("tokens_est", 0) or 0
        req_idx = event.get("request_index", 0)

        # Determine segment for this tool_output.
        # If request_index is beyond any known request, treat as last segment.
        seg = request_index_to_segment.get(req_idx)
        if seg is None:
            # request_index beyond last request: place in the last segment
            seg = compactions  # last segment index

        rereads = _rereads_in_segment(seg, req_idx)

        if category not in result:
            result[category] = {"insertion_tokens": 0, "reread_tokens": 0}
        result[category]["insertion_tokens"] += tokens_est
        result[category]["reread_tokens"] += tokens_est * rereads

    return result


def tool_ceilings(record: dict) -> dict:
    """Compute per-tool USD cost ceilings for a session record.

    Args:
        record: A normalized session record dict.  Expected keys:
                  "usage"       — canonical usage totals dict
                  "model"       — model string
                  "agent"       — agent string ("codex" or "claude-code")
                  "opportunity" — output of summarize() stored on the record
                  "cost_api_usd" — actual session cost (read-only, for context)

    Returns:
        Dict keyed by tool name, each value::

            {"insertion_only_usd": float, "with_rereads_usd": float}

        For headroom, an additional "note" key is present.
        Returns {} for unknown models (never guess).

    Ceiling formulas:
        For category-scoped tools (rtk, context-mode, repomix):
            insertion_only_usd = insertion_tokens × (fresh_input_per_m / 1e6)
            with_rereads_usd   = insertion_only_usd
                               + reread_tokens × (cache_read_per_m / 1e6)

        For headroom (wire-payload bound):
            wire_cost = (usage.fresh_input × fresh_rate
                         + usage.cache_read × cache_read_rate)
            insertion_only_usd = wire_cost  (same as with_rereads)
            with_rereads_usd   = wire_cost
            note: "wire-payload bound"

    Assumptions:
        - 100% compression (deliberate upper bound).
        - Cache-read repricing assumes prefix stability across compactions.
        - Subscription quota weighting for Claude is opaque; USD is API-equivalent.
    """
    agent = record.get("agent", "")
    model = record.get("model", "")
    usage = record.get("usage", {})
    opportunity = record.get("opportunity", {})

    tables = load_tables()
    usd_tables = [t for t in tables if t["kind"] == "api_usd"]
    resolved = resolve_model(usd_tables, agent, model)
    if resolved is None:
        return {}

    rates, _table_id = resolved
    fresh_rate = rates["fresh_input_per_m"] / 1_000_000
    cache_rate = rates["cache_read_per_m"] / 1_000_000

    ceilings: dict[str, Any] = {}

    # Wire-payload cost: the session's entire input bill. No input-side tool
    # can save more than this, so it caps every ceiling below. The raw reread
    # estimate can exceed it because the reread model assumes tool outputs
    # persist in context until compaction, while agents (notably Claude Code)
    # prune old tool results — the transcript cannot observe pruning.
    fresh_input = usage.get("fresh_input", 0) or 0
    cache_read = usage.get("cache_read", 0) or 0
    wire_cost = fresh_input * fresh_rate + cache_read * cache_rate

    # Category-scoped tools.
    for tool_name, categories in _TOOL_CATEGORIES.items():
        ins_tokens = 0
        reread_tokens = 0
        for cat in categories:
            cat_data = opportunity.get(cat, {})
            ins_tokens += cat_data.get("insertion_tokens", 0) or 0
            reread_tokens += cat_data.get("reread_tokens", 0) or 0

        insertion_only = ins_tokens * fresh_rate
        with_rereads = insertion_only + reread_tokens * cache_rate
        entry: dict[str, Any] = {
            "insertion_only_usd": insertion_only,
            "with_rereads_usd": with_rereads,
        }
        if wire_cost > 0 and with_rereads > wire_cost:
            entry["with_rereads_usd"] = wire_cost
            entry["capped"] = "wire-payload"
        ceilings[tool_name] = entry

    # Headroom: wire-payload bound = entire input bill.
    ceilings["headroom"] = {
        "insertion_only_usd": wire_cost,
        "with_rereads_usd": wire_cost,
        "note": "wire-payload bound",
    }

    return ceilings


def report(records: list[dict]) -> str:
    """Generate a human-readable markdown table of opportunity ceilings.

    Aggregates across all records per tool, showing:
      - Total insertion-only USD ceiling
      - Total with-rereads USD ceiling
      - Total actual cost_api_usd of the same records
      - Ceiling as % of actual cost
      - Count of sessions contributing vs skipped (unknown model)

    Args:
        records: List of normalized session records, each having been enriched
                 with "opportunity" (output of summarize()) and "cost_api_usd".

    Returns:
        A markdown-formatted string table with a caveats footer.
    """
    # Collect per-tool aggregates.
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
            # Unknown model — count as skipped for all tools.
            for t in tools:
                agg[t]["skipped"] += 1
            continue
        for t in tools:
            tc = ceilings.get(t, {})
            agg[t]["ins_usd"] += tc.get("insertion_only_usd", 0.0)
            agg[t]["reread_usd"] += tc.get("with_rereads_usd", 0.0)
            agg[t]["actual_usd"] += actual_cost
            agg[t]["sessions"] += 1

    # Render table.
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

    table = "\n".join(rows)

    caveats = (
        "\n"
        "**Caveats**\n"
        "- These are deliberate 100%-compression upper bounds, not predictions.\n"
        "- Cache-read repricing assumes prefix stability across compactions; "
        "actual savings will be lower if caching breaks.\n"
        "- Subscription quota weighting for Claude is opaque; figures are "
        "API-equivalent USD only and do not reflect subscription value.\n"
        "- Headroom ceilings represent the full input-token wire-payload cost; "
        "Headroom compression can also break prefix caching, which would reduce "
        "cache-read savings elsewhere.\n"
    )

    return table + caveats
