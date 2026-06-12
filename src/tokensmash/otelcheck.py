"""OTel second-meter cross-validation for tokensmash.

Parses OTLP JSON-lines exports of Claude Code's ``claude_code.token.usage``
counter metric and compares the per-session sums against the authoritative
transcript store records.

The transcript store is the **primary** source of truth.  OTel is an optional
independent meter: when both sources agree the transcript totals are confirmed;
when they disagree the discrepancy surfaces here for investigation.

OTLP path parsed
----------------
Line (JSON object)
  .resourceMetrics[]
    .scopeMetrics[]
      .metrics[]                      -- filter: name == "claude_code.token.usage"
        .sum.dataPoints[]
          .attributes[]               -- list of {key, value: {stringValue}}
          .asInt  OR  .asDouble       -- token count for this data point

Attribute keys used:
  ``type``       -- one of: input | output | cacheRead | cacheCreation
  ``session.id`` -- session identifier (joined to store records' session_id)

Canonical field mapping
-----------------------
  input          -> fresh_input
  cacheRead      -> cache_read
  cacheCreation  -> cache_write
  output         -> output

Stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Attribute → canonical field
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, str] = {
    "input": "fresh_input",
    "cacheRead": "cache_read",
    "cacheCreation": "cache_write",
    "output": "output",
}

_CANONICAL_FIELDS = ("fresh_input", "cache_read", "cache_write", "output")

_DISAGREEMENT_THRESHOLD = 0.01  # 1 %


def _empty_usage() -> dict[str, int]:
    return {f: 0 for f in _CANONICAL_FIELDS}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _extract_attrs(attributes: Any) -> dict[str, str]:
    """Return a flat {key: stringValue} dict from an OTLP attributes list.

    Skips any entry that is not a well-formed ``{key, value: {stringValue}}``.
    """
    result: dict[str, str] = {}
    if not isinstance(attributes, list):
        return result
    for attr in attributes:
        if not isinstance(attr, dict):
            continue
        key = attr.get("key")
        if not isinstance(key, str):
            continue
        val_wrapper = attr.get("value")
        if not isinstance(val_wrapper, dict):
            continue
        sv = val_wrapper.get("stringValue")
        if not isinstance(sv, str):
            continue
        result[key] = sv
    return result


def _dp_value(dp: dict[str, Any]) -> int | None:
    """Return the integer token count from a dataPoint, or None if absent."""
    if "asInt" in dp:
        raw = dp["asInt"]
        # OTLP JSON may encode int64 as a string
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError:
                return None
        if isinstance(raw, (int, float)):
            return int(raw)
    if "asDouble" in dp:
        raw = dp["asDouble"]
        if isinstance(raw, (int, float)):
            return int(raw)
    return None


def parse_otlp_jsonl(path: Path) -> dict[str, dict[str, int]]:
    """Parse an OTLP JSON-lines file and return per-session canonical usage.

    Returns a mapping ``{session_id: {fresh_input, cache_read, cache_write,
    output}}`` where all values are non-negative integers summed across all
    matching data points.  Malformed lines or data points are silently skipped.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file produced by an OTLP file-exporter
        (one ``ExportMetricsServiceRequest`` JSON object per line).
    """
    sessions: dict[str, dict[str, int]] = {}

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return sessions

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue

        resource_metrics = obj.get("resourceMetrics")
        if not isinstance(resource_metrics, list):
            continue

        for rm in resource_metrics:
            if not isinstance(rm, dict):
                continue
            scope_metrics_list = rm.get("scopeMetrics")
            if not isinstance(scope_metrics_list, list):
                continue

            for sm in scope_metrics_list:
                if not isinstance(sm, dict):
                    continue
                metrics = sm.get("metrics")
                if not isinstance(metrics, list):
                    continue

                for metric in metrics:
                    if not isinstance(metric, dict):
                        continue
                    if metric.get("name") != "claude_code.token.usage":
                        continue

                    sum_obj = metric.get("sum")
                    if not isinstance(sum_obj, dict):
                        continue
                    data_points = sum_obj.get("dataPoints")
                    if not isinstance(data_points, list):
                        continue

                    for dp in data_points:
                        if not isinstance(dp, dict):
                            continue
                        attrs = _extract_attrs(dp.get("attributes", []))
                        session_id = attrs.get("session.id")
                        token_type = attrs.get("type")
                        if not session_id or not token_type:
                            continue
                        canonical_field = _TYPE_MAP.get(token_type)
                        if canonical_field is None:
                            continue
                        value = _dp_value(dp)
                        if value is None or value < 0:
                            continue

                        if session_id not in sessions:
                            sessions[session_id] = _empty_usage()
                        sessions[session_id][canonical_field] += value

    return sessions


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    otel: dict[str, dict[str, int]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join OTel session sums against transcript store records.

    Only ``claude-code`` records are joined (OTel emits only Claude Code
    metrics).  For every session present in **both** sources the per-field
    absolute and relative deltas are computed (OTel minus store).

    Parameters
    ----------
    otel:
        Output of :func:`parse_otlp_jsonl`.
    records:
        List of transcript store dicts (each must have ``session_id``,
        ``agent``, and ``usage`` keys to be eligible; others are skipped).

    Returns
    -------
    dict with keys:
        ``matched``       -- int, sessions present in both sources
        ``otel_only``     -- int, sessions in OTel but not the store
        ``store_only``    -- int, claude-code store sessions not in OTel
        ``sessions``      -- list of per-session delta dicts
        ``disagreements`` -- list of session_ids with any field |rel delta| > 1%
    """
    # Index store records by session_id (claude-code only)
    store_index: dict[str, dict[str, Any]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("agent") != "claude-code":
            continue
        sid = rec.get("session_id")
        if not isinstance(sid, str) or not sid:
            continue
        usage = rec.get("usage")
        if not isinstance(usage, dict):
            continue
        store_index[sid] = rec

    store_sessions = set(store_index.keys())
    otel_sessions = set(otel.keys())

    matched_ids = store_sessions & otel_sessions
    otel_only_ids = otel_sessions - store_sessions
    store_only_ids = store_sessions - otel_sessions

    session_deltas: list[dict[str, Any]] = []
    disagreements: list[str] = []

    for sid in sorted(matched_ids):
        otel_usage = otel[sid]
        store_usage = store_index[sid]["usage"]

        fields: dict[str, dict[str, Any]] = {}
        has_disagreement = False

        for field in _CANONICAL_FIELDS:
            otel_val = otel_usage.get(field, 0)
            store_val = store_usage.get(field, 0)
            # Treat None as 0 for fields like reasoning_output absent in OTel
            if store_val is None:
                store_val = 0

            abs_delta = otel_val - store_val
            if store_val != 0:
                rel_delta = abs_delta / store_val
            elif otel_val != 0:
                rel_delta = float("inf")
            else:
                rel_delta = 0.0

            fields[field] = {
                "otel": otel_val,
                "store": store_val,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
            }

            if abs(rel_delta) > _DISAGREEMENT_THRESHOLD:
                has_disagreement = True

        session_deltas.append({"session_id": sid, "fields": fields})
        if has_disagreement:
            disagreements.append(sid)

    return {
        "matched": len(matched_ids),
        "otel_only": len(otel_only_ids),
        "store_only": len(store_only_ids),
        "sessions": session_deltas,
        "disagreements": disagreements,
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------


def report(result: dict[str, Any]) -> str:
    """Return a human-readable summary of a :func:`compare` result.

    The report explicitly states that:
    - OTel is an *optional* cross-validation meter, not the primary source.
    - The transcript store is authoritative.
    """
    matched = result.get("matched", 0)
    otel_only = result.get("otel_only", 0)
    store_only = result.get("store_only", 0)
    disagreements: list[str] = result.get("disagreements", [])
    sessions: list[dict[str, Any]] = result.get("sessions", [])

    lines: list[str] = [
        "OTel Second-Meter Cross-Validation",
        "===================================",
        "",
        "NOTE: The transcript store is the authoritative source of truth.",
        "OTel (claude_code.token.usage counter) is an optional independent",
        "meter used to confirm transcript totals — not to override them.",
        "",
        f"Sessions matched (both sources):  {matched}",
        f"Sessions in OTel only:            {otel_only}",
        f"Sessions in store only:           {store_only}",
        f"Sessions with >1% field delta:    {len(disagreements)}",
        "",
    ]

    lines += [
        "To enable OTel export from Claude Code, set:",
        "  CLAUDE_CODE_ENABLE_TELEMETRY=1",
        "  OTEL_METRICS_EXPORTER=otlp",
        "  OTEL_EXPORTER_OTLP_ENDPOINT=<your-collector-or-file-endpoint>",
        "",
    ]

    if matched == 0:
        lines.append("No sessions matched — nothing to compare.")
        return "\n".join(lines)

    if not disagreements:
        lines.append(
            f"All {matched} matched session(s) agree within 1% on every field."
        )
    else:
        lines.append(
            f"DISAGREEMENTS ({len(disagreements)} session(s) exceed 1% on at least one field):"
        )
        for sid in disagreements:
            lines.append(f"  {sid}")
            # Find the session delta entry
            for sd in sessions:
                if sd["session_id"] == sid:
                    for field, fd in sd["fields"].items():
                        abs_d = fd["abs_delta"]
                        rel_d = fd["rel_delta"]
                        if abs(rel_d) > _DISAGREEMENT_THRESHOLD:
                            pct = (
                                f"{rel_d * 100:+.1f}%"
                                if rel_d != float("inf")
                                else "+inf%"
                            )
                            lines.append(
                                f"    {field}: otel={fd['otel']}  store={fd['store']}"
                                f"  delta={abs_d:+d} ({pct})"
                            )
                    break

    return "\n".join(lines)
