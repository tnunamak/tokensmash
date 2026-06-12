"""Append-only JSONL session store.

Provides:
  append(path, record)          — write one canonical JSON line
  load_latest(path, key)        — read JSONL, dedupe by key (last wins), skip corrupt
  upsert_many(path, records)    — append only new/changed records; return (added, replaced)
  export_scrubbed(src, dst)     — scrub paths, validate, write; refuse forbidden keys

Stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tokensmash.schema as schema


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def append(path: Path, record: dict) -> None:
    """Append one canonical JSON line to *path*.

    The parent directory is created on first write.  The line is formatted via
    ``schema.dumps_record`` (sorted keys, compact separators) so the store is
    byte-stable: the same logical record always produces the same line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = schema.dumps_record(record)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_latest(
    path: Path,
    key: tuple[str, ...] = ("agent", "session_id"),
) -> dict[tuple, dict]:
    """Read a JSONL store and return a mapping of *key* → last-seen record.

    Rules:
    - Lines that are not valid JSON or whose values are not dicts are silently
      skipped (corrupt-line tolerance).
    - When the same key appears more than once, the **last** entry wins
      (idempotent upsert semantics).
    - Returns an empty dict when the file does not exist.
    """
    result: dict[tuple, dict] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            k = tuple(record.get(f) for f in key)
            result[k] = record
    return result


def upsert_many(
    path: Path,
    records: list[dict],
    key: tuple[str, ...] = ("agent", "session_id"),
) -> tuple[int, int]:
    """Append records that are new or have changed relative to the current store.

    Comparison is done on the **canonical JSON dump** so field-order differences
    are normalised out.  Returns ``(added, replaced)`` where:
    - *added*    — records whose key did not exist in the store before
    - *replaced* — records whose key existed but whose canonical dump differs

    The underlying file is an append-only log; "replaced" records are appended
    (the newer entry shadows the older one on next ``load_latest`` call).
    """
    existing = load_latest(path, key=key)
    added = 0
    replaced = 0
    for record in records:
        k = tuple(record.get(f) for f in key)
        new_line = schema.dumps_record(record)
        if k not in existing:
            append(path, record)
            added += 1
        elif schema.dumps_record(existing[k]) != new_line:
            append(path, record)
            replaced += 1
    return added, replaced


# ---------------------------------------------------------------------------
# Scrubbed export
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYS = frozenset({"prompt", "text", "content"})


def _scrub_value(value: Any) -> Any:
    """Recursively drop string values that look like absolute paths."""
    if isinstance(value, str):
        return None if value.startswith("/") else value
    if isinstance(value, dict):
        return _scrub_dict(value)
    if isinstance(value, list):
        return [_scrub_value(v) for v in value]
    return value


def _scrub_dict(record: dict) -> dict:
    """Return a copy of *record* with absolute-path string values removed."""
    out: dict = {}
    for k, v in record.items():
        scrubbed = _scrub_value(v)
        if scrubbed is not None or not isinstance(v, str):
            out[k] = scrubbed
    return out


def _check_forbidden_keys(record: dict, path: str = "") -> None:
    """Raise ValueError if any key in *record* (recursively) is in _FORBIDDEN_KEYS."""
    for k, v in record.items():
        full_key = f"{path}.{k}" if path else k
        if k in _FORBIDDEN_KEYS:
            raise ValueError(
                f"Record contains forbidden key '{full_key}'; "
                "export_scrubbed refuses to write raw text content."
            )
        if isinstance(v, dict):
            _check_forbidden_keys(v, full_key)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _check_forbidden_keys(item, f"{full_key}[{i}]")


def export_scrubbed(sessions_path: Path, out_path: Path) -> int:
    """Write a scrubbed copy of *sessions_path* to *out_path*.

    Per record:
    1. Drop the ``transcript_path`` key.
    2. Drop any string value that is an absolute path (starts with ``/``),
       recursively.  Hashes (hex strings) are kept because they do not start
       with ``/``.
    3. Refuse (raise ``ValueError``) if any record contains a key named
       ``"prompt"``, ``"text"``, or ``"content"`` (guards against accidentally
       exporting raw transcript text).
    4. Validate the scrubbed record via ``schema.validate_session_record``; skip
       invalid records (they indicate a malformed store, not a scrub error).

    Returns the count of records written.
    """
    # One record per transcript: session_id is shared by claude subagents and
    # resumed codex rollouts, so deduping on it would silently drop records.
    existing = load_latest(sessions_path, key=("agent", "transcript_id"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in existing.values():
            # Step 1: drop transcript_path
            record = dict(record)
            record.pop("transcript_path", None)

            # Step 2: scrub absolute path strings recursively
            record = _scrub_dict(record)

            # Step 3: refuse forbidden keys (raise — caller decides how to handle)
            _check_forbidden_keys(record)

            # Step 4: validate
            errors = schema.validate_session_record(record)
            if errors:
                continue  # malformed record; skip silently

            fh.write(schema.dumps_record(record) + "\n")
            written += 1
    return written
