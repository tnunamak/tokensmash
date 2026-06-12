"""Pricing tables and cost computation for tokensmash.

Data files live under src/tokensmash/data/pricing/*.json.  Each file has:
  - id, kind, agent, retrieved_at, source_urls, notes (optional)
  - models: {model_id: {fresh_input_per_m, cache_read_per_m, cache_write_per_m, output_per_m}}
  - match: [{pattern, model}]  — ordered substring-match rules for fuzzy resolution

kind values:
  "api_usd"       — dollar cost per million tokens
  "codex_credits" — Codex credit units per million tokens

All public functions return None for unknown models rather than guessing.
Stdlib only; no runtime dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data" / "pricing"

_REQUIRED_FILE_FIELDS = ("id", "kind", "agent", "retrieved_at", "source_urls", "models", "match")
_REQUIRED_MODEL_FIELDS = ("fresh_input_per_m", "cache_read_per_m", "cache_write_per_m", "output_per_m")
_VALID_KINDS = ("api_usd", "codex_credits")


def _validate_table(table: dict[str, Any], path: Path) -> None:
    """Raise ValueError if the table is structurally invalid."""
    for field in _REQUIRED_FILE_FIELDS:
        if field not in table:
            raise ValueError(f"{path.name}: missing required field '{field}'")
    if table["kind"] not in _VALID_KINDS:
        raise ValueError(f"{path.name}: unknown kind '{table['kind']}'; must be one of {_VALID_KINDS}")
    if not isinstance(table["models"], dict) or not table["models"]:
        raise ValueError(f"{path.name}: 'models' must be a non-empty dict")
    for model_id, rates in table["models"].items():
        for field in _REQUIRED_MODEL_FIELDS:
            if field not in rates:
                raise ValueError(f"{path.name}: model '{model_id}' missing field '{field}'")
            if not isinstance(rates[field], (int, float)):
                raise ValueError(f"{path.name}: model '{model_id}' field '{field}' must be numeric")
    if not isinstance(table["match"], list):
        raise ValueError(f"{path.name}: 'match' must be a list")
    for entry in table["match"]:
        if "pattern" not in entry or "model" not in entry:
            raise ValueError(f"{path.name}: each match entry must have 'pattern' and 'model'")
        if entry["model"] not in table["models"]:
            raise ValueError(
                f"{path.name}: match pattern '{entry['pattern']}' references unknown model '{entry['model']}'"
            )


def load_tables() -> list[dict]:
    """Load and validate all pricing data files from the data/pricing directory.

    Returns a list of validated table dicts.  Raises ValueError on any
    structural problem so bad data files are caught at startup.
    """
    tables: list[dict] = []
    for path in sorted(_DATA_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            table = json.load(fh)
        _validate_table(table, path)
        tables.append(table)
    return tables


def resolve_model(
    tables: list[dict], agent: str, model: str
) -> tuple[dict, str] | None:
    """Return (per-m rates dict, table_id) for the given agent+model, or None.

    Resolution order:
      1. Exact key match in tables whose agent matches.
      2. Substring match via the table's ordered `match` list (first win).

    The `agent` parameter filters tables: "codex" only looks at codex tables,
    "claude-code" only at claude-code tables.  For codex_credits tables the
    agent is always "codex".
    """
    model_lower = model.lower()

    # Pass 1: exact key match
    for table in tables:
        if table["agent"] != agent:
            continue
        if model in table["models"]:
            return dict(table["models"][model]), table["id"]
        # also try lower-case exact
        for mid in table["models"]:
            if mid.lower() == model_lower:
                return dict(table["models"][mid]), table["id"]

    # Pass 2: ordered substring match
    for table in tables:
        if table["agent"] != agent:
            continue
        for entry in table["match"]:
            if entry["pattern"].lower() in model_lower:
                mid = entry["model"]
                return dict(table["models"][mid]), table["id"]

    return None


def cost_usd(usage: dict, agent: str, model: str) -> tuple[float, str] | None:
    """Compute dollar cost for a canonical usage dict.

    usage keys: fresh_input, cache_read, cache_write, output, reasoning_output
    (reasoning_output is ignored for cost; reasoning tokens are billed as output
    at the provider level and should already be counted in output when non-None).

    Only considers tables with kind="api_usd".  Returns None if the model
    cannot be resolved.  cache_write cost is applied only when the table
    has cache_write_per_m > 0 (OpenAI does not charge a separate write cost).
    """
    tables = load_tables()
    usd_tables = [t for t in tables if t["kind"] == "api_usd"]
    result = resolve_model(usd_tables, agent, model)
    if result is None:
        return None
    rates, table_id = result

    fresh = usage.get("fresh_input", 0) or 0
    cache_read = usage.get("cache_read", 0) or 0
    cache_write = usage.get("cache_write", 0) or 0
    output = usage.get("output", 0) or 0

    total = (
        fresh / 1_000_000 * rates["fresh_input_per_m"]
        + cache_read / 1_000_000 * rates["cache_read_per_m"]
        + output / 1_000_000 * rates["output_per_m"]
    )
    if rates["cache_write_per_m"] > 0:
        total += cache_write / 1_000_000 * rates["cache_write_per_m"]

    return total, table_id


def codex_credits(usage: dict, model: str) -> tuple[float, str] | None:
    """Compute Codex credit cost for a canonical usage dict.

    Only considers tables with kind="codex_credits".  Returns None if the model
    cannot be resolved.  Agent is always "codex" for credit tables.
    cache_write cost is applied only when cache_write_per_m > 0 (currently 0
    for all Codex credit tables — caching is automatic, no write credit charged).
    """
    tables = load_tables()
    credit_tables = [t for t in tables if t["kind"] == "codex_credits"]
    result = resolve_model(credit_tables, "codex", model)
    if result is None:
        return None
    rates, table_id = result

    fresh = usage.get("fresh_input", 0) or 0
    cache_read = usage.get("cache_read", 0) or 0
    cache_write = usage.get("cache_write", 0) or 0
    output = usage.get("output", 0) or 0

    total = (
        fresh / 1_000_000 * rates["fresh_input_per_m"]
        + cache_read / 1_000_000 * rates["cache_read_per_m"]
        + output / 1_000_000 * rates["output_per_m"]
    )
    if rates["cache_write_per_m"] > 0:
        total += cache_write / 1_000_000 * rates["cache_write_per_m"]

    return total, table_id
