"""Power analysis for the tokensmash crossover study.

Estimates minimum detectable effect (MDE) for the two-sample comparison of
per-block API cost, given observed session data.

Formula (from PROTOCOL.md §7)
------------------------------
The MDE (absolute USD per block) for a two-sample comparison at equal allocation:

    delta = (z_{alpha/2} + z_beta) * sqrt(2 * sigma^2 / n)

where
    n     = blocks per arm = (observed_blocks_per_week * weeks) / 2
    sigma = standard deviation of per-block USD cost

This is the algebraic inversion of the sample-size formula:

    n = 2 * sigma^2 * (z_{alpha/2} + z_beta)^2 / delta^2

Standard cases handled by lookup table:
    alpha=0.05  -> z_{alpha/2} = 1.96
    power=0.80  -> z_beta      = 0.8416

For other values the inverse-normal is approximated using Beasley-Springer-Moro
(BSM) rational approximation (Moro 1995, "The Full Monte", Risk Magazine).

Effective-pairs caveat
-----------------------
The primary inferential test (PROTOCOL.md §5) is a paired-block permutation
test: within each group of 8 consecutive blocks, 4 are "on" and 4 are "off",
forming 4 on/off pairs.  A pair is **usable** only if BOTH blocks have at least
one session.  Empty blocks are excluded from the test (per the missing-data
rules).

If block occupancy is p (fraction of elapsed blocks that have >= 1 session),
the probability that both blocks in a pair are non-empty is approximately p^2
(assuming Bernoulli independence -- an approximation; real occupancy is
correlated within a work session but the approximation gives a conservative
bound).

Expected effective pairs per group of 8:
    E[effective_pairs] = 4 * p^2

The raw MDE assumes every block in both arms is occupied.  The effective-pairs
inflation factor is:

    mde_inflation = sqrt(raw_blocks_per_arm / effective_blocks_per_arm)
                  = sqrt(1 / p^2)
                  = 1 / p        [for the simple case; see below]

More precisely, we compute expected effective blocks per arm over a full study:
    effective_blocks = n_blocks_at_weeks * p
and inflate the MDE accordingly.

See `mde()` for full details.  The occupancy/independence assumption is
documented as a known limitation (D13).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from tokensmash.study.assign import block_index as _block_index

# ---------------------------------------------------------------------------
# Norm quantile helpers
# ---------------------------------------------------------------------------

_Z_ALPHA2_LOOKUP: dict[float, float] = {0.05: 1.96, 0.01: 2.5758, 0.10: 1.6449}
_Z_BETA_LOOKUP: dict[float, float] = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449}


def _norm_quantile(p: float) -> float:
    """Approximate inverse-normal CDF for p in (0,1).

    Uses the Beasley-Springer-Moro (BSM) rational approximation as described
    in Moro 1995, "The Full Monte", Risk Magazine.  Accurate to ~4 significant
    figures over the central range; tails are less precise.
    """
    if p <= 0 or p >= 1:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients from Moro 1995 central region
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511349,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ]

    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        num = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
        den = ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0)
        return num / den
    else:
        r = p if y < 0 else 1.0 - p
        r = math.log(-math.log(r))
        result = c[0] + r * (
            c[1]
            + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8]))))))
        )
        return result if y > 0 else -result


def _z_alpha2(alpha: float) -> float:
    """Two-sided z for significance level alpha."""
    if alpha in _Z_ALPHA2_LOOKUP:
        return _Z_ALPHA2_LOOKUP[alpha]
    return _norm_quantile(1.0 - alpha / 2.0)


def _z_beta(power: float) -> float:
    """z for power (one-sided)."""
    if power in _Z_BETA_LOOKUP:
        return _Z_BETA_LOOKUP[power]
    return _norm_quantile(power)


# ---------------------------------------------------------------------------
# ISO-8601 -> unix seconds
# ---------------------------------------------------------------------------


def _iso_to_unix(iso: str) -> float:
    """Parse an ISO-8601 timestamp to Unix seconds (UTC)."""
    # Python 3.11+ fromisoformat handles Z suffix; 3.10 does not.
    iso_clean = iso.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso_clean)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def block_costs(records: list[dict[str, Any]]) -> list[float]:
    """Aggregate session records into per-(repo_id, block) USD cost totals.

    Groups by (repo_id, block_index(started_at)) where started_at is
    ISO-8601.  Skips records where:
      - cost_api_usd is None
      - the record has a truthy "excluded" field

    Returns a list of per-block USD totals (one float per non-empty group).
    Order is not specified.
    """
    groups: dict[tuple[str, int], float] = {}
    for rec in records:
        if rec.get("excluded"):
            continue
        cost = rec.get("cost_api_usd")
        if cost is None:
            continue
        repo_id = rec.get("repo_id", "")
        started_at = rec.get("started_at", "")
        unix_s = _iso_to_unix(started_at)
        key = (repo_id, _block_index(unix_s))
        groups[key] = groups.get(key, 0.0) + float(cost)
    return list(groups.values())


def mde(
    records: list[dict[str, Any]],
    weeks: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[str, Any]:
    """Compute minimum detectable effect (MDE) for a given study duration.

    Parameters
    ----------
    records : list[dict]
        Normalized session records (same schema as CONTRACTS.md §D3).
    weeks : float
        Planned study duration.
    alpha : float
        Two-sided significance level (default 0.05).
    power : float
        Desired power (default 0.8).

    Returns
    -------
    dict with keys:
        n_sessions              total sessions after exclusions
        n_blocks                non-empty blocks (after exclusions/None-cost filter)
        observed_weeks          span from min to max started_at in weeks
        blocks_per_week         observed rate = n_blocks / observed_weeks
        mean_block_cost_usd     mean per-block cost
        sd_block_cost_usd       sample std dev of per-block cost
        cv                      coefficient of variation = sd / mean
        n_blocks_at_weeks       projected blocks per arm = (rate * weeks) / 2
        mde_usd_per_block       MDE in absolute USD per block (raw, all blocks occupied)
        mde_pct                 MDE as % of mean block cost

        occupancy_rate          occupied_blocks / elapsed_blocks (Bernoulli approx)
        effective_pairs_at_weeks expected number of usable on/off pairs
        mde_usd_per_block_effective  MDE inflated by sqrt(raw/effective) occupancy factor

    Formula
    -------
    delta = (z_{alpha/2} + z_beta) * sqrt(2 * sigma^2 / n)

    where n = (blocks_per_week * weeks) / 2  (blocks per arm)

    The occupancy-adjusted MDE inflates by 1/occupancy_rate (derived from
    Bernoulli per-block independence assumption -- see module docstring).

    Raises ValueError if there are fewer than 2 non-empty blocks.
    """
    # Collect filtered records
    filtered = [r for r in records if not r.get("excluded") and r.get("cost_api_usd") is not None]

    n_sessions = len(filtered)

    if n_sessions == 0:
        raise ValueError("No usable records (all excluded or all cost_api_usd=None).")

    # Build per-block costs AND collect started_at values
    groups: dict[tuple[str, int], float] = {}
    unix_times: list[float] = []
    all_keys: set[tuple[str, int]] = set()

    for rec in filtered:
        repo_id = rec.get("repo_id", "")
        started_at = rec.get("started_at", "")
        unix_s = _iso_to_unix(started_at)
        unix_times.append(unix_s)
        key = (repo_id, _block_index(unix_s))
        groups[key] = groups.get(key, 0.0) + float(rec["cost_api_usd"])
        all_keys.add(key)

    costs = list(groups.values())
    n_blocks = len(costs)

    if n_blocks < 2:
        raise ValueError(
            f"Need at least 2 non-empty blocks to estimate variance; got {n_blocks}."
        )

    # Observed time span
    min_unix = min(unix_times)
    max_unix = max(unix_times)
    observed_seconds = max_unix - min_unix
    SECONDS_PER_WEEK = 7 * 24 * 3600.0
    observed_weeks = observed_seconds / SECONDS_PER_WEEK
    if observed_weeks <= 0:
        # All records in same block; use 1/7th of a week as floor
        observed_weeks = 1.0 / 7.0

    blocks_per_week = n_blocks / observed_weeks

    # Descriptive stats
    mean_bc = sum(costs) / n_blocks
    variance = sum((c - mean_bc) ** 2 for c in costs) / (n_blocks - 1)
    sd_bc = math.sqrt(variance)
    cv = sd_bc / mean_bc if mean_bc > 0 else float("nan")

    # Projected blocks per arm
    n_blocks_at_weeks = (blocks_per_week * weeks) / 2.0

    # Normal quantiles
    za2 = _z_alpha2(alpha)
    zb = _z_beta(power)

    # MDE formula: delta = (z_{a/2} + z_b) * sqrt(2*sigma^2 / n)
    mde_usd = (za2 + zb) * math.sqrt(2.0 * variance / n_blocks_at_weeks)
    mde_pct = (mde_usd / mean_bc * 100.0) if mean_bc > 0 else float("nan")

    # ------------------------------------------------------------------
    # Occupancy / effective-pairs calculation
    # ------------------------------------------------------------------
    # Elapsed blocks: from first to last occupied block across all repos.
    # We use the total elapsed block span across the entire dataset.
    min_block = min(_block_index(t) for t in unix_times)
    max_block = max(_block_index(t) for t in unix_times)
    elapsed_blocks = max_block - min_block + 1  # inclusive

    occupancy_rate = min(1.0, n_blocks / elapsed_blocks)

    # Effective pairs per group of 8: E[pairs] = 4 * p^2
    # (Bernoulli independence; p = occupancy_rate per block)
    # Over the study, projected total blocks = blocks_per_week * weeks
    # Total groups of 8 blocks = total_blocks / 8
    # Each group produces 4 pairs; expected effective pairs = 4 * p^2 per group
    projected_total_blocks = blocks_per_week * weeks
    projected_groups = projected_total_blocks / 8.0
    effective_pairs_at_weeks = projected_groups * 4.0 * (occupancy_rate**2)

    # Inflation factor: MDE scales as sqrt(1/n_effective_per_arm).
    # Raw n per arm = blocks_per_week * weeks / 2
    # Effective n per arm = effective_pairs_at_weeks (each pair gives one observation)
    # Inflation = sqrt(raw_n / effective_n)
    if effective_pairs_at_weeks > 0:
        mde_inflation = math.sqrt(n_blocks_at_weeks / effective_pairs_at_weeks)
    else:
        mde_inflation = float("inf")

    mde_usd_effective = mde_usd * mde_inflation

    return {
        "n_sessions": n_sessions,
        "n_blocks": n_blocks,
        "observed_weeks": observed_weeks,
        "blocks_per_week": blocks_per_week,
        "mean_block_cost_usd": mean_bc,
        "sd_block_cost_usd": sd_bc,
        "cv": cv,
        "n_blocks_at_weeks": n_blocks_at_weeks,
        "mde_usd_per_block": mde_usd,
        "mde_pct": mde_pct,
        "occupancy_rate": occupancy_rate,
        "effective_pairs_at_weeks": effective_pairs_at_weeks,
        "mde_usd_per_block_effective": mde_usd_effective,
    }


def report(records: list[dict[str, Any]]) -> str:
    """Return a human-readable power report for week horizons 4, 8, 12.

    Prints the mde() dict for each horizon, plus a one-line interpretation
    for the 8-week result.
    """
    lines: list[str] = []
    lines.append("=" * 68)
    lines.append("Tokensmash Study Power Report")
    lines.append("=" * 68)

    result_8w = None
    for weeks in (4, 8, 12):
        lines.append(f"\n--- {weeks}-week horizon ---")
        try:
            result = mde(records, weeks)
        except ValueError as exc:
            lines.append(f"  (error: {exc})")
            continue

        if weeks == 8:
            result_8w = result

        lines.append(f"  Sessions (usable):         {result['n_sessions']}")
        lines.append(f"  Non-empty blocks observed: {result['n_blocks']}")
        lines.append(f"  Observed span (weeks):     {result['observed_weeks']:.2f}")
        lines.append(f"  Blocks/week rate:          {result['blocks_per_week']:.2f}")
        lines.append(f"  Mean block cost (USD):     ${result['mean_block_cost_usd']:.6f}")
        lines.append(f"  SD block cost (USD):       ${result['sd_block_cost_usd']:.6f}")
        lines.append(f"  CV:                        {result['cv']:.3f}")
        lines.append(f"  Blocks/arm at {weeks:2d}w:          {result['n_blocks_at_weeks']:.1f}")
        lines.append(f"  Raw MDE (USD/block):       ${result['mde_usd_per_block']:.6f}")
        lines.append(f"  Raw MDE (%):               {result['mde_pct']:.2f}%")
        lines.append(f"  Occupancy rate (p):        {result['occupancy_rate']:.3f}")
        lines.append(
            f"  Effective pairs at {weeks:2d}w:   {result['effective_pairs_at_weeks']:.1f}"
            "  [assumes Bernoulli per-block, independent occupancy]"
        )
        lines.append(
            f"  Effective MDE (USD/block): ${result['mde_usd_per_block_effective']:.6f}"
        )

    lines.append("")
    lines.append("-" * 68)
    if result_8w is not None:
        mde8 = result_8w["mde_usd_per_block_effective"]
        mean8 = result_8w["mean_block_cost_usd"]
        pct8 = (mde8 / mean8 * 100.0) if mean8 > 0 else float("nan")
        lines.append(
            f"Interpretation (8w): a tool must plausibly move "
            f">= ${mde8:.4f}/block ({pct8:.1f}%) to be detectable in 8 weeks."
        )
    else:
        lines.append("Interpretation: insufficient data for 8-week estimate.")
    lines.append("=" * 68)
    return "\n".join(lines)
