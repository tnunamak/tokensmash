"""Pre-registered statistical analysis for the tokensmash crossover study.

Implements PROTOCOL.md §5 exactly.  The formulas here are authoritative per
docs/CONTRACTS.md §9.  Stdlib only.

Public API:
    analyze(records, config, protocol_text) -> dict   # full result / audit trail
    report(result) -> str                             # human-readable summary

AnalysisRefused is raised (never returns) when any hard protocol guard fires.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timezone
from typing import Any

from tokensmash.study.assign import arm_for, block_index

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AnalysisRefused(Exception):
    """Raised when any protocol guard prevents the analysis from running.

    The message is a human-readable reason string.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp to an aware datetime."""
    # Python 3.11+ fromisoformat handles Z suffix; for 3.7-3.10 we do it manually.
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _block_for_ts(ts: str) -> int:
    """Return the 2-hour block index for an ISO-8601 timestamp string."""
    dt = _parse_iso(ts)
    return block_index(dt.timestamp())


def _norm_cdf(z: float) -> float:
    """Standard normal CDF via math.erfc.  Accurate to ~14 digits."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _two_sided_normal_p(z: float) -> float:
    return 2.0 * _norm_cdf(-abs(z))


# ---------------------------------------------------------------------------
# Hard guards
# ---------------------------------------------------------------------------


def _check_guards(records: list[dict], config: dict, protocol_text: str) -> None:
    """Raise AnalysisRefused if any hard protocol guard fires."""
    # Guard 1: protocol hash
    expected_hash = config.get("protocol_sha256")
    if expected_hash is None:
        raise AnalysisRefused(
            "config missing 'protocol_sha256'; cannot verify protocol integrity"
        )
    actual_hash = _sha256_hex(protocol_text)
    if actual_hash != expected_hash:
        raise AnalysisRefused(
            f"protocol_text hash mismatch: expected {expected_hash!r}, "
            f"got {actual_hash!r}. Analysis refused to prevent mixing protocol "
            "versions."
        )

    # Guard 2: protocol version on records
    config_version = config.get("protocol_version")
    for rec in records:
        rec_version = rec.get("protocol_version")
        if rec_version is not None and rec_version != config_version:
            raise AnalysisRefused(
                f"Record protocol_version {rec_version!r} differs from "
                f"config protocol_version {config_version!r}. "
                "Analysis refused to prevent mixing protocol epochs."
            )

    # Guard 3: live_started_at must exist and mode must be 'live'
    if not config.get("live_started_at"):
        raise AnalysisRefused(
            "config missing 'live_started_at'; analysis may only run on a live "
            "study epoch."
        )
    if config.get("mode") != "live":
        raise AnalysisRefused(
            f"config mode is {config.get('mode')!r}; analysis requires mode='live'."
        )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def _filter_records(records: list[dict], config: dict) -> tuple[list[dict], dict]:
    """Apply pre-registered inclusion/exclusion rules.

    Returns (kept_records, exclusion_tallies).

    Inclusion criteria (AND):
    - arm in ("on", "off")
    - excluded is falsy
    - cost_api_usd is not None
    - started_at >= live_started_at
    - repo_id not in config.exclude_repo_ids
    """
    live_started_at = _parse_iso(config["live_started_at"])
    exclude_repo_ids = set(config.get("exclude_repo_ids") or [])
    seed = bytes.fromhex(config["seed"])

    tallies: dict[str, int] = {
        "total_in": len(records),
        "excluded_field": 0,
        "no_cost": 0,
        "before_live": 0,
        "excluded_repo": 0,
        "no_arm": 0,
        "kept": 0,
    }

    kept: list[dict] = []
    for rec in records:
        if rec.get("excluded"):
            tallies["excluded_field"] += 1
            continue
        if rec.get("cost_api_usd") is None:
            tallies["no_cost"] += 1
            continue
        started_at = _parse_iso(rec["started_at"])
        if started_at < live_started_at:
            tallies["before_live"] += 1
            continue
        if rec.get("repo_id") in exclude_repo_ids:
            tallies["excluded_repo"] += 1
            continue
        arm_label = rec.get("arm")
        if arm_label not in ("on", "off"):
            tallies["no_arm"] += 1
            continue
        kept.append(rec)

    tallies["kept"] = len(kept)
    return kept, tallies


# ---------------------------------------------------------------------------
# Block-value aggregation and pair construction
# ---------------------------------------------------------------------------


def _build_block_values(
    records: list[dict],
    config: dict,
    cost_field: str = "cost_api_usd",
) -> tuple[dict, int]:
    """Aggregate records into {(repo_id, block): {"arm": str, "cost": float, "n": int}}.

    The arm of each block is RECOMPUTED via assign.arm_for from the config seed.
    Record arm labels are stored separately for mismatch counting.

    Returns (block_values, label_mismatches).
    block_values: {(repo_id, block): {"arm": str, "cost": float, "n": int}}
    """
    seed = bytes.fromhex(config["seed"])
    # Accumulate costs per (repo_id, block)
    buckets: dict[tuple, dict] = {}
    label_mismatches = 0

    for rec in records:
        repo_id = rec["repo_id"]
        blk = _block_for_ts(rec["started_at"])
        key = (repo_id, blk)

        # Recompute arm from PRF
        true_arm = arm_for(seed, repo_id, blk)

        # Compare with record label
        record_arm = rec.get("arm")
        if record_arm != true_arm:
            label_mismatches += 1

        cost = rec.get(cost_field)
        if cost is None:
            continue

        if key not in buckets:
            buckets[key] = {"arm": true_arm, "cost": 0.0, "n": 0}
        buckets[key]["cost"] += cost
        buckets[key]["n"] += 1

    return buckets, label_mismatches


def _build_pairs(
    block_values: dict,
    config: dict,
    cost_field: str = "cost_api_usd",
) -> list[dict]:
    """Construct paired (on, off) differences within each (repo, group).

    Within each (repo_id, group=block//8): collect non-empty on-blocks and
    non-empty off-blocks.  A pair exists only if both arms have >=1 non-empty
    block.

    pair dict keys: repo_id, group, d_i, n_on, n_off, mean_on, mean_off
    """
    # Group by (repo_id, group)
    groups: dict[tuple, dict] = {}
    for (repo_id, blk), bv in block_values.items():
        group = blk // 8
        key = (repo_id, group)
        if key not in groups:
            groups[key] = {"on": [], "off": []}
        groups[key][bv["arm"]].append(bv["cost"])

    pairs: list[dict] = []
    for (repo_id, group), arms in sorted(groups.items()):
        on_costs = arms["on"]
        off_costs = arms["off"]
        if not on_costs or not off_costs:
            continue  # PROTOCOL §5 missing data rule: drop one-armed groups
        mean_on = sum(on_costs) / len(on_costs)
        mean_off = sum(off_costs) / len(off_costs)
        d_i = mean_on - mean_off
        pairs.append(
            {
                "repo_id": repo_id,
                "group": group,
                "d_i": d_i,
                "n_on": len(on_costs),
                "n_off": len(off_costs),
                "mean_on": mean_on,
                "mean_off": mean_off,
            }
        )

    return pairs


# ---------------------------------------------------------------------------
# Primary test: sign-flip permutation
# ---------------------------------------------------------------------------


def _permutation_test(d_values: list[float]) -> dict:
    """Sign-flip permutation test with random.Random(20260612), 10,000 iterations.

    p = (1 + #{|T_perm| >= |T_obs|}) / 10001

    Returns dict with T_obs, p_two_sided, count_extreme, n_iterations.
    """
    n = len(d_values)
    if n == 0:
        return {"T_obs": None, "p_two_sided": None, "count_extreme": None, "n_iterations": 0}

    T_obs = sum(d_values) / n
    rng = random.Random(20260612)
    N = 10_000
    count_extreme = 0
    for _ in range(N):
        T_perm = sum(
            d * (1 if rng.random() >= 0.5 else -1) for d in d_values
        ) / n
        if abs(T_perm) >= abs(T_obs):
            count_extreme += 1

    p_two_sided = (1 + count_extreme) / (N + 1)
    return {
        "T_obs": T_obs,
        "p_two_sided": p_two_sided,
        "count_extreme": count_extreme,
        "n_iterations": N,
    }


# ---------------------------------------------------------------------------
# Secondary: CUPED
# ---------------------------------------------------------------------------


def _compute_cuped(
    pairs: list[dict],
    records_pre: list[dict],
    config: dict,
    cost_field: str = "cost_api_usd",
) -> dict:
    """CUPED covariate adjustment.

    X_i = pre-study mean block cost for the repo of pair i.
    Pre-study records: started_at < live_started_at, all repo_ids (no arm filter),
    cost_field not None, excluded falsy.

    OLS slope b = cov(d, X_centered) / var(X_centered).
    d_adj_i = d_i - b * (X_i - mean_X).

    Returns dict with keys: b, mean_X, X_values, d_adj, mean_d_adj, naive_se_d_adj,
    n_repos_with_pre_data.
    """
    if not pairs:
        return {
            "b": None,
            "mean_X": None,
            "X_values": [],
            "d_adj": [],
            "mean_d_adj": None,
            "naive_se_d_adj": None,
            "n_repos_with_pre_data": 0,
        }

    live_started_at = _parse_iso(config["live_started_at"])

    # Build pre-study block costs per repo
    pre_block_costs: dict[str, list[float]] = {}
    for rec in records_pre:
        if rec.get("excluded"):
            continue
        cost = rec.get(cost_field)
        if cost is None:
            continue
        ts = rec.get("started_at")
        if ts is None:
            continue
        if _parse_iso(ts) >= live_started_at:
            continue
        repo_id = rec["repo_id"]
        if repo_id not in pre_block_costs:
            pre_block_costs[repo_id] = []
        pre_block_costs[repo_id].append(cost)

    # X_i = pre-study mean block cost for pair's repo
    X_values: list[float | None] = []
    repos_with_pre_data: set[str] = set()
    for pair in pairs:
        repo_costs = pre_block_costs.get(pair["repo_id"])
        if repo_costs:
            X_values.append(sum(repo_costs) / len(repo_costs))
            repos_with_pre_data.add(pair["repo_id"])
        else:
            X_values.append(None)
    n_repos_with_pre_data = len(repos_with_pre_data)

    # Only regress on pairs where X is available
    d_vals_for_reg = [pairs[i]["d_i"] for i in range(len(pairs)) if X_values[i] is not None]
    X_vals_for_reg = [X_values[i] for i in range(len(pairs)) if X_values[i] is not None]

    if len(d_vals_for_reg) < 2:
        # Not enough data for regression; fall back to unadjusted
        d_adj = [p["d_i"] for p in pairs]
        mean_d_adj = sum(d_adj) / len(d_adj)
        return {
            "b": None,
            "mean_X": None,
            "X_values": X_values,
            "d_adj": d_adj,
            "mean_d_adj": mean_d_adj,
            "naive_se_d_adj": None,
            "n_repos_with_pre_data": n_repos_with_pre_data,
        }

    mean_X = sum(X_vals_for_reg) / len(X_vals_for_reg)
    mean_d_reg = sum(d_vals_for_reg) / len(d_vals_for_reg)
    X_centered_reg = [xi - mean_X for xi in X_vals_for_reg]
    d_centered_reg = [di - mean_d_reg for di in d_vals_for_reg]

    cov_num = sum(di * xi for di, xi in zip(d_centered_reg, X_centered_reg))
    var_denom = sum(xi ** 2 for xi in X_centered_reg)
    b = cov_num / var_denom if var_denom != 0.0 else 0.0

    # Apply adjustment to ALL pairs (use mean_X from regression pairs)
    d_adj: list[float] = []
    for i, pair in enumerate(pairs):
        xi = X_values[i]
        if xi is not None:
            d_adj.append(pair["d_i"] - b * (xi - mean_X))
        else:
            d_adj.append(pair["d_i"])

    mean_d_adj = sum(d_adj) / len(d_adj)

    # Naive SE of d_adj
    n = len(d_adj)
    if n > 1:
        naive_se_d_adj = math.sqrt(
            sum((di - mean_d_adj) ** 2 for di in d_adj) / (n * (n - 1))
        )
    else:
        naive_se_d_adj = None

    return {
        "b": b,
        "mean_X": mean_X,
        "X_values": X_values,
        "d_adj": d_adj,
        "mean_d_adj": mean_d_adj,
        "naive_se_d_adj": naive_se_d_adj,
        "n_repos_with_pre_data": n_repos_with_pre_data,
    }


# ---------------------------------------------------------------------------
# Robustness: cluster-robust SE (CR0)
# ---------------------------------------------------------------------------


def _cluster_robust_se(pairs: list[dict]) -> dict:
    """CR0 cluster-robust SE for mean(d_i), clustering by repo.

    SE^2 = Σ_repo (Σ_{i in repo} (d_i - mean_d))^2 / n_pairs^2 * G/(G-1)

    where G = number of repos.

    Returns dict with keys: SE, z, p_two_sided, G, n_pairs.
    """
    d_values = [p["d_i"] for p in pairs]
    n_pairs = len(d_values)
    if n_pairs < 2:
        return {"SE": None, "z": None, "p_two_sided": None, "G": 0, "n_pairs": n_pairs}

    mean_d = sum(d_values) / n_pairs

    # Group residuals by repo
    repo_residuals: dict[str, float] = {}
    for p in pairs:
        repo = p["repo_id"]
        residual = p["d_i"] - mean_d
        repo_residuals[repo] = repo_residuals.get(repo, 0.0) + residual

    G = len(repo_residuals)
    if G < 2:
        # Cannot apply G/(G-1) correction with G=1
        return {"SE": None, "z": None, "p_two_sided": None, "G": G, "n_pairs": n_pairs}

    sum_sq = sum(v ** 2 for v in repo_residuals.values())
    SE_sq = sum_sq / (n_pairs ** 2) * (G / (G - 1))
    SE = math.sqrt(SE_sq)
    z = mean_d / SE
    p_two_sided = _two_sided_normal_p(z)

    return {
        "SE": SE,
        "z": z,
        "p_two_sided": p_two_sided,
        "G": G,
        "n_pairs": n_pairs,
    }


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def _compute_guardrails(records: list[dict], config: dict) -> dict:
    """Per-arm means of user_turns, compactions, duration_ms; abandonment rate.

    Only considers in-window, non-excluded records with an arm assignment.
    Guardrails are report-only — no statistical test is run here.
    """
    live_started_at = _parse_iso(config["live_started_at"])
    exclude_repo_ids = set(config.get("exclude_repo_ids") or [])
    seed = bytes.fromhex(config["seed"])

    per_arm: dict[str, dict] = {
        "on": {"user_turns": [], "compactions": [], "duration_ms": [], "abandoned": []},
        "off": {"user_turns": [], "compactions": [], "duration_ms": [], "abandoned": []},
    }

    for rec in records:
        if rec.get("excluded"):
            continue
        started_at = _parse_iso(rec["started_at"])
        if started_at < live_started_at:
            continue
        repo_id = rec.get("repo_id")
        if repo_id in exclude_repo_ids:
            continue
        blk = _block_for_ts(rec["started_at"])
        arm = arm_for(seed, repo_id, blk)
        bucket = per_arm[arm]
        if rec.get("user_turns") is not None:
            bucket["user_turns"].append(rec["user_turns"])
        if rec.get("compactions") is not None:
            bucket["compactions"].append(rec["compactions"])
        if rec.get("duration_ms") is not None:
            bucket["duration_ms"].append(rec["duration_ms"])
        tool_calls = rec.get("tool_calls", 1)
        bucket["abandoned"].append(1 if tool_calls == 0 else 0)

    def _mean(lst: list) -> float | None:
        return sum(lst) / len(lst) if lst else None

    result: dict[str, Any] = {}
    for arm, bucket in per_arm.items():
        result[arm] = {
            "mean_user_turns": _mean(bucket["user_turns"]),
            "mean_compactions": _mean(bucket["compactions"]),
            "mean_duration_ms": _mean(bucket["duration_ms"]),
            "abandonment_rate": _mean(bucket["abandoned"]),
            "n_sessions": len(bucket["abandoned"]),
        }
    return result


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _compute_coverage(
    records: list[dict],
    config: dict,
    actuations: list[dict] | None = None,
) -> dict:
    """Coverage: among arm='on' sessions in-window, fraction with actuation record.

    actuations: list of actuation records from actuations.jsonl.
    Reports on/off session counts, pairs, label_mismatches.
    """
    # Build set of (agent, repo_id, block) for actuation records
    actuated: set[tuple] = set()
    if actuations:
        for act in actuations:
            agent = act.get("agent_command") or act.get("tool")
            repo_id = act.get("repo_id")
            blk = act.get("block")
            if repo_id is not None and blk is not None:
                actuated.add((repo_id, blk))

    live_started_at = _parse_iso(config["live_started_at"])
    exclude_repo_ids = set(config.get("exclude_repo_ids") or [])
    seed = bytes.fromhex(config["seed"])

    on_count = 0
    off_count = 0
    on_actuated = 0

    for rec in records:
        if rec.get("excluded"):
            continue
        if _parse_iso(rec["started_at"]) < live_started_at:
            continue
        repo_id = rec.get("repo_id")
        if repo_id in exclude_repo_ids:
            continue
        blk = _block_for_ts(rec["started_at"])
        arm = arm_for(seed, repo_id, blk)
        if arm == "on":
            on_count += 1
            if (repo_id, blk) in actuated:
                on_actuated += 1
        else:
            off_count += 1

    return {
        "on_sessions": on_count,
        "off_sessions": off_count,
        "on_actuated": on_actuated,
        "actuation_coverage": on_actuated / on_count if on_count > 0 else None,
    }


# ---------------------------------------------------------------------------
# Co-primary and secondary pipelines
# ---------------------------------------------------------------------------


def _run_pipeline(
    records: list[dict],
    config: dict,
    all_records: list[dict],
    cost_field: str,
    filter_agent: str | None = None,
) -> dict:
    """Run the full analysis pipeline for one outcome measure.

    cost_field: "cost_api_usd", "cost_codex_credits", or "usage.fresh_input"
    filter_agent: when set (e.g. "codex"), only include records from that agent.
    """

    def _get_cost(rec: dict, field: str) -> float | None:
        if "." in field:
            parts = field.split(".", 1)
            sub = rec.get(parts[0])
            if isinstance(sub, dict):
                v = sub.get(parts[1])
                return float(v) if v is not None else None
            return None
        v = rec.get(field)
        return float(v) if v is not None else None

    # Patch records with extracted cost field for this pipeline
    def _with_cost(rec: dict) -> dict:
        if "." not in cost_field:
            return rec
        c = _get_cost(rec, cost_field)
        return {**rec, cost_field: c}

    # Filter by agent if requested
    working_records = [
        _with_cost(r) for r in records
        if filter_agent is None or r.get("agent") == filter_agent
    ]
    working_all = [
        _with_cost(r) for r in all_records
        if filter_agent is None or r.get("agent") == filter_agent
    ]

    # Build the plain cost_field for downstream
    field = cost_field if "." not in cost_field else "__cost__"
    if "." in cost_field:
        working_records = [{**r, "__cost__": _get_cost(r, cost_field)} for r in records
                           if filter_agent is None or r.get("agent") == filter_agent]
        working_all = [{**r, "__cost__": _get_cost(r, cost_field)} for r in all_records
                       if filter_agent is None or r.get("agent") == filter_agent]
        field = "__cost__"

    block_values, label_mismatches = _build_block_values(working_records, config, cost_field=field)
    pairs = _build_pairs(block_values, config, cost_field=field)
    d_values = [p["d_i"] for p in pairs]

    perm_result = _permutation_test(d_values)
    cuped_result = _compute_cuped(pairs, working_all, config, cost_field=field)
    cr0_result = _cluster_robust_se(pairs)

    return {
        "pairs": pairs,
        "label_mismatches": label_mismatches,
        "n_pairs": len(pairs),
        "permutation": perm_result,
        "cuped": cuped_result,
        "cr0": cr0_result,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze(
    records: list[dict],
    config: dict,
    protocol_text: str,
    actuations: list[dict] | None = None,
) -> dict:
    """Run the pre-registered analysis.

    Parameters
    ----------
    records:
        Session records from sessions.jsonl, keyed by (agent, transcript_id).
        All records should be passed — pre-study records are used for CUPED X;
        in-window records are used for Y.
    config:
        Study config dict (from assign.load_study_config()).  Must contain:
        protocol_sha256, protocol_version, live_started_at, mode, seed,
        exclude_repo_ids.
    protocol_text:
        Raw text of PROTOCOL.md.  Its SHA-256 is verified against
        config["protocol_sha256"].
    actuations:
        Optional list of actuation records from actuations.jsonl, used for
        coverage computation.

    Returns
    -------
    Full result dict (the audit trail).  All intermediates are included.
    Raises AnalysisRefused if any hard guard fires.
    """
    _check_guards(records, config, protocol_text)

    # Filter to in-window records for Y
    kept_records, exclusion_tallies = _filter_records(records, config)

    if len(kept_records) == 0:
        return {
            "status": "insufficient_data",
            "reason": "no eligible records after applying pre-registered exclusions",
            "exclusion_tallies": exclusion_tallies,
            "primary": None,
            "coprimary_codex_credits": None,
            "secondary_fresh_input": None,
            "guardrails": None,
            "coverage": None,
        }

    # --- Primary: cost_api_usd ---
    primary = _run_pipeline(kept_records, config, records, cost_field="cost_api_usd")

    # --- Co-primary: cost_codex_credits (codex only) ---
    coprimary = _run_pipeline(
        kept_records, config, records,
        cost_field="cost_codex_credits",
        filter_agent="codex",
    )

    # --- Secondary: fresh_input tokens ---
    # usage.fresh_input — need special handling
    def _fresh_input(rec: dict) -> float | None:
        usage = rec.get("usage")
        if isinstance(usage, dict):
            v = usage.get("fresh_input")
            return float(v) if v is not None else None
        return None

    def _with_fresh(recs: list[dict]) -> list[dict]:
        return [{**r, "__fresh_input__": _fresh_input(r)} for r in recs]

    kept_with_fresh = _with_fresh(kept_records)
    all_with_fresh = _with_fresh(records)

    # Temporarily patch cost_api_usd filter to use __fresh_input__ field
    # We reuse the pipeline but with cost_api_usd → __fresh_input__ via patched records
    # Since filter_records filters on cost_api_usd, we need to re-filter for fresh_input
    # Re-filter but allow records where fresh_input != None
    fi_records = []
    live_started_at = _parse_iso(config["live_started_at"])
    exclude_repo_ids = set(config.get("exclude_repo_ids") or [])
    for rec in records:
        if rec.get("excluded"):
            continue
        if _fresh_input(rec) is None:
            continue
        if _parse_iso(rec["started_at"]) < live_started_at:
            continue
        if rec.get("repo_id") in exclude_repo_ids:
            continue
        if rec.get("arm") not in ("on", "off"):
            continue
        fi_records.append({**rec, "cost_api_usd": _fresh_input(rec)})

    all_fi = [{**r, "cost_api_usd": _fresh_input(r)} for r in records]

    secondary_fresh = _run_pipeline(fi_records, config, all_fi, cost_field="cost_api_usd")

    # --- Guardrails ---
    guardrails = _compute_guardrails(records, config)

    # --- Coverage ---
    coverage = _compute_coverage(records, config, actuations=actuations)

    # Inject label_mismatches from primary into coverage for audit
    coverage["label_mismatches"] = primary["label_mismatches"]

    return {
        "status": "ok",
        "config_study_id": config.get("study_id"),
        "config_protocol_version": config.get("protocol_version"),
        "exclusion_tallies": exclusion_tallies,
        "primary": primary,
        "coprimary_codex_credits": coprimary,
        "secondary_fresh_input": secondary_fresh,
        "guardrails": guardrails,
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(result: dict) -> str:
    """Render a human-readable summary of an analyze() result.

    Ends with the pre-registered interpretation rules:
    - Primary p vs alpha 0.05 two-sided
    - Guardrail degradation note per PROTOCOL §3
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("TOKENSMASH CROSSOVER STUDY — ANALYSIS RESULTS")
    lines.append("=" * 70)

    if result.get("status") == "insufficient_data":
        lines.append("")
        lines.append(f"Status: INSUFFICIENT DATA — {result.get('reason', '')}")
        tallies = result.get("exclusion_tallies") or {}
        if tallies:
            lines.append("")
            lines.append("Exclusion tallies:")
            for k, v in tallies.items():
                lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(_interpretation_rules())
        return "\n".join(lines)

    lines.append(f"Study ID     : {result.get('config_study_id', '(unknown)')}")
    lines.append(f"Protocol ver : {result.get('config_protocol_version', '(unknown)')}")
    lines.append("")

    # Exclusion tallies
    tallies = result.get("exclusion_tallies") or {}
    if tallies:
        lines.append("Data preparation (pre-registered exclusions):")
        for k, v in tallies.items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    def _format_pipeline(name: str, pipeline: dict | None) -> list[str]:
        out: list[str] = []
        out.append(f"--- {name} ---")
        if pipeline is None:
            out.append("  (no data)")
            return out
        pairs = pipeline.get("pairs") or []
        n_pairs = pipeline.get("n_pairs", 0)
        lm = pipeline.get("label_mismatches", 0)
        out.append(f"  N pairs        : {n_pairs}")
        out.append(f"  Label mismatches: {lm}")
        # Pairs detail
        for p in pairs:
            out.append(
                f"  pair repo={p['repo_id']} group={p['group']:3d} "
                f"d_i={p['d_i']:+.6f}  on={p['mean_on']:.6f}(n={p['n_on']})  "
                f"off={p['mean_off']:.6f}(n={p['n_off']})"
            )
        perm = pipeline.get("permutation") or {}
        T_obs = perm.get("T_obs")
        p_val = perm.get("p_two_sided")
        n_iters = perm.get("n_iterations", 10000)
        ce = perm.get("count_extreme")
        if T_obs is not None:
            out.append(f"  T_obs (mean d) : {T_obs:+.8f}")
            out.append(
                f"  Permutation p  : {p_val:.6f}  "
                f"(count_extreme={ce}/{n_iters}, formula=(1+{ce})/{n_iters+1})"
            )
        else:
            out.append("  T_obs          : (no pairs)")
        cuped = pipeline.get("cuped") or {}
        b = cuped.get("b")
        if b is not None:
            out.append(f"  CUPED b        : {b:.6f}")
            out.append(f"  mean(d_adj)    : {cuped.get('mean_d_adj'):+.8f}")
            se_adj = cuped.get("naive_se_d_adj")
            if se_adj is not None:
                out.append(f"  naive SE(d_adj): {se_adj:.8f}")
        cr0 = pipeline.get("cr0") or {}
        SE = cr0.get("SE")
        if SE is not None:
            out.append(
                f"  CR0 SE         : {SE:.8f}  "
                f"z={cr0.get('z'):.4f}  "
                f"p={cr0.get('p_two_sided'):.6f}  "
                f"G={cr0.get('G')}"
            )
        return out

    prim = result.get("primary")
    for line in _format_pipeline("PRIMARY — cost_api_usd (all agents)", prim):
        lines.append(line)
    lines.append("")

    coprim = result.get("coprimary_codex_credits")
    for line in _format_pipeline("CO-PRIMARY — cost_codex_credits (Codex only)", coprim):
        lines.append(line)
    lines.append("")

    sec = result.get("secondary_fresh_input")
    for line in _format_pipeline("SECONDARY — fresh_input tokens", sec):
        lines.append(line)
    lines.append("")

    # Coverage
    cov = result.get("coverage") or {}
    lines.append("--- Coverage ---")
    lines.append(f"  On sessions    : {cov.get('on_sessions', 0)}")
    lines.append(f"  Off sessions   : {cov.get('off_sessions', 0)}")
    lines.append(f"  On actuated    : {cov.get('on_actuated', 0)}")
    cov_rate = cov.get("actuation_coverage")
    lines.append(
        f"  Actuation cov  : "
        + (f"{cov_rate:.1%}" if cov_rate is not None else "(no on-sessions)")
    )
    lines.append(f"  Label mismatches: {cov.get('label_mismatches', 0)}")
    lines.append("")

    # Guardrails
    grails = result.get("guardrails") or {}
    lines.append("--- Guardrails (monitored, not tested) ---")
    for arm in ("on", "off"):
        ag = grails.get(arm) or {}
        n = ag.get("n_sessions", 0)
        ut = ag.get("mean_user_turns")
        cp = ag.get("mean_compactions")
        dur = ag.get("mean_duration_ms")
        ab = ag.get("abandonment_rate")
        ut_s = f"{ut:.2f}" if ut is not None else "—"
        cp_s = f"{cp:.2f}" if cp is not None else "—"
        dur_s = f"{dur:.0f}" if dur is not None else "—"
        ab_s = f"{ab:.1%}" if ab is not None else "—"
        lines.append(
            f"  arm={arm}  n={n}  "
            f"user_turns={ut_s}  "
            f"compactions={cp_s}  "
            f"duration_ms={dur_s}  "
            f"abandoned={ab_s}"
        )
    lines.append("")
    lines.append(_interpretation_rules())
    return "\n".join(lines)


def _interpretation_rules() -> str:
    return (
        "--- Pre-registered interpretation rules (PROTOCOL §3, §5) ---\n"
        "Primary test: paired-block sign-flip permutation, 10,000 iterations,\n"
        "  seed random.Random(20260612). Alpha = 0.05, two-sided.\n"
        "  Reject H0 (no effect) when p_two_sided < 0.05.\n"
        "  Do not use CR0 SE p-value for the primary decision.\n"
        "  CUPED-adjusted mean(d_adj) is a secondary estimate only.\n"
        "Guardrail note (PROTOCOL §3): A statistically significant increase in\n"
        "  any guardrail outcome (user_turns, compactions, duration_ms,\n"
        "  abandonment) in the on-arm requires a deviation log entry and pauses\n"
        "  enrollment for that tool pending review."
    )
