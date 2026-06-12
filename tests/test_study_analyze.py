"""Tests for tokensmash.study.analyze.

Coverage:
- Hand-computed tiny example: 4 pairs (2 repos × 2 groups), exact T_obs, p bounds,
  CR0 SE, and CUPED values verified in comments.
- Permutation determinism: same seed → same p-value.
- Each guard refusal (bad hash, bad version, missing live_started_at, bad mode).
- One-armed groups dropped (pair construction edge case).
- Pre-study records excluded from Y but used for CUPED X.
- Empty-data path returns a well-formed "insufficient data" result (no raise).

Hand-computed example
---------------------
Study seed: b"\\xaa" * 32 (64-char hex "aa"*32)
Protocol hash: sha256("FAKE_PROTOCOL_TEXT") = determined at test-collection time.

Repos: "repoA", "repoB"

Arm assignments (arm_for with seed=b"\\xaa"*32):
  repoA group 0 (blocks 0-7):  [on, off, on, off, off, off, on, on]
  repoA group 1 (blocks 8-15): [off, on, off, on, on, on, off, off]
  repoB group 0 (blocks 0-7):  [off, on, off, off, on, off, on, on]
  repoB group 1 (blocks 8-15): [on, on, off, off, on, off, off, on]

Sessions (started_at expressed as block × 7200 seconds past the Unix epoch,
formatted as ISO-8601 UTC):
  S1: repoA block 0 (on)  cost=0.10   → group 0
  S2: repoA block 1 (off) cost=0.04   → group 0
  S3: repoA block 9 (on)  cost=0.05   → group 1
  S4: repoA block 10 (off) cost=0.03  → group 1
  S5: repoB block 1 (on)  cost=0.20   → group 0
  S6: repoB block 0 (off) cost=0.12   → group 0
  S7: repoB block 8 (on)  cost=0.15   → group 1
  S8: repoB block 10 (off) cost=0.11  → group 1

Pairs (d_i = mean_on_i − mean_off_i):
  (repoA, group=0): on=[0.10] off=[0.04] → d_A0 = 0.10 − 0.04 = 0.06
  (repoA, group=1): on=[0.05] off=[0.03] → d_A1 = 0.05 − 0.03 = 0.02
  (repoB, group=0): on=[0.20] off=[0.12] → d_B0 = 0.20 − 0.12 = 0.08
  (repoB, group=1): on=[0.15] off=[0.11] → d_B1 = 0.15 − 0.11 = 0.04

T_obs = mean(d) = (0.06 + 0.02 + 0.08 + 0.04) / 4 = 0.20 / 4 = 0.05

Permutation test (random.Random(20260612), N=10000):
  count_extreme = 1225 (|T_perm| ≥ |0.05| in 1225 of 10000 permutations)
  p_two_sided = (1 + 1225) / 10001 = 1226 / 10001 ≈ 0.12259

CR0 SE (G=2 repos, n_pairs=4):
  Cluster A residuals: (d_A0 − mean_d) + (d_A1 − mean_d)
                     = (0.06−0.05) + (0.02−0.05) = 0.01 + (−0.03) = −0.02
  Cluster B residuals: (d_B0 − mean_d) + (d_B1 − mean_d)
                     = (0.08−0.05) + (0.04−0.05) = 0.03 + (−0.01) = +0.02
  SE^2 = [(−0.02)^2 + (0.02)^2] / (4^2) × (2/1)
       = [0.0004 + 0.0004] / 16 × 2
       = 0.0008 / 16 × 2
       = 0.0001
  SE = 0.01
  z = 0.05 / 0.01 = 5.0

CUPED:
  Pre-study records: one session per repo before live_started_at.
    repoA pre: cost = 0.08  → X_A = 0.08
    repoB pre: cost = 0.16  → X_B = 0.16
  X per pair (same for all pairs of a repo):
    [0.08, 0.08, 0.16, 0.16]  (pairs ordered: A/g0, A/g1, B/g0, B/g1)
  mean_X = (0.08+0.08+0.16+0.16)/4 = 0.48/4 = 0.12
  X_centered = [−0.04, −0.04, 0.04, 0.04]
  d_centered  = [0.01, −0.03, 0.03, −0.01]
  cov_num = (0.01)(−0.04)+(−0.03)(−0.04)+(0.03)(0.04)+(−0.01)(0.04)
          = −0.0004 + 0.0012 + 0.0012 − 0.0004 = 0.0016
  var_denom = (−0.04)^2+(−0.04)^2+(0.04)^2+(0.04)^2 = 4 × 0.0016 = 0.0064
  b = 0.0016 / 0.0064 = 0.25
  d_adj_A0 = 0.06 − 0.25×(0.08−0.12) = 0.06 + 0.01 = 0.07
  d_adj_A1 = 0.02 − 0.25×(0.08−0.12) = 0.02 + 0.01 = 0.03
  d_adj_B0 = 0.08 − 0.25×(0.16−0.12) = 0.08 − 0.01 = 0.07
  d_adj_B1 = 0.04 − 0.25×(0.16−0.12) = 0.04 − 0.01 = 0.03
  mean_d_adj = (0.07+0.03+0.07+0.03)/4 = 0.20/4 = 0.05
  naive_se_d_adj = sqrt(Σ(d_adj_i − mean_d_adj)^2 / (n*(n-1)))
    residuals: [0.02, −0.02, 0.02, −0.02]
    sum_sq = 4 × 0.0004 = 0.0016
    naive_se_d_adj = sqrt(0.0016 / (4 × 3)) = sqrt(0.0016/12)
                   = sqrt(1/7500) ≈ 0.011547
"""

from __future__ import annotations

import hashlib
import math
import unittest
from datetime import datetime, timezone

from tokensmash.study.analyze import (
    AnalysisRefused,
    analyze,
    report,
    _permutation_test,
    _cluster_robust_se,
    _compute_cuped,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

_SEED_HEX = "aa" * 32          # 64 hex chars → 32 bytes
_PROTOCOL_TEXT = "FAKE_PROTOCOL_TEXT"
_PROTOCOL_HASH = hashlib.sha256(_PROTOCOL_TEXT.encode()).hexdigest()
_BLOCK_SECS = 7200              # 2 hours in seconds
_LIVE_UNIX = 0                  # live_started_at = epoch (all sessions are in-window)


def _iso(unix_secs: float) -> str:
    return datetime.fromtimestamp(unix_secs, tz=timezone.utc).isoformat()


def _make_config(
    mode: str = "live",
    live_started_at: str | None = None,
    protocol_sha256: str | None = None,
    protocol_version: str = "0.1.0-draft",
    exclude_repo_ids: list | None = None,
) -> dict:
    return {
        "schema": "tokensmash-study-config/1",
        "study_id": "test-study-1",
        "seed": _SEED_HEX,
        "mode": mode,
        "protocol_version": protocol_version,
        "live_started_at": live_started_at if live_started_at is not None else _iso(_LIVE_UNIX),
        "protocol_sha256": protocol_sha256 if protocol_sha256 is not None else _PROTOCOL_HASH,
        "exclude_repo_ids": exclude_repo_ids or [],
        "created_at": _iso(0),
    }


def _make_session(
    repo_id: str,
    block: int,
    cost: float,
    agent: str = "claude-code",
    arm: str | None = None,
    excluded: str | None = None,
    pre_study: bool = False,
    user_turns: int = 2,
    compactions: int = 0,
    duration_ms: int = 5000,
    tool_calls: int = 3,
    protocol_version: str | None = None,
) -> dict:
    """Build a minimal synthetic session record.

    If pre_study=True, starts before live_started_at (at unix=-7200).
    Otherwise, starts at block * _BLOCK_SECS (>= live_started_at=0).
    """
    unix = -_BLOCK_SECS if pre_study else block * _BLOCK_SECS
    rec: dict = {
        "schema": "tokensmash-session/1",
        "agent": agent,
        "session_id": f"sess-{repo_id}-{block}",
        "transcript_id": f"tid-{repo_id}-{block}-{cost}",
        "machine_id": "machine-test",
        "started_at": _iso(unix),
        "model": "claude-sonnet-4",
        "repo_id": repo_id,
        "user_turns": user_turns,
        "tool_calls": tool_calls,
        "compactions": compactions,
        "duration_ms": duration_ms,
        "usage": {
            "fresh_input": int(cost * 1000),
            "cache_read": 0,
            "cache_write": 0,
            "output": 100,
            "reasoning_output": None,
        },
        "provider_raw": {},
        "cost_api_usd": cost,
    }
    if arm is not None:
        rec["arm"] = arm
    if excluded is not None:
        rec["excluded"] = excluded
    if protocol_version is not None:
        rec["protocol_version"] = protocol_version
    return rec


# ---------------------------------------------------------------------------
# Reference sessions (8 hand-computed sessions)
# ---------------------------------------------------------------------------

# arm_for("aa"*32, "repoA", 0) = "on", arm_for(..., 1) = "off"
# arm_for("aa"*32, "repoA", 9) = "on", arm_for(..., 10) = "off"
# arm_for("aa"*32, "repoB", 1) = "on", arm_for(..., 0) = "off"
# arm_for("aa"*32, "repoB", 8) = "on", arm_for(..., 10) = "off"

_SESSION_DATA = [
    # (repo_id, block, cost, arm-label matches PRF)
    ("repoA", 0,  0.10, "on"),   # group 0, on
    ("repoA", 1,  0.04, "off"),  # group 0, off
    ("repoA", 9,  0.05, "on"),   # group 1, on
    ("repoA", 10, 0.03, "off"),  # group 1, off
    ("repoB", 1,  0.20, "on"),   # group 0, on
    ("repoB", 0,  0.12, "off"),  # group 0, off
    ("repoB", 8,  0.15, "on"),   # group 1, on
    ("repoB", 10, 0.11, "off"),  # group 1, off
]

_LIVE_SESSIONS = [
    _make_session(repo, blk, cost, arm=arm_label)
    for repo, blk, cost, arm_label in _SESSION_DATA
]

# Pre-study sessions (used for CUPED X, not for Y)
_PRE_SESSIONS = [
    _make_session("repoA", 0, 0.08, pre_study=True),
    _make_session("repoB", 0, 0.16, pre_study=True),
]

_ALL_SESSIONS = _LIVE_SESSIONS + _PRE_SESSIONS


# ---------------------------------------------------------------------------
# TestHandComputed
# ---------------------------------------------------------------------------


class TestHandComputed(unittest.TestCase):
    """4 pairs, exact T_obs/p/CR0/CUPED verified in module docstring."""

    def setUp(self):
        self.config = _make_config()
        self.result = analyze(_ALL_SESSIONS, self.config, _PROTOCOL_TEXT)

    def test_status_ok(self):
        self.assertEqual(self.result["status"], "ok")

    def test_n_pairs(self):
        self.assertEqual(self.result["primary"]["n_pairs"], 4)

    def test_T_obs(self):
        T_obs = self.result["primary"]["permutation"]["T_obs"]
        self.assertAlmostEqual(T_obs, 0.05, places=10)

    def test_p_value(self):
        p = self.result["primary"]["permutation"]["p_two_sided"]
        # Pre-computed: count_extreme=1225, p=(1226)/10001
        self.assertAlmostEqual(p, 1226 / 10001, places=10)

    def test_count_extreme(self):
        ce = self.result["primary"]["permutation"]["count_extreme"]
        self.assertEqual(ce, 1225)

    def test_cr0_SE(self):
        SE = self.result["primary"]["cr0"]["SE"]
        # SE = 0.01 exactly (see module docstring derivation)
        self.assertAlmostEqual(SE, 0.01, places=10)

    def test_cr0_z(self):
        z = self.result["primary"]["cr0"]["z"]
        self.assertAlmostEqual(z, 5.0, places=8)

    def test_cr0_G(self):
        self.assertEqual(self.result["primary"]["cr0"]["G"], 2)

    def test_pairs_present(self):
        pairs = self.result["primary"]["pairs"]
        self.assertEqual(len(pairs), 4)
        d_values = sorted(p["d_i"] for p in pairs)
        self.assertAlmostEqual(d_values[0], 0.02, places=10)
        self.assertAlmostEqual(d_values[1], 0.04, places=10)
        self.assertAlmostEqual(d_values[2], 0.06, places=10)
        self.assertAlmostEqual(d_values[3], 0.08, places=10)

    def test_cuped_b(self):
        # b = 0.25 (see module docstring)
        b = self.result["primary"]["cuped"]["b"]
        self.assertAlmostEqual(b, 0.25, places=10)

    def test_cuped_mean_d_adj(self):
        # mean_d_adj = 0.05
        mean_d_adj = self.result["primary"]["cuped"]["mean_d_adj"]
        self.assertAlmostEqual(mean_d_adj, 0.05, places=10)

    def test_cuped_naive_se(self):
        # naive_se = sqrt(0.0016/12) ≈ 0.011547
        naive_se = self.result["primary"]["cuped"]["naive_se_d_adj"]
        expected = math.sqrt(0.0016 / 12.0)
        self.assertAlmostEqual(naive_se, expected, places=8)

    def test_pairs_include_audit_fields(self):
        """Pairs list must carry repo_id, group, d_i, n_on, n_off."""
        for p in self.result["primary"]["pairs"]:
            for key in ("repo_id", "group", "d_i", "n_on", "n_off"):
                self.assertIn(key, p, f"pair missing key {key!r}")

    def test_result_includes_exclusion_tallies(self):
        tallies = self.result["exclusion_tallies"]
        self.assertIn("kept", tallies)
        # 8 live sessions kept, 2 pre-study excluded (before_live)
        self.assertEqual(tallies["kept"], 8)
        self.assertEqual(tallies["before_live"], 2)

    def test_no_label_mismatches(self):
        """All record arm labels match PRF (we set them correctly)."""
        self.assertEqual(self.result["primary"]["label_mismatches"], 0)

    def test_coverage_counts(self):
        cov = self.result["coverage"]
        self.assertEqual(cov["on_sessions"], 4)
        self.assertEqual(cov["off_sessions"], 4)


# ---------------------------------------------------------------------------
# TestPermutationDeterminism
# ---------------------------------------------------------------------------


class TestPermutationDeterminism(unittest.TestCase):
    """Same seed → identical p-value on every run."""

    def test_same_seed_same_p(self):
        d = [0.06, 0.02, 0.08, 0.04]
        r1 = _permutation_test(d)
        r2 = _permutation_test(d)
        self.assertEqual(r1["p_two_sided"], r2["p_two_sided"])
        self.assertEqual(r1["count_extreme"], r2["count_extreme"])

    def test_exact_count_extreme(self):
        d = [0.06, 0.02, 0.08, 0.04]
        r = _permutation_test(d)
        self.assertEqual(r["count_extreme"], 1225)
        self.assertEqual(r["n_iterations"], 10000)

    def test_p_formula(self):
        d = [0.06, 0.02, 0.08, 0.04]
        r = _permutation_test(d)
        expected = (1 + r["count_extreme"]) / (r["n_iterations"] + 1)
        self.assertAlmostEqual(r["p_two_sided"], expected, places=15)

    def test_empty_d_values(self):
        r = _permutation_test([])
        self.assertIsNone(r["T_obs"])
        self.assertIsNone(r["p_two_sided"])


# ---------------------------------------------------------------------------
# TestGuardRefusals
# ---------------------------------------------------------------------------


class TestGuardRefusals(unittest.TestCase):
    """Each hard guard fires AnalysisRefused with a non-empty message."""

    def _base_config(self):
        return _make_config()

    def test_bad_protocol_hash(self):
        config = self._base_config()
        config["protocol_sha256"] = "0" * 64  # wrong hash
        with self.assertRaises(AnalysisRefused) as ctx:
            analyze(_LIVE_SESSIONS, config, _PROTOCOL_TEXT)
        self.assertIn("hash mismatch", str(ctx.exception))

    def test_missing_protocol_sha256(self):
        config = self._base_config()
        del config["protocol_sha256"]
        with self.assertRaises(AnalysisRefused) as ctx:
            analyze(_LIVE_SESSIONS, config, _PROTOCOL_TEXT)
        self.assertIn("protocol_sha256", str(ctx.exception))

    def test_record_protocol_version_mismatch(self):
        config = self._base_config()
        bad_rec = _make_session("repoA", 0, 0.10, arm="on", protocol_version="99.99.99")
        with self.assertRaises(AnalysisRefused) as ctx:
            analyze([bad_rec], config, _PROTOCOL_TEXT)
        self.assertIn("protocol_version", str(ctx.exception))

    def test_missing_live_started_at(self):
        config = self._base_config()
        del config["live_started_at"]
        with self.assertRaises(AnalysisRefused) as ctx:
            analyze(_LIVE_SESSIONS, config, _PROTOCOL_TEXT)
        self.assertIn("live_started_at", str(ctx.exception))

    def test_mode_not_live(self):
        config = self._base_config()
        config["mode"] = "dry-run"
        with self.assertRaises(AnalysisRefused) as ctx:
            analyze(_LIVE_SESSIONS, config, _PROTOCOL_TEXT)
        self.assertIn("mode", str(ctx.exception))

    def test_mode_missing(self):
        config = self._base_config()
        config["mode"] = None
        with self.assertRaises(AnalysisRefused) as ctx:
            analyze(_LIVE_SESSIONS, config, _PROTOCOL_TEXT)
        self.assertIn("mode", str(ctx.exception))


# ---------------------------------------------------------------------------
# TestOneArmedGroups
# ---------------------------------------------------------------------------


class TestOneArmedGroups(unittest.TestCase):
    """Groups with only one arm present are dropped from pairing."""

    def test_group_with_only_on_dropped(self):
        # Only on-blocks for repoA group 0; no off-block → no pair
        sessions = [
            _make_session("repoA", 0, 0.10, arm="on"),  # on
            # block 2 is also "on" for repoA group 0 → still no pair
        ]
        config = _make_config()
        result = analyze(sessions, config, _PROTOCOL_TEXT)
        # status should be insufficient_data or pairs=0
        primary = result.get("primary")
        if result["status"] == "ok":
            self.assertEqual(primary["n_pairs"], 0)
        else:
            self.assertEqual(result["status"], "insufficient_data")

    def test_group_with_only_off_dropped(self):
        # Only off-blocks for repoB
        sessions = [
            _make_session("repoB", 0, 0.12, arm="off"),  # off
            _make_session("repoB", 3, 0.10, arm="off"),  # off (block 3 is off too)
        ]
        config = _make_config()
        result = analyze(sessions, config, _PROTOCOL_TEXT)
        primary = result.get("primary")
        if result["status"] == "ok":
            self.assertEqual(primary["n_pairs"], 0)
        else:
            self.assertEqual(result["status"], "insufficient_data")

    def test_mixed_repos_one_armed_dropped(self):
        """repoA group 0 has both arms → pair; repoB group 0 has only off → no pair."""
        sessions = [
            _make_session("repoA", 0, 0.10, arm="on"),   # repoA, group 0, on
            _make_session("repoA", 1, 0.04, arm="off"),  # repoA, group 0, off
            _make_session("repoB", 0, 0.12, arm="off"),  # repoB, group 0, off only
        ]
        config = _make_config()
        result = analyze(sessions, config, _PROTOCOL_TEXT)
        if result["status"] == "ok":
            # Only 1 pair from repoA
            self.assertEqual(result["primary"]["n_pairs"], 1)


# ---------------------------------------------------------------------------
# TestPreStudyRecords
# ---------------------------------------------------------------------------


class TestPreStudyRecords(unittest.TestCase):
    """Pre-study records are excluded from Y but used for CUPED X."""

    def test_pre_study_excluded_from_pairs(self):
        # All pre-study records + no live records → no pairs
        config = _make_config()
        result = analyze(_PRE_SESSIONS, config, _PROTOCOL_TEXT)
        # Either insufficient data or 0 pairs
        if result["status"] == "ok":
            self.assertEqual(result["primary"]["n_pairs"], 0)
        else:
            self.assertEqual(result["status"], "insufficient_data")

    def test_pre_study_used_for_cuped(self):
        # With pre-study records, CUPED should find repos in pre data
        config = _make_config()
        result = analyze(_ALL_SESSIONS, config, _PROTOCOL_TEXT)
        cuped = result["primary"]["cuped"]
        self.assertEqual(cuped["n_repos_with_pre_data"], 2)
        self.assertIsNotNone(cuped["b"])

    def test_without_pre_study_cuped_b_may_be_none(self):
        # No pre-study records → no CUPED covariate → b is None or fallback
        config = _make_config()
        result = analyze(_LIVE_SESSIONS, config, _PROTOCOL_TEXT)
        cuped = result["primary"]["cuped"]
        # Without pre-study data, n_repos_with_pre_data == 0
        self.assertEqual(cuped["n_repos_with_pre_data"], 0)
        # b is None when there's no pre-data
        self.assertIsNone(cuped["b"])

    def test_exclusion_tally_before_live(self):
        config = _make_config()
        result = analyze(_ALL_SESSIONS, config, _PROTOCOL_TEXT)
        # 2 pre-study records filtered as before_live
        self.assertEqual(result["exclusion_tallies"]["before_live"], 2)


# ---------------------------------------------------------------------------
# TestEmptyData
# ---------------------------------------------------------------------------


class TestEmptyData(unittest.TestCase):
    """Empty data path returns a well-formed result, never raises."""

    def test_empty_records_no_raise(self):
        config = _make_config()
        result = analyze([], config, _PROTOCOL_TEXT)
        self.assertIn("status", result)

    def test_empty_records_insufficient_data(self):
        config = _make_config()
        result = analyze([], config, _PROTOCOL_TEXT)
        self.assertEqual(result["status"], "insufficient_data")

    def test_empty_result_has_exclusion_tallies(self):
        config = _make_config()
        result = analyze([], config, _PROTOCOL_TEXT)
        self.assertIn("exclusion_tallies", result)

    def test_all_excluded_insufficient_data(self):
        sessions = [
            _make_session("repoA", 0, 0.10, arm="on", excluded="block-boundary"),
            _make_session("repoB", 1, 0.20, arm="on", excluded="codex-superseded-rollout"),
        ]
        config = _make_config()
        result = analyze(sessions, config, _PROTOCOL_TEXT)
        self.assertEqual(result["status"], "insufficient_data")

    def test_all_none_cost_insufficient_data(self):
        sessions = [
            _make_session("repoA", 0, 0.10, arm="on"),
        ]
        # Override cost to None
        sessions[0]["cost_api_usd"] = None
        config = _make_config()
        result = analyze(sessions, config, _PROTOCOL_TEXT)
        self.assertEqual(result["status"], "insufficient_data")

    def test_insufficient_data_result_shape(self):
        config = _make_config()
        result = analyze([], config, _PROTOCOL_TEXT)
        # Must have these keys even when insufficient
        for key in ("status", "reason", "exclusion_tallies"):
            self.assertIn(key, result)


# ---------------------------------------------------------------------------
# TestCR0SE
# ---------------------------------------------------------------------------


class TestCR0SE(unittest.TestCase):
    """CR0 SE verified on the 2-repo 4-pair example."""

    def test_cr0_se_formula(self):
        # d values: [0.06, 0.02, 0.08, 0.04], repos: A has 0,1; B has 2,3
        pairs = [
            {"repo_id": "repoA", "group": 0, "d_i": 0.06, "n_on": 1, "n_off": 1,
             "mean_on": 0.10, "mean_off": 0.04},
            {"repo_id": "repoA", "group": 1, "d_i": 0.02, "n_on": 1, "n_off": 1,
             "mean_on": 0.05, "mean_off": 0.03},
            {"repo_id": "repoB", "group": 0, "d_i": 0.08, "n_on": 1, "n_off": 1,
             "mean_on": 0.20, "mean_off": 0.12},
            {"repo_id": "repoB", "group": 1, "d_i": 0.04, "n_on": 1, "n_off": 1,
             "mean_on": 0.15, "mean_off": 0.11},
        ]
        result = _cluster_robust_se(pairs)
        self.assertAlmostEqual(result["SE"], 0.01, places=10)
        self.assertAlmostEqual(result["z"], 5.0, places=8)
        self.assertEqual(result["G"], 2)
        self.assertEqual(result["n_pairs"], 4)

    def test_cr0_se_single_pair_no_se(self):
        pairs = [
            {"repo_id": "repoA", "group": 0, "d_i": 0.05, "n_on": 1, "n_off": 1,
             "mean_on": 0.10, "mean_off": 0.05},
        ]
        result = _cluster_robust_se(pairs)
        # n_pairs=1 → can't compute
        self.assertIsNone(result["SE"])

    def test_cr0_se_single_repo_no_se(self):
        # G=1 → can't apply G/(G-1)
        pairs = [
            {"repo_id": "repoA", "group": 0, "d_i": 0.05, "n_on": 1, "n_off": 1,
             "mean_on": 0.10, "mean_off": 0.05},
            {"repo_id": "repoA", "group": 1, "d_i": 0.03, "n_on": 1, "n_off": 1,
             "mean_on": 0.08, "mean_off": 0.05},
        ]
        result = _cluster_robust_se(pairs)
        self.assertIsNone(result["SE"])
        self.assertEqual(result["G"], 1)


# ---------------------------------------------------------------------------
# TestLabelMismatches
# ---------------------------------------------------------------------------


class TestLabelMismatches(unittest.TestCase):
    """Label mismatch counter tracks arm label vs PRF recompute disagreements."""

    def test_label_mismatch_counted(self):
        # Set arm label opposite to what PRF gives
        # arm_for(seed, "repoA", 0) = "on", so put "off"
        sessions = [
            _make_session("repoA", 0, 0.10, arm="off"),  # mismatch: PRF says "on"
            _make_session("repoA", 1, 0.04, arm="off"),  # correct: PRF says "off"
        ]
        config = _make_config()
        result = analyze(sessions, config, _PROTOCOL_TEXT)
        if result["status"] == "ok":
            # 1 mismatch for block 0 (arm label "off" vs PRF "on")
            self.assertEqual(result["primary"]["label_mismatches"], 1)

    def test_no_mismatch_when_labels_correct(self):
        config = _make_config()
        result = analyze(_ALL_SESSIONS, config, _PROTOCOL_TEXT)
        self.assertEqual(result["primary"]["label_mismatches"], 0)


# ---------------------------------------------------------------------------
# TestReport
# ---------------------------------------------------------------------------


class TestReport(unittest.TestCase):
    """report() returns a non-empty string with interpretation rules."""

    def test_report_ok_result(self):
        config = _make_config()
        result = analyze(_ALL_SESSIONS, config, _PROTOCOL_TEXT)
        text = report(result)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_report_contains_interpretation_rules(self):
        config = _make_config()
        result = analyze(_ALL_SESSIONS, config, _PROTOCOL_TEXT)
        text = report(result)
        # Pre-registered interpretation rules must appear
        self.assertIn("0.05", text)
        self.assertIn("two-sided", text)
        self.assertIn("Guardrail", text)

    def test_report_insufficient_data(self):
        config = _make_config()
        result = analyze([], config, _PROTOCOL_TEXT)
        text = report(result)
        self.assertIn("INSUFFICIENT DATA", text)

    def test_report_insufficient_data_has_interpretation(self):
        config = _make_config()
        result = analyze([], config, _PROTOCOL_TEXT)
        text = report(result)
        # Interpretation rules must appear even for insufficient data
        self.assertIn("0.05", text)


# ---------------------------------------------------------------------------
# TestExcludedRepos
# ---------------------------------------------------------------------------


class TestExcludedRepos(unittest.TestCase):
    """Sessions from excluded repos are not counted."""

    def test_excluded_repo_not_in_pairs(self):
        config = _make_config(exclude_repo_ids=["repoA"])
        result = analyze(_ALL_SESSIONS, config, _PROTOCOL_TEXT)
        if result["status"] == "ok":
            pairs = result["primary"]["pairs"]
            repos_in_pairs = {p["repo_id"] for p in pairs}
            self.assertNotIn("repoA", repos_in_pairs)

    def test_excluded_repo_counts_in_tally(self):
        config = _make_config(exclude_repo_ids=["repoA"])
        result = analyze(_ALL_SESSIONS, config, _PROTOCOL_TEXT)
        # 4 live repoA sessions (or pre-study) should count in excluded_repo tally
        self.assertGreater(result["exclusion_tallies"]["excluded_repo"], 0)


if __name__ == "__main__":
    unittest.main()
