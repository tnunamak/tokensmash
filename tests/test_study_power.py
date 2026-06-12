"""Tests for tokensmash.study.power.

Coverage:
- block_costs: normal grouping, None-cost skip, excluded skip, empty input
- mde: deterministic hand-verified result, single-block error, all-None error,
       empty-records error
- occupancy/effective-pairs math on a constructed example
- report: smoke test (returns non-empty string)

Hand-verification of the deterministic MDE case
------------------------------------------------
Records: 4 blocks, each with cost 0.10 USD; two repos, two blocks each.
  block_costs -> [0.10, 0.10, 0.10, 0.10]
  mean = 0.10, variance (sample) = 0.0
  -> variance = 0, so MDE = 0. Not useful; use a mixed set instead.

Deterministic test set (synthetic):
  Block costs: [0.05, 0.15, 0.10, 0.20]
  n_blocks = 4
  mean = 0.125
  variance = ((0.05-0.125)^2 + (0.15-0.125)^2 + (0.10-0.125)^2 + (0.20-0.125)^2) / 3
           = (0.005625 + 0.000625 + 0.000625 + 0.005625) / 3
           = 0.0125 / 3
           = 0.004166...

  We give the records timestamps spanning exactly 4 weeks so observed_weeks = 4.0
  (within floating-point tolerance). Actually we compute it from min/max unix.
  Let's set min unix = 0, max unix = 4 * 7 * 86400 = 2419200.
  observed_weeks = 2419200 / 604800 = 4.0
  blocks_per_week = 4 / 4.0 = 1.0

  For weeks=8:
    n = (1.0 * 8) / 2 = 4.0
    za2 = 1.96, zb = 0.8416
    delta = (1.96 + 0.8416) * sqrt(2 * (0.004166...) / 4.0)
          = 2.8016 * sqrt(0.002083...)
          = 2.8016 * 0.045644...
          = 2.8016 * sqrt(1/480)
    Exact: variance = 1/240, 2*variance/n = 2*(1/240)/4 = 1/480
           sqrt(1/480) = 0.045644...
           delta = 2.8016 * 0.045644... = 0.12789...

  Let's compute to 4 decimal places:
    variance = 0.0125 / 3 = 0.00416666...
    2 * variance / 4 = 0.00208333...
    sqrt(0.00208333...) = 0.04564354...
    2.8016 * 0.04564354... = 0.12789...

  More precisely:
    variance = (0.005625 + 0.000625 + 0.000625 + 0.005625) / 3
             = 0.012500 / 3
    sqrt(2 * 0.012500 / (3 * 4)) = sqrt(0.025 / 12) = sqrt(1/480)
    1/480 = 0.002083333...
    sqrt(1/480) = 0.045643546...
    (1.96 + 0.8416) = 2.8016
    delta = 2.8016 * 0.045643546... = 0.127865...

  We assert to 4 decimal places: 0.1279 (rounded).
"""

from __future__ import annotations

import math
import unittest

from tokensmash.study.power import block_costs, mde, report, _norm_quantile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EPOCH = 0  # 1970-01-01T00:00:00Z
_WEEK_SECS = 7 * 24 * 3600  # 604800
_BLOCK_SECS = 7200  # 2 hours


def _ts(unix_secs: float) -> str:
    """Format unix seconds as ISO-8601 UTC."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(unix_secs, tz=timezone.utc).isoformat()


def _rec(
    repo_id: str,
    unix_secs: float,
    cost: float | None,
    excluded: bool = False,
) -> dict:
    """Minimal session record."""
    r: dict = {
        "repo_id": repo_id,
        "started_at": _ts(unix_secs),
        "cost_api_usd": cost,
    }
    if excluded:
        r["excluded"] = True
    return r


# Deterministic test set:
# 4 records in 4 distinct (repo, block) groups, with costs [0.05, 0.15, 0.10, 0.20].
# repo "A": block at t=0 (cost 0.05) and block at t=4*WEEK_SECS (cost 0.20)
# repo "B": block at t=1*WEEK_SECS (cost 0.15) and block at t=2*WEEK_SECS (cost 0.10)
_DET_RECORDS = [
    _rec("A", 0, 0.05),
    _rec("A", 4 * _WEEK_SECS, 0.20),
    _rec("B", 1 * _WEEK_SECS, 0.15),
    _rec("B", 2 * _WEEK_SECS, 0.10),
]


# ---------------------------------------------------------------------------
# block_costs tests
# ---------------------------------------------------------------------------


class TestBlockCosts(unittest.TestCase):
    def test_empty_returns_empty(self):
        self.assertEqual(block_costs([]), [])

    def test_all_none_costs_returns_empty(self):
        records = [
            _rec("A", 0, None),
            _rec("A", _BLOCK_SECS, None),
        ]
        self.assertEqual(block_costs(records), [])

    def test_excluded_records_skipped(self):
        records = [
            _rec("A", 0, 1.00, excluded=True),
            _rec("A", 0, 0.50),  # same block, not excluded
        ]
        result = block_costs(records)
        self.assertEqual(result, [0.50])

    def test_same_repo_same_block_summed(self):
        # Two sessions in the same 2h block for same repo
        t1 = 100.0
        t2 = t1 + 3600  # 1 hour later, still same block
        records = [_rec("A", t1, 0.10), _rec("A", t2, 0.20)]
        result = block_costs(records)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 0.30, places=10)

    def test_different_repos_different_groups(self):
        t = 0.0
        records = [_rec("A", t, 0.10), _rec("B", t, 0.20)]
        result = block_costs(records)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(sorted(result)[0], 0.10, places=10)
        self.assertAlmostEqual(sorted(result)[1], 0.20, places=10)

    def test_different_blocks_same_repo_separate_groups(self):
        t1 = 0.0
        t2 = _BLOCK_SECS * 10  # 10 blocks later
        records = [_rec("A", t1, 0.10), _rec("A", t2, 0.30)]
        result = block_costs(records)
        self.assertEqual(len(result), 2)

    def test_mixed_none_and_valid(self):
        records = [
            _rec("A", 0, None),
            _rec("A", 0, 0.10),
            _rec("B", _WEEK_SECS, 0.20),
        ]
        result = block_costs(records)
        # None for A@0 is skipped; valid 0.10 for A@0 counts; B is separate
        self.assertEqual(len(result), 2)
        total = sum(result)
        self.assertAlmostEqual(total, 0.30, places=10)

    def test_det_records_four_distinct_blocks(self):
        result = block_costs(_DET_RECORDS)
        self.assertEqual(len(result), 4)
        for actual, expected in zip(sorted(result), sorted([0.05, 0.15, 0.10, 0.20])):
            self.assertAlmostEqual(actual, expected, places=10)


# ---------------------------------------------------------------------------
# mde tests
# ---------------------------------------------------------------------------


class TestMde(unittest.TestCase):
    def test_empty_records_raises(self):
        with self.assertRaises(ValueError):
            mde([], 8)

    def test_all_none_costs_raises(self):
        records = [_rec("A", 0, None), _rec("A", _BLOCK_SECS, None)]
        with self.assertRaises(ValueError):
            mde(records, 8)

    def test_single_block_raises(self):
        # All records land in the same (repo, block) -- need 2+ blocks
        records = [
            _rec("A", 0, 0.10),
            _rec("A", 100, 0.20),  # same 2h block
        ]
        with self.assertRaises(ValueError):
            mde(records, 8)

    def test_deterministic_mde(self):
        """Hand-verify MDE to 4 decimal places using _DET_RECORDS at 8 weeks.

        Derivation (see module docstring):
          costs = [0.05, 0.15, 0.10, 0.20]
          mean  = 0.125
          variance (sample, n-1) = 0.0125 / 3 = 0.00416666...
          observed_weeks = 4.0 (min t=0, max t=4*WEEK_SECS)
          blocks_per_week = 4 / 4.0 = 1.0
          n = (1.0 * 8) / 2 = 4.0
          delta = (1.96 + 0.8416) * sqrt(2 * 0.00416666... / 4.0)
                = 2.8016 * sqrt(1/480)
                = 2.8016 * 0.045643546...
                = 0.127866...
        """
        result = mde(_DET_RECORDS, 8)

        self.assertEqual(result["n_sessions"], 4)
        self.assertEqual(result["n_blocks"], 4)
        self.assertAlmostEqual(result["observed_weeks"], 4.0, places=4)
        self.assertAlmostEqual(result["blocks_per_week"], 1.0, places=6)
        self.assertAlmostEqual(result["mean_block_cost_usd"], 0.125, places=10)

        expected_var = 0.0125 / 3.0
        expected_sd = math.sqrt(expected_var)
        self.assertAlmostEqual(result["sd_block_cost_usd"], expected_sd, places=8)

        expected_mde = 2.8016 * math.sqrt(2.0 * expected_var / 4.0)
        self.assertAlmostEqual(result["mde_usd_per_block"], expected_mde, places=4)

    def test_mde_keys_present(self):
        result = mde(_DET_RECORDS, 8)
        expected_keys = {
            "n_sessions",
            "n_blocks",
            "observed_weeks",
            "blocks_per_week",
            "mean_block_cost_usd",
            "sd_block_cost_usd",
            "cv",
            "n_blocks_at_weeks",
            "mde_usd_per_block",
            "mde_pct",
            "occupancy_rate",
            "effective_pairs_at_weeks",
            "mde_usd_per_block_effective",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_mde_pct_consistent(self):
        result = mde(_DET_RECORDS, 8)
        expected_pct = result["mde_usd_per_block"] / result["mean_block_cost_usd"] * 100.0
        self.assertAlmostEqual(result["mde_pct"], expected_pct, places=8)

    def test_mde_scales_with_weeks(self):
        """More weeks -> smaller MDE."""
        r4 = mde(_DET_RECORDS, 4)
        r8 = mde(_DET_RECORDS, 8)
        r12 = mde(_DET_RECORDS, 12)
        self.assertGreater(r4["mde_usd_per_block"], r8["mde_usd_per_block"])
        self.assertGreater(r8["mde_usd_per_block"], r12["mde_usd_per_block"])

    def test_cv_positive(self):
        result = mde(_DET_RECORDS, 8)
        self.assertGreater(result["cv"], 0)

    def test_excluded_records_not_counted(self):
        # Add an excluded record with a very high cost that would shift stats
        extra = _rec("Z", 3 * _WEEK_SECS, 9999.0, excluded=True)
        records = _DET_RECORDS + [extra]
        result = mde(records, 8)
        self.assertEqual(result["n_sessions"], 4)  # excluded not counted

    def test_none_cost_records_not_counted(self):
        extra = _rec("Z", 3 * _WEEK_SECS, None)
        records = _DET_RECORDS + [extra]
        result = mde(records, 8)
        self.assertEqual(result["n_sessions"], 4)


# ---------------------------------------------------------------------------
# Occupancy / effective-pairs tests
# ---------------------------------------------------------------------------


class TestOccupancyEffectivePairs(unittest.TestCase):
    def _dense_records(self, n_blocks: int) -> list[dict]:
        """Create records occupying every block from t=0 for n_blocks blocks.

        Costs alternate to ensure non-zero variance.
        """
        _COSTS = [0.05, 0.15, 0.10, 0.20, 0.08, 0.12, 0.09, 0.11]
        records = []
        for i in range(n_blocks):
            records.append(_rec("R", float(i * _BLOCK_SECS), _COSTS[i % len(_COSTS)]))
        return records

    def test_full_occupancy_rate_is_one(self):
        """When every elapsed block has a session, occupancy_rate == 1."""
        records = self._dense_records(16)
        result = mde(records, 8)
        self.assertAlmostEqual(result["occupancy_rate"], 1.0, places=6)

    def test_full_occupancy_effective_mde_equals_raw(self):
        """When occupancy == 1, inflation factor is 1 so effective MDE == raw MDE."""
        records = self._dense_records(16)
        result = mde(records, 8)
        self.assertAlmostEqual(
            result["mde_usd_per_block_effective"],
            result["mde_usd_per_block"],
            places=6,
        )

    def test_partial_occupancy_effective_mde_greater_than_raw(self):
        """When occupancy < 1, effective MDE > raw MDE (harder to detect)."""
        # Create records with gaps: occupy every other block, with varying costs
        # so variance > 0 and both MDEs are non-zero.
        records = []
        costs = [0.05, 0.15, 0.10, 0.20, 0.08, 0.12, 0.09, 0.11]
        for idx, i in enumerate(range(0, 32, 2)):  # even blocks only -> occupancy ~ 0.5
            records.append(_rec("R", float(i * _BLOCK_SECS), costs[idx % len(costs)]))
        result = mde(records, 8)
        self.assertLess(result["occupancy_rate"], 1.0)
        self.assertGreater(
            result["mde_usd_per_block_effective"],
            result["mde_usd_per_block"],
        )

    def test_effective_pairs_formula(self):
        """Verify effective_pairs_at_weeks matches the documented formula.

        For p=1 occupancy, blocks_per_week w, weeks W:
          projected_groups = (w * W) / 8
          effective_pairs  = projected_groups * 4 * 1^2
                           = (w * W) / 8 * 4
                           = (w * W) / 2
                           = n_blocks_at_weeks
        """
        records = self._dense_records(16)
        result = mde(records, 8)
        expected_epairs = result["n_blocks_at_weeks"]  # p=1 case
        self.assertAlmostEqual(result["effective_pairs_at_weeks"], expected_epairs, places=4)

    def test_occupancy_partial_effective_pairs(self):
        """With occupancy p, effective_pairs = (bpw * weeks / 8) * 4 * p^2.

        We construct a scenario with known occupancy and verify the formula.
        """
        # 8 occupied blocks out of 16 elapsed -> p = 0.5
        records = []
        for i in range(0, 16, 2):
            records.append(_rec("S", float(i * _BLOCK_SECS), 0.10))
        result = mde(records, 8)

        p = result["occupancy_rate"]
        bpw = result["blocks_per_week"]
        projected_groups = (bpw * 8) / 8.0
        expected_epairs = projected_groups * 4.0 * (p**2)
        self.assertAlmostEqual(result["effective_pairs_at_weeks"], expected_epairs, places=4)

    def test_effective_mde_inflation_formula(self):
        """mde_effective = mde_raw * sqrt(n_blocks_at_weeks / effective_pairs)."""
        # Create 8 occupied blocks out of 15 elapsed (every other)
        _COSTS = [0.05, 0.15, 0.10, 0.20, 0.08, 0.12, 0.09, 0.11]
        records = [
            _rec("R", float(i * _BLOCK_SECS * 2), _COSTS[j % len(_COSTS)])
            for j, i in enumerate(range(8))
        ]
        result = mde(records, 8)
        n = result["n_blocks_at_weeks"]
        ep = result["effective_pairs_at_weeks"]
        expected_effective = result["mde_usd_per_block"] * math.sqrt(n / ep)
        self.assertAlmostEqual(
            result["mde_usd_per_block_effective"], expected_effective, places=6
        )


# ---------------------------------------------------------------------------
# Norm quantile tests
# ---------------------------------------------------------------------------


class TestNormQuantile(unittest.TestCase):
    def test_standard_cases(self):
        # z_{0.025} = 1.96, z_{0.84} ~ 0.9946 -- note: quantile(0.975) = 1.96
        self.assertAlmostEqual(_norm_quantile(0.975), 1.96, places=1)
        self.assertAlmostEqual(_norm_quantile(0.8416), 1.0, places=1)

    def test_symmetry(self):
        self.assertAlmostEqual(_norm_quantile(0.3), -_norm_quantile(0.7), places=4)

    def test_midpoint(self):
        self.assertAlmostEqual(_norm_quantile(0.5), 0.0, places=4)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _norm_quantile(0.0)
        with self.assertRaises(ValueError):
            _norm_quantile(1.0)


# ---------------------------------------------------------------------------
# report tests
# ---------------------------------------------------------------------------


class TestReport(unittest.TestCase):
    def test_report_returns_string(self):
        result = report(_DET_RECORDS)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_report_contains_week_horizons(self):
        result = report(_DET_RECORDS)
        self.assertIn("4-week", result)
        self.assertIn("8-week", result)
        self.assertIn("12-week", result)

    def test_report_contains_interpretation(self):
        result = report(_DET_RECORDS)
        self.assertIn("Interpretation", result)
        self.assertIn("8 weeks", result)

    def test_report_insufficient_data(self):
        # Empty records -> no usable data
        result = report([])
        self.assertIn("error", result.lower())

    def test_report_all_none_costs(self):
        records = [_rec("A", 0, None)]
        result = report(records)
        self.assertIn("error", result.lower())


if __name__ == "__main__":
    unittest.main()
