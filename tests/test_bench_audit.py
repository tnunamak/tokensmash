"""Tests for the wave-2 lab-bench repairs (docs/CONTRACTS.md §15).

Replaces a pytest-style file that never ran under this project's unittest
runner. Covers: position-design balance (including the single-replicate
cross-task case that motivated the repair), the suite-audit positions check,
baseline-mean pairing with the baseline-sd column, the code-version strict
gate, and min-N CI gating.
"""

from __future__ import annotations

import collections
import unittest

from tokensmash.cli import (
    BENCH_AUDIT_VERSION,
    _build_position_design,
    _code_version_stamp,
    _mean,
    _stddev,
    _strict_version_check,
    aggregate_rows,
    audit_suite_methodology,
    ci_cell,
)


def _tasks(n: int) -> list[dict]:
    return [{"id": f"t{i}"} for i in range(n)]


def _variants(n: int) -> list[dict]:
    return [{"id": f"v{i}"} for i in range(n)]


class TestPositionDesign(unittest.TestCase):
    def test_single_replicate_balances_across_tasks(self):
        # The regression that motivated the repair: with replicates=1, every
        # task must NOT run variants in the same order (cold-cache position
        # confound — in a real run one variant led on all three tasks).
        design = _build_position_design(_tasks(3), _variants(5), 1, seed=611)
        counts = collections.Counter((e["variant_id"], e["position"]) for e in design)
        self.assertEqual(max(counts.values()), 1)
        leaders = {e["task_id"]: e["variant_id"] for e in design if e["position"] == 1}
        self.assertEqual(len(set(leaders.values())), 3)

    def test_divisible_shape_is_perfectly_balanced(self):
        design = _build_position_design(_tasks(10), _variants(5), 1, seed=611)
        counts = collections.Counter((e["variant_id"], e["position"]) for e in design)
        self.assertEqual(set(counts.values()), {2})

    def test_replicates_rotate_within_task(self):
        design = _build_position_design(_tasks(1), _variants(3), 3, seed=1)
        counts = collections.Counter((e["variant_id"], e["position"]) for e in design)
        self.assertEqual(set(counts.values()), {1})

    def test_deterministic_for_seed(self):
        a = _build_position_design(_tasks(4), _variants(3), 2, seed=42)
        b = _build_position_design(_tasks(4), _variants(3), 2, seed=42)
        self.assertEqual(a, b)

    def test_every_slot_present(self):
        design = _build_position_design(_tasks(3), _variants(4), 2, seed=7)
        self.assertEqual(len(design), 3 * 4 * 2)
        for entry in design:
            self.assertIn("task_id", entry)
            self.assertIn("variant_id", entry)
            self.assertIn("position", entry)


def _suite_results(**overrides) -> dict:
    results = {
        "host_fingerprint_before": {"f": 1},
        "host_fingerprint_after": {"f": 1},
        "randomized_order": False,
        "runs": [{"status": "ok", "run_order": 1}],
    }
    results.update(overrides)
    return results


class TestSuiteAuditPositions(unittest.TestCase):
    def _check(self, audit: dict, label: str) -> dict:
        matches = [c for c in audit["checks"] if c["label"] == label]
        self.assertEqual(len(matches), 1, audit["checks"])
        return matches[0]

    def test_randomized_order_satisfies_positions_check(self):
        audit = audit_suite_methodology(_suite_results(randomized_order=True))
        self.assertTrue(self._check(audit, "positions balanced or randomized")["ok"])

    def test_position_design_satisfies_positions_check(self):
        audit = audit_suite_methodology(
            _suite_results(position_design=[{"task_id": "t0", "variant_id": "v0", "position": 1}])
        )
        self.assertTrue(self._check(audit, "positions balanced or randomized")["ok"])

    def test_neither_fails_positions_check_and_audit(self):
        audit = audit_suite_methodology(_suite_results())
        self.assertFalse(self._check(audit, "positions balanced or randomized")["ok"])
        self.assertFalse(audit["ok"])


def _run(
    variant_id: str,
    replicate: int,
    input_tokens: int,
    cached: int,
    output: int,
    task_id: str = "t0",
    is_baseline: bool = False,
) -> dict:
    total = input_tokens + output
    run = {
        "task_id": task_id,
        "variant_id": variant_id,
        "replicate": replicate,
        "success": True,
        "token_total": total,
        "token_usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached,
            "output_tokens": output,
            "reasoning_output_tokens": 0,
            "total_tokens": total,
        },
        "methodology_audit": {"ok": True},
    }
    if not is_baseline:
        run["mechanism_checks"] = {"required": True, "ok": True, "checks": []}
    return run


class TestBaselineMeanPairing(unittest.TestCase):
    """Hand-computed pairing against the mean of two baseline replicates.

    Baseline r1: input 90k (50k cached) + out 10k → total 100k, non-cached 50k
    Baseline r2: input 110k (60k cached) + out 10k → total 120k, non-cached 60k
    Task mean: total 110k, non-cached 55k. SD of totals (ddof=1) = 14142.13…
    Tool r1:    input 80k (40k cached) + out 5k → total 85k, non-cached 45k
    """

    def setUp(self):
        self.results = {
            "baseline": "baseline",
            "suite_methodology_audit": {"ok": True},
            "runs": [
                _run("baseline", 1, 90_000, 50_000, 10_000, is_baseline=True),
                _run("baseline", 2, 110_000, 60_000, 10_000, is_baseline=True),
                _run("tool", 1, 80_000, 40_000, 5_000),
            ],
        }

    def test_stddev_hand_value(self):
        self.assertAlmostEqual(_stddev([100_000.0, 120_000.0]), 14142.135623730951, places=6)
        self.assertEqual(_stddev([5.0]), 0.0)
        self.assertAlmostEqual(_mean([100_000.0, 120_000.0]), 110_000.0)

    def test_pairs_against_baseline_mean(self):
        rows = aggregate_rows([self.results])
        tool_rows = [r for r in rows if r[0] == "tool"]
        self.assertEqual(len(tool_rows), 1)
        row = tool_rows[0]
        self.assertEqual(row[1], "1")  # one pair
        # Non-cached: tool 45k < baseline mean 55k → positive
        self.assertEqual(row[13], "1/1")
        # Baseline SD column carries the across-replicate token SD (14,142)
        self.assertEqual(row[12].replace(",", ""), "14142")
        # Single pair → pilot CI gating, not a bootstrap interval
        self.assertEqual(row[11], "pilot (n<10)")

    def test_single_baseline_replicate_sd_zero_cell(self):
        results = {
            "baseline": "baseline",
            "suite_methodology_audit": {"ok": True},
            "runs": [
                _run("baseline", 1, 90_000, 50_000, 10_000, is_baseline=True),
                _run("tool", 1, 80_000, 40_000, 5_000),
            ],
        }
        rows = aggregate_rows([results])
        row = [r for r in rows if r[0] == "tool"][0]
        self.assertEqual(row[12].replace(",", ""), "0")


class TestCodeVersionStamp(unittest.TestCase):
    def test_stamp_carries_current_version(self):
        stamp = _code_version_stamp()
        self.assertEqual(stamp["bench_audit_version"], BENCH_AUDIT_VERSION)
        self.assertEqual(BENCH_AUDIT_VERSION, 2)

    def test_strict_check_accepts_current(self):
        ok = {"code_version": {"bench_audit_version": BENCH_AUDIT_VERSION}, "_path": "good.json"}
        self.assertEqual(_strict_version_check([ok]), [])

    def test_strict_check_flags_missing_and_old(self):
        missing = {"_path": "missing.json"}
        old = {"code_version": {"bench_audit_version": 1}, "_path": "old.json"}
        self.assertEqual(_strict_version_check([missing, old]), ["missing.json", "old.json"])


class TestMinNGating(unittest.TestCase):
    def test_below_ten_is_pilot(self):
        for n in (1, 5, 9):
            self.assertEqual(ci_cell([1.0] * n), "pilot (n<10)")

    def test_at_ten_is_bootstrap_interval(self):
        cell = ci_cell([float(i) for i in range(10)])
        self.assertIn("..", cell)

    def test_empty_is_na(self):
        self.assertEqual(ci_cell([]), "n/a")


if __name__ == "__main__":
    unittest.main()
