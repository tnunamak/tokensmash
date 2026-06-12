"""Tests for tokensmash.meta module.

Covers:
  - merge: two synthetic exports (overlapping keys → dedupe + conflict count),
    unscrubbed refusal (transcript_path), invalid-record skip counting,
    corrupt JSON skip, empty paths list.
  - report: per-machine blocks, combined section, expected numbers on a
    hand-built corpus, caveat line, 0-record edge case.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tokensmash.schema as schema
from tokensmash import meta


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

SESSION_SCHEMA = schema.SESSION_SCHEMA


def _make_record(
    machine_id: str = "aabbccdd11223344",
    agent: str = "codex",
    session_id: str = "sess-001",
    transcript_id: str = "tx-001",
    model: str = "gpt-5.5",
    started_at: str = "2026-01-01T10:00:00+00:00",
    cost_api_usd: float = 1.0,
    fresh_input: int = 1000,
    cache_read: int = 500,
    **extra,
) -> dict:
    rec = {
        "schema": SESSION_SCHEMA,
        "agent": agent,
        "session_id": session_id,
        "transcript_id": transcript_id,
        "machine_id": machine_id,
        "started_at": started_at,
        "model": model,
        "repo_id": "deadbeefcafe0001",
        "user_turns": 1,
        "tool_calls": 2,
        "compactions": 0,
        "duration_ms": 5000,
        "usage": {
            "fresh_input": fresh_input,
            "cache_read": cache_read,
            "cache_write": 0,
            "output": 80,
            "reasoning_output": None,
        },
        "provider_raw": {"input_tokens": fresh_input + cache_read, "output_tokens": 80},
        "cost_api_usd": cost_api_usd,
        "opportunity": {},
    }
    rec.update(extra)
    return rec


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Tests for merge()
# ---------------------------------------------------------------------------

class TestMergeBasic(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _path(self, name: str) -> Path:
        return Path(self._tmp) / name

    def test_single_file_returns_all_valid_records(self):
        p = self._path("a.jsonl")
        records = [
            _make_record(transcript_id="tx-1"),
            _make_record(transcript_id="tx-2"),
        ]
        _write_jsonl(p, records)
        result, _stats = meta.merge([p])
        self.assertEqual(len(result), 2)

    def test_two_files_no_overlap_merged(self):
        p1 = self._path("a.jsonl")
        p2 = self._path("b.jsonl")
        _write_jsonl(p1, [_make_record(machine_id="mach-A", transcript_id="tx-1")])
        _write_jsonl(p2, [_make_record(machine_id="mach-B", transcript_id="tx-2")])
        result, _stats = meta.merge([p1, p2])
        self.assertEqual(len(result), 2)

    def test_empty_paths_list_returns_empty(self):
        result, _stats = meta.merge([])
        self.assertEqual(result, [])

    def test_empty_file_returns_empty(self):
        p = self._path("empty.jsonl")
        p.write_text("")
        result, _stats = meta.merge([p])
        self.assertEqual(result, [])


class TestMergeDedupe(unittest.TestCase):
    """Overlapping (machine_id, agent, transcript_id) → last-wins dedupe."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _path(self, name: str) -> Path:
        return Path(self._tmp) / name

    def test_same_key_in_one_file_last_wins(self):
        p = self._path("a.jsonl")
        r1 = _make_record(transcript_id="tx-1", cost_api_usd=1.0)
        r2 = _make_record(transcript_id="tx-1", cost_api_usd=9.9)  # duplicate key, later
        _write_jsonl(p, [r1, r2])
        result, _stats = meta.merge([p])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["cost_api_usd"], 9.9)

    def test_same_key_across_files_last_wins(self):
        p1 = self._path("first.jsonl")
        p2 = self._path("second.jsonl")
        r1 = _make_record(transcript_id="tx-1", cost_api_usd=1.0)
        r2 = _make_record(transcript_id="tx-1", cost_api_usd=7.7)  # same key, second file
        _write_jsonl(p1, [r1])
        _write_jsonl(p2, [r2])
        result, _stats = meta.merge([p1, p2])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["cost_api_usd"], 7.7)

    def test_conflict_count_incremented(self):
        p1 = self._path("first.jsonl")
        p2 = self._path("second.jsonl")
        shared = _make_record(transcript_id="tx-shared")
        unique = _make_record(transcript_id="tx-unique")
        _write_jsonl(p1, [shared])
        _write_jsonl(p2, [shared, unique])  # shared is a conflict
        _, stats = meta.merge([p1, p2])
        self.assertEqual(stats["conflicts"], 1)

    def test_no_duplicates_zero_conflicts(self):
        p = self._path("a.jsonl")
        _write_jsonl(p, [
            _make_record(transcript_id="tx-1"),
            _make_record(transcript_id="tx-2"),
        ])
        _, stats = meta.merge([p])
        self.assertEqual(stats["conflicts"], 0)

    def test_multiple_conflicts_across_files(self):
        p1 = self._path("first.jsonl")
        p2 = self._path("second.jsonl")
        r_a1 = _make_record(transcript_id="tx-A", cost_api_usd=1.0)
        r_b1 = _make_record(transcript_id="tx-B", cost_api_usd=2.0)
        r_a2 = _make_record(transcript_id="tx-A", cost_api_usd=1.5)
        r_b2 = _make_record(transcript_id="tx-B", cost_api_usd=2.5)
        _write_jsonl(p1, [r_a1, r_b1])
        _write_jsonl(p2, [r_a2, r_b2])
        result, stats = meta.merge([p1, p2])
        self.assertEqual(len(result), 2)
        self.assertEqual(stats["conflicts"], 2)


class TestMergeUnscrubbed(unittest.TestCase):
    """Records containing transcript_path must be refused."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _path(self, name: str) -> Path:
        return Path(self._tmp) / name

    def test_top_level_transcript_path_raises(self):
        p = self._path("bad.jsonl")
        rec = _make_record(transcript_id="tx-1")
        rec["transcript_path"] = "/home/user/.codex/sessions/abc.jsonl"
        _write_jsonl(p, [rec])
        with self.assertRaises(ValueError) as ctx:
            meta.merge([p])
        self.assertIn("transcript_path", str(ctx.exception))

    def test_nested_transcript_path_raises(self):
        p = self._path("bad_nested.jsonl")
        rec = _make_record(transcript_id="tx-1")
        rec["meta"] = {"transcript_path": "/home/user/t.jsonl"}
        _write_jsonl(p, [rec])
        with self.assertRaises(ValueError):
            meta.merge([p])

    def test_valid_file_before_bad_file_still_raises(self):
        p_good = self._path("good.jsonl")
        p_bad = self._path("bad.jsonl")
        _write_jsonl(p_good, [_make_record(transcript_id="tx-ok")])
        bad_rec = _make_record(transcript_id="tx-bad")
        bad_rec["transcript_path"] = "/home/user/t.jsonl"
        _write_jsonl(p_bad, [bad_rec])
        with self.assertRaises(ValueError):
            meta.merge([p_good, p_bad])


class TestMergeInvalidRecords(unittest.TestCase):
    """Invalid records are skipped; count tracked."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def _path(self, name: str) -> Path:
        return Path(self._tmp) / name

    def test_invalid_schema_skipped(self):
        p = self._path("a.jsonl")
        bad = _make_record(transcript_id="tx-bad")
        bad["schema"] = "wrong-schema"
        good = _make_record(transcript_id="tx-good")
        _write_jsonl(p, [bad, good])
        result, _stats = meta.merge([p])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["transcript_id"], "tx-good")

    def test_invalid_skipped_count_tracked(self):
        p = self._path("a.jsonl")
        bad = _make_record(transcript_id="tx-bad")
        bad["schema"] = "wrong"
        good = _make_record(transcript_id="tx-good")
        _write_jsonl(p, [bad, good])
        _, stats = meta.merge([p])
        self.assertEqual(stats["invalid_skipped"], 1)

    def test_corrupt_json_skipped(self):
        p = self._path("corrupt.jsonl")
        with p.open("w") as fh:
            fh.write("{invalid json\n")
            fh.write(json.dumps(_make_record(transcript_id="tx-ok")) + "\n")
        result, stats = meta.merge([p])
        self.assertEqual(len(result), 1)
        self.assertEqual(stats["invalid_skipped"], 1)

    def test_missing_required_key_skipped(self):
        p = self._path("a.jsonl")
        bad = _make_record(transcript_id="tx-bad")
        del bad["machine_id"]
        good = _make_record(transcript_id="tx-good")
        _write_jsonl(p, [bad, good])
        result, _stats = meta.merge([p])
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# Tests for report()
# ---------------------------------------------------------------------------

class TestReportEmpty(unittest.TestCase):

    def test_empty_records(self):
        output = meta.report([])
        self.assertIn("0 machines", output)
        self.assertIn("0 sessions", output)
        self.assertIn("self-selected", output)

    def test_empty_returns_string(self):
        output = meta.report([])
        self.assertIsInstance(output, str)


class TestReportSingleMachine(unittest.TestCase):
    """Tiny hand-built corpus: one machine, two sessions."""

    def _records(self) -> list[dict]:
        return [
            _make_record(
                machine_id="mach-X",
                transcript_id="tx-1",
                cost_api_usd=2.0,
                fresh_input=800,
                cache_read=200,
            ),
            _make_record(
                machine_id="mach-X",
                transcript_id="tx-2",
                cost_api_usd=3.0,
                fresh_input=600,
                cache_read=400,
            ),
        ]

    def test_machine_count_in_header(self):
        output = meta.report(self._records())
        self.assertIn("1 machine", output)

    def test_session_count_in_header(self):
        output = meta.report(self._records())
        self.assertIn("2 sessions", output)

    def test_machine_id_prefix_present(self):
        output = meta.report(self._records())
        self.assertIn("mach-X", output)

    def test_total_cost_in_machine_block(self):
        output = meta.report(self._records())
        # total cost = $5.0000
        self.assertIn("5.0000", output)

    def test_cache_share_calculation(self):
        output = meta.report(self._records())
        # total fresh_input=1400, cache_read=600, total_input=2000
        # cache_read share = 600/2000 = 30.0%
        self.assertIn("30.0%", output)

    def test_caveat_line_present(self):
        output = meta.report(self._records())
        self.assertIn("self-selected", output)

    def test_combined_section_present(self):
        output = meta.report(self._records())
        self.assertIn("Combined", output)

    def test_opportunity_table_present(self):
        output = meta.report(self._records())
        # opportunity table header
        self.assertIn("Ins-only ceiling", output)

    def test_model_in_top_models(self):
        output = meta.report(self._records())
        self.assertIn("gpt-5.5", output)


class TestReportTwoMachines(unittest.TestCase):
    """Two machines; verify per-machine isolation and combined totals."""

    def _records(self) -> list[dict]:
        return [
            _make_record(
                machine_id="mach-A",
                transcript_id="tx-1",
                cost_api_usd=10.0,
                fresh_input=1000,
                cache_read=0,
            ),
            _make_record(
                machine_id="mach-B",
                transcript_id="tx-2",
                cost_api_usd=20.0,
                fresh_input=0,
                cache_read=2000,
            ),
        ]

    def test_two_machine_header(self):
        output = meta.report(self._records())
        self.assertIn("2 machines", output)

    def test_both_machine_ids_present(self):
        output = meta.report(self._records())
        self.assertIn("mach-A", output)
        self.assertIn("mach-B", output)

    def test_combined_total_cost(self):
        output = meta.report(self._records())
        # total = $30.0000
        self.assertIn("30.0000", output)

    def test_mach_a_cache_share_zero(self):
        output = meta.report(self._records())
        # mach-A: 1000 fresh, 0 cache → 0.0%
        self.assertIn("0.0%", output)

    def test_mach_b_cache_share_100(self):
        output = meta.report(self._records())
        # mach-B: 0 fresh, 2000 cache → 100.0%
        self.assertIn("100.0%", output)

    def test_self_selected_caveat(self):
        output = meta.report(self._records())
        self.assertIn("self-selected", output)
        self.assertIn("not a random sample", output)


class TestReportNumbers(unittest.TestCase):
    """Verify combined cache-read share arithmetic on a controlled corpus."""

    def test_combined_cache_share_arithmetic(self):
        # 3 sessions: total fresh=300, total cache=700 → 70.0%
        records = [
            _make_record(machine_id="m1", transcript_id="t1",
                         fresh_input=100, cache_read=300),
            _make_record(machine_id="m1", transcript_id="t2",
                         fresh_input=100, cache_read=200),
            _make_record(machine_id="m2", transcript_id="t3",
                         fresh_input=100, cache_read=200),
        ]
        output = meta.report(records)
        self.assertIn("70.0%", output)

    def test_sessions_count_in_combined(self):
        records = [
            _make_record(machine_id="m1", transcript_id=f"t{i}")
            for i in range(5)
        ]
        output = meta.report(records)
        self.assertIn("5 sessions", output)


if __name__ == "__main__":
    unittest.main()
