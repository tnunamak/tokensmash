"""Tests for tokensmash.store.

Covers:
- append: writes a canonical JSON line
- load_latest: deduplication (last wins), corrupt-line skipping, missing file
- upsert_many: added/replaced counts, idempotency
- export_scrubbed: transcript_path dropped, absolute-path values dropped,
  forbidden keys refused, validate_session_record gate, returns count
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tokensmash.schema as schema
from tokensmash import store


# ---------------------------------------------------------------------------
# Minimal valid session record factory
# ---------------------------------------------------------------------------


def _make_record(
    agent: str = "codex",
    session_id: str = "sess-001",
    model: str = "gpt-5.5",
    **extra,
) -> dict:
    rec = {
        "schema": schema.SESSION_SCHEMA,
        "agent": agent,
        "session_id": session_id,
        "machine_id": "aabbccdd11223344",
        "started_at": "2026-01-01T10:00:00+00:00",
        "model": model,
        "repo_id": "deadbeefcafe0001",
        "user_turns": 1,
        "tool_calls": 2,
        "compactions": 0,
        "duration_ms": 5000,
        "usage": {
            "fresh_input": 100,
            "cache_read": 50,
            "cache_write": 0,
            "output": 80,
            "reasoning_output": None,
        },
        "provider_raw": {"input_tokens": 150, "output_tokens": 80},
    }
    rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


class TestAppend(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._path = Path(self._tmp) / "sessions.jsonl"

    def test_creates_file(self):
        store.append(self._path, _make_record())
        self.assertTrue(self._path.exists())

    def test_writes_one_line(self):
        store.append(self._path, _make_record())
        lines = self._path.read_text().splitlines()
        self.assertEqual(len(lines), 1)

    def test_line_is_valid_json(self):
        rec = _make_record()
        store.append(self._path, rec)
        loaded = json.loads(self._path.read_text().strip())
        self.assertIsInstance(loaded, dict)

    def test_multiple_appends(self):
        store.append(self._path, _make_record(session_id="s1"))
        store.append(self._path, _make_record(session_id="s2"))
        lines = self._path.read_text().splitlines()
        self.assertEqual(len(lines), 2)

    def test_creates_parent_dirs(self):
        deep_path = Path(self._tmp) / "a" / "b" / "c.jsonl"
        store.append(deep_path, _make_record())
        self.assertTrue(deep_path.exists())

    def test_canonical_sorted_keys(self):
        rec = _make_record()
        store.append(self._path, rec)
        line = self._path.read_text().strip()
        parsed = json.loads(line)
        keys = list(parsed.keys())
        self.assertEqual(keys, sorted(keys))


# ---------------------------------------------------------------------------
# load_latest
# ---------------------------------------------------------------------------


class TestLoadLatest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._path = Path(self._tmp) / "sessions.jsonl"

    def test_returns_empty_dict_when_file_missing(self):
        result = store.load_latest(self._path)
        self.assertEqual(result, {})

    def test_single_record(self):
        rec = _make_record(session_id="s1")
        store.append(self._path, rec)
        result = store.load_latest(self._path)
        self.assertIn(("codex", "s1"), result)

    def test_deduplication_last_wins(self):
        r1 = _make_record(session_id="s1", user_turns=1)
        r2 = _make_record(session_id="s1", user_turns=99)
        store.append(self._path, r1)
        store.append(self._path, r2)
        result = store.load_latest(self._path)
        self.assertEqual(result[("codex", "s1")]["user_turns"], 99)

    def test_corrupt_lines_skipped(self):
        store.append(self._path, _make_record(session_id="s1"))
        with self._path.open("a") as fh:
            fh.write("this is not json }{{\n")
        store.append(self._path, _make_record(session_id="s2"))
        result = store.load_latest(self._path)
        self.assertIn(("codex", "s1"), result)
        self.assertIn(("codex", "s2"), result)
        self.assertEqual(len(result), 2)

    def test_non_dict_json_skipped(self):
        with self._path.open("w") as fh:
            fh.write("[1, 2, 3]\n")
        result = store.load_latest(self._path)
        self.assertEqual(result, {})

    def test_multiple_agents_distinct_keys(self):
        store.append(self._path, _make_record(agent="codex", session_id="s1"))
        store.append(self._path, _make_record(agent="claude-code", session_id="s1"))
        result = store.load_latest(self._path)
        self.assertIn(("codex", "s1"), result)
        self.assertIn(("claude-code", "s1"), result)
        self.assertEqual(len(result), 2)

    def test_custom_key(self):
        store.append(self._path, _make_record(session_id="s1"))
        result = store.load_latest(self._path, key=("session_id",))
        self.assertIn(("s1",), result)


# ---------------------------------------------------------------------------
# upsert_many
# ---------------------------------------------------------------------------


class TestUpsertMany(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._path = Path(self._tmp) / "sessions.jsonl"

    def test_new_record_counted_as_added(self):
        added, replaced = store.upsert_many(self._path, [_make_record(session_id="s1")])
        self.assertEqual(added, 1)
        self.assertEqual(replaced, 0)

    def test_identical_record_not_rewritten(self):
        rec = _make_record(session_id="s1")
        store.upsert_many(self._path, [rec])
        added2, replaced2 = store.upsert_many(self._path, [rec])
        self.assertEqual(added2, 0)
        self.assertEqual(replaced2, 0)

    def test_idempotency_line_count(self):
        rec = _make_record(session_id="s1")
        store.upsert_many(self._path, [rec])
        store.upsert_many(self._path, [rec])
        lines = self._path.read_text().splitlines()
        self.assertEqual(len(lines), 1)

    def test_changed_record_counted_as_replaced(self):
        r1 = _make_record(session_id="s1", user_turns=1)
        r2 = _make_record(session_id="s1", user_turns=5)
        store.upsert_many(self._path, [r1])
        added, replaced = store.upsert_many(self._path, [r2])
        self.assertEqual(added, 0)
        self.assertEqual(replaced, 1)

    def test_replaced_record_shadows_previous(self):
        r1 = _make_record(session_id="s1", user_turns=1)
        r2 = _make_record(session_id="s1", user_turns=99)
        store.upsert_many(self._path, [r1])
        store.upsert_many(self._path, [r2])
        result = store.load_latest(self._path)
        self.assertEqual(result[("codex", "s1")]["user_turns"], 99)

    def test_multiple_records_mixed(self):
        r1 = _make_record(session_id="s1")
        r2 = _make_record(session_id="s2")
        r3 = _make_record(session_id="s3")
        store.upsert_many(self._path, [r1, r2])
        # r1 changed, r2 unchanged, r3 new
        r1_changed = _make_record(session_id="s1", user_turns=99)
        added, replaced = store.upsert_many(self._path, [r1_changed, r2, r3])
        self.assertEqual(added, 1)
        self.assertEqual(replaced, 1)

    def test_empty_records_list(self):
        added, replaced = store.upsert_many(self._path, [])
        self.assertEqual(added, 0)
        self.assertEqual(replaced, 0)


# ---------------------------------------------------------------------------
# export_scrubbed
# ---------------------------------------------------------------------------


class TestExportScrubbed(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._src = Path(self._tmp) / "sessions.jsonl"
        self._out = Path(self._tmp) / "scrubbed.jsonl"

    def _write_record(self, rec: dict) -> None:
        store.append(self._src, rec)

    def _read_out(self) -> list[dict]:
        if not self._out.exists():
            return []
        return [json.loads(l) for l in self._out.read_text().splitlines() if l.strip()]

    def test_returns_count(self):
        self._write_record(_make_record(session_id="s1"))
        count = store.export_scrubbed(self._src, self._out)
        self.assertEqual(count, 1)

    def test_transcript_path_dropped(self):
        rec = _make_record(session_id="s1", transcript_path="/home/user/t.jsonl")
        self._write_record(rec)
        store.export_scrubbed(self._src, self._out)
        out_recs = self._read_out()
        self.assertEqual(len(out_recs), 1)
        self.assertNotIn("transcript_path", out_recs[0])

    def test_absolute_path_string_dropped(self):
        rec = _make_record(session_id="s1", some_path="/home/user/project")
        self._write_record(rec)
        store.export_scrubbed(self._src, self._out)
        out_recs = self._read_out()
        self.assertNotIn("some_path", out_recs[0])

    def test_non_path_string_kept(self):
        rec = _make_record(session_id="s1", tag="deadbeef1234")
        self._write_record(rec)
        store.export_scrubbed(self._src, self._out)
        out_recs = self._read_out()
        self.assertEqual(out_recs[0].get("tag"), "deadbeef1234")

    def test_hashes_kept(self):
        rec = _make_record(session_id="s1")
        self._write_record(rec)
        store.export_scrubbed(self._src, self._out)
        out_recs = self._read_out()
        self.assertEqual(out_recs[0]["repo_id"], rec["repo_id"])
        self.assertEqual(out_recs[0]["machine_id"], rec["machine_id"])

    def test_nested_absolute_path_dropped(self):
        rec = _make_record(
            session_id="s1",
            meta={"project_path": "/home/user/project", "name": "hello"},
        )
        self._write_record(rec)
        store.export_scrubbed(self._src, self._out)
        out_recs = self._read_out()
        meta = out_recs[0].get("meta", {})
        self.assertNotIn("project_path", meta)
        self.assertEqual(meta.get("name"), "hello")

    def test_forbidden_key_prompt_raises(self):
        rec = _make_record(session_id="s1", prompt="do the thing")
        self._write_record(rec)
        with self.assertRaises(ValueError):
            store.export_scrubbed(self._src, self._out)

    def test_forbidden_key_text_raises(self):
        rec = _make_record(session_id="s1", text="some raw text")
        self._write_record(rec)
        with self.assertRaises(ValueError):
            store.export_scrubbed(self._src, self._out)

    def test_forbidden_key_content_raises(self):
        rec = _make_record(session_id="s1", content="raw content")
        self._write_record(rec)
        with self.assertRaises(ValueError):
            store.export_scrubbed(self._src, self._out)

    def test_invalid_record_skipped(self):
        # Write a record missing required keys; should be skipped not raise
        bad = {"schema": schema.SESSION_SCHEMA, "agent": "codex", "session_id": "bad"}
        store.append(self._src, bad)
        count = store.export_scrubbed(self._src, self._out)
        self.assertEqual(count, 0)

    def test_empty_store(self):
        self._src.write_text("")
        count = store.export_scrubbed(self._src, self._out)
        self.assertEqual(count, 0)

    def test_creates_output_parent_dirs(self):
        deep_out = Path(self._tmp) / "a" / "b" / "out.jsonl"
        self._write_record(_make_record(session_id="s1"))
        store.export_scrubbed(self._src, deep_out)
        self.assertTrue(deep_out.exists())


if __name__ == "__main__":
    unittest.main()
