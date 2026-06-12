"""Tests for tokensmash.sessions.claude — Claude Code transcript parser.

Fixtures live under tests/fixtures/claude/ and contain synthetic data with
realistic structure but fake content. No real session content is committed.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokensmash.schema import validate_session_record  # noqa: E402
from tokensmash.sessions.claude import iter_session_files, parse_session  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "claude"


class TestNormalSession(unittest.TestCase):
    """tests/fixtures/claude/normal_session.jsonl — a clean session with tool calls."""

    def setUp(self) -> None:
        result = parse_session(FIXTURES / "normal_session.jsonl")
        self.assertIsNotNone(result)
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_validates(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_agent(self) -> None:
        self.assertEqual(self.record["agent"], "claude-code")

    def test_session_id_from_transcript(self) -> None:
        self.assertEqual(self.record["session_id"], "sess-normal-0001")

    def test_model_extracted(self) -> None:
        self.assertEqual(self.record["model"], "claude-test-model")

    def test_agent_version(self) -> None:
        self.assertEqual(self.record["agent_version"], "2.1.100")

    def test_usage_sums_correctly(self) -> None:
        # 5 assistant entries with usage:
        # msg-0003: fresh=500, write=1000, read=200, out=80
        # msg-0005: fresh=620, write=0,    read=1200, out=60
        # msg-0007: fresh=700, write=0,    read=1800, out=40
        # msg-0009: fresh=800, write=0,    read=2000, out=100
        # msg-0011: fresh=900, write=0,    read=2200, out=20
        usage = self.record["usage"]
        self.assertEqual(usage["fresh_input"], 500 + 620 + 700 + 800 + 900)
        self.assertEqual(usage["cache_write"], 1000 + 0 + 0 + 0 + 0)
        self.assertEqual(usage["cache_read"], 200 + 1200 + 1800 + 2000 + 2200)
        self.assertEqual(usage["output"], 80 + 60 + 40 + 100 + 20)
        self.assertIsNone(usage["reasoning_output"])

    def test_user_turns_count_genuine_only(self) -> None:
        # 2 genuine user messages: "Hello, please help me..." and "Great, thanks!"
        # 3 tool_result-only user entries are NOT counted
        self.assertEqual(self.record["user_turns"], 2)

    def test_tool_calls_count(self) -> None:
        # 3 tool_use blocks: Read, Bash, mcp__headroom__headroom_compress
        self.assertEqual(self.record["tool_calls"], 3)

    def test_compactions_zero(self) -> None:
        self.assertEqual(self.record["compactions"], 0)

    def test_parse_errors_zero(self) -> None:
        self.assertEqual(self.record["parse_errors"], 0)

    def test_no_sidechain_flag(self) -> None:
        self.assertNotIn("sidechain", self.record)

    def test_timeline_has_request_events(self) -> None:
        request_events = [e for e in self.timeline if e["kind"] == "request"]
        self.assertEqual(len(request_events), 5)

    def test_timeline_request_indices_sequential(self) -> None:
        request_events = [e for e in self.timeline if e["kind"] == "request"]
        for i, ev in enumerate(request_events):
            self.assertEqual(ev["index"], i)

    def test_timeline_tool_output_events(self) -> None:
        tool_events = [e for e in self.timeline if e["kind"] == "tool_output"]
        self.assertEqual(len(tool_events), 3)

    def test_tool_categorization_file_read(self) -> None:
        tool_events = [e for e in self.timeline if e["kind"] == "tool_output"]
        read_event = next(e for e in tool_events if e["tool_name"] == "Read")
        self.assertEqual(read_event["category"], "file_read")

    def test_tool_categorization_shell(self) -> None:
        tool_events = [e for e in self.timeline if e["kind"] == "tool_output"]
        bash_event = next(e for e in tool_events if e["tool_name"] == "Bash")
        self.assertEqual(bash_event["category"], "shell")

    def test_tool_categorization_mcp(self) -> None:
        tool_events = [e for e in self.timeline if e["kind"] == "tool_output"]
        mcp_event = next(e for e in tool_events if e["tool_name"] == "mcp__headroom__headroom_compress")
        self.assertEqual(mcp_event["category"], "mcp")

    def test_tokens_est_at_least_one(self) -> None:
        tool_events = [e for e in self.timeline if e["kind"] == "tool_output"]
        for ev in tool_events:
            self.assertGreaterEqual(ev["tokens_est"], 1)

    def test_tokens_est_formula(self) -> None:
        # Read result: "def main():\n    print('hello world')\n    return 0\n\nif __name__ == '__main__':\n    main()\n"
        # len = 71, 71//4 = 17
        tool_events = [e for e in self.timeline if e["kind"] == "tool_output"]
        read_event = next(e for e in tool_events if e["tool_name"] == "Read")
        text = "def main():\n    print('hello world')\n    return 0\n\nif __name__ == '__main__':\n    main()\n"
        expected = max(1, len(text) // 4)
        self.assertEqual(read_event["tokens_est"], expected)

    def test_provider_raw_contains_raw_fields(self) -> None:
        raw = self.record["provider_raw"]
        # provider_raw should be the last assistant message's raw usage
        self.assertIn("input_tokens", raw)
        self.assertIn("output_tokens", raw)

    def test_repo_id_is_string(self) -> None:
        self.assertIsInstance(self.record["repo_id"], str)
        self.assertTrue(len(self.record["repo_id"]) > 0)

    def test_duration_ms_nonnegative(self) -> None:
        self.assertGreaterEqual(self.record["duration_ms"], 0)

    def test_started_at_iso_format(self) -> None:
        ts = self.record["started_at"]
        # Should be parseable as ISO-8601
        from datetime import datetime
        # Should not raise
        datetime.fromisoformat(ts)

    def test_request_event_usage_structure(self) -> None:
        request_events = [e for e in self.timeline if e["kind"] == "request"]
        first = request_events[0]
        usage = first["usage"]
        for key in ("fresh_input", "cache_read", "cache_write", "output", "reasoning_output"):
            self.assertIn(key, usage)


class TestStreamedDuplicates(unittest.TestCase):
    """tests/fixtures/claude/streamed_duplicates.jsonl — same message id twice, last wins."""

    def setUp(self) -> None:
        result = parse_session(FIXTURES / "streamed_duplicates.jsonl")
        self.assertIsNotNone(result)
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_validates(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_dedupe_last_wins_for_output_tokens(self) -> None:
        # msg-stream-dup appears twice:
        #   first:  output_tokens=50
        #   second: output_tokens=120  (last wins)
        # msg-stream-final: output_tokens=30
        # Total output = 120 + 30 = 150
        usage = self.record["usage"]
        self.assertEqual(usage["output"], 120 + 30)

    def test_dedupe_correct_request_count(self) -> None:
        # Only 2 unique assistant messages with usage (after dedup)
        request_events = [e for e in self.timeline if e["kind"] == "request"]
        self.assertEqual(len(request_events), 2)

    def test_usage_totals_use_last_entry(self) -> None:
        # msg-stream-dup last: fresh=300, write=500, read=0, out=120
        # msg-stream-final:    fresh=450, write=0,   read=800, out=30
        usage = self.record["usage"]
        self.assertEqual(usage["fresh_input"], 300 + 450)
        self.assertEqual(usage["cache_write"], 500 + 0)
        self.assertEqual(usage["cache_read"], 0 + 800)

    def test_user_turns_genuine_only(self) -> None:
        # 1 genuine user turn ("Run some analysis")
        # 1 tool_result-only user entry (not counted)
        self.assertEqual(self.record["user_turns"], 1)


class TestWithCompaction(unittest.TestCase):
    """tests/fixtures/claude/with_compaction.jsonl — session with one compaction boundary."""

    def setUp(self) -> None:
        result = parse_session(FIXTURES / "with_compaction.jsonl")
        self.assertIsNotNone(result)
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_validates(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_compactions_count(self) -> None:
        self.assertEqual(self.record["compactions"], 1)

    def test_usage_sums_all_three_assistants(self) -> None:
        # msg-c003: fresh=400, write=2000, read=0, out=60
        # msg-c006: fresh=200, write=1500, read=0, out=40
        # msg-c008: fresh=350, write=0,    read=1700, out=30
        usage = self.record["usage"]
        self.assertEqual(usage["fresh_input"], 400 + 200 + 350)
        self.assertEqual(usage["cache_write"], 2000 + 1500 + 0)
        self.assertEqual(usage["cache_read"], 0 + 0 + 1700)
        self.assertEqual(usage["output"], 60 + 40 + 30)

    def test_user_turns_genuine_only(self) -> None:
        # 1 genuine user turn ("Start a long task")
        # compaction summary entry is NOT counted as user_turns
        # 2 tool_result-only user entries are not counted
        self.assertEqual(self.record["user_turns"], 1)

    def test_compaction_entry_not_in_timeline_as_request(self) -> None:
        # The isCompactSummary entry should not produce a request event
        request_events = [e for e in self.timeline if e["kind"] == "request"]
        self.assertEqual(len(request_events), 3)  # 3 assistant messages


class TestCorruptLines(unittest.TestCase):
    """tests/fixtures/claude/corrupt_lines.jsonl — has 2 corrupt/invalid JSON lines."""

    def setUp(self) -> None:
        result = parse_session(FIXTURES / "corrupt_lines.jsonl")
        self.assertIsNotNone(result)
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_validates(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_parse_errors_counted(self) -> None:
        # 2 corrupt lines: "{INVALID JSON LINE - this is corrupt}" and "not json at all!"
        self.assertEqual(self.record["parse_errors"], 2)

    def test_valid_entries_still_parsed(self) -> None:
        # 1 assistant entry with usage should still be parsed
        usage = self.record["usage"]
        self.assertEqual(usage["fresh_input"], 300)
        self.assertEqual(usage["cache_write"], 100)
        self.assertEqual(usage["cache_read"], 50)
        self.assertEqual(usage["output"], 10)

    def test_unknown_entry_types_silently_skipped(self) -> None:
        # The "unknown-future-type" entry should not cause errors
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [])


class TestNoUsage(unittest.TestCase):
    """tests/fixtures/claude/no_usage.jsonl — assistant entry has no usage block."""

    def test_returns_none(self) -> None:
        result = parse_session(FIXTURES / "no_usage.jsonl")
        self.assertIsNone(result)


class TestSidechainSession(unittest.TestCase):
    """tests/fixtures/claude/sidechain_session.jsonl — all entries have isSidechain=true."""

    def setUp(self) -> None:
        result = parse_session(FIXTURES / "sidechain_session.jsonl")
        self.assertIsNotNone(result)
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_validates(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_sidechain_flag_set(self) -> None:
        self.assertTrue(self.record.get("sidechain"))

    def test_usage_still_parsed(self) -> None:
        usage = self.record["usage"]
        self.assertEqual(usage["fresh_input"], 100)
        self.assertEqual(usage["output"], 20)


class TestIterSessionFiles(unittest.TestCase):
    """iter_session_files should yield JSONL files newest-last."""

    def test_yields_jsonl_files_in_mtime_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Create two project subdirs with JSONL files
            proj1 = root / "proj-a"
            proj1.mkdir()
            proj2 = root / "proj-b"
            proj2.mkdir()

            # Write files with distinct mtimes
            import time

            f1 = proj1 / "session1.jsonl"
            f1.write_text("{}\n")
            time.sleep(0.02)
            f2 = proj2 / "session2.jsonl"
            f2.write_text("{}\n")

            files = list(iter_session_files(root))
            # Should have found both files
            self.assertEqual(len(files), 2)
            # Newest last: f1 is older, f2 is newer → f1 first, f2 last
            self.assertEqual(files[-1], f2)
            self.assertEqual(files[0], f1)

    def test_skips_non_jsonl_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subdir = root / "proj"
            subdir.mkdir()
            (subdir / "notes.txt").write_text("ignore me")
            (subdir / "session.jsonl").write_text("{}\n")
            files = list(iter_session_files(root))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].suffix, ".jsonl")

    def test_empty_root_yields_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            files = list(iter_session_files(Path(tmp)))
            self.assertEqual(files, [])

    def test_nonexistent_root_yields_nothing(self) -> None:
        files = list(iter_session_files(Path("/nonexistent/path/that/does/not/exist")))
        self.assertEqual(files, [])

    def test_fixture_files_are_found(self) -> None:
        # The fixtures dir has a flat structure — one level of subdirs not needed
        # so let's test against the actual claude fixtures parent
        fixture_parent = FIXTURES.parent
        # iter_session_files expects subdirs, so make a temp wrapper
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            subdir = root / "claude-proj"
            subdir.mkdir()
            for fixture in FIXTURES.glob("*.jsonl"):
                import shutil
                shutil.copy(fixture, subdir / fixture.name)
            files = list(iter_session_files(root))
            self.assertGreater(len(files), 0)


class TestToolCategories(unittest.TestCase):
    """Verify all required tool category mappings work via the parser."""

    def _make_session_with_tool(self, tool_name: str, result_text: str) -> list[dict]:
        """Build a minimal in-memory JSONL and parse it."""
        import io
        lines = [
            json.dumps({"type": "user", "uuid": "u1", "sessionId": "s1",
                        "timestamp": "2026-01-01T00:00:00.000Z",
                        "cwd": "/tmp", "version": "2.0", "isSidechain": False,
                        "message": {"role": "user", "content": "go"}}),
            json.dumps({"type": "assistant", "uuid": "a1", "sessionId": "s1",
                        "timestamp": "2026-01-01T00:00:01.000Z",
                        "cwd": "/tmp", "version": "2.0", "isSidechain": False,
                        "message": {"id": "m1", "role": "assistant", "model": "test",
                                    "stop_reason": "tool_use",
                                    "content": [{"type": "tool_use", "id": "tid1",
                                                 "name": tool_name, "input": {}}],
                                    "usage": {"input_tokens": 10,
                                              "cache_creation_input_tokens": 0,
                                              "cache_read_input_tokens": 0,
                                              "output_tokens": 5}}}),
            json.dumps({"type": "user", "uuid": "u2", "sessionId": "s1",
                        "timestamp": "2026-01-01T00:00:02.000Z",
                        "cwd": "/tmp", "version": "2.0", "isSidechain": False,
                        "message": {"role": "user",
                                    "content": [{"type": "tool_result",
                                                 "tool_use_id": "tid1",
                                                 "content": result_text}]}}),
        ]
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "sess.jsonl"
            p.write_text("\n".join(lines) + "\n")
            result = parse_session(p)
        return result[1] if result else []  # type: ignore[index]

    def test_grep_is_search(self) -> None:
        timeline = self._make_session_with_tool("Grep", "foo bar")
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["category"], "search")

    def test_glob_is_search(self) -> None:
        timeline = self._make_session_with_tool("Glob", "*.py")
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["category"], "search")

    def test_webfetch_is_web(self) -> None:
        timeline = self._make_session_with_tool("WebFetch", "page content here")
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["category"], "web")

    def test_websearch_is_web(self) -> None:
        timeline = self._make_session_with_tool("WebSearch", "search results")
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["category"], "web")

    def test_mcp_prefixed_is_mcp(self) -> None:
        timeline = self._make_session_with_tool("mcp__headroom__compress", "compressed")
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["category"], "mcp")

    def test_unknown_tool_is_other(self) -> None:
        timeline = self._make_session_with_tool("TaskCreate", "created task")
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["category"], "other")

    def test_tokens_est_minimum_one(self) -> None:
        # Empty tool result should still give tokens_est = 1
        timeline = self._make_session_with_tool("Bash", "")
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["tokens_est"], 1)

    def test_tokens_est_formula_applied(self) -> None:
        text = "x" * 400  # 400 chars -> 400//4 = 100
        timeline = self._make_session_with_tool("Bash", text)
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        self.assertEqual(tool_ev["tokens_est"], 100)


class TestEdgeCases(unittest.TestCase):
    """Edge cases: nonexistent file, empty file, entries with no uuid."""

    def test_nonexistent_file_returns_none(self) -> None:
        result = parse_session(Path("/nonexistent/file.jsonl"))
        self.assertIsNone(result)

    def test_empty_file_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "empty.jsonl"
            p.write_text("")
            result = parse_session(p)
            self.assertIsNone(result)

    def test_only_whitespace_returns_none(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "blank.jsonl"
            p.write_text("\n\n   \n")
            result = parse_session(p)
            self.assertIsNone(result)

    def test_session_id_falls_back_to_filename(self) -> None:
        """If no sessionId in entries, session_id = path.stem."""
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "fallback-session-id.jsonl"
            p.write_text(json.dumps({
                "type": "assistant",
                "uuid": "x1",
                "timestamp": "2026-01-01T00:00:00.000Z",
                "cwd": "/tmp",
                "version": "2.0",
                "isSidechain": False,
                "message": {
                    "id": "m1", "role": "assistant", "model": "claude-test",
                    "stop_reason": "end_turn", "content": [],
                    "usage": {"input_tokens": 5, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0, "output_tokens": 2},
                },
            }) + "\n")
            result = parse_session(p)
            self.assertIsNotNone(result)
            record, _ = result  # type: ignore[misc]
            self.assertEqual(record["session_id"], "fallback-session-id")

    def test_tool_result_with_list_content(self) -> None:
        """tool_result with list-of-text-blocks content is handled correctly."""
        lines = [
            json.dumps({"type": "user", "uuid": "u1", "sessionId": "s1",
                        "timestamp": "2026-01-01T00:00:00.000Z",
                        "cwd": "/tmp", "version": "2.0", "isSidechain": False,
                        "message": {"role": "user", "content": "go"}}),
            json.dumps({"type": "assistant", "uuid": "a1", "sessionId": "s1",
                        "timestamp": "2026-01-01T00:00:01.000Z",
                        "cwd": "/tmp", "version": "2.0", "isSidechain": False,
                        "message": {"id": "m1", "role": "assistant", "model": "test",
                                    "stop_reason": "tool_use",
                                    "content": [{"type": "tool_use", "id": "tid1",
                                                 "name": "Bash", "input": {}}],
                                    "usage": {"input_tokens": 10,
                                              "cache_creation_input_tokens": 0,
                                              "cache_read_input_tokens": 0,
                                              "output_tokens": 5}}}),
            json.dumps({"type": "user", "uuid": "u2", "sessionId": "s1",
                        "timestamp": "2026-01-01T00:00:02.000Z",
                        "cwd": "/tmp", "version": "2.0", "isSidechain": False,
                        "message": {"role": "user",
                                    "content": [{"type": "tool_result",
                                                 "tool_use_id": "tid1",
                                                 "content": [
                                                     {"type": "text", "text": "hello"},
                                                     {"type": "text", "text": " world"},
                                                 ]}]}}),
        ]
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "list_content.jsonl"
            p.write_text("\n".join(lines) + "\n")
            result = parse_session(p)
        self.assertIsNotNone(result)
        _, timeline = result  # type: ignore[misc]
        tool_ev = next(e for e in timeline if e["kind"] == "tool_output")
        # "hello world" = 11 chars -> 11//4 = 2
        self.assertEqual(tool_ev["tokens_est"], max(1, len("hello world") // 4))


if __name__ == "__main__":
    unittest.main()
