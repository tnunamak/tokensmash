"""Tests for tokensmash.sessions.codex.

Uses synthetic fixtures under tests/fixtures/codex/; no real session content.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokensmash.schema import validate_session_record
from tokensmash.sessions.codex import iter_session_files, parse_session

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "codex"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fixture(name: str) -> Path:
    return FIXTURES / name


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# ---------------------------------------------------------------------------
# Normal session
# ---------------------------------------------------------------------------

class TestNormalSession(unittest.TestCase):
    def setUp(self) -> None:
        result = parse_session(_fixture("normal_session.jsonl"))
        self.assertIsNotNone(result, "parse_session must not return None for normal fixture")
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_valid(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_agent(self) -> None:
        self.assertEqual(self.record["agent"], "codex")

    def test_session_id_from_transcript(self) -> None:
        # Must use the internal id from payload, not the filename
        self.assertEqual(self.record["session_id"], "sess-codex-normal-001")

    def test_model(self) -> None:
        self.assertEqual(self.record["model"], "gpt-5.5")

    def test_agent_version(self) -> None:
        self.assertEqual(self.record.get("agent_version"), "0.139.0")

    def test_user_turns(self) -> None:
        # One user_message event in the fixture
        self.assertEqual(self.record["user_turns"], 1)

    def test_compactions(self) -> None:
        self.assertEqual(self.record["compactions"], 0)

    def test_parse_errors(self) -> None:
        self.assertEqual(self.record["parse_errors"], 0)

    def test_canonical_usage_session_level(self) -> None:
        """Session-level usage = final cumulative snapshot translated to canonical."""
        # Fixture has 3 token_count events with cumulative totals.
        # Final: input=29000, cached=18000, output=420, reasoning=10, total=29420
        # fresh_input = 29000 - 18000 = 11000
        # cache_read  = 18000
        # cache_write = 0
        # output      = 420
        # reasoning_output = 10
        usage = self.record["usage"]
        self.assertEqual(usage["fresh_input"], 11000)
        self.assertEqual(usage["cache_read"], 18000)
        self.assertEqual(usage["cache_write"], 0)
        self.assertEqual(usage["output"], 420)
        self.assertEqual(usage["reasoning_output"], 10)

    def test_provider_raw(self) -> None:
        pr = self.record["provider_raw"]
        self.assertEqual(pr["input_tokens"], 29000)
        self.assertEqual(pr["cached_input_tokens"], 18000)
        self.assertEqual(pr["output_tokens"], 420)
        self.assertEqual(pr["reasoning_output_tokens"], 10)
        self.assertEqual(pr["total_tokens"], 29420)

    def test_timeline_has_requests(self) -> None:
        requests = [e for e in self.timeline if e["kind"] == "request"]
        # 3 token_count events → 3 request events
        self.assertEqual(len(requests), 3)

    def test_request_indices_are_sequential(self) -> None:
        requests = [e for e in self.timeline if e["kind"] == "request"]
        indices = [e["index"] for e in requests]
        self.assertEqual(indices, list(range(len(requests))))

    def test_request_delta_first(self) -> None:
        """First request delta = first cumulative snapshot (no prior)."""
        req0 = next(e for e in self.timeline if e["kind"] == "request" and e["index"] == 0)
        u = req0["usage"]
        # First snapshot: input=8000, cached=1000, output=120, reasoning=0
        self.assertEqual(u["fresh_input"], 7000)   # 8000 - 1000
        self.assertEqual(u["cache_read"], 1000)
        self.assertEqual(u["cache_write"], 0)
        self.assertEqual(u["output"], 120)
        self.assertIsNone(u["reasoning_output"])    # 0 → None

    def test_request_delta_second(self) -> None:
        """Second request delta = diff between snapshot 1 and 2."""
        req1 = next(e for e in self.timeline if e["kind"] == "request" and e["index"] == 1)
        u = req1["usage"]
        # Snapshot 2: input=18500, cached=7500, output=290, reasoning=5
        # Snapshot 1: input=8000,  cached=1000, output=120, reasoning=0
        # Delta:      input=10500, cached=6500, output=170, reasoning=5
        # fresh_input = 10500 - 6500 = 4000
        self.assertEqual(u["fresh_input"], 4000)
        self.assertEqual(u["cache_read"], 6500)
        self.assertEqual(u["output"], 170)
        self.assertEqual(u["reasoning_output"], 5)

    def test_request_delta_third(self) -> None:
        """Third request delta = diff between snapshot 2 and 3."""
        req2 = next(e for e in self.timeline if e["kind"] == "request" and e["index"] == 2)
        u = req2["usage"]
        # Snapshot 3: input=29000, cached=18000, output=420, reasoning=10
        # Snapshot 2: input=18500, cached=7500,  output=290, reasoning=5
        # Delta:      input=10500, cached=10500,  output=130, reasoning=5
        # fresh_input = 10500 - 10500 = 0
        self.assertEqual(u["fresh_input"], 0)
        self.assertEqual(u["cache_read"], 10500)
        self.assertEqual(u["output"], 130)
        self.assertEqual(u["reasoning_output"], 5)

    def test_tool_output_events_present(self) -> None:
        tool_evs = [e for e in self.timeline if e["kind"] == "tool_output"]
        self.assertGreater(len(tool_evs), 0)

    def test_tool_output_categorisation(self) -> None:
        tool_evs = {e["tool_name"]: e for e in self.timeline if e["kind"] == "tool_output"}
        # exec_command with "cat helpers.py" → file_read
        cat_ev = next(
            (e for e in self.timeline
             if e["kind"] == "tool_output" and e.get("tool_name") == "exec_command"
             and e.get("category") == "file_read"),
            None,
        )
        # exec_command with "grep -n" → search
        grep_ev = next(
            (e for e in self.timeline
             if e["kind"] == "tool_output" and e.get("tool_name") == "exec_command"
             and e.get("category") == "search"),
            None,
        )
        # exec_command with "python -m pytest" → shell
        shell_ev = next(
            (e for e in self.timeline
             if e["kind"] == "tool_output" and e.get("tool_name") == "exec_command"
             and e.get("category") == "shell"),
            None,
        )
        # apply_patch → shell
        patch_ev = next(
            (e for e in self.timeline
             if e["kind"] == "tool_output" and e.get("tool_name") == "apply_patch"),
            None,
        )
        self.assertIsNotNone(cat_ev, "Expected a file_read categorised exec_command tool output")
        self.assertIsNotNone(grep_ev, "Expected a search categorised exec_command tool output")
        self.assertIsNotNone(shell_ev, "Expected a shell categorised exec_command tool output")
        self.assertIsNotNone(patch_ev, "Expected an apply_patch tool output")
        if patch_ev:
            self.assertEqual(patch_ev["category"], "shell")

    def test_tokens_est_formula(self) -> None:
        """tokens_est = max(1, len(output_text)//4)."""
        for ev in self.timeline:
            if ev["kind"] == "tool_output":
                self.assertGreaterEqual(ev["tokens_est"], 1)

    def test_no_anomalies(self) -> None:
        self.assertNotIn("anomalies", self.record)

    def test_duration_ms_positive(self) -> None:
        self.assertGreater(self.record["duration_ms"], 0)


# ---------------------------------------------------------------------------
# Restart session (negative delta / cumulative reset)
# ---------------------------------------------------------------------------

class TestRestartSession(unittest.TestCase):
    def setUp(self) -> None:
        result = parse_session(_fixture("restart_session.jsonl"))
        self.assertIsNotNone(result)
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_valid(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_session_id(self) -> None:
        self.assertEqual(self.record["session_id"], "sess-codex-restart-002")

    def test_model(self) -> None:
        self.assertEqual(self.record["model"], "gpt-5.4")

    def test_anomalies_recorded(self) -> None:
        """The fixture has a deliberate cumulative reset (negative delta)."""
        # Snapshot seq: 15200, 28400, 5080 (reset!), 16180
        # The drop from 28400 to 5080 should produce anomaly notes.
        self.assertIn("anomalies", self.record)
        self.assertGreater(len(self.record["anomalies"]), 0)

    def test_negative_deltas_clamped_to_zero(self) -> None:
        """All request usage fields must be >= 0 (negatives clamped)."""
        for ev in self.timeline:
            if ev["kind"] == "request":
                for k, v in ev["usage"].items():
                    if v is not None:
                        self.assertGreaterEqual(v, 0, f"Negative value for {k}: {v}")

    def test_final_usage_is_last_snapshot(self) -> None:
        """Session-level usage = final cumulative snapshot (last, not largest)."""
        # Final snapshot: input=16000, cached=4000, output=180, reasoning=0
        usage = self.record["usage"]
        self.assertEqual(usage["fresh_input"], 12000)   # 16000 - 4000
        self.assertEqual(usage["cache_read"], 4000)
        self.assertEqual(usage["output"], 180)


# ---------------------------------------------------------------------------
# Corrupt lines session
# ---------------------------------------------------------------------------

class TestCorruptLinesSession(unittest.TestCase):
    def setUp(self) -> None:
        result = parse_session(_fixture("corrupt_lines_session.jsonl"))
        self.assertIsNotNone(result)
        self.record, self.timeline = result  # type: ignore[misc]

    def test_schema_valid(self) -> None:
        errors = validate_session_record(self.record)
        self.assertEqual(errors, [], f"Validation errors: {errors}")

    def test_parse_errors_counted(self) -> None:
        # Two corrupt lines in the fixture
        self.assertEqual(self.record["parse_errors"], 2)

    def test_still_parses_usage(self) -> None:
        # One token_count event survives; usage should be non-zero
        usage = self.record["usage"]
        self.assertGreater(usage["fresh_input"] + usage["cache_read"] + usage["output"], 0)

    def test_does_not_crash(self) -> None:
        # Just getting here means no exception was raised
        self.assertIsNotNone(self.record)


# ---------------------------------------------------------------------------
# No-usage session (should return None)
# ---------------------------------------------------------------------------

class TestNoUsageSession(unittest.TestCase):
    def test_returns_none(self) -> None:
        result = parse_session(_fixture("no_usage_session.jsonl"))
        self.assertIsNone(result, "parse_session must return None when no token_count events present")


# ---------------------------------------------------------------------------
# Synthetic in-memory fixtures for edge cases
# ---------------------------------------------------------------------------

class TestSyntheticFixtures(unittest.TestCase):
    def _write_session(self, tmp_dir: str, rows: list[dict]) -> Path:
        p = Path(tmp_dir) / "session.jsonl"
        _write_jsonl(p, rows)
        return p

    def test_mcp_tool_categorisation(self) -> None:
        """ctx_ and mcp__-prefixed tool names → mcp category."""
        with TemporaryDirectory() as tmp:
            p = self._write_session(tmp, [
                {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta",
                 "payload": {"id": "syn-001", "cwd": "/tmp/proj", "originator": "codex_exec",
                             "cli_version": "0.100.0", "source": "exec",
                             "thread_source": "user", "model_provider": "openai",
                             "base_instructions": {"text": "You are Codex."}}},
                {"timestamp": "2026-01-01T00:00:01.000Z", "type": "turn_context",
                 "payload": {"turn_id": "t1", "cwd": "/tmp/proj", "workspace_roots": [],
                             "current_date": "2026-01-01", "timezone": "UTC",
                             "approval_policy": "never",
                             "sandbox_policy": {"type": "workspace-write"}, "model": "gpt-5.5"}},
                {"timestamp": "2026-01-01T00:00:02.000Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": "Do something."}},
                {"timestamp": "2026-01-01T00:00:03.000Z", "type": "response_item",
                 "payload": {"type": "function_call", "name": "ctx_search",
                             "arguments": "{}", "call_id": "c1"}},
                {"timestamp": "2026-01-01T00:00:04.000Z", "type": "response_item",
                 "payload": {"type": "function_call_output", "call_id": "c1",
                             "output": "search result text " * 20}},
                {"timestamp": "2026-01-01T00:00:05.000Z", "type": "response_item",
                 "payload": {"type": "function_call", "name": "mcp__amplitude__search",
                             "arguments": "{}", "call_id": "c2"}},
                {"timestamp": "2026-01-01T00:00:06.000Z", "type": "response_item",
                 "payload": {"type": "function_call_output", "call_id": "c2",
                             "output": "amplitude search result " * 10}},
                {"timestamp": "2026-01-01T00:00:07.000Z", "type": "event_msg",
                 "payload": {"type": "token_count", "info": {"total_token_usage": {
                     "input_tokens": 5000, "cached_input_tokens": 1000,
                     "output_tokens": 100, "reasoning_output_tokens": 0,
                     "total_tokens": 5100}}}},
            ])
            result = parse_session(p)
            self.assertIsNotNone(result)
            _, timeline = result  # type: ignore[misc]
            tool_evs = {e["tool_name"]: e for e in timeline if e["kind"] == "tool_output"}
            self.assertEqual(tool_evs["ctx_search"]["category"], "mcp")
            self.assertEqual(tool_evs["mcp__amplitude__search"]["category"], "mcp")

    def test_web_tool_categorisation(self) -> None:
        """web_url_read and browser_ tools → web category."""
        with TemporaryDirectory() as tmp:
            p = self._write_session(tmp, [
                {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta",
                 "payload": {"id": "syn-002", "cwd": "/tmp/proj", "originator": "codex_exec",
                             "cli_version": "0.100.0", "source": "exec",
                             "thread_source": "user", "model_provider": "openai",
                             "base_instructions": {"text": "You are Codex."}}},
                {"timestamp": "2026-01-01T00:00:01.000Z", "type": "turn_context",
                 "payload": {"turn_id": "t2", "cwd": "/tmp/proj", "workspace_roots": [],
                             "current_date": "2026-01-01", "timezone": "UTC",
                             "approval_policy": "never",
                             "sandbox_policy": {"type": "workspace-write"}, "model": "gpt-5.5"}},
                {"timestamp": "2026-01-01T00:00:02.000Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": "Fetch the docs."}},
                {"timestamp": "2026-01-01T00:00:03.000Z", "type": "response_item",
                 "payload": {"type": "function_call", "name": "web_url_read",
                             "arguments": "{\"url\": \"https://example.com\"}", "call_id": "c3"}},
                {"timestamp": "2026-01-01T00:00:04.000Z", "type": "response_item",
                 "payload": {"type": "function_call_output", "call_id": "c3",
                             "output": "Example Domain " * 30}},
                {"timestamp": "2026-01-01T00:00:05.000Z", "type": "response_item",
                 "payload": {"type": "function_call", "name": "browser_snapshot",
                             "arguments": "{}", "call_id": "c4"}},
                {"timestamp": "2026-01-01T00:00:06.000Z", "type": "response_item",
                 "payload": {"type": "function_call_output", "call_id": "c4",
                             "output": "browser snapshot data " * 20}},
                {"timestamp": "2026-01-01T00:00:07.000Z", "type": "event_msg",
                 "payload": {"type": "token_count", "info": {"total_token_usage": {
                     "input_tokens": 6000, "cached_input_tokens": 500,
                     "output_tokens": 150, "reasoning_output_tokens": 0,
                     "total_tokens": 6150}}}},
            ])
            result = parse_session(p)
            self.assertIsNotNone(result)
            _, timeline = result  # type: ignore[misc]
            tool_evs = {e["tool_name"]: e for e in timeline if e["kind"] == "tool_output"}
            self.assertEqual(tool_evs["web_url_read"]["category"], "web")
            self.assertEqual(tool_evs["browser_snapshot"]["category"], "web")

    def test_tokens_est_is_max_1_len_div_4(self) -> None:
        """tokens_est = max(1, len(output)//4)."""
        with TemporaryDirectory() as tmp:
            output_text = "x" * 400   # len=400 → 400//4 = 100
            p = self._write_session(tmp, [
                {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta",
                 "payload": {"id": "syn-003", "cwd": "/tmp/proj", "originator": "codex_exec",
                             "cli_version": "0.100.0", "source": "exec",
                             "thread_source": "user", "model_provider": "openai",
                             "base_instructions": {"text": "You are Codex."}}},
                {"timestamp": "2026-01-01T00:00:01.000Z", "type": "turn_context",
                 "payload": {"turn_id": "t3", "cwd": "/tmp/proj", "workspace_roots": [],
                             "current_date": "2026-01-01", "timezone": "UTC",
                             "approval_policy": "never",
                             "sandbox_policy": {"type": "workspace-write"}, "model": "gpt-5.5"}},
                {"timestamp": "2026-01-01T00:00:02.000Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": "Run something."}},
                {"timestamp": "2026-01-01T00:00:03.000Z", "type": "response_item",
                 "payload": {"type": "function_call", "name": "exec_command",
                             "arguments": "{\"cmd\": \"echo hello\"}", "call_id": "c5"}},
                {"timestamp": "2026-01-01T00:00:04.000Z", "type": "response_item",
                 "payload": {"type": "function_call_output", "call_id": "c5",
                             "output": output_text}},
                {"timestamp": "2026-01-01T00:00:05.000Z", "type": "event_msg",
                 "payload": {"type": "token_count", "info": {"total_token_usage": {
                     "input_tokens": 1000, "cached_input_tokens": 0,
                     "output_tokens": 50, "reasoning_output_tokens": 0,
                     "total_tokens": 1050}}}},
            ])
            result = parse_session(p)
            self.assertIsNotNone(result)
            _, timeline = result  # type: ignore[misc]
            tool_evs = [e for e in timeline if e["kind"] == "tool_output"]
            self.assertEqual(len(tool_evs), 1)
            self.assertEqual(tool_evs[0]["tokens_est"], 100)

    def test_tokens_est_minimum_is_1(self) -> None:
        """Empty output → tokens_est = 1."""
        with TemporaryDirectory() as tmp:
            p = self._write_session(tmp, [
                {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta",
                 "payload": {"id": "syn-004", "cwd": "/tmp/proj", "originator": "codex_exec",
                             "cli_version": "0.100.0", "source": "exec",
                             "thread_source": "user", "model_provider": "openai",
                             "base_instructions": {"text": "You are Codex."}}},
                {"timestamp": "2026-01-01T00:00:01.000Z", "type": "turn_context",
                 "payload": {"turn_id": "t4", "cwd": "/tmp/proj", "workspace_roots": [],
                             "current_date": "2026-01-01", "timezone": "UTC",
                             "approval_policy": "never",
                             "sandbox_policy": {"type": "workspace-write"}, "model": "gpt-5.5"}},
                {"timestamp": "2026-01-01T00:00:02.000Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": "Noop."}},
                {"timestamp": "2026-01-01T00:00:03.000Z", "type": "response_item",
                 "payload": {"type": "function_call", "name": "exec_command",
                             "arguments": "{\"cmd\": \"true\"}", "call_id": "c6"}},
                {"timestamp": "2026-01-01T00:00:04.000Z", "type": "response_item",
                 "payload": {"type": "function_call_output", "call_id": "c6",
                             "output": ""}},
                {"timestamp": "2026-01-01T00:00:05.000Z", "type": "event_msg",
                 "payload": {"type": "token_count", "info": {"total_token_usage": {
                     "input_tokens": 500, "cached_input_tokens": 0,
                     "output_tokens": 20, "reasoning_output_tokens": 0,
                     "total_tokens": 520}}}},
            ])
            result = parse_session(p)
            self.assertIsNotNone(result)
            _, timeline = result  # type: ignore[misc]
            tool_evs = [e for e in timeline if e["kind"] == "tool_output"]
            self.assertEqual(tool_evs[0]["tokens_est"], 1)

    def test_unknown_event_types_skipped_silently(self) -> None:
        """Unknown outer/payload types must not crash the parser."""
        with TemporaryDirectory() as tmp:
            p = self._write_session(tmp, [
                {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta",
                 "payload": {"id": "syn-005", "cwd": "/tmp/proj", "originator": "codex_exec",
                             "cli_version": "0.100.0", "source": "exec",
                             "thread_source": "user", "model_provider": "openai",
                             "base_instructions": {"text": "You are Codex."}}},
                {"timestamp": "2026-01-01T00:00:01.000Z", "type": "turn_context",
                 "payload": {"turn_id": "t5", "cwd": "/tmp/proj", "workspace_roots": [],
                             "current_date": "2026-01-01", "timezone": "UTC",
                             "approval_policy": "never",
                             "sandbox_policy": {"type": "workspace-write"}, "model": "gpt-5.5"}},
                # Unknown outer type
                {"timestamp": "2026-01-01T00:00:01.500Z", "type": "future_event_type",
                 "payload": {"some_field": "some_value"}},
                # Unknown payload type under event_msg
                {"timestamp": "2026-01-01T00:00:01.700Z", "type": "event_msg",
                 "payload": {"type": "future_inner_type", "data": 42}},
                {"timestamp": "2026-01-01T00:00:02.000Z", "type": "event_msg",
                 "payload": {"type": "user_message", "message": "Test."}},
                {"timestamp": "2026-01-01T00:00:03.000Z", "type": "event_msg",
                 "payload": {"type": "token_count", "info": {"total_token_usage": {
                     "input_tokens": 1000, "cached_input_tokens": 0,
                     "output_tokens": 30, "reasoning_output_tokens": 0,
                     "total_tokens": 1030}}}},
            ])
            result = parse_session(p)
            self.assertIsNotNone(result)
            record, _ = result  # type: ignore[misc]
            self.assertEqual(record["parse_errors"], 0)


# ---------------------------------------------------------------------------
# iter_session_files
# ---------------------------------------------------------------------------

class TestIterSessionFiles(unittest.TestCase):
    def test_yields_jsonl_files_newest_last(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Simulate Codex directory layout
            (root / "2025" / "11").mkdir(parents=True)
            (root / "2026" / "06").mkdir(parents=True)
            a = root / "2025" / "11" / "rollout-2025-11-01T10-00-00-aaaa.jsonl"
            b = root / "2026" / "06" / "rollout-2026-06-01T10-00-00-bbbb.jsonl"
            a.write_text("{}\n")
            b.write_text("{}\n")

            files = list(iter_session_files(root))
            self.assertEqual(len(files), 2)
            # newest last → a before b
            self.assertEqual(files[0].name, a.name)
            self.assertEqual(files[1].name, b.name)

    def test_non_existent_root_yields_nothing(self) -> None:
        files = list(iter_session_files(Path("/nonexistent/path/xyz")))
        self.assertEqual(files, [])

    def test_yields_from_real_fixtures_dir(self) -> None:
        files = list(iter_session_files(FIXTURES))
        self.assertGreater(len(files), 0)
        for f in files:
            self.assertTrue(f.suffix == ".jsonl")


if __name__ == "__main__":
    unittest.main()
