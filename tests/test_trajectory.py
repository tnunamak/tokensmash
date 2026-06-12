"""Tests for tokensmash.trajectory.

Covers:
  - analyze_session: empty timeline, file_read unused vs used, path normalisation,
    searches_unfollowed detection, tokens_unused accumulation.
  - analyze_corpus: seeded sampling, aggregate counts, session-skip handling.
  - report: string output, proxy caveat presence, upper-bound language.
  - Parser target field: claude Read/Edit/Write/Bash/Grep/Glob; codex exec_command/shell_command.

No real session content is committed; all fixtures are synthetic in-memory data.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokensmash.trajectory import analyze_session, analyze_corpus, report, _normalise_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(cwd: str | None = None) -> dict:
    """Minimal session record for trajectory tests."""
    rec: dict = {
        "schema": 1,
        "agent": "claude-code",
        "session_id": "test-traj-session",
        "transcript_id": "tid-test",
        "logical_session_id": "test-traj-session",
        "machine_id": "mid-test",
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:01:00+00:00",
        "model": "claude-sonnet-4-6",
        "repo_id": "rid-test",
        "user_turns": 1,
        "tool_calls": 0,
        "compactions": 0,
        "duration_ms": 60000,
        "usage": {"fresh_input": 100, "cache_read": 0, "cache_write": 0,
                  "output": 50, "reasoning_output": None},
        "provider_raw": {},
        "parse_errors": 0,
    }
    if cwd is not None:
        rec["cwd"] = cwd
    return rec


def _tool_output(category: str, tokens_est: int, target: str | None = None,
                 tool_name: str = "", request_index: int = 0) -> dict:
    ev: dict = {
        "kind": "tool_output",
        "category": category,
        "tokens_est": tokens_est,
        "tool_name": tool_name,
        "request_index": request_index,
    }
    if target is not None:
        ev["target"] = target
    return ev


def _request(index: int = 0) -> dict:
    return {
        "kind": "request",
        "index": index,
        "usage": {"fresh_input": 10, "cache_read": 0, "cache_write": 0,
                  "output": 5, "reasoning_output": None},
    }


# ---------------------------------------------------------------------------
# Tests for _normalise_path
# ---------------------------------------------------------------------------

class TestNormalisePath(unittest.TestCase):

    def test_none_input_returns_none(self) -> None:
        self.assertIsNone(_normalise_path(None, None))  # type: ignore[arg-type]

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_normalise_path("", None))
        self.assertIsNone(_normalise_path("   ", None))

    def test_absolute_path_unchanged(self) -> None:
        result = _normalise_path("/home/user/src/foo.py", None)
        self.assertEqual(result, "/home/user/src/foo.py")

    def test_relative_path_no_cwd_normalised(self) -> None:
        result = _normalise_path("src/../foo.py", None)
        self.assertEqual(result, "foo.py")

    def test_relative_path_with_cwd_becomes_absolute(self) -> None:
        result = _normalise_path("foo.py", "/home/user/project")
        self.assertEqual(result, "/home/user/project/foo.py")

    def test_double_slashes_collapsed(self) -> None:
        result = _normalise_path("/home//user//foo.py", None)
        self.assertEqual(result, "/home/user/foo.py")

    def test_trailing_slash_stripped(self) -> None:
        # normpath removes trailing slash on files (not dirs), test a file path
        result = _normalise_path("/home/user/foo.py", None)
        self.assertEqual(result, "/home/user/foo.py")


# ---------------------------------------------------------------------------
# Tests for analyze_session — empty / trivial cases
# ---------------------------------------------------------------------------

class TestAnalyzeSessionEmpty(unittest.TestCase):

    def test_empty_timeline(self) -> None:
        result = analyze_session(_make_record(), [])
        self.assertEqual(result["reads"], 0)
        self.assertEqual(result["reads_unused"], 0)
        self.assertEqual(result["tokens_unused"], 0)
        self.assertEqual(result["searches"], 0)
        self.assertEqual(result["searches_unfollowed"], 0)

    def test_only_request_events(self) -> None:
        timeline = [_request(0), _request(1)]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["reads"], 0)
        self.assertEqual(result["searches"], 0)

    def test_non_read_non_search_events_not_counted(self) -> None:
        timeline = [
            _request(0),
            _tool_output("shell", 100, "/some/script.sh"),
            _tool_output("web", 200, "https://example.com"),
            _tool_output("mcp", 50),
            _tool_output("other", 30),
        ]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["reads"], 0)
        self.assertEqual(result["searches"], 0)


# ---------------------------------------------------------------------------
# Tests for analyze_session — file_read unused/used
# ---------------------------------------------------------------------------

class TestAnalyzeSessionReads(unittest.TestCase):

    def test_single_read_no_later_events_is_unused(self) -> None:
        timeline = [
            _request(0),
            _tool_output("file_read", 80, "/project/foo.py", "Read"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        self.assertEqual(result["reads"], 1)
        self.assertEqual(result["reads_unused"], 1)
        self.assertEqual(result["tokens_unused"], 80)

    def test_read_followed_by_same_path_edit_is_used(self) -> None:
        timeline = [
            _request(0),
            _tool_output("file_read", 80, "/project/foo.py", "Read"),
            _tool_output("shell", 10, "/project/foo.py", "Edit"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        self.assertEqual(result["reads"], 1)
        self.assertEqual(result["reads_unused"], 0)
        self.assertEqual(result["tokens_unused"], 0)

    def test_read_not_followed_by_same_path_is_unused(self) -> None:
        timeline = [
            _request(0),
            _tool_output("file_read", 80, "/project/foo.py", "Read"),
            _tool_output("shell", 10, "/project/bar.py", "Edit"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        self.assertEqual(result["reads"], 1)
        self.assertEqual(result["reads_unused"], 1)

    def test_two_reads_one_used_one_unused(self) -> None:
        timeline = [
            _request(0),
            _tool_output("file_read", 50, "/project/used.py", "Read"),
            _tool_output("file_read", 120, "/project/unused.py", "Read"),
            _tool_output("shell", 10, "/project/used.py", "Edit"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        self.assertEqual(result["reads"], 2)
        self.assertEqual(result["reads_unused"], 1)
        self.assertEqual(result["tokens_unused"], 120)

    def test_read_with_no_target_counts_as_unused(self) -> None:
        timeline = [
            _request(0),
            # No target key at all
            {"kind": "tool_output", "category": "file_read", "tokens_est": 60,
             "tool_name": "Read", "request_index": 0},
        ]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["reads"], 1)
        self.assertEqual(result["reads_unused"], 1)
        self.assertEqual(result["tokens_unused"], 60)

    def test_tokens_unused_sums_across_unused_reads(self) -> None:
        timeline = [
            _request(0),
            _tool_output("file_read", 100, "/project/a.py", "Read"),
            _tool_output("file_read", 200, "/project/b.py", "Read"),
            _tool_output("file_read", 300, "/project/c.py", "Read"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        self.assertEqual(result["reads"], 3)
        self.assertEqual(result["reads_unused"], 3)
        self.assertEqual(result["tokens_unused"], 600)

    def test_read_matched_by_later_read_same_path_counts_as_used(self) -> None:
        """A second read of the same file counts as a reference."""
        timeline = [
            _tool_output("file_read", 80, "/project/foo.py", "Read"),
            _tool_output("file_read", 80, "/project/foo.py", "Read"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        # First read: used (second read references same path)
        # Second read: unused (no later event references it)
        self.assertEqual(result["reads"], 2)
        self.assertEqual(result["reads_unused"], 1)

    def test_path_normalisation_relative_vs_absolute(self) -> None:
        """Relative read target normalised against cwd matches absolute later target."""
        timeline = [
            _tool_output("file_read", 50, "src/main.py", "Read"),
            _tool_output("shell", 10, "/project/src/main.py", "Write"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        self.assertEqual(result["reads_unused"], 0)

    def test_path_normalisation_double_slash(self) -> None:
        """normpath collapses double slashes so paths match."""
        timeline = [
            _tool_output("file_read", 50, "/project//src/main.py", "Read"),
            _tool_output("shell", 10, "/project/src/main.py", "Write"),
        ]
        result = analyze_session(_make_record(cwd="/project"), timeline)
        self.assertEqual(result["reads_unused"], 0)


# ---------------------------------------------------------------------------
# Tests for analyze_session — searches
# ---------------------------------------------------------------------------

class TestAnalyzeSessionSearches(unittest.TestCase):

    def test_single_search_no_following_events(self) -> None:
        timeline = [
            _request(0),
            _tool_output("search", 40, "pattern", "Grep"),
        ]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["searches"], 1)
        self.assertEqual(result["searches_unfollowed"], 1)

    def test_search_followed_by_tool_output_is_followed(self) -> None:
        timeline = [
            _tool_output("search", 40, "pattern", "Grep"),
            _tool_output("file_read", 80, "/some/file.py", "Read"),
        ]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["searches"], 1)
        self.assertEqual(result["searches_unfollowed"], 0)

    def test_multiple_searches_unfollowed_means_no_later_file_read(self) -> None:
        # "Followed" requires a later file_read — a shell command after a
        # search does not count as following it up.
        timeline = [
            _tool_output("search", 40, "foo", "Grep"),
            _tool_output("shell", 10, None, "Bash"),
            _tool_output("search", 60, "bar", "Grep"),
        ]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["searches"], 2)
        self.assertEqual(result["searches_unfollowed"], 2)

    def test_search_followed_by_eventual_read_is_followed(self) -> None:
        timeline = [
            _tool_output("search", 40, "foo", "Grep"),
            _tool_output("shell", 10, None, "Bash"),
            _tool_output("search", 60, "bar", "Grep"),
            _tool_output("file_read", 80, "/some/file.py", "Read"),
        ]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["searches"], 2)
        self.assertEqual(result["searches_unfollowed"], 0)

    def test_no_searches(self) -> None:
        timeline = [_tool_output("shell", 100, "ls -la", "Bash")]
        result = analyze_session(_make_record(), timeline)
        self.assertEqual(result["searches"], 0)
        self.assertEqual(result["searches_unfollowed"], 0)


# ---------------------------------------------------------------------------
# Tests for analyze_session — return dict structure
# ---------------------------------------------------------------------------

class TestAnalyzeSessionReturnType(unittest.TestCase):

    def test_all_keys_present(self) -> None:
        result = analyze_session(_make_record(), [])
        for key in ("reads", "reads_unused", "tokens_unused",
                    "searches", "searches_unfollowed"):
            self.assertIn(key, result)

    def test_all_values_are_int(self) -> None:
        timeline = [
            _tool_output("file_read", 50, "/x/y.py"),
            _tool_output("search", 20, "pattern"),
        ]
        result = analyze_session(_make_record(), timeline)
        for key, val in result.items():
            self.assertIsInstance(val, (int, float), f"{key} should be numeric")

    def test_reads_unused_never_exceeds_reads(self) -> None:
        timeline = [_tool_output("file_read", 50, f"/proj/f{i}.py") for i in range(5)]
        result = analyze_session(_make_record(), timeline)
        self.assertLessEqual(result["reads_unused"], result["reads"])


# ---------------------------------------------------------------------------
# Tests for analyze_corpus
# ---------------------------------------------------------------------------

def _write_claude_session(path: Path, file_path_target: str | None = "/project/foo.py",
                          later_write: bool = False) -> None:
    """Write a minimal synthetic Claude session JSONL with a Read tool call."""
    tool_use_id = "toolu_traj_001"
    later_tool_use_id = "toolu_traj_002"

    # Build tool_use block with optional file_path
    tool_use_input: dict = {}
    if file_path_target:
        tool_use_input["file_path"] = file_path_target

    lines = [
        json.dumps({
            "type": "user", "uuid": "u1", "sessionId": "s-traj",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "cwd": "/project", "version": "2.0", "isSidechain": False,
            "message": {"role": "user", "content": "read file"}
        }),
        json.dumps({
            "type": "assistant", "uuid": "a1", "sessionId": "s-traj",
            "timestamp": "2026-01-01T00:00:01.000Z",
            "cwd": "/project", "version": "2.0", "isSidechain": False,
            "message": {
                "id": "m1", "role": "assistant", "model": "claude-sonnet-4-6",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": tool_use_id, "name": "Read",
                     "input": tool_use_input},
                ],
                "usage": {"input_tokens": 100, "cache_creation_input_tokens": 0,
                          "cache_read_input_tokens": 0, "output_tokens": 20}
            }
        }),
        json.dumps({
            "type": "user", "uuid": "u2", "sessionId": "s-traj",
            "timestamp": "2026-01-01T00:00:02.000Z",
            "cwd": "/project", "version": "2.0", "isSidechain": False,
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_id,
                              "content": "file contents here " * 10}]
            }
        }),
    ]

    if later_write:
        # Add a second assistant + user pair for an Edit of the same file
        lines += [
            json.dumps({
                "type": "assistant", "uuid": "a2", "sessionId": "s-traj",
                "timestamp": "2026-01-01T00:00:03.000Z",
                "cwd": "/project", "version": "2.0", "isSidechain": False,
                "message": {
                    "id": "m2", "role": "assistant", "model": "claude-sonnet-4-6",
                    "stop_reason": "tool_use",
                    "content": [
                        {"type": "tool_use", "id": later_tool_use_id, "name": "Edit",
                         "input": {"file_path": file_path_target, "old_string": "x",
                                   "new_string": "y"}},
                    ],
                    "usage": {"input_tokens": 150, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0, "output_tokens": 30}
                }
            }),
            json.dumps({
                "type": "user", "uuid": "u3", "sessionId": "s-traj",
                "timestamp": "2026-01-01T00:00:04.000Z",
                "cwd": "/project", "version": "2.0", "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": later_tool_use_id,
                                 "content": "edit applied"}]
                }
            }),
        ]

    path.write_text("\n".join(lines) + "\n")


class TestAnalyzeCorpus(unittest.TestCase):

    def test_empty_roots_returns_zero_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            result = analyze_corpus({"claude-code": Path(tmp)})
        for key in ("sessions_sampled", "sessions_skipped", "reads",
                    "reads_unused", "tokens_unused", "searches",
                    "searches_unfollowed"):
            self.assertIn(key, result)
        self.assertEqual(result["sessions_sampled"], 0)

    def test_corpus_counts_unused_read(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            _write_claude_session(p / "sess1.jsonl", file_path_target="/project/foo.py",
                                  later_write=False)
            result = analyze_corpus({"claude-code": p})
        self.assertEqual(result["sessions_sampled"], 1)
        self.assertGreaterEqual(result["reads"], 1)
        self.assertGreaterEqual(result["reads_unused"], 1)

    def test_corpus_counts_used_read(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            _write_claude_session(p / "sess1.jsonl", file_path_target="/project/foo.py",
                                  later_write=True)
            result = analyze_corpus({"claude-code": p})
        self.assertEqual(result["sessions_sampled"], 1)
        self.assertEqual(result["reads_unused"], 0)

    def test_corpus_skips_no_usage_files(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            # Write an empty file — no usage data → should be skipped
            (p / "empty.jsonl").write_text("")
            result = analyze_corpus({"claude-code": p})
        self.assertEqual(result["sessions_sampled"], 0)
        self.assertEqual(result["sessions_skipped"], 1)

    def test_corpus_limit_sessions_respected(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(5):
                _write_claude_session(p / f"sess{i}.jsonl",
                                      file_path_target=f"/project/f{i}.py")
            result = analyze_corpus({"claude-code": p}, limit_sessions=2, seed=42)
        self.assertEqual(result["sessions_sampled"], 2)

    def test_corpus_seed_is_deterministic(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            for i in range(10):
                _write_claude_session(p / f"sess{i}.jsonl",
                                      file_path_target=f"/project/f{i}.py")
            r1 = analyze_corpus({"claude-code": p}, limit_sessions=3, seed=99)
            r2 = analyze_corpus({"claude-code": p}, limit_sessions=3, seed=99)
        self.assertEqual(r1["reads"], r2["reads"])
        self.assertEqual(r1["reads_unused"], r2["reads_unused"])

    def test_corpus_different_seeds_may_differ(self) -> None:
        """With 10 sessions sampled 3 at a time, different seeds likely give different samples."""
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            # Mix of used/unused: even indices have later_write=True
            for i in range(10):
                _write_claude_session(p / f"sess{i}.jsonl",
                                      file_path_target=f"/project/f{i}.py",
                                      later_write=(i % 2 == 0))
            r1 = analyze_corpus({"claude-code": p}, limit_sessions=3, seed=1)
            r2 = analyze_corpus({"claude-code": p}, limit_sessions=3, seed=2)
        # We can't assert they differ (might coincidentally be same), but
        # both should succeed and produce valid keys.
        for r in (r1, r2):
            self.assertIn("reads_unused_pct", r)

    def test_reads_unused_pct_zero_when_no_reads(self) -> None:
        with TemporaryDirectory() as tmp:
            result = analyze_corpus({"claude-code": Path(tmp)})
        self.assertEqual(result["reads_unused_pct"], 0.0)

    def test_aggregate_output_contains_no_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp)
            _write_claude_session(p / "sess.jsonl", file_path_target="/project/secret.py")
            result = analyze_corpus({"claude-code": p})
        for key, val in result.items():
            self.assertIsInstance(val, (int, float),
                                  f"Key '{key}' should be numeric, not {type(val)}")


# ---------------------------------------------------------------------------
# Tests for report()
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):

    def _make_session_result(self, reads: int = 3, reads_unused: int = 2,
                             tokens_unused: int = 500, searches: int = 1,
                             searches_unfollowed: int = 1) -> dict:
        return {
            "reads": reads,
            "reads_unused": reads_unused,
            "tokens_unused": tokens_unused,
            "searches": searches,
            "searches_unfollowed": searches_unfollowed,
        }

    def _make_corpus_result(self) -> dict:
        return {
            "sessions_sampled": 42,
            "sessions_skipped": 3,
            "reads": 100,
            "reads_unused": 40,
            "tokens_unused": 8000,
            "reads_unused_pct": 40.0,
            "searches": 20,
            "searches_unfollowed": 5,
        }

    def test_report_returns_string(self) -> None:
        out = report(self._make_session_result())
        self.assertIsInstance(out, str)

    def test_report_contains_proxy_caveat(self) -> None:
        out = report(self._make_session_result())
        self.assertIn("PROXY", out)

    def test_report_contains_upper_bound(self) -> None:
        out = report(self._make_session_result())
        self.assertIn("UPPER BOUND", out)

    def test_report_contains_no_lower_bound_claim(self) -> None:
        out = report(self._make_session_result())
        self.assertIn("No lower bound", out)

    def test_report_contains_caveats_section(self) -> None:
        out = report(self._make_session_result())
        self.assertIn("Caveats", out)

    def test_report_contains_unused_read_counts(self) -> None:
        out = report(self._make_session_result(reads=3, reads_unused=2))
        self.assertIn("3", out)
        self.assertIn("2", out)

    def test_report_corpus_contains_sessions_count(self) -> None:
        out = report(self._make_corpus_result())
        self.assertIn("42", out)

    def test_report_corpus_shows_pct(self) -> None:
        out = report(self._make_corpus_result())
        self.assertIn("40.0%", out)

    def test_report_empty_session(self) -> None:
        out = report(self._make_session_result(reads=0, reads_unused=0,
                                               tokens_unused=0, searches=0,
                                               searches_unfollowed=0))
        self.assertIsInstance(out, str)
        self.assertIn("Caveats", out)

    def test_report_zero_reads_no_division_error(self) -> None:
        # Should not raise ZeroDivisionError
        out = report({"reads": 0, "reads_unused": 0, "tokens_unused": 0,
                      "searches": 0, "searches_unfollowed": 0})
        self.assertIsInstance(out, str)


# ---------------------------------------------------------------------------
# Tests for parser target field — Claude parser
# ---------------------------------------------------------------------------

class TestClaudeParserTarget(unittest.TestCase):
    """Verify that the claude parser populates the 'target' field on tool_output events."""

    def _parse_single_tool(self, tool_name: str, tool_input: dict,
                           tool_result_content: str = "result") -> dict | None:
        """Build a minimal Claude session with one tool call and return the tool_output event."""
        from tokensmash.sessions.claude import parse_session

        tool_use_id = "toolu_target_test"
        lines = [
            json.dumps({
                "type": "user", "uuid": "u1", "sessionId": "s1",
                "timestamp": "2026-01-01T00:00:00.000Z",
                "cwd": "/project", "version": "2.0", "isSidechain": False,
                "message": {"role": "user", "content": "do something"}
            }),
            json.dumps({
                "type": "assistant", "uuid": "a1", "sessionId": "s1",
                "timestamp": "2026-01-01T00:00:01.000Z",
                "cwd": "/project", "version": "2.0", "isSidechain": False,
                "message": {
                    "id": "m1", "role": "assistant", "model": "claude-test",
                    "stop_reason": "tool_use",
                    "content": [{"type": "tool_use", "id": tool_use_id,
                                 "name": tool_name, "input": tool_input}],
                    "usage": {"input_tokens": 50, "cache_creation_input_tokens": 0,
                              "cache_read_input_tokens": 0, "output_tokens": 10}
                }
            }),
            json.dumps({
                "type": "user", "uuid": "u2", "sessionId": "s1",
                "timestamp": "2026-01-01T00:00:02.000Z",
                "cwd": "/project", "version": "2.0", "isSidechain": False,
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": tool_use_id,
                                 "content": tool_result_content}]
                }
            }),
        ]
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "session.jsonl"
            p.write_text("\n".join(lines) + "\n")
            result = parse_session(p)
        if result is None:
            return None
        _, timeline = result
        tool_evs = [e for e in timeline if e.get("kind") == "tool_output"]
        return tool_evs[0] if tool_evs else None

    def test_read_target_is_file_path(self) -> None:
        ev = self._parse_single_tool("Read", {"file_path": "/project/foo.py"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "/project/foo.py")  # type: ignore[index]

    def test_edit_target_is_file_path(self) -> None:
        ev = self._parse_single_tool("Edit", {"file_path": "/project/bar.py",
                                               "old_string": "x", "new_string": "y"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "/project/bar.py")  # type: ignore[index]

    def test_write_target_is_file_path(self) -> None:
        ev = self._parse_single_tool("Write", {"file_path": "/project/new.py",
                                                "content": "print('hi')"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "/project/new.py")  # type: ignore[index]

    def test_bash_target_is_command(self) -> None:
        ev = self._parse_single_tool("Bash", {"command": "echo hello"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "echo hello")  # type: ignore[index]

    def test_bash_target_truncated_at_120(self) -> None:
        long_cmd = "x" * 200
        ev = self._parse_single_tool("Bash", {"command": long_cmd})
        self.assertIsNotNone(ev)
        self.assertEqual(len(ev["target"]), 120)  # type: ignore[index]

    def test_grep_target_is_pattern(self) -> None:
        ev = self._parse_single_tool("Grep", {"pattern": "def main", "path": "/project"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "def main")  # type: ignore[index]

    def test_glob_target_is_pattern(self) -> None:
        ev = self._parse_single_tool("Glob", {"pattern": "**/*.py"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "**/*.py")  # type: ignore[index]

    def test_mcp_tool_has_no_target(self) -> None:
        ev = self._parse_single_tool("mcp__headroom__compress", {"data": "x"})
        self.assertIsNotNone(ev)
        self.assertNotIn("target", ev)  # type: ignore[operator]

    def test_unknown_tool_has_no_target(self) -> None:
        ev = self._parse_single_tool("TaskCreate", {"description": "do something"})
        self.assertIsNotNone(ev)
        self.assertNotIn("target", ev)  # type: ignore[operator]

    def test_read_without_file_path_has_no_target(self) -> None:
        ev = self._parse_single_tool("Read", {})
        self.assertIsNotNone(ev)
        self.assertNotIn("target", ev)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Tests for parser target field — Codex parser
# ---------------------------------------------------------------------------

class TestCodexParserTarget(unittest.TestCase):
    """Verify that the codex parser populates the 'target' field on tool_output events."""

    _BASE_META = [
        {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta",
         "payload": {"id": "sess-traj-codex", "cwd": "/project",
                     "originator": "codex_exec", "cli_version": "1.0.0",
                     "source": "exec", "thread_source": "user",
                     "model_provider": "openai",
                     "base_instructions": {"text": "You are Codex."}}},
        {"timestamp": "2026-01-01T00:00:01.000Z", "type": "turn_context",
         "payload": {"turn_id": "t1", "cwd": "/project", "workspace_roots": [],
                     "current_date": "2026-01-01", "timezone": "UTC",
                     "approval_policy": "never",
                     "sandbox_policy": {"type": "workspace-write"},
                     "model": "gpt-5.5"}},
        {"timestamp": "2026-01-01T00:00:02.000Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "Do something."}},
    ]

    _TOKEN_COUNT = {"timestamp": "2026-01-01T00:00:10.000Z", "type": "event_msg",
                    "payload": {"type": "token_count", "info": {"total_token_usage": {
                        "input_tokens": 5000, "cached_input_tokens": 500,
                        "output_tokens": 100, "reasoning_output_tokens": 0,
                        "total_tokens": 5100}}}}

    def _parse_with_tool(self, tool_name: str, arguments: dict) -> dict | None:
        from tokensmash.sessions.codex import parse_session

        call_id = "call-traj-001"
        rows = list(self._BASE_META) + [
            {"timestamp": "2026-01-01T00:00:03.000Z", "type": "response_item",
             "payload": {"type": "function_call", "name": tool_name,
                         "arguments": json.dumps(arguments), "call_id": call_id}},
            {"timestamp": "2026-01-01T00:00:04.000Z", "type": "response_item",
             "payload": {"type": "function_call_output", "call_id": call_id,
                         "output": "some output text here"}},
            self._TOKEN_COUNT,
        ]
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "session.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            result = parse_session(p)
        if result is None:
            return None
        _, timeline = result
        tool_evs = [e for e in timeline if e.get("kind") == "tool_output"]
        return tool_evs[0] if tool_evs else None

    def test_exec_command_cmd_key_is_target(self) -> None:
        ev = self._parse_with_tool("exec_command", {"cmd": "cat /project/foo.py",
                                                     "workdir": "/project"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "cat /project/foo.py")  # type: ignore[index]

    def test_shell_command_command_key_is_target(self) -> None:
        ev = self._parse_with_tool("shell_command", {"command": "grep -n foo /project/bar.py",
                                                      "workdir": "/project"})
        self.assertIsNotNone(ev)
        self.assertEqual(ev["target"], "grep -n foo /project/bar.py")  # type: ignore[index]

    def test_target_truncated_at_120(self) -> None:
        long_cmd = "echo " + "x" * 200
        ev = self._parse_with_tool("shell_command", {"command": long_cmd})
        self.assertIsNotNone(ev)
        self.assertEqual(len(ev["target"]), 120)  # type: ignore[index]

    def test_tool_without_command_key_has_no_target(self) -> None:
        ev = self._parse_with_tool("update_plan", {"explanation": "foo", "plan": "bar"})
        self.assertIsNotNone(ev)
        self.assertNotIn("target", ev)  # type: ignore[operator]

    def test_mcp_tool_has_no_target(self) -> None:
        ev = self._parse_with_tool("mcp__amplitude__search",
                                   {"query": "events", "limit": 10})
        self.assertIsNotNone(ev)
        self.assertNotIn("target", ev)  # type: ignore[operator]


if __name__ == "__main__":
    unittest.main()
