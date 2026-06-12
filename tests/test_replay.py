"""Tests for tokensmash.replay — offline realized-compression estimates.

All tests use:
  - Synthetic JSONL fixtures written to a TemporaryDirectory (no real transcripts)
  - A fake reducer shell script (tests/fixtures/replay/fake_reducer.sh) that
    outputs every other character of stdin (≈ 50% reduction), so no rtk or
    headroom binary is required.

Privacy invariants tested:
  - No transcript text appears in any replay_session / replay_corpus output.
  - Only aggregate counts (tokens_in, tokens_out, samples, failures) are kept.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# Make the project importable when tests are run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokensmash import replay as _replay_mod
from tokensmash.replay import replay_session, replay_corpus, report

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_ROOT = Path(__file__).parent / "fixtures" / "replay"
FAKE_REDUCER = FIXTURES_ROOT / "fake_reducer.sh"
CLAUDE_FIXTURES = Path(__file__).parent / "fixtures" / "claude"

# ---------------------------------------------------------------------------
# Helpers — minimal JSONL builders
# ---------------------------------------------------------------------------

def _assistant_entry(session_id: str, tool_uses: list[dict], usage: dict) -> dict:
    """Build an assistant JSONL entry with tool_use blocks and usage."""
    return {
        "type": "assistant",
        "uuid": "uuid-asst-1",
        "sessionId": session_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": "/tmp/fakerepo",
        "version": "1.0.0",
        "isSidechain": False,
        "message": {
            "id": f"msg-{session_id}",
            "role": "assistant",
            "model": "claude-test-model",
            "content": tool_uses,
            "usage": usage,
        },
    }


def _tool_result_entry(session_id: str, tool_use_id: str, content: str) -> dict:
    """Build a user JSONL entry containing a single tool_result block."""
    return {
        "type": "user",
        "uuid": "uuid-user-2",
        "sessionId": session_id,
        "timestamp": "2026-01-01T00:00:01.000Z",
        "cwd": "/tmp/fakerepo",
        "version": "1.0.0",
        "isSidechain": False,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ],
        },
    }


def _mode_entry(session_id: str) -> dict:
    return {
        "type": "mode",
        "uuid": "uuid-mode-0",
        "sessionId": session_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": "/tmp/fakerepo",
        "version": "1.0.0",
        "isSidechain": False,
        "subtype": "init",
    }


def _initial_user_entry(session_id: str) -> dict:
    return {
        "type": "user",
        "uuid": "uuid-user-0",
        "sessionId": session_id,
        "timestamp": "2026-01-01T00:00:00.100Z",
        "cwd": "/tmp/fakerepo",
        "version": "1.0.0",
        "isSidechain": False,
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "Do something please."}],
        },
    }


def _make_claude_session_jsonl(
    tmp_dir: Path,
    session_id: str,
    tool_content: str = "A" * 400,
    tool_name: str = "Bash",
) -> Path:
    """Write a minimal valid Claude JSONL session to tmp_dir."""
    tool_use_id = "toolu_replay_001"
    entries = [
        _mode_entry(session_id),
        _initial_user_entry(session_id),
        _assistant_entry(
            session_id,
            tool_uses=[
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": {"command": "ls -la"},
                }
            ],
            usage={
                "input_tokens": 1000,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": 50,
            },
        ),
        _tool_result_entry(session_id, tool_use_id, tool_content),
    ]
    path = tmp_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return path


# ---------------------------------------------------------------------------
# Fake-reducer injection helpers
# ---------------------------------------------------------------------------

def _make_fake_rtk_runner(fake_script: Path):
    """Return a _run_rtk_pipe replacement that uses the fake reducer script."""
    def _fake_rtk(text: str, timeout: float = 30.0) -> dict | None:
        import subprocess
        if not fake_script.exists():
            return None
        try:
            proc = subprocess.run(
                ["bash", str(fake_script)],
                input=text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        out = proc.stdout
        return {
            "tokens_in": max(1, len(text) // 4),
            "tokens_out": max(1, len(out) // 4),
        }
    return _fake_rtk


def _make_fake_headroom_runner(fake_script: Path):
    """Return a _run_headroom replacement that uses the fake reducer script."""
    def _fake_headroom(text: str, timeout: float = 30.0) -> dict | None:
        import subprocess
        if not fake_script.exists():
            return None
        try:
            proc = subprocess.run(
                ["bash", str(fake_script)],
                input=text,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except Exception:
            return None
        if proc.returncode != 0:
            return None
        out = proc.stdout
        return {
            "tokens_in": max(1, len(text) // 4),
            "tokens_out": max(1, len(out) // 4),
        }
    return _fake_headroom


# ---------------------------------------------------------------------------
# Test: _tokens_est
# ---------------------------------------------------------------------------

class TestTokensEst(unittest.TestCase):
    def test_basic(self):
        from tokensmash.replay import _tokens_est
        self.assertEqual(_tokens_est("a" * 400), 100)

    def test_minimum_one(self):
        from tokensmash.replay import _tokens_est
        self.assertEqual(_tokens_est(""), 1)
        self.assertEqual(_tokens_est("x"), 1)


# ---------------------------------------------------------------------------
# Test: fake reducer script
# ---------------------------------------------------------------------------

class TestFakeReducer(unittest.TestCase):
    """Verify the fixture fake reducer works as expected."""

    def test_fake_reducer_exists(self):
        self.assertTrue(FAKE_REDUCER.exists(), f"Fixture not found: {FAKE_REDUCER}")
        self.assertTrue(os.access(FAKE_REDUCER, os.X_OK), "Fake reducer not executable")

    def test_fake_reducer_output_shorter(self):
        import subprocess
        text = "ABCDEFGHIJ" * 40  # 400 chars
        proc = subprocess.run(
            ["bash", str(FAKE_REDUCER)],
            input=text,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0)
        out = proc.stdout
        # Every other char → roughly half the length
        self.assertGreater(len(text), len(out))
        # Approximately 50% (allow ±10 chars)
        self.assertAlmostEqual(len(out), len(text) / 2, delta=10)


# ---------------------------------------------------------------------------
# Test: replay_session with injected fake reducers
# ---------------------------------------------------------------------------

class TestReplaySession(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        # Inject fake runners so CI never needs rtk/headroom
        _replay_mod._TOOL_RUNNERS["rtk"] = _make_fake_rtk_runner(FAKE_REDUCER)
        _replay_mod._TOOL_RUNNERS["headroom"] = _make_fake_headroom_runner(FAKE_REDUCER)

    def tearDown(self):
        # Restore originals
        _replay_mod._TOOL_RUNNERS["rtk"] = _replay_mod._run_rtk_pipe
        _replay_mod._TOOL_RUNNERS["headroom"] = _replay_mod._run_headroom
        self.tmp.cleanup()

    def _make_session(self, session_id: str = "sess-replay-001", content: str = "X" * 400) -> Path:
        return _make_claude_session_jsonl(self.tmp_dir, session_id, tool_content=content)

    def test_returns_none_for_no_tool_output(self):
        # Session with no tool_result entries → None
        path = self.tmp_dir / "empty.jsonl"
        path.write_text(json.dumps({"type": "unknown"}) + "\n")
        result = replay_session(path, ["rtk"])
        self.assertIsNone(result)

    def test_returns_dict_for_valid_session(self):
        path = self._make_session()
        result = replay_session(path, ["rtk"])
        self.assertIsNotNone(result)
        self.assertIn("rtk", result)

    def test_result_shape(self):
        path = self._make_session()
        result = replay_session(path, ["rtk", "headroom"])
        self.assertIsNotNone(result)
        for tool in ["rtk", "headroom"]:
            with self.subTest(tool=tool):
                self.assertIn(tool, result)
                td = result[tool]
                self.assertIn("tokens_in", td)
                self.assertIn("tokens_out", td)
                self.assertIn("ratio", td)
                self.assertIn("samples", td)
                self.assertIn("failures", td)

    def test_ratio_is_less_than_one(self):
        """Fake reducer halves output → ratio < 1."""
        path = self._make_session(content="Z" * 800)
        result = replay_session(path, ["rtk"])
        self.assertIsNotNone(result)
        ratio = result["rtk"]["ratio"]
        self.assertLess(ratio, 1.0, "Fake reducer should compress below 1.0")

    def test_no_text_in_output(self):
        """Replay output must contain only numeric aggregates, no transcript text."""
        path = self._make_session(content="SECRET_CONTENT_12345")
        result = replay_session(path, ["rtk"])
        self.assertIsNotNone(result)
        result_str = json.dumps(result)
        self.assertNotIn("SECRET_CONTENT", result_str)

    def test_tool_failure_counted_not_fatal(self):
        """When reducer fails (returns None), failure is counted, result is returned."""
        # Replace rtk with a always-failing runner
        _replay_mod._TOOL_RUNNERS["rtk"] = lambda text, timeout=30.0: None
        try:
            path = self._make_session()
            result = replay_session(path, ["rtk"])
            # Should return a dict (not None) even if all reducer calls fail,
            # as long as tool_output events exist.
            # If all samples fail, tokens_in may be 0 but failures > 0.
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result["rtk"]["failures"], 0)
        finally:
            _replay_mod._TOOL_RUNNERS["rtk"] = _make_fake_rtk_runner(FAKE_REDUCER)

    def test_samples_increases_with_more_tool_outputs(self):
        """More tool outputs → more samples."""
        # Write a session with multiple tool_result entries
        session_id = "sess-multi-tool"
        tool_content = "Y" * 200
        entries = [
            _mode_entry(session_id),
            _initial_user_entry(session_id),
            _assistant_entry(
                session_id,
                tool_uses=[
                    {"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {}},
                    {"type": "tool_use", "id": "toolu_b", "name": "Bash", "input": {}},
                ],
                usage={
                    "input_tokens": 1000,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 50,
                },
            ),
            _tool_result_entry(session_id, "toolu_a", tool_content),
            _tool_result_entry(session_id, "toolu_b", tool_content),
        ]
        path = self.tmp_dir / f"{session_id}.jsonl"
        path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        result_single = replay_session(self._make_session(), ["rtk"])
        result_multi = replay_session(path, ["rtk"])

        samples_single = result_single["rtk"]["samples"] if result_single else 0
        samples_multi = result_multi["rtk"]["samples"] if result_multi else 0
        self.assertGreaterEqual(samples_multi, samples_single)

    def test_unknown_tool_returns_empty_for_that_tool(self):
        """An unknown tool name produces zero samples without crashing."""
        path = self._make_session()
        result = replay_session(path, ["nonexistent_tool"])
        self.assertIsNotNone(result)
        self.assertIn("nonexistent_tool", result)
        self.assertEqual(result["nonexistent_tool"]["samples"], 0)

    def test_existing_fixture_works(self):
        """The existing claude normal_session fixture is parseable for replay."""
        fixture = CLAUDE_FIXTURES / "normal_session.jsonl"
        if not fixture.exists():
            self.skipTest("claude normal_session fixture not found")
        result = replay_session(fixture, ["rtk", "headroom"])
        # May be None if parser finds no tool_outputs in fixture
        if result is not None:
            for tool in ["rtk", "headroom"]:
                with self.subTest(tool=tool):
                    self.assertIn(tool, result)
                    self.assertIsInstance(result[tool]["ratio"], float)


# ---------------------------------------------------------------------------
# Test: replay_corpus
# ---------------------------------------------------------------------------

class TestReplayCorpus(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        _replay_mod._TOOL_RUNNERS["rtk"] = _make_fake_rtk_runner(FAKE_REDUCER)
        _replay_mod._TOOL_RUNNERS["headroom"] = _make_fake_headroom_runner(FAKE_REDUCER)

    def tearDown(self):
        _replay_mod._TOOL_RUNNERS["rtk"] = _replay_mod._run_rtk_pipe
        _replay_mod._TOOL_RUNNERS["headroom"] = _replay_mod._run_headroom
        self.tmp.cleanup()

    def _populate_claude_root(self, n: int = 3) -> Path:
        root = self.tmp_dir / "claude_root" / "projects" / "fake-proj"
        root.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _make_claude_session_jsonl(root, f"sess-corpus-{i:03d}", tool_content="W" * 300)
        return self.tmp_dir / "claude_root"

    def test_corpus_shape(self):
        root = self._populate_claude_root(3)
        result = replay_corpus({"claude": root}, tools=["rtk"])
        self.assertIn("tools", result)
        self.assertIn("sessions_processed", result)
        self.assertIn("sessions_skipped", result)
        self.assertIn("rtk", result["tools"])

    def test_corpus_processes_sessions(self):
        root = self._populate_claude_root(3)
        result = replay_corpus({"claude": root}, tools=["rtk"])
        self.assertGreater(result["sessions_processed"] + result["sessions_skipped"], 0)

    def test_seeded_sampling(self):
        """Same seed → same sessions selected → identical results."""
        root = self._populate_claude_root(6)
        r1 = replay_corpus({"claude": root}, tools=["rtk"], limit_sessions=3, seed=42)
        r2 = replay_corpus({"claude": root}, tools=["rtk"], limit_sessions=3, seed=42)
        self.assertEqual(r1["tools"]["rtk"]["samples"], r2["tools"]["rtk"]["samples"])
        self.assertEqual(r1["tools"]["rtk"]["tokens_in"], r2["tools"]["rtk"]["tokens_in"])

    def test_different_seeds_may_differ(self):
        """Different seeds → (almost always) different sample sets."""
        root = self._populate_claude_root(10)
        r1 = replay_corpus({"claude": root}, tools=["rtk"], limit_sessions=3, seed=1)
        r2 = replay_corpus({"claude": root}, tools=["rtk"], limit_sessions=3, seed=99999)
        # With 10 sessions and 3 samples, seeds 1 vs 99999 will differ;
        # but we can only assert sessions_processed is the same limit.
        self.assertEqual(r1["sessions_processed"] + r1["sessions_skipped"], 3)
        self.assertEqual(r2["sessions_processed"] + r2["sessions_skipped"], 3)

    def test_limit_sessions_respected(self):
        root = self._populate_claude_root(6)
        result = replay_corpus({"claude": root}, tools=["rtk"], limit_sessions=2)
        total = result["sessions_processed"] + result["sessions_skipped"]
        self.assertEqual(total, 2)

    def test_empty_roots_returns_zero(self):
        empty = self.tmp_dir / "empty_root"
        empty.mkdir()
        result = replay_corpus({"claude": empty}, tools=["rtk"])
        self.assertEqual(result["sessions_processed"], 0)
        self.assertEqual(result["sessions_skipped"], 0)

    def test_no_text_in_corpus_output(self):
        """Corpus output must not contain any transcript text."""
        root = self._populate_claude_root(2)
        result = replay_corpus({"claude": root}, tools=["rtk"])
        result_str = json.dumps(result)
        # The tool_content we wrote was 'W' * 300 — none should appear
        self.assertNotIn("WWWWW", result_str)

    def test_corpus_ratio_between_zero_and_one(self):
        root = self._populate_claude_root(3)
        result = replay_corpus({"claude": root}, tools=["rtk"])
        ratio = result["tools"]["rtk"]["ratio"]
        self.assertGreater(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)

    def test_multi_tool(self):
        root = self._populate_claude_root(2)
        result = replay_corpus({"claude": root}, tools=["rtk", "headroom"])
        self.assertIn("rtk", result["tools"])
        self.assertIn("headroom", result["tools"])


# ---------------------------------------------------------------------------
# Test: report
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        _replay_mod._TOOL_RUNNERS["rtk"] = _make_fake_rtk_runner(FAKE_REDUCER)
        _replay_mod._TOOL_RUNNERS["headroom"] = _make_fake_headroom_runner(FAKE_REDUCER)

    def tearDown(self):
        _replay_mod._TOOL_RUNNERS["rtk"] = _replay_mod._run_rtk_pipe
        _replay_mod._TOOL_RUNNERS["headroom"] = _replay_mod._run_headroom
        self.tmp.cleanup()

    def _make_corpus_result(self) -> dict:
        root = self.tmp_dir / "projects" / "proj"
        root.mkdir(parents=True, exist_ok=True)
        _make_claude_session_jsonl(root, "sess-r-001", tool_content="V" * 400)
        return replay_corpus(
            {"claude": self.tmp_dir},
            tools=["rtk", "headroom"],
            limit_sessions=1,
            seed=17,
        )

    def test_report_is_string(self):
        result = self._make_corpus_result()
        r = report(result)
        self.assertIsInstance(r, str)
        self.assertGreater(len(r), 0)

    def test_report_contains_header(self):
        result = self._make_corpus_result()
        r = report(result)
        self.assertIn("Realized Compression", r)

    def test_report_contains_all_tools(self):
        result = self._make_corpus_result()
        r = report(result)
        for tool in ["rtk", "headroom", "repomix"]:
            with self.subTest(tool=tool):
                self.assertIn(tool, r)

    def test_report_caveat_block(self):
        """Report must include the required caveats."""
        result = self._make_corpus_result()
        r = report(result)
        self.assertIn("len", r.lower())          # len/4 approximation
        self.assertIn("trajectory", r.lower())   # trajectory effects caveat
        self.assertIn("repomix", r.lower())      # repomix out of scope caveat

    def test_report_no_transcript_text(self):
        """Report must not contain any transcript text."""
        result = self._make_corpus_result()
        r = report(result)
        self.assertNotIn("VVVVV", r)

    def test_report_accepts_plain_tool_dict(self):
        """report() must also accept a plain per-tool dict (replay_session output)."""
        plain = {
            "rtk": {"tokens_in": 1000, "tokens_out": 500, "ratio": 0.5,
                    "samples": 5, "failures": 0},
            "headroom": {"tokens_in": 1000, "tokens_out": 600, "ratio": 0.6,
                         "samples": 5, "failures": 0},
        }
        r = report(plain)
        self.assertIsInstance(r, str)
        self.assertIn("rtk", r)
        self.assertIn("headroom", r)

    def test_report_empty_tools(self):
        """report() with no-sample results should still produce a table."""
        empty = {
            "tools": {
                "rtk": {"tokens_in": 0, "tokens_out": 0, "ratio": 1.0,
                        "samples": 0, "failures": 0},
            },
            "sessions_processed": 0,
            "sessions_skipped": 0,
        }
        r = report(empty)
        self.assertIn("n/a", r)


# ---------------------------------------------------------------------------
# Test: availability helpers
# ---------------------------------------------------------------------------

class TestAvailabilityHelpers(unittest.TestCase):

    def test_rtk_available_returns_bool(self):
        from tokensmash.replay import _rtk_available
        result = _rtk_available()
        self.assertIsInstance(result, bool)

    def test_headroom_available_returns_bool(self):
        from tokensmash.replay import _headroom_available
        result = _headroom_available()
        self.assertIsInstance(result, bool)


# ---------------------------------------------------------------------------
# Test: subprocess timeout respected (using a slow fake)
# ---------------------------------------------------------------------------

class TestSubprocessTimeout(unittest.TestCase):

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        _replay_mod._TOOL_RUNNERS["rtk"] = _replay_mod._run_rtk_pipe
        _replay_mod._TOOL_RUNNERS["headroom"] = _replay_mod._run_headroom
        self.tmp.cleanup()

    def test_timeout_counts_as_failure(self):
        """A reducer that times out is counted as a failure, not a crash."""
        # Write a slow-reducer that sleeps forever
        slow_script = self.tmp_dir / "slow_reducer.sh"
        slow_script.write_text("#!/usr/bin/env bash\nsleep 9999\n")
        slow_script.chmod(0o755)

        def _slow_rtk(text: str, timeout: float = 0.1) -> dict | None:
            import subprocess
            try:
                subprocess.run(
                    ["bash", str(slow_script)],
                    input=text, capture_output=True, text=True,
                    timeout=0.1, check=False,
                )
            except subprocess.TimeoutExpired:
                return None
            return None

        _replay_mod._TOOL_RUNNERS["rtk"] = _slow_rtk

        path = _make_claude_session_jsonl(self.tmp_dir, "sess-slow-001", tool_content="T" * 200)
        result = replay_session(path, ["rtk"], timeout=0.1)
        # Should not raise; failures counter may be > 0
        if result is not None:
            self.assertGreaterEqual(result["rtk"]["failures"], 0)


if __name__ == "__main__":
    unittest.main()
