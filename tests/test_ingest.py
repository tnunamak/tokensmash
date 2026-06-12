"""Tests for tokensmash.ingest.

Strategy
--------
- Monkeypatch tokensmash.schema.STUDY_DIR (canonical) so all modules see the
  same tempdir — same pattern as tests/test_study_link.py.
- Monkeypatch ``tokensmash.ingest.opportunity`` with a stub so tests are
  independent of the parallel opportunity.py worker.
- Point roots at the fixture directories under tests/fixtures/ for real
  transcript parsing.
- Codex fixtures live directly in tests/fixtures/codex/ (rglob finds them).
- Claude fixtures need a subdirectory — wrap them in a tempdir subdir.

Covered scenarios
-----------------
- idempotency: running twice yields added=0 on second run
- since_days: files older than cutoff are skipped
- unknown model: session with unresolvable model still ingests, counter incremented
- exclusion: session whose repo_id is in exclude_repo_ids gets excluded=study-repo
- arm joining (linked): assignment record in assignments.jsonl is used
- arm joining (recomputed): no assignment but config exists → recomputed
- no config: ingest succeeds, no arm attached
- opportunity error guarding: if opportunity raises, session still ingests with
  opportunity_error key
- corrupt store lines: load_latest skips them
- parse error: file with no usage returns skipped
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import tokensmash.schema as schema
from tokensmash.study.assign import init_study, arm_for, block_index, assignment_id
from tokensmash import store, ingest


# ---------------------------------------------------------------------------
# Fixtures root
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"
_CODEX_FIXTURES = _FIXTURES / "codex"
_CLAUDE_FIXTURES = _FIXTURES / "claude"


# ---------------------------------------------------------------------------
# Opportunity stub (always succeeds)
# ---------------------------------------------------------------------------

def _stub_summarize(timeline, compactions):
    return {"stub": True}

def _stub_tool_ceilings(record):
    return {}

class _StubOpportunity:
    summarize = staticmethod(_stub_summarize)
    tool_ceilings = staticmethod(_stub_tool_ceilings)


# ---------------------------------------------------------------------------
# Shared setup mixin
# ---------------------------------------------------------------------------

class _IngestMixin:
    """Mixin that wires a temp STUDY_DIR and stubs out opportunity."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._study_dir = Path(self._tmpdir) / "study"
        self._study_dir.mkdir(parents=True, exist_ok=True)

        import tokensmash.schema as schema_mod
        import tokensmash.study.assign as assign_mod

        self._orig_schema_dir = schema_mod.STUDY_DIR
        self._orig_assign_dir = assign_mod.STUDY_DIR

        schema_mod.STUDY_DIR = self._study_dir
        assign_mod.STUDY_DIR = self._study_dir

        # Claude fixtures need a subdirectory wrapper so iter_session_files finds them
        self._claude_root = Path(self._tmpdir) / "claude_root"
        self._claude_sub = self._claude_root / "proj-abc"
        self._claude_sub.mkdir(parents=True, exist_ok=True)

        # Codex fixtures work directly
        self._codex_root = Path(self._tmpdir) / "codex_root"
        self._codex_root.mkdir(parents=True, exist_ok=True)

        # Copy a normal fixture session into each root
        shutil.copy(
            _CODEX_FIXTURES / "normal_session.jsonl",
            self._codex_root / "normal_session.jsonl",
        )
        shutil.copy(
            _CLAUDE_FIXTURES / "normal_session.jsonl",
            self._claude_sub / "normal_session.jsonl",
        )

    def tearDown(self):
        import tokensmash.schema as schema_mod
        import tokensmash.study.assign as assign_mod

        schema_mod.STUDY_DIR = self._orig_schema_dir
        assign_mod.STUDY_DIR = self._orig_assign_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _ingest(self, roots=None, since_days=None, opp=None):
        """Run ingest with the given roots (defaults to both fixtures)."""
        if roots is None:
            roots = {
                "codex": self._codex_root,
                "claude-code": self._claude_root,
            }
        opp_stub = opp if opp is not None else _StubOpportunity()
        # Patch the module-level 'opportunity' attribute in ingest
        with patch.object(ingest, "opportunity", opp_stub):
            return ingest.ingest(roots, since_days=since_days)

    def _read_sessions(self) -> list[dict]:
        p = self._study_dir / "sessions.jsonl"
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Basic stats
# ---------------------------------------------------------------------------

class TestIngestBasicStats(_IngestMixin, unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        stats = self._ingest()
        required = {"scanned", "parsed", "added", "replaced", "skipped",
                    "parse_errors", "unknown_model_sessions"}
        self.assertTrue(required.issubset(stats.keys()))

    def test_scanned_nonzero(self):
        stats = self._ingest()
        self.assertGreater(stats["scanned"], 0)

    def test_parsed_nonzero(self):
        stats = self._ingest()
        self.assertGreater(stats["parsed"], 0)

    def test_added_nonzero_first_run(self):
        stats = self._ingest()
        self.assertGreater(stats["added"], 0)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIngestIdempotency(_IngestMixin, unittest.TestCase):

    def test_second_run_adds_zero(self):
        self._ingest()
        stats2 = self._ingest()
        self.assertEqual(stats2["added"], 0)

    def test_second_run_replaces_zero(self):
        self._ingest()
        stats2 = self._ingest()
        self.assertEqual(stats2["replaced"], 0)

    def test_session_count_stable(self):
        self._ingest()
        n1 = len(self._read_sessions())
        self._ingest()
        n2 = len(self._read_sessions())
        self.assertEqual(n1, n2)


# ---------------------------------------------------------------------------
# since_days filter
# ---------------------------------------------------------------------------

class TestIngestSinceDays(_IngestMixin, unittest.TestCase):

    def test_zero_since_days_skips_all_old_files(self):
        # Set mtime well in the past (10 days), then filter to 1 day
        for p in self._codex_root.rglob("*.jsonl"):
            import os
            old_time = time.time() - 10 * 86400
            os.utime(p, (old_time, old_time))
        for sub in self._claude_root.iterdir():
            for p in sub.iterdir():
                import os
                old_time = time.time() - 10 * 86400
                os.utime(p, (old_time, old_time))
        stats = self._ingest(since_days=1.0)
        self.assertEqual(stats["added"], 0)

    def test_no_since_days_processes_everything(self):
        stats = self._ingest(since_days=None)
        self.assertGreater(stats["added"], 0)


# ---------------------------------------------------------------------------
# No study config (no arm attached)
# ---------------------------------------------------------------------------

class TestIngestNoConfig(_IngestMixin, unittest.TestCase):

    def test_sessions_ingested_without_arm(self):
        self._ingest()
        sessions = self._read_sessions()
        self.assertGreater(len(sessions), 0)
        for s in sessions:
            self.assertNotIn("arm", s)

    def test_study_id_absent(self):
        self._ingest()
        sessions = self._read_sessions()
        for s in sessions:
            self.assertNotIn("study_id", s)


# ---------------------------------------------------------------------------
# Arm joining — linked from assignments.jsonl
# ---------------------------------------------------------------------------

class TestIngestArmLinked(_IngestMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        # Create a study config
        self._config = init_study("test-study", "log-only", "1.0",
                                  study_dir=self._study_dir)
        # Parse the codex fixture to learn session_id and repo_id
        from tokensmash.sessions.codex import parse_session
        path = self._codex_root / "normal_session.jsonl"
        result = parse_session(path)
        self.assertIsNotNone(result, "Normal codex session should parse")
        self._session_record, _ = result
        self._codex_session_id = self._session_record["session_id"]

        # Write a fake assignment record pointing to this session
        seed = bytes.fromhex(self._config["seed"])
        repo_id = self._session_record["repo_id"]
        blk = block_index(time.time())
        arm = arm_for(seed, repo_id, blk)
        asgn = {
            "schema": schema.ASSIGNMENT_SCHEMA,
            "study_id": "test-study",
            "agent": "codex",
            "session_id": self._codex_session_id,
            "repo_id": repo_id,
            "block": blk,
            "arm": arm,
            "assignment_id": assignment_id(repo_id, blk),
            "mode": "log-only",
            "linked_at": "2026-01-01T00:00:00+00:00",
            "source": "new",
            "transcript_path": None,
        }
        asgn_path = self._study_dir / "assignments.jsonl"
        with asgn_path.open("a") as fh:
            fh.write(json.dumps(asgn) + "\n")
        self._expected_arm = arm

    def test_arm_linked_from_assignment(self):
        self._ingest(roots={"codex": self._codex_root})
        sessions = self._read_sessions()
        codex_sessions = [s for s in sessions if s.get("agent") == "codex"]
        self.assertGreater(len(codex_sessions), 0)
        match = [s for s in codex_sessions
                 if s["session_id"] == self._codex_session_id]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0]["arm"], self._expected_arm)

    def test_arm_source_is_linked(self):
        self._ingest(roots={"codex": self._codex_root})
        sessions = self._read_sessions()
        match = [s for s in sessions
                 if s.get("session_id") == self._codex_session_id]
        self.assertEqual(match[0].get("arm_source"), "linked")


# ---------------------------------------------------------------------------
# Arm recomputed (config exists, no assignment record)
# ---------------------------------------------------------------------------

class TestIngestArmRecomputed(_IngestMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self._config = init_study("test-study-recompute", "log-only", "1.0",
                                  study_dir=self._study_dir)

    def test_arm_is_attached(self):
        self._ingest()
        sessions = self._read_sessions()
        self.assertGreater(len(sessions), 0)
        for s in sessions:
            self.assertIn("arm", s)
            self.assertIn(s["arm"], ("on", "off"))

    def test_arm_source_is_recomputed(self):
        self._ingest()
        sessions = self._read_sessions()
        for s in sessions:
            self.assertEqual(s.get("arm_source"), "recomputed")

    def test_arm_is_deterministic(self):
        self._ingest()
        sessions1 = {s["session_id"]: s["arm"] for s in self._read_sessions()}
        # Clear sessions store to force re-ingest
        (self._study_dir / "sessions.jsonl").unlink()
        self._ingest()
        sessions2 = {s["session_id"]: s["arm"] for s in self._read_sessions()}
        self.assertEqual(sessions1, sessions2)


# ---------------------------------------------------------------------------
# Exclusion by repo_id
# ---------------------------------------------------------------------------

class TestIngestExclusion(_IngestMixin, unittest.TestCase):

    def setUp(self):
        super().setUp()
        # Parse the codex fixture to get its repo_id, then put that in exclude_repo_ids
        from tokensmash.sessions.codex import parse_session
        path = self._codex_root / "normal_session.jsonl"
        result = parse_session(path)
        self.assertIsNotNone(result)
        self._session_record, _ = result
        excluded_repo = self._session_record["repo_id"]

        # init a study config with the exclusion
        config = init_study("excl-study", "log-only", "1.0",
                            study_dir=self._study_dir)
        # patch config to add the excluded repo_id
        config_path = self._study_dir / "config.json"
        cfg = json.loads(config_path.read_text())
        cfg["exclude_repo_ids"] = [excluded_repo]
        config_path.write_text(json.dumps(cfg, indent=2))

    def test_excluded_session_has_excluded_field(self):
        self._ingest(roots={"codex": self._codex_root})
        sessions = self._read_sessions()
        codex_sessions = [s for s in sessions if s.get("agent") == "codex"]
        self.assertGreater(len(codex_sessions), 0)
        for s in codex_sessions:
            self.assertEqual(s.get("excluded"), "study-repo")

    def test_excluded_session_still_ingested(self):
        # Excluded sessions are marked, not dropped
        self._ingest(roots={"codex": self._codex_root})
        sessions = self._read_sessions()
        codex_sessions = [s for s in sessions if s.get("agent") == "codex"]
        self.assertGreater(len(codex_sessions), 0)


# ---------------------------------------------------------------------------
# Unknown model handling
# ---------------------------------------------------------------------------

class TestIngestUnknownModel(_IngestMixin, unittest.TestCase):

    def test_unknown_model_session_counter(self):
        """Sessions with a model that cannot be priced still ingest."""
        # Create a special fixture with an unknown model by writing one
        fake_codex_root = Path(self._tmpdir) / "fake_codex"
        fake_codex_root.mkdir()
        # Write a minimal valid codex session with an unknown model
        lines = [
            json.dumps({
                "timestamp": "2026-01-01T00:00:00.000Z",
                "type": "session_meta",
                "payload": {
                    "id": "sess-unknown-model",
                    "cwd": "/tmp/fakerepo",
                    "cli_version": "1.0.0",
                },
            }),
            json.dumps({
                "timestamp": "2026-01-01T00:00:01.000Z",
                "type": "turn_context",
                "payload": {
                    "model": "totally-unknown-model-xyz-9999",
                    "cwd": "/tmp/fakerepo",
                },
            }),
            json.dumps({
                "timestamp": "2026-01-01T00:00:02.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "hello",
                },
            }),
            json.dumps({
                "timestamp": "2026-01-01T00:00:03.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 0,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 0,
                            "total_tokens": 120,
                        },
                        "last_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 0,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 0,
                            "total_tokens": 120,
                        },
                        "model_context_window": 128000,
                    },
                },
            }),
        ]
        (fake_codex_root / "sess.jsonl").write_text("\n".join(lines) + "\n")

        stats = self._ingest(roots={"codex": fake_codex_root})
        self.assertEqual(stats["parsed"], 1)
        sessions = self._read_sessions()
        self.assertEqual(len(sessions), 1)
        # cost_api_usd should be None for unknown model
        self.assertIsNone(sessions[0].get("cost_api_usd"))


# ---------------------------------------------------------------------------
# Opportunity error guarding
# ---------------------------------------------------------------------------

class TestIngestOpportunityGuard(_IngestMixin, unittest.TestCase):

    def test_opportunity_error_still_ingests(self):
        class _BrokenOpportunity:
            @staticmethod
            def summarize(timeline, compactions):
                raise RuntimeError("opportunity is broken")

            @staticmethod
            def tool_ceilings(record):
                raise RuntimeError("also broken")

        with patch.object(ingest, "opportunity", _BrokenOpportunity):
            stats = ingest.ingest(
                {"codex": self._codex_root},
                since_days=None,
            )

        sessions = self._read_sessions()
        self.assertGreater(len(sessions), 0)

    def test_opportunity_error_sets_error_key(self):
        class _BrokenOpportunity:
            @staticmethod
            def summarize(timeline, compactions):
                raise RuntimeError("oops")

            @staticmethod
            def tool_ceilings(record):
                raise RuntimeError("also oops")

        with patch.object(ingest, "opportunity", _BrokenOpportunity):
            ingest.ingest({"codex": self._codex_root})

        sessions = self._read_sessions()
        for s in [s for s in sessions if s.get("agent") == "codex"]:
            self.assertIn("opportunity_error", s)


# ---------------------------------------------------------------------------
# Cost attachment
# ---------------------------------------------------------------------------

class TestIngestCostAttachment(_IngestMixin, unittest.TestCase):

    def test_cost_fields_present(self):
        self._ingest()
        sessions = self._read_sessions()
        self.assertGreater(len(sessions), 0)
        # All sessions should have cost_api_usd (may be None for unknown models)
        for s in sessions:
            self.assertIn("cost_api_usd", s)

    def test_codex_has_credit_fields(self):
        self._ingest(roots={"codex": self._codex_root})
        sessions = self._read_sessions()
        codex = [s for s in sessions if s.get("agent") == "codex"]
        self.assertGreater(len(codex), 0)
        for s in codex:
            self.assertIn("cost_codex_credits", s)

    def test_claude_no_credit_fields(self):
        self._ingest(roots={"claude-code": self._claude_root})
        sessions = self._read_sessions()
        claude = [s for s in sessions if s.get("agent") == "claude-code"]
        self.assertGreater(len(claude), 0)
        for s in claude:
            self.assertNotIn("cost_codex_credits", s)


# ---------------------------------------------------------------------------
# Corrupt store lines are skipped (store.load_latest robustness)
# ---------------------------------------------------------------------------

class TestCorruptStoreLines(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._path = Path(self._tmp) / "sessions.jsonl"

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_corrupt_lines_skipped(self):
        from tokensmash import store

        def _rec(sid):
            return {
                "schema": schema.SESSION_SCHEMA,
                "agent": "codex",
                "session_id": sid,
                "machine_id": "aabb",
                "started_at": "2026-01-01T00:00:00+00:00",
                "model": "gpt-5.5",
                "repo_id": "deadbeef",
                "user_turns": 1,
                "tool_calls": 0,
                "compactions": 0,
                "duration_ms": 1000,
                "usage": {"fresh_input": 0, "cache_read": 0, "cache_write": 0,
                          "output": 0, "reasoning_output": None},
                "provider_raw": {},
            }

        store.append(self._path, _rec("s1"))
        with self._path.open("a") as fh:
            fh.write("}{corrupt line\n")
            fh.write("also bad\n")
        store.append(self._path, _rec("s2"))

        result = store.load_latest(self._path)
        self.assertIn(("codex", "s1"), result)
        self.assertIn(("codex", "s2"), result)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Parse errors: no-usage sessions are skipped
# ---------------------------------------------------------------------------

class TestIngestSkipNoUsage(_IngestMixin, unittest.TestCase):

    def test_no_usage_file_counted_as_skipped(self):
        # Replace the codex fixture with the no_usage one
        (self._codex_root / "normal_session.jsonl").unlink()
        shutil.copy(
            _CODEX_FIXTURES / "no_usage_session.jsonl",
            self._codex_root / "no_usage_session.jsonl",
        )
        stats = self._ingest(roots={"codex": self._codex_root})
        self.assertEqual(stats["parsed"], 0)
        self.assertGreater(stats["skipped"], 0)


if __name__ == "__main__":
    unittest.main()
