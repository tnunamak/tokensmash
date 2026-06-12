"""Tests for tokensmash.study.link.

STUDY_DIR test strategy
-----------------------
We monkeypatch tokensmash.schema.STUDY_DIR (the canonical source) and the
cached reference inside tokensmash.study.link so both modules see the same
tempdir.  link_from_hook reads _schema.STUDY_DIR at call time (not at import
time) so monkeypatching the schema module attribute is sufficient; we also
patch it on the assign module for consistency.

Silence contract
----------------
Every test that calls link_from_hook wraps the call in a redirect of
sys.stdout and sys.stderr and asserts both remain empty afterward.  The
redirect uses io.StringIO so nothing leaks to the real file descriptors.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hook_json(
    session_id: str = "sess-abc",
    cwd: str = "/tmp/myrepo",
    transcript_path: str | None = "/tmp/t.jsonl",
    model: str | None = "claude-opus-4-5",
    source: str | None = None,
) -> str:
    payload: dict = {"session_id": session_id, "cwd": cwd}
    if transcript_path is not None:
        payload["transcript_path"] = transcript_path
    if model is not None:
        payload["model"] = model
    if source is not None:
        payload["source"] = source
    return json.dumps(payload)


class _PatchedStudyDirMixin:
    """Mixin that patches STUDY_DIR on all relevant modules for each test."""

    def setUp(self):
        import shutil
        self._tmpdir = tempfile.mkdtemp()
        self._study_dir = Path(self._tmpdir) / "study"

        import tokensmash.schema as schema_mod
        import tokensmash.study.assign as assign_mod
        import tokensmash.study.link as link_mod

        self._orig_schema_dir = schema_mod.STUDY_DIR
        self._orig_assign_dir = assign_mod.STUDY_DIR

        schema_mod.STUDY_DIR = self._study_dir
        assign_mod.STUDY_DIR = self._study_dir
        # link.py uses _schema.STUDY_DIR at call time (via the alias), so
        # patching schema_mod is sufficient — but keep the assign alias in sync.

    def tearDown(self):
        import shutil
        import tokensmash.schema as schema_mod
        import tokensmash.study.assign as assign_mod

        schema_mod.STUDY_DIR = self._orig_schema_dir
        assign_mod.STUDY_DIR = self._orig_assign_dir
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _init_study(self, study_id: str = "test-study") -> dict:
        from tokensmash.study.assign import init_study
        return init_study(study_id, "log-only", "1.0", study_dir=self._study_dir)

    def _call_link(self, stdin_text: str, agent: str = "claude-code", now: float | None = None) -> tuple[int, str, str]:
        """Call link_from_hook, capture and return (exit_code, stdout, stderr)."""
        from tokensmash.study.link import link_from_hook
        old_out, old_err = sys.stdout, sys.stderr
        buf_out, buf_err = io.StringIO(), io.StringIO()
        sys.stdout = buf_out
        sys.stderr = buf_err
        try:
            rc = link_from_hook(stdin_text, agent, now=now)
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        return rc, buf_out.getvalue(), buf_err.getvalue()


class TestLinkFromHookNoConfig(_PatchedStudyDirMixin, unittest.TestCase):
    """When no study config exists, link_from_hook is a silent no-op."""

    def test_returns_zero(self):
        rc, out, err = self._call_link(_make_hook_json())
        self.assertEqual(rc, 0)

    def test_stdout_empty(self):
        _, out, _ = self._call_link(_make_hook_json())
        self.assertEqual(out, "")

    def test_stderr_empty(self):
        _, _, err = self._call_link(_make_hook_json())
        self.assertEqual(err, "")

    def test_no_files_created(self):
        self._call_link(_make_hook_json())
        self.assertFalse((self._study_dir / "assignments.jsonl").exists())
        self.assertFalse((self._study_dir / "errors.log").exists())


class TestLinkFromHookMalformedJSON(_PatchedStudyDirMixin, unittest.TestCase):
    """Malformed JSON must exit 0, log to errors.log, and be silent."""

    def setUp(self):
        super().setUp()
        self._init_study()

    def test_returns_zero_on_malformed_json(self):
        rc, out, err = self._call_link("not valid json }{")
        self.assertEqual(rc, 0)

    def test_stdout_empty_on_malformed_json(self):
        _, out, _ = self._call_link("not valid json }{")
        self.assertEqual(out, "")

    def test_stderr_empty_on_malformed_json(self):
        _, _, err = self._call_link("not valid json }{")
        self.assertEqual(err, "")

    def test_errors_log_written_on_malformed_json(self):
        self._call_link("not valid json }{")
        errors_log = self._study_dir / "errors.log"
        self.assertTrue(errors_log.exists(), "errors.log should be created")
        content = errors_log.read_text()
        self.assertTrue(len(content) > 0, "errors.log should be non-empty")

    def test_assignments_not_written_on_malformed_json(self):
        self._call_link("not valid json }{")
        self.assertFalse((self._study_dir / "assignments.jsonl").exists())


class TestLinkFromHookValid(_PatchedStudyDirMixin, unittest.TestCase):
    """A valid hook payload appends a correct assignment record."""

    def setUp(self):
        super().setUp()
        self._config = self._init_study("study-42")
        self._now = 1_700_000_000.0  # fixed Unix timestamp for determinism

    def _read_assignments(self) -> list[dict]:
        p = self._study_dir / "assignments.jsonl"
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def test_returns_zero(self):
        rc, _, _ = self._call_link(_make_hook_json(), now=self._now)
        self.assertEqual(rc, 0)

    def test_stdout_empty(self):
        _, out, _ = self._call_link(_make_hook_json(), now=self._now)
        self.assertEqual(out, "")

    def test_stderr_empty(self):
        _, _, err = self._call_link(_make_hook_json(), now=self._now)
        self.assertEqual(err, "")

    def test_appends_one_record(self):
        self._call_link(_make_hook_json(session_id="sess-1"), now=self._now)
        records = self._read_assignments()
        self.assertEqual(len(records), 1)

    def test_schema_field(self):
        from tokensmash.schema import ASSIGNMENT_SCHEMA
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["schema"], ASSIGNMENT_SCHEMA)

    def test_study_id_field(self):
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["study_id"], "study-42")

    def test_agent_field(self):
        self._call_link(_make_hook_json(), agent="codex", now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["agent"], "codex")

    def test_session_id_field(self):
        self._call_link(_make_hook_json(session_id="my-session"), now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["session_id"], "my-session")

    def test_block_field(self):
        from tokensmash.study.assign import block_index
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["block"], block_index(self._now))

    def test_arm_field_valid(self):
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        self.assertIn(record["arm"], ("on", "off"))

    def test_assignment_id_field(self):
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        repo_id = record["repo_id"]
        block = record["block"]
        self.assertEqual(record["assignment_id"], f"{repo_id}-{block}")

    def test_mode_field(self):
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["mode"], "log-only")

    def test_linked_at_field_present(self):
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        self.assertIn("linked_at", record)

    def test_source_field_default(self):
        self._call_link(_make_hook_json(), now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["source"], "new")

    def test_transcript_path_field(self):
        self._call_link(_make_hook_json(transcript_path="/home/user/t.jsonl"), now=self._now)
        record = self._read_assignments()[0]
        self.assertEqual(record["transcript_path"], "/home/user/t.jsonl")

    def test_multiple_sessions_append(self):
        self._call_link(_make_hook_json(session_id="s1"), now=self._now)
        self._call_link(_make_hook_json(session_id="s2"), now=self._now)
        records = self._read_assignments()
        self.assertEqual(len(records), 2)
        session_ids = {r["session_id"] for r in records}
        self.assertEqual(session_ids, {"s1", "s2"})

    def test_arm_is_deterministic(self):
        """Same session_id / timestamp should produce the same arm both times."""
        self._call_link(_make_hook_json(session_id="det"), now=self._now)
        self._call_link(_make_hook_json(session_id="det"), now=self._now)
        records = self._read_assignments()
        self.assertEqual(records[0]["arm"], records[1]["arm"])
        self.assertEqual(records[0]["block"], records[1]["block"])


class TestLinkFromHookResumedSource(_PatchedStudyDirMixin, unittest.TestCase):
    """source == 'resume' must set resumed = True on the record."""

    def setUp(self):
        super().setUp()
        self._init_study()

    def _read_assignments(self) -> list[dict]:
        p = self._study_dir / "assignments.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def test_resumed_flag_set(self):
        rc, out, err = self._call_link(
            _make_hook_json(source="resume"),
            now=1_700_000_000.0,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertEqual(err, "")
        record = self._read_assignments()[0]
        self.assertTrue(record.get("resumed"), "resumed should be True for source=resume")
        self.assertEqual(record["source"], "resume")

    def test_non_resume_source_has_no_resumed_flag(self):
        self._call_link(_make_hook_json(source="new"), now=1_700_000_000.0)
        record = self._read_assignments()[0]
        self.assertNotIn("resumed", record)


class TestLinkFromHookMissingFields(_PatchedStudyDirMixin, unittest.TestCase):
    """Missing required fields in hook JSON should produce exit 0 + errors.log."""

    def setUp(self):
        super().setUp()
        self._init_study()

    def test_missing_session_id_exits_zero(self):
        payload = json.dumps({"cwd": "/tmp/x"})
        rc, out, err = self._call_link(payload, now=1_700_000_000.0)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertEqual(err, "")
        self.assertTrue((self._study_dir / "errors.log").exists())

    def test_missing_cwd_exits_zero(self):
        payload = json.dumps({"session_id": "s"})
        rc, out, err = self._call_link(payload, now=1_700_000_000.0)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertEqual(err, "")


class TestLinkFromHookInternalException(_PatchedStudyDirMixin, unittest.TestCase):
    """Any unexpected exception must not surface; exit 0, errors.log written."""

    def setUp(self):
        super().setUp()
        self._init_study()

    def test_exception_logged_not_raised(self):
        # Pass a payload that parses as JSON but will trigger an error
        # (empty dict lacks required keys).
        payload = json.dumps({})
        rc, out, err = self._call_link(payload, now=1_700_000_000.0)
        self.assertEqual(rc, 0)
        self.assertEqual(out, "")
        self.assertEqual(err, "")
        self.assertTrue((self._study_dir / "errors.log").exists())


if __name__ == "__main__":
    unittest.main()
