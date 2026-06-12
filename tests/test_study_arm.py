"""Tests for live-mode arm resolution (arm_for_cwd) and actuation logging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tokensmash.schema as schema
from tokensmash.study import assign


class TestArmForCwd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.study_dir = Path(self._tmp.name) / "study"
        self._orig = schema.STUDY_DIR
        schema.STUDY_DIR = self.study_dir
        self.cwd = Path(self._tmp.name) / "some-repo"
        self.cwd.mkdir()

    def tearDown(self):
        schema.STUDY_DIR = self._orig
        self._tmp.cleanup()

    def test_no_config_returns_none(self):
        self.assertIsNone(assign.arm_for_cwd(self.cwd, now=1_000_000))

    def test_log_only_mode_returns_none(self):
        assign.init_study("s1", "log-only", "0.1.0", study_dir=self.study_dir)
        self.assertIsNone(assign.arm_for_cwd(self.cwd, now=1_000_000))

    def test_live_mode_resolves_and_matches_link_path(self):
        config = assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        resolution = assign.arm_for_cwd(self.cwd, now=1_000_000)
        self.assertIsNotNone(resolution)
        self.assertIn(resolution["arm"], ("on", "off"))
        # Must agree exactly with the linkage-path computation.
        expected = assign.arm_for(
            bytes.fromhex(config["seed"]),
            schema.stable_id("repo", schema.repo_identity(self.cwd)),
            assign.block_index(1_000_000),
        )
        self.assertEqual(resolution["arm"], expected)

    def test_excluded_repo_is_off(self):
        config = assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        repo_id = schema.stable_id("repo", schema.repo_identity(self.cwd))
        config["exclude_repo_ids"] = [repo_id]
        (self.study_dir / "config.json").write_text(json.dumps(config))
        resolution = assign.arm_for_cwd(self.cwd, now=1_000_000)
        self.assertEqual(resolution["arm"], "off")

    def test_log_actuation_appends_and_never_raises(self):
        assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        resolution = assign.arm_for_cwd(self.cwd, now=1_000_000)
        assign.log_actuation(resolution, tool="headroom", agent_command="claude")
        lines = (self.study_dir / "actuations.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["tool"], "headroom")
        self.assertEqual(record["arm"], resolution["arm"])
        # Unwritable dir must not raise.
        assign.log_actuation(resolution, tool="headroom", agent_command="claude", study_dir=Path("/proc/nope"))


if __name__ == "__main__":
    unittest.main()
