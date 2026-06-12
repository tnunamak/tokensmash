"""Tests for tokensmash.study.launchctl (contract §10).

Monkeypatching pattern follows tests/test_study_arm.py:
- patch tokensmash.schema.STUDY_DIR so arm_for_cwd / load_study_config see the
  tempdir study.
- use TOKENSMASH_CLAUDE_SETTINGS / TOKENSMASH_CODEX_HOOKS env vars so
  install_report uses tempdir files instead of real config.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tokensmash.schema as schema
from tokensmash.study import assign, launchctl


class LaunchctlBase(unittest.TestCase):
    """Base with tempdir study_dir and a fake 'claude' binary."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.study_dir = Path(self._tmp.name) / "study"
        self._orig_study_dir = schema.STUDY_DIR
        schema.STUDY_DIR = self.study_dir

        self.cwd = Path(self._tmp.name) / "some-repo"
        self.cwd.mkdir()

        # Create a fake 'claude' binary so shutil.which resolves it.
        self.fake_bin_dir = Path(self._tmp.name) / "bin"
        self.fake_bin_dir.mkdir()
        for name in ("claude", "codex", "headroom"):
            p = self.fake_bin_dir / name
            p.write_text("#!/bin/sh\nexec $0 $@\n")
            p.chmod(0o755)

        # Patch PATH so shutil.which finds our fakes first.
        self._orig_path = os.environ.get("PATH", "")
        os.environ["PATH"] = str(self.fake_bin_dir) + os.pathsep + self._orig_path

        # Remove recursion guard from the environment for each test (tests
        # that need it set will set it themselves).
        self._orig_active = os.environ.pop("TOKENSMASH_LAUNCH_ACTIVE", None)

    def tearDown(self):
        schema.STUDY_DIR = self._orig_study_dir
        os.environ["PATH"] = self._orig_path
        if self._orig_active is not None:
            os.environ["TOKENSMASH_LAUNCH_ACTIVE"] = self._orig_active
        else:
            os.environ.pop("TOKENSMASH_LAUNCH_ACTIVE", None)
        self._tmp.cleanup()


class TestResolveLaunchRecursionGuard(LaunchctlBase):
    """TOKENSMASH_LAUNCH_ACTIVE set → plain exec immediately."""

    def test_active_guard_plain_exec(self):
        os.environ["TOKENSMASH_LAUNCH_ACTIVE"] = "1"
        assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        d = launchctl.resolve_launch("claude", ["--version"], study_dir=self.study_dir)
        self.assertIsNotNone(d["exec"])
        self.assertEqual(d["arm"], None)
        self.assertIsNone(d["resolution"])
        self.assertNotIn("error", d)

    def test_active_env_always_set_in_returned_env(self):
        """TOKENSMASH_LAUNCH_ACTIVE=1 must be in env even when guard fires."""
        os.environ["TOKENSMASH_LAUNCH_ACTIVE"] = "1"
        d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        self.assertEqual(d["env"].get("TOKENSMASH_LAUNCH_ACTIVE"), "1")


class TestResolveLaunchNoStudy(LaunchctlBase):
    """No config → plain exec, no error."""

    def test_no_config_plain_exec(self):
        d = launchctl.resolve_launch("claude", ["--help"], study_dir=self.study_dir)
        self.assertIsNotNone(d["exec"])
        self.assertIn("claude", d["exec"][0])
        self.assertNotIn("headroom", d["exec"])
        self.assertEqual(d["arm"], None)
        self.assertEqual(d["env"].get("TOKENSMASH_LAUNCH_ACTIVE"), "1")
        self.assertNotIn("error", d)

    def test_log_only_mode_plain_exec(self):
        assign.init_study("s1", "log-only", "0.1.0", study_dir=self.study_dir)
        d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        self.assertIsNone(d["resolution"])
        self.assertEqual(d["arm"], None)
        self.assertNotIn("headroom", d["exec"])


class TestResolveLaunchArmOn(LaunchctlBase):
    """Live study, arm forced to 'on' → headroom wrap argv."""

    def _force_arm_on(self):
        """Write a study config and patch arm_for_cwd to return arm='on'."""
        assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)

    def test_arm_on_uses_on_command_template(self):
        self._force_arm_on()
        with mock.patch.object(
            assign,
            "arm_for_cwd",
            return_value={
                "arm": "on",
                "repo_id": "aabbccdd00112233",
                "block": 500,
                "assignment_id": "aabbccdd00112233-500",
                "study_id": "s1",
                "mode": "live",
            },
        ):
            d = launchctl.resolve_launch("claude", ["--resume"], study_dir=self.study_dir)
        self.assertEqual(d["arm"], "on")
        self.assertIsNotNone(d["exec"])
        # on_command: ["headroom", "wrap", "<tool>", "--no-serena", "--", "--resume"]
        self.assertEqual(d["exec"][0], "headroom")
        self.assertIn("claude", d["exec"])
        self.assertIn("--no-serena", d["exec"])
        self.assertIn("--", d["exec"])
        self.assertIn("--resume", d["exec"])
        self.assertEqual(d["env"].get("TOKENSMASH_LAUNCH_ACTIVE"), "1")

    def test_arm_on_sets_registry_env(self):
        """Registry env (HEADROOM_TELEMETRY=off) merges into decision env."""
        self._force_arm_on()
        with mock.patch.object(
            assign,
            "arm_for_cwd",
            return_value={
                "arm": "on",
                "repo_id": "aabbccdd00112233",
                "block": 500,
                "assignment_id": "aabbccdd00112233-500",
                "study_id": "s1",
                "mode": "live",
            },
        ):
            d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        self.assertEqual(d["env"].get("HEADROOM_TELEMETRY"), "off")

    def test_arm_on_logs_actuation(self):
        self._force_arm_on()
        with mock.patch.object(
            assign,
            "arm_for_cwd",
            return_value={
                "arm": "on",
                "repo_id": "aabbccdd00112233",
                "block": 500,
                "assignment_id": "aabbccdd00112233-500",
                "study_id": "s1",
                "mode": "live",
            },
        ):
            launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        log_path = self.study_dir / "actuations.jsonl"
        self.assertTrue(log_path.exists())
        record = json.loads(log_path.read_text().strip())
        self.assertEqual(record["arm"], "on")
        self.assertEqual(record["agent_command"], "claude")


class TestResolveLaunchArmOff(LaunchctlBase):
    """Live study, arm = 'off' → plain exec of real binary."""

    def test_arm_off_plain_exec(self):
        assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        with mock.patch.object(
            assign,
            "arm_for_cwd",
            return_value={
                "arm": "off",
                "repo_id": "aabbccdd00112233",
                "block": 500,
                "assignment_id": "aabbccdd00112233-500",
                "study_id": "s1",
                "mode": "live",
            },
        ):
            d = launchctl.resolve_launch("claude", ["--help"], study_dir=self.study_dir)
        self.assertEqual(d["arm"], "off")
        self.assertNotIn("headroom", d["exec"][0])
        self.assertIn("claude", d["exec"][0])
        self.assertIn("--help", d["exec"])
        self.assertEqual(d["env"].get("TOKENSMASH_LAUNCH_ACTIVE"), "1")

    def test_arm_off_logs_actuation(self):
        assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        with mock.patch.object(
            assign,
            "arm_for_cwd",
            return_value={
                "arm": "off",
                "repo_id": "aabbccdd00112233",
                "block": 500,
                "assignment_id": "aabbccdd00112233-500",
                "study_id": "s1",
                "mode": "live",
            },
        ):
            launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        log_path = self.study_dir / "actuations.jsonl"
        self.assertTrue(log_path.exists())
        record = json.loads(log_path.read_text().strip())
        self.assertEqual(record["arm"], "off")


class TestResolveLaunchEnvMerge(LaunchctlBase):
    """Env merge: TOKENSMASH_LAUNCH_ACTIVE=1 always present; registry env merged."""

    def test_launch_active_always_in_env(self):
        d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        self.assertEqual(d["env"]["TOKENSMASH_LAUNCH_ACTIVE"], "1")

    def test_launch_active_overrides_existing_value(self):
        os.environ["TOKENSMASH_LAUNCH_ACTIVE"] = "0"
        d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        # The module pops it for the no-guard path; returned env must have 1.
        self.assertEqual(d["env"]["TOKENSMASH_LAUNCH_ACTIVE"], "1")


class TestResolveLaunchFailOpen(LaunchctlBase):
    """Missing binary / registry / bad config → plain exec + error note."""

    def test_missing_registry_fail_open(self):
        """If config references a tool whose registry file doesn't exist, fail open."""
        config = assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        # Write a config that references a non-existent tool.
        config["tool"] = "nonexistent-tool-xyz"
        (self.study_dir / "config.json").write_text(json.dumps(config))
        with mock.patch.object(
            assign,
            "arm_for_cwd",
            return_value={
                "arm": "on",
                "repo_id": "aabbccdd00112233",
                "block": 500,
                "assignment_id": "aabbccdd00112233-500",
                "study_id": "s1",
                "mode": "live",
            },
        ):
            d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        # Must still return a dict (not raise), with an error note.
        self.assertIn("error", d)
        self.assertIsNotNone(d)  # never raises

    def test_missing_binary_returns_none_exec_with_error(self):
        """Binary not on PATH → exec=None, error key present, no raise."""
        os.environ["PATH"] = "/nonexistent-path-xyz"
        d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        self.assertIsNone(d["exec"])
        self.assertIn("error", d)
        self.assertIsNotNone(d)  # never raises

    def test_arm_for_cwd_exception_fail_open(self):
        """arm_for_cwd raising → plain exec fallback with error."""
        assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
        with mock.patch.object(assign, "arm_for_cwd", side_effect=RuntimeError("boom")):
            d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
        self.assertIn("error", d)
        self.assertIsNotNone(d)

    def test_resolve_launch_never_raises(self):
        """Catastrophic internal exception must not propagate."""
        with mock.patch("tokensmash.study.launchctl._load_registry", side_effect=Exception("kaboom")):
            with mock.patch.object(
                assign,
                "arm_for_cwd",
                return_value={
                    "arm": "on",
                    "repo_id": "aabbccdd00112233",
                    "block": 500,
                    "assignment_id": "x",
                    "study_id": "s1",
                    "mode": "live",
                },
            ):
                assign.init_study("s1", "live", "0.1.0", study_dir=self.study_dir)
                try:
                    d = launchctl.resolve_launch("claude", [], study_dir=self.study_dir)
                except Exception as exc:
                    self.fail(f"resolve_launch raised: {exc}")
                self.assertIsInstance(d, dict)


# ---------------------------------------------------------------------------
# install_report tests
# ---------------------------------------------------------------------------


class InstallReportBase(unittest.TestCase):
    """Base with tempdir paths for Claude settings and Codex hooks."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.claude_settings = Path(self._tmp.name) / "claude" / "settings.json"
        self.codex_hooks = Path(self._tmp.name) / "codex" / "hooks.json"
        self._orig_env = {}
        for k in ("TOKENSMASH_CLAUDE_SETTINGS", "TOKENSMASH_CODEX_HOOKS"):
            self._orig_env[k] = os.environ.get(k)
        os.environ["TOKENSMASH_CLAUDE_SETTINGS"] = str(self.claude_settings)
        os.environ["TOKENSMASH_CODEX_HOOKS"] = str(self.codex_hooks)

    def tearDown(self):
        for k, v in self._orig_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()


class TestInstallReportCheck(InstallReportBase):
    """Check mode (apply=False)."""

    def test_check_missing_files(self):
        r = launchctl.install_report(apply=False)
        self.assertFalse(r["claude"]["present"])
        self.assertFalse(r["codex"]["present"])
        self.assertEqual(r["claude"]["path"], str(self.claude_settings))
        self.assertEqual(r["codex"]["path"], str(self.codex_hooks))
        # Snippets must be non-empty strings.
        self.assertIsInstance(r["claude"]["snippet"], str)
        self.assertIsInstance(r["codex"]["snippet"], str)

    def test_check_detects_existing_claude_hook(self):
        self.claude_settings.parent.mkdir(parents=True)
        data = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": launchctl._CLAUDE_HOOK_CMD}]}
                ]
            }
        }
        self.claude_settings.write_text(json.dumps(data))
        r = launchctl.install_report(apply=False)
        self.assertTrue(r["claude"]["present"])
        self.assertFalse(r["codex"]["present"])

    def test_check_detects_existing_codex_hook(self):
        self.codex_hooks.parent.mkdir(parents=True)
        data = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": launchctl._CODEX_HOOK_CMD}]}
                ]
            }
        }
        self.codex_hooks.write_text(json.dumps(data))
        r = launchctl.install_report(apply=False)
        self.assertFalse(r["claude"]["present"])
        self.assertTrue(r["codex"]["present"])

    def test_snippet_contains_correct_commands(self):
        r = launchctl.install_report(apply=False)
        self.assertIn(launchctl._CLAUDE_HOOK_CMD, r["claude"]["snippet"])
        self.assertIn(launchctl._CODEX_HOOK_CMD, r["codex"]["snippet"])

    def test_shell_snippet_never_applied(self):
        r = launchctl.install_report(apply=False)
        self.assertIn("shell", r)
        self.assertFalse(r["shell"]["present"])
        self.assertIsInstance(r["shell"]["snippet"], str)


class TestInstallReportApply(InstallReportBase):
    """apply=True edits the two JSON files."""

    def test_apply_creates_missing_claude_settings(self):
        r = launchctl.install_report(apply=True)
        self.assertTrue(r["claude"]["apply_result"]["ok"])
        self.assertTrue(self.claude_settings.exists())
        data = json.loads(self.claude_settings.read_text())
        cmds = [
            h["command"]
            for e in data["hooks"]["SessionStart"]
            for h in e.get("hooks", [])
        ]
        self.assertIn(launchctl._CLAUDE_HOOK_CMD, cmds)

    def test_apply_creates_missing_codex_hooks(self):
        r = launchctl.install_report(apply=True)
        self.assertTrue(r["codex"]["apply_result"]["ok"])
        self.assertTrue(self.codex_hooks.exists())
        data = json.loads(self.codex_hooks.read_text())
        cmds = [
            h["command"]
            for e in data["hooks"]["SessionStart"]
            for h in e.get("hooks", [])
        ]
        self.assertIn(launchctl._CODEX_HOOK_CMD, cmds)

    def test_apply_preserves_existing_entries(self):
        """Existing hooks must not be removed or duplicated."""
        self.claude_settings.parent.mkdir(parents=True)
        existing = {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "some-other-hook"}]}
                ]
            }
        }
        self.claude_settings.write_text(json.dumps(existing))
        launchctl.install_report(apply=True)
        data = json.loads(self.claude_settings.read_text())
        cmds = [
            h["command"]
            for e in data["hooks"]["SessionStart"]
            for h in e.get("hooks", [])
        ]
        self.assertIn("some-other-hook", cmds)
        self.assertIn(launchctl._CLAUDE_HOOK_CMD, cmds)

    def test_apply_twice_no_duplicates(self):
        """Idempotent: applying twice must not produce duplicate entries."""
        launchctl.install_report(apply=True)
        launchctl.install_report(apply=True)
        data = json.loads(self.claude_settings.read_text())
        cmds = [
            h["command"]
            for e in data["hooks"]["SessionStart"]
            for h in e.get("hooks", [])
        ]
        self.assertEqual(cmds.count(launchctl._CLAUDE_HOOK_CMD), 1)

        data2 = json.loads(self.codex_hooks.read_text())
        cmds2 = [
            h["command"]
            for e in data2["hooks"]["SessionStart"]
            for h in e.get("hooks", [])
        ]
        self.assertEqual(cmds2.count(launchctl._CODEX_HOOK_CMD), 1)

    def test_apply_reports_already_present(self):
        launchctl.install_report(apply=True)
        r2 = launchctl.install_report(apply=True)
        self.assertTrue(r2["claude"]["apply_result"].get("already_present"))
        self.assertTrue(r2["codex"]["apply_result"].get("already_present"))

    def test_apply_updates_present_flag(self):
        r = launchctl.install_report(apply=True)
        self.assertTrue(r["claude"]["present"])
        self.assertTrue(r["codex"]["present"])

    def test_apply_never_edits_shell_rc(self):
        """Shell snippet must never be written anywhere automatically."""
        r = launchctl.install_report(apply=True)
        self.assertFalse(r["shell"]["present"])
        # Verify no rc files were created in the temp dir.
        rc_files = list(Path(self._tmp.name).glob("*.rc")) + list(Path(self._tmp.name).glob(".bash*"))
        self.assertEqual(rc_files, [])


class TestInstallReportMalformedJSON(InstallReportBase):
    """Malformed JSON in existing file → refuse with error, do not corrupt."""

    def test_malformed_claude_settings_returns_error_no_corrupt(self):
        self.claude_settings.parent.mkdir(parents=True)
        self.claude_settings.write_text("not valid json {{{")
        r = launchctl.install_report(apply=True)
        self.assertIn("error", r["claude"])
        # File must be unchanged.
        self.assertEqual(self.claude_settings.read_text(), "not valid json {{{")

    def test_malformed_codex_hooks_returns_error_no_corrupt(self):
        self.codex_hooks.parent.mkdir(parents=True)
        self.codex_hooks.write_text("{broken")
        r = launchctl.install_report(apply=True)
        self.assertIn("error", r["codex"])
        self.assertEqual(self.codex_hooks.read_text(), "{broken")

    def test_check_malformed_json_returns_error(self):
        self.claude_settings.parent.mkdir(parents=True)
        self.claude_settings.write_text("}}}")
        r = launchctl.install_report(apply=False)
        self.assertIn("error", r["claude"])
        self.assertFalse(r["claude"]["present"])


if __name__ == "__main__":
    unittest.main()
