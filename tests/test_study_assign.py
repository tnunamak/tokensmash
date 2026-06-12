"""Tests for tokensmash.study.assign.

STUDY_DIR test strategy
-----------------------
schema.STUDY_DIR is a module-level constant computed at import time from the
TOKENSMASH_STUDY_DIR environment variable.  We monkeypatch the attribute on
the schema module *and* on the assign module (which imports from schema) so
that both references see the same tempdir.  This is simpler than juggling
importlib.reload and avoids polluting ~/.local/state.

Each test that touches the filesystem uses a fresh tempdir via setUp /
teardown so there is no inter-test state.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import random
import secrets
import tempfile
import time
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed() -> bytes:
    return secrets.token_bytes(32)


def _expected_arm(seed: bytes, repo_id: str, block: int) -> str:
    """Reference implementation that mirrors the production logic."""
    group = block // 8
    msg = f"{repo_id}:{group}".encode()
    digest = hmac.new(seed, msg, hashlib.sha256).digest()
    rng = random.Random(digest)
    perm = ["on", "off"] * 4
    rng.shuffle(perm)
    return perm[block % 8]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBlockIndex(unittest.TestCase):
    def _bi(self, ts):
        from tokensmash.study.assign import block_index
        return block_index(ts)

    def test_zero(self):
        self.assertEqual(self._bi(0), 0)

    def test_boundary_exactly(self):
        # 7200 seconds = start of block 1
        self.assertEqual(self._bi(7200), 1)
        self.assertEqual(self._bi(7200.0), 1)

    def test_just_before_boundary(self):
        self.assertEqual(self._bi(7199.9), 0)

    def test_arbitrary(self):
        ts = 1_700_000_000
        self.assertEqual(self._bi(ts), ts // 7200)

    def test_group_boundary(self):
        # Block 8 starts a new group (group 1)
        self.assertEqual(self._bi(8 * 7200), 8)

    def test_large_timestamp(self):
        ts = 9_999_999_999
        self.assertEqual(self._bi(ts), ts // 7200)


class TestArmFor(unittest.TestCase):
    def setUp(self):
        from tokensmash.study import assign
        self.arm_for = assign.arm_for

    def test_returns_on_or_off(self):
        seed = _make_seed()
        for block in range(32):
            arm = self.arm_for(seed, "repo-abc", block)
            self.assertIn(arm, ("on", "off"))

    def test_deterministic_same_call(self):
        seed = _make_seed()
        arm1 = self.arm_for(seed, "repo-x", 42)
        arm2 = self.arm_for(seed, "repo-x", 42)
        self.assertEqual(arm1, arm2)

    def test_deterministic_matches_reference(self):
        seed = _make_seed()
        for block in range(64):
            self.assertEqual(
                self.arm_for(seed, "my-repo", block),
                _expected_arm(seed, "my-repo", block),
            )

    def test_exact_balance_within_every_group(self):
        """Every group of 8 consecutive blocks must have exactly 4 on and 4 off."""
        seed = _make_seed()
        repo = "balance-test-repo"
        n_groups = 20
        for g in range(n_groups):
            blocks = list(range(g * 8, (g + 1) * 8))
            arms = [self.arm_for(seed, repo, b) for b in blocks]
            on_count = arms.count("on")
            off_count = arms.count("off")
            self.assertEqual(on_count, 4, f"group {g}: {arms}")
            self.assertEqual(off_count, 4, f"group {g}: {arms}")

    def test_balance_for_200_random_seed_repo_draws(self):
        """Balance holds across 200 random (seed, repo_id) combinations."""
        rng = random.Random(0xDEADBEEF)
        failures = []
        for _ in range(200):
            seed = secrets.token_bytes(32)
            repo = f"repo-{rng.randint(0, 10**9)}"
            n_groups = 5
            for g in range(n_groups):
                blocks = list(range(g * 8, (g + 1) * 8))
                arms = [self.arm_for(seed, repo, b) for b in blocks]
                on_count = arms.count("on")
                if on_count != 4:
                    failures.append((repo, g, arms))
        self.assertEqual(failures, [], f"Balance failures (first 3): {failures[:3]}")

    def test_different_repos_differ(self):
        """Different repos should produce different permutations (statistically)."""
        seed = _make_seed()
        repos = [f"repo-{i}" for i in range(50)]
        # For block 0 (group 0), collect arms; expect both on and off to appear.
        arms = [self.arm_for(seed, r, 0) for r in repos]
        self.assertIn("on", arms, "All 50 repos have the same arm — very unlikely")
        self.assertIn("off", arms, "All 50 repos have the same arm — very unlikely")

    def test_different_groups_differ(self):
        """Different groups should produce different permutations (statistically)."""
        seed = _make_seed()
        repo = "one-repo"
        # Sample first block of each of 50 groups
        arms = [self.arm_for(seed, repo, g * 8) for g in range(50)]
        self.assertIn("on", arms)
        self.assertIn("off", arms)

    def test_does_not_perturb_global_rng(self):
        """arm_for must not affect the global random state."""
        seed = _make_seed()
        random.seed(12345)
        before = random.random()
        random.seed(12345)
        self.arm_for(seed, "repo", 0)
        after = random.random()
        self.assertEqual(before, after)


class TestAssignmentId(unittest.TestCase):
    def test_format(self):
        from tokensmash.study.assign import assignment_id
        result = assignment_id("abc123", 7)
        self.assertEqual(result, "abc123-7")

    def test_deterministic(self):
        from tokensmash.study.assign import assignment_id
        self.assertEqual(assignment_id("r", 0), assignment_id("r", 0))


class TestInitAndLoadStudyConfig(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._study_dir = Path(self._tmpdir) / "study"
        # Monkeypatch both schema and assign so they share the same path.
        import tokensmash.schema as schema_mod
        import tokensmash.study.assign as assign_mod
        self._orig_schema_dir = schema_mod.STUDY_DIR
        self._orig_assign_dir = assign_mod.STUDY_DIR
        schema_mod.STUDY_DIR = self._study_dir
        assign_mod.STUDY_DIR = self._study_dir

    def tearDown(self):
        import tokensmash.schema as schema_mod
        import tokensmash.study.assign as assign_mod
        schema_mod.STUDY_DIR = self._orig_schema_dir
        assign_mod.STUDY_DIR = self._orig_assign_dir
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_load_returns_none_when_absent(self):
        from tokensmash.study.assign import load_study_config
        self.assertIsNone(load_study_config(self._study_dir))

    def test_init_creates_config(self):
        from tokensmash.study.assign import init_study, load_study_config
        cfg = init_study("s1", "log-only", "1.0", study_dir=self._study_dir)
        self.assertEqual(cfg["study_id"], "s1")
        self.assertEqual(cfg["mode"], "log-only")
        self.assertEqual(cfg["protocol_version"], "1.0")
        self.assertIn("seed", cfg)
        self.assertEqual(len(cfg["seed"]), 64)  # 32 bytes = 64 hex chars
        self.assertEqual(cfg["exclude_repo_ids"], [])
        # Persisted and loadable.
        loaded = load_study_config(self._study_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["study_id"], "s1")

    def test_init_refuses_overwrite(self):
        from tokensmash.study.assign import init_study
        init_study("s1", "log-only", "1.0", study_dir=self._study_dir)
        with self.assertRaises(FileExistsError):
            init_study("s2", "live", "1.1", study_dir=self._study_dir)

    def test_seed_is_32_bytes_hex(self):
        from tokensmash.study.assign import init_study
        cfg = init_study("s1", "live", "1.0", study_dir=self._study_dir)
        seed_bytes = bytes.fromhex(cfg["seed"])
        self.assertEqual(len(seed_bytes), 32)

    def test_each_init_generates_unique_seed(self):
        """Two fresh studies should have different seeds."""
        from tokensmash.study.assign import init_study
        import shutil
        cfg1 = init_study("s1", "log-only", "1.0", study_dir=self._study_dir)
        # Remove config to allow a second init.
        (self._study_dir / "config.json").unlink()
        cfg2 = init_study("s2", "log-only", "1.0", study_dir=self._study_dir)
        self.assertNotEqual(cfg1["seed"], cfg2["seed"])


class TestDeterminismAcrossProcesses(unittest.TestCase):
    """Verify determinism by re-running the reference formula independently."""

    def test_arm_matches_reference_formula(self):
        from tokensmash.study.assign import arm_for
        for _ in range(100):
            seed = _make_seed()
            repo = f"repo-{secrets.token_hex(4)}"
            for blk in [0, 1, 7, 8, 15, 16, 23, 63, 64, 127]:
                self.assertEqual(
                    arm_for(seed, repo, blk),
                    _expected_arm(seed, repo, blk),
                    f"Mismatch: seed={seed.hex()[:8]} repo={repo} block={blk}",
                )


if __name__ == "__main__":
    unittest.main()
