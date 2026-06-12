"""Study arm assignment: deterministic permuted-block design.

See docs/CONTRACTS.md §3 and docs/study-architecture.md D4.

Block unit: 2-hour UTC windows (floor(unix_seconds / 7200)).
Permuted blocks of 8 per repo: group = block // 8.  The 8 arms within a group
are a deterministic shuffle of ["on","off"]*4 keyed by
HMAC-SHA256(seed, f"{repo_id}:{group}").  random.Random is seeded from the
HMAC digest bytes so the shuffle is both portable and deterministic.

arm = perm[block % 8]
"""

from __future__ import annotations

import hashlib
import hmac
import json
import random
import secrets
from datetime import datetime, timezone
from pathlib import Path

import tokensmash.schema as _schema

# Re-export so callers can import from here if convenient.
ASSIGNMENT_SCHEMA = _schema.ASSIGNMENT_SCHEMA
STUDY_DIR = _schema.STUDY_DIR

_BLOCK_SECONDS = 7200  # 2 hours
_GROUP_SIZE = 8
_BASE_ARMS = ["on", "off"] * 4  # 8 elements, balanced 4/4


# ---------------------------------------------------------------------------
# Core assignment functions
# ---------------------------------------------------------------------------


def block_index(unix_seconds: float) -> int:
    """Return the 2-hour UTC block index for a Unix timestamp."""
    return int(unix_seconds) // _BLOCK_SECONDS


def arm_for(seed: bytes, repo_id: str, block: int) -> str:
    """Return "on" or "off" for the given seed, repo, and block.

    Uses a permuted-block design: within each group of 8 consecutive blocks
    the 8 arms are a keyed shuffle of ["on","off"]*4, guaranteeing exact 4/4
    balance within every group.
    """
    group = block // _GROUP_SIZE
    msg = f"{repo_id}:{group}".encode()
    digest = hmac.new(seed, msg, hashlib.sha256).digest()
    # Seed Python's random.Random from the HMAC bytes for a portable, deterministic
    # shuffle that does not perturb the global random state.
    rng = random.Random(digest)
    perm = _BASE_ARMS[:]
    rng.shuffle(perm)
    return perm[block % _GROUP_SIZE]


def assignment_id(repo_id: str, block: int) -> str:
    """Stable assignment identifier for a (repo, block) pair."""
    return f"{repo_id}-{block}"


def arm_for_cwd(
    cwd: str | Path,
    now: float | None = None,
    config: dict | None = None,
    study_dir: Path | None = None,
) -> dict | None:
    """Resolve the actuation arm for a working directory at a moment in time.

    Returns None unless a study config exists AND mode == "live" — log-only
    studies must never actuate. Excluded repos resolve to arm "off" so the
    launcher leaves them untouched. Uses the same repo_identity/block/PRF
    path as the link hook, so actuation and linkage can never disagree.
    """
    import time as _time

    config = config if config is not None else load_study_config(study_dir)
    if config is None or config.get("mode") != "live":
        return None
    ts = now if now is not None else _time.time()
    blk = block_index(ts)
    repo_id = _schema.stable_id("repo", _schema.repo_identity(cwd))
    if repo_id in (config.get("exclude_repo_ids") or []):
        arm = "off"
    else:
        arm = arm_for(bytes.fromhex(config["seed"]), repo_id, blk)
    return {
        "arm": arm,
        "repo_id": repo_id,
        "block": blk,
        "assignment_id": assignment_id(repo_id, blk),
        "study_id": config.get("study_id"),
        "mode": config.get("mode"),
    }


def log_actuation(resolution: dict, tool: str, agent_command: str, study_dir: Path | None = None) -> None:
    """Append an actuation record; must never raise (fail-open launcher)."""
    import datetime as _dt

    try:
        sd = study_dir or _schema.STUDY_DIR
        sd.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": "tokensmash-actuation/1",
            "tool": tool,
            "agent_command": agent_command,
            "logged_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            **resolution,
        }
        with (sd / "actuations.jsonl").open("a") as fh:
            fh.write(_schema.dumps_record(record) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Study config helpers
# ---------------------------------------------------------------------------


def load_study_config(study_dir: Path | None = None) -> dict | None:
    """Return the parsed config.json, or None if it does not exist."""
    path = (study_dir or _schema.STUDY_DIR) / "config.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def init_study(
    study_id: str,
    mode: str,
    protocol_version: str,
    study_dir: Path | None = None,
) -> dict:
    """Create STUDY_DIR/config.json and return it.

    Raises FileExistsError if a config already exists (never overwrite a live
    study config; create a new study_id instead).

    Fields written:
        schema, study_id, seed (32-byte hex), mode, protocol_version,
        created_at (ISO-8601 UTC), exclude_repo_ids ([])
    """
    sd = study_dir or _schema.STUDY_DIR
    sd.mkdir(parents=True, exist_ok=True)
    config_path = sd / "config.json"
    if config_path.exists():
        raise FileExistsError(
            f"Study config already exists at {config_path}. "
            "Delete it manually or start a new study with a different study_id."
        )
    seed = secrets.token_bytes(32)
    config = {
        "schema": "tokensmash-study-config/1",
        "study_id": study_id,
        "seed": seed.hex(),
        "mode": mode,
        "protocol_version": protocol_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exclude_repo_ids": [],
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config
