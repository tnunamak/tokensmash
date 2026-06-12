"""Regression tests for transcript identity (one transcript file = one record).

Covers two real-world collisions discovered against live data:
- Claude subagent transcripts share the parent's sessionId; keying by
  session_id silently dropped subagent spend.
- Resumed Codex sessions write multiple rollout files with the same internal
  session id and CUMULATIVE token counters; summing rollouts double-counts.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tokensmash.schema as schema
from tokensmash import ingest as ingest_mod
from tokensmash import store


def _claude_transcript(session_id: str, n_assistant: int = 1) -> str:
    lines = []
    for i in range(n_assistant):
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": f"{session_id}-msg-{i}",
                    "sessionId": session_id,
                    "timestamp": f"2026-06-01T10:0{i}:00.000Z",
                    "cwd": "/tmp/fake-repo",
                    "version": "2.0.0",
                    "message": {
                        "model": "claude-sonnet-4-6",
                        "usage": {
                            "input_tokens": 100,
                            "cache_read_input_tokens": 50,
                            "cache_creation_input_tokens": 10,
                            "output_tokens": 20,
                        },
                    },
                }
            )
        )
    return "\n".join(lines) + "\n"


def _codex_transcript(session_id: str, total_tokens: int) -> str:
    lines = [
        json.dumps(
            {
                "timestamp": "2026-06-01T10:00:00.000Z",
                "type": "session_meta",
                "payload": {"id": session_id, "cwd": "/tmp/fake-repo", "cli_version": "1.0.0"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-06-01T10:00:01.000Z",
                "type": "turn_context",
                "payload": {"model": "gpt-5.5", "cwd": "/tmp/fake-repo"},
            }
        ),
        json.dumps(
            {
                "timestamp": "2026-06-01T10:05:00.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "total_token_usage": {
                            "input_tokens": total_tokens - 20,
                            "cached_input_tokens": total_tokens - 120,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 0,
                            "total_tokens": total_tokens,
                        }
                    },
                },
            }
        ),
    ]
    return "\n".join(lines) + "\n"


class TestTranscriptIdentity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self._orig_study_dir = schema.STUDY_DIR
        schema.STUDY_DIR = tmp / "study"
        self.claude_root = tmp / "claude-projects"
        self.codex_root = tmp / "codex-sessions"

    def tearDown(self):
        schema.STUDY_DIR = self._orig_study_dir
        self._tmp.cleanup()

    def _ingest(self):
        return ingest_mod.ingest(
            {"codex": self.codex_root, "claude-code": self.claude_root}
        )

    def test_claude_subagents_sharing_session_id_all_stored(self):
        project = self.claude_root / "-tmp-fake-repo"
        subagents = project / "abc-123" / "subagents"
        subagents.mkdir(parents=True)
        (project / "abc-123.jsonl").write_text(_claude_transcript("abc-123", 2))
        (subagents / "agent-one.jsonl").write_text(_claude_transcript("abc-123"))
        (subagents / "agent-two.jsonl").write_text(_claude_transcript("abc-123"))

        stats = self._ingest()
        self.assertEqual(stats["added"], 3)

        records = list(
            store.load_latest(
                schema.STUDY_DIR / "sessions.jsonl", key=("agent", "transcript_id")
            ).values()
        )
        self.assertEqual(len(records), 3)
        self.assertEqual(len({r["transcript_id"] for r in records}), 3)
        self.assertEqual({r["logical_session_id"] for r in records}, {"abc-123"})
        # Spend sums across all three transcripts (200+100+100 fresh).
        self.assertEqual(sum(r["usage"]["fresh_input"] for r in records), 400)

    def test_codex_superseded_rollouts_excluded_keeper_kept(self):
        day = self.codex_root / "2026" / "06" / "01"
        day.mkdir(parents=True)
        (day / "rollout-a.jsonl").write_text(_codex_transcript("sess-1", 1_000))
        (day / "rollout-b.jsonl").write_text(_codex_transcript("sess-1", 5_000))

        stats = self._ingest()
        self.assertEqual(stats["added"], 2)

        records = list(
            store.load_latest(
                schema.STUDY_DIR / "sessions.jsonl", key=("agent", "transcript_id")
            ).values()
        )
        keepers = [r for r in records if not r.get("excluded")]
        superseded = [r for r in records if r.get("excluded") == "codex-superseded-rollout"]
        self.assertEqual(len(keepers), 1)
        self.assertEqual(len(superseded), 1)
        self.assertEqual(keepers[0]["provider_raw"]["total_tokens"], 5_000)

    def test_reingest_is_idempotent(self):
        project = self.claude_root / "-tmp-fake-repo"
        project.mkdir(parents=True)
        (project / "abc-123.jsonl").write_text(_claude_transcript("abc-123"))
        day = self.codex_root / "2026" / "06" / "01"
        day.mkdir(parents=True)
        (day / "rollout-a.jsonl").write_text(_codex_transcript("sess-1", 1_000))
        (day / "rollout-b.jsonl").write_text(_codex_transcript("sess-1", 5_000))

        first = self._ingest()
        second = self._ingest()
        self.assertEqual(first["added"], 3)
        self.assertEqual(second["added"], 0)
        self.assertEqual(second["replaced"], 0)


if __name__ == "__main__":
    unittest.main()
