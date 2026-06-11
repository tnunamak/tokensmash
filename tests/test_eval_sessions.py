import argparse
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokensmash.cli import eval_sessions  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


class EvalSessionsTest(unittest.TestCase):
    def test_codex_pressure_rows_require_real_mechanisms(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            sessions = root / "sessions"
            repo.mkdir()
            sessions.mkdir()
            (repo / "main.py").write_text("print('hello')\n" * 100)
            session = sessions / "session.jsonl"
            write_jsonl(
                session,
                [
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "rtk read main.py"}),
                            "call_id": "rtk-call",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "rtk-call",
                            "output": "Original token count: 1000\nOutput:\nshort",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "echo rtk should not count"}),
                            "call_id": "noise-call",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "noise-call",
                            "output": "Original token count: 5000\nOutput:\nrtk in prose only",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "ctx_batch_execute",
                            "namespace": "mcp__context_mode__",
                            "arguments": "{}",
                            "call_id": "ctx-call",
                        },
                    },
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "ctx-call",
                            "output": "Executed 2 commands (40 lines, 4.0KB). Indexed 2 sections.",
                        },
                    },
                ],
            )
            args = argparse.Namespace(
                agent="codex",
                codex_root=str(sessions),
                repo_root=str(repo),
                latest=1,
                session=[],
                tools="rtk,context-mode,headroom",
                semmap_bin="semmap",
                headroom_perf=None,
                no_artifacts=True,
                output=None,
            )

            result = eval_sessions(args)
            rows = {row["tool_id"]: row for row in result["rows"]}

            self.assertEqual(rows["rtk"]["sample_count"], 1)
            self.assertEqual(rows["rtk"]["token_pressure_before"], 1000)
            self.assertEqual(rows["context-mode"]["sample_count"], 1)
            self.assertEqual(rows["context-mode"]["token_pressure_before"], 1024)
            self.assertEqual(rows["headroom"]["mechanism_fired"], "no")
            self.assertEqual(rows["headroom"]["future_session_confidence_percent"], 0)


if __name__ == "__main__":
    unittest.main()
