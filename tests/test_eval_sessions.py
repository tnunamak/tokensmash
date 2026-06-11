import argparse
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokensmash.cli import audit_run_methodology, aggregate_rows, comparison_rows, eval_sessions, prepare_variant_codex_home  # noqa: E402


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
                    {
                        "type": "event_msg",
                        "payload": {
                            "type": "token_count",
                            "info": {
                                "total_token_usage": {
                                    "input_tokens": 900,
                                    "cached_input_tokens": 300,
                                    "output_tokens": 50,
                                    "reasoning_output_tokens": 25,
                                    "total_tokens": 1275,
                                }
                            },
                        },
                    },
                ],
            )
            args = argparse.Namespace(
                agent="codex",
                codex_root=str(sessions),
                repo_root=str(repo),
                latest=1,
                session=[str(sessions)],
                sample_name="fixture sample",
                tools="rtk,context-mode,headroom",
                semmap_bin="semmap",
                headroom_perf=None,
                no_artifacts=True,
                output=None,
            )

            result = eval_sessions(args)
            rows = {row["tool_id"]: row for row in result["rows"]}

            self.assertEqual(result["sample"]["name"], "fixture sample")
            self.assertEqual(result["sample"]["session_count"], 1)
            self.assertEqual(result["sample"]["total_tokens"], 1275)
            self.assertEqual(result["sample"]["cached_input_tokens"], 300)
            self.assertEqual(result["sample"]["non_cached_tokens"], 975)
            self.assertEqual(rows["rtk"]["sample_count"], 1)
            self.assertEqual(rows["rtk"]["token_pressure_before"], 1000)
            self.assertEqual(rows["context-mode"]["sample_count"], 1)
            self.assertEqual(rows["context-mode"]["token_pressure_before"], 1024)
            self.assertEqual(rows["headroom"]["mechanism_fired"], "no")
            self.assertEqual(rows["headroom"]["future_session_confidence_percent"], 0)

    def test_comparison_rows_include_non_cached_spend(self) -> None:
        result = {
            "baseline": "baseline",
            "runs": [
                {
                    "success": True,
                    "task_id": "task",
                    "replicate": 1,
                    "variant_id": "baseline",
                    "token_total": 1000,
                    "token_usage": {"total_tokens": 1000, "cached_input_tokens": 700},
                },
                {
                    "success": True,
                    "task_id": "task",
                    "replicate": 1,
                    "variant_id": "rtk",
                    "token_total": 800,
                    "token_usage": {"input_tokens": 760, "cached_input_tokens": 650, "output_tokens": 40, "total_tokens": 800},
                    "mechanism_checks": {"required": True, "ok": True, "checks": []},
                },
            ],
        }

        rows = comparison_rows(result, "baseline")

        self.assertEqual(rows[0][0], "rtk")
        self.assertEqual(rows[0][1], "1,000")
        self.assertEqual(rows[0][2], "800")
        self.assertEqual(rows[0][3], "+20.0%")
        self.assertEqual(rows[0][4], "300")
        self.assertEqual(rows[0][5], "150")
        self.assertEqual(rows[0][6], "+50.0%")

    def test_aggregate_rows_sum_paired_successful_runs(self) -> None:
        result_one = {
            "baseline": "baseline",
            "runs": [
                {
                    "success": True,
                    "task_id": "task-a",
                    "replicate": 1,
                    "variant_id": "baseline",
                    "token_total": 1000,
                    "token_usage": {"input_tokens": 950, "cached_input_tokens": 700, "output_tokens": 50, "total_tokens": 1000},
                },
                {
                    "success": True,
                    "task_id": "task-a",
                    "replicate": 1,
                    "variant_id": "rtk",
                    "token_total": 800,
                    "token_usage": {"input_tokens": 760, "cached_input_tokens": 650, "output_tokens": 40, "total_tokens": 800},
                    "mechanism_checks": {"required": True, "ok": True, "checks": []},
                },
            ],
        }
        result_two = {
            "baseline": "baseline",
            "runs": [
                {
                    "success": True,
                    "task_id": "task-b",
                    "replicate": 1,
                    "variant_id": "baseline",
                    "token_total": 2000,
                    "token_usage": {"input_tokens": 1900, "cached_input_tokens": 1000, "output_tokens": 100, "total_tokens": 2000},
                },
                {
                    "success": True,
                    "task_id": "task-b",
                    "replicate": 1,
                    "variant_id": "rtk",
                    "token_total": 2200,
                    "token_usage": {"input_tokens": 2050, "cached_input_tokens": 900, "output_tokens": 150, "total_tokens": 2200},
                    "mechanism_checks": {"required": True, "ok": True, "checks": []},
                },
            ],
        }

        rows = aggregate_rows([result_one, result_two])

        self.assertEqual(rows[0][0], "rtk")
        self.assertEqual(rows[0][1], "2")
        self.assertEqual(rows[0][2], "3,000")
        self.assertEqual(rows[0][3], "3,000")
        self.assertEqual(rows[0][4], "+0.0%")
        self.assertEqual(rows[0][5], "1,300")
        self.assertEqual(rows[0][6], "1,450")
        self.assertEqual(rows[0][7], "-11.5%")
        self.assertEqual(rows[0][12], "1/2")
        self.assertEqual(rows[0][14], "pass")

    def test_isolated_codex_home_sets_home_under_case_dir(self) -> None:
        with TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            repo_dir = case_dir / "repo"
            repo_dir.mkdir(parents=True)
            env = {}

            prepare_variant_codex_home(
                {"codex_home": True, "isolated_home": True},
                case_dir,
                repo_dir,
                env,
            )

            self.assertEqual(env["HOME"], str(case_dir / "home"))
            self.assertEqual(env["CODEX_HOME"], str(case_dir / "home" / ".codex"))
            self.assertTrue((case_dir / "home" / ".codex" / "auth.json").exists())

    def test_methodology_audit_requires_isolated_session_path(self) -> None:
        with TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "case"
            session = case_dir / "home" / ".codex" / "sessions" / "session.jsonl"
            session.parent.mkdir(parents=True)
            session.write_text("{}\n")
            result = {
                "agent_exit_code": 0,
                "agent_timed_out": False,
                "case_dir": str(case_dir),
                "session_path": str(session),
                "success_commands": [{"exit_code": 0}],
                "token_total": 100,
                "token_usage": {
                    "input_tokens": 90,
                    "cached_input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 100,
                },
                "variant_id": "baseline",
            }

            audit = audit_run_methodology(result, {"isolated_home": True}, "baseline")

            self.assertTrue(audit["ok"])

            result["session_path"] = str(Path(tmp) / "outside.jsonl")
            audit = audit_run_methodology(result, {"isolated_home": True}, "baseline")

            self.assertFalse(audit["ok"])


if __name__ == "__main__":
    unittest.main()
