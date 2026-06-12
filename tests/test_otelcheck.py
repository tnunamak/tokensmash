"""Tests for tokensmash.otelcheck — OTel second-meter cross-validation.

All OTLP fixtures are synthetic (built in-process; no real session data).

OTLP path under test
--------------------
  Line (JSON object)
    .resourceMetrics[]
      .scopeMetrics[]
        .metrics[]          name == "claude_code.token.usage"
          .sum.dataPoints[]
            .attributes[]   [{key, value:{stringValue}}, ...]
            .asInt / .asDouble
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tokensmash.otelcheck import compare, parse_otlp_jsonl, report

# ---------------------------------------------------------------------------
# Helpers to build synthetic OTLP fixtures
# ---------------------------------------------------------------------------


def _make_dp(
    session_id: str,
    token_type: str,
    value: int,
    use_double: bool = False,
) -> dict:
    """Build one OTLP sum dataPoint for claude_code.token.usage."""
    dp: dict = {
        "attributes": [
            {"key": "type", "value": {"stringValue": token_type}},
            {"key": "session.id", "value": {"stringValue": session_id}},
        ],
    }
    if use_double:
        dp["asDouble"] = float(value)
    else:
        dp["asInt"] = value
    return dp


def _make_line(data_points: list[dict]) -> str:
    """Wrap dataPoints in a minimal ExportMetricsServiceRequest JSON line."""
    obj = {
        "resourceMetrics": [
            {
                "scopeMetrics": [
                    {
                        "metrics": [
                            {
                                "name": "claude_code.token.usage",
                                "sum": {"dataPoints": data_points},
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return json.dumps(obj)


def _write_jsonl(lines: list[str]) -> Path:
    """Write lines to a temp .jsonl file and return the Path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for line in lines:
        tmp.write(line + "\n")
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _make_record(
    session_id: str,
    fresh_input: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    output: int = 0,
) -> dict:
    """Minimal transcript store record for a claude-code session."""
    return {
        "agent": "claude-code",
        "session_id": session_id,
        "usage": {
            "fresh_input": fresh_input,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "output": output,
            "reasoning_output": None,
        },
    }


# ---------------------------------------------------------------------------
# parse_otlp_jsonl
# ---------------------------------------------------------------------------


class TestParseOtlpJsonl(unittest.TestCase):
    def test_single_session_all_four_types(self):
        """All four token types parse into correct canonical fields."""
        dps = [
            _make_dp("sess-1", "input", 100),
            _make_dp("sess-1", "cacheRead", 200),
            _make_dp("sess-1", "cacheCreation", 50),
            _make_dp("sess-1", "output", 75),
        ]
        path = _write_jsonl([_make_line(dps)])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result, {
            "sess-1": {
                "fresh_input": 100,
                "cache_read": 200,
                "cache_write": 50,
                "output": 75,
            }
        })

    def test_multiple_sessions(self):
        """Data points for different session.id values stay separate."""
        dps = [
            _make_dp("sess-A", "input", 10),
            _make_dp("sess-B", "input", 20),
            _make_dp("sess-A", "output", 5),
        ]
        path = _write_jsonl([_make_line(dps)])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result["sess-A"]["fresh_input"], 10)
        self.assertEqual(result["sess-A"]["output"], 5)
        self.assertEqual(result["sess-B"]["fresh_input"], 20)

    def test_multiple_lines_summed(self):
        """Values across multiple JSON-lines are summed per session."""
        line1 = _make_line([_make_dp("sess-1", "input", 100)])
        line2 = _make_line([_make_dp("sess-1", "input", 50)])
        path = _write_jsonl([line1, line2])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result["sess-1"]["fresh_input"], 150)

    def test_as_double_accepted(self):
        """asDouble values are cast to int."""
        dps = [_make_dp("sess-d", "output", 33, use_double=True)]
        path = _write_jsonl([_make_line(dps)])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result["sess-d"]["output"], 33)

    def test_as_int_string_accepted(self):
        """asInt encoded as a JSON string (int64 transport) is accepted."""
        dp = {
            "attributes": [
                {"key": "type", "value": {"stringValue": "input"}},
                {"key": "session.id", "value": {"stringValue": "sess-s"}},
            ],
            "asInt": "9999",
        }
        path = _write_jsonl([_make_line([dp])])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result["sess-s"]["fresh_input"], 9999)

    def test_unknown_type_skipped(self):
        """Data points with an unrecognised type attribute are skipped."""
        dps = [
            _make_dp("sess-1", "unknown_type", 999),
            _make_dp("sess-1", "output", 5),
        ]
        path = _write_jsonl([_make_line(dps)])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result["sess-1"]["output"], 5)
        # No extra keys from unknown type
        self.assertNotIn("unknown_type", result["sess-1"])

    def test_missing_session_id_skipped(self):
        """Data points without session.id are silently dropped."""
        dp = {
            "attributes": [
                {"key": "type", "value": {"stringValue": "input"}},
            ],
            "asInt": 100,
        }
        path = _write_jsonl([_make_line([dp])])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result, {})

    def test_malformed_json_line_skipped(self):
        """Malformed JSON lines are skipped; valid lines still parse."""
        good_line = _make_line([_make_dp("sess-ok", "input", 10)])
        path = _write_jsonl(["NOT JSON {{{", good_line])
        result = parse_otlp_jsonl(path)
        self.assertIn("sess-ok", result)

    def test_empty_file_returns_empty(self):
        """Empty file produces empty dict."""
        path = _write_jsonl([])
        self.assertEqual(parse_otlp_jsonl(path), {})

    def test_missing_file_returns_empty(self):
        """Non-existent file returns empty dict without raising."""
        result = parse_otlp_jsonl(Path("/tmp/__nonexistent_otel_test__.jsonl"))
        self.assertEqual(result, {})

    def test_wrong_metric_name_skipped(self):
        """Metrics with a different name are not parsed."""
        obj = {
            "resourceMetrics": [
                {
                    "scopeMetrics": [
                        {
                            "metrics": [
                                {
                                    "name": "other.metric",
                                    "sum": {
                                        "dataPoints": [
                                            _make_dp("sess-x", "input", 100)
                                        ]
                                    },
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        path = _write_jsonl([json.dumps(obj)])
        self.assertEqual(parse_otlp_jsonl(path), {})

    def test_blank_lines_ignored(self):
        """Blank lines in the file are not treated as errors."""
        good = _make_line([_make_dp("sess-b", "output", 7)])
        path = _write_jsonl(["", good, ""])
        result = parse_otlp_jsonl(path)
        self.assertIn("sess-b", result)

    def test_multiple_resource_scope_metric_nesting(self):
        """Deeply nested OTLP objects (multiple resourceMetrics) are fully traversed."""
        obj = {
            "resourceMetrics": [
                {
                    "scopeMetrics": [
                        {
                            "metrics": [
                                {
                                    "name": "claude_code.token.usage",
                                    "sum": {
                                        "dataPoints": [
                                            _make_dp("sess-n", "input", 10)
                                        ]
                                    },
                                }
                            ]
                        }
                    ]
                },
                {
                    "scopeMetrics": [
                        {
                            "metrics": [
                                {
                                    "name": "claude_code.token.usage",
                                    "sum": {
                                        "dataPoints": [
                                            _make_dp("sess-n", "output", 3)
                                        ]
                                    },
                                }
                            ]
                        }
                    ]
                },
            ]
        }
        path = _write_jsonl([json.dumps(obj)])
        result = parse_otlp_jsonl(path)
        self.assertEqual(result["sess-n"]["fresh_input"], 10)
        self.assertEqual(result["sess-n"]["output"], 3)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare(unittest.TestCase):
    def test_perfect_match(self):
        """Sessions that agree exactly produce zero deltas and no disagreements."""
        otel = {
            "sess-1": {
                "fresh_input": 100,
                "cache_read": 50,
                "cache_write": 10,
                "output": 30,
            }
        }
        records = [_make_record("sess-1", 100, 50, 10, 30)]
        result = compare(otel, records)
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["otel_only"], 0)
        self.assertEqual(result["store_only"], 0)
        self.assertEqual(result["disagreements"], [])
        session = result["sessions"][0]
        for field in ("fresh_input", "cache_read", "cache_write", "output"):
            self.assertEqual(session["fields"][field]["abs_delta"], 0)
            self.assertEqual(session["fields"][field]["rel_delta"], 0.0)

    def test_otel_only_session(self):
        """Session present in OTel but not the store is counted in otel_only."""
        otel = {
            "sess-x": {
                "fresh_input": 10,
                "cache_read": 0,
                "cache_write": 0,
                "output": 5,
            }
        }
        result = compare(otel, [])
        self.assertEqual(result["otel_only"], 1)
        self.assertEqual(result["matched"], 0)

    def test_store_only_session(self):
        """claude-code store session absent from OTel is counted in store_only."""
        result = compare({}, [_make_record("sess-y", 10, 0, 0, 5)])
        self.assertEqual(result["store_only"], 1)
        self.assertEqual(result["matched"], 0)

    def test_codex_records_excluded(self):
        """Codex records are not joined (OTel only covers claude-code)."""
        otel = {
            "sess-c": {
                "fresh_input": 10,
                "cache_read": 0,
                "cache_write": 0,
                "output": 5,
            }
        }
        rec = _make_record("sess-c", 10, 0, 0, 5)
        rec["agent"] = "codex"
        result = compare(otel, [rec])
        # The codex record is ignored → sess-c appears only in OTel
        self.assertEqual(result["otel_only"], 1)
        self.assertEqual(result["store_only"], 0)
        self.assertEqual(result["matched"], 0)

    def test_disagreement_flagged_above_one_percent(self):
        """Sessions with >1% delta on any field are in disagreements."""
        otel = {
            "sess-d": {
                "fresh_input": 102,  # +2% from 100
                "cache_read": 0,
                "cache_write": 0,
                "output": 0,
            }
        }
        records = [_make_record("sess-d", 100, 0, 0, 0)]
        result = compare(otel, records)
        self.assertIn("sess-d", result["disagreements"])

    def test_no_disagreement_at_one_percent(self):
        """Exactly 1% delta (== threshold) does NOT trigger disagreement."""
        otel = {
            "sess-e": {
                "fresh_input": 101,  # +1% from 100 → rel_delta exactly 0.01
                "cache_read": 0,
                "cache_write": 0,
                "output": 0,
            }
        }
        records = [_make_record("sess-e", 100, 0, 0, 0)]
        result = compare(otel, records)
        self.assertNotIn("sess-e", result["disagreements"])

    def test_reasoning_output_none_treated_as_zero(self):
        """reasoning_output=None in store does not crash comparison."""
        otel = {
            "sess-r": {
                "fresh_input": 10,
                "cache_read": 0,
                "cache_write": 0,
                "output": 5,
            }
        }
        rec = _make_record("sess-r", 10, 0, 0, 5)
        rec["usage"]["reasoning_output"] = None
        result = compare(otel, [rec])
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["disagreements"], [])

    def test_delta_direction(self):
        """abs_delta is otel minus store (positive when OTel is higher)."""
        otel = {
            "sess-dir": {
                "fresh_input": 120,
                "cache_read": 0,
                "cache_write": 0,
                "output": 0,
            }
        }
        records = [_make_record("sess-dir", 100, 0, 0, 0)]
        result = compare(otel, records)
        fi = result["sessions"][0]["fields"]["fresh_input"]
        self.assertEqual(fi["abs_delta"], 20)
        self.assertAlmostEqual(fi["rel_delta"], 0.2)

    def test_both_zero_no_disagreement(self):
        """When both OTel and store report zero the rel_delta is 0.0."""
        otel = {
            "sess-0": {
                "fresh_input": 0,
                "cache_read": 0,
                "cache_write": 0,
                "output": 0,
            }
        }
        records = [_make_record("sess-0", 0, 0, 0, 0)]
        result = compare(otel, records)
        self.assertEqual(result["disagreements"], [])

    def test_multiple_sessions_partial_overlap(self):
        """Correctly partitions matched / otel_only / store_only across many sessions."""
        otel = {
            "shared": {"fresh_input": 10, "cache_read": 0, "cache_write": 0, "output": 0},
            "otel-only": {"fresh_input": 5, "cache_read": 0, "cache_write": 0, "output": 0},
        }
        records = [
            _make_record("shared", 10, 0, 0, 0),
            _make_record("store-only", 7, 0, 0, 0),
        ]
        result = compare(otel, records)
        self.assertEqual(result["matched"], 1)
        self.assertEqual(result["otel_only"], 1)
        self.assertEqual(result["store_only"], 1)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReport(unittest.TestCase):
    def _run(
        self,
        otel: dict,
        records: list[dict],
    ) -> str:
        return report(compare(otel, records))

    def test_authoritativeness_statement_present(self):
        """Report always names the transcript store as authoritative."""
        text = self._run({}, [])
        self.assertIn("authoritative", text.lower())
        self.assertIn("transcript store", text.lower())

    def test_optional_meter_statement_present(self):
        """Report always describes OTel as an optional meter."""
        text = self._run({}, [])
        self.assertIn("optional", text.lower())

    def test_agreement_message_when_clean(self):
        """Agreement message shown when all sessions agree within threshold."""
        otel = {
            "sess-1": {
                "fresh_input": 100,
                "cache_read": 0,
                "cache_write": 0,
                "output": 50,
            }
        }
        records = [_make_record("sess-1", 100, 0, 0, 50)]
        text = self._run(otel, records)
        self.assertIn("agree", text.lower())

    def test_disagreement_flagged_in_report(self):
        """Disagreeing sessions are listed in the report output."""
        otel = {
            "sess-bad": {
                "fresh_input": 200,
                "cache_read": 0,
                "cache_write": 0,
                "output": 0,
            }
        }
        records = [_make_record("sess-bad", 100, 0, 0, 0)]
        text = self._run(otel, records)
        self.assertIn("sess-bad", text)
        self.assertIn("DISAGREEMENT", text)

    def test_otel_env_vars_mentioned(self):
        """Report mentions the key OTel environment variables."""
        text = self._run({}, [])
        self.assertIn("CLAUDE_CODE_ENABLE_TELEMETRY", text)
        self.assertIn("OTEL_METRICS_EXPORTER", text)

    def test_no_match_message(self):
        """No-match case produces a clear message."""
        text = self._run({}, [])
        self.assertIn("nothing to compare", text.lower())

    def test_counts_in_report(self):
        """Matched / otel_only / store_only counts appear in the report."""
        otel = {
            "shared": {"fresh_input": 10, "cache_read": 0, "cache_write": 0, "output": 0},
            "extra": {"fresh_input": 5, "cache_read": 0, "cache_write": 0, "output": 0},
        }
        records = [
            _make_record("shared", 10, 0, 0, 0),
            _make_record("store-side", 3, 0, 0, 0),
        ]
        text = report(compare(otel, records))
        self.assertIn("1", text)  # matched = 1


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
