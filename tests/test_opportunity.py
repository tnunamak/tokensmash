"""Tests for tokensmash.opportunity module.

Covers:
  - summarize: empty timeline, single request, multiple requests + tool_outputs,
    compaction segment splitting, request_index beyond last request,
    compactions > number of requests, unknown categories defaulting to "other".
  - tool_ceilings: known model, unknown model returns {}, zero-cost sessions,
    headroom wire-payload bound, all tool mappings.
  - report: aggregation table, skipped sessions, zero actual cost, caveats footer.
"""

import unittest

from tokensmash.opportunity import summarize, tool_ceilings, report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(index: int) -> dict:
    return {"kind": "request", "index": index, "usage": {
        "fresh_input": 100, "cache_read": 0, "cache_write": 0,
        "output": 10, "reasoning_output": None,
    }}


def _make_tool_output(request_index: int, category: str, tokens: int,
                      tool_name: str = "Bash") -> dict:
    return {
        "kind": "tool_output",
        "request_index": request_index,
        "category": category,
        "tokens_est": tokens,
        "tool_name": tool_name,
    }


def _make_record(model: str = "claude-sonnet-4-6",
                 agent: str = "claude-code",
                 opportunity: dict | None = None,
                 cost_api_usd: float = 1.0,
                 usage: dict | None = None) -> dict:
    return {
        "schema": "tokensmash-session/1",
        "agent": agent,
        "model": model,
        "session_id": "s1",
        "machine_id": "m1",
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T01:00:00Z",
        "repo_id": "r1",
        "user_turns": 1,
        "tool_calls": 1,
        "compactions": 0,
        "duration_ms": 3600000,
        "usage": usage or {
            "fresh_input": 1000,
            "cache_read": 500,
            "cache_write": 0,
            "output": 100,
            "reasoning_output": None,
        },
        "provider_raw": {},
        "cost_api_usd": cost_api_usd,
        "opportunity": opportunity or {},
    }


# ---------------------------------------------------------------------------
# Tests for summarize()
# ---------------------------------------------------------------------------

class TestSummarizeEmpty(unittest.TestCase):
    """Empty and trivial timelines."""

    def test_empty_timeline(self):
        result = summarize([], 0)
        self.assertEqual(result, {})

    def test_only_requests_no_tool_outputs(self):
        timeline = [_make_request(0), _make_request(1)]
        result = summarize(timeline, 0)
        self.assertEqual(result, {})

    def test_only_tool_outputs_no_requests(self):
        timeline = [_make_tool_output(0, "shell", 100)]
        result = summarize(timeline, 0)
        # No requests in the segment, so rereads = 0.
        self.assertIn("shell", result)
        self.assertEqual(result["shell"]["insertion_tokens"], 100)
        self.assertEqual(result["shell"]["reread_tokens"], 0)


class TestSummarizeSingleSegment(unittest.TestCase):
    """Compactions=0: whole session is one segment."""

    def test_single_tool_output_no_later_requests(self):
        # tool_output at request 0, then no later requests
        timeline = [_make_request(0), _make_tool_output(0, "shell", 200)]
        result = summarize(timeline, 0)
        self.assertEqual(result["shell"]["insertion_tokens"], 200)
        self.assertEqual(result["shell"]["reread_tokens"], 0)

    def test_tool_output_with_two_later_requests(self):
        # tool_output at request 0; requests at 0, 1, 2 => 2 later requests
        timeline = [
            _make_request(0),
            _make_tool_output(0, "shell", 100),
            _make_request(1),
            _make_request(2),
        ]
        result = summarize(timeline, 0)
        self.assertEqual(result["shell"]["insertion_tokens"], 100)
        self.assertEqual(result["shell"]["reread_tokens"], 200)  # 100 * 2

    def test_multiple_categories(self):
        timeline = [
            _make_request(0),
            _make_tool_output(0, "shell", 100),
            _make_tool_output(0, "file_read", 50),
            _make_request(1),
        ]
        result = summarize(timeline, 0)
        self.assertEqual(result["shell"]["insertion_tokens"], 100)
        self.assertEqual(result["shell"]["reread_tokens"], 100)  # 100 * 1
        self.assertEqual(result["file_read"]["insertion_tokens"], 50)
        self.assertEqual(result["file_read"]["reread_tokens"], 50)  # 50 * 1

    def test_multiple_tool_outputs_same_category(self):
        timeline = [
            _make_request(0),
            _make_tool_output(0, "shell", 100),
            _make_request(1),
            _make_tool_output(1, "shell", 200),
            _make_request(2),
        ]
        result = summarize(timeline, 0)
        # Requests: indices [0, 1, 2].
        # tool_output at req 0: later requests with index > 0 in seg = [1, 2] -> 2 rereads
        # tool_output at req 1: later requests with index > 1 in seg = [2]    -> 1 reread
        self.assertEqual(result["shell"]["insertion_tokens"], 300)
        self.assertEqual(result["shell"]["reread_tokens"], 100 * 2 + 200 * 1)

    def test_unknown_category_maps_to_other(self):
        timeline = [
            _make_request(0),
            _make_tool_output(0, "not_a_real_category", 80),
            _make_request(1),
        ]
        result = summarize(timeline, 0)
        self.assertIn("other", result)
        self.assertEqual(result["other"]["insertion_tokens"], 80)
        self.assertEqual(result["other"]["reread_tokens"], 80)  # 80 * 1


class TestSummarizeCompactions(unittest.TestCase):
    """Compaction segment splitting."""

    def test_one_compaction_two_segments(self):
        # 4 requests => 2 segments of 2 each (boundary at ordinal 2).
        # tool_output at request_index 0 (segment 0): later requests in seg 0 = req[1] -> 1
        # tool_output at request_index 2 (segment 1): later requests in seg 1 = req[3] -> 1
        timeline = [
            _make_request(0),
            _make_request(1),
            _make_request(2),
            _make_request(3),
            _make_tool_output(0, "shell", 100),
            _make_tool_output(2, "shell", 200),
        ]
        result = summarize(timeline, 1)
        self.assertEqual(result["shell"]["insertion_tokens"], 300)
        # req 0 in seg0: later in seg0 = [1] -> 1 reread
        # req 2 in seg1: later in seg1 = [3] -> 1 reread
        self.assertEqual(result["shell"]["reread_tokens"], 100 * 1 + 200 * 1)

    def test_compaction_resets_reread_count(self):
        # Without compaction, tool_output at req 0 with 3 later requests = 3 rereads.
        # With compaction splitting into 2 segments of 2, only 1 reread per tool_output.
        timeline = [
            _make_request(0),
            _make_request(1),
            _make_request(2),
            _make_request(3),
            _make_tool_output(0, "shell", 100),
        ]
        result_no_compact = summarize(timeline, 0)
        result_one_compact = summarize(timeline, 1)
        self.assertEqual(result_no_compact["shell"]["reread_tokens"], 300)  # 100 * 3
        self.assertEqual(result_one_compact["shell"]["reread_tokens"], 100)  # 100 * 1

    def test_compactions_more_than_requests(self):
        # If C > n-1, each request is its own segment; no rereads possible.
        timeline = [
            _make_request(0),
            _make_tool_output(0, "shell", 100),
            _make_request(1),
        ]
        result = summarize(timeline, 10)
        # With many compactions each request gets its own segment: 0 rereads.
        self.assertEqual(result["shell"]["reread_tokens"], 0)

    def test_codex_zero_compactions_whole_session(self):
        """Codex sessions with compactions=0 use the whole session as one segment."""
        timeline = [
            _make_request(0),
            _make_tool_output(0, "shell", 50),
            _make_request(1),
            _make_request(2),
        ]
        result = summarize(timeline, 0)
        self.assertEqual(result["shell"]["reread_tokens"], 50 * 2)


class TestSummarizeEdgeCases(unittest.TestCase):
    """Edge cases: out-of-range request_index, non-request events ignored."""

    def test_tool_output_request_index_beyond_last_request(self):
        # request_index 99 is not in any known request; should fall back to last segment.
        timeline = [
            _make_request(0),
            _make_request(1),
            _make_tool_output(99, "shell", 100),
        ]
        result = summarize(timeline, 0)
        # Placed in last segment; no requests at index > 99 in that segment.
        self.assertIn("shell", result)
        self.assertEqual(result["shell"]["insertion_tokens"], 100)
        self.assertEqual(result["shell"]["reread_tokens"], 0)

    def test_unknown_event_kinds_ignored(self):
        timeline = [
            {"kind": "compaction", "detail": "ignored"},
            _make_request(0),
            _make_tool_output(0, "shell", 100),
            {"kind": "unknown_future_event", "data": 42},
        ]
        result = summarize(timeline, 0)
        self.assertIn("shell", result)
        self.assertEqual(result["shell"]["insertion_tokens"], 100)

    def test_tokens_est_none_treated_as_zero(self):
        timeline = [
            _make_request(0),
            {"kind": "tool_output", "request_index": 0, "category": "shell",
             "tokens_est": None, "tool_name": "Bash"},
            _make_request(1),
        ]
        result = summarize(timeline, 0)
        # tokens_est None -> 0; should not appear or have 0 tokens.
        if "shell" in result:
            self.assertEqual(result["shell"]["insertion_tokens"], 0)


# ---------------------------------------------------------------------------
# Tests for tool_ceilings()
# ---------------------------------------------------------------------------

class TestToolCeilingsUnknownModel(unittest.TestCase):
    """Unknown model returns empty dict."""

    def test_unknown_model_returns_empty(self):
        record = _make_record(model="gpt-unknown-xyz-9999")
        result = tool_ceilings(record)
        self.assertEqual(result, {})

    def test_unknown_agent_returns_empty(self):
        # claude-code agent with a codex-only model should also resolve to None.
        record = _make_record(model="gpt-4o", agent="claude-code")
        result = tool_ceilings(record)
        self.assertEqual(result, {})


class TestToolCeilingsKnownModel(unittest.TestCase):
    """Known model produces correct ceilings."""

    def setUp(self):
        # claude-sonnet-4-6: fresh_input_per_m=3.0, cache_read_per_m=0.30
        self.model = "claude-sonnet-4-6"
        self.agent = "claude-code"
        self.fresh_rate = 3.0 / 1_000_000
        self.cache_rate = 0.30 / 1_000_000

    def test_rtk_shell_only(self):
        opp = {"shell": {"insertion_tokens": 1000, "reread_tokens": 2000}}
        record = _make_record(model=self.model, agent=self.agent, opportunity=opp)
        result = tool_ceilings(record)
        self.assertIn("rtk", result)
        expected_ins = 1000 * self.fresh_rate
        # Raw reread estimate (0.0036) exceeds the wire-payload cost of the
        # default usage (1000 fresh + 500 cache_read = 0.00315), so the
        # ceiling is capped: savings on input cannot exceed the input bill.
        wire_cap = 1000 * self.fresh_rate + 500 * self.cache_rate
        self.assertAlmostEqual(result["rtk"]["insertion_only_usd"], expected_ins, places=9)
        self.assertAlmostEqual(result["rtk"]["with_rereads_usd"], wire_cap, places=9)
        self.assertEqual(result["rtk"]["capped"], "wire-payload")

    def test_context_mode_shell_search_mcp(self):
        opp = {
            "shell": {"insertion_tokens": 500, "reread_tokens": 100},
            "search": {"insertion_tokens": 200, "reread_tokens": 50},
            "mcp": {"insertion_tokens": 300, "reread_tokens": 80},
        }
        record = _make_record(model=self.model, agent=self.agent, opportunity=opp)
        result = tool_ceilings(record)
        self.assertIn("context-mode", result)
        total_ins = (500 + 200 + 300) * self.fresh_rate
        total_reread = total_ins + (100 + 50 + 80) * self.cache_rate
        self.assertAlmostEqual(result["context-mode"]["insertion_only_usd"], total_ins, places=9)
        self.assertAlmostEqual(result["context-mode"]["with_rereads_usd"], total_reread, places=9)

    def test_repomix_file_read(self):
        opp = {"file_read": {"insertion_tokens": 4000, "reread_tokens": 8000}}
        record = _make_record(model=self.model, agent=self.agent, opportunity=opp)
        result = tool_ceilings(record)
        self.assertIn("repomix", result)
        expected_ins = 4000 * self.fresh_rate
        # Raw reread estimate (0.0144) far exceeds the default usage's
        # wire-payload cost (0.00315) → capped.
        wire_cap = 1000 * self.fresh_rate + 500 * self.cache_rate
        self.assertAlmostEqual(result["repomix"]["insertion_only_usd"], expected_ins, places=9)
        self.assertAlmostEqual(result["repomix"]["with_rereads_usd"], wire_cap, places=9)
        self.assertEqual(result["repomix"]["capped"], "wire-payload")

    def test_headroom_wire_payload_bound(self):
        usage = {
            "fresh_input": 10000,
            "cache_read": 5000,
            "cache_write": 0,
            "output": 500,
            "reasoning_output": None,
        }
        record = _make_record(model=self.model, agent=self.agent, usage=usage)
        result = tool_ceilings(record)
        self.assertIn("headroom", result)
        wire_cost = (10000 * self.fresh_rate + 5000 * self.cache_rate)
        self.assertAlmostEqual(result["headroom"]["insertion_only_usd"], wire_cost, places=9)
        self.assertAlmostEqual(result["headroom"]["with_rereads_usd"], wire_cost, places=9)
        self.assertIn("note", result["headroom"])
        self.assertIn("wire-payload", result["headroom"]["note"])

    def test_headroom_equals_both_bounds(self):
        """Both headroom bounds must be identical (wire-payload semantics)."""
        record = _make_record(model=self.model, agent=self.agent)
        result = tool_ceilings(record)
        self.assertEqual(
            result["headroom"]["insertion_only_usd"],
            result["headroom"]["with_rereads_usd"],
        )

    def test_zero_opportunity_zero_ceiling(self):
        record = _make_record(model=self.model, agent=self.agent, opportunity={})
        result = tool_ceilings(record)
        self.assertIn("rtk", result)
        self.assertEqual(result["rtk"]["insertion_only_usd"], 0.0)
        self.assertEqual(result["rtk"]["with_rereads_usd"], 0.0)

    def test_zero_usage_headroom_zero(self):
        usage = {"fresh_input": 0, "cache_read": 0, "cache_write": 0,
                 "output": 0, "reasoning_output": None}
        record = _make_record(model=self.model, agent=self.agent, usage=usage)
        result = tool_ceilings(record)
        self.assertEqual(result["headroom"]["insertion_only_usd"], 0.0)
        self.assertEqual(result["headroom"]["with_rereads_usd"], 0.0)

    def test_all_expected_tools_present(self):
        record = _make_record(model=self.model, agent=self.agent)
        result = tool_ceilings(record)
        for tool in ("rtk", "context-mode", "repomix", "headroom"):
            self.assertIn(tool, result, f"Expected tool '{tool}' in ceilings")

    def test_codex_model_returns_ceilings(self):
        """Codex sessions with a known model also produce ceilings."""
        record = _make_record(model="gpt-5.4", agent="codex")
        result = tool_ceilings(record)
        # gpt-5.4 resolves via openai-api table for codex agent.
        self.assertIn("rtk", result)
        self.assertIn("headroom", result)


# ---------------------------------------------------------------------------
# Tests for report()
# ---------------------------------------------------------------------------

class TestReport(unittest.TestCase):
    """Human-readable report formatting and aggregation."""

    def setUp(self):
        self.model = "claude-sonnet-4-6"
        self.agent = "claude-code"

    def _make_session(self, ins_shell: int = 1000, reread_shell: int = 2000,
                      cost: float = 1.0,
                      usage: dict | None = None) -> dict:
        opp = {"shell": {"insertion_tokens": ins_shell, "reread_tokens": reread_shell}}
        return _make_record(model=self.model, agent=self.agent,
                            opportunity=opp, cost_api_usd=cost, usage=usage)

    def test_report_contains_tool_names(self):
        records = [self._make_session()]
        out = report(records)
        for tool in ("rtk", "context-mode", "repomix", "headroom"):
            self.assertIn(tool, out)

    def test_report_contains_caveats(self):
        records = [self._make_session()]
        out = report(records)
        self.assertIn("upper bounds", out)
        self.assertIn("prefix stability", out)
        self.assertIn("opaque", out)

    def test_report_aggregates_two_sessions(self):
        r1 = self._make_session(ins_shell=1000, reread_shell=0, cost=1.0)
        r2 = self._make_session(ins_shell=1000, reread_shell=0, cost=1.0)
        out = report([r1, r2])
        self.assertIn("2", out)  # sessions count

    def test_report_skips_unknown_model(self):
        good = self._make_session(cost=1.0)
        bad = _make_record(model="totally-unknown-model-xyz", cost_api_usd=0.5)
        out = report([good, bad])
        # The bad record should register as skipped.
        self.assertIn("1", out)  # at least one skipped

    def test_report_zero_actual_cost(self):
        # Should not raise ZeroDivisionError.
        record = self._make_session(cost=0.0)
        out = report([record])
        self.assertIn("0.0%", out)

    def test_report_empty_records(self):
        out = report([])
        self.assertIsInstance(out, str)
        self.assertIn("Caveats", out)

    def test_report_is_string(self):
        out = report([self._make_session()])
        self.assertIsInstance(out, str)

    def test_report_has_header_row(self):
        out = report([self._make_session()])
        self.assertIn("Tool", out)
        self.assertIn("Actual cost", out)

    def test_report_ceiling_percentage_nonzero(self):
        """When ceilings exceed zero and actual cost is nonzero, % should appear."""
        record = self._make_session(ins_shell=1_000_000, reread_shell=0, cost=1.0)
        out = report([record])
        # Should have a non-zero % for rtk row.
        self.assertIn("%", out)


# ---------------------------------------------------------------------------
# Integration: summarize -> tool_ceilings pipeline
# ---------------------------------------------------------------------------

class TestSummarizeToToolCeilingsPipeline(unittest.TestCase):
    """End-to-end: parse timeline, compute opportunity, compute ceilings."""

    def test_full_pipeline_known_model(self):
        timeline = [
            _make_request(0),
            _make_tool_output(0, "shell", 1000),
            _make_request(1),
            _make_request(2),
        ]
        opp = summarize(timeline, 0)
        # shell: insertion=1000, rereads=1000*2=2000
        self.assertEqual(opp["shell"]["insertion_tokens"], 1000)
        self.assertEqual(opp["shell"]["reread_tokens"], 2000)

        record = _make_record(
            model="claude-sonnet-4-6",
            agent="claude-code",
            opportunity=opp,
        )
        ceilings = tool_ceilings(record)
        fresh_rate = 3.0 / 1_000_000
        cache_rate = 0.30 / 1_000_000
        expected_ins = 1000 * fresh_rate
        # Raw reread (0.0036) exceeds the default usage's wire-payload cost
        # (0.00315) → capped at the input bill.
        wire_cap = 1000 * fresh_rate + 500 * cache_rate
        self.assertAlmostEqual(ceilings["rtk"]["insertion_only_usd"], expected_ins, places=9)
        self.assertAlmostEqual(ceilings["rtk"]["with_rereads_usd"], wire_cap, places=9)
        self.assertEqual(ceilings["rtk"]["capped"], "wire-payload")

    def test_full_pipeline_unknown_model(self):
        timeline = [_make_request(0), _make_tool_output(0, "shell", 500)]
        opp = summarize(timeline, 0)
        record = _make_record(model="mystery-model-v99", opportunity=opp)
        ceilings = tool_ceilings(record)
        self.assertEqual(ceilings, {})


if __name__ == "__main__":
    unittest.main()
