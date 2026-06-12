"""Tests for tokensmash.pricing module.

Covers:
  - Table loading and validation (every data file loads, has required fields)
  - Exact cost arithmetic against hand-computed values
  - Model resolution: exact key, date-suffix IDs, bare fuzzy strings, None for unknowns
  - codex_credits math
  - cost_usd None for unknown models / unknown agents
"""

import unittest
from tokensmash.pricing import load_tables, resolve_model, cost_usd, codex_credits


class TestLoadTables(unittest.TestCase):
    """Data file structural validation."""

    def setUp(self):
        self.tables = load_tables()

    def test_tables_non_empty(self):
        self.assertGreater(len(self.tables), 0, "No pricing tables loaded")

    def test_required_file_fields(self):
        required = ("id", "kind", "agent", "retrieved_at", "source_urls", "models", "match")
        for table in self.tables:
            for field in required:
                self.assertIn(field, table, f"Table '{table.get('id')}' missing field '{field}'")

    def test_required_model_fields(self):
        rate_fields = ("fresh_input_per_m", "cache_read_per_m", "cache_write_per_m", "output_per_m")
        for table in self.tables:
            for model_id, rates in table["models"].items():
                for field in rate_fields:
                    self.assertIn(
                        field, rates,
                        f"Table '{table['id']}' model '{model_id}' missing field '{field}'"
                    )
                    self.assertIsInstance(
                        rates[field], (int, float),
                        f"Table '{table['id']}' model '{model_id}' field '{field}' must be numeric"
                    )

    def test_valid_kind_values(self):
        valid_kinds = {"api_usd", "codex_credits"}
        for table in self.tables:
            self.assertIn(table["kind"], valid_kinds, f"Table '{table['id']}' has invalid kind")

    def test_valid_agent_values(self):
        valid_agents = {"codex", "claude-code"}
        for table in self.tables:
            self.assertIn(table["agent"], valid_agents, f"Table '{table['id']}' has invalid agent")

    def test_match_references_existing_models(self):
        for table in self.tables:
            for entry in table["match"]:
                self.assertIn(
                    entry["model"], table["models"],
                    f"Table '{table['id']}' match entry '{entry['pattern']}' "
                    f"references unknown model '{entry['model']}'"
                )

    def test_retrieved_at_present(self):
        for table in self.tables:
            self.assertTrue(table["retrieved_at"], f"Table '{table['id']}' has empty retrieved_at")

    def test_source_urls_non_empty(self):
        for table in self.tables:
            self.assertGreater(
                len(table["source_urls"]), 0,
                f"Table '{table['id']}' has no source_urls"
            )

    def test_anthropic_table_present(self):
        ids = {t["id"] for t in self.tables}
        self.assertTrue(
            any("anthropic" in tid for tid in ids),
            "No Anthropic pricing table found"
        )

    def test_openai_api_table_present(self):
        ids = {t["id"] for t in self.tables}
        self.assertTrue(
            any("openai-api" in tid for tid in ids),
            "No OpenAI API pricing table found"
        )

    def test_codex_credits_table_present(self):
        kinds = {t["kind"] for t in self.tables}
        self.assertIn("codex_credits", kinds, "No codex_credits table found")


class TestResolveModel(unittest.TestCase):
    """Model resolution logic."""

    def setUp(self):
        self.tables = load_tables()

    def test_exact_key_sonnet_4_6(self):
        result = resolve_model(self.tables, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        rates, table_id = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 3.0)
        self.assertAlmostEqual(rates["output_per_m"], 15.0)

    def test_exact_key_opus_4_8(self):
        result = resolve_model(self.tables, "claude-code", "claude-opus-4-8")
        self.assertIsNotNone(result)
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 5.0)
        self.assertAlmostEqual(rates["output_per_m"], 25.0)

    def test_exact_key_haiku_4_5(self):
        result = resolve_model(self.tables, "claude-code", "claude-haiku-4-5")
        self.assertIsNotNone(result)
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 1.0)
        self.assertAlmostEqual(rates["output_per_m"], 5.0)

    def test_date_suffix_sonnet_4_6(self):
        """Model IDs with date suffixes like claude-sonnet-4-6-20251001 should resolve."""
        result = resolve_model(self.tables, "claude-code", "claude-sonnet-4-6-20251001")
        self.assertIsNotNone(result, "Date-suffix model ID should resolve via match patterns")
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 3.0)

    def test_bare_sonnet_resolves(self):
        """Bare 'sonnet' should resolve to a sonnet model."""
        result = resolve_model(self.tables, "claude-code", "sonnet")
        self.assertIsNotNone(result)
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 3.0)

    def test_bare_opus_resolves(self):
        result = resolve_model(self.tables, "claude-code", "opus")
        self.assertIsNotNone(result)
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 5.0)

    def test_bare_haiku_resolves(self):
        result = resolve_model(self.tables, "claude-code", "haiku")
        self.assertIsNotNone(result)
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 1.0)

    def test_unknown_model_returns_none(self):
        result = resolve_model(self.tables, "claude-code", "gpt-9000-ultra")
        self.assertIsNone(result)

    def test_wrong_agent_returns_none(self):
        """Claude models should not resolve for the codex agent (api_usd tables)."""
        # codex api_usd table has gpt-5.5 / gpt-5.4; claude-sonnet should not appear there
        usd_tables = [t for t in self.tables if t["kind"] == "api_usd"]
        result = resolve_model(usd_tables, "codex", "claude-sonnet-4-6")
        self.assertIsNone(result)

    def test_gpt55_exact_key(self):
        # Resolve against api_usd tables only (the codex_credits table also has gpt-5.5)
        usd_tables = [t for t in self.tables if t["kind"] == "api_usd"]
        result = resolve_model(usd_tables, "codex", "gpt-5.5")
        self.assertIsNotNone(result)
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 5.0)
        self.assertAlmostEqual(rates["output_per_m"], 30.0)

    def test_gpt54_exact_key(self):
        usd_tables = [t for t in self.tables if t["kind"] == "api_usd"]
        result = resolve_model(usd_tables, "codex", "gpt-5.4")
        self.assertIsNotNone(result)
        rates, _ = result
        self.assertAlmostEqual(rates["fresh_input_per_m"], 2.5)

    def test_returns_table_id(self):
        result = resolve_model(self.tables, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        _, table_id = result
        self.assertIsInstance(table_id, str)
        self.assertTrue(table_id, "table_id must not be empty")


class TestCostUsd(unittest.TestCase):
    """cost_usd arithmetic against hand-computed values."""

    def _usage(self, fresh=0, cache_read=0, cache_write=0, output=0):
        return {
            "fresh_input": fresh,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "output": output,
            "reasoning_output": None,
        }

    def test_sonnet_4_6_fresh_only(self):
        # 1M fresh input at $3.00/M = $3.00
        usage = self._usage(fresh=1_000_000)
        result = cost_usd(usage, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 3.0, places=6)

    def test_sonnet_4_6_output_only(self):
        # 1M output at $15.00/M = $15.00
        usage = self._usage(output=1_000_000)
        result = cost_usd(usage, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 15.0, places=6)

    def test_sonnet_4_6_cache_read(self):
        # 1M cache_read at $0.30/M = $0.30
        usage = self._usage(cache_read=1_000_000)
        result = cost_usd(usage, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 0.30, places=6)

    def test_sonnet_4_6_cache_write(self):
        # 1M cache_write at $3.75/M = $3.75
        usage = self._usage(cache_write=1_000_000)
        result = cost_usd(usage, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 3.75, places=6)

    def test_sonnet_4_6_combined(self):
        # 500K fresh ($1.50) + 200K cache_read ($0.06) + 100K cache_write ($0.375) + 300K output ($4.50)
        # = 1.50 + 0.06 + 0.375 + 4.50 = 6.435
        usage = self._usage(fresh=500_000, cache_read=200_000, cache_write=100_000, output=300_000)
        result = cost_usd(usage, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        total, table_id = result
        expected = 500_000 / 1e6 * 3.0 + 200_000 / 1e6 * 0.30 + 100_000 / 1e6 * 3.75 + 300_000 / 1e6 * 15.0
        self.assertAlmostEqual(total, expected, places=9)
        self.assertIn("anthropic", table_id)

    def test_haiku_4_5_combined(self):
        # fresh_input_per_m=1.0, cache_read_per_m=0.10, cache_write_per_m=1.25, output_per_m=5.0
        usage = self._usage(fresh=1_000_000, cache_read=500_000, cache_write=250_000, output=200_000)
        result = cost_usd(usage, "claude-code", "claude-haiku-4-5")
        self.assertIsNotNone(result)
        total, _ = result
        expected = 1.0 + 0.05 + 0.3125 + 1.0
        self.assertAlmostEqual(total, expected, places=9)

    def test_opus_4_8_combined(self):
        # fresh_input_per_m=5.0, cache_read_per_m=0.50, cache_write_per_m=6.25, output_per_m=25.0
        usage = self._usage(fresh=1_000_000, cache_read=1_000_000, cache_write=1_000_000, output=1_000_000)
        result = cost_usd(usage, "claude-code", "claude-opus-4-8")
        self.assertIsNotNone(result)
        total, _ = result
        expected = 5.0 + 0.50 + 6.25 + 25.0
        self.assertAlmostEqual(total, expected, places=9)

    def test_gpt55_cost(self):
        # gpt-5.5: fresh_input_per_m=5.0, cache_read_per_m=0.50, cache_write_per_m=0.0, output_per_m=30.0
        usage = self._usage(fresh=1_000_000, cache_read=500_000, output=1_000_000)
        result = cost_usd(usage, "codex", "gpt-5.5")
        self.assertIsNotNone(result)
        total, _ = result
        expected = 5.0 + 0.25 + 30.0
        self.assertAlmostEqual(total, expected, places=9)

    def test_gpt55_cache_write_not_charged(self):
        # OpenAI cache_write_per_m=0.0; providing cache_write tokens should not add cost
        usage_no_write = self._usage(fresh=1_000_000, output=1_000_000)
        usage_with_write = self._usage(fresh=1_000_000, cache_write=1_000_000, output=1_000_000)
        r1 = cost_usd(usage_no_write, "codex", "gpt-5.5")
        r2 = cost_usd(usage_with_write, "codex", "gpt-5.5")
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        self.assertAlmostEqual(r1[0], r2[0], places=9)

    def test_zero_usage_zero_cost(self):
        usage = self._usage()
        result = cost_usd(usage, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 0.0, places=9)

    def test_unknown_model_returns_none(self):
        usage = self._usage(fresh=1_000_000)
        result = cost_usd(usage, "claude-code", "unknown-model-xyz")
        self.assertIsNone(result)

    def test_unknown_agent_returns_none(self):
        usage = self._usage(fresh=1_000_000)
        result = cost_usd(usage, "mystery-agent", "claude-sonnet-4-6")
        self.assertIsNone(result)

    def test_date_suffix_model_resolves(self):
        """Models like claude-sonnet-4-6-20251001 must resolve."""
        usage = self._usage(fresh=1_000_000, output=1_000_000)
        result = cost_usd(usage, "claude-code", "claude-sonnet-4-6-20251001")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 3.0 + 15.0, places=6)

    def test_bare_sonnet_resolves(self):
        usage = self._usage(fresh=1_000_000)
        result = cost_usd(usage, "claude-code", "sonnet")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 3.0, places=6)

    def test_reasoning_output_ignored_in_cost(self):
        """reasoning_output tokens must not be double-counted; they are billed as output."""
        usage_a = {"fresh_input": 1_000_000, "cache_read": 0, "cache_write": 0,
                   "output": 1_000_000, "reasoning_output": None}
        usage_b = {"fresh_input": 1_000_000, "cache_read": 0, "cache_write": 0,
                   "output": 1_000_000, "reasoning_output": 500_000}
        r1 = cost_usd(usage_a, "claude-code", "claude-sonnet-4-6")
        r2 = cost_usd(usage_b, "claude-code", "claude-sonnet-4-6")
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        # reasoning_output does not add cost; cost should be identical
        self.assertAlmostEqual(r1[0], r2[0], places=9)


class TestCodexCredits(unittest.TestCase):
    """codex_credits arithmetic."""

    def _usage(self, fresh=0, cache_read=0, cache_write=0, output=0):
        return {
            "fresh_input": fresh,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "output": output,
            "reasoning_output": None,
        }

    def test_gpt55_fresh_only(self):
        # 1M fresh_input at 125 credits/M = 125.0 credits
        usage = self._usage(fresh=1_000_000)
        result = codex_credits(usage, "gpt-5.5")
        self.assertIsNotNone(result)
        total, table_id = result
        self.assertAlmostEqual(total, 125.0, places=6)
        self.assertIn("codex", table_id)

    def test_gpt55_output_only(self):
        # 1M output at 750 credits/M = 750.0 credits
        usage = self._usage(output=1_000_000)
        result = codex_credits(usage, "gpt-5.5")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 750.0, places=6)

    def test_gpt55_cache_read(self):
        # 1M cache_read at 12.5 credits/M = 12.5 credits
        usage = self._usage(cache_read=1_000_000)
        result = codex_credits(usage, "gpt-5.5")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 12.5, places=6)

    def test_gpt55_combined(self):
        # 2M fresh (250) + 1M cached (12.5) + 0.5M output (375)
        usage = self._usage(fresh=2_000_000, cache_read=1_000_000, output=500_000)
        result = codex_credits(usage, "gpt-5.5")
        self.assertIsNotNone(result)
        total, _ = result
        expected = 2.0 * 125.0 + 1.0 * 12.5 + 0.5 * 750.0
        self.assertAlmostEqual(total, expected, places=9)

    def test_gpt54_rates(self):
        # GPT-5.4: 62.5 fresh / 6.25 cached / 375 output
        usage = self._usage(fresh=1_000_000, cache_read=1_000_000, output=1_000_000)
        result = codex_credits(usage, "gpt-5.4")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 62.5 + 6.25 + 375.0, places=9)

    def test_gpt55_cache_write_not_charged(self):
        # Codex caching is automatic; cache_write_per_m=0 so write tokens are free
        usage_no_write = self._usage(fresh=1_000_000)
        usage_with_write = self._usage(fresh=1_000_000, cache_write=1_000_000)
        r1 = codex_credits(usage_no_write, "gpt-5.5")
        r2 = codex_credits(usage_with_write, "gpt-5.5")
        self.assertIsNotNone(r1)
        self.assertIsNotNone(r2)
        self.assertAlmostEqual(r1[0], r2[0], places=9)

    def test_zero_usage_zero_credits(self):
        usage = self._usage()
        result = codex_credits(usage, "gpt-5.5")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 0.0, places=9)

    def test_unknown_model_returns_none(self):
        usage = self._usage(fresh=1_000_000)
        result = codex_credits(usage, "gpt-99-ultra")
        self.assertIsNone(result)

    def test_claude_model_returns_none(self):
        """Claude models have no codex_credits table."""
        usage = self._usage(fresh=1_000_000)
        result = codex_credits(usage, "claude-sonnet-4-6")
        self.assertIsNone(result)

    def test_gpt54_mini_rates(self):
        # GPT-5.4-mini: 18.75 fresh / 1.875 cached / 113 output
        usage = self._usage(fresh=1_000_000, cache_read=1_000_000, output=1_000_000)
        result = codex_credits(usage, "gpt-5.4-mini")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 18.75 + 1.875 + 113.0, places=9)

    def test_gpt55_pattern_match(self):
        """Pattern match on a string containing gpt-5.5."""
        usage = self._usage(fresh=1_000_000)
        result = codex_credits(usage, "gpt-5.5-turbo-preview")
        self.assertIsNotNone(result)
        total, _ = result
        self.assertAlmostEqual(total, 125.0, places=6)


if __name__ == "__main__":
    unittest.main()
