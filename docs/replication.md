# Replication guide

This document covers installing tokensmash, running a study on your own session
history, understanding what leaves your machine, and submitting results.

---

## Install

```bash
uv tool install git+https://github.com/tnunamak/tokensmash
tokensmash --help
```

`uv tool install` puts the binary on your PATH and isolates the virtualenv.
Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

---

## Ingest your session history

```bash
tokensmash ingest
```

Walks `~/.codex/sessions` and `~/.claude/projects`, parses per-request usage
from provider-reported fields, and writes normalized records to
`~/.local/state/tokensmash/study/sessions.jsonl`. No prompt text, file paths,
or tool outputs are stored.

Options:
- `--codex-root <path>` — override Codex session directory
- `--claude-root <path>` — override Claude projects directory
- `--since-days <n>` — restrict to files modified in the last N days

---

## Check opportunity ceilings (Layer 0)

```bash
tokensmash opportunity
```

Prints per-tool upper bounds on potential savings from your ingested history.
These are generous ceilings (100% compression assumed); see
[`docs/study-architecture.md`](study-architecture.md) §D9. If a tool's ceiling
is small relative to your usage variance, the study will not have enough power
to detect an effect — the ceiling is the result.

---

## Initialize a study

Before initializing, fill and hash [`PROTOCOL.template.md`](../PROTOCOL.template.md)
as described in that file's header, then commit the hash.

```bash
tokensmash study init --study-id <your-id> --protocol-version <version>
```

This writes a config to `~/.local/state/tokensmash/study/config.json` containing
a random seed (printed only once; kept in the config, not echoed). The config
sets `mode: log-only` by default — sessions get arm labels but no actuation.

---

## Dry run (log-only mode)

Run in log-only mode for approximately one week before going live. This
validates that:

- Sessions are linked to assignments (`study link` is firing from the
  SessionStart hook).
- Block balance looks correct — about half on, half off.
- Ingest is parsing your sessions without parse errors.

Check linkage:

```bash
tokensmash ingest
tokensmash opportunity
```

Look for `"arm": "on"` and `"arm": "off"` in the ingested records. If sessions
show `"arm": null`, the SessionStart hook is not firing — see below.

### Checking hook installation (`study install`, landing this week)

```bash
tokensmash study install --check
```

Reports whether the SessionStart hook line is present in `~/.claude/settings.json`
and `~/.codex/hooks.json`, and shows the shell function snippet needed for the
launcher wrapper. When the `--check` flag is absent, it applies the hook
entries idempotently (but never edits shell rc files — you add the shell
function yourself from the printed snippet).

---

## Power gate

Before switching to live mode, confirm the target tool's ceiling exceeds the
minimum detectable effect:

```bash
tokensmash study power
```

The report shows mean block cost, variance, blocks per week, and per-tool MDE
comparison. A tool only enters live mode if `ceiling_usd_per_block > MDE_usd_per_block`
at 8 weeks, α = 0.05, power = 0.80 (PROTOCOL.md §6–7). If a tool fails the
gate, the Layer-0 bound from `tokensmash opportunity` is the reportable result
for that tool.

---

## Live flip

Once the dry run validates linkage and the power gate passes, switch to live
mode by re-running init with `--mode live` or editing the config, then redeploy
the launcher wrappers. Sessions started after the mode change are actuated (tool
enabled on on-arm, absent on off-arm). The study seed and assignment rule do not
change.

---

## Analyze (`study analyze`, landing this week)

```bash
tokensmash study analyze
```

Implements the pre-registered analysis plan (PROTOCOL.md §5): paired-block
permutation test, CUPED secondary estimate, cluster-robust SE. Reads
`sessions.jsonl` + `assignments.jsonl`, applies pre-registered exclusions, and
refuses to mix protocol versions in one analysis run.

---

## What gets shared

The scrubbed export is the shareable artifact:

```bash
tokensmash study export -o export.jsonl
```

Fields in the export (schema `tokensmash-session/1`, D3):

| Field | Description |
|---|---|
| `schema` | `"tokensmash-session/1"` |
| `agent` | `"codex"` or `"claude-code"` |
| `session_id` | HMAC-keyed hash of the original session identifier |
| `machine_id` | HMAC-keyed hash of the machine; not reversible without the key |
| `study_id` | Study identifier set at init |
| `protocol_version` | Protocol version string |
| `started_at`, `ended_at` | ISO 8601 timestamps |
| `model` | Model identifier string |
| `agent_version` | Agent version string |
| `repo_id` | HMAC-keyed hash of the canonical repo identity |
| `user_turns` | Count of user turns in the session |
| `tool_calls` | Count of tool calls in the session |
| `compactions` | Count of compaction events |
| `duration_ms` | Wall-clock session duration |
| `usage.fresh_input` | Tokens billed at full input rate |
| `usage.cache_read` | Tokens billed at cache-read rate |
| `usage.cache_write` | Tokens billed at cache-write rate (Anthropic only) |
| `usage.output` | Output tokens |
| `usage.reasoning_output` | Reasoning output tokens where reported; null otherwise |
| `provider_raw` | Numeric usage fields only from the provider response |
| `cost_api_usd` | API-equivalent USD (versioned pricing) |
| `pricing_id` | Pricing data file identifier |
| `cost_codex_credits` | Codex credit units (Codex sessions only) |
| `credit_rate_id` | Credit rate table identifier |
| `arm` | `"on"` or `"off"` |
| `assignment_id` | Assignment record identifier |
| `excluded` | Exclusion reason string, or null |

### What never leaves the machine

The following are never written to the store and are therefore absent from
exports:

- Prompt text or assistant message content
- File paths (replaced by HMAC-keyed hashes)
- Tool output text
- The study seed (kept in `config.json`, never printed or exported)
- The `machine_id` HMAC key (derived from a local secret; hashes cannot be
  reversed without it)

---

## Sending results back

Open a GitHub issue or pull request on the tokensmash repository and attach the
scrubbed export file. Include:

- Your `tokensmash study power` output (block statistics, gate result)
- The SHA-256 hash of your registered `PROTOCOL.md`
- Agent versions and approximate date range of sessions

Multiple export files from different machines or users can be merged by simple
concatenation; the `(agent, session_id)` key pair is idempotent across exports
from different machines.

---

## Optional second meter: OTel cross-validation

Claude Code can emit token-usage counters via OpenTelemetry, providing an
independent check on the transcript store totals.  To enable it, set
`CLAUDE_CODE_ENABLE_TELEMETRY=1` and configure an OTLP exporter — for local
validation use `OTEL_METRICS_EXPORTER=otlp` with a file-based collector writing
one `ExportMetricsServiceRequest` JSON object per line.  Once you have a
`.jsonl` export, run `tokensmash.otelcheck.parse_otlp_jsonl()` to extract
per-session sums and `compare()` to join them against your store records;
`report()` prints a human-readable summary that flags any session where OTel
and the store disagree by more than 1% on any token field.  **The transcript
store is the authoritative source**; OTel is a confirmatory second meter only —
discrepancies warrant investigation but do not override store values.
