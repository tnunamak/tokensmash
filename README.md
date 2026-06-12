# Tokensmash

Tokensmash is a measurement instrument for whether token-saving agent tools are
worth enabling. It measures from provider-billed usage in your own session logs
— not from tool self-reporting, not from vendor claims. It is not itself a
token-saving tool.

Two layers:

- **Layer 0 (opportunity):** reads your real session history and computes an
  upper bound on what each tool could save if it were 100% effective. These are
  deliberate ceilings, not predictions.
- **Layer 1 (crossover study):** a pre-registered within-repo A/B design that
  measures whether enabling a tool as the default policy actually moves
  billed-equivalent cost per 2-hour work block.

---

## 10-minute quickstart

```bash
# Install
uv tool install git+https://github.com/tnunamak/tokensmash

# Parse your local agent transcripts into the normalized store
tokensmash ingest

# Print Layer-0 opportunity ceilings
tokensmash opportunity
```

`ingest` reads `~/.codex/sessions` and `~/.claude/projects` by default. It
writes to `~/.local/state/tokensmash/study/sessions.jsonl`. No prompt text,
file content, or tool output is stored — only numeric usage counters and keyed
hashes of identifiers.

`opportunity` prints per-tool ceilings in API-equivalent USD (per-session
records additionally carry exact Codex credit costs). Read the numbers as
"maximum plausible saving if this tool compressed its target tokens to zero." Most
output-compressor ceilings are small in the reference dataset; interpret them as
such before deciding whether to run a study.

---

## What the numbers mean

**Ceilings are upper bounds, not estimates.** The `with_rereads_usd` column
assumes 100% compression of the tool's addressable tokens and includes
cache-read savings on subsequent requests. The actual saving will be lower —
often much lower.

**USD figures are API-equivalent.** They are computed from provider-published
per-token-type rates with versioned pricing files (`pricing_id` per session).

**Codex credits are exact.** OpenAI publishes per-token-type credit rates; the
`cost_codex_credits` field is computed from those rates (`credit_rate_id` per
session).

**Claude subscription quota is opaque.** Anthropic does not publish the
weighting applied to subscription quota. Only API-equivalent USD is claimed for
Claude Code sessions; the tool does not attempt to model quota pressure.

**Cache reads dominate real agent spend.** In the reference dataset, a large
fraction of billed tokens are cache reads. This is why most tool ceilings are
small: the tools target fresh-input tokens, but most tokens are already cached
by the time the agent reaches steady state in a session.

---

## Crossover study

If a tool's ceiling exceeds the minimum detectable effect at your usage level
(`tokensmash study power`), you can run a pre-registered crossover study that
measures the causal effect of the policy "tool default-enabled."

The design: repo × UTC 2-hour block is the randomization unit. Arms alternate
deterministically (on / off) across consecutive groups of 8 blocks, guaranteeing
4/4 balance per group. Actuation is launcher-based and never injects text into
the agent's context. The estimand is intention-to-treat: arm assignment, not
whether the tool was invoked, determines group membership.

See [`PROTOCOL.md`](PROTOCOL.md) for the full pre-registered protocol and
[`docs/replication.md`](docs/replication.md) for the replication kit.

**Current gate result (2026-06-12):** at 47.4 weeks of pre-study history,
RTK, context-mode, and Repomix ceilings fall below the 8-week MDE — documented
as unmeasurable at current usage levels. Headroom (wire-payload ceiling 77.3%
of mean block cost) exceeds the raw MDE and is the sole tool in the current
epoch.

### Commands

| Command | Status |
|---|---|
| `tokensmash ingest` | available |
| `tokensmash opportunity` | available |
| `tokensmash study init` | available |
| `tokensmash study link` | available (SessionStart hook entrypoint) |
| `tokensmash study arm` | available (launcher entrypoint) |
| `tokensmash study power` | available |
| `tokensmash study export` | available |
| `tokensmash study analyze` | available (pre-registered inference) |
| `tokensmash study install` | available (config check/apply) |
| `tokensmash launch` | available (actuation launcher) |
| `tokensmash replay` | available (realized-compression estimates) |
| `tokensmash trajectory` | available (wasted-exploration analysis) |
| `tokensmash meta` | available (multi-machine aggregation) |

---

## Honest findings (reference dataset)

All figures are from one subject's session history. They should not be
generalized without replication.

- Cache reads account for the majority of billed tokens in steady-state agent
  sessions. Tools that compress fresh-input tokens show small ceilings because
  most tokens are no longer fresh by the time they re-enter context.
- RTK, context-mode, and Repomix ceilings are each below 21% of mean block
  cost in the reference dataset.
- Headroom's ceiling is higher (77% of mean block cost) because it targets the
  full wire payload including cache-read traffic, but its mechanism can also
  break prefix caching; the net effect is the thing the study measures.
- "Ceiling is small" is a result, not a failure of the measurement. It means
  the tool cannot be worth much at current usage levels regardless of how well
  it works.

---

## Legacy synthetic benchmark

The original tokensmash benchmark ran synthetic tasks against cloned repos with
controlled variants. That infrastructure (`tokensmash run`, `tokensmash plan`,
`tokensmash table`, `tokensmash aggregate`, `tokensmash report`) is still
present and usable, but **results from it should be treated as directional
only**.

Known limitations:

- **Provider-cache position confounds.** ~~Randomized order records a seed but
  does not balance positions, so a variant can land first on every task by
  chance.~~ **Repaired** (`--balance-positions`): generates a Latin-square
  rotation so each variant occupies each within-task position equally; design
  matrix is recorded in `results.json`; strict mode now requires either
  `--balance-positions` or `--randomize-order`.
- **Single baseline run per task.** ~~Baseline stochastic variance is
  unmeasured and shared across all tool comparisons, correlating the tool
  rows.~~ **Repaired** (`--baseline-replicates N`): runs the baseline N times
  per task; aggregate pairs each tool run against the mean of baseline
  replicates and reports a "baseline sd" column.
- **Strict gate is not code-version-stamped.** **Repaired**: `results.json`
  now carries a `code_version` stamp (`bench_audit_version`, `git`, `package`);
  `aggregate --strict` refuses files lacking `bench_audit_version==2`.
- **Toy tasks.** Small synthetic repair fixtures mostly measure fixed tool
  overhead, not the savings the tools exist to deliver. Not yet repaired.

Until all repairs are in production runs, use the synthetic benchmark
only for tool × task-class exploration, not for publishable improvement claims.

### Run the synthetic benchmark

```bash
git clone https://github.com/tnunamak/tokensmash
cd tokensmash

export TARGET_REPO=~/code/my-project
export TARGET_TEST_COMMAND='npm test'
export TARGET_PROMPT='Fix the failing test with the smallest correct change.'

uv run tokensmash plan
uv run tokensmash run --live --variants rtk --run-id my-project-rtk
uv run tokensmash table ~/.local/state/tokensmash/ab-runs/my-project-rtk
```

`run` is dry-run by default. Add `--live` only when intentionally spending
model quota.

---

## Repository layout

```text
src/tokensmash/       CLI and study harness
docs/                 design notes (study-architecture.md, CONTRACTS.md, ...)
suites/               synthetic benchmark suite definitions
tasks/                reusable synthetic task cards
results/              sanitized result summaries only
reports/              benchmark-card Markdown reports
PROTOCOL.md           pre-registered crossover study protocol
```
