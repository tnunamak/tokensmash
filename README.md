# Tokensmash

Tokensmash measures whether agent tooling actually reduces end-to-end token
spend on successful coding tasks.

It does not trust tool self-reported savings. The primary metric is the
provider-reported session total:

```text
total_token_usage.total_tokens per successful task
```

## Current Valid Result

No maintainer-mode token deltas are currently published.

The first smoke run was withdrawn because it measured task success and provider
tokens, but did not require proof that each tool's token-saving mechanism
actually participated. Tokensmash now reports a token result only when both are
true:

1. The task oracle passes.
2. The configured mechanism check is observed.

Rows that pass the task but fail mechanism evidence are shown as:

| tool | baseline token spend | token spend with tool | percent improvement | confidence | mechanism |
| --- | --- | --- | --- | --- | --- |
| example_tool | not reported | not reported | not measured | none | not observed |

## Benchmark Scope

The bundled suite is aimed at focused maintenance tasks in existing tested
repositories. A task should ask the agent to modify code and then pass a real
verification command, for example:

```bash
go test ./...
```

Results from this suite should not be generalized to greenfield builds, large
refactors, UI work, documentation tasks, research tasks, or long multi-session
debugging without adding tasks that represent those workflows.

Validity rules:

- Passing the task oracle is required but not sufficient.
- A tool row must also pass its mechanism check.
- Confidence describes whether the reported percent change is actionable for
  this benchmark sample, not whether a tool is universally better.
- Rows from different result batches must be compared only to their paired
  baseline from the same batch, task, and replicate.
- Claude totals include provider-reported input, cache creation, cache read, and
  output tokens from `claude --output-format json`.

## Run Your Own Benchmark

The bundled suite is generic. It does not assume your repos, tools, or paths.

```bash
git clone https://github.com/tnunamak/tokensmash
cd tokensmash

export TARGET_REPO=~/code/my-project
export TARGET_BASE_REF=HEAD
export TARGET_PROMPT='Fix the failing test with the smallest correct change.'
export TARGET_TEST_COMMAND='npm test'
export TARGET_CONTEXT_INCLUDE='src/**/*.ts,package.json'

uv run tokensmash plan --variants context_mode
uv run tokensmash run --live --variants context_mode --run-id my-project-context-mode
uv run tokensmash report ~/.local/state/tokensmash/ab-runs/my-project-context-mode/results.json -o reports/my-project-context-mode.md
```

`run` is dry-run by default. Add `--live` only when intentionally spending model
quota. Variants skip cleanly when required commands or environment variables are
missing.

To run Claude Sonnet rows:

```bash
export TOKENSMASH_AGENT=claude
export TOKENSMASH_CLAUDE_MODEL=sonnet
export TOKENSMASH_MAX_BUDGET_USD=1.50

uv run tokensmash run --live --variants rtk,repomix --run-id my-project-claude-sonnet
```

## Tool Conditions Included

The generic suite includes:

- `baseline_no_user_config`
- `context_mode`
- `headroom`
- `rtk`
- `semmap`
- `repomix`
- `gitingest`

`semmap` requires `SEMMAP_BIN`. `gitingest` requires
`TARGET_CONTEXT_INCLUDE`, and optionally accepts `TARGET_CONTEXT_EXCLUDE`.

Each non-baseline variant includes a mechanism check. Examples: context-mode
and Repomix must appear in agent tool calls; Headroom must report nonzero routed
requests; SEMMAP and Gitingest must generate their artifact and the agent must
use it from a tool call.

## Session Log Audits

Tokensmash can summarize local agent session logs without copying raw
transcripts into the repo:

```bash
uv run tokensmash sessions --agent all --days 7
uv run tokensmash sessions --agent codex --codex-root ~/.codex/sessions -o results/my-codex-sessions.json
uv run tokensmash sessions --agent claude --claude-root ~/.claude/projects -o results/my-claude-sessions.json
uv run tokensmash sessions --agent gemini --gemini-root ~/.gemini -o results/my-gemini-sessions.json
```

The output is sanitized aggregate JSON: file hashes, token counters, tool-call
counts, and byte counts. Do not commit raw session logs. See
[`docs/session-privacy.md`](docs/session-privacy.md).

## Improving The Benchmark

To turn early results into stronger evidence:

- Run many public tasks across languages and repo sizes.
- Use 3-5 replicates per tool and report median/range.
- Randomize run order.
- Record actual tool-use evidence per row.
- Test each tool in its real default integration mode.
- Publish task manifests, base refs, prompts, oracle commands, patch hashes, and
  sanitized token summaries.

## Repository Layout

```text
src/tokensmash/            CLI and benchmark harness
suites/                    runnable benchmark suites and templates
tasks/                     optional reusable task cards
results/                   sanitized result summaries only
reports/                   benchmark-card Markdown reports
docs/                      design notes and privacy rules
```
