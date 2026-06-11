# Tokensmash

Tokensmash measures whether agent tooling actually reduces end-to-end token
spend on successful coding tasks.

It does not trust tool self-reported savings. The primary metric is the
provider-reported session total:

```text
total_token_usage.total_tokens per successful task
```

## Current Smoke Result

Reference run: 2026-06-11, one Go maintenance task in a tested desktop utility repo, one
replicate per row. Each row uses the paired baseline from the same batch.

| tool | agent / model | maintainer-mode condition tested | baseline tokens | tool tokens | token result | oracle |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [context-mode](https://github.com/mksglu/context-mode) | Codex `gpt-5.5` low | MCP plus lifecycle hooks in isolated `CODEX_HOME` | 796319 | 657650 | +17.4% | pass |
| [context-mode](https://github.com/mksglu/context-mode) | Claude Sonnet low | MCP plus Claude-style lifecycle hooks | 559867 | 967624 | -72.8% | pass |
| [Headroom](https://github.com/chopratejas/headroom) | Codex `gpt-5.5` low | `headroom wrap codex --no-serena` | 796319 | 730580 | +8.3% | pass |
| [Headroom](https://github.com/chopratejas/headroom) | Claude Sonnet low | `headroom wrap claude --no-serena` | 559867 | 861004 | -53.8% | pass |
| [RTK](https://github.com/rtk-ai/rtk) | Codex `gpt-5.5` low | Codex `AGENTS.md`/`RTK.md` instructions, ultra-compact wrappers | 796319 | 638112 | +19.9% | pass |
| [RTK](https://github.com/rtk-ai/rtk) | Claude Sonnet low | Claude `PreToolUse` Bash rewrite hook, ultra-compact mode | 559867 | 505027 | +9.8% | pass |
| [SEMMAP](https://github.com/junovhs/semmap) | Codex `gpt-5.5` low | `semmap generate --chat-output`, prompt requires minimal working set | 1021707 | 617922 | +39.5% | pass |
| [SEMMAP](https://github.com/junovhs/semmap) | Claude Sonnet low | `semmap generate --chat-output`, prompt requires minimal working set | 417949 | 1459564 | -249.2% | pass |
| [Repomix](https://github.com/yamadashy/repomix) | Codex `gpt-5.5` low | Repomix MCP, incremental focused packing/retrieval | 796319 | 857928 | -7.7% | pass |
| [Repomix](https://github.com/yamadashy/repomix) | Claude Sonnet low | Repomix MCP, incremental focused packing/retrieval | 559867 | 376623 | +32.7% | pass |
| [Gitingest](https://github.com/coderamp-labs/gitingest) | Codex `gpt-5.5` low | focused digest with explicit include patterns | 1021707 | 933131 | +8.7% | pass |
| [Gitingest](https://github.com/coderamp-labs/gitingest) | Claude Sonnet low | focused digest with explicit include patterns | 417949 | 1777156 | -325.2% | pass |

`oracle = pass` means the final code passed the task's verification command. It
does not mean the tool was token-efficient. `token result` is the benchmark
outcome.

## What This Result Means

The reference task was one focused maintenance bug fix in a tested Go
repository. The agent had to modify code and tests, then pass:

```bash
go test ./...
```

The smoke result is most relevant to small-to-medium coding tasks in existing
tested repositories. It should not be generalized to greenfield builds, large
refactors, UI work, documentation tasks, research tasks, or long multi-session
debugging.

Important caveats:

- One task and one replicate per row.
- Rows came from multiple bounded batches; compare each row only to its paired
  baseline.
- Tool exposure is not always tool use. The Codex Headroom run launched the
  maintainer wrapper, but the Headroom proxy log showed `0` routed requests in
  this environment, so that row is not causal proof of Headroom savings.
- Claude totals include provider-reported input, cache creation, cache read, and
  output tokens from `claude --output-format json`.
- This is a single smoke task, not a leaderboard.

Treat these as benchmark harness checks and directional evidence.

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

To turn smoke results into stronger evidence:

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
