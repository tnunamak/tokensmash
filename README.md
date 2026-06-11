# Tokensmash

Tokensmash measures whether agent tooling actually reduces end-to-end token
spend on successful coding tasks.

It does not trust tool self-reported savings. The primary metric is the
provider-reported session total:

```text
total_token_usage.total_tokens per successful task
```

## Current Smoke Result

Reference run: 2026-06-11, Codex CLI, `gpt-5.5`, low reasoning.

| tool | tool condition tested | baseline tokens | tool tokens | token result | oracle |
| --- | --- | ---: | ---: | ---: | --- |
| [context-mode](https://github.com/mksglu/context-mode) | Codex MCP server plus lifecycle hooks in isolated `CODEX_HOME` | 1265375 | 643229 | +49.2% | pass |
| [Headroom](https://github.com/chopratejas/headroom) | Codex MCP server exposed; prompt allowed optional context compression | 1265375 | 591837 | +53.2% | pass |
| [RTK](https://github.com/rtk-ai/rtk) | Prompt instructed Codex to prefer `rtk` shell wrappers; not tested as automatic hook | 1265375 | 696266 | +45.0% | pass |
| [SEMMAP](https://github.com/junovhs/semmap) | Generated `SEMMAP.md`; prompt instructed Codex to inspect it before broad search | 999151 | 879352 | +12.0% | pass |
| [Repomix](https://github.com/yamadashy/repomix) | Generated full-repo `repomix.md`; prompt exposed the pack path | 620319 | 1309307 | -111.1% | pass |
| [Gitingest](https://github.com/coderamp-labs/gitingest) | Generated full-repo `gitingest.txt`; prompt exposed the digest path | 620319 | 949669 | -53.1% | pass |

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

Key caveats:

- One task and one replicate per row.
- Codex CLI only.
- Rows came from multiple bounded batches; each row uses its own paired
  baseline from the same batch.
- Tool exposure is not always tool use. In particular, the Headroom condition
  exposed the MCP server, but the checked session log did not prove meaningful
  Headroom MCP calls.
- RTK was tested as prompted shell-wrapper usage, not as a transparent default
  hook.
- Full-repo packers were tested by putting a generated pack/digest path in the
  prompt. That is appropriate for this comparison, but not every possible
  Repomix or Gitingest workflow.

Treat this as a first smoke result, not a leaderboard.

## Run Your Own Benchmark

The bundled suite is generic. It does not assume your repos, tools, or paths.

```bash
git clone https://github.com/tnunamak/tokensmash
cd tokensmash

export TARGET_REPO=~/code/my-project
export TARGET_BASE_REF=HEAD
export TARGET_PROMPT='Fix the failing test with the smallest correct change.'
export TARGET_TEST_COMMAND='npm test'

uv run tokensmash plan --variants context_mode
uv run tokensmash run --live --variants context_mode --run-id my-project-context-mode
uv run tokensmash report ~/.local/state/tokensmash/ab-runs/my-project-context-mode/results.json -o reports/my-project-context-mode.md
```

`run` is dry-run by default. Add `--live` only when intentionally spending model
quota. Variants skip cleanly when required commands or environment variables are
missing.

## Tool Conditions Included

The generic suite includes:

- `baseline_no_user_config`
- `context_mode`
- `headroom`
- `rtk`
- `semmap`
- `repomix`
- `gitingest`

Use the `tool condition tested` column above to decide whether the built-in
condition matches the way you would actually use that tool.

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
