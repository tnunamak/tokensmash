# Tokensmash

Tokensmash is a benchmark harness for answering one practical question:

> Does enabling this agent tool reduce total token spend for normal successful agent tasks?

The primary metric is provider-reported session tokens, not a tool's self-reported savings.

## Current Results

Reference smoke run, 2026-06-11:

| tool | how it was applied | baseline tokens | tool tokens | token result | model | reasoning | oracle result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| [context-mode](https://github.com/mksglu/context-mode) | enabled as a Codex MCP server with Codex lifecycle hooks in an isolated `CODEX_HOME` | 1265375 | 643229 | +49.2% | gpt-5.5 | low | pass |
| [Headroom](https://github.com/chopratejas/headroom) | exposed as a Codex MCP server and prompted as optional context compression | 1265375 | 591837 | +53.2% | gpt-5.5 | low | pass |
| [RTK](https://github.com/rtk-ai/rtk) | prompted Codex to prefer `rtk` shell wrappers; not tested as an automatic hook | 1265375 | 696266 | +45.0% | gpt-5.5 | low | pass |
| [SEMMAP](https://github.com/junovhs/semmap) | generated `SEMMAP.md` before the run and prompted Codex to inspect it before broad search | 999151 | 879352 | +12.0% | gpt-5.5 | low | pass |
| [Repomix](https://github.com/yamadashy/repomix) | generated a full-repo `repomix.md` pack before the run and exposed its path in the prompt | 620319 | 1309307 | -111.1% | gpt-5.5 | low | pass |
| [Gitingest](https://github.com/coderamp-labs/gitingest) | generated a full-repo `gitingest.txt` digest before the run and exposed its path in the prompt | 620319 | 949669 | -53.1% | gpt-5.5 | low | pass |

Interpretation: `oracle result` only says whether the coding task succeeded.
`token result` is the benchmark outcome. This is a smoke result, not a
leaderboard. It is one task, one replicate per row, Codex CLI only, and rows
came from multiple bounded batches with paired baselines per batch.

## What The Task Was

The reference task was a single maintenance bug fix in a Go application
repository. The agent had to change code and tests for a CLI/provider path
detection issue, then pass:

```bash
go test ./...
```

That makes this result most relevant to small-to-medium coding tasks in a
tested repository. It is not evidence for greenfield implementation, large
refactors, UI work, documentation tasks, research tasks, or long multi-session
debugging.

## What It Measures

Tokensmash runs paired baseline/tool variants in disposable Git checkouts, verifies the task with an execution oracle, and reads final Codex session token totals from local session logs.

Default metric:

```text
total_token_usage.total_tokens per successful task
```

## Methodology Risks

The current smoke result is useful, but easy to over-interpret.

Main critiques:

- **Single task:** one Go maintenance task cannot represent all agent work.
- **Single replicate:** agent variance can move token totals without any tool
  being responsible.
- **Different batches:** rows use paired baselines, but not one shared baseline.
- **Tool exposure vs tool use:** the Headroom run exposed the MCP server, but
  the checked session log did not prove Headroom MCP calls were used.
- **Integration mismatch:** RTK was tested by instruction to use `rtk` shell
  wrappers, not as a transparent default hook.
- **Packers are task-sensitive:** full-repo packs can be useful for audit or
  handoff tasks, but they were inefficient for this focused bug fix.
- **Oracle is narrow:** `go test ./...` checks task success, not patch quality,
  maintainability, or user experience.

How to make the benchmark stronger:

- Run 20-50 public tasks across languages and repo sizes.
- Run 3-5 replicates per tool and report median, range, and confidence
  intervals.
- Randomize run order to reduce time/rate-limit/cache effects.
- Record actual tool-use evidence per row, not just tool availability.
- Test each tool in the way users would actually enable it by default.
- Publish task manifests, base refs, prompts, oracle commands, patch hashes, and
  sanitized token summaries.

## Repo Structure

```text
src/tokensmash/            CLI and benchmark harness
suites/                    runnable benchmark suites and templates
tasks/                     optional reusable task cards
results/                   sanitized result summaries only
reports/                   benchmark-card Markdown reports
docs/                      design notes and privacy rules
```

## Quick Start

```bash
cd ~/code/tokensmash
uv run tokensmash plan
```

The bundled default suite is generic. It does not assume your repositories, tools, or paths.

Run a small comparison against your own repo:

```bash
export TARGET_REPO=~/code/my-project
export TARGET_BASE_REF=HEAD
export TARGET_PROMPT='Fix the failing test with the smallest correct change.'
export TARGET_TEST_COMMAND='npm test'

uv run tokensmash plan --variants context_mode
uv run tokensmash run --live --variants context_mode --run-id my-project-context-mode
uv run tokensmash report ~/.local/state/tokensmash/ab-runs/my-project-context-mode/results.json -o reports/my-project-context-mode.md
```

`run` is dry-run by default. Add `--live` only when intentionally spending model quota.

## Built-In Tool Conditions

The generic suite includes baseline plus context-mode, Headroom, RTK, SEMMAP,
Repomix, and Gitingest. Variants skip cleanly when required environment
variables or commands are missing.

Use the `how it was applied` column above to decide whether the current variant
matches the way you would expect to use that tool.

## Auditing Your Own Session Logs

Tokensmash can summarize local agent session logs without copying raw transcripts into the repo:

```bash
uv run tokensmash sessions --agent all --days 7
uv run tokensmash sessions --agent codex --codex-root ~/.codex/sessions -o results/my-codex-sessions.json
uv run tokensmash sessions --agent claude --claude-root ~/.claude/projects -o results/my-claude-sessions.json
uv run tokensmash sessions --agent gemini --gemini-root ~/.gemini -o results/my-gemini-sessions.json
```

The output is sanitized aggregate JSON: file hashes, token counters, tool-call counts, and byte counts. Do not commit raw session logs. See [`docs/session-privacy.md`](docs/session-privacy.md).

## Adding A Tool

Add a `variants[]` entry to a suite:

- `id`: stable row key.
- `codex_args`: Codex exec args for the variant.
- `codex_home` and `codex_config`: optional isolated Codex config.
- `setup_commands`: optional commands that generate context artifacts in `{run_dir}`.
- `prompt_prefix`: optional instructions for how the agent should use the tool.
- `requires`: optional `env` and `commands` gates.

## Adding A Task

A task should include:

- `id`
- `repo`
- `base_ref`
- `prompt`
- `success_commands`

Use execution-based oracles whenever possible. Weak oracles produce weak benchmark conclusions.

## Current Limitations

- Codex CLI is the only live A/B runner today.
- Claude/Gemini support is currently session-log auditing, not live paired runs.
- One run is not a confidence interval; use multiple tasks, repos, and replicates before changing global defaults.
- Tool exposure is not always tool use; inspect session logs before making a default-policy decision.
