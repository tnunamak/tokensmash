# Tokensmash

Tokensmash is a benchmark harness for answering one practical question:

> Does enabling this agent tool reduce total token spend for normal successful agent tasks?

The primary metric is provider-reported session tokens, not a tool's self-reported savings.

## Current Results

Reference smoke run, 2026-06-11:

| tool         | baseline tokens | tool tokens | improvement | model   | reasoning | task class                 | oracle        | result |
| ------------ | --------------- | ----------- | ----------- | ------- | --------- | -------------------------- | ------------- | ------ |
| context_mode | 1265375         | 643229      | +49.2%      | gpt-5.5 | low       | single Go repository task  | go test ./... | pass   |
| headroom     | 1265375         | 591837      | +53.2%      | gpt-5.5 | low       | single Go repository task  | go test ./... | pass   |
| rtk          | 1265375         | 696266      | +45.0%      | gpt-5.5 | low       | single Go repository task  | go test ./... | pass   |
| semmap       | 999151          | 879352      | +12.0%      | gpt-5.5 | low       | single Go repository task  | go test ./... | pass   |
| repomix      | 620319          | 1309307     | -111.1%     | gpt-5.5 | low       | single Go repository task  | go test ./... | pass   |
| gitingest    | 620319          | 949669      | -53.1%      | gpt-5.5 | low       | single Go repository task  | go test ./... | pass   |

Interpretation: this is a smoke result, not a leaderboard. It is one task, one
replicate per row, Codex CLI only, and rows came from multiple bounded batches
with paired baselines per batch.

## What It Measures

Tokensmash runs paired baseline/tool variants in disposable Git checkouts, verifies the task with an execution oracle, and reads final Codex session token totals from local session logs.

Default metric:

```text
total_token_usage.total_tokens per successful task
```

## Prior Art

The repo layout borrows from established evaluation projects:

- [SWE-bench](https://github.com/swe-bench/SWE-bench): real repository tasks, reproducible evaluation logs, execution-based oracles.
- [SWE-bench harness docs](https://www.swebench.com/SWE-bench/reference/harness/): isolated evaluation environments.
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness): config-defined tasks and broad model/task extensibility.
- [lm-evaluation-harness task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md): task definitions as reusable configuration.
- [OpenCompass](https://github.com/open-compass/opencompass): model/dataset/config separation and benchmark-card style reporting across many datasets.

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

## Built-In Variants

The generic suite includes these tool conditions:

- `baseline_no_user_config`
- `context_mode`
- `headroom`
- `rtk`
- `semmap`
- `repomix`
- `gitingest`

Variants skip cleanly when required environment variables or commands are missing.

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
