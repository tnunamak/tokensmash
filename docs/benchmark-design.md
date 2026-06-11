# Benchmark Design

Tokensmash evaluates agent tools by end-to-end task cost.

## Metric

Primary metric:

```text
provider-reported total_token_usage.total_tokens per successful task
```

This deliberately avoids:

- reducer self-reported savings,
- raw shell output byte deltas,
- one-off context pack sizes,
- non-verified agent runs.

Those can be useful diagnostics, but they do not answer whether the whole agent session got cheaper.

## Task Shape

The task shape follows SWE-bench's core lesson: evaluate real repo work against an execution oracle.

Each task captures:

- repository path,
- immutable base ref,
- user-facing prompt,
- success command.

Portable suite files should use environment placeholders such as `${TARGET_REPO}`,
`${TARGET_PROMPT}`, and `${TARGET_TEST_COMMAND}` instead of host-specific absolute
paths.

## Variant Shape

A variant is a tool condition. It can change:

- Codex config,
- Codex home,
- MCP servers,
- hooks,
- setup artifacts,
- prompt instructions.

Each variant is paired with a baseline from the same task and replicate.

## Report Shape

Reports are intentionally similar to model cards and benchmark cards:

- model and reasoning effort,
- task and oracle,
- baseline tokens,
- tool tokens,
- improvement,
- token breakdown,
- limitations and provenance.

## Interpretation Rules

- Positive one-task results are directional only.
- Negative packer results are more actionable when the mechanism is obvious: handing a whole repo digest to the agent can dominate context.
- A tool-exposed row is not always a tool-used row. Session logs should be audited for actual calls before enabling defaults.
- Global defaults need multiple tasks, repos, agents, and replicates.

## Session Log Audits

`tokensmash sessions` accepts local Claude, Codex, and Gemini roots and emits
sanitized aggregate summaries. It does not copy raw logs or print raw prompts.

Session audits are not causal benchmarks. Use them to understand real workload
shape, token volume, and tool-output pressure before choosing which live A/B
suites to run.
