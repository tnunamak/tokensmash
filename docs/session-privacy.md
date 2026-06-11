# Session Privacy

Tokensmash is designed to benchmark and audit agent sessions without publishing
private transcripts.

## Do Not Commit

Do not commit:

- raw Claude, Codex, or Gemini session logs,
- JSONL transcripts,
- SQLite session databases,
- raw tool outputs,
- prompts or assistant messages copied from private sessions.

The repository `.gitignore` excludes common raw-session paths and file types.

## Safe To Commit

Commit only sanitized summaries:

- aggregate token totals,
- tool-call counts,
- byte counts,
- hashed file identifiers,
- benchmark-card reports,
- manually reviewed result summaries.

## Local Audits

Use:

```bash
uv run tokensmash sessions --agent all --days 7 -o results/my-session-summary.json
```

The command reads local logs in place and writes aggregate JSON. It does not copy
the source logs into the repository.

## Agent Support

- Codex: high confidence when `token_count` events exist.
- Claude: medium-high confidence when per-message `usage` fields exist.
- Gemini: best-effort schema heuristics; pass `--gemini-root` for your local log path.
