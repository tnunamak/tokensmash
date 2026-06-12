# Module Contracts (build spec)

Authoritative interfaces for the study implementation. Workers implement
exactly these signatures; integration wires them into `cli.py`. Background:
`docs/study-architecture.md`. Ground rules for all modules:

- Python 3.11+, **stdlib only** (this project has zero runtime deps; keep it).
- Tests: `unittest`, under `tests/`, discoverable via
  `uv run python -m unittest discover -s tests -p 'test_*.py'`.
- **Never** commit real session content. You may read files under
  `~/.codex/sessions` and `~/.claude/projects` to learn formats; committed
  fixtures must be synthetic (realistic structure, fake content).
- Do not modify `src/tokensmash/cli.py` (integration happens separately).
- Import the schema contract from `tokensmash.schema` (already written):
  `SESSION_SCHEMA`, `USAGE_KEYS`, `empty_usage()`, `validate_session_record()`,
  `stable_id()`, `repo_identity()`, `machine_id()`, `STUDY_DIR`.

## Normalized session record

See `schema.py` docstrings and `docs/study-architecture.md` D3. Canonical
usage translation (the most important correctness rule in the codebase):

| canonical      | codex (`total_token_usage`)            | claude-code (`message.usage`)   |
|----------------|----------------------------------------|---------------------------------|
| fresh_input    | `input_tokens - cached_input_tokens`   | `input_tokens`                  |
| cache_read     | `cached_input_tokens`                  | `cache_read_input_tokens`       |
| cache_write    | `0`                                    | `cache_creation_input_tokens`   |
| output         | `output_tokens`                        | `output_tokens`                 |
| reasoning_output | `reasoning_output_tokens` or None    | `None`                          |

## 1. `tokensmash/sessions/codex.py` and `tokensmash/sessions/claude.py`

```python
def parse_session(path: Path) -> tuple[dict, list[dict]] | None:
    """Parse one transcript JSONL into (session_record, timeline).

    Returns None if the file contains no usable usage data.
    session_record: validates clean against schema.validate_session_record,
      with started_at/ended_at as UTC ISO-8601, model from the transcript,
      agent_version when present, session_id from the transcript (not the
      filename when an internal id exists), repo_id = stable_id("repo",
      repo_identity(cwd-from-transcript)), provider_raw = the final cumulative
      provider usage fields untouched.
    timeline: ordered list of events for opportunity analysis (below).
    """

def iter_session_files(root: Path) -> Iterator[Path]:
    """Yield transcript files under the agent's default root, newest last."""
```

Timeline event shapes (only these two kinds):

```jsonc
{"kind": "request", "index": 3, "usage": {…canonical usage DELTA for this API request…}}
{"kind": "tool_output", "request_index": 3, "category": "shell|file_read|search|mcp|web|other",
 "tokens_est": 1234, "tool_name": "Bash"}
```

Identity fields (set by both parsers): `transcript_id =
stable_id("transcript", str(path.resolve()))` — the store key; one transcript
file = one record — and `logical_session_id` — groups transcripts that belong
to one logical session (claude subagents share the parent's sessionId; resumed
codex sessions write multiple rollout files with the same internal id) and is
the arm-join key. `session_id` is NOT unique across records.

`tokens_est` = `max(1, len(text)//4)`. Per-request deltas for codex come from
diffing successive cumulative `token_count` events; suppress negative deltas
(session restarts) by clamping at 0 and record `"anomalies": [...]` notes on
the session record. Claude: one `request` event per assistant message bearing
usage; **dedupe streamed duplicates by message id, last wins**. Handle
sidechain/subagent transcripts: do not double-count a sidechain file into its
parent; a sidechain file parsed on its own returns its own record with
`"sidechain": true`. Count `compactions` (claude: compact boundary entries;
codex: compaction/`turn_context` markers if present, else 0) and `user_turns`
(genuine user messages, not tool results or meta). Unknown/extra JSONL entry
types must be skipped silently — parsers never crash on unknown input; corrupt
lines are skipped and counted in `"parse_errors": N`.

## 2. `tokensmash/pricing.py` + `src/tokensmash/data/pricing/*.json`

Data files (one per source, versioned by id):

```jsonc
{"id": "anthropic-api-2026-06", "kind": "api_usd", "agent": "claude-code",
 "retrieved_at": "2026-06-12", "source_urls": ["…"],
 "models": {"<model-id>": {"fresh_input_per_m": 3.0, "cache_read_per_m": 0.3,
   "cache_write_per_m": 3.75, "output_per_m": 15.0}},
 "match": [{"pattern": "sonnet-4", "model": "<model-id>"}]}
```

Same shape for `openai-api-…` (kind `api_usd`, agent `codex`) and
`codex-credits-…` (kind `codex_credits`, fields `…_per_m` in credits;
cache_write_per_m = 0). Prices/credits MUST be verified against the live
provider docs at build time (WebFetch), with the URLs recorded; do not trust
memory. Claude cache_write priced at the 5-minute multiplier; note in the file.

```python
def load_tables() -> list[dict]                      # all data files, validated
def resolve_model(tables, agent, model) -> tuple[dict, str] | None
    # (per-m rates, table_id) via exact match then `match` patterns; None if unknown
def cost_usd(usage: dict, agent: str, model: str) -> tuple[float, str] | None
def codex_credits(usage: dict, model: str) -> tuple[float, str] | None
    # both return None for unknown models — never guess
```

## 3. `tokensmash/study/assign.py`

Spec (architecture D4): block = `floor(unix_seconds / 7200)` (2h UTC blocks).
Permuted blocks of 8 per repo: `group = block // 8`; the 8 arms within a group
are a deterministic shuffle of `["on","off"]*4` keyed by
`HMAC(seed, f"{repo_id}:{group}")`; `arm = perm[block % 8]`.

```python
def block_index(unix_seconds: float) -> int
def arm_for(seed: bytes, repo_id: str, block: int) -> str          # "on"|"off"
def assignment_id(repo_id: str, block: int) -> str                 # f"{repo_id}-{block}"
def load_study_config() -> dict | None        # STUDY_DIR/config.json or None
def init_study(study_id: str, mode: str, protocol_version: str) -> dict
    # creates config.json with fresh hex seed; refuses to overwrite
```

Tests must prove: determinism, exact 4/4 balance within every group for many
random (seed, repo) draws, and different repos/groups get different
permutations (almost always).

## 4. `tokensmash/study/link.py`

```python
def link_from_hook(stdin_text: str, agent: str, now: float | None = None) -> int
    # parses hook JSON {session_id, cwd, transcript_path?, model?, source?},
    # computes repo_id/block/arm, appends ASSIGNMENT_SCHEMA record to
    # STUDY_DIR/assignments.jsonl. ALWAYS returns 0. NEVER writes to
    # stdout/stderr (SessionStart stdout is injected into agent context).
    # On any error: append one line to STUDY_DIR/errors.log and return 0.
    # If no study config exists: no-op, return 0.
    # source == "resume": record linkage but mark {"resumed": true}.
```

Assignment record: `{"schema": ASSIGNMENT_SCHEMA, "study_id", "agent",
"session_id", "repo_id", "block", "arm", "assignment_id", "mode",
"linked_at", "source", "transcript_path", "resumed"?}`.

## 5. `tokensmash/store.py`

```python
def append(path: Path, record: dict) -> None          # canonical line via schema.dumps_record
def load_latest(path: Path, key=("agent","session_id")) -> dict[tuple, dict]
    # read JSONL, dedupe by key, last wins, skip corrupt lines
def upsert_many(path: Path, records: list[dict]) -> tuple[int, int]
    # (added, replaced): append only records that are new or changed vs load_latest
def export_scrubbed(sessions_path: Path, out_path: Path) -> int
    # drop keys: transcript_path and any absolute path fields; keep hashes;
    # validate every record before writing; returns count
```

## 6. `tokensmash/ingest.py`

```python
def ingest(roots: dict[str, Path], since_days: float | None = None) -> dict
    # walks codex+claude roots via sessions.iter_session_files, parses,
    # computes costs (pricing) + opportunity aggregates (opportunity.summarize),
    # attaches arm/assignment by joining assignments.jsonl on
    # (agent, logical_session_id), falls back to deterministic recompute by
    # (repo_id, block) when a study config exists, upserts into
    # STUDY_DIR/sessions.jsonl keyed by (agent, transcript_id), marks
    # excluded="codex-superseded-rollout" on all but the max-cumulative
    # rollout per logical codex session (counters are cumulative across
    # resumes; summing would double-count), and excluded="block-boundary"
    # on sessions spanning 2h blocks (PROTOCOL §4),
    # returns {"scanned": n, "parsed": n, "added": n, "replaced": n, "skipped": n,
    #          "parse_errors": n, "unknown_model_sessions": n}
```

Sessions whose repo_id matches `config.exclude_repo_ids` (e.g. the tokensmash
repo itself) get `"excluded": "study-repo"`.

## 7. `tokensmash/opportunity.py`

```python
def summarize(timeline: list[dict], compactions: int) -> dict
    # per category ∈ {shell, file_read, search, mcp, web, other}:
    #   {"insertion_tokens": Σ tokens_est,
    #    "reread_tokens": Σ tokens_est * rereads_until_compaction}
    # where rereads = number of later request events before the next compaction
    # boundary (approximate compaction positions as evenly splitting the
    # request sequence into compactions+1 segments).
def tool_ceilings(record: dict) -> dict
    # maps categories to per-tool USD ceilings using the session's model pricing:
    # rtk → shell; context-mode → shell+search+mcp; repomix/packers → file_read;
    # headroom → all request payload (fresh_input + cache_read).
    # Each ceiling reported as {"insertion_only_usd": x, "with_rereads_usd": y}.
    # Assumes 100% compression — these are deliberate upper bounds.
def report(records: list[dict]) -> str    # human-readable table + caveats
```

## 8. `tokensmash/study/power.py`

```python
def block_costs(records: list[dict]) -> list[float]      # per (repo, block) USD totals
def mde(records: list[dict], weeks: float, alpha=0.05, power=0.8) -> dict
    # two-sample MDE on block-level costs: uses observed block-cost mean/var,
    # blocks-per-week observed rate, returns {"blocks_per_week": r, "cv": …,
    # "mde_pct": …, "n_blocks": …}; document formula in docstring
def report(records: list[dict]) -> str
```

## CLI integration (done at integration time, not by workers)

`tokensmash ingest`, `tokensmash opportunity`, `tokensmash study init|link|power|export`.

---

# Wave-1 Contracts (analyze, launch, install, replay, trajectory, meta)

Same ground rules as above: stdlib only, unittest, never touch cli.py, never
commit real session content, fixtures synthetic.

## 9. `tokensmash/study/analyze.py` — pre-registered inference

Implements PROTOCOL.md §5 exactly. Inputs: session records (transcript-keyed),
assignments, study config.

```python
def analyze(records, config, protocol_text: str) -> dict   # full result object
def report(result: dict) -> str                            # human-readable
```

Hard guards (raise AnalysisRefused with a reason string):
- sha256(protocol_text) != config["protocol_sha256"]
- any record with study fields whose protocol_version != config protocol_version
- config missing live_started_at or mode != "live"

Data preparation:
1. Keep records with arm in ("on","off"), excluded falsy, cost_api_usd not
   None, started_at >= live_started_at, repo_id not in exclude_repo_ids.
2. Block value Y(repo, block) = Σ cost_api_usd of that block's records.
   Arm of a block = PRF arm (recompute via assign.arm_for; ignore record
   labels if they disagree — and count disagreements as "label_mismatches").
3. Pair construction: within each (repo, group=block//8): mean Y over
   non-empty on-blocks vs mean Y over non-empty off-blocks. Pair exists only
   if both arms have >=1 non-empty block. d_i = mean_on_i - mean_off_i.

Primary test (sign-flip permutation):
- T_obs = mean(d_i). For 10,000 iterations with random.Random(20260612):
  flip each d_i sign with p=0.5, record mean. p_two_sided =
  (1 + #{|T_perm| >= |T_obs|}) / (10_001).
Secondary (CUPED): X_i = repo pre-study mean block cost (records with
  started_at < live_started_at, same block-value construction, no arm filter).
  theta = cov(d's component...) — NO: CUPED adjusts Y_blocks: for each pair,
  adjusted d_i' = d_i (X cancels within-repo pairing; document this — within-
  repo pairs already difference out repo level, so CUPED here adjusts for
  group-level drift instead): regress d_i on (X_i - mean(X)) via OLS slope b;
  d_adj_i = d_i - b*(X_i - mean(X)); report mean(d_adj), its naive SE, and b.
Robustness: cluster-robust (CR0) SE for mean(d_i) clustering pairs by repo:
  SE^2 = Σ_repo (Σ_{i in repo} (d_i - mean(d)))^2 / (n_pairs^2) scaled by
  G/(G-1) with G = #repos; z = mean(d)/SE; two-sided normal p.
Co-primary (codex credits) and secondary (fresh_input tokens): same pipeline
with Y = cost_codex_credits (codex records only) and Y = usage.fresh_input.
Guardrails (report-only, no tests): per-arm means of user_turns, compactions,
duration_ms per session; abandonment = sessions with tool_calls == 0.
Coverage: among arm="on" sessions in-window, fraction with a matching
actuation record (join actuations.jsonl on (agent-ish tool field, repo_id,
block)); report on/off counts, pairs, label_mismatches, excluded counts.
Result dict carries every intermediate (pairs list, T_obs, p, theta/b, SEs,
coverage, guardrails) — analyze is also the audit trail.
Tests: hand-computed tiny example (4 pairs, exact T_obs/p bounds), guard
refusals, pair construction edge cases (one-armed groups dropped), CR0 SE on
a 2-repo example, deterministic permutation (seed fixed).

## 10. `tokensmash/study/launchctl.py` — portable actuation

```python
def resolve_launch(agent_cli: str, argv: list[str], study_dir=None) -> dict
    # {"exec": [argv...], "arm": "on"|"off"|None, "resolution": {...}|None}
def install_report(apply: bool = False) -> dict   # see install semantics
```

resolve_launch:
- agent_cli ∈ {"claude","codex"}; maps to agent name {"claude-code","codex"}.
- If env TOKENSMASH_LAUNCH_ACTIVE set → plain exec of the real binary.
- Resolution via assign.arm_for_cwd(os.getcwd()); None → plain exec.
- arm "on": load the registry file data/tools/<config['tool']>.json;
  build exec argv from its on_command with "<tool>" replaced by agent_cli,
  appending argv after the literal "--"; merge registry "env" dict into
  os.environ for the exec; log via assign.log_actuation (arm "off" logs too).
- Real-binary resolution: shutil.which(agent_cli) — document that PATH shims
  are not supported (we shadow via shell functions, so which() finds the real
  binary). If which fails → return {"exec": None, "error": ...}.
- NEVER raises; any exception → plain exec fallback with "error" noted.
- Add "env" key to data/tools/headroom.json: {"HEADROOM_TELEMETRY": "off"}.

install semantics (install_report):
- Computes desired state: (a) SessionStart hook line for Claude settings.json,
  (b) SessionStart hook entry for codex hooks.json, (c) shell function snippet.
- --check (apply=False): returns per-item {present: bool, path, snippet}.
- apply=True: idempotently inserts (a) and (b) by JSON edit (create file if
  missing, preserve existing entries, never duplicate); NEVER edits shell rc
  files — returns the snippet text for the user to add. Respects env
  TOKENSMASH_CLAUDE_SETTINGS / TOKENSMASH_CODEX_HOOKS path overrides (defaults
  ~/.claude/settings.json, ~/.codex/hooks.json) so tests use tempdirs.
Tests: registry argv construction, recursion guard, fail-open on missing
config/registry, install idempotency (apply twice → no dupes), check mode.

## 11. `tokensmash/replay.py` — offline realized-compression estimates

Purpose: convert Layer-0 ceilings (100% compression) into realized point
estimates by running the actual tools over tool outputs recorded in
transcripts. Per-tool feasibility decided by INVESTIGATION (record findings in
module docstring): rtk (binary on PATH; legacy script at
~/.tmp/tokensmash-legacy-20260612/tokensmash shows a working pattern for
piping recorded output through reducers — mine it), headroom (legacy script
used its python lib offline), repomix (no repo state at session time —
explicitly out of scope; document).

```python
def replay_session(path: Path, tools: list[str]) -> dict | None
    # per category: {"tokens_in": n, "tokens_out": n, "ratio": float, "samples": n}
def replay_corpus(roots: dict[str, Path], tools, limit_sessions=None, seed=17) -> dict
    # random sample of sessions (seeded), aggregates + per-tool realized ratio
def report(result: dict) -> str
    # ceilings table reprinted with "realized estimate" column =
    # insertion ceiling × realized ratio (+ capped reread × ratio), with the
    # honest caveat block
```

Subprocess calls get timeouts; tool failures counted, never fatal. Tests use a
fake reducer script (fixture shell script) so CI never depends on rtk/headroom
being installed; real-tool runs happen at integration.

## 12. `tokensmash/trajectory.py` + parser `target` field

Parsers (sessions/codex.py, sessions/claude.py) gain ONE additive field on
tool_output timeline events: "target" — claude: file_path for Read/Edit/Write,
first ~120 chars of command for Bash, pattern for Grep/Glob; codex: same idea
from arguments. Targets are in-memory only (timeline is never persisted) —
existing tests must keep passing; update fixtures minimally if needed.

```python
def analyze_session(record, timeline) -> dict
    # {"reads": n, "reads_unused": n, "tokens_unused": n, "searches": n,
    #  "searches_unfollowed": n, ...}
def analyze_corpus(roots, limit_sessions=None, seed=17) -> dict
def report(result: dict) -> str
```

"Unused read" = file_read whose target path never appears later in the session
(in any subsequent tool target or in an Edit/Write/apply_patch). The metric is
a proxy and the report must say so. Aggregates only; no paths in output
(counts and token totals).

## 13. `tokensmash/meta.py` — multi-user aggregation

```python
def merge(paths: list[Path]) -> list[dict]    # validate, dedupe (machine_id,
                                              # agent, transcript_id), count conflicts
def report(records: list[dict]) -> str        # per-machine + combined: sessions,
                                              # cost, cache share, opportunity
                                              # ceilings (reuse opportunity.report
                                              # internals where exported)
```

Accepts scrubbed exports only (refuse records with transcript_path).

## 14. README + replication kit (docs worker)

- Rewrite README.md: what tokensmash is now (measurement instrument: ingest /
  opportunity / study / replay), the 10-minute quickstart, honest caveats
  (API-equivalent USD, Claude quota opacity, ceilings-vs-estimates), pointer to
  PROTOCOL.md + docs/study-architecture.md, legacy benchmark relegated to a
  section with its known limitations.
- PROTOCOL.template.md: blank registration template (structure of PROTOCOL.md
  with <fill> markers and instructions, §7 formulas retained).
- docs/replication.md: install (uv tool install), study init → dry-run →
  power gate → live, what gets shared (scrubbed export schema), one-command
  install via `tokensmash study install --check`.

## 15. Lab-bench repairs (wave 2; MAY edit cli.py bench region only)

Repairs to the synthetic A/B harness, fixing the README "Known limitations":
1. **Latin-square position balancing**: `run --balance-positions` generates the
   case order so that across the suite, each variant occupies each within-task
   position equally often (for V variants and T tasks with T % V == 0 use a
   Latin square; otherwise closest balanced design; record the design matrix in
   results.json). `--randomize-order` remains for backward compat; strict mode
   now REQUIRES results["randomized_order"] is True OR balanced_positions
   present, plus run order complete.
2. **Baseline replicates**: `--baseline-replicates N` runs the baseline N times
   per task (replicate ids baseline r1..rN); aggregate pairs each tool run
   against the MEAN of baseline replicates and reports baseline SD per task.
3. **Code-version stamp**: results.json gains "code_version" (git describe or
   package __version__ + git sha when available); `aggregate --strict` refuses
   result files lacking it or differing in major behavior version (a
   BENCH_AUDIT_VERSION constant bumped with audit-semantics changes).
4. **Min-N gating**: aggregate hides bootstrap CIs below 10 pairs (prints
   "pilot (n<10)" instead) and labels all tables with pairs count.
Tests for each (results-dict level; no live agent runs in CI).

## 16. `tokensmash/otelcheck.py` — second-meter cross-validation

```python
def parse_otlp_jsonl(path: Path) -> dict[str, dict]   # session.id → canonical usage sums
def compare(otel: dict, records: list[dict]) -> dict  # per-session deltas, match rate
def report(result: dict) -> str
```
Reads OTLP JSON-lines exports of Claude Code's `claude_code.token.usage`
metric (attributes: type ∈ input/output/cacheRead/cacheCreation, session.id).
Canonical mapping: input→fresh_input, cacheRead→cache_read,
cacheCreation→cache_write, output→output. compare joins on session_id for
claude-code records and reports absolute/relative deltas; report flags any
session disagreeing by >1%. docs: one paragraph in docs/replication.md on the
OTel env vars (CLAUDE_CODE_ENABLE_TELEMETRY=1, OTEL_METRICS_EXPORTER, file
collector) as the optional second meter. Synthetic fixtures only.
