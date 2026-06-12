# Tokensmash Study Architecture

Status: draft for review. This document works out the architectural decisions for
turning tokensmash into a defensible measurement instrument: (Layer 0) observational
opportunity bounds over real sessions, (Layer 1) a randomized crossover study wired
into real agent harnesses, (Layer 2) the existing lab benchmark, demoted to
supporting evidence. Each decision is numbered so review can reference them.

## D1. Repo boundary: instrument vs deployment

- `tokensmash` (this repo) is the portable instrument: session parsers, normalized
  accounting, pricing data, arm assignment, analysis CLI, protocol templates, and
  the existing live benchmark.
- `~/code/dotfiles` is one reference deployment: launcher wrappers and hook entries
  that *call* tokensmash. Nothing machine-specific lives in tokensmash; nothing
  measurement-critical lives in dotfiles.
- Anyone else's deployment = install the CLI + add two hook lines + a launcher
  wrapper. That separation is what makes results comparable across users.

## D2. Data store: append-only JSONL, not SQLite

`~/.local/state/tokensmash/study/` holds:

- `sessions.jsonl` — one normalized record per agent session (schema below).
- `assignments.jsonl` — one record per (repo-block) arm assignment and per
  session linkage event.

Records are idempotent by key (`agent`, `session_id`); re-ingest is a no-op.
Rationale: transparent and inspectable by reviewers, diffable, mergeable across
machines/users by concatenation, trivially exportable. Volumes are thousands of
records; a database adds opacity without solving a problem we have. A SQLite cache
can be derived later if analysis wants it.

## D3. Normalized session record (the spec others implement)

The single most error-prone detail in this whole design is that **providers define
`input_tokens` differently**:

- **Codex/OpenAI**: `cached_input_tokens ⊆ input_tokens`; fresh input =
  `input_tokens − cached_input_tokens`; `total = input + output`.
- **Anthropic**: `input_tokens` *excludes* cache activity; total input =
  `input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, and cache
  writes are billed at a premium (1.25x for 5-min TTL, 2x for 1-hr).

The normalized schema therefore stores five canonical fields and never reuses
provider names:

```jsonc
{
  "schema": "tokensmash-session/1",
  "agent": "codex | claude-code",
  "session_id": "...",
  "machine_id": "hmac(...)",          // keyed hash, never hostname
  "study_id": "...", "protocol_version": "...",
  "started_at": "...", "ended_at": "...",
  "model": "...", "agent_version": "...",
  "repo_id": "hmac(git remote or cwd)",
  // workload covariates (for CUPED / stratification)
  "user_turns": 0, "tool_calls": 0, "compactions": 0, "duration_ms": 0,
  // canonical usage (sum over all API requests in session)
  "usage": {
    "fresh_input": 0,        // tokens billed at full input rate
    "cache_read": 0,         // tokens billed at cache-read rate
    "cache_write": 0,        // Anthropic only; 0 for Codex
    "output": 0,
    "reasoning_output": null // subset of output where reported
  },
  "provider_raw": { ... },   // untouched provider fields, for audit
  // derived, always with data-file provenance
  "cost_api_usd": 0.0, "pricing_id": "openai-2026-05",
  "cost_codex_credits": 0, "credit_rate_id": "codex-credits-2026-05",
  // study fields (absent outside a study)
  "arm": "on | off", "assignment_id": "...",
  "excluded": null           // e.g. "block-boundary", "resumed-cross-block"
}
```

No prompt or output text ever enters this store; paths and identifiers are keyed
HMACs. The scrubbed export is the same schema, so "release the raw data" is the
default, not an afterthought.

Primary meter: the agent's own transcript JSONL (`~/.codex/sessions`,
`~/.claude/projects`), because both agents persist per-request provider-reported
usage there. Claude Code's official OTel metrics (`claude_code.token.usage`, which
carry session.id and the four token types) are an optional second, independent
meter — running both and showing they agree is cheap credibility.

## D4. Arm assignment: deterministic, blocked, stratified

Assignment unit is **repo × UTC time block (2 hours)**, not session:

- Provider prompt caches persist 5–60 minutes. Sessions on the same repo within
  the cache window share cache state, so arms must not interleave within it.
  Blocks of 2h ≥ 2× the longest cache TTL make carryover structural rather than
  statistical. Sessions spanning a block boundary are flagged `excluded`.
- Arm is computed, not stored: `arm = PRF(study_seed, repo_id, block_index)` with
  permuted-block balancing per repo (seeded PRP over block indices guarantees
  near-balance in any window). Determinism means concurrent sessions agree, no
  state is needed at assignment time, any component can recompute the arm, and the
  whole assignment sequence is reproducible from the seed in the protocol.
- Stratification by repo is automatic (assignment is within-repo); model and agent
  version are recorded covariates, not strata.

Blinding: nothing in the user-visible session ever prints the arm. The launcher
and shims are silent. (Single-subject studies can't fully blind the operator, but
they can avoid advertising the arm at every prompt.)

## D5. Linkage: SessionStart hook, fallback at ingest

A SessionStart hook on each agent pipes its JSON stdin (which includes
`session_id`, `cwd`, `model`, `source`) to `tokensmash study link`, which appends
an assignment record. Two hard requirements:

- **The hook must print nothing.** SessionStart hook stdout is injected into agent
  context; any output would contaminate the measurement it exists to support.
- **The hook is installed identically in both arms.** Linkage is instrumentation,
  not treatment.

`source: resume` keeps the original session's arm; `clear`/`compact` re-link
(deterministic assignment makes this idempotent). Fallback when the hook didn't
run: ingest matches sessions to assignments by `repo_id` + start-time within the
block — and because arms are deterministic functions of (repo, time), a missing
linkage record can always be reconstructed.

## D6. Actuation: a tool registry with three actuation classes

Tools differ in *where* their treatment lives, so one toggle mechanism can't cover
them. `data/tools/*.json` declares each tool's class:

1. **hook-shim** (RTK, context-mode lifecycle hooks): the agent's hook entry
   points at `tokensmash shim <tool>` permanently, in both arms. The shim
   recomputes the arm and either execs the real tool hook or passes stdin through
   unchanged. Settings files are byte-identical across arms — zero config-diff
   confound, zero context-size diff (hook *definitions* don't enter context).
2. **mcp-overlay** (context-mode MCP, Repomix MCP): MCP tool schemas enter context
   at startup, and "off" must mean *no schemas*. Actuated at launch: the `claude`
   wrapper appends `--settings overlays/<arm>.json --mcp-config <arm-servers>
   --strict-mcp-config`; the `codex` wrapper appends `--profile ts-<arm>`
   (profiles layer `$CODEX_HOME/<name>.config.toml` over base config). Verified
   surfaces; no global config editing, no restow.
3. **wrapper** (Headroom): the launcher chooses `headroom wrap claude …` vs
   `claude …`.

The registry is the community extension point: adding a tool = one JSON file
declaring its class and on/off config, no harness code.

## D7. Estimand and analysis: intention-to-treat at the block level

- **Estimand:** the effect of the *policy* "tool default-enabled" on
  billed-equivalent cost per block of natural work — intention-to-treat. Whether
  the agent actually used the tool is an outcome, not a filter. (The lab
  benchmark's mechanism-conditioning is a selection bias we deliberately do not
  import; mechanism-stratified analyses are secondary/exploratory.)
- **Primary outcome:** USD-equivalent cost per repo-block, computed from canonical
  usage with versioned pricing. **Codex co-primary:** credits per block — OpenAI
  publishes per-token-type credit rates, so subscription quota pressure is exactly
  computable for Codex. For Claude subscriptions the weighting is opaque; we
  report API-equivalent USD and say so plainly.
- **Variance reduction:** CUPED using each repo's pre-study mean block cost as the
  covariate; cluster-robust inference with repo as the cluster (or a simple
  paired-block permutation test — pre-registered, both specified in advance).
- **Guardrails:** user turns, compactions, session abandonment, wall-clock per
  block. A tool that "saves tokens" by degrading the agent must show up as a cost.
- `tokensmash study analyze` implements exactly the pre-registered plan and
  refuses to mix protocol versions in one analysis.

## D8. Power gating: Layer 0 decides whether Layer 1 runs at all

`tokensmash study power` estimates block-cost variance from ingested real sessions
and reports the minimum detectable effect for a given study duration. The
protocol's go/no-go rule: a tool enters the RCT only if its Layer-0 opportunity
bound exceeds the MDE achievable in ~8 weeks. Honest consequence: a tool with a
0.2% ceiling is *unmeasurable and not worth measuring* — publishing the bound is
the result. This is the token-budget answer: experiments are only run where an
effect is both plausible and detectable.

## D9. Opportunity bounds (Layer 0): generous, documented ceilings

Per-tool addressable spend is computed from the per-request timeline of real
sessions with formulas that deliberately over-estimate what the tool could save
(100% compression assumed), so they are defensible as upper bounds:

- A tool output inserted at turn *t* costs fresh-input once **plus cache-read on
  every subsequent request until the next compaction** (smaller context also means
  cheaper re-reads — this multiplier works in the tools' favor and must be
  included for the bound to be fair).
- Report both insertion-only and re-read-inclusive ceilings.
- Tool scopes: RTK → shell-output tokens; context-mode → tool outputs above its
  interception threshold; Repomix/packers → file-read tokens; Headroom → total
  request payload (its compression can also *break* prefix caching, so its bound
  is reported alongside a cache-breakage sensitivity note).

## D10. Pricing as versioned data, never constants

`src/tokensmash/data/pricing/*.json`: per-model API prices (including Anthropic
cache-write multipliers) and Codex credit rates, each with effective dates and
source URLs. Every derived cost in every record carries the pricing file id. The
hard-coded `GPT55_SHORT_CONTEXT_PRICES` constant in `cli.py` migrates here.

## D11. Multi-machine and multi-user

Every record carries `machine_id` and `study_id`. `tokensmash study export
--scrub` emits the shareable schema; merging datasets is concatenation thanks to
idempotent keys. Cross-user replication is the long-term credibility play:
tokensmash publishes the protocol and instrument, each participant owns their
data, aggregate analyses cite N users × N weeks.

## D12. Code structure

`cli.py` (3,043 lines, 6 tests) splits as Layer 0 is built — extraction, not
rewrite:

```
src/tokensmash/
  sessions/codex.py, claude.py   # transcript parsers → normalized records
  pricing.py                     # loads data/pricing, computes costs
  opportunity.py                 # Layer-0 ceilings
  study/assign.py, link.py, analyze.py, power.py, shim.py
  bench/                         # existing live A/B harness, unchanged for now
  data/pricing/, data/tools/, data/suites/
```

Parsers get golden-fixture tests from scrubbed real sessions (the repo already has
`docs/session-privacy.md`; the scrub pipeline precedent exists). The assignment
PRF, normalization, and cost math get exhaustive unit tests — they are the parts a
skeptical reviewer will re-derive.

## D13. Known limitations and threats (stated, not hidden)

- **Single subject** until others replicate; mitigated by within-repo blocking,
  pre-registration, and the portable kit.
- **No full blinding**; mitigated by silent arms and pre-registered analysis.
  Behavior drift is monitored by comparing off-arm cost against pre-study baseline.
- **Claude subscription quota weighting is opaque**; only API-equivalent USD and
  Codex credits are claimed.
- **Model/agent upgrades mid-study** are recorded per session and handled as
  covariates; a major model change triggers a protocol-versioned analysis split.
- **Tokensmash's own development sessions** are excluded by repo_id rule listed in
  the protocol.
- **Lab benchmark (Layer 2)** remains subject to its known repairs (position
  balancing, baseline replicates, version-stamped strict gate, session-derived
  tasks) and is cited only for tool × task-class questions.

## Build order

1. **M0 — Accounting + Layer 0:** pricing data files, normalized parsers for both
   agents, `tokensmash ingest`, opportunity report over full session history.
   Forces the module split; produces the first publishable number (the bounds).
2. **M1 — Protocol + power:** `PROTOCOL.md` (estimand, arms, blocks, exclusions,
   analysis plan, go/no-go rule) committed and hashed before any assignment;
   `study power` over M0 data.
3. **M2 — Dry-run study:** assignment + shims + wrappers + link hooks deployed in
   dotfiles with actuation in *log-only* mode for ~1 week — validates linkage,
   balance, and ingest end-to-end without treating anything.
4. **M3 — Live study** on whichever tools pass the M1 gate.
5. **M4 — Replication kit:** scrubbed export, public protocol, install docs for
   non-bespoke deployments; Layer-2 repairs as demanded by reviewers.
