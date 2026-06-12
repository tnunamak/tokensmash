# Tokensmash Crossover Study Protocol

**Protocol version:** 0.1.0-draft  
**Status:** DRAFT — not yet registered. This file must be SHA-256 hashed and
committed (with the hash recorded in `PROTOCOL.md` itself or `assignments.jsonl`
header) before any arm assignment is made. Any post-assignment edit to this file
constitutes a registered deviation.

---

## 1. Research Questions and Estimand

**Primary question:** Does enabling a token-saving agent tool (RTK, context-mode,
Repomix, Headroom) as the default policy reduce billed-equivalent cost per unit
of natural agent work?

**Estimand:** Intention-to-treat (ITT) effect of the policy "tool default-enabled"
on USD-equivalent block cost per repo × 2h-block of natural agent work, computed
from canonical usage with versioned pricing (D7). Whether the agent invoked the
tool within any given block is an outcome, not an inclusion criterion.

**Co-primary for Codex sessions:** effect on published credit units per block,
computed from Codex per-token-type credit rates (`credit_rate_id` recorded per
session). This is the subscription-quota-pressure metric for Codex. For Claude
Code, subscription quota weighting is opaque; only API-equivalent USD is claimed
(D13).

---

## 2. Design

**Unit of randomization:** repo × UTC 2h block (`floor(unix_seconds / 7200)`,
block index per D4 / `assign.block_index`). Sessions within a block share prompt
cache state; 2h ≥ 2× the longest cache TTL (60 min) makes carryover structural
rather than statistical.

**Arms:** on (tool default-enabled) vs off (tool absent), per the tool under
test. One tool per study epoch; arm definitions are fixed in `data/tools/<tool>.json`
at study init.

**Assignment rule:** Deterministic PRF with permuted blocks of 8.
`arm = PRF(study_seed, repo_id, block_index)` where the permutation over each
group of 8 consecutive blocks is a keyed shuffle of `["on","off"]*4`, keyed by
`HMAC(seed, f"{repo_id}:{group}")`, and `group = block // 8`. This guarantees
exact 4/4 balance within every group (D4, `assign.arm_for`).

**Actuation:** Launcher-based, never exposed to the agent's *context*. Two
acknowledged limitations: (1) `headroom wrap` prints a banner at launch, so the
operator is not blind to the arm; (2) only sessions started via the shell
wrapper functions are actuated — sessions launched by scripts, IDEs, or
devcontainers carry arm labels but no actuation, diluting the ITT estimate
toward zero (conservative). Actuation coverage is measured by joining
`actuations.jsonl` against `assignments.jsonl` and reported with results.
Hook definitions are byte-identical across arms (D4, D6).
Actuation class per tool is declared in its registry file:
- hook-shim: RTK, context-mode lifecycle hooks  
- mcp-overlay: context-mode MCP, Repomix MCP  
- wrapper: Headroom  

**Subject count:** Single subject (D13). Cross-user replication uses the shared
export schema (D11) and is the credibility roadmap, not a condition of this
study epoch.

---

## 3. Outcomes

### Primary
`cost_api_usd` per repo × 2h block, computed from canonical usage fields
(`fresh_input`, `cache_read`, `cache_write`, `output`) with per-model rates from
the versioned pricing file (`pricing_id`). Each session record carries its
`pricing_id`; the analysis uses the pricing file active at session end.

### Co-primary (Codex only)
`cost_codex_credits` per repo × 2h block, from the versioned credit table
(`credit_rate_id`, per D10).

### Secondary
`fresh_input` tokens per block (canonical field, D3). Isolates the reduction in
fully-billed input, stripped of cache effects.

### Guardrail outcomes (monitored, not tested)
`user_turns`, `compactions`, session abandonment rate (sessions with no tool
calls after turn 1), and `duration_ms` (wall-clock) per block. A statistically
significant increase in any guardrail outcome in the on-arm triggers a mandatory
deviation log entry and pauses enrollment for that tool pending review (D7, D13).

---

## 4. Eligibility and Exclusions

**Included sessions:** Any agent session linked to the active study epoch whose
block assignment is unambiguous and whose repo is not excluded.

**Excluded from primary analysis (field `excluded` set at ingest):**

| Exclusion rule | `excluded` value | Rationale |
|---|---|---|
| Session spans a block boundary (started_at block ≠ ended_at block) | `"block-boundary"` | Ambiguous arm (D4) |
| Resumed session crosses block boundary from original block | `"resumed-cross-block"` | Arm is that of original session; cross-block carry-over violates block independence |
| Repo is the tokensmash development repo | `"study-repo"` | Measurement confound (D13); excluded via `config.exclude_repo_ids` |
| Superseded rollout file of a resumed Codex session | `"codex-superseded-rollout"` | Codex token counters are cumulative across resumed rollout files of one logical session; only the max-cumulative snapshot counts, others would double-count |

**Resumed sessions (no boundary crossing):** keep the original session's arm;
`"resumed": true` is set in the assignment record (D5). These sessions are
included in the primary analysis.

**Model changes during study:**
- Minor version changes (patch, suffix): recorded as covariate; analysis
  proceeds with `model` as a covariate.
- Major model change (e.g., Sonnet 4 → Sonnet 5 or a pricing discontinuity):
  triggers an analysis split by protocol version. The post-change epoch is
  treated as a new study epoch; the two epochs are not pooled in the primary
  analysis. A deviation log entry is required. (D13)

---

## 5. Analysis Plan

### Estimator

**Primary inferential test:** Paired-block permutation test. For each repo,
blocks are paired on-vs-off within each group of 8 (4 pairs). The test statistic
is the mean within-pair difference in block cost. The reference distribution is
generated by randomly flipping pair signs (sign-flip permutation, 10,000
iterations). Two-sided p-value from this distribution. This is the pre-committed
primary test.

**Secondary estimator (CUPED):** Adjust each block cost by the regression of
block cost on each repo's pre-study mean block cost (computed from ingested
sessions prior to study init, per D7). The CUPED-adjusted difference is reported
as a secondary estimate alongside the permutation-test p-value.

**Cluster-robust SE:** Also reported as a robustness check, with repo as the
cluster, using HC3 sandwich variance. Not used for the primary p-value; included
for comparison with conventional reporting.

**Alpha:** 0.05, two-sided.

**Analysis implementation:** `tokensmash study analyze`, which reads
`sessions.jsonl` + `assignments.jsonl`, applies the pre-registered exclusions,
and refuses to mix protocol versions in one analysis run (D7).

### Interim analyses
No unblinded interim tests. The following pre-specified power checks are
permitted without adjustment:

1. At 25% of target blocks: confirm observed block-cost variance is within 2×
   the pre-study estimate used for the sample size calculation. If variance
   exceeds 2×, extend sample size proportionally and record as a deviation.
2. At 50% of target blocks: confirm guardrail outcomes show no statistically
   significant degradation (α = 0.01 for guardrails at interim).

### Missing data rules
- Sessions with unknown model (pricing lookup returns None): excluded from
  `cost_api_usd` outcome; included in `fresh_input` outcome if usage is present.
- Sessions with `parse_errors > 0` but non-zero usage: included with a flag;
  `provider_raw` is retained for audit.
- Blocks with zero sessions in either arm: excluded from the paired test for
  that repo in that group.

---

## 6. Go/No-Go Gate

A tool may enter the live study (M3) only if its Layer-0 opportunity ceiling
(`tool_ceilings.with_rereads_usd` from `tokensmash opportunity`) exceeds the
minimum detectable effect achievable in approximately 8 weeks of natural work,
as reported by `tokensmash study power` (D8).

Formally: **tool enters if `ceiling_usd_per_block > MDE_usd_per_block`** where
`MDE_usd_per_block` is the 8-week MDE from the power report at α = 0.05,
power = 0.8. If the ceiling is below the MDE, the tool is documented as
"unmeasurable at current usage levels" and the Layer-0 bound is the reportable
result.

---

## 7. Sample Size

**Filled from `tokensmash study power` over 47.4 weeks of pre-study sessions
(2026-06-12):** mean block cost $8.31, SD $14.27 (CV 1.72), 34.1 non-empty
blocks/week, block occupancy p = 0.41. At 8 weeks: 136 blocks/arm, raw MDE
$4.84/block (58.3% of mean); effective pairs ≈ 22.5, effective MDE
$11.92/block (143% of mean).

**Gate outcome:** rtk (ceiling ≤19.1% of cost), context-mode (≤20.8%), and
repomix (≤18.4%) fall below the MDE — documented as unmeasurable at current
usage levels; their Layer-0 bounds are the reportable result. **headroom**
(wire-payload ceiling 77.3%) exceeds the raw MDE and is the sole tool in this
epoch. Target: 8 weeks of live blocks, re-evaluated at the §5 interim checks.

**Formula (two-sample comparison of block costs, assuming equal allocation):**

$$n = \frac{2 \sigma^2 (z_{\alpha/2} + z_\beta)^2}{\delta^2}$$

where:
- $\sigma^2$ = observed variance of per-block USD cost (from pre-study ingested
  sessions, estimated by `power.block_costs`)
- $\delta$ = minimum detectable effect (absolute USD per block, set to the
  Layer-0 opportunity ceiling for the tool under test or a fraction thereof)
- $z_{\alpha/2} = 1.96$ (two-sided, $\alpha = 0.05$)
- $z_\beta = 0.84$ ($\text{power} = 0.80$)
- $n$ = blocks per arm; total blocks = $2n$

Blocks-per-week rate and coefficient of variation are reported by
`tokensmash study power`. Final values will be inserted here and the file
re-hashed before M3.

---

## 8. Data Availability

All data is stored in `~/.local/state/tokensmash/study/sessions.jsonl` and
`assignments.jsonl` per schema `tokensmash-session/1` (D3). The scrubbed export
(`tokensmash study export --scrub`) is the shareable artifact:

- Absolute paths dropped; all identifiers are HMAC-keyed hashes (keyed by
  `machine_id` secret, never reversible without the key).
- No prompt text, file content, or tool output text ever enters the store.
- `provider_raw` retains only numeric usage fields from the provider response.
- The export schema is identical to the study schema; releasing raw export files
  is the default posture, not an exception requiring review.

Cross-machine or cross-user merge: concatenation of export files; idempotent
keys `(agent, session_id)` prevent double-counting (D11).

---

## 9. Deviations Log

| # | Date | Description | Impact | Resolution |
|---|------|-------------|--------|------------|
| — | — | — | — | — |

*Any deviation from this protocol must be recorded here, dated, and committed
before the next analysis is run. Post-hoc deviations discovered during analysis
must be recorded with the analysis run's git commit hash.*

---

## Appendix: Key Term Glossary

| Term | Definition |
|---|---|
| block | UTC 2h window; `floor(unix_seconds / 7200)` |
| repo_id | `stable_id("repo", repo_identity(cwd))` — HMAC of canonical repo identity |
| fresh_input | Tokens billed at full input rate (provider-specific translation per D3) |
| canonical usage | The five fields `fresh_input, cache_read, cache_write, output, reasoning_output` |
| pricing_id | Identifier of the versioned pricing data file used to compute `cost_api_usd` |
| ITT | Intention-to-treat; arm assignment, not tool invocation, determines group membership |
| Layer 0 | Observational opportunity ceiling: maximum plausible saving if tool were 100% effective |
| CUPED | Controlled-experiment using pre-experiment data; covariate = pre-study repo mean block cost |
