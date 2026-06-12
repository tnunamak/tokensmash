# Tokensmash Crossover Study Protocol

**Protocol version:** <fill: semver, e.g. 0.2.0>  
**Status:** DRAFT — not yet registered.

---

## How to register this protocol

1. Fill every `<fill: …>` marker below with your actual values.
2. Run `tokensmash study power` and paste §7 output into the §7 block.
3. Compute the SHA-256 hash of this file:
   ```bash
   sha256sum PROTOCOL.md
   ```
4. Record the hash in the header of `assignments.jsonl` OR paste it back into
   this file under a `**Registered hash:**` line, then re-hash the amended file
   and record the second hash as the canonical registration hash.
5. Commit the file and hash to version control **before** running
   `tokensmash study init`. Any edit after the first assignment is a registered
   deviation (§9).

---

## 1. Research Questions and Estimand

**Primary question:** Does enabling <fill: tool name> as the default policy
reduce billed-equivalent cost per unit of natural agent work?

**Estimand:** Intention-to-treat (ITT) effect of the policy "tool
default-enabled" on USD-equivalent block cost per repo × 2h-block of natural
agent work, computed from canonical usage with versioned pricing (D7). Whether
the agent invoked the tool within any given block is an outcome, not an
inclusion criterion.

**Co-primary for Codex sessions:** effect on published credit units per block,
computed from Codex per-token-type credit rates (`credit_rate_id` recorded per
session). For Claude Code, subscription quota weighting is opaque; only
API-equivalent USD is claimed (D13).

---

## 2. Design

**Unit of randomization:** repo × UTC 2h block (`floor(unix_seconds / 7200)`,
block index per D4 / `assign.block_index`). Sessions within a block share prompt
cache state; 2h ≥ 2× the longest cache TTL (60 min) makes carryover structural
rather than statistical.

**Arms:** on (tool default-enabled) vs off (tool absent). One tool per study
epoch; arm definitions are fixed in `data/tools/<fill: tool>.json` at study
init.

**Assignment rule:** Deterministic PRF with permuted blocks of 8.
`arm = PRF(study_seed, repo_id, block_index)` where the permutation over each
group of 8 consecutive blocks is a keyed shuffle of `["on","off"]*4`, keyed by
`HMAC(seed, f"{repo_id}:{group}")`, and `group = block // 8`. This guarantees
exact 4/4 balance within every group (D4, `assign.arm_for`).

**Actuation class for <fill: tool name>:** <fill: hook-shim | mcp-overlay | wrapper>

**Subject count:** <fill: number of subjects, e.g. 1>. Cross-user replication
uses the shared export schema (D11).

---

## 3. Outcomes

### Primary
`cost_api_usd` per repo × 2h block, computed from canonical usage fields
(`fresh_input`, `cache_read`, `cache_write`, `output`) with per-model rates from
the versioned pricing file (`pricing_id`).

### Co-primary (Codex only)
`cost_codex_credits` per repo × 2h block, from the versioned credit table
(`credit_rate_id`, per D10).

### Secondary
`fresh_input` tokens per block (canonical field, D3).

### Guardrail outcomes (monitored, not tested)
`user_turns`, `compactions`, session abandonment rate, and `duration_ms` per
block. A statistically significant increase in any guardrail in the on-arm
triggers a mandatory deviation log entry and pauses enrollment pending review.

---

## 4. Eligibility and Exclusions

**Included sessions:** Any agent session linked to the active study epoch whose
block assignment is unambiguous and whose repo is not excluded.

**Excluded from primary analysis (`excluded` field set at ingest):**

| Exclusion rule | `excluded` value | Rationale |
|---|---|---|
| Session spans a block boundary | `"block-boundary"` | Ambiguous arm |
| Resumed session crosses block boundary | `"resumed-cross-block"` | Arm is that of original session |
| Repo is the tokensmash development repo | `"study-repo"` | Measurement confound |
| Superseded rollout file of a resumed Codex session | `"codex-superseded-rollout"` | Cumulative counters would double-count |

**Additional excluded repo IDs for this epoch:**
<fill: list HMAC'd repo_ids to exclude, or "none">

**Model changes during study:** minor version changes recorded as covariate;
major changes trigger a protocol-versioned analysis split and a deviation log
entry.

---

## 5. Analysis Plan

### Estimator

**Primary inferential test:** Paired-block permutation test. For each repo,
blocks are paired on-vs-off within each group of 8 (4 pairs). Test statistic:
mean within-pair difference in block cost. Reference distribution: sign-flip
permutation, 10,000 iterations. Two-sided p-value. This is the pre-committed
primary test.

**Secondary estimator (CUPED):** Adjust each block cost by regression on each
repo's pre-study mean block cost (D7). Reported as a secondary estimate
alongside the permutation-test p-value.

**Cluster-robust SE:** Reported as a robustness check (HC3, repo as cluster).
Not used for primary p-value.

**Alpha:** 0.05, two-sided.

**Analysis implementation:** `tokensmash study analyze` (landing this week),
which reads `sessions.jsonl` + `assignments.jsonl`, applies pre-registered
exclusions, and refuses to mix protocol versions in one analysis run.

### Interim analyses

No unblinded interim tests. Pre-specified power checks permitted without
adjustment:

1. At 25% of target blocks: confirm observed block-cost variance is within 2×
   the pre-study estimate. If exceeded, extend sample size proportionally and
   record as a deviation.
2. At 50% of target blocks: confirm guardrail outcomes show no statistically
   significant degradation (α = 0.01 for guardrails at interim).

### Missing data rules
- Sessions with unknown model: excluded from `cost_api_usd`; included in
  `fresh_input` outcome if usage is present.
- Sessions with `parse_errors > 0` but non-zero usage: included with a flag;
  `provider_raw` retained for audit.
- Blocks with zero sessions in either arm: excluded from the paired test for
  that repo in that group.

---

## 6. Go/No-Go Gate

A tool may enter the live study (M3) only if its Layer-0 opportunity ceiling
(`tool_ceilings.with_rereads_usd` from `tokensmash opportunity`) exceeds the
minimum detectable effect achievable in approximately 8 weeks of natural work,
as reported by `tokensmash study power` (D8).

**Gate outcome for this epoch:** <fill: paste from `tokensmash study power`>

---

## 7. Sample Size

*Fill from `tokensmash study power` before registration.*

**Observed statistics (<fill: weeks> weeks of pre-study sessions, <fill: date>):**
mean block cost $<fill>, SD $<fill> (CV <fill>), <fill> non-empty blocks/week,
block occupancy p = <fill>.

**Gate outcome:** <fill: paste tool-by-tool gate result from power report>

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

---

## 8. Data Availability

All data is stored in `~/.local/state/tokensmash/study/sessions.jsonl` and
`assignments.jsonl` per schema `tokensmash-session/1` (D3). The scrubbed export
(`tokensmash study export -o export.jsonl`) is the shareable artifact:

- Absolute paths dropped; all identifiers are HMAC-keyed hashes (keyed by
  `machine_id` secret, not reversible without the key).
- No prompt text, file content, or tool output text ever enters the store.
- `provider_raw` retains only numeric usage fields from the provider response.
- The export schema is identical to the study schema; releasing raw export files
  is the default posture.

Cross-machine or cross-user merge: concatenation of export files; idempotent
keys `(agent, session_id)` prevent double-counting.

---

## 9. Deviations Log

| # | Date | Description | Impact | Resolution |
|---|------|-------------|--------|------------|
| — | — | — | — | — |

*Any deviation from this protocol must be recorded here, dated, and committed
before the next analysis is run.*

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
