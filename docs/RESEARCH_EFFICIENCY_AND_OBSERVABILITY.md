# Research efficiency and observability

This document describes the production behavior introduced for long-running
ResearchClaw/RSI campaigns.

## 1. Durable literature memory

Stage 4 uses InfoHub as a shared, persistent library:

1. Search the local InfoHub corpus for every Stage-3 query.
2. If the corpus has sufficient coverage, continue without external academic
   API calls.
3. If coverage is below `infohub_min_results`, ask InfoHub to refresh from
   arXiv/Scholar/Bing and persist those results.
4. If ResearchClaw's legacy OpenAlex/Semantic Scholar/arXiv clients discover
   additional papers, upsert those papers into InfoHub as well.
5. If InfoHub is unavailable, continue through the existing providers.

Configuration lives under `literature_search.infohub_*`.

`search_meta.json` records memory hits, refresh counts, newly inserted items,
and errors. This makes API outages visible without making them fatal.

## 2. Incremental Stage 4-8 cache

Stages 4 through 8 use a content-addressed cache. A fingerprint includes:

- hashes of declared input artifacts;
- the relevant research, prompt, role/model, literature and web-search config;
- a cache schema version.

Outputs are copied into a shared cache only after a stage completes, passes
its output contract and PRM/approval/HITL review, and remains in `DONE` state.
A cache hit restores all artifacts and records:

- `stage-NN/cache_restore.json`;
- `stage-NN/stage_health.json.cache`;
- a `cache_hit` decision in `decision.json`;
- a `cache_hit` event in `pipeline_events.jsonl`.

Any input/config/hash mismatch is a cache miss. Topic/prompt/model changes
therefore invalidate only affected stages and their natural downstream
fingerprints rather than forcing unconditional recomputation.

For an RSI campaign, the default shared cache path is:

```text
<campaign>/shared/stage-cache/
```

Set `runtime.stage_cache_enabled: false` to disable it or
`runtime.stage_cache_dir` to use a different shared location.

Stage 4 and Stage 8 also use
`runtime.stage_cache_literature_ttl_hours` (24 hours by default). This prevents
an otherwise identical topic/query from permanently hiding papers newly added
to InfoHub. Set the value to `0` only when a permanently frozen literature
snapshot is intentional.

## 3. Resume from the failed stage across RSI cycles

When a cycle ends as `failed` or `paused`, the next cycle:

1. copies only the prefix covered by `checkpoint.json`;
2. writes `resume_manifest.json`;
3. writes the checkpoint last;
4. launches the normal pipeline with `--resume`.

The failed stage itself is not copied as a completed artifact and is rerun.
Topic pivots and accepted topic refinements do not inherit the old prefix.

This preserves immutable cycle directories for diagnosis while avoiding the old
behavior where Stage 9 failure caused Stages 1-8 to run again.

## 4. Structured logs

Human-readable `pipeline.log` remains available. Machine-readable events are
written to:

```text
<run>/pipeline_events.jsonl
<campaign>/events.jsonl
```

Pipeline events include:

- pipeline start/end and resume point;
- every stage start/end/failure;
- elapsed time, status, decision and artifacts;
- cache hits and cache provenance.

Role-level model calls are written under:

```text
<run>/audit/llm-<role>.jsonl
```

Each row records role/stage/provider, requested and response model, attempted
fallback models, retry/fallback counts, elapsed time, token usage, finish and
truncation status, generation controls, and a SHA-256 request fingerprint.
Prompt bodies, API keys, and chain-of-thought are not stored.

At the end of each pipeline invocation, the raw journals are aggregated into:

```text
<run>/observability_summary.json
```

The summary contains stage count/failures/elapsed time, cache and literature
memory activity, plus per-role LLM call count, latency, token usage, retries,
fallbacks, truncation, errors, and model distribution. It is derived data only;
the append-only JSONL files remain authoritative.

The RSI dashboard can review:

- human-readable Pipeline/Supervisor/experiment logs;
- `pipeline_events.jsonl`;
- the latest or role-specific `audit/llm-<role>.jsonl`; and
- `observability_summary.json`.

Factory mode additionally writes:

```text
<factory>/events.jsonl
<factory>/ideas/<idea_id>/events.jsonl
<factory>/ideas/<idea_id>/operational_events.jsonl
<factory>/observability_summary.json
```

The Idea-local journal records every status transition, Work Item attempt,
resource request, gate decision, lease allocation/release, result, repair, and
exit reason. The Factory dashboard exposes this as an itemized per-Idea
timeline, so a failed or slow Idea can be reconstructed without parsing worker
stdout.

The exact selected-topic contract and generated pipeline config are preserved
per Work Item attempt under
`ideas/<idea_id>/contract/<work_item_id>/attempt-NN/`, so a retrospective can
replay the prompt/config inputs that produced a particular log and result.

`operational_events.jsonl` is the cross-layer replay journal. Factory workers
propagate the same Factory/Idea/Work Item/attempt identifiers into the pipeline,
which records worker launch/exit, pipeline start/end, stage start/end, elapsed
time, outcome, reason code, and artifact names. It intentionally stores
metadata rather than prompt bodies or model reasoning. Obvious credential keys
(`api_key`, authorization, passwords, secrets, and credential-bearing access,
auth, bearer, or refresh token fields) are recursively redacted before any
structured JSONL event is appended. Usage counters such as `total_tokens` and
`completion_tokens` remain visible for cost and efficiency analysis.

The Factory dashboard `/health` response also reports Factory status, tick age,
running Work Item count, and active Lease count. A Factory that still claims
`running` but has not refreshed its state within the configured threshold is
reported as `degraded` with `factory_tick_stale`.

High-frequency JSON snapshots use atomic rename without forcing a CephFS
`fsync` on every tick. Factory event journals are flushed on every append and
remain append-only; LLM audit rows additionally use `fsync` because they are
low-frequency and expensive to reproduce. This keeps retrospective evidence
authoritative at the application level without making filesystem durability
barriers the scheduler bottleneck.

The Factory summary is regenerated from the durable journal and state on every
tick/dashboard refresh. It reports event/failure/gate counts, queue and runtime
p50/p95, Factory tick p50/p95, throughput, screen conversion/terminal yield,
retries, aggregate GPU-hours, LLM calls, and engineering repairs. Candidate
de-duplication is also journaled, rather than silently dropping repeated
generator output. Dashboard event APIs read a bounded tail rather than loading
an unbounded 24-hour JSONL file into memory.

If the aggregation window reaches its configured event cap,
`events.window_truncated` is true. Lifetime counters remain available, but
per-hour rates are withheld rather than dividing lifetime outcomes by a
truncated time window.

Queue wait and runtime metrics are correlated by `(item_id, attempt)`, so a
failed attempt followed by a retry does not merge two queue intervals into one
inflated latency sample. Terminal `work_item_failed` rows carry the structured
`failure_reason`, `profile`, and `kind`. Research yield counts only
`COMPLETED` and `COMPLETED_NEGATIVE`; rejected, parked, and failed Ideas are
reported as exits rather than successful scientific output.

Stage 10 also writes:

```text
<run>/stage-10/scientific_code_alignment.json
```

For an authoritative selected-topic contract, this fail-closed gate verifies
that generated code contains executable paths for the declared real
model/dataset and does not replace the experiment with a synthetic DGP,
simulated trajectory, mock inference, random scientific outcomes, or a
"production will replace this" placeholder. Normal random seeds and legitimate
Monte Carlo methods are not rejected merely for using randomness.

Use the append-only journals for diagnosis and the summaries for fast
comparison. A practical retrospective loop is:

1. Group failures and gate exits by `reason_code`, profile, and Idea family.
2. Compare queue p95, runtime p95, and Factory-tick p95 before/after a change.
3. Rank promoted and rejected Ideas by GPU-hours, LLM calls, and repair count.
4. Open the Idea-local timeline and worker stdout/stderr only for the outliers.
   Use `operational_events.jsonl` first when the failure crosses worker,
   pipeline, and stage boundaries.
5. Change one scheduler, prompt, gate, or budget policy at a time and preserve
   the previous summary as the experiment baseline.

Campaign events include:

- cycle start/completion;
- cross-cycle resume seeding;
- source/target cycle, copied stage prefix and resume stage.

These logs support retrospective metrics such as:

- p50/p95 latency by stage and role;
- cache hit rate and time saved by stage;
- failure rate by stage/error signature;
- number of external literature refreshes and InfoHub reuse rate;
- repeated retries without input changes;
- time from Idea creation to first GPU pilot and evidence gate.
- Idea conversion and early-exit rates by family, tier, and reason;
- GPU-hours spent on promoted, repaired, parked, rejected, and negative-result
  Ideas; and
- queue/lease latency and utilization by Work Item profile.

Do not store API keys, prompts containing secrets, or full model reasoning in
these logs.
