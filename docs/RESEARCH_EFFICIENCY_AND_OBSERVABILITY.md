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

Outputs are copied into a shared cache only after a stage completes and passes
its output contract. A cache hit restores all artifacts and records:

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

Do not store API keys, prompts containing secrets, or full model reasoning in
these logs.
