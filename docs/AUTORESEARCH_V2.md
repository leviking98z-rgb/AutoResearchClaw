# AutoResearch v2

AutoResearch v2 is the production-oriented, pipeline-independent execution
core for continuous multi-Idea research.

## Boundary

v2 **does not import or execute** the legacy 23-stage pipeline, RSI supervisor,
or CodeAgent. It directly reuses the stable parts of AutoResearchClaw:

- the three configured model tiers and local LLM Bridge;
- deterministic code validators;
- the already-owned asynchronous ClusterBridge/Ray pool;
- persistent InfoHub literature memory;
- dashboard styling and the existing FastAPI deployment stack.

The durable source of truth is one SQLite database. Every generated or repaired
project is written to a new immutable attempt directory and is copied to
`current/` only after deterministic validation succeeds.

## Lifecycle

```text
Idea board
  -> DESIGN
  -> BUILD
  -> PILOT
  -> SCALE
  -> REPORT
```

The board is backed by a persistent reservoir. The Controller keeps generating
and screening candidates even while other Ideas are in Build/Pilot/Scale, and
admits the highest-value diverse Ideas whenever an active slot opens.

Any job may end in retry, rejection, informative negative completion, or
quarantine. The Controller continuously refills active slots; completing one
paper never stops the system.

## Directory model

```text
<state_dir>/
├── autoresearch.db
├── events.jsonl
└── ideas/<idea-id>/
    ├── idea.json
    ├── current/                 # latest accepted snapshot
    └── attempts/<job-id>/
        └── attempt-NN/
            ├── attempt.json
            ├── candidate/       # immutable candidate snapshot
            ├── stdout.log
            └── stderr.log
```

Failed, truncated, or partial model output never mutates `current/`.

## Model policy

Only three model clients are constructed:

| Tier | Current model | Use |
|---|---|---|
| Decision | `codebuddy/gpt-5.6-sol` | Idea board and consequential gates |
| Worker | `codebuddy/claude-sonnet-5` | Design, implementation, analysis, report |
| Utility | `codebuddy/claude-haiku-4.5` | High-volume extraction and organization |

The exact models come from `config.rsi.yaml` model tiers. v2 does not use
per-stage model proliferation.

## GPU policy

One `GPUBroker` adopts an already claimed and prepared pool. It never claims,
prepares, or releases physical nodes. Jobs request malleable ranges:

```text
min_gpus <= allocated_gpus <= preferred_gpus <= max_gpus
```

The scheduler applies:

- global available capacity;
- per-Idea fair-share caps;
- short-job/backfill ordering;
- deterministic pool task IDs for idempotent submission;
- concurrent Pilot/Scale tasks from independent Ideas.

Every physical GPU task must write both:

```text
metrics.json
runtime_evidence.json
```

into the absolute shared attempt output directory. A zero return code without
those artifacts is an invalid experiment and can never promote `current/`.

`max_gpu_jobs`, GPU-hour budgets, wall-clock/no-progress limits, exact GPU
accounting, pool utilization, and per-Idea fair share are enforced by the v2
Controller.

## Decision policy

The Worker tier writes plans, code, analysis, and reports. The Decision tier
only returns bounded JSON verdicts for:

- design scientific review;
- Pilot promote/retry/reject/informative-negative;
- Scale promote/retry/reject/informative-negative;
- final claim-evidence review.

The Decision tier is never given a code-editing interface.

## Dashboard

```bash
autoresearch-v2-dashboard \
  -c config.autoresearch-v2.yaml \
  --host 127.0.0.1 \
  --port 8120
```

The dashboard displays the reservoir and five lifecycle lanes, job/attempt
drill-down, stdout/stderr, decision reasons, token/GPU cost, active GPU jobs,
pool utilization, events, and pause/resume/stop controls.

## Run locally in simulation

```bash
cp config.autoresearch-v2.example.yaml config.autoresearch-v2.yaml
# set autoresearch_v2.enabled: true and a writable state_dir

autoresearch-v2 \
  -c config.autoresearch-v2.yaml \
  start \
  --simulation-candidates /path/to/candidates.json \
  --max-ticks 100
```

Inspect:

```bash
autoresearch-v2 -c config.autoresearch-v2.yaml status
autoresearch-v2 -c config.autoresearch-v2.yaml ideas
```

## Current verification

The v2 suite covers:

- SQLite/WAL persistence;
- immutable attempts and atomic promotion;
- high-quality candidate schema and diversity admission;
- multiple Ideas progressing concurrently;
- bounded retry and failure isolation;
- interrupted job recovery;
- fair-share/malleable GPU allocation;
- three independent Ideas filling one six-GPU test pool;
- strict GPU artifact validation and failed-attempt isolation;
- explicit `max_gpu_jobs`;
- persistent reservoir refill/admission;
- dashboard data and controls;
- architecture guard preventing legacy control-plane imports.
