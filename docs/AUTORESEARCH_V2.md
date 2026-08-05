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

This is a **reference-and-infrastructure reuse boundary**, not a wrapper around
the old framework. The reusable adapters can be extracted into standalone
packages later; replacing or deleting the legacy `pipeline` and `rsi`
directories does not change the v2 lifecycle or durable state machine.

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
├── autoresearch.db-wal
├── controller.lock
├── control/
├── events.jsonl
├── llm-audit/
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
| Utility | `codebuddy/claude-haiku-4.5` | Literature landscape extraction and query organization |

The exact models come from `config.rsi.yaml` model tiers. v2 does not use
per-stage model proliferation. InfoHub remains the durable retrieval layer;
the Utility tier extracts a compact landscape for the Idea board, while the
Decision tier retains final admission and consequential gate authority.

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
- bounded tolerance for transient pool probe failures.

Every physical GPU task must write both:

```text
metrics.json
runtime_evidence.json
```

into the absolute shared attempt output directory. A zero return code without
those artifacts is an invalid experiment and can never promote `current/`.
The runtime evidence must agree with the actual GPU allocation and the
preregistered pilot envelope; Scale must increase example or seed coverage.
Build admission also requires an executable model loader, benchmark loader,
artifact writes, and source hashes.

`max_gpu_jobs`, GPU-hour budgets, wall-clock/no-progress limits,
allocated-GPU-time accounting, pool utilization, and per-Idea fair share are
enforced by the v2 Controller.

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

## Run unattended with systemd

Install the checked-in units and one instance environment:

```bash
install -m 0644 deploy/systemd/autoresearch-v2@.service \
  /etc/systemd/system/
install -m 0644 deploy/systemd/autoresearch-v2-dashboard@.service \
  /etc/systemd/system/
install -d -m 0750 /etc/autoresearch-v2
install -m 0640 deploy/systemd/autoresearch-v2-canary.env.example \
  /etc/autoresearch-v2/rsi-canary2.env
systemctl daemon-reload
systemctl enable --now autoresearch-v2@rsi-canary2.service
systemctl enable --now autoresearch-v2-dashboard@rsi-canary2.service
```

The controller unit runs the dependency installer before every start, restarts
after failures, translates `SIGTERM`/`SIGINT` into a cooperative stop, drains
already admitted local model calls, and writes stdout/stderr to journald.
Infrastructure interruptions are recorded but refunded from the scientific
attempt budget. `TimeoutStopSec=15min` allows long in-flight model calls to
finish before systemd escalates. The dashboard uses the controller tick plus
the recorded controller PID for health: a stale lock is reported as
stopped/degraded rather than falsely shown as running.

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

`--max-ticks` is a bounded scheduler probe. It waits only for work already
submitted by the final tick; it does not silently advance every active Idea to
a terminal state.

## Production preflight

- `state_dir` must be inside `gpu.shared_workspace_root`;
- the GPU pool must already be claimed and prepared;
- Bridge and InfoHub must be reachable;
- missing small runtime packages such as `arxiv` are installed by
  `bin/researchclaw-ensure-deps`;
- production GPU-required jobs fail closed when GPU execution is disabled.

## Current verification

The v2 suite covers:

- SQLite/WAL persistence;
- immutable attempts and atomic promotion;
- high-quality candidate schema and diversity admission;
- deterministic designability scoring before active-slot admission;
- phase-aware screening-pilot contracts and finite-sample arithmetic gates;
- Design retries that edit the previous plan against its prior review;
- multiple Ideas progressing concurrently;
- bounded retry and failure isolation;
- graceful stop plus interrupted job recovery without scientific retry loss;
- fair-share/malleable GPU allocation;
- three independent Ideas filling one six-GPU test pool;
- strict GPU artifact validation and failed-attempt isolation;
- explicit `max_gpu_jobs`;
- persistent reservoir refill/admission;
- focused InfoHub refresh and grounded novelty admission;
- preregistration workload/estimand/decision-table checks;
- real model and benchmark implementation checks;
- allocation/runtime and Pilot/Scale contract checks;
- accepted-attempt and interrupted-attempt restart reconciliation;
- transient GPU probe fault tolerance;
- bounded `max_ticks` behavior;
- dashboard data and controls;
- architecture guard preventing legacy control-plane imports.

The suite is necessary but not sufficient evidence for a 24-hour production
soak. A real-model GPU canary and long-running fault-injection run remain
required before declaring the deployment production-complete.
