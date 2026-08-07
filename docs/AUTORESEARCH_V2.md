# AutoResearch v2

AutoResearch v2 is the production-oriented, pipeline-independent execution
core for continuous multi-Idea research.

## Boundary

v2 **does not import or execute** the legacy 23-stage pipeline, RSI supervisor,
or CodeAgent. It directly reuses the stable parts of AutoResearchClaw:

- the three configured model tiers and local LLM Bridge;
- deterministic code validators;
- ClusterBridge/Ray execution, either from a static preclaimed pool or an
  elastic central-resource-manager allocation;
- persistent InfoHub literature memory;
- dashboard styling and the existing FastAPI deployment stack.

This is a **reference-and-infrastructure reuse boundary**, not a wrapper around
the old framework. The reusable adapters can be extracted into standalone
packages later; replacing or deleting the legacy `pipeline` and `rsi`
directories does not change the v2 lifecycle or durable state machine.

The durable source of truth is one SQLite database. In production the hot
database should live on controller-local NVMe, while an atomic SQLite backup is
periodically projected into the shared state directory. Every generated or
repaired project is written to a new immutable attempt directory and is copied
to `current/` only after deterministic validation succeeds.

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
├── autoresearch.db.backup       # atomic shared recovery snapshot (optional)
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
            ├── execution_contract.json
            ├── execution_attestation.json
            ├── stdout.log
            └── stderr.log
```

Failed, truncated, or partial model output never mutates `current/`.

The hot SQLite path is configurable independently:

```yaml
storage:
  database_path: /root/.local/state/autoresearch-v2/my-run/autoresearch.db
  database_backup_path: /root/shared/.clusters/.workdir/autoresearch-v2/runs/my-run/autoresearch.db.backup
  backup_interval_sec: 60
```

Controller, CLI, and Dashboard use the same `database_path`. The Controller
uses SQLite's online backup API on a background thread, writes a temporary
database beside `database_backup_path`, verifies it, and atomically replaces
the previous backup. If the local database is absent at startup, it is restored
from that shared backup before schema initialization. Leaving `storage`
unconfigured preserves the single-file `<state_dir>/autoresearch.db` behavior
used by local development and tests.

## InfoHub Research Memory

AutoResearch projects every Idea into one idempotently updated InfoHub
Research Note:

```text
autoresearch-v2:<system_id>:<idea_id>
```

The note contains the question, hypothesis, literature context, plan, current
Pilot/Scale metrics, decision history, final review, paper Markdown, and a
SHA-256 manifest of accepted artifacts. This makes completed work, negative
results, and in-progress decisions searchable and reusable across later Idea
cycles.

InfoHub is intentionally a **projection**, not the scientific source of truth.
The configured hot database, immutable attempt directories, signed execution
evidence, and `current/` remain canonical. The shared database backup is the
restart/recovery copy. The Controller periodically reconciles all Ideas, uses
an idempotent HTTP upsert, and records sync success/failure as events. InfoHub
downtime never blocks Design, Build, GPU execution, or Report.

```yaml
research_memory:
  enabled: true
  url: http://127.0.0.1:8077
  timeout_sec: 10
  reconcile_interval_ticks: 15
```

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

`gpu.mode` chooses the physical-capacity lifecycle:

- `static_pool` preserves the original behavior. `pool_config` names an
  already claimed and prepared ClusterBridge/Ray pool. AutoResearch adopts it
  but never claims, prepares, or releases its physical nodes.
- `resource_manager` delegates node ownership to the central ClusterBridge
  resource manager. `ResourceManagedGPUManager` requests capacity, renews the
  allocation, materializes a pool from granted nodes, and hot-attaches or
  replaces the `GPUBroker` without restarting the Controller.

The logical scheduler remains independent of either lifecycle. A shared SQLite
lease registry performs atomic cross-Controller reservations, so two
Controllers cannot overbook the same physical GPU pool. Jobs request malleable
ranges:

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
- crash-safe global leases with heartbeat, task-probe recovery, and adoption.
- durable Job metadata records both allocation and pool identity; a task is
  adopted only when the active pool still owns its local metadata.
- ClusterBridge task submission runs in background workers, so a slow control
  plane cannot stall the Controller heartbeat.

### Elastic resource-manager configuration

```yaml
gpu:
  enabled: true
  mode: resource_manager
  # Not used in resource_manager mode. Keep pool_config for static_pool only.
  pool_config: ""
  resource_manager:
    owner: 00000000-0000-0000-0000-000000000000
    cb_command: /root/shared/.clusters/.tools/clusterbridge.sh
    project: AutoResearchClaw-v2
    purpose: AutoResearch v2 continuous multi-Idea experiments
    max_gpus: 64
    duration_min: 1440
    renew_ttl_min: 1440
    renew_interval_sec: 900
    reconcile_interval_sec: 15
    allow_cross_cluster: true
    gpu_type: H20
    priority: normal
    release_on_shutdown: false
    log_root: /root/shared/.clusters/.tmp/autoresearch-v2/elastic-pools
```

The capacity bounds satisfy:

```text
max_gpus > 0
```

Runtime requests equal the current dispatchable durable GPU demand and are
capped by `max_gpus`; the resource manager may grant a whole node even when
the requested card count is smaller. The allocation is returned immediately
after pending and running GPU Jobs both reach zero.

AutoResearch does not manage cluster GPU spin. A granted allocation is treated
as in use throughout cache preparation, Ray startup, and experiment execution.
After release, the cluster owns any automatic spin/idle behavior.

Immutable cache archives may include an adjacent `<archive>.sha256` sidecar.
Pool preparation reads that digest without scanning the full archive on every
allocation and serializes extraction per node with `flock`.

`owner` must be a stable ClusterBridge owner identity for the unattended
service. `project`, `purpose`, `gpu_type`, `priority`, and
`allow_cross_cluster` are passed to the central resource request. `log_root`
holds generated per-allocation pool state and task logs.

`release_on_shutdown: false` is recommended for unattended services. A normal
zero-demand reconciliation still returns the allocation immediately, while a
brief Controller restart preserves the allocation long enough to adopt live
durable GPU tasks instead of interrupting them. Operators may set it to `true`
for intentionally terminal, scale-to-zero deployments.

The public manager contract is intentionally small:

```text
ResourceManagedGPUManager
  properties: broker, configured_capacity
  methods:    bootstrap(), reconcile(), snapshot(), close()
```

`bootstrap()` performs the initial best-effort reconciliation.
`reconcile()` owns the ongoing request/renew/hot-attach lifecycle and is called
from the Controller tick. `snapshot()` returns manager observability state.
`close()` is idempotent and applies the configured release policy. Controller
and dashboard code must not depend on the concrete resource client.

### Elastic lifecycle and failure behavior

At startup, `bootstrap()` performs a best-effort reconciliation:

1. observe any active allocation owned by the configured `owner` and `project`;
2. stay at zero capacity when no durable GPU Job is dispatchable;
3. request rounded demand only when GPU work becomes ready;
4. remain available for Idea generation, Design, and Build while queued;
5. attach the broker and release immediately after durable GPU demand drains.

Every Controller tick calls the manager reconciliation path. A later grant,
allocation replacement, or capacity change therefore updates
`gpu_broker` and `configured_gpu_capacity` in-process. GPU-required jobs stay
`READY` while no broker is attached; infrastructure waiting must not be
misclassified as a scientific failure. Reconciliation and renewal failures are
reported as degraded/unavailable state and retried on later ticks.

The dashboard snapshot includes the normal broker metrics plus an `elastic`
object with manager state, requested bounds, allocation ID, granted capacity,
nodes, queue state, and last error.

Every physical GPU task must write both:

```text
metrics.json
runtime_evidence.json
```

into the absolute shared attempt output directory. A zero return code without
those artifacts is an invalid experiment and can never promote `current/`.
The runtime evidence must agree with the actual GPU allocation and the
preregistered pilot envelope. Scale must increase **both** examples and
independent seed coverage and must use a distinct, untouched, preregistered
confirmatory split. Build admission also requires an executable model loader,
benchmark loader, artifact writes, source hashes, and a successful direct-argv
smoke execution.

When GPU execution is enabled, the dependency-bearing smoke run is a separate
one-GPU job in the same ClusterBridge/Ray environment as Pilot. It is recorded
as a durable Build sub-attempt, and Pilot is not scheduled until its signed
attestation succeeds. `execution.smoke_environment` may be `auto`
(recommended), `gpu_pool`, or `local`; `auto` selects the GPU pool when GPU
execution is enabled and local execution otherwise.

For offline pools, the immutable cache is also a scheduling contract. Configure
`execution.available_models` and `execution.available_datasets` with canonical
identifiers that are physically staged in `gpu_cache_dir`. The Idea board and
typed Design compiler may select only those identifiers; a deterministic
offline cache miss is returned to Build rather than retried forever as transient
infrastructure.

Generated shell programs are not accepted. Build commands are normalized into
direct Python `argv` and checked against an allowlist. Before Pilot/Scale, the
Controller writes `execution_contract.json` containing the exact argv, bound
plan/build hashes, paths, resource limits, and allowed environment keys. After
execution it writes a controller-side HMAC-signed
`execution_attestation.json` over the contract, return code, allocated GPUs,
stdout/stderr hashes, and the complete artifact manifest. Generated code cannot
self-sign this evidence because the key never enters its environment.

`max_gpu_jobs`, `max_gpu_submissions_per_tick`, GPU-hour budgets,
wall-clock/no-progress limits, allocated-GPU-time accounting, pool utilization,
and per-Idea fair share are enforced by the v2 Controller. Detached GPU result
probes run asynchronously with a transport cooldown; a slow ClusterBridge probe
does not block the Controller heartbeat and is not reported as a scientific
failure.

## Decision policy

The Worker tier writes plans, code, analysis, and reports. The Decision tier
only returns bounded JSON verdicts for:

- design scientific review;
- Pilot promote/retry/reject/informative-negative;
- Scale promote/retry/reject/informative-negative;
- final claim-evidence review.

The Decision tier is never given a code-editing interface.

Design uses one immutable Job attempt with a bounded Worker/Decision repair
loop. A Decision `retry` is fed directly back to the Worker with the current
compiled plan and exact required changes; the Worker may revise the typed
scientific fields up to `budgets.max_design_revisions` times. A hard scientific
`reject` exits immediately. This avoids spending several full Job attempts
rediscovering the same protocol defects while preserving every revision in the
attempt audit record.

Design uses a typed protocol compiler rather than reconciling several
free-form rules after generation. The raw scientific endpoint and the
promotion statistic are separate:

```json
{
  "primary_metric": "selection regret",
  "metric_direction": "minimize",
  "gate_statistic": {
    "name": "relative_regret_reduction",
    "direction": "maximize",
    "threshold": {"value": 0.2, "scale": "proportion"},
    "undefined_policy": "reject"
  }
}
```

The model supplies a small typed conjunction of operational
`validity_criteria` and scientific `promotion_criteria`. The Controller
compiles exactly three exhaustive outcomes:

```text
invalid evidence                         -> retry
valid + every promotion criterion passes -> promote
valid otherwise                          -> reject
```

Thus an undefined denominator, flat outcome, low event rate, or confidence
interval crossing a boundary cannot be retried until favorable; it is a valid
futility/rejection result unless an independently preregistered operational
validity criterion failed.

Dataset access is also typed. `input_access`, `within_episode_feedback`,
`cross_example_adaptation`, `hidden_labels_for_tuning`, and
`threshold_tuning` are independent controls. This permits an online-memory
experiment to consume its own prior within-episode outcome without incorrectly
claiming that the entire screening split is non-adaptive. Confirmatory inputs
may be presented exactly once at Scale, but their labels/assertions remain
unavailable for tuning, selection, calibration, memory, or threshold changes.

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
after failures, and writes stdout/stderr to journald. `SIGTERM` is deliberately
lock-free and immediate: already admitted model calls remain durable as
RUNNING, and startup recovery records and refunds the interruption without
consuming the scientific attempt budget. `TimeoutStopSec=10s` is a last-resort
systemd bound rather than a model-call drain window. `SIGINT` remains a
cooperative foreground stop. The dashboard uses the controller tick plus the
recorded controller PID for health: a stale lock is reported as
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
- choose exactly one GPU lifecycle:
  - `static_pool`: `pool_config` must name an already claimed/prepared pool;
  - `resource_manager`: `owner`, request metadata, valid capacity bounds,
    positive lease intervals, and a writable `log_root` are required;
- the configured `cb_command` must reach the central resource manager, and the
  service owner must be authorized to request, renew, inspect, and optionally
  release its own allocation;
- Bridge and InfoHub must be reachable;
- missing small runtime packages such as `arxiv` are installed by
  `bin/researchclaw-ensure-deps`;
- production GPU-required jobs fail closed when GPU execution is disabled.
- `execution.python_executable` must exist on the GPU nodes;
- `execution.smoke_environment: auto` prevents Controller-host dependency
  gaps from being misclassified as scientific Build failures;
- `execution.attestation_key_file` should reside outside Idea candidates and
  be readable only by the Controller.

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
- direct Build smoke execution with shell-command rejection;
- controller-issued execution contracts and signed artifact attestations;
- cross-Controller global GPU lease accounting;
- elastic GPU request, queued startup, hot attachment, capacity replacement,
  lease renewal, and configurable shutdown release;
- allocation/runtime and Pilot/Scale contract checks;
- strict Scale expansion and untouched confirmatory-split checks;
- accepted-attempt and interrupted-attempt restart reconciliation;
- transient GPU probe fault tolerance;
- non-blocking GPU probes and submissions, allocation-aware adoption, and safe
  orphan leases;
- bounded `max_ticks` behavior;
- dashboard data and controls;
- architecture guard preventing legacy control-plane imports.

The suite is necessary but not sufficient evidence for a 24-hour production
soak. A real-model GPU canary and long-running fault-injection run remain
required before declaring the deployment production-complete.
