# Continuous Multi-Idea Research Factory

**Status:** Proposed architecture and implementation roadmap

**Scope:** An optional, backward-compatible execution mode for AutoResearchClaw

**Primary goal:** Keep generating, screening, validating, and completing multiple
research ideas continuously while sharing one GPU pool efficiently.

## 1. Motivation

The current AutoResearchClaw execution model is optimized for one selected topic:

```text
candidate generation -> select one topic -> run one 23-stage pipeline
```

That remains useful for focused research, but it leaves two opportunities:

1. only one idea is scientifically validated at a time; and
2. GPU capacity can be idle while the active pipeline is using the LLM, reading
   literature, writing code, or analyzing results.

The Research Factory adds a steady-state portfolio mode:

```text
generate ideas continuously
    -> admit several independent ideas
    -> decompose them into work items
    -> share one GPU pool
    -> stop weak ideas early
    -> keep or publish every idea that independently passes its evidence gates
    -> immediately refill released idea slots
```

There is no global winner and no global "paper completed, therefore stop"
condition. Several ideas may progress or produce papers concurrently.

## 2. Design principles

1. **One physical pool owner.** Idea workers never claim ClusterBridge nodes
   directly. One broker owns the allocation, lease renewal, Ray lifecycle, and
   recovery.
2. **Many independent ideas.** Each idea has its own hypothesis, workspace,
   budget, state, evidence, role sessions, and terminal outcome.
3. **Schedule work items, not monolithic pipelines.** Conditions, seeds,
   baselines, ablations, analyses, and writing jobs are explicit units of work.
4. **Absolute scientific gates.** Ideas progress when they meet predefined
   validity and evidence requirements, not only because they rank above another
   idea.
5. **Early exit by default.** Most ideas should stop before expensive validation
   or paper writing.
6. **Negative results are first-class outcomes.** A rigorous negative result may
   enter paper production; an invalid or uninformative experiment may not.
7. **Durable and restartable.** All decisions, budgets, leases, and work-item
   transitions are persisted and reconciled after restart.
8. **Backward compatible.** The current single-topic RSI campaign remains
   available. Factory mode is enabled explicitly.
9. **No automatic external publication.** Paper packages remain local and
   require human review before any submission or release.

## 3. System architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│                   Research Factory Control Plane                 │
│                                                                  │
│  Continuous Idea Generator                                      │
│          │                                                       │
│          ▼                                                       │
│  Candidate Reservoir ──> Admission + Deduplication               │
│          ▲                         │                              │
│          │                         ▼                              │
│  literature / results /     Idea Actor Manager                   │
│  failures / crossovers       ├── Idea A                          │
│                              ├── Idea B                          │
│                              ├── Idea C                          │
│                              └── ...                             │
│                                     │                            │
│                                     ▼                            │
│                              Work-Item DAGs                      │
│                                     │                            │
│                                     ▼                            │
│  Evidence Gates <────────── Global Scheduler ───────> LLM Queue  │
│                                     │                            │
└─────────────────────────────────────┼────────────────────────────┘
                                      ▼
                              GPU Resource Broker
                                      │
                                      ▼
                          Shared ClusterBridge / Ray Pool
                                  (for example 32 H20s)
```

The first implementation should be a **modular monolith with isolated worker
processes**, not a fleet of independent services:

```text
researchclaw-factory.service
├── generator loop
├── admission loop
├── scheduling loop
├── evidence-gate loop
├── GPU broker
└── worker manager
    ├── transient worker for idea-000001
    ├── transient worker for idea-000002
    └── ...

researchclaw-factory-monitor.service
researchclaw-factory-dashboard.service
```

This keeps deployment and recovery simple while isolating failures in individual
idea workers.

## 4. Core domain model

### 4.1 Factory

The Factory is the global single writer for portfolio state. It:

- maintains candidate and active-population watermarks;
- generates and admits new ideas;
- owns global LLM and GPU concurrency limits;
- starts and reconciles idea workers;
- applies evidence-gate decisions;
- archives completed, rejected, parked, and failed ideas; and
- continues running until an operator pauses or stops it.

Completing one paper never completes the Factory.

### 4.2 Idea

An Idea is an independently evaluated research project.

```json
{
  "idea_id": "idea-000127",
  "title": "Example research question",
  "family": "verifier-gating",
  "source": "negative_result",
  "parent_ids": ["idea-000043"],
  "status": "pilot_running",
  "primary_metric": "held-out accuracy",
  "budget_tier": "pilot",
  "priority": 0.73
}
```

Each Idea owns:

- a concrete falsifiable hypothesis;
- a preregistered primary metric and early-stopping boundaries;
- a code and artifact workspace;
- role-scoped LLM sessions;
- a work-item DAG;
- a resource budget and usage ledger;
- evidence, gate decisions, and provenance;
- lineage to parent, child, mutation, and crossover ideas; and
- one terminal or suspended outcome.

### 4.3 Work item

A Work Item is the smallest schedulable and retryable unit.

Examples:

- `literature-query-03`
- `novelty-screen`
- `experiment-blueprint`
- `code-smoke-test`
- `baseline-seed-0`
- `proposed-seed-2`
- `no-memory-ablation`
- `pilot-analysis`
- `paper-methods-draft`

```yaml
item_id: idea-000127-proposed-seed-2
idea_id: idea-000127
kind: gpu_experiment
dependencies:
  - idea-000127-code-validation
resources:
  min_gpus: 1
  preferred_gpus: 2
  max_gpus: 4
  cpus: 8
  timeout_sec: 3600
placement: single_node
preemptible: true
checkpointable: true
attempt_limit: 2
```

Work-item states:

```text
PENDING -> READY -> QUEUED -> ADMITTED -> RUNNING
        -> SUCCEEDED | FAILED | CANCELLED | TIMED_OUT

FAILED -> RETRY_WAIT -> READY
```

Transitions must be idempotent and append an audit event.

### 4.4 Resource request and lease

An Idea requests a logical lease; it never claims physical nodes:

```text
REQUESTED -> QUEUED -> ADMITTED -> RUNNING -> RELEASED
                                  └────────> EXPIRED
```

```json
{
  "idea_id": "idea-000127",
  "item_id": "idea-000127-proposed-seed-2",
  "min_gpus": 1,
  "preferred_gpus": 2,
  "max_gpus": 4,
  "estimated_seconds": 3600,
  "priority": 70,
  "preemptible": true
}
```

### 4.5 Gate decision

Every budget increase is an explicit, auditable decision:

```json
{
  "decision": "PROMOTE",
  "reason_code": "PILOT_SIGNAL_VALID",
  "current_tier": "pilot",
  "next_tier": "validation",
  "evidence_refs": [
    "results/baseline-seed-0.json",
    "analysis/pilot-summary.json"
  ],
  "decided_at": "2026-08-04T12:00:00Z"
}
```

Allowed decisions:

- `CONTINUE`
- `PROMOTE`
- `REPAIR`
- `PARK`
- `REJECT`
- `COMPLETE_NEGATIVE`
- `COMPLETE`

## 5. Continuous idea supply

The generator is a persistent loop, independent of an individual pipeline or
paper.

### 5.1 Generation triggers

- **Low watermark:** generate a new batch when the reservoir falls below its
  configured minimum.
- **Periodic refresh:** scan current literature and open-source developments on
  a configurable interval.
- **Experimental event:** derive explanatory or boundary-condition ideas from
  anomalous, positive, or negative results.
- **Mutation:** produce a bounded variation of an active or completed idea.
- **Fork:** branch a promising mechanism into a distinct falsifiable question.
- **Crossover:** combine complementary mechanisms from different idea families.

Idea source values:

```text
de_novo
literature_gap
mutation
fork
negative_result
crossover
```

### 5.2 Admission controls

Before an Idea consumes an active slot, admission checks:

- semantic duplication against candidates, active ideas, and archive;
- maximum active ideas in the same mechanism family;
- novelty and closest prior work;
- falsifiability and a predeclared primary endpoint;
- availability and licensing of data and models;
- presence of a no-self-improvement or equivalent control;
- existence of a cheap discriminating pilot;
- maximum pilot cost; and
- scientific value if the hypothesis is false.

The admission model may recommend a decision, but deterministic validation must
reject malformed or policy-violating proposals.

## 6. Steady-state population

The server maintains watermarks instead of running fixed generations:

```yaml
factory:
  enabled: true

  reservoir:
    low_watermark: 12
    target_size: 24
    generation_batch_size: 6

  population:
    max_active_ideas: 10
    max_screening_ideas: 4
    max_pilot_ideas: 6
    max_validation_ideas: 3
    max_paper_ideas: 2
    max_same_family_active: 2
```

When an Idea leaves an active slot:

```text
release resources
    -> archive its state and evidence
    -> admit the next eligible candidate
    -> generate more candidates if the reservoir is below its watermark
```

These are concurrency limits, not static GPU partitions. Paper and literature
work may coexist with GPU experiments.

## 7. Idea lifecycle and early exit

```text
CANDIDATE
  -> DESK_SCREENING
     -> REJECTED
     -> ADMITTED
        -> BUILDING
           -> SMOKE_TEST
              -> REPAIR
              -> ENGINEERING_INFEASIBLE
              -> PILOT
                 -> PILOT_EVALUATION
                    -> REJECTED
                    -> PARKED
                    -> NEGATIVE_RESULT
                    -> REPAIR
                    -> VALIDATION
                       -> SCALE_UP
                          -> REJECTED
                          -> NEGATIVE_RESULT
                          -> PAPER
                             -> COMPLETED
```

### 7.1 Budget tiers

Each tier has a hard budget. An Idea must pass the current gate before receiving
the next allocation.

| Tier | Typical budget | Purpose |
| --- | ---: | --- |
| Desk screening | 0 GPU | Deduplication, novelty, feasibility, licensing |
| Smoke test | 0.1-0.5 GPU-hour | Prove that code, data, and metrics execute |
| Cheap pilot | 1-4 GPU-hours | Detect a primary signal or informative null |
| Validation | 8-40 GPU-hours | Baselines, multiple seeds, uncertainty |
| Scale-up | Explicit gate | Robustness, transfer, and full ablations |
| Paper | Primarily LLM | Only for valid positive or negative evidence |

Example:

```yaml
budgets:
  desk_llm_calls: 12
  smoke_gpu_hours: 0.5
  pilot_gpu_hours: 4
  validation_gpu_hours: 32
  max_wall_clock_hours: 72
  max_engineering_repairs: 2
  max_no_progress_rounds: 2
```

### 7.2 Early-stopping policy

Pilot and validation plans preregister their stopping rules:

```yaml
early_stopping:
  minimum_seeds: 3
  minimum_effect_size: 0.03
  success_probability: 0.95
  futility_probability: 0.95
  maximum_gpu_hours: 12
```

Examples:

- cancel remaining seeds when futility is established;
- promote when the primary signal and validity gates are satisfied;
- repair, rather than reject, when the failure is an implementation defect;
- reject after the repair budget is exhausted and no valid minimal experiment
  can be produced; and
- preserve a rigorous, informative null as `COMPLETE_NEGATIVE`.

Stopping boundaries must be declared before observing the corresponding results
to avoid opportunistic stopping.

### 7.3 Structured exit reasons

```text
DUPLICATE
NOVELTY_INVALIDATED
NOT_FALSIFIABLE
DATA_UNAVAILABLE
LICENSE_BLOCKED
COMPUTE_INFEASIBLE
ENGINEERING_INFEASIBLE
LEAKAGE_DETECTED
NO_PRIMARY_SIGNAL
INSUFFICIENT_INFORMATION_GAIN
SAFETY_FAILURE
NEGATIVE_BUT_INFORMATIVE
```

All rejected or parked ideas retain their evidence, cost, and rationale. This
history is input to future generation and deduplication.

## 8. Relationship to the existing 23-stage pipeline

The current pipeline remains the reusable per-Idea workflow engine. It is no
longer the single global pipeline.

Suggested execution profiles:

| Profile | Existing stages | Factory purpose |
| --- | --- | --- |
| Scientific screen | 1-8 | Literature, synthesis, hypotheses, novelty |
| Build | 9-11 | Design, code, and resource plan |
| Smoke | bounded execution after Stage 10 | Verify executable minimum |
| Pilot / validation | 12-15 | Run, refine, analyze, decide |
| Paper | 16-23 | Write, review, quality gate, citation verification |

At any moment:

```text
Idea A: Stage 04, literature collection
Idea B: Stage 10, code generation
Idea C: Stage 12, pilot experiments
Idea D: Stage 14, result analysis
Idea E: Stage 18, peer review
```

In Factory mode:

- Stage 15 is an advisory scientific decision and evidence producer;
- the Factory Evidence Gate controls budget promotion and paper admission; and
- exhausted retries must never force an invalid experiment into paper writing.

The existing single-Idea behavior remains unchanged when Factory mode is
disabled.

## 9. Shared GPU pool

### 9.1 Single ownership boundary

Only the GPU Broker may:

- request and renew the ClusterBridge allocation;
- prepare or recover the Ray cluster;
- inspect global resources;
- submit, probe, collect, and cancel GPU tasks; and
- release physical resources.

Idea workers submit logical requests to the Broker. This prevents competing
claims, duplicate Ray lifecycles, and one Idea shutting down another Idea's
cluster.

### 9.2 Scheduler policy

The initial scheduler should combine:

1. hard resource and placement constraints;
2. weighted fairness between Ideas;
3. per-Idea GPU-share caps;
4. expected information gain per estimated GPU-hour;
5. queue-age compensation;
6. short-job backfilling; and
7. preemption only for explicitly checkpointable tasks.

```yaml
scheduler:
  llm_slots: 2
  gpu_target_utilization: 0.90
  reserved_gpus: 2
  max_gpu_share_per_idea: 0.50
  pilot_max_gpus_per_idea: 4
  validation_max_gpus_per_idea: 8
  backfill_enabled: true
  checkpoint_preemption: false
```

Checkpoint preemption should remain disabled until restart correctness is
demonstrated. Task-boundary scheduling and backfill are sufficient for the
first production version.

### 9.3 Ray execution

Single-task requests use Ray resource annotations. Gang-scheduled multi-GPU
jobs use placement groups.

```python
@ray.remote(num_gpus=2, num_cpus=8)
def execute_work_item(spec: dict) -> dict:
    ...
```

The existing blocking pool API should evolve from:

```python
run_task(...)
```

to a durable asynchronous interface:

```python
submit_task(...)
probe_task(...)
collect_task(...)
cancel_task(...)
```

Task IDs must be deterministic per Idea, Work Item, and attempt so recovery
does not duplicate expensive experiments.

## 10. LLM scheduling and roles

Each Idea continues to use role-aware routing:

- topic selector
- research director
- literature researcher
- idea scientist
- experiment designer
- coding engineer
- compute operator
- result analyst
- paper writer
- skeptical reviewer
- citation auditor

The session identity becomes:

```text
idea_id / role / phase-or-round
```

Examples:

```text
idea-000127/coding_engineer/pilot-1
idea-000127/result_analyst/pilot-1
idea-000203/literature_researcher/screen-1
```

A global LLM queue enforces backend concurrency and rate limits. Different
roles may use different providers or models; an Idea cannot bypass the global
admission limit.

## 11. Persistence and recovery

Suggested durable layout:

```text
research-factory/
├── factory.json
├── state.json
├── events.jsonl
├── reservoir/
│   └── candidates.jsonl
├── scheduler/
│   ├── queue.jsonl
│   ├── leases.json
│   └── snapshot.json
├── ideas/
│   └── idea-000127/
│       ├── idea.json
│       ├── state.json
│       ├── budget.json
│       ├── lineage.json
│       ├── work_items.jsonl
│       ├── events.jsonl
│       ├── evidence/
│       ├── workspace/
│       └── runs/
└── shared-cache/
    ├── literature/
    ├── datasets/
    └── models/
```

Recovery rules:

1. The Factory is the single writer of global state.
2. An Idea worker writes only under its Idea directory.
3. Snapshots use atomic replacement; event logs are append-only.
4. Work-item transitions are idempotent.
5. On restart, reconcile persisted leases with Ray and pool task state.
6. Adopt a matching live task or cancel an unknown orphan.
7. Release expired leases and return safe tasks to `READY`.
8. Failure in one Idea never stops unrelated Ideas.
9. Pause and shutdown stop admission first, then drain or checkpoint workers.

## 12. Dashboard

The Factory dashboard is a control-plane view above the existing single-Idea
dashboard.

Primary Kanban columns:

```text
Reservoir | Screening | Build | Pilot | Validation | Paper | Completed | Rejected
```

Each Idea card should show:

- title, family, and lineage;
- current stage and gate;
- primary metric and evidence validity;
- GPU-hours, LLM calls, and wall-clock budget;
- active and queued Work Items;
- retries and repair budget;
- priority and current GPU allocation; and
- promotion, parking, or rejection reason.

Additional views:

- all 32 GPUs and their current Idea / Work Item assignment;
- global GPU utilization and queue depth;
- Idea generation, admission, rejection, and completion rates;
- lineage graph;
- evidence and budget audit trail; and
- drill-down into the existing 23-stage Idea detail page.

The dashboard is not the source of truth. It reads durable Factory, scheduler,
pool, and Idea state.

## 13. Proposed module boundaries

```text
researchclaw/factory/
├── models.py          # Idea, WorkItem, Lease, GateDecision
├── store.py           # atomic snapshots and append-only events
├── generator.py       # continuous idea generation
├── admission.py       # validation, deduplication, family quotas
├── orchestrator.py    # steady-state control loop
├── actor.py           # per-Idea state machine and pipeline adapter
├── scheduler.py       # global LLM/CPU/GPU queues
├── gpu_broker.py      # sole ClusterBridge/Ray pool owner
├── gates.py           # promotion, early exit, paper admission
├── budgets.py         # resource ledger and quota enforcement
├── recovery.py        # restart reconciliation
└── dashboard.py       # Factory API and UI
```

Reuse:

- `researchclaw/rsi/topic_selection.py` for structured proposal generation;
- `researchclaw/llm/roles.py` for role-aware routing;
- `researchclaw/pipeline/runner.py` as the per-Idea workflow engine;
- `researchclaw/rsi/evidence.py` for claim and evidence extraction;
- `researchclaw/experiment/clusterbridge_pool.py` for physical pool lifecycle;
  and
- the current RSI dashboard for per-Idea details.

## 14. Roadmap

The roadmap deliberately separates orchestration correctness from aggressive GPU
utilization. Each milestone must be independently usable and reversible.

### Phase 0 — Architecture contract and safety invariants

**Goal:** Freeze schemas, ownership boundaries, and compatibility expectations.

Deliverables:

- versioned schemas for Idea, Work Item, Lease, Budget, and Gate Decision;
- Factory configuration section with `enabled: false` by default;
- explicit single-pool-owner invariant;
- no-publication and evidence-validity invariants;
- deterministic ID and event semantics; and
- a migration rule for importing an existing RSI topic as an Idea.

Exit criteria:

- schema tests and state-transition property tests pass;
- current single-Idea tests are unchanged and passing; and
- no runtime behavior changes when Factory mode is disabled.

### Phase 1 — Durable Idea registry and steady-state reservoir

**Goal:** Continuously create, deduplicate, admit, and archive Ideas without GPU
execution.

Deliverables:

- `factory.models`, `factory.store`, `factory.generator`, and
  `factory.admission`;
- candidate low/target watermarks;
- family quotas and semantic duplicate records;
- Idea lineage and structured archive outcomes;
- CLI commands for start, status, pause, resume, and list; and
- deterministic fake-LLM integration tests.

Exit criteria:

- a 24-hour simulated run keeps reservoir and active-slot watermarks;
- restart produces no duplicate Idea IDs or admissions;
- rejected Ideas are retained with reason and provenance; and
- generation failures back off without stopping the Factory.

### Phase 2 — Multi-Idea CPU/LLM execution

**Goal:** Run several Idea workflows concurrently through screening and build,
without sharing GPUs yet.

Deliverables:

- Idea Actor Manager and isolated worker processes;
- per-Idea pipeline profiles and work-item DAGs;
- global LLM semaphore and rate-aware queue;
- role session isolation by `idea_id / role / round`;
- bounded retries, cancellation, and failure isolation; and
- conversion of pipeline artifacts into Work Item and Evidence records.

Exit criteria:

- at least six Ideas can progress concurrently in a local test;
- one worker crash does not stop or corrupt other Ideas;
- pause/resume and process restart recover every Idea deterministically; and
- no Idea enters GPU Pilot without passing screen/build gates.

### Phase 3 — Asynchronous pool API and GPU Broker

**Goal:** Share one prepared ClusterBridge/Ray pool among independent Idea Work
Items.

Deliverables:

- `submit_task`, `probe_task`, `collect_task`, and idempotent `cancel_task`;
- logical lease state machine and durable task IDs;
- single pool-owner Broker;
- Ray resource admission and per-Idea share caps;
- queue aging and short-job backfill;
- orphan-task reconciliation; and
- GPU assignment and usage accounting events.

Exit criteria:

- multiple Ideas run independent GPU tasks concurrently in one pool;
- no Idea can claim, release, or reinitialize the physical pool;
- Broker restart adopts or safely cancels every live task;
- cancelled or failed tasks release resources;
- observed GPU assignment never exceeds declared limits; and
- a sustained mixed-workload test reaches the configured utilization target
  when runnable work exists.

### Phase 4 — Experimental matrix Itemization

**Goal:** Decompose experiments into independently schedulable conditions,
seeds, baselines, and ablations.

Deliverables:

- standard experiment-manifest schema;
- compiler from Stage 9/10 artifacts into a Work Item DAG;
- dataset/model preparation dependencies;
- seed, condition, baseline, and ablation task templates;
- result aggregation with provenance and missing-item detection; and
- task-boundary retry and backfill.

Exit criteria:

- a multi-condition, multi-seed experiment fills available GPUs without
  duplicate runs;
- aggregation refuses missing, stale, or invalid results;
- rerunning the scheduler is idempotent; and
- all numerical claims trace to concrete completed Work Items.

### Phase 5 — Evidence gates, budgets, and asynchronous early exit

**Goal:** Spend additional compute only when an Idea earns the next budget tier.

Deliverables:

- desk, smoke, pilot, validation, scale, and paper gates;
- resource and repair budgets;
- preregistered success and futility rules;
- `PROMOTE`, `REPAIR`, `PARK`, `REJECT`, and negative-result outcomes;
- cancellation of unnecessary queued/running Work Items; and
- replacement admission after an Idea exits an active slot.

Exit criteria:

- invalid experiments cannot enter paper stages;
- engineering failures receive bounded repair rather than scientific rejection;
- futility decisions cancel remaining optional work;
- valid negative results remain eligible for papers;
- budget overruns are impossible without an auditable override; and
- exited slots are automatically refilled.

### Phase 6 — Factory dashboard and operations

**Goal:** Make the continuous portfolio observable and safely operable.

Deliverables:

- Factory Kanban and Idea drill-down;
- append-only global and per-Idea itemized event journals;
- GPU map, queues, utilization, budgets, and leases;
- lineage and gate-decision views;
- cooperative per-Idea pause, park, resume, and reject controls;
- Factory monitor, alerts, and systemd deployment; and
- operational runbook and backup/restore procedure.

Exit criteria:

- every displayed status is derived from durable state;
- every Idea and Work Item transition can be replayed from its local timeline;
- stale heartbeats and orphan tasks are visible;
- control operations are idempotent and auditable;
- a restart drill restores Factory, Broker, workers, and dashboard; and
- a 24-hour unattended soak test completes without global deadlock.

### Phase 7 — Adaptive scheduling and idea evolution

**Goal:** Improve throughput and scientific yield after the deterministic core is
proven.

Deliverables:

- information-gain-per-GPU-hour priority updates;
- online duration and resource estimation;
- safe checkpoint-boundary preemption;
- result-triggered mutation, fork, and crossover;
- shared-cache provenance and cache hit accounting; and
- diversity-aware population control.

Exit criteria:

- adaptive scheduling improves throughput over FIFO in replay tests;
- preemption never loses an accepted checkpoint or duplicates a result;
- derived Ideas preserve lineage and evidence provenance; and
- diversity constraints prevent one mechanism family from monopolizing the
  active population.

## 15. Production acceptance criteria

Factory mode is production-ready only when all of the following hold:

1. one physical GPU pool has exactly one lifecycle owner;
2. at least six independent Ideas can be active without state contamination;
3. a worker, Broker, or Factory restart is recoverable without duplicate
   experiments;
4. every GPU task has an Idea, Work Item, lease, budget, and provenance record;
5. invalid evidence cannot reach paper generation;
6. early exit reliably frees queued and running resources;
7. candidate watermarks refill after completion or rejection;
8. an informative negative result can complete independently;
9. the current single-Idea mode remains supported; and
10. a 24-hour unattended run sustains progress, recovers expected failures, and
    requires no manual lease or process repair.

## 16. Initial implementation defaults

Conservative defaults for the first 32-GPU deployment:

```yaml
factory:
  enabled: false

  reservoir:
    low_watermark: 12
    target_size: 24
    generation_batch_size: 6

  population:
    max_active_ideas: 10
    max_screening_ideas: 4
    max_pilot_ideas: 6
    max_validation_ideas: 3
    max_paper_ideas: 2
    max_same_family_active: 2

  scheduler:
    llm_slots: 2
    gpu_target_utilization: 0.90
    reserved_gpus: 2
    pilot_max_gpus_per_idea: 4
    validation_max_gpus_per_idea: 8
    max_gpu_share_per_idea: 0.50
    backfill_enabled: true
    checkpoint_preemption: false

  budgets:
    smoke_gpu_hours: 0.5
    pilot_gpu_hours: 4
    validation_gpu_hours: 32
    max_engineering_repairs: 2
    max_no_progress_rounds: 2
```

These values are policy defaults, not hard-coded assumptions. They should be
tuned from observed duration, utilization, queue, and evidence-quality data.
