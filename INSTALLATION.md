# End-to-End AutoResearch / RSI Pipeline

This checkout is the first end-to-end AutoResearch pipeline for the current
machine.

## Installed components

- AutoResearchClaw 0.5.0, pinned by the current Git checkout.
- Python 3.11 virtual environment at `.venv/`.
- Local Bridge Server at `http://127.0.0.1:8787/v1`.
- CodeBuddy model pinned to `codebuddy/deepseek-v4-pro-ioa`.
- Persistent RSI supervisor, monitor, pause/stop/resume controls, evidence
  scorecards, and cross-cycle A-Evolve memory.
- Background `rsi-submit` automatically starts the independent monitor with a
  durable `rsi-resume` restart command.
- The detached monitor runs under a lightweight watchdog, so a monitor process
  crash is restarted independently of the campaign supervisor.
- ClusterBridge/Ray execution on 4 claimed H20 nodes (32 GPUs total).
- LaTeX is available through the host `pdflatex`.

## Safety defaults

- RSI cycles use full-auto pipeline execution but keep automatic publication
  and submission disabled.
- Generated GPU experiments execute through ClusterBridge; SSH is never used.
- `config.rsi.yaml` submits experiment code to the prepared 4-node Ray pool.
- The pool lifecycle remains external to a paper cycle so one failed experiment
  cannot silently claim, release, or clean another user's nodes.
- Pool state restoration, renewal, release, and monitoring fail closed unless
  all four live ClusterBridge claims still match the configured owner and
  purpose. The detached lease keeper is single-instance, writes a heartbeat,
  and exits instead of recreating a missing or foreign claim.
- A campaign-level `flock` prevents duplicate supervisors from writing the same
  cycle. Background start/resume uses an explicit `rsi-submit` entrypoint and
  waits for matching `running` state plus heartbeat before reporting success.
- Candidate diagnosis/A-Evolve changes are staged per cycle and promoted only
  when the evidence comparison accepts that cycle. A plateau counter pauses
  repeated successful-but-non-improving cycles.
- Pause/stop signals terminate the local pipeline process group and use
  persisted pool task metadata to cancel detached ClusterBridge tasks. An
  interrupted cycle re-enters ResearchClaw with `--resume` when a checkpoint
  exists.
- A full isolated network namespace cannot reach Ray GCS; pool experiments keep
  the Ray control plane reachable. Install a `researchclaw-ray-netns` policy
  helper if restricted egress is required.
- The pipeline refuses graceful degradation for production runs so it cannot
  silently turn failed evidence into a polished paper.
- API keys are never stored in the checked-in YAML.

## Backend note

The production reasoning/code-generation backend is the local Bridge Server
using `codebuddy/deepseek-v4-pro-ioa`. The placeholder environment variable
required by the OpenAI-compatible client is:

```bash
export BRIDGE_LOCAL_API_KEY=local-bridge
```

The managed RSI launcher checks the exact Python interpreter used by the
supervisor and automatically installs small required runtime packages when
missing. The same idempotent check can be run manually:

```bash
./bin/researchclaw-ensure-deps --python /usr/bin/python3
./bin/researchclaw-ensure-deps --python /usr/bin/python3 --check-only
```

This currently covers the `arxiv` client used by literature collection.

## First use

```bash
cd /data/workspace/autoresearch-stack/AutoResearchClaw
source .venv/bin/activate

# Check Bridge + pool state
curl -fsS http://127.0.0.1:8787/health
./bin/cluster-pool --config config.cluster32.yaml status --probe

# Launch one bounded RSI cycle from a detailed brief
./bin/rsi-submit --brief-file research-brief.md --single-cycle

# Or launch the persistent campaign supervisor
./bin/rsi-submit --brief-file research-brief.md

# Keep iterating until an explicit operator pause/stop
./bin/rsi-submit --brief-file research-brief.md --continuous
```

The default campaign budget is 20 RSI cycles with an automatic pause after
five consecutive non-improving cycles or three consecutive pipeline failures.
`--continuous` is the explicit always-on policy: it persists across monitor
restarts/resume, ignores the cycle/failure/plateau limits and LLM stop
recommendations, but still honors `rsi-pause` and `rsi-stop`. `--max-cycles 0`
remains available for an unbounded campaign that retains the automatic safety
pauses.

The persistent form starts both the supervisor and `rsi-monitor`. The monitor
checks Bridge health, supervisor heartbeats, checkpoints, exact claim
ownership, all 32 GPUs, the four-node Ray resource map, and the detached lease
keeper heartbeat. It suppresses an automatic supervisor restart while a
required dependency is failed, avoiding restart storms against a lost GPU
pool.

Inspect/control it with:

```bash
./bin/rsi-status <campaign-id>
./bin/rsi-monitor /absolute/path/to/campaign --max-iterations 1
./bin/rsi-pause <campaign-id> "manual review"
./bin/rsi-resume <campaign-id>
./bin/rsi-stop <campaign-id> "finished"
```

`rsi-resume` inherits the campaign's persisted model, Bridge endpoint, cycle
policy, thresholds, timing settings, and pipeline arguments unless an explicit
override is supplied. `rsi-pause` keeps the independent monitor running for
restart-on-crash and later resume; `rsi-stop` requests both the supervisor and
that campaign's monitor to exit, while leaving the externally managed GPU pool
claimed.

Do not submit or publish generated papers without human review.

## 32-GPU pool

```bash
cd /data/workspace/autoresearch-stack/AutoResearchClaw

# Claims must already belong to this session.
./bin/cluster-pool --config config.cluster32.yaml status --probe
./bin/cluster-pool --config config.cluster32.yaml validate
```

The active pool contains:

- `28.83.2.169`
- `28.83.50.39`
- `28.85.33.47`
- `28.85.50.103`

Each node has 8 NVIDIA H20 GPUs. The environment is
`/opt/conda/envs/torch-base/bin/python3` with PyTorch 2.7.1, CUDA 12.9,
NCCL 2.27.3, and Ray 2.46.0.

Validated on August 4, 2026:

- Ray exposed exactly 4 configured node IPs and 32 GPUs.
- A real sandbox task scheduled 32 concurrent one-GPU Ray workers, 8 per node.
- A 32-rank `torch.distributed` NCCL all-reduce completed correctly.
- NCCL logs confirmed the `NET/IB` transport.
