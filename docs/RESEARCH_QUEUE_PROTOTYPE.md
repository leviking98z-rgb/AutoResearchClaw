# Continuous Research Queue Prototype

This is the small testable implementation of the architecture described in
`CONTINUOUS_RESEARCH_QUEUE_ROADMAP.md`. It is deliberately **not** a
production daemon.

## What it demonstrates

```text
generate -> admit -> prepare -> run(B0/B1/B2) -> review -> conclude -> refill
```

- several independent Ideas advance asynchronously;
- each Idea has only `candidate`, `active`, `concluded`, or `quarantined`;
- B0, B1, and B2 use one Run backend;
- GPU experiments scale their requested capacity and timeout with B0/B1/B2,
  while explicitly CPU-only revisions remain CPU-only;
- Review returns only `run_more`, `escalate`, `revise`, or `conclude`;
- local mode exercises GPU slot scheduling without requesting hardware;
- ClusterBridge mode reuses the existing elastic allocation adapter and routes
  concurrently harvested results back to the correct Run;
- finite static Idea sources signal exhaustion, while cycling simulation Ideas
  receive globally unique titles;
- every concluded Idea writes a local `research_note.md`.

It does not provide production restart adoption, long-lived leases, InfoHub,
systemd, a dashboard, or a 24-hour SLA.

## Simulated local run (no model tokens, no real GPUs)

Copy and enable the example:

```bash
cp config.research-queue.example.yaml /tmp/research-queue.yaml
sed -i 's/enabled: false/enabled: true/' /tmp/research-queue.yaml
sed -i 's/max_total_ideas: 0/max_total_ideas: 4/' /tmp/research-queue.yaml
```

Run the finite prototype:

```bash
research-queue -c /tmp/research-queue.yaml start --until-idle
```

Inspect:

```bash
research-queue -c /tmp/research-queue.yaml status
research-queue -c /tmp/research-queue.yaml ideas
research-queue -c /tmp/research-queue.yaml runs
```

Artifacts:

```text
workspace/research-queue-prototype/
├── research_queue.db
├── events.jsonl
└── ideas/<idea-id>/
    ├── idea.json
    ├── research_note.md
    ├── revisions/revision-001/
    └── runs/<run-id>/
```

## Real LLM, local experiment

Set:

```yaml
execution:
  backend: local
  simulation: false
```

The existing three-tier model configuration is read from
`models.researchclaw_config`. The Worker model generates Ideas and executable
revisions; the Decision model reviews measured evidence.

Use `max_total_ideas: 1` for the first real-model smoke test.

## ClusterBridge prototype

Before any real cluster operation, follow `/root/shared/.clusters/README.md`.
Then set:

```yaml
execution:
  backend: clusterbridge
  simulation: false

gpu:
  max_total_gpus: 4
  max_gpus_per_run: 2
  resource_manager:
    owner: <stable AutoResearch owner UUID>
```

The backend requests capacity only when a fully prepared Run is ready and
reconciles demand to zero after the last Run finishes. The prototype does not
manage GPU spin.
