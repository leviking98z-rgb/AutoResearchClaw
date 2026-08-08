# PortfolioBench

`PortfolioBench` measures the Queue architecture with the headline metric
`VCO@window`: scientifically valid, conclusive positive **or negative**
outcomes completed inside a fixed wall-time window.

## Frozen inputs

- `ideas.v1.json`: 16 versioned calibration Ideas.
- `benchmark.cifar10.v1.yaml`: pinned CIFAR-10/ResNet20 corruption protocol
  with five independent seed pairs.
- The runner records the Git commit and SHA-256 of the Idea pool, Benchmark,
  model config, and rendered Queue config.

## Implemented variants

| Variant | Meaning |
|---|---|
| `rq-sequential` | Same Queue and scientific gate, one active Idea/Run |
| `rq-no-early-exit` | Four active Ideas, every Idea goes directly to the frozen Benchmark |
| `rq-full` | Four asynchronous Ideas with B0/B1/B2 early exit and dynamic promotion |

The direct variant is an ablation, not the normal production configuration.
The AutoResearchClaw loop adapter remains a separate baseline TODO because it
must export the same `ResearchSpec`/Benchmark evidence rather than being scored
by its incompatible legacy “paper generated” terminal state.

## Fast deterministic smoke

This runs the full state machine, revisions, budgets, scientific gate,
BenchmarkProfile compatibility check, treatment preflight, logits-cache
adapter, VCO aggregation, and cross-variant comparison without LLM/GPU cost:

```bash
python experiments/portfolio_bench/run.py \
  --mode synthetic \
  --variants rq-sequential,rq-no-early-exit,rq-full \
  --duration-sec 300 \
  --repeat 1
```

The synthetic fixture is only an integration test; it is not a scientific
result.

## Real run

Real mode uses the configured three model roles and ClusterBridge. It copies
the exact source commit to the shared run directory so remote nodes execute the
same code. With no `--logits-cache`, each promoted treatment requests one GPU
on demand. With a trusted five-seed cache, treatment evaluation is CPU-only.

```bash
python experiments/portfolio_bench/run.py \
  --mode real \
  --variants rq-sequential,rq-full \
  --duration-sec 7200 \
  --token-budget 1200000 \
  --max-gpus 8 \
  --repeat 1
```

Start with a short 2–4 Idea smoke before a complete run. Formal results require
three independent repeats and a common baseline adapter.

## Outputs

Each immutable suite directory contains:

```text
suite-manifest.json
aggregate.json
comparisons.json
<variant>-rXX/
  manifest.json
  config.yaml
  benchmark.yaml
  ideas.json
  controller.log
  artifacts/
  portfolio/
    portfolio-report.json
    summary.json
    funnel.json
    usage.json
    timeline.json
```

Real mode fails if its final ClusterBridge owner snapshot contains an
allocation or queued request.
