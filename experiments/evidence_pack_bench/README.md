# Evidence-Pack Benchmark

This benchmark isolates scientific-decision correctness from Idea generation,
code generation, and GPU scheduling.

It evaluates 25 versioned cases covering:

- valid positive and valid negative outcomes;
- insufficient independent pairs;
- missing metrics;
- NLL/accuracy regressions;
- aggregate accuracy equality with per-example argmax changes;
- wrong calibration/evaluation protocol;
- incomplete compute attestation;
- NaN evidence;
- failed execution with residual metrics; and
- a paired CI crossing zero.

Run:

```bash
python experiments/evidence_pack_bench/run.py \
  --output-dir /tmp/evidence-pack-bench
```

The comparison is:

```text
legacy_mean_only
vs
ResearchSpec + deterministic scientific gate
```

The legacy reviewer intentionally models the previously observed failure mode:
if execution succeeded and aggregate ECE improved, accept. The deterministic
reviewer independently checks the complete Evidence Pack.

`packs.json` contains the exact ResearchSpec, evidence, and gold verdict;
`report.json` contains verdict accuracy, false accepts/rejects, latency, and
per-case decisions.
