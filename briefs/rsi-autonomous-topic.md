# Autonomous Research Meta-Brief: Recursive Self-Improvement for Large Language Models

## Mission

Autonomously identify, select, execute, evaluate, and iteratively improve one
publishable research project on **recursive self-improvement (RSI) /
self-iterative optimization of large language models and LLM agents**. The
system is responsible for proposing the concrete research question rather than
receiving a preselected topic.

The final target is a rigorous paper package backed by reproducible experiments,
not merely a survey, opinion essay, or speculative architecture proposal.

## Autonomous topic-selection procedure

Before committing to a project:

1. Map the recent literature and open-source landscape around LLM/agent
   self-improvement, including iterative self-refinement, self-training,
   self-rewarding models, automated prompt/workflow evolution, memory/skill
   evolution, test-time learning, code-agent improvement, and safe recursive
   optimization.
2. Generate at least 12 concrete candidate research questions.
3. For every candidate, record:
   - falsifiable hypothesis;
   - closest prior work and the precise novelty gap;
   - required datasets, models, compute, and wall-clock time;
   - primary metric and strongest relevant baselines;
   - key ablations and failure/safety tests;
   - implementation and licensing feasibility;
   - expected information gain if the hypothesis is false as well as true.
4. Score candidates on novelty, scientific importance, falsifiability,
   experimental tractability on the available 32 H20 GPUs, reproducibility,
   risk, and likelihood of producing a meaningful negative or positive result.
5. Select the strongest feasible question. Preserve the candidate matrix and
   selection rationale as artifacts. Do not choose a topic solely because it
   is easy or likely to yield a positive result.
6. Run a cheap discriminating pilot before expensive scaling. If evidence
   invalidates feasibility or novelty, explicitly reject the topic and select
   the next candidate rather than polishing a failed premise.

## Preferred research shape

Prefer questions that study or improve the **mechanism of self-iteration**, for
example how an LLM agent can use execution evidence, verifier feedback,
long-term memory, learned skills, or population-based evolution to improve its
own research/coding policy across cycles. Strong projects should compare
against non-RSI controls and isolate which feedback or memory mechanisms cause
real, transferable improvement.

Possible directions are prompts, not fixed assignments:

- evidence-gated self-improvement versus unconstrained self-reflection;
- preventing self-improvement collapse, reward hacking, or regression;
- cross-task transfer of evolved skills or memories;
- credit assignment across multi-step self-improvement cycles;
- population-based versus single-trajectory agent evolution;
- compute-efficient stopping and allocation policies for recursive iteration;
- verifier ensembles and uncertainty-aware acceptance gates;
- reproducible benchmarks for measuring genuine agent improvement rather than
  benchmark overfitting.

The system may choose a better direction discovered through literature review.

## Mandatory scientific requirements

- State one primary falsifiable hypothesis and predeclare a primary metric.
- Use strong, current baselines and at least one no-self-improvement control.
- Separate development/tuning tasks from held-out evaluation tasks to test
  generalization and reduce benchmark leakage.
- Use multiple random seeds where stochasticity matters and report uncertainty
  or confidence intervals.
- Include ablations that isolate feedback, memory, mutation, selection, and
  compute effects as applicable.
- Measure compute/token/GPU cost and improvement per unit of compute, not only
  final quality.
- Track regressions, catastrophic forgetting, reward hacking, invalid outputs,
  and unsuccessful iterations.
- Preserve configs, prompts, code, logs, checkpoints, raw metrics, environment
  information, and exact provenance for every numerical claim.
- Treat negative results as valid evidence; never fabricate, smooth, or infer
  unobserved experimental results.
- Verify citations against primary sources and clearly distinguish measured
  results from hypotheses or interpretations.

## Resources and execution constraints

- Reasoning and code-generation backend: local Bridge Server using
  `codebuddy/deepseek-v4-pro-ioa`.
- Available compute: up to 32 NVIDIA H20 GPUs through the prepared
  ClusterBridge/Ray pool.
- Start with low-cost pilots and scale only when evidence justifies it.
- Reuse legal/open datasets and model checkpoints when appropriate. Do not
  assume access to proprietary model weights or private datasets.
- Keep experiments reproducible and resumable after supervisor, monitor, node,
  or task failure.

## RSI campaign policy

After each cycle:

1. Evaluate evidence quality and the primary hypothesis.
2. Diagnose the largest scientific or engineering bottleneck.
3. Propose a bounded mutation to the hypothesis, method, experiment, analysis,
   prompts, memory, or reusable skills.
4. Run the next discriminating experiment.
5. Promote mutations only when evidence improves the project; retain failed
   attempts and rejection reasons.
6. Continue autonomously until explicitly paused or stopped by the operator.

The campaign may pivot among candidate topics when evidence warrants it, but it
must log the pivot rationale and preserve comparability wherever possible.

## Deliverables

Maintain a continuously updated package containing:

- candidate-topic matrix and selection/pivot rationale;
- literature map with verified citations;
- research question, hypothesis, protocol, and preregistered primary metric;
- executable implementation and environment/configuration files;
- raw and processed results, plots, tables, uncertainty estimates, and
  compute-cost accounting;
- ablations, robustness checks, failure analysis, and limitations;
- manuscript source and compiled paper when tooling permits;
- reproducibility instructions and artifact manifest;
- an explicit claim-to-evidence ledger.

## Safety and publication boundary

Automatic paper submission, public release, repository publication, email to
venues, or external posting is prohibited. Produce local artifacts for human
review only. Do not weaken this restriction during self-modification.
