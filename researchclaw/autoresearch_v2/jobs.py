"""Pipeline-independent v2 research jobs.

Every job writes only to its immutable attempt directory.  The controller
commits accepted candidates to ``current/`` atomically.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .attestation import sha256_file
from .gates import DecisionGate, design_preflight
from .llm import StructuredRole, StructuredValidationError
from .models import (
    AttemptRecord,
    AttemptStatus,
    IdeaRecord,
    JobKind,
    JobRecord,
)
from .protocols import (
    SUPPORTED_PROTOCOLS,
    compile_screening_protocol,
    infer_protocol_template,
)
from .runtime_wrapper import (
    WRAPPER_FILENAME,
    RuntimeArtifactError,
    compile_build_output,
)
from .store import V2Store
from .validation import (
    validate_execution_argv,
    validate_experiment_artifacts,
    validate_plan,
    validate_python_tree,
    validate_research_implementation,
    validate_runtime_against_contract,
)


@dataclass(frozen=True, slots=True)
class JobOutcome:
    success: bool
    decision: str
    reason: str
    result: dict[str, Any]
    tokens: int = 0
    elapsed_sec: float = 0.0


class JobExecutor(Protocol):
    def execute(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> JobOutcome: ...


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class DesignJobExecutor:
    def __init__(
        self,
        role: StructuredRole,
        *,
        decision_gate: DecisionGate | None = None,
        max_revisions: int = 1,
        available_models: tuple[str, ...] = (),
        available_datasets: tuple[str, ...] = (),
    ) -> None:
        self.role = role
        self.decision_gate = decision_gate
        self.max_revisions = max(0, int(max_revisions))
        self.available_models = tuple(
            str(value).strip()
            for value in available_models
            if str(value).strip()
        )
        self.available_datasets = tuple(
            str(value).strip()
            for value in available_datasets
            if str(value).strip()
        )

    def execute(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> JobOutcome:
        started = time.monotonic()
        preflight = (
            design_preflight(idea)
            if self.decision_gate is not None
            else None
        )
        if preflight is not None:
            attempt.validation = {
                "ok": False,
                "decision_gate": preflight.raw or {},
                "preflight": True,
            }
            attempt.status = AttemptStatus.REJECTED
            attempt.error = preflight.reason
            store.save_attempt(attempt)
            return JobOutcome(
                True,
                "reject",
                preflight.reason,
                {"decision_gate": preflight.raw or {}, "preflight": True},
                elapsed_sec=time.monotonic() - started,
            )
        candidate = store.prepare_candidate(attempt)
        previous_plan, previous_review = self._previous_attempt_context(
            job=job,
            attempt=attempt,
            store=store,
        )
        prompt = self._prompt(
            idea,
            prior_failure=job.result,
            previous_plan=previous_plan,
            previous_review=previous_review,
        )
        total_tokens = 0
        design_revisions: list[dict[str, Any]] = []
        pending_draft: dict[str, Any] | None = None
        pending_errors: list[str] = []
        for revision in range(self.max_revisions + 1):
            try:
                if pending_draft is None:
                    result = self.role.call(
                        prompt,
                        max_tokens=10000,
                        temperature=0.25 if revision == 0 else 0.10,
                        retry_context=self._validation_repair_context,
                    )
                else:
                    repair = getattr(self.role, "repair", None)
                    if not callable(repair):
                        raise ValueError(
                            "structured role cannot repair a compiler-rejected "
                            "draft"
                        )
                    result = repair(
                        pending_draft,
                        pending_errors,
                        retry_context=self._validation_repair_context,
                        max_tokens=10000,
                        temperature=0.10,
                    )
            except StructuredValidationError as exc:
                total_tokens += exc.total_tokens
                if (
                    exc.previous_value is not None
                    and revision < self.max_revisions
                ):
                    design_revisions.append(
                        {
                            "revision": revision,
                            "decision": "draft_validation_retry",
                            "reason": str(exc),
                            "required_changes": list(exc.errors),
                        }
                    )
                    pending_draft = exc.previous_value
                    pending_errors = list(exc.errors)
                    continue
                attempt.validation = {
                    "ok": False,
                    "protocol_compiler": {"error": str(exc)},
                    "design_revisions": design_revisions,
                }
                attempt.status = AttemptStatus.REJECTED
                attempt.error = f"protocol_draft_failed: {exc}"
                store.save_attempt(attempt)
                return JobOutcome(
                    False,
                    "retry",
                    attempt.error,
                    {"validation": attempt.validation},
                    tokens=total_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
            except ValueError as exc:
                attempt.validation = {
                    "ok": False,
                    "protocol_compiler": {"error": str(exc)},
                    "design_revisions": design_revisions,
                }
                attempt.status = AttemptStatus.REJECTED
                attempt.error = f"protocol_draft_failed: {exc}"
                store.save_attempt(attempt)
                return JobOutcome(
                    False,
                    "retry",
                    attempt.error,
                    {"validation": attempt.validation},
                    tokens=total_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
            total_tokens += result.total_tokens
            pending_draft = None
            pending_errors = []
            try:
                plan = compile_screening_protocol(
                    idea,
                    result.value,
                    available_models=self.available_models,
                    available_datasets=self.available_datasets,
                )
            except (TypeError, ValueError) as exc:
                design_revisions.append(
                    {
                        "revision": revision,
                        "decision": "compiler_retry",
                        "reason": str(exc),
                        "required_changes": [str(exc)],
                    }
                )
                if revision < self.max_revisions:
                    pending_draft = dict(result.value)
                    pending_errors = [str(exc)]
                    continue
                attempt.validation = {
                    "ok": False,
                    "protocol_compiler": {
                        "error": str(exc),
                    },
                    "design_revisions": design_revisions,
                }
                attempt.status = AttemptStatus.REJECTED
                attempt.error = f"protocol_compile_failed: {exc}"
                store.save_attempt(attempt)
                return JobOutcome(
                    False,
                    "retry",
                    attempt.error,
                    {"validation": attempt.validation},
                    tokens=total_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
            deterministic_errors = validate_plan(plan)
            if deterministic_errors:
                design_revisions.append(
                    {
                        "revision": revision,
                        "decision": "plan_validation_retry",
                        "reason": "; ".join(deterministic_errors),
                        "required_changes": deterministic_errors,
                    }
                )
                if revision < self.max_revisions:
                    pending_draft = dict(result.value)
                    pending_errors = deterministic_errors
                    continue
                attempt.validation = {
                    "ok": False,
                    "plan_validation": {
                        "errors": deterministic_errors,
                    },
                    "design_revisions": design_revisions,
                }
                attempt.status = AttemptStatus.REJECTED
                attempt.error = (
                    "plan_validation_failed: "
                    + "; ".join(deterministic_errors)
                )
                store.save_attempt(attempt)
                return JobOutcome(
                    False,
                    "retry",
                    attempt.error,
                    {"validation": attempt.validation},
                    tokens=total_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
            _write_json(candidate / "plan.json", plan)
            if self.decision_gate is None:
                break
            verdict = self.decision_gate.review_design(idea, plan)
            total_tokens += verdict.tokens
            review = verdict.raw or {
                "decision": verdict.decision,
                "reason": verdict.reason,
                "confidence": verdict.confidence,
                "risks": list(verdict.risks),
                "required_changes": list(verdict.required_changes),
            }
            _write_json(candidate / "design_review.json", review)
            design_revisions.append(
                {
                    "revision": revision,
                    "decision": verdict.decision,
                    "reason": verdict.reason,
                    "blocker_codes": list(verdict.blocker_codes),
                    "required_changes": list(verdict.required_changes),
                }
            )
            if verdict.decision == "promote":
                break
            if verdict.decision == "reject":
                attempt.output_manifest = {
                    "files": ["plan.json", "design_review.json"]
                }
                attempt.validation = {
                    "ok": False,
                    "decision_gate": review,
                    "design_revisions": design_revisions,
                }
                attempt.status = AttemptStatus.REJECTED
                attempt.error = verdict.reason
                store.save_attempt(attempt)
                return JobOutcome(
                    True,
                    "reject",
                    verdict.reason,
                    {
                        "decision_gate": review,
                        "design_revisions": design_revisions,
                    },
                    tokens=total_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
            reason = (
                "design_gate_contract_failure: validated semantic gate "
                f"returned forbidden decision {verdict.decision!r}"
            )
            attempt.output_manifest = {
                "files": ["plan.json", "design_review.json"]
            }
            attempt.validation = {
                "ok": False,
                "decision_gate": review,
                "design_revisions": design_revisions,
            }
            attempt.status = AttemptStatus.REJECTED
            attempt.error = reason
            store.save_attempt(attempt)
            return JobOutcome(
                True,
                "reject",
                reason,
                {
                    "decision_gate": review,
                    "design_revisions": design_revisions,
                },
                tokens=total_tokens,
                elapsed_sec=time.monotonic() - started,
            )
        attempt.output_manifest = {"files": ["plan.json"]}
        attempt.validation = {
            "ok": True,
            "design_revisions": design_revisions,
        }
        attempt.status = AttemptStatus.VALIDATING
        store.save_attempt(attempt)
        store.commit_candidate(attempt)
        return JobOutcome(
            True,
            "promote",
            "design_accepted",
            {"plan_path": str(store.current_dir(idea.idea_id) / "plan.json")},
            tokens=total_tokens,
            elapsed_sec=time.monotonic() - started,
        )

    @staticmethod
    def _previous_attempt_context(
        *,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load the immediately preceding rejected design for targeted repair."""

        if attempt.number <= 1:
            return {}, {}
        previous = [
            item
            for item in store.list_attempts(job_id=job.job_id)
            if item.number < attempt.number
        ]
        if not previous:
            return {}, {}
        prior = max(previous, key=lambda item: item.number)
        candidate = store.attempt_dir(prior) / "candidate"
        return (
            _read_json(candidate / "plan.json"),
            _read_json(candidate / "design_review.json"),
        )

    def _prompt(
        self,
        idea: IdeaRecord,
        *,
        prior_failure: Mapping[str, Any],
        previous_plan: Mapping[str, Any] | None = None,
        previous_review: Mapping[str, Any] | None = None,
    ) -> str:
        previous_plan = dict(previous_plan or {})
        previous_review = dict(previous_review or {})
        prior_failure = dict(prior_failure or {})
        repair = ""
        if previous_plan:
            blocker_codes = previous_review.get("blocker_codes", [])
            if not (
                isinstance(blocker_codes, list)
                and bool(blocker_codes)
                and all(
                    isinstance(item, str) and item
                    for item in blocker_codes
                )
            ):
                previous_review = {}
                if prior_failure.get("decision") in {"retry", "reject"}:
                    prior_failure = {
                        key: value
                        for key, value in prior_failure.items()
                        if key
                        not in {
                            "decision",
                            "reason",
                            "decision_gate",
                            "design_revisions",
                        }
                    }
            repair = f"""
This is a REVISION attempt. Do not design a different study and do not start
over. Preserve the Idea, primary estimand, and all scientific parts of the
prior plan. Preserve its public model and benchmark only when they remain in
the executable resource manifest below; otherwise replace them with exact
manifest entries. Repair only deterministic draft/compiler/plan validation
failures recorded in prior failed-attempt feedback. Legacy open-ended
Design-review requests are not authoritative and must not be replayed.

PRIOR PLAN TO EDIT:
{json.dumps(previous_plan, ensure_ascii=False, indent=2)[:18000]}

PRIOR CLOSED-TAXONOMY DESIGN REVIEW (informational only):
{json.dumps(previous_review, ensure_ascii=False, indent=2)[:10000]}
"""
        inferred_template = infer_protocol_template(idea)
        return f"""\
Fill the scientific variables for one supported typed SCREENING-PILOT.
The Controller will compile all mechanical fields after your response.

SUPPORTED PROTOCOL TEMPLATES:
{json.dumps(sorted(SUPPORTED_PROTOCOLS), ensure_ascii=False)}

INFERRED TEMPLATE FOR THIS IDEA:
{json.dumps(inferred_template, ensure_ascii=False)}

EXECUTABLE RESOURCE MANIFEST:
- subject models: {json.dumps(list(self.available_models), ensure_ascii=False)}
- public benchmarks: {json.dumps(list(self.available_datasets), ensure_ascii=False)}
When either list is non-empty, the typed draft must use an exact value from
that list. Do not choose a nearby model size or a benchmark alias.

IDEA:
{json.dumps(idea.candidate, ensure_ascii=False, indent=2)}

Prior failed attempt feedback, if any:
{json.dumps(prior_failure, ensure_ascii=False, indent=2)[:8000]}
{repair}

Return exactly:
{{
  "protocol_template": "{inferred_template or 'one supported template'}",
  "pilot_objective": "one sentence: feasibility, protocol validation, or coarse go/no-go signal",
  "pilot_claim_scope": "what this small pilot can and cannot conclude",
  "research_question": "...",
  "hypothesis": "...",
  "primary_metric": "...",
  "metric_direction": "maximize|minimize",
  "unit_of_analysis": "one paired example, task, modification, or stream",
  "dataset": "one public benchmark; Controller creates disjoint dev/screening/confirmatory split IDs",
  "screening_access_policy": {{
    "input_access": true,
    "within_episode_feedback": false,
    "cross_example_adaptation": false,
    "hidden_labels_for_tuning": false,
    "threshold_tuning": false
  }},
  "models": [{{"name": "...", "role": "subject|verifier"}}],
  "baselines": ["include no-self-improvement"],
  "ablations": ["..."],
  "arms": [
    {{"name": "treatment", "role": "treatment"}},
    {{"name": "no-self-improvement", "role": "control"}}
  ],
  "pilot": {{
    "max_gpus": 1, "development_examples": 16,
    "max_examples": 32, "max_seeds": 1, "timeout_sec": 7200
  }},
  "call_ledger": {{
    "components": [
      {{
        "name": "candidate_generation",
        "scope": "per_example_seed",
        "dataset_role": "screening",
        "calls_per_unit": 1
      }},
      {{
        "name": "verifier_scoring",
        "scope": "per_arm_example_seed",
        "dataset_role": "screening",
        "arms": ["treatment"],
        "calls_per_unit": 1
      }}
    ]
  }},
  "gate_statistic": {{
    "name": "machine_identifier_for_the_promotion_statistic",
    "definition": "exact formula, sign, denominator and aggregation",
    "direction": "maximize|minimize",
    "threshold": {{"value": 0.15, "scale": "proportion"}},
    "undefined_policy": "reject"
  }},
  "uncertainty": {{
    "method": "paired_cluster_bootstrap|paired_bootstrap|exact_binomial|none",
    "cluster_unit": "the independent resampling unit",
    "confidence_level": 0.90,
    "resamples": 2000
  }},
  "validity_criteria": [
    {{
      "id": "minimum_completed_examples",
      "metric": "completed_examples",
      "operator": ">=",
      "value": 28,
      "scale": "absolute",
      "description": "operational validity only, never a favorable outcome"
    }}
  ],
  "promotion_criteria": [
    {{
      "id": "primary_effect",
      "metric": "same identifier as gate_statistic.name",
      "operator": ">=",
      "value": 0.15,
      "scale": "proportion",
      "description": "primary coarse screening effect"
    }},
    {{
      "id": "uncertainty_support",
      "metric": "primary_effect_ci_lower",
      "operator": ">",
      "value": 0.0,
      "scale": "absolute",
      "description": "paired uncertainty excludes no effect"
    }}
  ],
  "estimand": "exact unit, treatment contrast and aggregation",
  "sample_size_rationale": "screening precision/resolution, not confirmatory power",
  "workload_budget": {{"max_new_tokens": 512}},
  "confirmatory_followup": {{
    "claim": "the stronger paper-level claim that only Scale may support"
  }}
}}

Scientific responsibilities that remain yours:
- choose the protocol template, intervention, 2-3 arms, estimand, metric,
  gate statistic, threshold, uncertainty, typed validity/promotion criteria,
  pilot claim, screening access policy, and exact model-call components;
- include an independent no-self-improvement/reference control;
- distinguish adaptation calls from final evaluation calls;
- distinguish the raw endpoint direction from gate_statistic.direction. For
  example, raw regret may be minimized while relative regret reduction is
  maximized; promotion follows the gate statistic, never the raw endpoint;
- declare within-episode feedback separately from cross-example adaptation.
  Screening inputs may be visible and an online protocol may consume its own
  prior outcome inside the same episode, while hidden labels, thresholds, and
  cross-example state remain frozen unless explicitly declared;
- never use confirmatory labels/assertions for tuning, selection, calibration,
  prompts, memory, inheritance, or stopping thresholds before Scale. Scale may
  present confirmatory inputs for generation and score them exactly once.
- keep the Pilot deliberately minimal: one public benchmark, one open-weight
  subject model, one seed, one GPU, 16-32 complete paired examples, two primary
  arms plus one independent no-self-improvement reference, at most 512 model
  calls, and at most one primary plus one support promotion criterion;
- specify the complete executable algorithm in the model-owned text fields:
  input partition, candidate/update order, frozen parameters, pairing, random
  streams, selection/promotion rule, tie/duplicate/unscorable handling, metric
  numerator and denominator, and comparable outcome for every arm;
- use complete-case operational validity when missing pairs could alter the
  coarse gate. Scientific zero events, absent classes, undefined ratios,
  low variation, CI crossing, or futility are valid rejects, not retry;
- keep endpoint data independent of adaptation and selection. If screening
  outcomes drive cross-example state, define a separate frozen evaluation
  partition for the endpoint instead of scoring the adaptation stream itself.
- defer extra baselines, ablations, datasets, and stronger claims to Scale;
  do not list them as Pilot measurements unless their exact calls are in the
  ledger and they are required for the one go/no-go question.

Mechanical fields that the Controller owns and will overwrite:
- dataset roles and stable split identifiers;
- sample_accounting and workload arithmetic;
- exhaustive invalid / meets-all-promotion / valid-otherwise regions;
- decision_contract plus synchronized promotion/early-stop prose;
- Scale examples/seeds and untouched confirmatory split;
- required runtime evidence.

Decision semantics are strict:
- invalid operational evidence -> retry;
- valid evidence satisfying every promotion criterion -> promote;
- every other valid result, including a CI crossing the boundary, zero or
  undefined denominator, low event count, flat outcome, or unfavorable
  secondary gate -> reject. Never label scientific inconclusiveness invalid.

Each validity/promotion criterion must contain exactly:
id, metric, operator, value, scale, description. Use only <, <=, >, >=, ==.
The primary promotion criterion must reference gate_statistic.name exactly,
use its exact threshold value/scale, and use an operator consistent with the
gate statistic direction. Keep total criteria small and conjunctive.

Allowed call_ledger component names are:
adaptation, candidate_generation, verifier_scoring, calibration,
memory_writing, shadow_continuation, baseline_reference, final_evaluation.
The same component name may appear more than once when its scope,
dataset_role, or exact arm subset differs; do not duplicate the exact same
name/scope/dataset_role/arms tuple.
Allowed scopes are:
- per_arm_example_seed: multiplied by selected arms, dataset examples, seeds;
- per_example_seed: one shared call per dataset example and seed;
- per_arm_seed: fixed per selected arm and seed, without example multiplier;
- per_seed: fixed shared calls per seed;
- fixed: one fixed call.
Do not invent aliases such as per_task, per_example, per_arm, once, or global.
For a call repeated once per benchmark item use per_example_seed; for a call
repeated once per arm and item use per_arm_example_seed.
For scopes containing "arm", omit arms to apply to all arms or name the exact
subset. For scopes containing "example", dataset_role must be development or
screening. Other scopes use dataset_role=none. Include zero-cost operations in
prose, not as components. This pilot must use 16-32 screening examples, at most
32 development examples, one seed and one GPU, and at most 512 total model
calls.
"""

    @staticmethod
    def _validation_repair_context(
        previous_value: Mapping[str, Any],
        errors: list[str],
    ) -> str:
        """Keep typed-draft repair local to model-owned scientific fields."""

        return f"""\
Repair the exact prior JSON below; do not redesign the study, add arms, change
the Idea, or expand the scientific protocol. Preserve every field not named by
the validation errors.

This is a typed scientific draft, not the compiled plan. Do not add or edit
sample_accounting, workload totals, split IDs, decision_table, or required
runtime evidence; the Controller derives those fields.

For decision fields:
- primary_metric + metric_direction describe the raw endpoint;
- gate_statistic independently defines the signed promotion statistic;
- gate_statistic.undefined_policy must be reject;
- validity_criteria contain only operational/protocol validity checks;
- promotion_criteria contain every scientific go/no-go gate as a conjunction;
- exactly one promotion criterion must reference gate_statistic.name, use its
  exact threshold value/scale, and point in gate_statistic.direction;
- valid but unfavorable, undefined, low-event, flat, or CI-crossing outcomes
  must reject rather than retry.

For screening_access_policy, provide all five booleans. input_access must be
true; hidden_labels_for_tuning and threshold_tuning must be false. Use
within_episode_feedback for feedback inside one episode and
cross_example_adaptation only for state carried between independent examples.

For call_ledger:
- allowed names are adaptation, candidate_generation, verifier_scoring,
  calibration, memory_writing, shadow_continuation, baseline_reference,
  final_evaluation;
- the same name may be reused only for a different scope, dataset_role, or
  exact arm subset; never duplicate the same name/scope/dataset_role/arms;
- allowed scopes are per_arm_example_seed, per_example_seed, per_arm_seed,
  per_seed, fixed;
- never invent aliases such as per_task, per_example, per_arm, once, or global;
- scopes containing "arm" may name an exact arm subset;
- scopes containing "example" require dataset_role development or screening;
- every calls_per_unit must be a positive integer.

For gate_statistic.threshold.scale and every criterion scale use exactly
proportion, percentage_points, or absolute. Bootstrap methods require at least
200 resamples. Do not weaken the scientific threshold merely to pass
validation.

PRIOR JSON TO REPAIR:
{json.dumps(dict(previous_value), ensure_ascii=False, indent=2)[:24000]}

DETERMINISTIC ERRORS TO FIX:
{json.dumps(errors, ensure_ascii=False, indent=2)[:8000]}
"""

class BuildJobExecutor:
    def __init__(
        self,
        role: StructuredRole,
        *,
        smoke_timeout_sec: float = 300.0,
        python_executable: str = "",
        execute_smoke_locally: bool = True,
    ) -> None:
        self.role = role
        self.smoke_timeout_sec = max(1.0, float(smoke_timeout_sec))
        self.python_executable = str(python_executable).strip()
        self.execute_smoke_locally = bool(execute_smoke_locally)

    def execute(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> JobOutcome:
        started = time.monotonic()
        current = store.current_dir(idea.idea_id)
        plan = json.loads((current / "plan.json").read_text(encoding="utf-8"))
        candidate = store.snapshot_current(
            attempt,
            include_artifacts=False,
        )
        result = self.role.call(
            self._prompt(idea, plan, prior_failure=job.result),
            # Build is a small executable scaffold, not a generated framework.
            # Bounding the response prevents one oversized project snapshot
            # from monopolizing a Bridge slot for tens of minutes.
            max_tokens=12000,
            temperature=0.20,
        )
        try:
            result, output, validation, implementation = (
                self._materialize_and_validate(
                    result=result,
                    candidate=candidate,
                    plan=plan,
                )
            )
        except RuntimeArtifactError as exc:
            validation = {
                "ok": False,
                "errors": [
                    {
                        "file": "",
                        "category": "runtime_wrapper",
                        "message": str(exc),
                    }
                ],
                "warnings": [],
                "files_checked": [],
            }
            output = dict(result.value)
            implementation = {
                "ok": False,
                "errors": [str(exc)],
                "evidence": {},
            }
        if not validation["ok"]:
            repair_errors = [
                str(error)
                for error in implementation.get("errors", [])
            ]
            repair_errors.extend(
                str(issue.get("message", "") or "")
                for issue in validation.get("errors", [])
                if str(issue.get("message", "") or "")
            )
            repair = getattr(self.role, "repair", None)
            if repair_errors and callable(repair):
                shutil.rmtree(candidate, ignore_errors=True)
                candidate = store.snapshot_current(
                    attempt,
                    include_artifacts=False,
                )
                result = repair(
                    output,
                    repair_errors,
                    retry_context=self._validation_repair_context,
                    max_tokens=12000,
                    temperature=0.10,
                )
                try:
                    result, output, validation, implementation = (
                        self._materialize_and_validate(
                            result=result,
                            candidate=candidate,
                            plan=plan,
                        )
                    )
                except RuntimeArtifactError as exc:
                    validation = {
                        "ok": False,
                        "errors": [
                            {
                                "file": "",
                                "category": "runtime_wrapper",
                                "message": str(exc),
                            }
                        ],
                        "warnings": [],
                        "files_checked": [],
                    }
                    output = dict(result.value)
                    implementation = {
                        "ok": False,
                        "errors": [str(exc)],
                        "evidence": {},
                    }
        validation["research_contract"] = implementation
        validation["ok"] = bool(
            validation.get("ok") and implementation.get("ok")
        )
        if self.execute_smoke_locally:
            smoke = self._run_smoke(
                candidate=candidate,
                command=output["commands"]["smoke"],
                attempt=attempt,
                store=store,
            )
        else:
            smoke = {
                "ok": True,
                "executed": False,
                "environment": "gpu_pool",
                "reason": "deferred_to_controller_managed_remote_smoke",
            }
        validation["smoke"] = smoke
        validation["ok"] = bool(
            validation.get("ok") and smoke.get("ok")
        )
        attempt.validation = validation
        attempt.output_manifest = {
            "files": sorted(output["files"]),
            "commands": output["commands"],
        }
        if not validation["ok"]:
            attempt.status = AttemptStatus.REJECTED
            attempt.error = "deterministic code validation failed"
            store.save_attempt(attempt)
            return JobOutcome(
                False,
                "retry",
                "code_validation_failed",
                {"validation": validation},
                tokens=result.total_tokens,
                elapsed_sec=time.monotonic() - started,
            )
        attempt.status = AttemptStatus.VALIDATING
        store.save_attempt(attempt)
        store.commit_candidate(attempt)
        return JobOutcome(
            True,
            "promote",
            "build_accepted",
            {
                "commands": output["commands"],
                "validation": validation,
            },
            tokens=result.total_tokens,
            elapsed_sec=time.monotonic() - started,
        )

    @staticmethod
    def _materialize_and_validate(
        *,
        result: Any,
        candidate: Path,
        plan: dict[str, Any],
    ) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
        output = compile_build_output(result.value)
        for filename, content in output["files"].items():
            target = (candidate / filename).resolve()
            if not target.is_relative_to(candidate.resolve()):
                raise ValueError(f"path traversal: {filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        _write_json(candidate / "build.json", output)
        validation = validate_python_tree(
            candidate,
            trusted_files={WRAPPER_FILENAME},
        )
        implementation = validate_research_implementation(
            candidate,
            plan=plan,
            controller_runtime=True,
        )
        validation["ok"] = bool(
            validation.get("ok") and implementation.get("ok")
        )
        return result, output, validation, implementation

    @staticmethod
    def _validation_repair_context(
        previous_value: Mapping[str, Any],
        errors: list[str],
    ) -> str:
        return f"""\
Repair the exact compact project snapshot below. Preserve the experiment's
scientific design, commands, model, benchmark, budgets, and overall file
structure. Change only what is necessary to satisfy the deterministic
Build-to-Runtime validation errors.

The Controller now owns the final artifact envelope and decision schema.
Generated code must only make every measured criterion metric available as a
finite scalar and report the exact model identifier, real datasets, examples,
seed identifiers, and compiled call-ledger component counts.

PREVIOUS PROJECT JSON:
{json.dumps(dict(previous_value), ensure_ascii=False, indent=2)[:24000]}

DETERMINISTIC ERRORS:
{json.dumps(errors, ensure_ascii=False, indent=2)[:8000]}
"""

    @staticmethod
    def _prompt(
        idea: IdeaRecord,
        plan: Mapping[str, Any],
        *,
        prior_failure: Mapping[str, Any],
    ) -> str:
        return f"""\
Implement the complete experiment below as a small Python project.

IDEA:
{json.dumps(idea.candidate, ensure_ascii=False, indent=2)}

PLAN:
{json.dumps(plan, ensure_ascii=False, indent=2)}

PRIOR FAILED ATTEMPT FEEDBACK, IF ANY:
{json.dumps(dict(prior_failure), ensure_ascii=False, indent=2)[:8000]}

Return JSON:
{{
  "files": {{
    "main.py": "complete compact experiment implementation"
  }},
  "commands": {{
    "smoke": "python main.py --mode smoke --output artifacts/smoke",
    "pilot": "python main.py --mode pilot --output artifacts/pilot",
    "scale": "python main.py --mode scale --output artifacts/scale"
  }},
  "dependencies": ["transformers", "datasets"],
  "expected_outputs": ["metrics.json", "runtime_evidence.json"]
}}

Requirements:
- Every returned file is complete, never a patch or prefix.
- Keep the project deliberately small: prefer one main.py, allow at most three
  Python files, no file may exceed 800 lines, and do not generate a framework,
  package boilerplate, notebooks, documentation, vendored code, or tests.
- Reuse transformers/datasets/torch directly instead of reimplementing model,
  dataset, statistics, or orchestration libraries.
- Use canonical namespaced Hugging Face dataset ids. In particular, use
  "openai/gsm8k" instead of "gsm8k" and
  "google-research-datasets/mbpp" instead of "mbpp"; current datasets/hub
  clients reject those legacy unnamespaced aliases.
- The runtime must write raw metrics.json and runtime_evidence.json. The
  Controller wraps your commands and owns the final artifact envelope,
  criterion_results schema, dataset/split declarations, GPU count, and gate
  decision. Your raw runtime_evidence must still report the measured model id,
  datasets, examples, seeds, call counts, and scalar scientific metrics.
- In runtime_evidence, examples_processed is the number of endpoint
  screening/confirmatory evaluation units, not development plus endpoint
  examples. Report development and endpoint counts separately in
  examples_by_role.
- call_counts keys must exactly match the names in plan.call_ledger.components.
  Increment each component at the actual call site. Never collapse multiple
  components into generic generation_calls/verifier_calls totals.
- Smoke mode is a real integration test, not a dry run: load the declared
  pretrained LLM through the actual transformers/model loader, load at least
  one real example from a declared benchmark, move the model and tensors to
  CUDA, execute at least one model forward/generation, synchronize CUDA, and
  write the measured artifacts. Never fake CUDA, GPU UUID, utilization, memory,
  model, dataset, or benchmark evidence; Controller-owned trusted telemetry
  independently verifies the physical execution.
- Emit measured values for every metric referenced by validity_criteria and
  promotion_criteria. The Controller mechanically applies the compiled
  operators, thresholds, and retry/promote/reject decision table.
- If any promotion criterion references an uncertainty metric (for example a
  confidence-interval bound), runtime_evidence must include an ``uncertainty``
  object with the configured method, confidence level, resamples, RNG seed,
  and the frozen item/cluster-level observations used to resample. The
  observations must reproduce gate_statistic.name as either their sum
  (absolute-count effects) or mean (proportion effects). The Controller owns
  and recomputes interval bounds; aggregate counts or precomputed CI scalars
  alone are not auditable and will fail the deterministic runtime contract.
- The metric named by gate_statistic.name must be emitted in both metrics
  artifacts. Do not derive promotion direction from primary_metric or
  metric_direction.
- Enforce each dataset access_policy: within-episode feedback and cross-example
  adaptation are distinct; confirmatory inputs are first opened at Scale and
  confirmatory labels/assertions never tune prompts, thresholds, memory, or
  selection.
- Respect plan budgets. Never synthesize scientific outcomes.
- Every commands value must be either a direct argv list or a shell-free
  Python command such as "python main.py --mode pilot --output ...".
- Do not use pipes, redirects, command substitution, shell builtins, package
  installation, or multi-command strings.
"""

    def _run_smoke(
        self,
        *,
        candidate: Path,
        command: object,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> dict[str, Any]:
        try:
            argv = _command_argv(command)
        except ValueError as exc:
            return {
                "ok": False,
                "returncode": -1,
                "errors": [f"invalid commands.smoke: {exc}"],
            }
        errors = validate_execution_argv(argv, path="commands.smoke")
        if errors:
            return {"ok": False, "returncode": -1, "errors": errors}
        env = _execution_env(
            idea_id=attempt.idea_id,
            job_id=attempt.job_id,
            attempt_id=attempt.attempt_id,
            output_dir=candidate / "artifacts" / "smoke",
            gpu_count=0,
        )
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        resolved_argv = list(argv)
        if self.python_executable:
            resolved_argv[0] = self.python_executable
        try:
            completed = subprocess.run(
                resolved_argv,
                cwd=candidate,
                env=env,
                text=True,
                capture_output=True,
                timeout=self.smoke_timeout_sec,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "ok": False,
                "argv": resolved_argv,
                "returncode": -1,
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
        attempt_dir = store.attempt_dir(attempt)
        stdout_path = attempt_dir / "smoke.stdout.log"
        stderr_path = attempt_dir / "smoke.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        return {
            "ok": completed.returncode == 0,
            "executed": True,
            "environment": "local",
            "argv": resolved_argv,
            "returncode": completed.returncode,
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
            "errors": (
                []
                if completed.returncode == 0
                else [f"smoke returncode={completed.returncode}"]
            ),
        }


class ExperimentJobExecutor:
    """CPU/local executor; GPU jobs use the same command through GPUBroker."""

    def __init__(
        self,
        *,
        decision_gate: DecisionGate | None = None,
    ) -> None:
        self.decision_gate = decision_gate

    def execute(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> JobOutcome:
        started = time.monotonic()
        candidate = store.snapshot_current(attempt)
        build = json.loads((candidate / "build.json").read_text(encoding="utf-8"))
        mode = "pilot" if job.kind is JobKind.PILOT else "scale"
        argv = _command_argv(build["commands"][mode])
        command_errors = validate_execution_argv(
            argv,
            path=f"commands.{mode}",
        )
        if command_errors:
            raise ValueError("; ".join(command_errors))
        artifacts = candidate / "artifacts" / mode
        artifacts.mkdir(parents=True, exist_ok=True)
        env = _execution_env(
            idea_id=idea.idea_id,
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            output_dir=artifacts,
            gpu_count=0,
        )
        completed = subprocess.run(
            argv,
            cwd=candidate,
            env=env,
            text=True,
            capture_output=True,
            timeout=job.timeout_sec,
            check=False,
        )
        (store.attempt_dir(attempt) / "stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (store.attempt_dir(attempt) / "stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        metrics_path = artifacts / "metrics.json"
        validation = validate_experiment_artifacts(artifacts)
        runtime_errors = validate_runtime_against_contract(
            plan=_read_json(candidate / "plan.json"),
            runtime_evidence=validation.get("runtime_evidence", {}),
            allocated_gpus=int(
                validation.get("runtime_evidence", {}).get(
                    "gpu_count",
                    0,
                )
                or 0
            ),
            mode=mode,
            pilot_runtime=(
                _read_json(
                    candidate
                    / "artifacts"
                    / "pilot"
                    / "runtime_evidence.json"
                )
                if mode == "scale"
                else None
            ),
        )
        if runtime_errors:
            validation["errors"].extend(runtime_errors)
            validation["ok"] = False
        attempt.validation = validation
        attempt.output_manifest = {
            "argv": argv,
            "returncode": completed.returncode,
            "metrics_path": str(metrics_path),
            "runtime_evidence_path": str(
                artifacts / "runtime_evidence.json"
            ),
        }
        if completed.returncode != 0 or not validation["ok"]:
            attempt.status = AttemptStatus.REJECTED
            attempt.error = (
                f"returncode={completed.returncode}; "
                + "; ".join(validation.get("errors", []))
            )
            store.save_attempt(attempt)
            return JobOutcome(
                False,
                "retry",
                "experiment_invalid",
                {
                    "returncode": completed.returncode,
                    "validation": validation,
                },
                elapsed_sec=time.monotonic() - started,
            )
        metrics = validation["metrics"]
        gate_tokens = 0
        if self.decision_gate is not None:
            plan = _read_json(candidate / "plan.json")
            verdict = self.decision_gate.review_experiment(
                idea,
                kind=job.kind,
                plan=plan,
                metrics=metrics,
                runtime_evidence=validation["runtime_evidence"],
            )
            gate = {
                "decision": verdict.decision,
                "reason": verdict.reason,
                "confidence": verdict.confidence,
                "risks": list(verdict.risks),
                "required_changes": list(verdict.required_changes),
            }
            gate_tokens = verdict.tokens
        else:
            gate = _experiment_gate(metrics)
        gate = resolve_experiment_lifecycle_gate(
            runtime_evidence=validation["runtime_evidence"],
            gate=gate,
        )
        decision_path = store.attempt_dir(attempt) / "decision_review.json"
        _write_json(decision_path, gate)
        attempt.output_manifest["decision_review_path"] = str(decision_path)
        if gate["decision"] == "retry":
            attempt.status = AttemptStatus.REJECTED
            attempt.error = str(gate["reason"])
            attempt.validation["decision_gate"] = gate
            store.save_attempt(attempt)
            return JobOutcome(
                False,
                "retry",
                str(gate["reason"]),
                {"metrics": metrics, "gate": gate},
                tokens=gate_tokens,
                elapsed_sec=time.monotonic() - started,
            )
        attempt.status = AttemptStatus.VALIDATING
        attempt.validation["decision_gate"] = gate
        store.save_attempt(attempt)
        store.commit_candidate(attempt)
        return JobOutcome(
            True,
            gate["decision"],
            gate["reason"],
            {"metrics": metrics, "gate": gate},
            tokens=gate_tokens,
            elapsed_sec=time.monotonic() - started,
        )


def _experiment_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    decision = str(metrics.get("decision", "") or "").casefold()
    success_probability = float(metrics.get("success_probability", 0.0) or 0.0)
    futility_probability = float(
        metrics.get("futility_probability", 0.0) or 0.0
    )
    informative_null = bool(metrics.get("informative_null", False))
    if decision in {"reject", "stop"} or futility_probability >= 0.95:
        return {
            "decision": "complete_negative" if informative_null else "reject",
            "reason": (
                "informative_negative"
                if informative_null
                else "pilot_futility"
            ),
        }
    if decision in {"promote", "continue"} or success_probability >= 0.95:
        return {"decision": "promote", "reason": "primary_signal_valid"}
    return {"decision": "reject", "reason": "insufficient_information_gain"}


def resolve_experiment_lifecycle_gate(
    *,
    runtime_evidence: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate scientific promotion rejection from lifecycle rejection.

    A controller-compiled runtime ``reject`` means the evidence was valid but
    failed one or more preregistered promotion criteria. That is a reportable
    negative result, not an instruction to discard the Idea. Strong-model
    review may enrich the reason and risks, but it cannot silently skip the
    Report stage for valid measured evidence.
    """

    resolved = dict(gate)
    runtime_decision = str(
        runtime_evidence.get("gate_decision", "") or ""
    ).casefold()
    if (
        runtime_decision == "reject"
        and runtime_evidence.get("evidence_valid") is True
        and runtime_evidence.get("gate_statistic_defined") is True
    ):
        criterion_results = runtime_evidence.get("criterion_results")
        measured_criteria = (
            isinstance(criterion_results, Mapping)
            and bool(criterion_results)
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("passed"), bool)
                and not isinstance(row.get("value"), bool)
                and isinstance(row.get("value"), (int, float))
                for row in criterion_results.values()
            )
        )
        if measured_criteria:
            resolved["decision"] = "complete_negative"
            resolved["reason"] = str(
                resolved.get("reason", "") or "informative_negative"
            )
            resolved["report_disposition"] = "reportable_negative"
            resolved["scientific_decision"] = "reject"
    return resolved


class ReportJobExecutor:
    def __init__(
        self,
        role: StructuredRole,
        *,
        decision_gate: DecisionGate | None = None,
        max_revisions: int = 3,
    ) -> None:
        self.role = role
        self.decision_gate = decision_gate
        self.max_revisions = max(0, int(max_revisions))

    def execute(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> JobOutcome:
        started = time.monotonic()
        candidate = store.snapshot_current(attempt)
        context = self._report_context(
            idea=idea,
            candidate=candidate,
        )
        previous_report, previous_review = self._previous_attempt_context(
            job=job,
            attempt=attempt,
            store=store,
        )
        result = self.role.call(
            self._prompt(
                context,
                previous_report=previous_report,
                previous_review=previous_review,
            ),
            max_tokens=18000,
            temperature=0.10 if previous_report else 0.20,
        )
        total_tokens = result.total_tokens
        report = dict(result.value)
        review_history: list[dict[str, Any]] = []
        gate_tokens = 0
        verdict = None
        for revision in range(self.max_revisions + 1):
            _write_json(candidate / "report.json", report)
            (candidate / "paper.md").write_text(
                str(report["paper_markdown"]),
                encoding="utf-8",
            )
            if self.decision_gate is None:
                break
            verdict = self.decision_gate.review_report(
                idea,
                report,
                context,
            )
            gate_tokens += verdict.tokens
            review = verdict.raw or {
                "decision": verdict.decision,
                "reason": verdict.reason,
                "confidence": verdict.confidence,
                "risks": list(verdict.risks),
                "required_changes": list(verdict.required_changes),
            }
            review_history.append(dict(review))
            _write_json(candidate / "final_review.json", review)
            if verdict.decision == "complete":
                break
            if (
                verdict.decision != "retry"
                or revision >= self.max_revisions
            ):
                attempt.status = AttemptStatus.REJECTED
                attempt.error = verdict.reason
                attempt.validation = {
                    "ok": False,
                    "decision_gate": review,
                    "report_revisions": review_history,
                }
                attempt.output_manifest = {
                    "files": [
                        "report.json",
                        "paper.md",
                        "final_review.json",
                    ]
                }
                store.save_attempt(attempt)
                return JobOutcome(
                    verdict.decision == "reject",
                    verdict.decision,
                    verdict.reason,
                    {
                        "decision_gate": review,
                        "report_revisions": review_history,
                    },
                    tokens=total_tokens + gate_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
            repaired = self.role.call(
                self._repair_prompt(
                    context=context,
                    report=report,
                    review=review,
                ),
                max_tokens=18000,
                temperature=0.05,
            )
            total_tokens += repaired.total_tokens
            report = dict(repaired.value)
        attempt.validation = {"ok": True}
        if review_history:
            attempt.validation["decision_gate"] = review_history[-1]
            attempt.validation["report_revisions"] = review_history
        attempt.output_manifest = {
            "files": [
                "report.json",
                "paper.md",
                *(
                    ["final_review.json"]
                    if self.decision_gate is not None
                    else []
                ),
            ]
        }
        attempt.status = AttemptStatus.VALIDATING
        store.save_attempt(attempt)
        # Startup recovery may remove an interrupted candidate while its
        # detached worker is still alive. Such a worker can later recreate
        # only report.json, paper.md, and final_review.json. Merge the latest
        # committed snapshot without overwriting those accepted report files.
        store.merge_current_into_candidate(attempt)
        store.commit_candidate(attempt)
        return JobOutcome(
            True,
            "complete",
            "paper_package_generated",
            {"paper": str(store.current_dir(idea.idea_id) / "paper.md")},
            tokens=total_tokens + gate_tokens,
            elapsed_sec=time.monotonic() - started,
        )

    @staticmethod
    def _report_context(
        *,
        idea: IdeaRecord,
        candidate: Path,
    ) -> dict[str, Any]:
        plan = _read_json(candidate / "plan.json")
        evidence = _collect_report_evidence(candidate / "artifacts")
        return {
            "idea": {
                "idea_id": idea.idea_id,
                "title": idea.title,
                "research_question": idea.research_question,
                "falsifiable_hypothesis": idea.falsifiable_hypothesis,
                "primary_metric": idea.primary_metric,
                "final_outcome": idea.candidate.get("final_outcome", ""),
                "scientific_exit_reason": idea.exit_reason,
                "models": idea.candidate.get("models", []),
                "datasets": idea.candidate.get("datasets", []),
                "baselines": idea.candidate.get("baselines", []),
                "ablations": idea.candidate.get("ablations", []),
                "novelty_gap": idea.candidate.get("novelty_gap", ""),
                "closest_prior_work": idea.candidate.get(
                    "closest_prior_work",
                    [],
                ),
            },
            "plan": plan,
            "evidence": evidence,
        }

    @staticmethod
    def _previous_attempt_context(
        *,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if attempt.number <= 1:
            return {}, {}
        previous = [
            item
            for item in store.list_attempts(job_id=job.job_id)
            if item.number < attempt.number
        ]
        if not previous:
            return {}, {}
        prior = max(previous, key=lambda item: item.number)
        candidate = store.attempt_dir(prior) / "candidate"
        return (
            _read_json(candidate / "report.json"),
            _read_json(candidate / "final_review.json"),
        )

    @staticmethod
    def _prompt(
        context: Mapping[str, Any],
        *,
        previous_report: Mapping[str, Any] | None = None,
        previous_review: Mapping[str, Any] | None = None,
    ) -> str:
        revision = ""
        if previous_report:
            revision = f"""\

This is a REVISION of a report that failed evidence review. Preserve the
research question and measured results, but correct every review finding.

PREVIOUS REPORT:
{json.dumps(dict(previous_report), ensure_ascii=False, indent=2)[:30000]}

PREVIOUS FINAL REVIEW:
{json.dumps(dict(previous_review or {}), ensure_ascii=False, indent=2)[:12000]}
"""
        return f"""\
Produce an evidence-bounded research report from:
{json.dumps(context, ensure_ascii=False, indent=2)[:70000]}
{revision}

Return JSON:
{{
  "title": "...",
  "claims": [
    {{
      "claim": "...",
      "evidence_paths": [
        "/evidence/pilot/metrics/endpoint_correct_diff"
      ],
      "strength": "measured|hypothesis"
    }}
  ],
  "limitations": ["..."],
  "next_experiments": ["..."],
  "paper_markdown": "# Title\\n..."
}}
Do not invent results, citations, or runs. Clearly report negative results.
The EVIDENCE object is authoritative for measured outcomes. Every measured
claim must cite at least one exact JSON-pointer path beginning with
"/evidence/"; it may additionally cite exact "/plan/" or "/idea/" paths for
pre-specified thresholds, denominators, decision contracts, or design
provenance. Never cite Python-style paths such as idea.*, plan.*, or
artifacts/*. Reconcile measured call counts with the supplied plan rather than
silently asserting protocol adherence. Report disagreement, tie, and
abstention rates as separate observables; aggregate rates do not prove that
one outcome converted into another. Do not call ties abstentions unless the
supplied protocol explicitly defines that equivalence. Recommend future work
as a recommendation, not as an already pre-specified or untouched study,
unless those properties are directly supported by exact context paths. Use
"pre-specified", not "preregistered", unless the supplied evidence contains
an immutable, time-stamped preregistration artifact.
"""

    @staticmethod
    def _repair_prompt(
        *,
        context: Mapping[str, Any],
        report: Mapping[str, Any],
        review: Mapping[str, Any],
    ) -> str:
        return f"""\
Repair the exact report below using the authoritative context and final-review
findings. Return one complete corrected JSON report, not a patch. Preserve
valid prose and change only claims, evidence paths, numeric results,
limitations, and interpretations that the review identifies as wrong.

AUTHORITATIVE CONTEXT:
{json.dumps(dict(context), ensure_ascii=False, indent=2)[:70000]}

REJECTED REPORT:
{json.dumps(dict(report), ensure_ascii=False, indent=2)[:30000]}

FINAL REVIEW:
{json.dumps(dict(review), ensure_ascii=False, indent=2)[:16000]}

Every measured claim must cite at least one exact JSON-pointer path beginning
with "/evidence/"; it may additionally cite exact "/plan/" or "/idea/" paths
for pre-specified thresholds, denominators, decision contracts, or design
provenance. Never claim measured metrics are unavailable when they appear in
the context. Reconcile planned and measured call counts explicitly. Keep tie,
abstention, and disagreement statements separate. Do not infer that aggregate
rates represent transition-level conversions. Remove unsupported provenance
claims instead of preserving them. Return:
{{
  "title": "...",
  "claims": [
    {{"claim": "...", "evidence_paths": ["/evidence/..."], "strength": "measured|hypothesis"}}
  ],
  "limitations": ["..."],
  "next_experiments": ["..."],
  "paper_markdown": "# Title\\n..."
}}
"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _command_argv(value: object) -> list[str]:
    if isinstance(value, list) and all(
        isinstance(item, str) for item in value
    ):
        return list(value)
    if isinstance(value, str):
        import shlex

        return shlex.split(value)
    return []


def _execution_env(
    *,
    idea_id: str,
    job_id: str,
    attempt_id: str,
    output_dir: Path,
    gpu_count: int,
) -> dict[str, str]:
    """Pass only a minimal allow-listed environment to generated code."""

    inherited = {
        key: value
        for key in (
            "CUDA_HOME",
            "CUDA_DEVICE_ORDER",
            "CUDA_VISIBLE_DEVICES",
            "HF_HOME",
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "PATH",
            "PYTHONPATH",
            "TORCH_HOME",
            "NVIDIA_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
        )
        if (value := os.environ.get(key))
    }
    inherited.update(
        {
            "AUTORESEARCH_V2_IDEA_ID": idea_id,
            "AUTORESEARCH_V2_JOB_ID": job_id,
            "AUTORESEARCH_V2_ATTEMPT_ID": attempt_id,
            "AUTORESEARCH_V2_GPU_COUNT": str(max(0, int(gpu_count))),
            "AUTORESEARCH_V2_OUTPUT_DIR": str(output_dir.resolve()),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return inherited


def _collect_metrics(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "value": _read_json(path),
        }
        for path in sorted(root.rglob("metrics.json"))
    ]


def _collect_report_evidence(root: Path) -> dict[str, Any]:
    """Expose one deterministic, path-stable evidence object to report roles."""

    evidence: dict[str, Any] = {}
    if not root.is_dir():
        return evidence
    for mode_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    ):
        metrics = _read_json(mode_dir / "metrics.json")
        runtime = _read_json(mode_dir / "runtime_evidence.json")
        if not metrics and not runtime:
            continue
        evidence[mode_dir.name] = {
            "metrics": dict(metrics.get("metrics", {}) or {}),
            "decision": str(metrics.get("decision", "") or ""),
            "result_valid": metrics.get("result_valid"),
            "runtime": {
                key: runtime.get(key)
                for key in (
                    "evidence_valid",
                    "gate_decision",
                    "gate_statistic_defined",
                    "criterion_results",
                    "call_counts",
                    "examples_processed",
                    "examples_by_role",
                    "gpu_count",
                    "model_loaded",
                    "datasets_loaded",
                    "dataset_roles",
                    "seeds",
                    "split_identifiers",
                    "uncertainty",
                )
                if key in runtime
            },
        }
    return evidence


class SimulatedJobExecutor:
    """Deterministic end-to-end executor for controller/recovery tests."""

    def execute(
        self,
        *,
        idea: IdeaRecord,
        job: JobRecord,
        attempt: AttemptRecord,
        store: V2Store,
    ) -> JobOutcome:
        candidate = store.snapshot_current(attempt)
        if job.kind is JobKind.DESIGN:
            _write_json(
                candidate / "plan.json",
                {
                    "research_question": idea.research_question,
                    "hypothesis": idea.falsifiable_hypothesis,
                    "primary_metric": idea.primary_metric,
                    "datasets": ["synthetic-test-fixture"],
                    "models": ["deterministic-test-model"],
                    "baselines": ["single-pass"],
                    "ablations": ["remove mechanism"],
                    "pilot": {
                        "max_gpus": 1,
                        "max_examples": 10,
                        "max_seeds": 1,
                        "timeout_sec": 60,
                    },
                    "promotion_rule": "success_probability >= .95",
                    "early_stop_rule": "futility_probability >= .95",
                },
            )
            decision = "promote"
        elif job.kind is JobKind.BUILD:
            (candidate / "main.py").write_text(
                "print('simulated fixture only')\n",
                encoding="utf-8",
            )
            _write_json(
                candidate / "build.json",
                {
                    "files": {"main.py": "print('simulated fixture only')\n"},
                    "commands": {
                        "smoke": f"{sys.executable} main.py",
                        "pilot": f"{sys.executable} main.py",
                        "scale": f"{sys.executable} main.py",
                    },
                },
            )
            decision = "promote"
        elif job.kind in {JobKind.PILOT, JobKind.SCALE}:
            output = candidate / "artifacts" / job.kind.value
            _write_json(
                output / "metrics.json",
                {
                    "result_valid": True,
                    "metrics": {"accuracy": 0.6},
                    "success_probability": 0.99,
                    "decision": "promote",
                },
            )
            _write_json(
                output / "runtime_evidence.json",
                {
                    "model_loaded": "deterministic-test-model",
                    "datasets_loaded": ["synthetic-test-fixture"],
                    "examples_processed": 10,
                    "seeds": [0],
                    "gpu_count": 1,
                    "gate_decision": "promote",
                    "metrics": {"accuracy": 0.6},
                },
            )
            decision = "promote"
        else:
            (candidate / "paper.md").write_text(
                f"# {idea.title}\n\nDeterministic test report.\n",
                encoding="utf-8",
            )
            decision = "complete"
        attempt.status = AttemptStatus.VALIDATING
        attempt.validation = {"ok": True}
        store.save_attempt(attempt)
        store.commit_candidate(attempt)
        return JobOutcome(True, decision, f"{job.kind.value}_simulated", {})
