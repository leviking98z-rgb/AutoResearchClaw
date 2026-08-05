"""Pipeline-independent v2 research jobs.

Every job writes only to its immutable attempt directory.  The controller
commits accepted candidates to ``current/`` atomically.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .gates import DecisionGate
from .llm import StructuredRole
from .models import (
    AttemptRecord,
    AttemptStatus,
    IdeaRecord,
    JobKind,
    JobRecord,
)
from .store import V2Store
from .validation import (
    validate_experiment_artifacts,
    validate_python_tree,
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
    ) -> None:
        self.role = role
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
        candidate = store.prepare_candidate(attempt)
        result = self.role.call(
            self._prompt(idea, prior_failure=job.result),
            max_tokens=10000,
            temperature=0.25,
        )
        _write_json(candidate / "plan.json", result.value)
        gate_tokens = 0
        if self.decision_gate is not None:
            verdict = self.decision_gate.review_design(idea, result.value)
            gate_tokens = verdict.tokens
            _write_json(
                candidate / "design_review.json",
                verdict.raw or {
                    "decision": verdict.decision,
                    "reason": verdict.reason,
                    "confidence": verdict.confidence,
                    "risks": list(verdict.risks),
                    "required_changes": list(verdict.required_changes),
                },
            )
            if verdict.decision != "promote":
                attempt.output_manifest = {
                    "files": ["plan.json", "design_review.json"]
                }
                attempt.validation = {
                    "ok": False,
                    "decision_gate": verdict.raw or {},
                }
                attempt.status = AttemptStatus.REJECTED
                attempt.error = verdict.reason
                store.save_attempt(attempt)
                return JobOutcome(
                    verdict.decision == "reject",
                    verdict.decision,
                    verdict.reason,
                    {"decision_gate": verdict.raw or {}},
                    tokens=result.total_tokens + gate_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
        attempt.output_manifest = {"files": ["plan.json"]}
        attempt.validation = {"ok": True}
        attempt.status = AttemptStatus.VALIDATING
        store.save_attempt(attempt)
        store.commit_candidate(attempt)
        return JobOutcome(
            True,
            "promote",
            "design_accepted",
            {"plan_path": str(store.current_dir(idea.idea_id) / "plan.json")},
            tokens=result.total_tokens + gate_tokens,
            elapsed_sec=time.monotonic() - started,
        )

    @staticmethod
    def _prompt(
        idea: IdeaRecord,
        *,
        prior_failure: Mapping[str, Any],
    ) -> str:
        return f"""\
Create an executable preregistered experiment plan for this Idea:
{json.dumps(idea.candidate, ensure_ascii=False, indent=2)}

Prior failed attempt feedback, if any:
{json.dumps(dict(prior_failure), ensure_ascii=False, indent=2)[:8000]}

Return exactly:
{{
  "research_question": "...",
  "hypothesis": "...",
  "primary_metric": "...",
  "metric_direction": "maximize|minimize",
  "datasets": [{{"name": "...", "split_role": "dev|heldout"}}],
  "models": [{{"name": "...", "role": "subject|verifier"}}],
  "baselines": ["include no-self-improvement"],
  "ablations": ["..."],
  "pilot": {{
    "max_gpus": 1, "max_examples": 50, "max_seeds": 1,
    "timeout_sec": 7200
  }},
  "promotion_rule": "numeric evidence threshold",
  "early_stop_rule": "futility/invalidity threshold",
  "required_runtime_evidence": [
    "model_loaded", "datasets_loaded", "examples_processed",
    "gpu_count", "gate_decision", "metrics"
  ]
}}
The plan must be answerable by the cheap pilot. Keep held-out data isolated.
"""


class BuildJobExecutor:
    def __init__(self, role: StructuredRole) -> None:
        self.role = role

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
        candidate = store.snapshot_current(attempt)
        result = self.role.call(
            self._prompt(idea, plan, prior_failure=job.result),
            max_tokens=24000,
            temperature=0.20,
        )
        output = result.value
        for filename, content in output["files"].items():
            target = (candidate / filename).resolve()
            if not target.is_relative_to(candidate.resolve()):
                raise ValueError(f"path traversal: {filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        _write_json(candidate / "build.json", output)
        validation = validate_python_tree(candidate)
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
    "main.py": "complete file",
    "data.py": "complete file",
    "model.py": "complete file"
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
- The runtime must write metrics.json with result_valid and metrics.
- The runtime must write runtime_evidence.json with exact model, datasets,
  examples, seeds, GPU count and any accept/reject/rollback decision.
- Respect plan budgets. Never synthesize scientific outcomes.
"""


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
        command = str(build["commands"][mode])
        artifacts = candidate / "artifacts" / mode
        artifacts.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["AUTORESEARCH_V2_IDEA_ID"] = idea.idea_id
        env["AUTORESEARCH_V2_ATTEMPT_ID"] = attempt.attempt_id
        completed = subprocess.run(
            command,
            shell=True,
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
        attempt.validation = validation
        attempt.output_manifest = {
            "command": command,
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
        _write_json(artifacts / "decision_review.json", gate)
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


class ReportJobExecutor:
    def __init__(
        self,
        role: StructuredRole,
        *,
        decision_gate: DecisionGate | None = None,
    ) -> None:
        self.role = role
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
        context = {
            "idea": idea.to_dict(),
            "plan": _read_json(candidate / "plan.json"),
            "metrics": _collect_metrics(candidate / "artifacts"),
        }
        result = self.role.call(
            self._prompt(context),
            max_tokens=18000,
            temperature=0.20,
        )
        _write_json(candidate / "report.json", result.value)
        (candidate / "paper.md").write_text(
            str(result.value["paper_markdown"]),
            encoding="utf-8",
        )
        gate_tokens = 0
        if self.decision_gate is not None:
            verdict = self.decision_gate.review_report(
                idea,
                result.value,
                context["metrics"],
            )
            gate_tokens = verdict.tokens
            _write_json(
                candidate / "final_review.json",
                verdict.raw or {
                    "decision": verdict.decision,
                    "reason": verdict.reason,
                    "confidence": verdict.confidence,
                    "risks": list(verdict.risks),
                    "required_changes": list(verdict.required_changes),
                },
            )
            if verdict.decision != "complete":
                attempt.status = AttemptStatus.REJECTED
                attempt.error = verdict.reason
                attempt.validation = {
                    "ok": False,
                    "decision_gate": verdict.raw or {},
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
                    {"decision_gate": verdict.raw or {}},
                    tokens=result.total_tokens + gate_tokens,
                    elapsed_sec=time.monotonic() - started,
                )
        attempt.validation = {"ok": True}
        attempt.output_manifest = {"files": ["report.json", "paper.md"]}
        attempt.status = AttemptStatus.VALIDATING
        store.save_attempt(attempt)
        store.commit_candidate(attempt)
        return JobOutcome(
            True,
            "complete",
            "paper_package_generated",
            {"paper": str(store.current_dir(idea.idea_id) / "paper.md")},
            tokens=result.total_tokens + gate_tokens,
            elapsed_sec=time.monotonic() - started,
        )

    @staticmethod
    def _prompt(context: Mapping[str, Any]) -> str:
        return f"""\
Produce an evidence-bounded research report from:
{json.dumps(context, ensure_ascii=False, indent=2)[:40000]}

Return JSON:
{{
  "title": "...",
  "claims": [
    {{"claim": "...", "evidence_paths": ["..."], "strength": "measured|hypothesis"}}
  ],
  "limitations": ["..."],
  "next_experiments": ["..."],
  "paper_markdown": "# Title\\n..."
}}
Do not invent results, citations, or runs. Clearly report negative results.
"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _collect_metrics(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "value": _read_json(path),
        }
        for path in sorted(root.rglob("metrics.json"))
    ]


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
