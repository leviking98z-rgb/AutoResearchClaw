"""Idea generation, experiment preparation, and evidence review workers."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from researchclaw.autoresearch_v2.llm import StructuredRole

from .config import BudgetSpec, ResearchQueueConfig
from .models import (
    BudgetLevel,
    Conclusion,
    GenerationBatch,
    IdeaProposal,
    IdeaRecord,
    PreparedRevision,
    ReviewAction,
    ReviewDecision,
    RunRecord,
)


def usage_from_result(result: Any) -> dict[str, Any]:
    return {
        "model": str(getattr(result, "model", "") or ""),
        "prompt_tokens": int(getattr(result, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(result, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(result, "total_tokens", 0) or 0),
    }


class IdeaProducer(Protocol):
    def generate(
        self,
        count: int,
        *,
        existing: Sequence[IdeaRecord],
    ) -> GenerationBatch: ...


class PreparationWorker(Protocol):
    def prepare(
        self,
        idea: IdeaRecord,
        *,
        revision: int,
        budget: BudgetSpec,
        previous_revision: Mapping[str, Any] | None,
        feedback: str,
    ) -> PreparedRevision: ...


class ReviewWorker(Protocol):
    def review(
        self,
        idea: IdeaRecord,
        *,
        run: RunRecord,
        history: Sequence[RunRecord],
        limits: Mapping[str, Any],
    ) -> ReviewDecision: ...


def _proposal_errors(value: Mapping[str, Any]) -> list[str]:
    ideas = value.get("ideas")
    if not isinstance(ideas, list) or not ideas:
        return ["ideas must be a non-empty list"]
    errors: list[str] = []
    for index, item in enumerate(ideas):
        if not isinstance(item, Mapping):
            errors.append(f"ideas[{index}] must be an object")
            continue
        for field in (
            "title",
            "question",
            "hypothesis",
            "treatment",
            "control",
            "primary_metric",
        ):
            if not str(item.get(field, "") or "").strip():
                errors.append(f"ideas[{index}] missing {field}")
    return errors


def _prepare_errors(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not str(value.get("method_summary", "") or "").strip():
        errors.append("missing method_summary")
    if not str(value.get("treatment", "") or "").strip():
        errors.append("missing treatment")
    if not str(value.get("control", "") or "").strip():
        errors.append("missing control")
    if not str(value.get("primary_metric", "") or "").strip():
        errors.append("missing primary_metric")
    command = value.get("command")
    if not isinstance(command, list) or not command:
        errors.append("command must be a non-empty argv list")
    elif any(not str(item).strip() for item in command):
        errors.append("command contains an empty argv item")
    source_files = value.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        errors.append("source_files must be a non-empty object")
    elif not any(str(path).endswith(".py") for path in source_files):
        errors.append("source_files must contain a Python file")
    requested_gpus = int(value.get("requested_gpus", 0) or 0)
    if requested_gpus < 0:
        errors.append("requested_gpus cannot be negative")
    return errors


def _review_errors(value: Mapping[str, Any]) -> list[str]:
    try:
        action = ReviewAction(str(value.get("action", "")))
    except ValueError:
        return ["invalid action"]
    errors: list[str] = []
    if not str(value.get("reason", "") or "").strip():
        errors.append("missing reason")
    if action is ReviewAction.CONCLUDE:
        try:
            Conclusion(str(value.get("conclusion", "")))
        except ValueError:
            errors.append("conclude requires a valid conclusion")
    if action is ReviewAction.ESCALATE:
        try:
            BudgetLevel(str(value.get("next_budget", "")))
        except ValueError:
            errors.append("escalate requires a valid next_budget")
    return errors


class LLMIdeaProducer:
    def __init__(self, *, client: Any, brief: str) -> None:
        self.brief = brief
        self.role = StructuredRole(
            client=client,
            system=(
                "You produce diverse, falsifiable, compute-bounded research "
                "ideas. Return one JSON object and no prose."
            ),
            validator=_proposal_errors,
            max_attempts=2,
        )

    def generate(
        self,
        count: int,
        *,
        existing: Sequence[IdeaRecord],
    ) -> GenerationBatch:
        existing_titles = [idea.title for idea in existing[-30:]]
        result = self.role.call(
            f"""
RESEARCH BRIEF:
{self.brief}

Generate exactly {count} new research ideas.
Avoid these existing titles:
{json.dumps(existing_titles, ensure_ascii=False)}

Each idea must have:
- title
- question
- falsifiable hypothesis
- compute-matched treatment
- compute-matched control
- one primary metric
- optional short tags
- priority from 0 to 1

Return:
{{
  "ideas": [
    {{
      "title": "...",
      "question": "...",
      "hypothesis": "...",
      "treatment": "...",
      "control": "...",
      "primary_metric": "...",
      "tags": ["..."],
      "priority": 0.7
    }}
  ]
}}
""".strip(),
            max_tokens=5000,
            temperature=0.7,
        )
        proposals = [
            IdeaProposal.from_mapping(item)
            for item in result.value["ideas"][:count]
            if isinstance(item, Mapping)
        ]
        return GenerationBatch(
            ideas=proposals,
            usage=usage_from_result(result),
        )


class LLMPreparationWorker:
    def __init__(
        self,
        *,
        client: Any,
        python_executable: str,
        max_gpus_per_run: int,
        max_tokens: int = 8000,
    ) -> None:
        self.python_executable = python_executable
        self.max_gpus_per_run = max(0, int(max_gpus_per_run))
        self.max_tokens = max(1000, int(max_tokens))
        self.role = StructuredRole(
            client=client,
            system=(
                "You are a research engineer building a minimal executable "
                "experiment. Return a complete JSON project, not a patch."
            ),
            validator=_prepare_errors,
            max_attempts=2,
        )

    def prepare(
        self,
        idea: IdeaRecord,
        *,
        revision: int,
        budget: BudgetSpec,
        previous_revision: Mapping[str, Any] | None,
        feedback: str,
    ) -> PreparedRevision:
        result = self.role.call(
            f"""
IDEA:
{json.dumps(idea.to_dict(), ensure_ascii=False, indent=2)}

REVISION: {revision}
BUDGET TEMPLATE:
{json.dumps(budget.to_dict(), ensure_ascii=False, indent=2)}

PREVIOUS REVISION:
{json.dumps(previous_revision or {}, ensure_ascii=False, indent=2)}

REVIEW FEEDBACK:
{feedback or "none"}

Build the cheapest executable experiment that compares Treatment and Control.
The project runs from its revision directory. It MUST write result.json in the
directory supplied by environment variable RESEARCH_QUEUE_OUTPUT_DIR.
result.json must contain:
{{
  "status": "ok",
  "metrics": {{
    "primary_value": number,
    "treatment_value": number,
    "control_value": number,
    "effect": number
  }},
  "artifacts": []
}}

Use only declared local dependencies. Do not download huge models or datasets
inside the experiment unless the research brief explicitly requires them.
Return:
{{
  "method_summary": "...",
  "treatment": "...",
  "control": "...",
  "primary_metric": "...",
  "requested_gpus": 0,
  "timeout_sec": 120,
  "command": ["{self.python_executable}", "experiment.py"],
  "source_files": {{
    "experiment.py": "complete source"
  }},
  "plan": {{
    "cheap_test": "...",
    "success_signal": "...",
    "limitations": ["..."]
  }}
}}
""".strip(),
            max_tokens=self.max_tokens,
            temperature=0.25,
        )
        value = result.value
        requested_gpus = min(
            self.max_gpus_per_run,
            max(0, int(value.get("requested_gpus", budget.gpus) or 0)),
        )
        plan = {
            "method_summary": str(value.get("method_summary", "")),
            "treatment": str(value.get("treatment", "")),
            "control": str(value.get("control", "")),
            "primary_metric": str(value.get("primary_metric", "")),
            "source_files": {
                str(path): str(content)
                for path, content in dict(value.get("source_files", {}) or {}).items()
            },
            **dict(value.get("plan", {}) or {}),
        }
        command = tuple(str(item) for item in value.get("command", ()) or ())
        return PreparedRevision(
            revision=revision,
            command=command,
            requested_gpus=requested_gpus,
            timeout_sec=min(
                max(1.0, float(value.get("timeout_sec", budget.timeout_sec))),
                max(1.0, budget.timeout_sec),
            ),
            plan=plan,
            usage=usage_from_result(result),
        )


class LLMReviewWorker:
    def __init__(self, *, client: Any) -> None:
        self.role = StructuredRole(
            client=client,
            system=(
                "You are a conservative scientific decision maker. Use only "
                "the supplied evidence and return one bounded JSON decision."
            ),
            validator=_review_errors,
            max_attempts=2,
        )

    def review(
        self,
        idea: IdeaRecord,
        *,
        run: RunRecord,
        history: Sequence[RunRecord],
        limits: Mapping[str, Any],
    ) -> ReviewDecision:
        history_payload = [
            {
                "budget": item.budget.value,
                "revision": item.revision,
                "status": item.status.value,
                "result": item.result,
                "error": item.error,
            }
            for item in history
        ]
        result = self.role.call(
            f"""
IDEA:
{json.dumps(idea.to_dict(), ensure_ascii=False, indent=2)}

LATEST RUN:
{json.dumps(run.to_dict(), ensure_ascii=False, indent=2)}

RUN HISTORY:
{json.dumps(history_payload, ensure_ascii=False, indent=2)}

LIMITS:
{json.dumps(dict(limits), ensure_ascii=False, indent=2)}

Choose exactly one:
- run_more: repeat the current budget only when another independent run is
  necessary and allowed.
- escalate: move exactly B0->B1 or B1->B2 when valid evidence warrants more
  compute.
- revise: prepare a new immutable revision when the scientific implementation
  should change.
- conclude: positive, negative, or inconclusive.

Return:
{{
  "action": "run_more|escalate|revise|conclude",
  "reason": "...",
  "next_budget": "B1|B2|null",
  "conclusion": "positive|negative|inconclusive|null"
}}
""".strip(),
            max_tokens=1800,
            temperature=0.1,
        )
        value = result.value
        action = ReviewAction(str(value["action"]))
        conclusion_raw = value.get("conclusion")
        next_budget_raw = value.get("next_budget")
        return ReviewDecision(
            action=action,
            reason=str(value.get("reason", "") or ""),
            conclusion=(
                Conclusion(str(conclusion_raw))
                if conclusion_raw not in {None, "", "null"}
                else None
            ),
            next_budget=(
                BudgetLevel(str(next_budget_raw))
                if next_budget_raw not in {None, "", "null"}
                else None
            ),
            usage=usage_from_result(result),
        )


class StaticIdeaProducer:
    """Deterministic finite or cycling Idea source for prototype tests."""

    def __init__(
        self,
        proposals: Sequence[IdeaProposal | Mapping[str, Any]],
        *,
        cycle: bool = False,
    ) -> None:
        self._proposals = [
            item if isinstance(item, IdeaProposal) else IdeaProposal.from_mapping(item)
            for item in proposals
        ]
        self._cursor = 0
        self._emitted = 0
        self._cycle = bool(cycle)

    def generate(
        self,
        count: int,
        *,
        existing: Sequence[IdeaRecord],
    ) -> GenerationBatch:
        del existing
        output: list[IdeaProposal] = []
        while len(output) < count and self._proposals:
            if self._cursor >= len(self._proposals):
                if not self._cycle:
                    break
                self._cursor = 0
            base = self._proposals[self._cursor]
            self._cursor += 1
            self._emitted += 1
            suffix = self._emitted if self._cycle else 0
            output.append(
                IdeaProposal(
                    title=(f"{base.title} #{suffix}" if suffix else base.title),
                    question=base.question,
                    hypothesis=base.hypothesis,
                    treatment=base.treatment,
                    control=base.control,
                    primary_metric=base.primary_metric,
                    tags=base.tags,
                    priority=base.priority,
                    metadata=dict(base.metadata),
                )
            )
        exhausted = not self._cycle and self._cursor >= len(self._proposals)
        return GenerationBatch(ideas=output, exhausted=exhausted)


class SimulatedPreparationWorker:
    """Write a tiny real Python experiment with deterministic trajectories."""

    def __init__(self, *, python_executable: str = sys.executable) -> None:
        self.python_executable = python_executable

    def prepare(
        self,
        idea: IdeaRecord,
        *,
        revision: int,
        budget: BudgetSpec,
        previous_revision: Mapping[str, Any] | None,
        feedback: str,
    ) -> PreparedRevision:
        del previous_revision, feedback
        scenario = str(idea.metadata.get("scenario", "positive") or "positive")
        sleep_sec = max(
            0.0,
            float(idea.metadata.get("sleep_sec", 0.01) or 0.0),
        )
        requested = int(idea.metadata.get("requested_gpus", budget.gpus) or 0)
        source = _simulated_experiment_source(
            idea_id=idea.idea_id,
            scenario=scenario,
            revision=revision,
            sleep_sec=sleep_sec,
        )
        return PreparedRevision(
            revision=revision,
            command=(self.python_executable, "experiment.py"),
            requested_gpus=max(0, requested),
            timeout_sec=budget.timeout_sec,
            plan={
                "method_summary": "Deterministic prototype experiment",
                "treatment": idea.treatment,
                "control": idea.control,
                "primary_metric": idea.primary_metric,
                "cheap_test": "Run the deterministic low-budget comparison.",
                "source_files": {"experiment.py": source},
                "scenario": scenario,
            },
        )


class SimulatedReviewWorker:
    """Deterministic review policy exercising all four legal actions."""

    def review(
        self,
        idea: IdeaRecord,
        *,
        run: RunRecord,
        history: Sequence[RunRecord],
        limits: Mapping[str, Any],
    ) -> ReviewDecision:
        scenario = str(idea.metadata.get("scenario", "positive") or "positive")
        if not bool(run.result.get("ok", False)):
            return ReviewDecision(
                action=ReviewAction.REVISE,
                reason="The experiment failed and should be revised.",
            )
        effect = _numeric_effect(run.result)
        budget_runs = sum(
            1
            for item in history
            if item.budget is run.budget
            and item.revision == run.revision
            and bool(item.result.get("ok", False))
        )
        if scenario == "negative":
            return ReviewDecision(
                action=ReviewAction.CONCLUDE,
                reason="The cheap test produced a null or adverse effect.",
                conclusion=Conclusion.NEGATIVE,
            )
        if scenario == "inconclusive":
            return ReviewDecision(
                action=ReviewAction.CONCLUDE,
                reason="The observed effect remains too small to interpret.",
                conclusion=Conclusion.INCONCLUSIVE,
            )
        if scenario == "revise" and idea.current_revision < 2:
            return ReviewDecision(
                action=ReviewAction.REVISE,
                reason="The first revision exposed a design weakness.",
            )
        if scenario == "run_more" and budget_runs < 2:
            return ReviewDecision(
                action=ReviewAction.RUN_MORE,
                reason="One more independent run is needed at this budget.",
            )
        if effect > 0 and run.budget.next() is not None:
            return ReviewDecision(
                action=ReviewAction.ESCALATE,
                reason="Positive signal warrants the next budget.",
                next_budget=run.budget.next(),
            )
        return ReviewDecision(
            action=ReviewAction.CONCLUDE,
            reason="The confirmatory budget retained a positive effect.",
            conclusion=Conclusion.POSITIVE,
        )


def materialize_revision(
    revision_dir: Any,
    prepared: PreparedRevision,
) -> None:
    directory = revision_dir
    directory.mkdir(parents=True, exist_ok=True)
    source_files = dict(prepared.plan.get("source_files", {}) or {})
    for relative, content in source_files.items():
        path = directory / str(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content), encoding="utf-8")
    (directory / "plan.json").write_text(
        json.dumps(prepared.plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "command.json").write_text(
        json.dumps(list(prepared.command), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "revision.json").write_text(
        json.dumps(
            prepared.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _simulated_experiment_source(
    *,
    idea_id: str,
    scenario: str,
    revision: int,
    sleep_sec: float,
) -> str:
    seed = int(
        hashlib.sha256(f"{idea_id}:{scenario}:{revision}".encode()).hexdigest()[:8],
        16,
    )
    base_effect = {
        "positive": 0.12,
        "negative": -0.02,
        "inconclusive": 0.005,
        "revise": -0.01 if revision == 1 else 0.10,
        "run_more": 0.08,
    }.get(scenario, 0.06)
    return f"""\
import json
import os
import pathlib
import random
import time

random.seed({seed})
time.sleep({sleep_sec!r})
budget = os.environ.get("RESEARCH_QUEUE_BUDGET", "B0")
scale = {{"B0": 1.0, "B1": 1.1, "B2": 1.2}}.get(budget, 1.0)
effect = ({base_effect!r} * scale) + random.uniform(-0.002, 0.002)
treatment = 0.5 + effect
control = 0.5
output = pathlib.Path(os.environ["RESEARCH_QUEUE_OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)
payload = {{
    "status": "ok",
    "metrics": {{
        "primary_value": effect,
        "treatment_value": treatment,
        "control_value": control,
        "effect": effect,
    }},
    "artifacts": [],
}}
(output / "result.json").write_text(json.dumps(payload, indent=2))
"""


def _numeric_effect(result: Mapping[str, Any]) -> float:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return 0.0
    for name in ("effect", "primary_value", "delta"):
        value = metrics.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return 0.0


def title_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.casefold())
        if token
    }


def title_similarity(left: str, right: str) -> float:
    a = title_tokens(left)
    b = title_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate_proposal(proposal: IdeaProposal) -> list[str]:
    errors: list[str] = []
    for field in (
        "title",
        "question",
        "hypothesis",
        "treatment",
        "control",
        "primary_metric",
    ):
        if not str(getattr(proposal, field)).strip():
            errors.append(f"missing {field}")
    if proposal.treatment.strip().casefold() == proposal.control.strip().casefold():
        errors.append("treatment and control are identical")
    return errors


def build_workers(
    config: ResearchQueueConfig,
) -> tuple[IdeaProducer, PreparationWorker, ReviewWorker]:
    if config.execution.simulation:
        proposals = [
            {
                "title": "Prototype positive signal",
                "question": "Does the treatment improve the primary metric?",
                "hypothesis": "Treatment has a positive effect.",
                "treatment": "Use the proposed mechanism.",
                "control": "Use the matched baseline.",
                "primary_metric": "effect",
                "metadata": {"scenario": "positive"},
            },
            {
                "title": "Prototype negative signal",
                "question": "Does a weak mechanism improve the metric?",
                "hypothesis": "The weak mechanism has no useful effect.",
                "treatment": "Use the weak mechanism.",
                "control": "Use the matched baseline.",
                "primary_metric": "effect",
                "metadata": {"scenario": "negative"},
            },
            {
                "title": "Prototype revision path",
                "question": "Can a revised implementation recover the signal?",
                "hypothesis": "Revision two is better than revision one.",
                "treatment": "Use the revised mechanism.",
                "control": "Use the matched baseline.",
                "primary_metric": "effect",
                "metadata": {"scenario": "revise"},
            },
            {
                "title": "Prototype repeated run",
                "question": "Does a second run confirm the effect?",
                "hypothesis": "Two independent runs retain the signal.",
                "treatment": "Use the proposed mechanism.",
                "control": "Use the matched baseline.",
                "primary_metric": "effect",
                "metadata": {"scenario": "run_more"},
            },
        ]
        return (
            StaticIdeaProducer(proposals, cycle=True),
            SimulatedPreparationWorker(
                python_executable=config.execution.python_executable
            ),
            SimulatedReviewWorker(),
        )
    from researchclaw.autoresearch_v2.llm import RoleRouter

    router = RoleRouter(
        config.models.researchclaw_config,
        audit_root=config.root / "llm-audit",
        decision_role=config.models.decision_role,
        worker_role=config.models.worker_role,
        utility_role=config.models.utility_role,
    )
    return (
        LLMIdeaProducer(
            client=router.worker,
            brief=config.resolved_brief(),
        ),
        LLMPreparationWorker(
            client=router.worker,
            python_executable=config.execution.python_executable,
            max_gpus_per_run=config.gpu.max_gpus_per_run,
        ),
        LLMReviewWorker(client=router.decision),
    )
