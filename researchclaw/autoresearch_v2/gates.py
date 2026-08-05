"""Consequential, evidence-bound decisions made by the decision tier."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .llm import StructuredRole
from .models import IdeaRecord, JobKind


@dataclass(frozen=True, slots=True)
class GateVerdict:
    decision: str
    reason: str
    confidence: float
    risks: tuple[str, ...] = ()
    required_changes: tuple[str, ...] = ()
    tokens: int = 0
    raw: dict[str, Any] | None = None


class DecisionGate(Protocol):
    def review_design(
        self,
        idea: IdeaRecord,
        plan: Mapping[str, Any],
    ) -> GateVerdict: ...

    def review_experiment(
        self,
        idea: IdeaRecord,
        *,
        kind: JobKind,
        plan: Mapping[str, Any],
        metrics: Mapping[str, Any],
        runtime_evidence: Mapping[str, Any],
    ) -> GateVerdict: ...

    def review_report(
        self,
        idea: IdeaRecord,
        report: Mapping[str, Any],
        evidence: list[dict[str, Any]],
    ) -> GateVerdict: ...


def _validate_design(value: Mapping[str, Any]) -> list[str]:
    errors = _validate_common(value, {"promote", "retry", "reject"})
    if not isinstance(value.get("required_changes"), list):
        errors.append("required_changes must be a list")
    return errors


def _validate_experiment(value: Mapping[str, Any]) -> list[str]:
    return _validate_common(
        value,
        {
            "promote",
            "retry",
            "reject",
            "complete_negative",
        },
    )


def _validate_report(value: Mapping[str, Any]) -> list[str]:
    return _validate_common(value, {"complete", "retry", "reject"})


def _validate_common(
    value: Mapping[str, Any],
    decisions: set[str],
) -> list[str]:
    errors: list[str] = []
    if str(value.get("decision", "") or "") not in decisions:
        errors.append("invalid decision")
    if not str(value.get("reason", "") or "").strip():
        errors.append("missing reason")
    try:
        confidence = float(value.get("confidence", -1))
    except (TypeError, ValueError):
        errors.append("invalid confidence")
    else:
        if not 0 <= confidence <= 1:
            errors.append("confidence must be in [0,1]")
    if not isinstance(value.get("risks"), list):
        errors.append("risks must be a list")
    return errors


class LLMDecisionGate:
    """Decision model emits JSON verdicts and is never allowed to edit code."""

    def __init__(self, *, client: Any) -> None:
        system = (
            "You are the final scientific review board. Return only JSON. "
            "You may make a decision, but you may not write or edit code. "
            "Fail closed when evidence is missing or internally inconsistent."
        )
        self._design = StructuredRole(
            client=client,
            system=system,
            validator=_validate_design,
        )
        self._experiment = StructuredRole(
            client=client,
            system=system,
            validator=_validate_experiment,
        )
        self._report = StructuredRole(
            client=client,
            system=system,
            validator=_validate_report,
        )

    def review_design(
        self,
        idea: IdeaRecord,
        plan: Mapping[str, Any],
    ) -> GateVerdict:
        preflight = _design_preflight(idea)
        if preflight is not None:
            return preflight
        result = self._design.call(
            f"""\
Review this preregistered design before implementation.

IDEA:
{json.dumps(idea.candidate, ensure_ascii=False, indent=2)[:24000]}

PLAN:
{json.dumps(dict(plan), ensure_ascii=False, indent=2)[:24000]}

This plan is explicitly a SCREENING PILOT, not the confirmatory paper study.
Judge whether it can produce a valid, inexpensive go/no-go decision,
validate the protocol, or expose a coarse signal worth scaling. Do not require
16-50 examples or one seed to precisely establish the eventual paper-level
effect. Instead require:
- a precise unit of analysis and paired/comparable outcomes;
- 2-3 primary arms with an independent no-self-improvement control;
- exact, internally consistent sample/call arithmetic;
- a threshold no finer than finite-sample metric resolution;
- disjoint promotion, retry/invalidity, and rejection/futility regions;
- no adaptation, calibration, selection, or memory writing on heldout data;
- an explicit confirmatory follow-up with more examples/seeds at Scale;
- claims explicitly limited to screening feasibility or a coarse signal.

Still fail closed on leakage, missing controls, impossible compute, ambiguous
outcomes, manipulated metrics, or a pilot framed as confirmatory evidence.
Check novelty evidence, falsifiability, controls, metric alignment, screening
discrimination, and compute feasibility. Grounded closest-paper evidence has
already passed deterministic preflight: do not require an exhaustive
bibliographic review merely to run a screening pilot. Treat incomplete
coverage as a recorded risk/follow-up unless the supplied evidence reveals a
direct duplicate or the claimed gap is contradicted.

Return:
{{
  "decision": "promote|retry|reject",
  "reason": "specific evidence-bound reason",
  "confidence": 0.0,
  "risks": ["..."],
  "required_changes": ["..."]
}}
""",
            max_tokens=5000,
            temperature=0.05,
        )
        return _verdict(result.value, result.total_tokens)

    def review_experiment(
        self,
        idea: IdeaRecord,
        *,
        kind: JobKind,
        plan: Mapping[str, Any],
        metrics: Mapping[str, Any],
        runtime_evidence: Mapping[str, Any],
    ) -> GateVerdict:
        result = self._experiment.call(
            f"""\
Decide whether this {kind.value} run should advance.

IDEA:
{json.dumps(idea.candidate, ensure_ascii=False, indent=2)[:16000]}

PREREGISTERED PLAN:
{json.dumps(dict(plan), ensure_ascii=False, indent=2)[:16000]}

MEASURED METRICS:
{json.dumps(dict(metrics), ensure_ascii=False, indent=2)[:16000]}

RUNTIME EVIDENCE:
{json.dumps(dict(runtime_evidence), ensure_ascii=False, indent=2)[:16000]}

Use only measured evidence. Reject leakage, missing controls, synthetic or
unverifiable outcomes. Preserve a valid informative null as complete_negative.

Return:
{{
  "decision": "promote|retry|reject|complete_negative",
  "reason": "specific evidence-bound reason",
  "confidence": 0.0,
  "risks": ["..."],
  "required_changes": ["..."]
}}
""",
            max_tokens=5000,
            temperature=0.05,
        )
        return _verdict(result.value, result.total_tokens)

    def review_report(
        self,
        idea: IdeaRecord,
        report: Mapping[str, Any],
        evidence: list[dict[str, Any]],
    ) -> GateVerdict:
        result = self._report.call(
            f"""\
Audit the final claim-evidence package.

IDEA:
{json.dumps(idea.candidate, ensure_ascii=False, indent=2)[:12000]}

REPORT:
{json.dumps(dict(report), ensure_ascii=False, indent=2)[:24000]}

MEASURED EVIDENCE:
{json.dumps(evidence, ensure_ascii=False, indent=2)[:24000]}

Every measured claim must name an existing evidence path and must not be
stronger than the evidence. Hypotheses must be labelled as hypotheses.

Return:
{{
  "decision": "complete|retry|reject",
  "reason": "specific evidence-bound reason",
  "confidence": 0.0,
  "risks": ["..."],
  "required_changes": ["..."]
}}
""",
            max_tokens=5000,
            temperature=0.05,
        )
        return _verdict(result.value, result.total_tokens)


def _verdict(value: Mapping[str, Any], tokens: int) -> GateVerdict:
    return GateVerdict(
        decision=str(value["decision"]),
        reason=str(value["reason"]),
        confidence=float(value["confidence"]),
        risks=tuple(str(item) for item in value.get("risks", [])),
        required_changes=tuple(
            str(item) for item in value.get("required_changes", [])
        ),
        tokens=max(0, int(tokens)),
        raw=dict(value),
    )


def _design_preflight(idea: IdeaRecord) -> GateVerdict | None:
    evidence = idea.candidate.get("novelty_evidence", {})
    if not isinstance(evidence, Mapping):
        reason = "novelty evidence is missing"
    elif evidence.get("available") is not True:
        reason = "literature service was unavailable during novelty review"
    elif not evidence.get("closest_papers"):
        reason = (
            "novelty search returned no grounded closest papers; refresh "
            "focused literature queries before spending a design-model call"
        )
    else:
        return None
    raw = {
        "decision": "reject",
        "reason": reason,
        "confidence": 1.0,
        "risks": ["unsupported novelty claim"],
        "required_changes": [
            "regenerate or refresh the Idea with grounded closest prior work"
        ],
    }
    return GateVerdict(
        decision="reject",
        reason=reason,
        confidence=1.0,
        risks=("unsupported novelty claim",),
        required_changes=(
            "regenerate or refresh the Idea with grounded closest prior work",
        ),
        raw=raw,
    )


def design_preflight(idea: IdeaRecord) -> GateVerdict | None:
    """Public deterministic preflight used before spending a worker-model call."""

    return _design_preflight(idea)
