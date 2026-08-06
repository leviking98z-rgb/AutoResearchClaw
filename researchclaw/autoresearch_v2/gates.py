"""Consequential, evidence-bound decisions made by the decision tier."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .llm import StructuredRole
from .models import IdeaRecord, JobKind
from .protocols import COMPILER_OWNED_DESIGN_FIELDS, design_gate_view

DESIGN_BLOCKER_CODES = frozenset(
    {
        "direct_duplicate",
        "template_mismatch",
        "non_identifiable_contrast",
        "unmeasurable_endpoint",
        "unfalsifiable_hypothesis",
        "claim_scope_exceeds_screening",
        "unsafe_or_unlicensed",
    }
)

DESIGN_BLOCKER_EVIDENCE_PREFIXES = {
    "direct_duplicate": (
        "/idea/novelty_gap",
        "/idea/novelty_evidence/closest_papers",
    ),
    "template_mismatch": (
        "/plan/protocol_template",
        "/plan/research_question",
        "/plan/hypothesis",
    ),
    "non_identifiable_contrast": (
        "/plan/arms",
        "/plan/estimand",
    ),
    "unmeasurable_endpoint": (
        "/plan/primary_metric",
        "/plan/gate_statistic",
        "/plan/resources",
    ),
    "unfalsifiable_hypothesis": (
        "/plan/hypothesis",
        "/plan/research_question",
    ),
    "claim_scope_exceeds_screening": (
        "/plan/pilot_objective",
        "/plan/pilot_claim_scope",
    ),
    "unsafe_or_unlicensed": (
        "/idea/implementation_feasibility",
        "/idea/licensing_feasibility",
        "/plan/resources",
    ),
}


@dataclass(frozen=True, slots=True)
class GateVerdict:
    decision: str
    reason: str
    confidence: float
    risks: tuple[str, ...] = ()
    required_changes: tuple[str, ...] = ()
    tokens: int = 0
    raw: dict[str, Any] | None = None
    blocker_codes: tuple[str, ...] = ()


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
    errors: list[str] = []
    if value.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if "decision" in value:
        errors.append("decision is Controller-derived and must be omitted")
    if "required_changes" in value:
        errors.append("required_changes must be omitted")
    if not str(value.get("reason", "") or "").strip():
        errors.append("missing reason")
    try:
        confidence = float(value.get("confidence", -1))
    except (TypeError, ValueError):
        errors.append("invalid confidence")
    else:
        if not 0 <= confidence <= 1:
            errors.append("confidence must be in [0,1]")
    risks = value.get("risks")
    if not isinstance(risks, list) or not all(
        isinstance(item, str) for item in risks
    ):
        errors.append("risks must be a list of strings")
    blockers = value.get("blockers")
    if not isinstance(blockers, list):
        errors.append("blockers must be a list")
        return errors
    if len(blockers) > 3:
        errors.append("blockers must contain at most 3 entries")
    seen: set[str] = set()
    for index, blocker in enumerate(blockers):
        if not isinstance(blocker, Mapping):
            errors.append(f"blockers[{index}] must be an object")
            continue
        code = str(blocker.get("code", "") or "")
        if code not in DESIGN_BLOCKER_CODES:
            errors.append(f"blockers[{index}].code is not allowed")
            continue
        if code in seen:
            errors.append(f"duplicate blocker code: {code}")
        seen.add(code)
        paths = blocker.get("evidence_paths")
        if not isinstance(paths, list) or not paths or not all(
            isinstance(path, str) and path.startswith("/")
            for path in paths
        ):
            errors.append(
                f"blockers[{index}].evidence_paths must be a non-empty "
                "list of JSON-pointer strings"
            )
        else:
            allowed = DESIGN_BLOCKER_EVIDENCE_PREFIXES[code]
            for path in paths:
                if not any(
                    path == prefix or path.startswith(prefix + "/")
                    for prefix in allowed
                ):
                    errors.append(
                        f"blockers[{index}].evidence_paths contains "
                        f"disallowed path for {code}: {path}"
                    )
        if not str(blocker.get("explanation", "") or "").strip():
            errors.append(f"blockers[{index}].explanation is required")
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
        scientific_plan = design_gate_view(plan)
        idea_view = {
            key: idea.candidate.get(key)
            for key in (
                "title",
                "research_question",
                "falsifiable_hypothesis",
                "novelty_gap",
                "novelty_evidence",
                "implementation_feasibility",
                "licensing_feasibility",
            )
        }
        result = self._design.call(
            f"""\
Audit the scientific semantics of this compiled SCREENING-PILOT.

IDEA:
{json.dumps(idea_view, ensure_ascii=False, indent=2)[:16000]}

SCIENTIFIC PLAN VIEW:
{json.dumps(scientific_plan, ensure_ascii=False, indent=2)[:18000]}

DETERMINISTIC ATTESTATION:
{json.dumps({
    "validate_plan": "passed",
    "compiler_version": 2,
    "compiler_owned_fields": list(COMPILER_OWNED_DESIGN_FIELDS),
}, ensure_ascii=False, indent=2)}

The Controller has already compiled and validated every mechanical field.
You must not block on call arithmetic, sample counts, split identifiers,
screening access flags, seeds, parser/tie/missingness conventions, bootstrap
mechanics, threshold resolution, decision regions, or runtime evidence fields.
Implementation detail, limited sample size, single-model scope, missing extra
ablations, and desirable Scale follow-ups are risks, never blockers.

The only legal blockers are:
- direct_duplicate: supplied prior work directly covers the same mechanism,
  contrast, and endpoint;
- template_mismatch: the scientific mechanism cannot be expressed by the
  selected supported protocol without changing the research question;
- non_identifiable_contrast: the stated arms and estimand scientifically
  confound the claimed causal contrast;
- unmeasurable_endpoint: the endpoint cannot be observed or computed from the
  named resources;
- unfalsifiable_hypothesis: no valid pilot outcome could falsify the claim;
- claim_scope_exceeds_screening: the Pilot itself claims confirmatory or
  generalized evidence rather than a coarse go/no-go signal;
- unsafe_or_unlicensed: execution has a hard safety, data, or license blocker.

Use only evidence paths allowed by the schema. If none of those blockers is
present, return an empty blockers list. Do not return a decision and do not
request revisions. The Controller derives reject when blockers are non-empty
and promote when blockers are empty.

Return:
{{
  "schema_version": 2,
  "reason": "concise evidence-bound summary",
  "confidence": 0.0,
  "blockers": [
    {{
      "code": "one allowed blocker code",
      "evidence_paths": ["/plan/estimand"],
      "explanation": "why the cited evidence is terminal"
    }}
  ],
  "risks": ["non-blocking limitation or Scale follow-up"]
}}
""",
            max_tokens=3000,
            temperature=0.05,
        )
        return _design_verdict(result.value, result.total_tokens)

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
unverifiable outcomes. First audit runtime_evidence.evidence_valid,
gate_statistic_defined, criterion_results, and gate_decision against the
compiled decision_contract. Do not reinterpret raw metric_direction as the
promotion direction. Protocol-invalid evidence may retry; every valid outcome
that fails any promotion criterion, including an undefined gate statistic,
must reject or complete_negative rather than retry. Preserve a valid
informative null as complete_negative.

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


def _design_verdict(value: Mapping[str, Any], tokens: int) -> GateVerdict:
    blockers = value.get("blockers", [])
    codes = tuple(
        str(item["code"]) for item in blockers if isinstance(item, Mapping)
    )
    decision = "reject" if codes else "promote"
    raw = {
        **dict(value),
        "decision": decision,
        "blocker_codes": list(codes),
        "required_changes": [],
    }
    return GateVerdict(
        decision=decision,
        reason=str(value["reason"]),
        confidence=float(value["confidence"]),
        risks=tuple(str(item) for item in value.get("risks", [])),
        required_changes=(),
        tokens=max(0, int(tokens)),
        raw=raw,
        blocker_codes=codes,
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
    blocker = {
        "code": "novelty_evidence_missing",
        "evidence_paths": ["/idea/novelty_evidence"],
        "explanation": reason,
    }
    raw = {
        "schema_version": 2,
        "decision": "reject",
        "reason": reason,
        "confidence": 1.0,
        "blocker_codes": ["novelty_evidence_missing"],
        "blockers": [blocker],
        "risks": ["unsupported novelty claim"],
        "required_changes": [],
    }
    return GateVerdict(
        decision="reject",
        reason=reason,
        confidence=1.0,
        risks=("unsupported novelty claim",),
        required_changes=(),
        raw=raw,
        blocker_codes=("novelty_evidence_missing",),
    )


def design_preflight(idea: IdeaRecord) -> GateVerdict | None:
    """Public deterministic preflight used before spending a worker-model call."""

    return _design_preflight(idea)
