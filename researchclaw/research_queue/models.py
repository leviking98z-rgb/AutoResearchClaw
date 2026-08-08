"""Small domain model for the Continuous Research Queue prototype."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class IdeaStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    CONCLUDED = "concluded"
    QUARANTINED = "quarantined"


class Conclusion(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"
    NOT_ADMITTED = "not_admitted"
    CANCELLED = "cancelled"


class BudgetLevel(str, Enum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"

    def next(self) -> BudgetLevel | None:
        return {
            BudgetLevel.B0: BudgetLevel.B1,
            BudgetLevel.B1: BudgetLevel.B2,
            BudgetLevel.B2: None,
        }[self]


class ReviewAction(str, Enum):
    RUN_MORE = "run_more"
    ESCALATE = "escalate"
    REVISE = "revise"
    CONCLUDE = "conclude"


class RunStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MetricRelation(str, Enum):
    """Machine-executable relation for a secondary metric."""

    NO_WORSE = "no_worse"
    EQUAL = "equal"


@dataclass(frozen=True, slots=True)
class MetricGuardrail:
    """One deterministic metric constraint in a ResearchSpec."""

    metric: str
    direction: MetricDirection
    relation: MetricRelation = MetricRelation.NO_WORSE
    tolerance: float = 0.0
    require_effect_ci: bool = False
    per_pair: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "relation": self.relation.value,
            "tolerance": self.tolerance,
            "require_effect_ci": self.require_effect_ci,
            "per_pair": self.per_pair,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MetricGuardrail:
        return cls(
            metric=str(value.get("metric", "") or "").strip(),
            direction=MetricDirection(
                str(
                    value.get(
                        "direction",
                        MetricDirection.MAXIMIZE.value,
                    )
                    or MetricDirection.MAXIMIZE.value
                ).strip()
            ),
            relation=MetricRelation(
                str(
                    value.get(
                        "relation",
                        MetricRelation.NO_WORSE.value,
                    )
                    or MetricRelation.NO_WORSE.value
                ).strip()
            ),
            tolerance=max(0.0, float(value.get("tolerance", 0.0) or 0.0)),
            require_effect_ci=bool(value.get("require_effect_ci", False)),
            per_pair=bool(value.get("per_pair", False)),
        )


@dataclass(slots=True)
class ResearchSpec:
    """Executable scientific contract attached to one Idea."""

    question: str
    hypothesis: str
    treatment: str
    control: str
    primary_metric: str
    metric_direction: MetricDirection
    guardrails: tuple[str, ...] = ()
    validity_conditions: tuple[str, ...] = ()
    compute_matching: tuple[str, ...] = ()
    stopping_rules: tuple[str, ...] = ()
    benchmark_id: str = ""
    treatment_api: str = ""
    minimum_effect: float = 0.0
    primary_requires_effect_ci: bool = False
    guardrail_metrics: tuple[MetricGuardrail, ...] = ()
    minimum_pairs: int = 1
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["metric_direction"] = self.metric_direction.value
        value["guardrail_metrics"] = [item.to_dict() for item in self.guardrail_metrics]
        for field_name in (
            "guardrails",
            "validity_conditions",
            "compute_matching",
            "stopping_rules",
        ):
            value[field_name] = list(value[field_name])
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ResearchSpec:
        def strings(name: str) -> tuple[str, ...]:
            raw = value.get(name, ())
            if isinstance(raw, str):
                return (raw.strip(),) if raw.strip() else ()
            return tuple(str(item).strip() for item in (raw or ()) if str(item).strip())

        direction_raw = str(
            value.get("metric_direction", MetricDirection.MAXIMIZE.value)
            or MetricDirection.MAXIMIZE.value
        ).strip()
        guardrail_metrics_raw = value.get("guardrail_metrics", ())
        if isinstance(guardrail_metrics_raw, Mapping):
            guardrail_metrics_raw = (guardrail_metrics_raw,)
        return cls(
            question=str(value.get("question", "") or "").strip(),
            hypothesis=str(value.get("hypothesis", "") or "").strip(),
            treatment=str(value.get("treatment", "") or "").strip(),
            control=str(value.get("control", "") or "").strip(),
            primary_metric=str(value.get("primary_metric", "") or "").strip(),
            metric_direction=MetricDirection(direction_raw),
            guardrails=strings("guardrails"),
            validity_conditions=strings("validity_conditions"),
            compute_matching=strings("compute_matching"),
            stopping_rules=strings("stopping_rules"),
            benchmark_id=str(value.get("benchmark_id", "") or "").strip(),
            treatment_api=str(value.get("treatment_api", "") or "").strip(),
            minimum_effect=max(
                0.0,
                float(value.get("minimum_effect", 0.0) or 0.0),
            ),
            primary_requires_effect_ci=bool(
                value.get("primary_requires_effect_ci", False)
            ),
            guardrail_metrics=tuple(
                MetricGuardrail.from_mapping(item)
                for item in (guardrail_metrics_raw or ())
                if isinstance(item, Mapping)
            ),
            minimum_pairs=max(
                1,
                int(value.get("minimum_pairs", 1) or 1),
            ),
            confidence_level=min(
                0.999,
                max(
                    0.5,
                    float(value.get("confidence_level", 0.95) or 0.95),
                ),
            ),
        )


@dataclass(slots=True)
class IdeaProposal:
    title: str
    question: str
    hypothesis: str
    treatment: str
    control: str
    primary_metric: str
    research_spec: ResearchSpec | None = None
    tags: tuple[str, ...] = ()
    priority: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["research_spec"] = (
            self.research_spec.to_dict() if self.research_spec is not None else None
        )
        value["tags"] = list(self.tags)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> IdeaProposal:
        tags_raw = value.get("tags", ())
        if isinstance(tags_raw, str):
            tags = (tags_raw.strip(),) if tags_raw.strip() else ()
        else:
            tags = tuple(
                str(item).strip() for item in (tags_raw or ()) if str(item).strip()
            )
        spec_raw = value.get("research_spec")
        return cls(
            title=str(value.get("title", "") or "").strip(),
            question=str(value.get("question", "") or "").strip(),
            hypothesis=str(value.get("hypothesis", "") or "").strip(),
            treatment=str(value.get("treatment", "") or "").strip(),
            control=str(value.get("control", "") or "").strip(),
            primary_metric=str(value.get("primary_metric", "") or "").strip(),
            research_spec=(
                ResearchSpec.from_mapping(spec_raw)
                if isinstance(spec_raw, Mapping)
                else None
            ),
            tags=tags,
            priority=float(value.get("priority", 0.5) or 0.5),
            metadata=dict(value.get("metadata", {}) or {}),
        )


@dataclass(slots=True)
class GenerationBatch:
    ideas: list[IdeaProposal]
    usage: dict[str, Any] = field(default_factory=dict)
    exhausted: bool = False


@dataclass(slots=True)
class IdeaRecord:
    idea_id: str
    title: str
    question: str
    hypothesis: str
    treatment: str
    control: str
    primary_metric: str
    research_spec: ResearchSpec | None = None
    tags: tuple[str, ...] = ()
    priority: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    status: IdeaStatus = IdeaStatus.CANDIDATE
    conclusion: Conclusion | None = None
    current_revision: int = 0
    current_budget: BudgetLevel = BudgetLevel.B0
    next_action: str = "prepare"
    step_count: int = 0
    infra_failures: int = 0
    total_tokens: int = 0
    gpu_seconds: float = 0.0
    last_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def from_proposal(cls, proposal: IdeaProposal) -> IdeaRecord:
        return cls(
            idea_id=new_id("idea"),
            title=proposal.title,
            question=proposal.question,
            hypothesis=proposal.hypothesis,
            treatment=proposal.treatment,
            control=proposal.control,
            primary_metric=proposal.primary_metric,
            research_spec=proposal.research_spec,
            tags=proposal.tags,
            priority=max(0.0, min(1.0, proposal.priority)),
            metadata=proposal.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["research_spec"] = (
            self.research_spec.to_dict() if self.research_spec is not None else None
        )
        value["tags"] = list(self.tags)
        value["status"] = self.status.value
        value["conclusion"] = (
            self.conclusion.value if self.conclusion is not None else None
        )
        value["current_budget"] = self.current_budget.value
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> IdeaRecord:
        conclusion_raw = value.get("conclusion")
        spec_raw = value.get("research_spec")
        return cls(
            idea_id=str(value["idea_id"]),
            title=str(value.get("title", "") or ""),
            question=str(value.get("question", "") or ""),
            hypothesis=str(value.get("hypothesis", "") or ""),
            treatment=str(value.get("treatment", "") or ""),
            control=str(value.get("control", "") or ""),
            primary_metric=str(value.get("primary_metric", "") or ""),
            research_spec=(
                ResearchSpec.from_mapping(spec_raw)
                if isinstance(spec_raw, Mapping)
                else None
            ),
            tags=tuple(str(item) for item in value.get("tags", ()) or ()),
            priority=float(value.get("priority", 0.5) or 0.5),
            metadata=dict(value.get("metadata", {}) or {}),
            status=IdeaStatus(str(value.get("status", IdeaStatus.CANDIDATE.value))),
            conclusion=(
                Conclusion(str(conclusion_raw))
                if conclusion_raw not in {None, ""}
                else None
            ),
            current_revision=int(value.get("current_revision", 0) or 0),
            current_budget=BudgetLevel(
                str(value.get("current_budget", BudgetLevel.B0.value))
            ),
            next_action=str(value.get("next_action", "prepare") or ""),
            step_count=int(value.get("step_count", 0) or 0),
            infra_failures=int(value.get("infra_failures", 0) or 0),
            total_tokens=int(value.get("total_tokens", 0) or 0),
            gpu_seconds=float(value.get("gpu_seconds", 0.0) or 0.0),
            last_reason=str(value.get("last_reason", "") or ""),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass(slots=True)
class PreparedRevision:
    revision: int
    command: tuple[str, ...]
    requested_gpus: int
    timeout_sec: float
    plan: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "command": list(self.command),
            "requested_gpus": self.requested_gpus,
            "timeout_sec": self.timeout_sec,
            "plan": self.plan,
            "usage": self.usage,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PreparedRevision:
        return cls(
            revision=int(value["revision"]),
            command=tuple(str(item) for item in value.get("command", ()) or ()),
            requested_gpus=max(
                0,
                int(value.get("requested_gpus", 0) or 0),
            ),
            timeout_sec=max(
                1.0,
                float(value.get("timeout_sec", 300.0) or 300.0),
            ),
            plan=dict(value.get("plan", {}) or {}),
            usage=dict(value.get("usage", {}) or {}),
        )


@dataclass(slots=True)
class RunResult:
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    returncode: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunRecord:
    run_id: str
    idea_id: str
    revision: int
    budget: BudgetLevel
    requested_gpus: int
    timeout_sec: float
    command: tuple[str, ...]
    output_dir: str
    status: RunStatus = RunStatus.WAITING
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: str = field(default_factory=utc_now)
    started_at: str = ""
    finished_at: str = ""
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["budget"] = self.budget.value
        value["command"] = list(self.command)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RunRecord:
        return cls(
            run_id=str(value["run_id"]),
            idea_id=str(value["idea_id"]),
            revision=int(value.get("revision", 0) or 0),
            budget=BudgetLevel(str(value.get("budget", "B0"))),
            requested_gpus=int(value.get("requested_gpus", 0) or 0),
            timeout_sec=float(value.get("timeout_sec", 300.0) or 300.0),
            command=tuple(str(item) for item in value.get("command", ()) or ()),
            output_dir=str(value.get("output_dir", "") or ""),
            status=RunStatus(str(value.get("status", RunStatus.WAITING.value))),
            result=dict(value.get("result", {}) or {}),
            error=str(value.get("error", "") or ""),
            created_at=str(value.get("created_at", utc_now())),
            started_at=str(value.get("started_at", "") or ""),
            finished_at=str(value.get("finished_at", "") or ""),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass(slots=True)
class ReviewDecision:
    action: ReviewAction
    reason: str
    conclusion: Conclusion | None = None
    next_budget: BudgetLevel | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "conclusion": (
                self.conclusion.value if self.conclusion is not None else None
            ),
            "next_budget": (
                self.next_budget.value if self.next_budget is not None else None
            ),
            "usage": self.usage,
        }
