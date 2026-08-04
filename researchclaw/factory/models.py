"""Versioned domain models for the continuous multi-Idea research factory.

The factory deliberately keeps its durable schemas independent from the
single-run pipeline classes.  A pipeline run is an implementation detail of an
Idea; these records are the control-plane source of truth.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, TypeVar

SCHEMA_VERSION = 1

_SPACE_RE = re.compile(r"\s+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def utc_now() -> str:
    """Return a stable UTC timestamp without depending on RSI internals."""

    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def normalize_title(value: str) -> str:
    """Normalize a title for deterministic duplicate detection."""

    return _SPACE_RE.sub(" ", str(value).strip().casefold())


def stable_key(value: str, *, length: int = 16) -> str:
    """Return a filesystem-safe deterministic key for arbitrary text."""

    normalized = normalize_title(value)
    slug = _SLUG_RE.sub("-", normalized).strip("-")[:40] or "idea"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]
    return f"{slug}-{digest}"


class IdeaStatus(str, Enum):
    CANDIDATE = "candidate"
    SCREENING = "screening"
    BUILDING = "building"
    SMOKE = "smoke"
    PILOT = "pilot"
    VALIDATING = "validating"
    PAPER = "paper"
    REPAIR = "repair"
    PARKED = "parked"
    REJECTED = "rejected"
    FAILED = "failed"
    COMPLETED_NEGATIVE = "completed_negative"
    COMPLETED = "completed"


ACTIVE_IDEA_STATUSES = frozenset(
    {
        IdeaStatus.SCREENING,
        IdeaStatus.BUILDING,
        IdeaStatus.SMOKE,
        IdeaStatus.PILOT,
        IdeaStatus.VALIDATING,
        IdeaStatus.PAPER,
        IdeaStatus.REPAIR,
    }
)
TERMINAL_IDEA_STATUSES = frozenset(
    {
        IdeaStatus.PARKED,
        IdeaStatus.REJECTED,
        IdeaStatus.FAILED,
        IdeaStatus.COMPLETED_NEGATIVE,
        IdeaStatus.COMPLETED,
    }
)


class WorkItemStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    QUEUED = "queued"
    ADMITTED = "admitted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    RETRY_WAIT = "retry_wait"


TERMINAL_WORK_ITEM_STATUSES = frozenset(
    {
        WorkItemStatus.SUCCEEDED,
        WorkItemStatus.FAILED,
        WorkItemStatus.CANCELLED,
        WorkItemStatus.TIMED_OUT,
    }
)


class LeaseStatus(str, Enum):
    REQUESTED = "requested"
    QUEUED = "queued"
    ADMITTED = "admitted"
    RUNNING = "running"
    RELEASED = "released"
    EXPIRED = "expired"


class GateAction(str, Enum):
    CONTINUE = "continue"
    PROMOTE = "promote"
    REPAIR = "repair"
    PARK = "park"
    REJECT = "reject"
    COMPLETE_NEGATIVE = "complete_negative"
    COMPLETE = "complete"


class BudgetTier(str, Enum):
    DESK = "desk"
    SMOKE = "smoke"
    PILOT = "pilot"
    VALIDATION = "validation"
    SCALE = "scale"
    PAPER = "paper"


class WorkKind(str, Enum):
    PIPELINE = "pipeline"
    GPU_EXPERIMENT = "gpu_experiment"
    ANALYSIS = "analysis"
    PAPER = "paper"


@dataclass(slots=True)
class ResourceRequest:
    min_gpus: int = 0
    preferred_gpus: int = 0
    max_gpus: int = 0
    cpus: int = 1
    timeout_sec: float = 3600.0
    placement: str = "any"
    preemptible: bool = False
    checkpointable: bool = False

    def __post_init__(self) -> None:
        values = (self.min_gpus, self.preferred_gpus, self.max_gpus, self.cpus)
        if any(isinstance(value, bool) or int(value) < 0 for value in values):
            raise ValueError("resource counts must be non-negative integers")
        if not self.min_gpus <= self.preferred_gpus <= self.max_gpus:
            raise ValueError("GPU request must satisfy min <= preferred <= max")
        if self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ResourceRequest:
        values = dict(data or {})
        return cls(
            min_gpus=int(values.get("min_gpus", 0)),
            preferred_gpus=int(values.get("preferred_gpus", 0)),
            max_gpus=int(values.get("max_gpus", 0)),
            cpus=int(values.get("cpus", 1)),
            timeout_sec=float(values.get("timeout_sec", 3600.0)),
            placement=str(values.get("placement", "any")),
            preemptible=bool(values.get("preemptible", False)),
            checkpointable=bool(values.get("checkpointable", False)),
        )


@dataclass(slots=True)
class Idea:
    idea_id: str
    title: str
    research_question: str
    falsifiable_hypothesis: str
    primary_metric: str
    family: str = "unclassified"
    source: str = "de_novo"
    parent_ids: tuple[str, ...] = ()
    status: IdeaStatus = IdeaStatus.CANDIDATE
    budget_tier: BudgetTier = BudgetTier.DESK
    priority: float = 0.0
    candidate: dict[str, Any] = field(default_factory=dict)
    current_item_id: str = ""
    exit_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.idea_id.strip():
            raise ValueError("idea_id is required")
        for name in (
            "title",
            "research_question",
            "falsifiable_hypothesis",
            "primary_metric",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} is required")
        if not 0.0 <= float(self.priority) <= 1.0:
            raise ValueError("priority must be between 0 and 1")

    @property
    def normalized_title(self) -> str:
        return normalize_title(self.title)

    @property
    def active(self) -> bool:
        return self.status in ACTIVE_IDEA_STATUSES

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_IDEA_STATUSES

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["budget_tier"] = self.budget_tier.value
        value["parent_ids"] = list(self.parent_ids)
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Idea:
        return cls(
            idea_id=str(data["idea_id"]),
            title=str(data["title"]),
            research_question=str(data["research_question"]),
            falsifiable_hypothesis=str(data["falsifiable_hypothesis"]),
            primary_metric=str(data["primary_metric"]),
            family=str(data.get("family", "unclassified")),
            source=str(data.get("source", "de_novo")),
            parent_ids=tuple(str(item) for item in data.get("parent_ids", ())),
            status=IdeaStatus(str(data.get("status", IdeaStatus.CANDIDATE.value))),
            budget_tier=BudgetTier(
                str(data.get("budget_tier", BudgetTier.DESK.value))
            ),
            priority=float(data.get("priority", 0.0)),
            candidate=dict(data.get("candidate", {}) or {}),
            current_item_id=str(data.get("current_item_id", "") or ""),
            exit_reason=str(data.get("exit_reason", "") or ""),
            created_at=str(data.get("created_at", utc_now())),
            updated_at=str(data.get("updated_at", utc_now())),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(slots=True)
class WorkItem:
    item_id: str
    idea_id: str
    kind: WorkKind
    profile: str
    dependencies: tuple[str, ...] = ()
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    status: WorkItemStatus = WorkItemStatus.PENDING
    attempt: int = 0
    attempt_limit: int = 2
    command: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.item_id or not self.idea_id:
            raise ValueError("item_id and idea_id are required")
        if self.attempt < 0 or self.attempt_limit < 1:
            raise ValueError("invalid attempt counters")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_WORK_ITEM_STATUSES

    def deterministic_task_id(self) -> str:
        return f"{self.item_id}-attempt-{self.attempt + 1:02d}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["status"] = self.status.value
        value["dependencies"] = list(self.dependencies)
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkItem:
        return cls(
            item_id=str(data["item_id"]),
            idea_id=str(data["idea_id"]),
            kind=WorkKind(str(data.get("kind", WorkKind.PIPELINE.value))),
            profile=str(data.get("profile", "")),
            dependencies=tuple(
                str(item) for item in data.get("dependencies", ())
            ),
            resources=ResourceRequest.from_dict(data.get("resources")),
            status=WorkItemStatus(
                str(data.get("status", WorkItemStatus.PENDING.value))
            ),
            attempt=int(data.get("attempt", 0)),
            attempt_limit=int(data.get("attempt_limit", 2)),
            command=str(data.get("command", "") or ""),
            metadata=dict(data.get("metadata", {}) or {}),
            result=dict(data.get("result", {}) or {}),
            created_at=str(data.get("created_at", utc_now())),
            updated_at=str(data.get("updated_at", utc_now())),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(slots=True)
class ResourceLease:
    lease_id: str
    idea_id: str
    item_id: str
    requested_gpus: int
    allocated_gpus: int = 0
    status: LeaseStatus = LeaseStatus.REQUESTED
    priority: float = 0.0
    submitted_at: str = field(default_factory=utc_now)
    started_at: str = ""
    released_at: str = ""
    pool_task_id: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResourceLease:
        return cls(
            lease_id=str(data["lease_id"]),
            idea_id=str(data["idea_id"]),
            item_id=str(data["item_id"]),
            requested_gpus=int(data.get("requested_gpus", 0)),
            allocated_gpus=int(data.get("allocated_gpus", 0)),
            status=LeaseStatus(
                str(data.get("status", LeaseStatus.REQUESTED.value))
            ),
            priority=float(data.get("priority", 0.0)),
            submitted_at=str(data.get("submitted_at", utc_now())),
            started_at=str(data.get("started_at", "") or ""),
            released_at=str(data.get("released_at", "") or ""),
            pool_task_id=str(data.get("pool_task_id", "") or ""),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(slots=True)
class GateDecision:
    decision: GateAction
    reason_code: str
    current_tier: BudgetTier
    next_tier: BudgetTier | None = None
    next_status: IdeaStatus | None = None
    evidence_refs: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    decided_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.value
        value["current_tier"] = self.current_tier.value
        value["next_tier"] = self.next_tier.value if self.next_tier else None
        value["next_status"] = self.next_status.value if self.next_status else None
        value["evidence_refs"] = list(self.evidence_refs)
        return value

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GateDecision:
        raw_next_tier = data.get("next_tier")
        raw_next_status = data.get("next_status")
        return cls(
            decision=GateAction(str(data["decision"])),
            reason_code=str(data["reason_code"]),
            current_tier=BudgetTier(str(data["current_tier"])),
            next_tier=(
                BudgetTier(str(raw_next_tier)) if raw_next_tier else None
            ),
            next_status=(
                IdeaStatus(str(raw_next_status)) if raw_next_status else None
            ),
            evidence_refs=tuple(
                str(item) for item in data.get("evidence_refs", ())
            ),
            details=dict(data.get("details", {}) or {}),
            decided_at=str(data.get("decided_at", utc_now())),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )


ModelT = TypeVar("ModelT", Idea, WorkItem, ResourceLease, GateDecision)


def model_dict(value: ModelT) -> dict[str, Any]:
    """Serialize any Factory model."""

    return value.to_dict()
