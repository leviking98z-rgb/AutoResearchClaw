"""Durable domain models for AutoResearch v2."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def stable_id(value: str, *, prefix: str, length: int = 12) -> str:
    normalized = " ".join(str(value).strip().casefold().split())
    slug = _SLUG_RE.sub("-", normalized).strip("-")[:36] or prefix
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{slug}-{digest}"


class IdeaStatus(str, Enum):
    RESERVOIR = "reservoir"
    NEW = "new"
    DESIGNING = "designing"
    BUILDING = "building"
    PILOTING = "piloting"
    SCALING = "scaling"
    REPORTING = "reporting"
    COMPLETED = "completed"
    COMPLETED_NEGATIVE = "completed_negative"
    REJECTED = "rejected"
    RETRYABLE = "retryable"
    QUARANTINED = "quarantined"


ACTIVE_IDEA_STATUSES = frozenset(
    {
        IdeaStatus.NEW,
        IdeaStatus.DESIGNING,
        IdeaStatus.BUILDING,
        IdeaStatus.PILOTING,
        IdeaStatus.SCALING,
        IdeaStatus.REPORTING,
        IdeaStatus.RETRYABLE,
    }
)


class JobKind(str, Enum):
    DESIGN = "design"
    BUILD = "build"
    PILOT = "pilot"
    SCALE = "scale"
    REPORT = "report"


class JobStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY_WAIT = "retry_wait"


class AttemptStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    VALIDATING = "validating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class IdeaRecord:
    idea_id: str
    title: str
    research_question: str
    falsifiable_hypothesis: str
    primary_metric: str
    candidate: dict[str, Any]
    score: float = 0.0
    family: str = "other"
    priority: float = 0.0
    status: IdeaStatus = IdeaStatus.NEW
    current_job_id: str = ""
    exit_reason: str = ""
    llm_tokens_spent: int = 0
    gpu_seconds_spent: float = 0.0
    llm_calls: int = 0
    last_progress_at: str = field(default_factory=utc_now)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> IdeaRecord:
        return cls(
            idea_id=str(value["idea_id"]),
            title=str(value["title"]),
            research_question=str(value["research_question"]),
            falsifiable_hypothesis=str(value["falsifiable_hypothesis"]),
            primary_metric=str(value["primary_metric"]),
            candidate=dict(value.get("candidate", {}) or {}),
            score=float(value.get("score", 0.0)),
            family=str(value.get("family", "other")),
            priority=float(value.get("priority", 0.0)),
            status=IdeaStatus(str(value.get("status", IdeaStatus.NEW.value))),
            current_job_id=str(value.get("current_job_id", "") or ""),
            exit_reason=str(value.get("exit_reason", "") or ""),
            llm_tokens_spent=int(value.get("llm_tokens_spent", 0)),
            gpu_seconds_spent=float(value.get("gpu_seconds_spent", 0.0)),
            llm_calls=int(value.get("llm_calls", 0)),
            last_progress_at=str(
                value.get("last_progress_at", value.get("updated_at", utc_now()))
            ),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(slots=True)
class JobRecord:
    job_id: str
    idea_id: str
    kind: JobKind
    status: JobStatus = JobStatus.READY
    attempt: int = 0
    attempt_limit: int = 2
    requires_gpu: bool = False
    min_gpus: int = 0
    preferred_gpus: int = 0
    max_gpus: int = 0
    timeout_sec: float = 3600.0
    command: str = ""
    expected_output_dir: str = ""
    attempt_id: str = ""
    submitted_task_id: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JobRecord:
        return cls(
            job_id=str(value["job_id"]),
            idea_id=str(value["idea_id"]),
            kind=JobKind(str(value["kind"])),
            status=JobStatus(str(value.get("status", JobStatus.READY.value))),
            attempt=int(value.get("attempt", 0)),
            attempt_limit=int(value.get("attempt_limit", 2)),
            requires_gpu=bool(value.get("requires_gpu", False)),
            min_gpus=int(value.get("min_gpus", 0)),
            preferred_gpus=int(value.get("preferred_gpus", 0)),
            max_gpus=int(value.get("max_gpus", 0)),
            timeout_sec=float(value.get("timeout_sec", 3600.0)),
            command=str(value.get("command", "") or ""),
            expected_output_dir=str(
                value.get("expected_output_dir", "") or ""
            ),
            attempt_id=str(value.get("attempt_id", "") or ""),
            submitted_task_id=str(
                value.get("submitted_task_id", "") or ""
            ),
            result=dict(value.get("result", {}) or {}),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )


@dataclass(slots=True)
class AttemptRecord:
    attempt_id: str
    idea_id: str
    job_id: str
    number: int
    status: AttemptStatus = AttemptStatus.CREATED
    input_manifest: dict[str, Any] = field(default_factory=dict)
    output_manifest: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    started_at: str = ""
    finished_at: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttemptRecord:
        return cls(
            attempt_id=str(value["attempt_id"]),
            idea_id=str(value["idea_id"]),
            job_id=str(value["job_id"]),
            number=int(value["number"]),
            status=AttemptStatus(
                str(value.get("status", AttemptStatus.CREATED.value))
            ),
            input_manifest=dict(value.get("input_manifest", {}) or {}),
            output_manifest=dict(value.get("output_manifest", {}) or {}),
            validation=dict(value.get("validation", {}) or {}),
            error=str(value.get("error", "") or ""),
            started_at=str(value.get("started_at", "") or ""),
            finished_at=str(value.get("finished_at", "") or ""),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )
