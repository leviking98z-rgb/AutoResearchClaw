"""Durable per-Idea resource accounting and hard budget enforcement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import BudgetConfig
from .models import BudgetTier, utc_now


@dataclass(slots=True)
class BudgetLedger:
    idea_id: str
    gpu_seconds_by_tier: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    engineering_repairs: int = 0
    no_progress_rounds: int = 0
    usage_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    started_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def record_gpu_seconds(
        self,
        tier: BudgetTier,
        *,
        gpu_count: int,
        elapsed_sec: float,
    ) -> float:
        if gpu_count < 0 or elapsed_sec < 0:
            raise ValueError("GPU usage cannot be negative")
        amount = float(gpu_count) * float(elapsed_sec)
        key = tier.value
        self.gpu_seconds_by_tier[key] = (
            float(self.gpu_seconds_by_tier.get(key, 0.0)) + amount
        )
        self.updated_at = utc_now()
        return amount

    def gpu_hours(self, tier: BudgetTier | None = None) -> float:
        if tier is not None:
            seconds = self.gpu_seconds_by_tier.get(tier.value, 0.0)
        else:
            seconds = sum(self.gpu_seconds_by_tier.values())
        return round(float(seconds) / 3600.0, 6)

    def record_llm_call(self, count: int = 1) -> None:
        if count < 0:
            raise ValueError("LLM call count cannot be negative")
        self.llm_calls += count
        self.updated_at = utc_now()

    def record_repair(self) -> None:
        self.engineering_repairs += 1
        self.updated_at = utc_now()

    def record_attempt_usage(
        self,
        accounting_id: str,
        tier: BudgetTier,
        *,
        gpu_count: int,
        elapsed_sec: float,
        llm_calls: int,
    ) -> tuple[dict[str, Any], bool]:
        """Commit one attempt exactly once and return ``(record, created)``."""

        key = str(accounting_id).strip()
        if not key:
            raise ValueError("accounting_id is required")
        existing = self.usage_records.get(key)
        if isinstance(existing, Mapping):
            return dict(existing), False
        if gpu_count < 0 or elapsed_sec < 0 or llm_calls < 0:
            raise ValueError("attempt usage cannot be negative")
        gpu_seconds = self.record_gpu_seconds(
            tier,
            gpu_count=gpu_count,
            elapsed_sec=elapsed_sec,
        )
        self.record_llm_call(llm_calls)
        record = {
            "accounting_id": key,
            "budget_tier": tier.value,
            "allocated_gpus": int(gpu_count),
            "elapsed_sec": round(float(elapsed_sec), 6),
            "gpu_seconds": round(float(gpu_seconds), 6),
            "llm_calls": int(llm_calls),
            "committed_at": utc_now(),
        }
        self.usage_records[key] = record
        self.updated_at = utc_now()
        return dict(record), True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BudgetLedger:
        return cls(
            idea_id=str(data["idea_id"]),
            gpu_seconds_by_tier={
                str(key): float(value)
                for key, value in dict(
                    data.get("gpu_seconds_by_tier", {}) or {}
                ).items()
            },
            llm_calls=int(data.get("llm_calls", 0)),
            engineering_repairs=int(data.get("engineering_repairs", 0)),
            no_progress_rounds=int(data.get("no_progress_rounds", 0)),
            usage_records={
                str(key): dict(value)
                for key, value in dict(
                    data.get("usage_records", {}) or {}
                ).items()
                if isinstance(value, Mapping)
            },
            started_at=str(data.get("started_at", utc_now())),
            updated_at=str(data.get("updated_at", utc_now())),
        )


def tier_limit_hours(config: BudgetConfig, tier: BudgetTier) -> float | None:
    return {
        BudgetTier.DESK: None,
        BudgetTier.SMOKE: config.smoke_gpu_hours,
        BudgetTier.PILOT: config.pilot_gpu_hours,
        BudgetTier.VALIDATION: config.validation_gpu_hours,
        BudgetTier.SCALE: config.scale_gpu_hours,
        BudgetTier.PAPER: None,
    }[tier]


def budget_allows(
    ledger: BudgetLedger,
    config: BudgetConfig,
    tier: BudgetTier,
    *,
    requested_gpu_seconds: float = 0.0,
) -> bool:
    if requested_gpu_seconds < 0:
        raise ValueError("requested_gpu_seconds cannot be negative")
    limit = tier_limit_hours(config, tier)
    if limit is None:
        if tier is BudgetTier.DESK:
            return ledger.llm_calls < config.desk_llm_calls
        return True
    projected = ledger.gpu_hours(tier) + requested_gpu_seconds / 3600.0
    return projected <= limit + 1e-12


def repair_allowed(ledger: BudgetLedger, config: BudgetConfig) -> bool:
    return ledger.engineering_repairs < config.max_engineering_repairs
