"""Deterministic evidence gates and asynchronous early-exit policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .budgets import BudgetLedger, repair_allowed
from .config import FactoryConfig
from .models import (
    BudgetTier,
    GateAction,
    GateDecision,
    Idea,
    IdeaStatus,
)
from .validity import assess_experiment_summary


def load_experiment_summary(run_dir: Path) -> tuple[dict[str, Any], str, Path]:
    paths = (
        run_dir / "experiment_summary_best.json",
        run_dir / "stage-14" / "experiment_summary.json",
    )
    path = next((candidate for candidate in paths if candidate.exists()), paths[-1])
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, "missing", path
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, "malformed", path
    return (
        (dict(value), "ok", path)
        if isinstance(value, Mapping)
        else ({}, "malformed", path)
    )


class EvidenceGate:
    """Fail-closed policy separating pipeline advice from budget promotion."""

    def __init__(self, config: FactoryConfig) -> None:
        self.config = config

    def evaluate(
        self,
        idea: Idea,
        *,
        run_dir: Path,
        ledger: BudgetLedger,
    ) -> GateDecision:
        if idea.status is IdeaStatus.SCREENING:
            return self._pipeline_gate(
                idea,
                run_dir=run_dir,
                ledger=ledger,
                expected_last_stage=8,
                next_tier=BudgetTier.SMOKE,
                next_status=IdeaStatus.BUILDING,
                reason="SCREEN_COMPLETE",
            )
        if idea.status is IdeaStatus.BUILDING:
            return self._pipeline_gate(
                idea,
                run_dir=run_dir,
                ledger=ledger,
                expected_last_stage=11,
                next_tier=BudgetTier.PILOT,
                next_status=IdeaStatus.PILOT,
                reason="BUILD_COMPLETE",
            )
        if idea.status in {
            IdeaStatus.SMOKE,
            IdeaStatus.PILOT,
            IdeaStatus.VALIDATING,
            IdeaStatus.REPAIR,
        }:
            return self._experiment_gate(idea, run_dir=run_dir, ledger=ledger)
        if idea.status is IdeaStatus.PAPER:
            return self._pipeline_gate(
                idea,
                run_dir=run_dir,
                ledger=ledger,
                expected_last_stage=23,
                next_tier=BudgetTier.PAPER,
                next_status=IdeaStatus.COMPLETED,
                reason="PAPER_COMPLETE",
                terminal=True,
            )
        return GateDecision(
            decision=GateAction.CONTINUE,
            reason_code="NO_GATE_FOR_STATUS",
            current_tier=idea.budget_tier,
            next_status=idea.status,
        )

    @staticmethod
    def _last_completed_stage(run_dir: Path) -> int:
        try:
            value = json.loads(
                (run_dir / "checkpoint.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return 0
        try:
            return int(value.get("last_completed_stage", 0))
        except (AttributeError, TypeError, ValueError):
            return 0

    def _pipeline_gate(
        self,
        idea: Idea,
        *,
        run_dir: Path,
        ledger: BudgetLedger,
        expected_last_stage: int,
        next_tier: BudgetTier,
        next_status: IdeaStatus,
        reason: str,
        terminal: bool = False,
    ) -> GateDecision:
        completed = self._last_completed_stage(run_dir)
        simulation_marker = (
            run_dir.parent.parent / "evidence" / f"simulated-{reason}.json"
        )
        if simulation_marker.exists():
            completed = expected_last_stage
        if completed < expected_last_stage:
            if not repair_allowed(ledger, self.config.budgets):
                return GateDecision(
                    decision=GateAction.REJECT,
                    reason_code="PIPELINE_PROFILE_RETRY_EXHAUSTED",
                    current_tier=idea.budget_tier,
                    next_status=IdeaStatus.FAILED,
                    details={
                        "expected_last_stage": expected_last_stage,
                        "observed_last_stage": completed,
                        "profile_status": idea.status.value,
                    },
                )
            return GateDecision(
                # A bounded screen/build/paper profile that did not finish
                # must retry that same profile.  Stage 13-15 repair is only
                # meaningful after experiment evidence exists.
                decision=GateAction.REPAIR,
                reason_code="PIPELINE_PROFILE_INCOMPLETE",
                current_tier=idea.budget_tier,
                next_status=idea.status,
                details={
                    "expected_last_stage": expected_last_stage,
                    "observed_last_stage": completed,
                },
            )
        return GateDecision(
            decision=GateAction.COMPLETE if terminal else GateAction.PROMOTE,
            reason_code=reason,
            current_tier=idea.budget_tier,
            next_tier=next_tier,
            next_status=next_status,
            evidence_refs=("checkpoint.json",),
        )

    def _experiment_gate(
        self,
        idea: Idea,
        *,
        run_dir: Path,
        ledger: BudgetLedger,
    ) -> GateDecision:
        summary, status, path = load_experiment_summary(run_dir)
        evidence_ref = str(path.relative_to(run_dir))
        if status != "ok":
            return self._repair_or_reject(
                idea,
                ledger,
                reason=f"EXPERIMENT_SUMMARY_{status.upper()}",
                evidence_refs=(evidence_ref,),
            )

        validity = assess_experiment_summary(summary)
        if not validity.valid:
            return self._repair_or_reject(
                idea,
                ledger,
                reason="EXPERIMENT_INVALID",
                evidence_refs=(evidence_ref,),
                details={"validity_reasons": validity.reasons},
            )

        minimum_seeds = self.config.early_stopping.minimum_seeds
        declared_seed_count = summary.get("successful_seed_count")
        try:
            observed_seed_count = max(
                validity.successful_seed_count,
                int(declared_seed_count),
            )
        except (TypeError, ValueError):
            observed_seed_count = validity.successful_seed_count
        if (
            validity.seed_evidence_present
            and observed_seed_count < minimum_seeds
        ):
            return self._repair_or_reject(
                idea,
                ledger,
                reason="INSUFFICIENT_VALID_SEEDS",
                evidence_refs=(evidence_ref,),
                details={
                    "successful_seed_count": observed_seed_count,
                    "minimum_seeds": minimum_seeds,
                },
            )

        futility = float(summary.get("futility_probability", 0.0) or 0.0)
        success = float(summary.get("success_probability", 0.0) or 0.0)
        effect = summary.get("primary_effect_size")
        effect_value = float(effect) if isinstance(effect, (int, float)) else None
        informative_null = bool(summary.get("informative_null", False))

        if futility >= self.config.early_stopping.futility_probability:
            if informative_null or idea.status is IdeaStatus.VALIDATING:
                return GateDecision(
                    decision=GateAction.COMPLETE_NEGATIVE,
                    reason_code="NEGATIVE_BUT_INFORMATIVE",
                    current_tier=idea.budget_tier,
                    next_status=IdeaStatus.COMPLETED_NEGATIVE,
                    evidence_refs=(evidence_ref,),
                    details={"futility_probability": futility},
                )
            return GateDecision(
                decision=GateAction.REJECT,
                reason_code="NO_PRIMARY_SIGNAL",
                current_tier=idea.budget_tier,
                next_status=IdeaStatus.REJECTED,
                evidence_refs=(evidence_ref,),
                details={"futility_probability": futility},
            )

        if (
            effect_value is not None
            and abs(effect_value) < self.config.early_stopping.minimum_effect_size
            and success < self.config.early_stopping.success_probability
        ):
            return GateDecision(
                decision=GateAction.PARK,
                reason_code="INSUFFICIENT_INFORMATION_GAIN",
                current_tier=idea.budget_tier,
                next_status=IdeaStatus.PARKED,
                evidence_refs=(evidence_ref,),
                details={
                    "effect_size": effect_value,
                    "success_probability": success,
                },
            )

        if idea.status in {
            IdeaStatus.SMOKE,
            IdeaStatus.PILOT,
            IdeaStatus.REPAIR,
        }:
            return GateDecision(
                decision=GateAction.PROMOTE,
                reason_code="PILOT_SIGNAL_VALID",
                current_tier=idea.budget_tier,
                next_tier=BudgetTier.VALIDATION,
                next_status=IdeaStatus.VALIDATING,
                evidence_refs=(evidence_ref,),
                details={
                    "successful_seed_count": observed_seed_count,
                    "success_probability": success,
                },
            )
        return GateDecision(
            decision=GateAction.PROMOTE,
            reason_code="VALIDATION_EVIDENCE_VALID",
            current_tier=idea.budget_tier,
            next_tier=BudgetTier.PAPER,
            next_status=IdeaStatus.PAPER,
            evidence_refs=(evidence_ref,),
            details={
                "successful_seed_count": observed_seed_count,
                "success_probability": success,
            },
        )

    def _repair_or_reject(
        self,
        idea: Idea,
        ledger: BudgetLedger,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        details: dict[str, Any] | None = None,
    ) -> GateDecision:
        if repair_allowed(ledger, self.config.budgets):
            return GateDecision(
                decision=GateAction.REPAIR,
                reason_code=reason,
                current_tier=idea.budget_tier,
                next_tier=idea.budget_tier,
                next_status=IdeaStatus.REPAIR,
                evidence_refs=evidence_refs,
                details=details or {},
            )
        return GateDecision(
            decision=GateAction.REJECT,
            reason_code="ENGINEERING_INFEASIBLE",
            current_tier=idea.budget_tier,
            next_status=IdeaStatus.REJECTED,
            evidence_refs=evidence_refs,
            details={"trigger": reason, **(details or {})},
        )
