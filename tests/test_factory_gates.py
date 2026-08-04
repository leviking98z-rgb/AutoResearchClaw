from __future__ import annotations

import json
from pathlib import Path

from researchclaw.factory.budgets import BudgetLedger
from researchclaw.factory.config import FactoryConfig
from researchclaw.factory.gates import EvidenceGate
from researchclaw.factory.models import (
    BudgetTier,
    GateAction,
    Idea,
    IdeaStatus,
)


def _idea(status: IdeaStatus, tier: BudgetTier) -> Idea:
    return Idea(
        idea_id="idea-gate",
        title="Gate study",
        research_question="question",
        falsifiable_hypothesis="hypothesis",
        primary_metric="accuracy",
        status=status,
        budget_tier=tier,
    )


def _write_summary(run_dir: Path, value: object) -> None:
    path = run_dir / "stage-14" / "experiment_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _valid_summary(**extra: object) -> dict[str, object]:
    return {
        "result_valid": True,
        "successful_seed_count": 3,
        "success_probability": 0.99,
        "primary_effect_size": 0.1,
        "best_run": {
            "status": "completed",
            "returncode": 0,
            "result_valid": True,
            "metrics": {
                "baseline/seed-0/accuracy": 0.5,
                "method/seed-0/accuracy": 0.6,
            },
        },
        "condition_summaries": {
            "baseline": {"metrics": {"accuracy": 0.5}},
            "method": {"metrics": {"accuracy": 0.6}},
        },
        **extra,
    }


def test_invalid_pilot_repairs_then_rejects(tmp_path: Path) -> None:
    config = FactoryConfig.from_mapping({"factory": {}})
    gate = EvidenceGate(config)
    idea = _idea(IdeaStatus.PILOT, BudgetTier.PILOT)
    _write_summary(
        tmp_path,
        {
            "result_valid": False,
            "best_run": {
                "status": "failed",
                "returncode": 0,
                "stdout": "DatasetNotFoundError: dataset missing",
                "metrics": {"method/accuracy": 0.0},
            },
        },
    )
    ledger = BudgetLedger(idea_id=idea.idea_id)
    first = gate.evaluate(idea, run_dir=tmp_path, ledger=ledger)
    assert first.decision is GateAction.REPAIR

    ledger.engineering_repairs = config.budgets.max_engineering_repairs
    exhausted = gate.evaluate(idea, run_dir=tmp_path, ledger=ledger)
    assert exhausted.decision is GateAction.REJECT
    assert exhausted.reason_code == "ENGINEERING_INFEASIBLE"


def test_valid_pilot_promotes_and_valid_null_completes_negative(
    tmp_path: Path,
) -> None:
    config = FactoryConfig.from_mapping({"factory": {}})
    gate = EvidenceGate(config)
    pilot = _idea(IdeaStatus.PILOT, BudgetTier.PILOT)
    _write_summary(tmp_path, _valid_summary())
    decision = gate.evaluate(
        pilot,
        run_dir=tmp_path,
        ledger=BudgetLedger(pilot.idea_id),
    )
    assert decision.next_status is IdeaStatus.VALIDATING

    validating = _idea(IdeaStatus.VALIDATING, BudgetTier.VALIDATION)
    _write_summary(
        tmp_path,
        _valid_summary(futility_probability=0.99, informative_null=True),
    )
    null = gate.evaluate(
        validating,
        run_dir=tmp_path,
        ledger=BudgetLedger(validating.idea_id),
    )
    assert null.decision is GateAction.COMPLETE_NEGATIVE
