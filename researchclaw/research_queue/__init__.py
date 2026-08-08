"""Lightweight continuous multi-Idea research prototype.

The prototype intentionally keeps a much smaller control plane than the
production-oriented ``autoresearch_v2`` package:

* four Idea states;
* one Controller as the state owner;
* one Prepare/Run/Review loop;
* one Run protocol for B0/B1/B2 budgets; and
* pluggable local or ClusterBridge execution backends.
"""

from .config import ResearchQueueConfig
from .controller import ResearchQueueController
from .models import (
    BudgetLevel,
    Conclusion,
    IdeaRecord,
    IdeaStatus,
    MetricDirection,
    MetricGuardrail,
    MetricRelation,
    ResearchSpec,
    ReviewAction,
    RunRecord,
    RunStatus,
)
from .store import ResearchQueueStore

__all__ = [
    "BudgetLevel",
    "Conclusion",
    "IdeaRecord",
    "IdeaStatus",
    "MetricDirection",
    "MetricGuardrail",
    "MetricRelation",
    "ResearchQueueConfig",
    "ResearchQueueController",
    "ResearchQueueStore",
    "ResearchSpec",
    "ReviewAction",
    "RunRecord",
    "RunStatus",
]
