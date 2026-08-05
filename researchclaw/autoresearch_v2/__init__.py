"""Transactional, pipeline-independent AutoResearch execution core.

The v2 control plane intentionally does not import ``researchclaw.pipeline`` or
``researchclaw.rsi.supervisor``.  It reuses stateless infrastructure (role LLM
clients, validators, literature adapters, and the GPU pool) behind small
interfaces while keeping durable workflow state here.
"""

from .models import AttemptStatus, IdeaRecord, JobKind, JobRecord, JobStatus
from .store import V2Store

__all__ = [
    "AttemptStatus",
    "IdeaRecord",
    "JobKind",
    "JobRecord",
    "JobStatus",
    "V2Store",
]
