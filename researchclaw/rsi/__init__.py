"""Campaign-level recursive self-improvement (RSI) supervisor."""

from .storage import (
    CampaignStore,
    EventLog,
    atomic_write_json,
    atomic_write_text,
    cleanup_atomic_temp_files,
)
from .supervisor import (
    DEFAULT_CAMPAIGN_ROOT,
    CampaignSupervisor,
    SupervisorOptions,
    campaign_status,
    new_campaign_id,
    resolve_campaign,
)

__all__ = [
    "DEFAULT_CAMPAIGN_ROOT",
    "CampaignStore",
    "CampaignSupervisor",
    "EventLog",
    "SupervisorOptions",
    "atomic_write_json",
    "atomic_write_text",
    "campaign_status",
    "cleanup_atomic_temp_files",
    "new_campaign_id",
    "resolve_campaign",
]
