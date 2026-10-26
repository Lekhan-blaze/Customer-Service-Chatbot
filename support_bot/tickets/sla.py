"""SLA deadline calculator and breach escalation engine."""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from support_bot.clock import Clock, add_business_time
from support_bot.tickets.models import TicketPriority


def calculate_sla_deadlines(
    priority: TicketPriority,
    clock: Clock,
    config: Dict[str, Any],
    start_time: Optional[datetime] = None,
) -> Tuple[datetime, datetime]:
    """Calculate (sla_deadline, sla_warning_at) in business hours.
    
    Excludes weekends and holidays using support_bot.clock.add_business_time.
    Warning fires when 75% (configurable) of SLA time is consumed.
    """
    ticket_cfg = config.get("tickets", {})
    sla_hours_map = ticket_cfg.get(
        "sla_hours_by_priority",
        {"P1": 4, "P2": 8, "P3": 24, "P4": 48}
    )
    pri_key = priority.value if isinstance(priority, TicketPriority) else str(priority)
    allowed_hours = sla_hours_map.get(pri_key, 24)
    total_minutes = int(allowed_hours * 60)

    warning_ratio = ticket_cfg.get("warning_threshold_ratio", 0.75)
    warning_minutes = int(total_minutes * warning_ratio)

    start = start_time or clock.now()

    # Calculate deadline and warning time honoring business hours, weekends and holidays
    sla_deadline = add_business_time(start, minutes=total_minutes, cfg=config)
    sla_warning_at = add_business_time(start, minutes=warning_minutes, cfg=config)

    return sla_deadline, sla_warning_at
