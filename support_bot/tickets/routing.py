"""Ticket routing engine based on skills, availability, workload, and business hours."""

from typing import Any, Dict, Optional
from support_bot.clock import Clock, is_business_hours
from support_bot.tickets.models import TicketPriority


def route_ticket(
    issue_type: str,
    priority: TicketPriority,
    config: Dict[str, Any],
    clock: Optional[Clock] = None,
) -> str:
    """Determine assigned team for ticket.
    
    Checks config['tickets']['teams']:
    - skills: list of matching issue types/keywords
    - available: boolean flag
    - current_workload < max_workload
    If outside business hours:
    - P1 routes to on_call queue (or configured p1_queue)
    - P2-P4 routes to next-working-day queue if configured, or falls back.
    """
    ticket_cfg = config.get("tickets", {})
    teams = ticket_cfg.get("teams", {})
    fallback_team = ticket_cfg.get("fallback_team", "general_support")

    # In business hours check
    in_hours = True
    if clock is not None:
        in_hours = is_business_hours(clock, config)

    # Outside business hours routing
    if not in_hours:
        after_hours_cfg = ticket_cfg.get("after_hours", {})
        # If P1, route to on-call
        if priority == TicketPriority.P1:
            on_call_target = after_hours_cfg.get("p1_queue", "on_call")
            return on_call_target
        
        # P2-P4 route to next_working_day or configured after-hours queue
        after_hours_queue = after_hours_cfg.get("default_queue") or ticket_cfg.get("after_hours_queue")
        if after_hours_queue:
            return after_hours_queue

    best_team = None

    for team_name, team_info in teams.items():
        if not team_info.get("available", True):
            continue

        skills = team_info.get("skills", [])
        if issue_type.lower() in [s.lower() for s in skills]:
            current_workload = team_info.get("current_workload", 0)
            max_workload = team_info.get("max_workload", 100)
            if current_workload < max_workload:
                best_team = team_name
                break

    return best_team or fallback_team
