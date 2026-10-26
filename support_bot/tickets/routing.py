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
    If after-hours or no team available/under capacity, routes to fallback queue.
    """
    ticket_cfg = config.get("tickets", {})
    teams = ticket_cfg.get("teams", {})
    fallback_team = ticket_cfg.get("fallback_team", "general_support")

    # In business hours check
    in_hours = True
    if clock is not None:
        in_hours = is_business_hours(clock, config)

    # If priority is P1 and outside business hours, can route to on_call or escalation team
    if not in_hours and priority == TicketPriority.P1:
        if "on_call" in teams and teams["on_call"].get("available", True):
            return "on_call"

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
