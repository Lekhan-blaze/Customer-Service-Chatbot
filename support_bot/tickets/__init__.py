"""Module 2: Structured Support Tickets & SLA Management."""

from support_bot.tickets.models import (
    HandoffSummary,
    Ticket,
    TicketPriority,
    TicketStatus,
)
from support_bot.tickets.service import TicketService

__all__ = [
    "Ticket",
    "TicketPriority",
    "TicketStatus",
    "HandoffSummary",
    "TicketService",
]
