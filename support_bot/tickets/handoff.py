"""Masked handoff summary generator for support agents."""

from typing import Any, Dict
from support_bot.masking import mask_payment, mask_pii
from support_bot.tickets.models import HandoffSummary, Ticket


def generate_masked_handoff_summary(ticket: Ticket) -> HandoffSummary:
    """Generate a sanitized handoff summary card for human support agents.
    
    Masks PII (emails, phone numbers, customer name if sensitive) and payment data (card numbers).
    Includes only what an agent needs to resolve the ticket.
    """
    raw_text = f"Customer reported: {ticket.description}"
    masked_desc = mask_payment(mask_pii(raw_text))

    summary_lines = [
        f"[{ticket.priority.value}] {ticket.issue_type.upper()} - {ticket.product_or_course}",
        f"Order/Ref: {ticket.order_or_enrollment_id}",
        f"Details: {masked_desc}",
        f"SLA Target: {ticket.sla_deadline}",
    ]
    masked_summary = "\n".join(summary_lines)

    masked_customer = mask_pii(ticket.customer_name)

    return HandoffSummary(
        ticket_id=ticket.id,
        priority=ticket.priority.value,
        assigned_team=ticket.assigned_team,
        sla_deadline=ticket.sla_deadline,
        customer_display=masked_customer,
        masked_summary=masked_summary,
        evidence_refs=ticket.evidence_refs,
    )
