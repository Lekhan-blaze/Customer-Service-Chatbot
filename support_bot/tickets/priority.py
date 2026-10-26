"""Ticket priority engine based on severity, sentiment, impact, and SLA configuration."""

from typing import Any, Dict, Optional
from support_bot.tickets.models import TicketPriority


def calculate_ticket_priority(
    issue_type: str,
    severity: Optional[str] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    escalation_data: Optional[Dict[str, Any]] = None,
    wait_time_minutes: int = 0,
    customer_impact: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> TicketPriority:
    """Calculate ticket priority P1, P2, P3, or P4.
    
    Priority levels:
    - P1 (Critical): High financial risk, duplicate payment, account compromise, legal threat,
                     widespread customer impact, high severity escalation, or wait >= 120m.
    - P2 (High): Total blockage (login locked out, course access denied), repeated negative sentiment
                 escalation, high customer impact, or wait >= 60m.
    - P3 (Medium): Standard requests, normal refunds, video buffering errors.
    - P4 (Low): General inquiries, informational feedback.
    """
    # 1. Customer Impact check (widespread / critical -> P1; high -> P2)
    if customer_impact:
        impact_lower = customer_impact.lower()
        if impact_lower in ("critical", "widespread", "all_students", "enterprise", "outage"):
            return TicketPriority.P1
        elif impact_lower in ("high", "multiple_users", "team"):
            return TicketPriority.P2

    # 2. Escalation Record checks (from stored escalation records)
    if escalation_data:
        rule_name = escalation_data.get("rule_name", "")
        esc_severity = escalation_data.get("severity", "")
        routing_outcome = escalation_data.get("routing_outcome", "")
        if "high_risk" in rule_name or esc_severity == "high" or "on_call" in routing_outcome or "urgent" in rule_name:
            return TicketPriority.P1
        if "repeated_negative" in rule_name or esc_severity == "medium":
            return TicketPriority.P2

    # 3. Sentiment & Risk checks
    if sentiment_data:
        risk_cat = sentiment_data.get("risk_category")
        if risk_cat in ("duplicate_payment", "account_compromise", "legal_threat", "data_breach"):
            return TicketPriority.P1
        dominant = sentiment_data.get("dominant_label", "")
        if dominant in ("angry", "frustrated") and sentiment_data.get("calibrated_scores", {}).get(dominant, 0) > 0.7:
            return TicketPriority.P1

    # 4. Explicit severity checks
    if severity:
        sev_lower = severity.lower()
        if sev_lower in ("critical", "p1", "blocker"):
            return TicketPriority.P1
        elif sev_lower in ("high", "p2", "major"):
            return TicketPriority.P2

    # 5. Issue type defaults
    if issue_type in ("duplicate_payment", "account_compromise", "legal_threat"):
        return TicketPriority.P1
    if issue_type in ("course_access", "login", "access_blocked"):
        return TicketPriority.P2

    # 6. Wait time elevation
    if wait_time_minutes >= 120:
        return TicketPriority.P1
    elif wait_time_minutes >= 60:
        return TicketPriority.P2

    if issue_type in ("refund", "video_error", "technical"):
        return TicketPriority.P3

    return TicketPriority.P3
