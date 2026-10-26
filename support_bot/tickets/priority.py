"""Ticket priority engine based on severity, sentiment, impact, and SLA configuration."""

from typing import Any, Dict, Optional
from support_bot.tickets.models import TicketPriority


def calculate_ticket_priority(
    issue_type: str,
    severity: Optional[str] = None,
    sentiment_data: Optional[Dict[str, Any]] = None,
    escalation_data: Optional[Dict[str, Any]] = None,
    wait_time_minutes: int = 0,
    config: Optional[Dict[str, Any]] = None,
) -> TicketPriority:
    """Calculate ticket priority P1, P2, P3, or P4.
    
    P1 (Critical): High financial risk, account compromise, legal threat, duplicate payment, or severe anger/high negative escalation.
    P2 (High): Total blockage (course access, login locked out), repeated negative escalation, or wait > 60m.
    P3 (Medium): Standard requests, normal refunds, video buffering errors.
    P4 (Low): General inquiries, informational feedback.
    """
    # 1. Escalation Record checks
    if escalation_data:
        rule_name = escalation_data.get("rule_name", "")
        esc_severity = escalation_data.get("severity", "")
        if "high_risk" in rule_name or esc_severity == "high" or "on_call" in escalation_data.get("routing_outcome", ""):
            return TicketPriority.P1
        if "repeated_negative" in rule_name or esc_severity == "medium":
            return TicketPriority.P2

    # 2. Risk category & Sentiment checks
    if sentiment_data:
        risk_cat = sentiment_data.get("risk_category")
        if risk_cat in ("duplicate_payment", "account_compromise", "legal_threat", "data_breach"):
            return TicketPriority.P1
        dominant = sentiment_data.get("dominant_label", "")
        if dominant in ("angry", "frustrated") and sentiment_data.get("calibrated_scores", {}).get(dominant, 0) > 0.7:
            return TicketPriority.P1

    # 3. Issue type & Explicit severity checks
    if severity == "critical" or issue_type in ("duplicate_payment", "account_compromise", "legal_threat"):
        return TicketPriority.P1

    if severity == "high" or issue_type in ("course_access", "login", "access_blocked"):
        return TicketPriority.P2

    # 4. Wait time elevation
    if wait_time_minutes >= 120:
        return TicketPriority.P1
    elif wait_time_minutes >= 60:
        return TicketPriority.P2

    if issue_type in ("refund", "video_error", "technical"):
        return TicketPriority.P3

    return TicketPriority.P3
