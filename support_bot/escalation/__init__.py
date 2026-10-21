"""Escalation package — rule-based escalation with job scheduling."""

from support_bot.escalation.service import (
    EscalationRecord,
    SentimentEscalationService,
)

__all__ = ["SentimentEscalationService", "EscalationRecord"]
