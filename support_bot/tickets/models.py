"""Data models and enums for Module 2: Structured Support Tickets."""

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional


class TicketPriority(str, Enum):
    P1 = "P1"  # Critical (4h SLA) - payment duplicate, account compromise, legal threat
    P2 = "P2"  # High (8h SLA) - access blocked, broken video
    P3 = "P3"  # Medium (24h SLA) - general queries, standard refunds
    P4 = "P4"  # Low (48h SLA) - feedback, minor non-blocking questions


class TicketStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


@dataclass
class Ticket:
    id: str
    session_id: Optional[str]
    customer_name: str
    contact_info: Optional[str]
    order_or_enrollment_id: str
    product_or_course: str
    issue_type: str
    description: str
    summary: str
    evidence_refs: List[str] = field(default_factory=list)
    priority: TicketPriority = TicketPriority.P3
    assigned_team: str = "general_support"
    status: TicketStatus = TicketStatus.OPEN
    sla_deadline: str = ""
    sla_warning_at: str = ""
    sla_breached: bool = False
    sla_warning_sent: bool = False
    duplicate_of_ticket_id: Optional[str] = None
    related_ticket_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "customer_name": self.customer_name,
            "contact_info": self.contact_info,
            "order_or_enrollment_id": self.order_or_enrollment_id,
            "product_or_course": self.product_or_course,
            "issue_type": self.issue_type,
            "description": self.description,
            "summary": self.summary,
            "evidence_refs": self.evidence_refs,
            "priority": self.priority.value if isinstance(self.priority, TicketPriority) else str(self.priority),
            "assigned_team": self.assigned_team,
            "status": self.status.value if isinstance(self.status, TicketStatus) else str(self.status),
            "sla_deadline": self.sla_deadline,
            "sla_warning_at": self.sla_warning_at,
            "sla_breached": bool(self.sla_breached),
            "sla_warning_sent": bool(self.sla_warning_sent),
            "duplicate_of_ticket_id": self.duplicate_of_ticket_id,
            "related_ticket_ids": self.related_ticket_ids,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> "Ticket":
        ev_refs = row["evidence_refs"]
        if isinstance(ev_refs, str):
            try:
                ev_refs = json.loads(ev_refs)
            except Exception:
                ev_refs = []
        
        rel_ids = row["related_ticket_ids"] if "related_ticket_ids" in row.keys() else "[]"
        if isinstance(rel_ids, str):
            try:
                rel_ids = json.loads(rel_ids)
            except Exception:
                rel_ids = []

        meta = row["metadata"] if "metadata" in row.keys() else "{}"
        if isinstance(meta, str) and meta:
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        return cls(
            id=row["id"],
            session_id=row["session_id"],
            customer_name=row["customer_name"],
            contact_info=row["contact_info"],
            order_or_enrollment_id=row["order_or_enrollment_id"],
            product_or_course=row["product_or_course"],
            issue_type=row["issue_type"],
            description=row["description"],
            summary=row["summary"],
            evidence_refs=ev_refs or [],
            priority=TicketPriority(row["priority"]),
            assigned_team=row["assigned_team"],
            status=TicketStatus(row["status"]),
            sla_deadline=row["sla_deadline"],
            sla_warning_at=row["sla_warning_at"],
            sla_breached=bool(row["sla_breached"]),
            sla_warning_sent=bool(row["sla_warning_sent"]),
            duplicate_of_ticket_id=row["duplicate_of_ticket_id"],
            related_ticket_ids=rel_ids or [],
            metadata=meta or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


@dataclass
class ExtractionResult:
    ticket_data: Optional[Dict[str, Any]]
    missing_fields: List[str]
    clarification_prompt: Optional[str] = None
    confidence: float = 1.0


@dataclass
class HandoffSummary:
    ticket_id: str
    priority: str
    assigned_team: str
    sla_deadline: str
    customer_display: str
    masked_summary: str
    evidence_refs: List[str]
