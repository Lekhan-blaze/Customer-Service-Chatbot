"""Ticket Service: Main orchestrator for tickets, routing, duplicate detection, and SLA tracking."""

import json
import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, SystemClock
from support_bot.db import run_migrations
from support_bot.jobs import JobQueue
from support_bot.tickets.duplicate import check_duplicate_or_related_tickets
from support_bot.tickets.extractor import extract_ticket_data
from support_bot.tickets.handoff import generate_masked_handoff_summary
from support_bot.tickets.models import (
    ExtractionResult,
    HandoffSummary,
    Ticket,
    TicketPriority,
    TicketStatus,
)
from support_bot.tickets.priority import calculate_ticket_priority
from support_bot.tickets.routing import route_ticket
from support_bot.tickets.sla import calculate_sla_deadlines


class TicketService:
    """Service orchestrating tickets, duplicate detection, routing, and SLA tracking."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: Optional[Clock] = None,
        config: Optional[Dict[str, Any]] = None,
        audit: Optional[AuditLogger] = None,
        job_queue: Optional[JobQueue] = None,
        auto_migrate: bool = True,
    ) -> None:
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        if auto_migrate:
            run_migrations(self.conn)
        self.clock = clock or SystemClock()
        self.config = config or {}
        self.audit = audit or AuditLogger(self.conn, self.clock)
        self.job_queue = job_queue or JobQueue(self.conn, self.clock, self.config)

    def _log_ticket_audit(self, ticket_id: str, action: str, actor: str, details: Dict[str, Any]) -> None:
        """Store ticket-specific audit trail."""
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        self.conn.execute(
            "INSERT INTO ticket_audit (ticket_id, action, actor, details, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (ticket_id, action, actor, json.dumps(details), now),
        )
        self.conn.commit()

    def create_ticket_from_conversation(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        sentiment_data: Optional[Dict[str, Any]] = None,
        escalation_data: Optional[Dict[str, Any]] = None,
        actor: str = "customer_service_bot",
    ) -> Tuple[Optional[Ticket], Optional[str]]:
        """Create a structured ticket from conversation messages.
        
        Returns (Ticket, None) if successful, or (None, clarification_prompt) if mandatory fields are missing.
        """
        # 1. Extraction
        ext_res: ExtractionResult = extract_ticket_data(
            messages=messages,
            session_id=session_id,
            config=self.config,
        )

        if ext_res.missing_fields or not ext_res.ticket_data:
            self.audit.log(
                "ticket_extraction_clarification_requested",
                session_id=session_id,
                missing_fields=ext_res.missing_fields,
            )
            return None, ext_res.clarification_prompt

        t_data = ext_res.ticket_data

        # 2. Load stored escalation record if available
        if escalation_data is None and session_id:
            try:
                row = self.conn.execute(
                    "SELECT * FROM escalation_records WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                    (session_id,)
                ).fetchone()
                if row:
                    escalation_data = dict(row)
            except Exception:
                pass

        # Priority calculation
        priority = calculate_ticket_priority(
            issue_type=t_data["issue_type"],
            sentiment_data=sentiment_data,
            escalation_data=escalation_data,
            config=self.config,
        )

        # 3. Duplicate & Related detection
        dup_id, rel_ids = check_duplicate_or_related_tickets(
            conn=self.conn,
            customer_name=t_data["customer_name"],
            order_or_enrollment_id=t_data["order_or_enrollment_id"],
            product_or_course=t_data["product_or_course"],
            description=t_data["description"],
        )

        # 4. Routing
        assigned_team = route_ticket(
            issue_type=t_data["issue_type"],
            priority=priority,
            config=self.config,
            clock=self.clock,
        )

        # 5. SLA Calculation
        deadline_dt, warning_dt = calculate_sla_deadlines(
            priority=priority,
            clock=self.clock,
            config=self.config,
        )
        sla_deadline = deadline_dt.isoformat()
        sla_warning_at = warning_dt.isoformat()

        # 6. Create Ticket object
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        ticket_id = f"TICK-{uuid.uuid4().hex[:8].upper()}"

        ticket = Ticket(
            id=ticket_id,
            session_id=session_id,
            customer_name=t_data["customer_name"],
            contact_info=t_data.get("contact_info"),
            order_or_enrollment_id=t_data["order_or_enrollment_id"],
            product_or_course=t_data["product_or_course"],
            issue_type=t_data["issue_type"],
            description=t_data["description"],
            summary=t_data["summary"],
            evidence_refs=t_data.get("evidence_refs", []),
            priority=priority,
            assigned_team=assigned_team,
            status=TicketStatus.OPEN,
            sla_deadline=sla_deadline,
            sla_warning_at=sla_warning_at,
            sla_breached=False,
            sla_warning_sent=False,
            duplicate_of_ticket_id=dup_id,
            related_ticket_ids=rel_ids,
            created_at=now,
            updated_at=now,
        )

        # 7. Persist to SQLite
        self.conn.execute(
            "INSERT INTO tickets ("
            "id, session_id, customer_name, contact_info, order_or_enrollment_id, "
            "product_or_course, issue_type, description, summary, evidence_refs, "
            "priority, assigned_team, status, sla_deadline, sla_warning_at, "
            "sla_breached, sla_warning_sent, duplicate_of_ticket_id, related_ticket_ids, "
            "created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ticket.id,
                ticket.session_id,
                ticket.customer_name,
                ticket.contact_info,
                ticket.order_or_enrollment_id,
                ticket.product_or_course,
                ticket.issue_type,
                ticket.description,
                ticket.summary,
                json.dumps(ticket.evidence_refs),
                ticket.priority.value,
                ticket.assigned_team,
                ticket.status.value,
                ticket.sla_deadline,
                ticket.sla_warning_at,
                0,
                0,
                ticket.duplicate_of_ticket_id,
                json.dumps(ticket.related_ticket_ids),
                ticket.created_at,
                ticket.updated_at,
            ),
        )
        self.conn.commit()

        # 8. Audit trail
        self._log_ticket_audit(
            ticket.id,
            action="ticket_created",
            actor=actor,
            details={
                "priority": ticket.priority.value,
                "team": ticket.assigned_team,
                "duplicate_of": dup_id,
                "related_tickets": rel_ids,
                "sla_deadline": sla_deadline,
            },
        )
        self.audit.log(
            "ticket_created",
            ticket_id=ticket.id,
            session_id=session_id,
            priority=ticket.priority.value,
            assigned_team=ticket.assigned_team,
            duplicate_of=dup_id,
        )

        # 9. Enqueue background SLA warning and breach checks
        if self.job_queue:
            self.job_queue.enqueue(
                job_type="check_ticket_sla",
                payload={"ticket_id": ticket.id},
                scheduled_at=ticket.sla_warning_at,
            )

        return ticket, None

    def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Fetch ticket by ID."""
        row = self.conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if not row:
            return None
        return Ticket.from_row(row)

    def list_tickets(
        self,
        status: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Ticket]:
        """List tickets with optional status or session filter."""
        query = "SELECT * FROM tickets WHERE 1=1"
        params: List[Any] = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        return [Ticket.from_row(r) for r in rows]

    def check_sla_status(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """Evaluate ticket SLA deadline and warning against current clock.
        
        If 75% warning consumed -> mark sla_warning_sent, log audit.
        If deadline breached -> mark sla_breached, escalate ticket, log audit.
        """
        ticket = self.get_ticket(ticket_id)
        if not ticket or ticket.status in (TicketStatus.RESOLVED, TicketStatus.CLOSED):
            return None

        from datetime import datetime
        now = self.clock.now()
        # Parse ISO string
        deadline_dt = datetime.fromisoformat(ticket.sla_deadline)
        warning_dt = datetime.fromisoformat(ticket.sla_warning_at)

        res = {"ticket_id": ticket_id, "action": None}

        # Check breach first
        if now >= deadline_dt and not ticket.sla_breached:
            self.conn.execute(
                "UPDATE tickets SET sla_breached = 1, status = 'escalated', updated_at = ? WHERE id = ?",
                (now.isoformat(), ticket_id),
            )
            self.conn.commit()
            self._log_ticket_audit(
                ticket_id,
                action="sla_breached",
                actor="system_sla_monitor",
                details={"deadline": ticket.sla_deadline, "breached_at": now.isoformat()},
            )
            self.audit.log("ticket_sla_breached", ticket_id=ticket_id, deadline=ticket.sla_deadline)
            res["action"] = "breached"
            return res

        # Check 75% warning
        if now >= warning_dt and not ticket.sla_warning_sent:
            self.conn.execute(
                "UPDATE tickets SET sla_warning_sent = 1, updated_at = ? WHERE id = ?",
                (now.isoformat(), ticket_id),
            )
            self.conn.commit()
            self._log_ticket_audit(
                ticket_id,
                action="sla_warning",
                actor="system_sla_monitor",
                details={"warning_at": ticket.sla_warning_at, "triggered_at": now.isoformat()},
            )
            self.audit.log("ticket_sla_warning", ticket_id=ticket_id, warning_at=ticket.sla_warning_at)
            res["action"] = "warning"
            return res

        return res

    def process_all_due_sla_checks(self) -> List[Dict[str, Any]]:
        """Scan open tickets and run SLA checks."""
        rows = self.conn.execute(
            "SELECT id FROM tickets WHERE status NOT IN ('resolved', 'closed')"
        ).fetchall()
        actions = []
        for r in rows:
            act = self.check_sla_status(r["id"])
            if act and act.get("action"):
                actions.append(act)
        return actions

    def get_ticket_audit(self, ticket_id: str) -> List[Dict[str, Any]]:
        """Retrieve audit history for a ticket."""
        rows = self.conn.execute(
            "SELECT * FROM ticket_audit WHERE ticket_id = ? ORDER BY id ASC",
            (ticket_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_masked_handoff(self, ticket_id: str) -> Optional[HandoffSummary]:
        """Generate sanitized handoff summary for a ticket."""
        ticket = self.get_ticket(ticket_id)
        if not ticket:
            return None
        return generate_masked_handoff_summary(ticket)
