"""Tests for Module 2: Support Tickets, Routing, Duplicate Detection, and SLA Tracking."""

from datetime import datetime, timedelta
import json
import sqlite3
import pytest
from zoneinfo import ZoneInfo

from support_bot.audit import AuditLogger
from support_bot.clock import FakeClock
from support_bot.db import run_migrations
from support_bot.jobs import JobQueue
from support_bot.tickets.models import TicketPriority, TicketStatus
from support_bot.tickets.service import TicketService


_WEDNESDAY_10AM = datetime(2026, 4, 15, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
_FRIDAY_5PM = datetime(2026, 4, 17, 17, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
_SUNDAY_10AM = datetime(2026, 4, 19, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))


@pytest.fixture
def ticket_env():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)

    clock = FakeClock(_WEDNESDAY_10AM)
    config = {
        "company_timezone": "Asia/Kolkata",
        "business_hours": {
            "start": "09:00",
            "end": "18:00",
            "days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        },
        "holidays": ["2026-04-16"],  # Thursday 16 April is a holiday
        "tickets": {
            "mandatory_fields": [
                "customer_name",
                "order_or_enrollment_id",
                "product_or_course",
                "issue_type",
                "description",
            ],
            "sla_hours_by_priority": {
                "P1": 4,
                "P2": 8,
                "P3": 24,
                "P4": 48,
            },
            "warning_threshold_ratio": 0.75,
            "teams": {
                "billing": {
                    "skills": ["duplicate_payment", "refund", "invoice", "fee"],
                    "available": True,
                    "max_workload": 5,
                    "current_workload": 0,
                },
                "technical": {
                    "skills": ["course_access", "video_error", "login"],
                    "available": True,
                    "max_workload": 5,
                    "current_workload": 0,
                },
            },
            "fallback_team": "general_support",
        },
    }

    audit = AuditLogger(conn, clock)
    job_queue = JobQueue(conn, clock, config)
    svc = TicketService(
        conn=conn,
        clock=clock,
        config=config,
        audit=audit,
        job_queue=job_queue,
        auto_migrate=False,
    )
    return {"svc": svc, "conn": conn, "clock": clock, "config": config}


def test_missing_mandatory_fields_asks_for_clarification(ticket_env):
    svc: TicketService = ticket_env["svc"]

    # Missing customer name and order ID
    messages = [
        {"sender": "user", "text": "I was charged twice for Python course! Please help."},
    ]
    ticket, clarification = svc.create_ticket_from_conversation(
        session_id="sess_missing",
        messages=messages,
    )
    assert ticket is None
    assert clarification is not None
    assert "your name" in clarification
    assert "enrollment ID or order number" in clarification


def test_successful_ticket_creation_and_routing(ticket_env):
    svc: TicketService = ticket_env["svc"]

    messages = [
        {"sender": "user", "text": "My name is Alice Smith. Order ORD-998811 for Python course was charged twice!"},
    ]
    ticket, clarification = svc.create_ticket_from_conversation(
        session_id="sess_ok",
        messages=messages,
    )
    assert clarification is None
    assert ticket is not None
    assert ticket.customer_name == "Alice Smith"
    assert ticket.order_or_enrollment_id == "ORD-998811"
    assert ticket.product_or_course == "Python"
    assert ticket.issue_type == "duplicate_payment"
    assert ticket.priority == TicketPriority.P1  # duplicate payment is high risk P1
    assert ticket.assigned_team == "billing"      # matches billing skills
    assert ticket.status == TicketStatus.OPEN


def test_unavailable_team_falls_back_to_general_support(ticket_env):
    svc: TicketService = ticket_env["svc"]
    # Mark billing team as unavailable in config
    ticket_env["config"]["tickets"]["teams"]["billing"]["available"] = False

    messages = [
        {"sender": "user", "text": "My name is Bob. Order ORD-12345 for Python was charged twice."},
    ]
    ticket, _ = svc.create_ticket_from_conversation("sess_unavail", messages)
    assert ticket is not None
    assert ticket.assigned_team == "general_support"  # Fallback


def test_duplicate_ticket_linked_to_original(ticket_env):
    svc: TicketService = ticket_env["svc"]

    messages1 = [
        {"sender": "user", "text": "My name is John. Order ORD-777 for Python course charged twice."},
    ]
    t1, _ = svc.create_ticket_from_conversation("sess_1", messages1)
    assert t1 is not None
    assert t1.duplicate_of_ticket_id is None

    # Customer sends identical issue again
    messages2 = [
        {"sender": "user", "text": "My name is John. Order ORD-777 for Python course charged twice again."},
    ]
    t2, _ = svc.create_ticket_from_conversation("sess_2", messages2)
    assert t2 is not None
    assert t2.duplicate_of_ticket_id == t1.id


def test_related_vs_unrelated_issues(ticket_env):
    svc: TicketService = ticket_env["svc"]

    # 1. First ticket: billing issue for ORD-888 Python
    messages1 = [
        {"sender": "user", "text": "My name is Carol. Order ORD-888 for Python was charged twice."},
    ]
    t1, _ = svc.create_ticket_from_conversation("sess_c1", messages1)

    # 2. Second ticket: same order ORD-888 Python, but completely different issue: video playback error
    messages2 = [
        {"sender": "user", "text": "My name is Carol. Order ORD-888 for Python course video error playback is broken black screen."},
    ]
    t2, _ = svc.create_ticket_from_conversation("sess_c2", messages2)
    assert t2.duplicate_of_ticket_id is None
    assert t1.id in t2.related_ticket_ids  # Grouped as related!

    # 3. Third ticket: completely different order and course
    messages3 = [
        {"sender": "user", "text": "My name is Dave. Order ORD-555 for Data Science cannot access course."},
    ]
    t3, _ = svc.create_ticket_from_conversation("sess_d1", messages3)
    assert t3.duplicate_of_ticket_id is None
    assert len(t3.related_ticket_ids) == 0  # Unrelated


def test_weekend_and_holiday_skipping_in_sla_deadline(ticket_env):
    svc: TicketService = ticket_env["svc"]
    clock: FakeClock = ticket_env["clock"]

    # Start: Friday 5:00 PM (17:00). Business day ends at 18:00 (1 hour remaining).
    # P1 ticket has 4 hours SLA.
    # Friday 17:00 to 18:00 consumes 1 hour. 3 hours remain.
    # Saturday & Sunday are non-working days.
    # Next working day is Monday 20 April at 09:00 AM.
    # 3 hours into Monday -> deadline must be Monday 20 April at 12:00 PM (12:00).
    clock.set(_FRIDAY_5PM)

    messages = [
        {"sender": "user", "text": "My name is Eve. Order ORD-999 for Python charged twice."},
    ]
    t, _ = svc.create_ticket_from_conversation("sess_fri", messages)
    assert t is not None

    deadline_dt = datetime.fromisoformat(t.sla_deadline)
    # Target date: Monday 2026-04-20 at 12:00
    assert deadline_dt.strftime("%Y-%m-%d %H:%M") == "2026-04-20 12:00"


def test_sla_warning_at_75_percent_and_breach_escalation(ticket_env):
    svc: TicketService = ticket_env["svc"]
    clock: FakeClock = ticket_env["clock"]

    # Wednesday 10:00 AM. P1 ticket -> 4 business hours SLA (ends at 14:00).
    # 75% warning ratio -> 3 business hours (at 13:00).
    clock.set(_WEDNESDAY_10AM)

    messages = [
        {"sender": "user", "text": "My name is Frank. Order ORD-333 for Python charged twice."},
    ]
    t, _ = svc.create_ticket_from_conversation("sess_sla", messages)

    # 1. At 11:00 AM (1h elapsed) -> No warning, no breach
    clock.advance(hours=1)
    act1 = svc.check_sla_status(t.id)
    assert act1["action"] is None
    updated_t = svc.get_ticket(t.id)
    assert updated_t.sla_warning_sent is False
    assert updated_t.sla_breached is False

    # 2. At 13:05 PM (3h 5m elapsed > 75%) -> SLA warning sent
    clock.advance(hours=2, minutes=5)
    act2 = svc.check_sla_status(t.id)
    assert act2["action"] == "warning"
    updated_t = svc.get_ticket(t.id)
    assert updated_t.sla_warning_sent is True
    assert updated_t.sla_breached is False

    # 3. At 14:05 PM (4h 5m elapsed > deadline) -> SLA breach and auto-escalation
    clock.advance(hours=1)
    act3 = svc.check_sla_status(t.id)
    assert act3["action"] == "breached"
    updated_t = svc.get_ticket(t.id)
    assert updated_t.sla_breached is True
    assert updated_t.status == TicketStatus.ESCALATED


def test_runtime_sla_config_change(ticket_env):
    svc: TicketService = ticket_env["svc"]
    clock: FakeClock = ticket_env["clock"]
    clock.set(_WEDNESDAY_10AM)

    # Modify P1 SLA at runtime from 4 hours to 2 hours
    ticket_env["config"]["tickets"]["sla_hours_by_priority"]["P1"] = 2

    messages = [
        {"sender": "user", "text": "My name is Grace. Order ORD-111 for Python charged twice."},
    ]
    t, _ = svc.create_ticket_from_conversation("sess_cfg", messages)
    deadline_dt = datetime.fromisoformat(t.sla_deadline)
    # 10:00 AM + 2 hours -> 12:00 PM
    assert deadline_dt.strftime("%H:%M") == "12:00"


def test_masked_handoff_summary(ticket_env):
    svc: TicketService = ticket_env["svc"]

    messages = [
        {"sender": "user", "text": "My name is Helen. Order ORD-444 for Python charged twice on card 4111222233334444! Contact me at helen@example.com or 555-123-4567."},
    ]
    t, _ = svc.create_ticket_from_conversation("sess_hand", messages)

    handoff = svc.get_masked_handoff(t.id)
    assert handoff is not None
    # Credit card masked
    assert "4111222233334444" not in handoff.masked_summary
    assert "4111" in handoff.masked_summary or "CARD" in handoff.masked_summary or "..." in handoff.masked_summary
    # Email and phone masked
    assert "helen@example.com" not in handoff.masked_summary
    assert "555-123-4567" not in handoff.masked_summary
