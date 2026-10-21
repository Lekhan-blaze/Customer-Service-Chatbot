"""Tests for support_bot.audit — structured JSON audit logging to SQLite and logs."""

import json
from support_bot.audit import AuditLogger


def test_audit_log_and_query(db_conn, fake_clock):
    logger = AuditLogger(db_conn, fake_clock)

    # Log an event
    row_id = logger.log(
        "query_processed",
        decision="answered",
        rule="faq_match",
        references=["faq_row_12", "faq_row_13"],
        user_id="user_123",
        duration_ms=45,
    )
    assert row_id == 1

    # Log a second event
    logger.log(
        "escalation_triggered",
        decision="escalated_to_human",
        rule="sentiment_negative",
        ticket_id="TICK-999",
    )

    # Query all events
    records = logger.query()
    assert len(records) == 2
    # Records ordered DESC by id
    assert records[0]["event"] == "escalation_triggered"
    assert records[1]["event"] == "query_processed"
    assert records[1]["decision"] == "answered"
    assert records[1]["rule"] == "faq_match"

    refs = json.loads(records[1]["references"])
    assert refs == ["faq_row_12", "faq_row_13"]

    meta = json.loads(records[1]["meta"])
    assert meta["user_id"] == "user_123"
    assert meta["duration_ms"] == 45


def test_audit_query_filter(db_conn, fake_clock):
    logger = AuditLogger(db_conn, fake_clock)
    logger.log("login_success", user="alice")
    logger.log("login_failure", user="bob")
    logger.log("login_success", user="charlie")

    failures = logger.query(event="login_failure")
    assert len(failures) == 1
    assert failures[0]["event"] == "login_failure"

    successes = logger.query(event="login_success")
    assert len(successes) == 2
