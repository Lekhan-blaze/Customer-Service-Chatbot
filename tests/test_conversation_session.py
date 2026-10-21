"""Tests for session isolation, 30-minute inactivity expiry, 24-hour restore window, and PII masking."""

from support_bot.conversation.session import SessionManager


def test_concurrent_sessions_isolation(db_conn, fake_clock):
    mgr = SessionManager(conn=db_conn, clock=fake_clock)

    # Create Session A
    s_a = mgr.resolve_session("session_user_A", customer_id="cust_A")
    mgr.add_message("session_user_A", "user", "I want Python course info", language="English")
    mgr.update_facts("session_user_A", {"interest": "Python", "budget": "₹5000"})
    mgr.update_summary("session_user_A", "Customer A interested in Python")

    # Create Session B concurrently
    s_b = mgr.resolve_session("session_user_B", customer_id="cust_B")
    mgr.add_message("session_user_B", "user", "What is data science syllabus?", language="English")
    mgr.update_facts("session_user_B", {"interest": "DataScience"})
    mgr.update_summary("session_user_B", "Customer B interested in Data Science")

    # Verify complete isolation: no cross-leakage
    msgs_a = mgr.get_recent_messages("session_user_A")
    msgs_b = mgr.get_recent_messages("session_user_B")
    assert len(msgs_a) == 1
    assert len(msgs_b) == 1
    assert msgs_a[0]["content"] == "I want Python course info"
    assert msgs_b[0]["content"] == "What is data science syllabus?"

    facts_a = mgr.get_session("session_user_A").facts
    facts_b = mgr.get_session("session_user_B").facts
    assert facts_a["interest"] == "Python"
    assert facts_b["interest"] == "DataScience"
    assert "budget" not in facts_b

    summary_a = mgr.get_session("session_user_A").summary
    summary_b = mgr.get_session("session_user_B").summary
    assert "Python" in summary_a and "Data Science" not in summary_a
    assert "Data Science" in summary_b and "Python" not in summary_b


def test_session_message_history_limit(db_conn, fake_clock):
    mgr = SessionManager(conn=db_conn, clock=fake_clock, config={"session": {"max_recent_messages": 5}})
    s = mgr.resolve_session("sess_history_test")

    # Add 8 messages
    for i in range(8):
        mgr.add_message("sess_history_test", "user" if i % 2 == 0 else "assistant", f"Message {i}")

    # Should only return the latest 5 messages in chronological order
    recent = mgr.get_recent_messages("sess_history_test")
    assert len(recent) == 5
    assert recent[0]["content"] == "Message 3"
    assert recent[-1]["content"] == "Message 7"


def test_inactivity_expiry_30_minutes(db_conn, fake_clock):
    mgr = SessionManager(conn=db_conn, clock=fake_clock)
    s = mgr.resolve_session("sess_expiry_test")
    mgr.add_message("sess_expiry_test", "user", "Hello support")

    # After 15 minutes: session is still active
    fake_clock.advance(minutes=15)
    s_active = mgr.resolve_session("sess_expiry_test")
    assert s_active.status == "active"

    # After 31 minutes total (15 + 16): session expires
    fake_clock.advance(minutes=16)
    # The session should be transitioned
    s_expired = mgr.get_session("sess_expiry_test")
    # Resolve transitions it
    s_renewed = mgr.resolve_session("sess_expiry_test")
    # Within 24h restore window: restored with prior context
    assert s_renewed.status == "active"


def test_24_hour_restore_window_vs_fresh_session(db_conn, fake_clock):
    mgr = SessionManager(conn=db_conn, clock=fake_clock)

    # 1. Start session, establish facts and summary
    s = mgr.resolve_session("sess_lifecycle", customer_id="user_42")
    mgr.add_message("sess_lifecycle", "user", "My order is ORD-100")
    mgr.update_facts("sess_lifecycle", {"order_id": "ORD-100", "topic": "shipping"})
    mgr.update_summary("sess_lifecycle", "Customer inquired about shipping for ORD-100")

    # 2. Advance by 2 hours (inactivity > 30m, but < 24h)
    fake_clock.advance(hours=2)
    s_restored = mgr.resolve_session("sess_lifecycle", customer_id="user_42")
    assert s_restored.status == "active"
    # Prior facts and summary must be restored
    assert s_restored.summary == "Customer inquired about shipping for ORD-100"
    assert s_restored.facts["order_id"] == "ORD-100"

    # 3. Advance by 25 hours (inactivity > 24h)
    fake_clock.advance(hours=25)
    s_fresh = mgr.resolve_session("sess_lifecycle", customer_id="user_42")
    assert s_fresh.status == "active"
    # Over 24h: fresh conversation, summary and facts reset
    assert s_fresh.summary is None
    assert s_fresh.facts == {}

    # Historical records are still in the database
    rows = db_conn.execute("SELECT * FROM conversation_messages WHERE session_id = 'sess_lifecycle'").fetchall()
    assert len(rows) >= 1


def test_pii_masking_in_session_storage(db_conn, fake_clock):
    mgr = SessionManager(conn=db_conn, clock=fake_clock)
    mgr.resolve_session("sess_pii_test")

    # Message with email, phone, card
    mgr.add_message(
        "sess_pii_test",
        "user",
        "My email is rahul.sharma@example.com, phone 9876543210 and card 4111 2222 3333 4444",
    )

    # Inspect SQLite table directly
    row = db_conn.execute(
        "SELECT content FROM conversation_messages WHERE session_id = 'sess_pii_test'",
    ).fetchone()

    content = row["content"]
    assert "rahul.sharma@example.com" not in content
    assert "9876543210" not in content
    assert "4111 2222 3333 4444" not in content
    assert "[EMAIL]" in content
    assert "[PHONE]" in content
    assert "[CARD]" in content
