"""Tests for support_bot.jobs — SQLite job queue, retries, backoff, and after-hours scheduling."""

from datetime import timedelta
from support_bot.jobs import (
    JobQueue,
    enqueue,
    enqueue_after_hours,
    get_job,
    tick,
)


def test_immediate_job_execution(db_conn, fake_clock):
    executed = []

    def handle_test(payload):
        executed.append(payload)

    job_id = enqueue(db_conn, fake_clock, "send_email", {"to": "user@example.com"})
    assert job_id > 0

    processed = tick(db_conn, fake_clock, {"send_email": handle_test})
    assert processed == [job_id]
    assert executed == [{"to": "user@example.com"}]

    job = get_job(db_conn, job_id)
    assert job.status == "done"
    assert job.attempts == 1


def test_delayed_job_execution(db_conn, fake_clock):
    executed = []
    job_id = enqueue(
        db_conn,
        fake_clock,
        "cleanup_session",
        {"session_id": "s123"},
        delay_minutes=30,
    )

    # At t=0, job should NOT run
    processed = tick(db_conn, fake_clock, {"cleanup_session": executed.append})
    assert processed == []
    assert get_job(db_conn, job_id).status == "pending"

    # Advance clock by 29 minutes — still not ready
    fake_clock.advance(minutes=29)
    processed = tick(db_conn, fake_clock, {"cleanup_session": executed.append})
    assert processed == []

    # Advance 2 more minutes (total 31 min) — now ready!
    fake_clock.advance(minutes=2)
    processed = tick(db_conn, fake_clock, {"cleanup_session": executed.append})
    assert processed == [job_id]
    assert len(executed) == 1
    assert get_job(db_conn, job_id).status == "done"


def test_retry_timing_and_backoff(db_conn, fake_clock):
    attempts_seen = []

    def failing_handler(payload):
        attempts_seen.append(fake_clock.now().isoformat())
        raise RuntimeError("Transient network error")

    # max_attempts=4 (1 initial + 3 retries with backoff [15, 30, 60])
    job_id = enqueue(
        db_conn,
        fake_clock,
        "sync_data",
        {"id": 1},
        max_attempts=4,
    )

    # 1st attempt at t=0 -> fails, next retry scheduled in 15 mins
    tick(db_conn, fake_clock, {"sync_data": failing_handler}, backoff_minutes=[15, 30, 60])
    job = get_job(db_conn, job_id)
    assert job.status == "pending"
    assert job.attempts == 1
    assert "Transient network error" in job.last_error

    # At t=10 min -> not ready
    fake_clock.advance(minutes=10)
    tick(db_conn, fake_clock, {"sync_data": failing_handler}, backoff_minutes=[15, 30, 60])
    assert get_job(db_conn, job_id).attempts == 1

    # At t=15 min -> 2nd attempt -> fails, next retry in 30 mins
    fake_clock.advance(minutes=5)
    tick(db_conn, fake_clock, {"sync_data": failing_handler}, backoff_minutes=[15, 30, 60])
    job = get_job(db_conn, job_id)
    assert job.attempts == 2

    # At t=45 min (15 + 30) -> 3rd attempt -> fails, next retry in 60 mins
    fake_clock.advance(minutes=30)
    tick(db_conn, fake_clock, {"sync_data": failing_handler}, backoff_minutes=[15, 30, 60])
    job = get_job(db_conn, job_id)
    assert job.attempts == 3

    # At t=105 min (45 + 60) -> 4th attempt -> fails permanently (max_attempts reached)
    fake_clock.advance(minutes=60)
    tick(db_conn, fake_clock, {"sync_data": failing_handler}, backoff_minutes=[15, 30, 60])
    job = get_job(db_conn, job_id)
    assert job.status == "failed"
    assert job.attempts == 4
    assert len(attempts_seen) == 4


def test_after_hours_scheduling(db_conn, fake_clock):
    # fake_clock is at Mon 10:00 IST
    cfg = {
        "company_timezone": "Asia/Kolkata",
        "maintenance_window": {"start": "02:00", "end": "04:00"},
    }
    job_id = enqueue_after_hours(db_conn, fake_clock, cfg, "reindex_vector_db")
    job = get_job(db_conn, job_id)

    # Should be scheduled for tomorrow 02:00 IST (since 10:00 > 02:00 today)
    assert "2025-01-06T20:30:00" in job.scheduled_at or "2025-01-07" in job.scheduled_at

    # Should not execute now
    processed = tick(db_conn, fake_clock, {"reindex_vector_db": lambda p: None})
    assert processed == []


def test_job_queue_class_wrapper(container):
    executed = []
    q = container.job_queue

    job_id = q.enqueue("task_x", {"a": 1})
    assert job_id > 0

    processed = q.tick({"task_x": executed.append})
    assert processed == [job_id]
    assert executed == [{"a": 1}]
    assert q.get(job_id).status == "done"
