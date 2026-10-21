"""SQLite-backed job queue — delayed, scheduled, retryable, after-hours jobs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable

from support_bot.clock import Clock


@dataclass
class Job:
    """In-memory representation of a job row."""

    id: int
    job_type: str
    payload: dict | None
    status: str
    scheduled_at: str
    attempts: int
    max_attempts: int
    last_error: str | None = None


def enqueue(
    conn: sqlite3.Connection,
    clock: Clock,
    job_type: str,
    payload: dict | None = None,
    *,
    delay_minutes: int = 0,
    scheduled_at: str | None = None,
    max_attempts: int = 4,
) -> int:
    """Add a job to the queue. Returns the job id.

    *delay_minutes* offsets from now; *scheduled_at* is an explicit ISO
    timestamp. If both are 0/None the job is scheduled for immediate
    execution.
    """
    from zoneinfo import ZoneInfo
    from datetime import timedelta

    now = clock.now(ZoneInfo("UTC"))
    if scheduled_at:
        sched = scheduled_at
    elif delay_minutes:
        sched = (now + timedelta(minutes=delay_minutes)).isoformat()
    else:
        sched = now.isoformat()

    cur = conn.execute(
        "INSERT INTO jobs (job_type, payload, scheduled_at, max_attempts) "
        "VALUES (?, ?, ?, ?)",
        (job_type, json.dumps(payload) if payload else None, sched, max_attempts),
    )
    conn.commit()
    return cur.lastrowid


def _fetch_ready(conn: sqlite3.Connection, now_iso: str) -> list[Job]:
    """Fetch jobs that are ready to run (pending and scheduled_at <= now)."""
    rows = conn.execute(
        "SELECT * FROM jobs WHERE status = 'pending' AND scheduled_at <= ? "
        "ORDER BY scheduled_at",
        (now_iso,),
    ).fetchall()
    return [
        Job(
            id=r["id"],
            job_type=r["job_type"],
            payload=json.loads(r["payload"]) if r["payload"] else None,
            status=r["status"],
            scheduled_at=r["scheduled_at"],
            attempts=r["attempts"],
            max_attempts=r["max_attempts"],
            last_error=r["last_error"],
        )
        for r in rows
    ]


def tick(
    conn: sqlite3.Connection,
    clock: Clock,
    handlers: dict[str, Callable[[dict | None], Any]],
    *,
    backoff_minutes: list[int] | None = None,
) -> list[int]:
    """Process ready jobs — call the matching handler for each.

    Returns a list of job ids that were processed (success or final failure).
    This function is designed to be called from a worker loop, or driven
    manually step-by-step in tests via ``FakeClock.advance()``.
    """
    from zoneinfo import ZoneInfo
    from datetime import timedelta

    backoff = backoff_minutes or [15, 30, 60]
    now = clock.now(ZoneInfo("UTC"))
    ready = _fetch_ready(conn, now.isoformat())
    processed: list[int] = []

    for job in ready:
        handler = handlers.get(job.job_type)
        if handler is None:
            conn.execute(
                "UPDATE jobs SET status = 'failed', last_error = ?, "
                "finished_at = ? WHERE id = ?",
                (f"No handler for '{job.job_type}'", now.isoformat(), job.id),
            )
            conn.commit()
            processed.append(job.id)
            continue

        # Mark as running
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ?, "
            "attempts = attempts + 1 WHERE id = ?",
            (now.isoformat(), job.id),
        )
        conn.commit()

        try:
            handler(job.payload)
            conn.execute(
                "UPDATE jobs SET status = 'done', finished_at = ? WHERE id = ?",
                (now.isoformat(), job.id),
            )
            conn.commit()
        except Exception as exc:
            new_attempts = job.attempts + 1
            if new_attempts >= job.max_attempts:
                conn.execute(
                    "UPDATE jobs SET status = 'failed', last_error = ?, "
                    "finished_at = ? WHERE id = ?",
                    (str(exc), now.isoformat(), job.id),
                )
            else:
                # Schedule retry with backoff
                idx = min(new_attempts - 1, len(backoff) - 1)
                retry_at = now + timedelta(minutes=backoff[idx])
                conn.execute(
                    "UPDATE jobs SET status = 'pending', last_error = ?, "
                    "scheduled_at = ? WHERE id = ?",
                    (str(exc), retry_at.isoformat(), job.id),
                )
            conn.commit()

        processed.append(job.id)

    return processed


def get_job(conn: sqlite3.Connection, job_id: int) -> Job | None:
    """Fetch a single job by id."""
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return Job(
        id=row["id"],
        job_type=row["job_type"],
        payload=json.loads(row["payload"]) if row["payload"] else None,
        status=row["status"],
        scheduled_at=row["scheduled_at"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        last_error=row["last_error"],
    )
