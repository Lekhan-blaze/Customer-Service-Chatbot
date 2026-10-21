"""Structured audit logging — JSON records to SQLite and a log file."""

from __future__ import annotations

import json
import logging
from typing import Any

from support_bot.clock import Clock

logger = logging.getLogger("support_bot.audit")


class AuditLogger:
    """Write structured audit records to the database and Python logging."""

    def __init__(self, conn, clock: Clock):
        self._conn = conn
        self._clock = clock

    def log(
        self,
        event: str,
        *,
        decision: str | None = None,
        rule: str | None = None,
        references: dict | list | None = None,
        **extra: Any,
    ) -> int:
        """Record an audit event. Returns the row id."""
        from zoneinfo import ZoneInfo

        ts = self._clock.now(ZoneInfo("UTC")).isoformat()
        refs_json = json.dumps(references) if references else None
        meta_json = json.dumps(extra) if extra else None

        cur = self._conn.execute(
            "INSERT INTO audit_log (timestamp, event, decision, rule, "
            '  "references", meta) VALUES (?, ?, ?, ?, ?, ?)',
            (ts, event, decision, rule, refs_json, meta_json),
        )
        self._conn.commit()
        row_id = cur.lastrowid

        # Also emit to Python logging (for file/console sinks)
        record = {
            "id": row_id,
            "timestamp": ts,
            "event": event,
            "decision": decision,
            "rule": rule,
            "references": references,
            **extra,
        }
        logger.info(json.dumps(record, default=str))
        return row_id

    def query(
        self, event: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Retrieve recent audit records, optionally filtered by event type."""
        if event:
            rows = self._conn.execute(
                "SELECT * FROM audit_log WHERE event = ? ORDER BY id DESC LIMIT ?",
                (event, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
