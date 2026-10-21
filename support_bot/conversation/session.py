"""Session and message persistence in SQLite with isolation and lifecycle management."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from support_bot.clock import Clock
from support_bot.conversation.analyzer import MessageAnalyzer
from support_bot.masking import mask_pii


@dataclass
class Session:
    """Represents an active or historic conversation session."""

    session_id: str
    customer_id: str | None = None
    language: str = "English"
    summary: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    status: str = "active"  # active | expired | ended
    created_at: str = ""
    updated_at: str = ""
    ended_at: str | None = None


class SessionManager:
    """Manages session creation, restoration, expiry, and message storage."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: Clock,
        config: dict[str, Any] | None = None,
        analyzer: MessageAnalyzer | None = None,
    ):
        self.conn = conn
        self.clock = clock
        self.config = config or {}
        self.analyzer = analyzer

    def _now_dt(self) -> datetime:
        return self.clock.now(ZoneInfo("UTC"))

    def _now_iso(self) -> str:
        return self._now_dt().isoformat()

    def get_session(self, session_id: str) -> Session | None:
        """Fetch a session record by id."""
        row = self.conn.execute(
            "SELECT * FROM conversation_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return None
        return Session(
            session_id=row["session_id"],
            customer_id=row["customer_id"],
            language=row["language"],
            summary=row["summary"],
            facts=json.loads(row["facts"]) if row["facts"] else {},
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            ended_at=row["ended_at"],
        )

    def resolve_session(
        self,
        session_id: str,
        customer_id: str | None = None,
    ) -> Session:
        """Resolve or transition a session based on inactivity and restore windows.

        Lifecycle rules:
        - Active session <= inactivity_timeout (default 30m): return session.
        - Inactivity > 30m and <= restore_window (default 24h): expire old session,
          create/restore a new active session with the prior conversation summary and facts.
        - Inactivity > 24h: expire old session, start a completely fresh session (prior
          records retained in database).
        """
        now = self._now_dt()
        now_iso = self._now_iso()

        # Configurable timeouts
        session_cfg = self.config.get("session", {})
        inactivity_min = session_cfg.get("inactivity_timeout_minutes", 30)
        restore_window_hrs = session_cfg.get("restore_window_hours", 24)

        existing = self.get_session(session_id)

        if existing is not None:
            updated_dt = datetime.fromisoformat(existing.updated_at)
            diff_seconds = (now - updated_dt).total_seconds()
            diff_minutes = diff_seconds / 60.0
            diff_hours = diff_seconds / 3600.0

            if existing.status == "active":
                if diff_minutes <= inactivity_min:
                    # Still active
                    return existing

                # Session expired due to inactivity
                self.conn.execute(
                    "UPDATE conversation_sessions SET status = 'expired', ended_at = ? WHERE session_id = ?",
                    (now_iso, existing.session_id),
                )
                self.conn.commit()

            # Session was expired or ended — check restore window
            if diff_hours <= restore_window_hrs:
                # Restore prior summary and facts into a renewed active session
                restored_id = f"{session_id}_renewed" if existing.status == "expired" else session_id
                # Or reuse session_id by updating it to active if same client key
                new_session = Session(
                    session_id=session_id,
                    customer_id=customer_id or existing.customer_id,
                    language=existing.language,
                    summary=existing.summary,
                    facts=existing.facts,
                    status="active",
                    created_at=now_iso,
                    updated_at=now_iso,
                )
                self.conn.execute(
                    "INSERT INTO conversation_sessions (session_id, customer_id, language, summary, facts, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 'active', ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET status = 'active', summary = excluded.summary, facts = excluded.facts, updated_at = excluded.updated_at, ended_at = NULL",
                    (
                        new_session.session_id,
                        new_session.customer_id,
                        new_session.language,
                        new_session.summary,
                        json.dumps(new_session.facts),
                        now_iso,
                        now_iso,
                    ),
                )
                self.conn.commit()
                return new_session
            else:
                # Outside 24h restore window: fresh session
                fresh = Session(
                    session_id=session_id,
                    customer_id=customer_id or existing.customer_id,
                    language="English",
                    summary=None,
                    facts={},
                    status="active",
                    created_at=now_iso,
                    updated_at=now_iso,
                )
                self.conn.execute(
                    "INSERT INTO conversation_sessions (session_id, customer_id, language, summary, facts, status, created_at, updated_at) "
                    "VALUES (?, ?, 'English', NULL, '{}', 'active', ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET status = 'active', language = 'English', summary = NULL, facts = '{}', updated_at = excluded.updated_at, ended_at = NULL",
                    (
                        fresh.session_id,
                        fresh.customer_id,
                        now_iso,
                        now_iso,
                    ),
                )
                self.conn.commit()
                return fresh

        # First-time new session
        new_s = Session(
            session_id=session_id,
            customer_id=customer_id,
            language="English",
            summary=None,
            facts={},
            status="active",
            created_at=now_iso,
            updated_at=now_iso,
        )
        self.conn.execute(
            "INSERT INTO conversation_sessions (session_id, customer_id, language, summary, facts, status, created_at, updated_at) "
            "VALUES (?, ?, 'English', NULL, '{}', 'active', ?, ?)",
            (session_id, customer_id, now_iso, now_iso),
        )
        self.conn.commit()
        return new_s

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        language: str | None = None,
        intents: list[str] | None = None,
    ) -> int:
        """Record a message with PII and payment data masked."""
        now_iso = self._now_iso()
        # Mask PII in storage
        masked_content = mask_pii(content)
        intents_json = json.dumps(intents) if intents else None

        cur = self.conn.execute(
            "INSERT INTO conversation_messages (session_id, role, content, language, intents, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, masked_content, language, intents_json, now_iso),
        )
        # Touch session updated_at
        lang_update = ", language = ?" if language else ""
        params = [now_iso]
        if language:
            params.append(language)
        params.append(session_id)
        self.conn.execute(
            f"UPDATE conversation_sessions SET updated_at = ?{lang_update} WHERE session_id = ?",
            params,
        )
        self.conn.commit()
        return cur.lastrowid

    def get_recent_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the latest N messages for this session in chronological order."""
        max_msgs = limit or self.config.get("session", {}).get("max_recent_messages", 10)
        rows = self.conn.execute(
            "SELECT * FROM ("
            "  SELECT id, session_id, role, content, language, intents, created_at "
            "  FROM conversation_messages WHERE session_id = ? "
            "  ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (session_id, max_msgs),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "language": r["language"],
                "intents": json.loads(r["intents"]) if r["intents"] else [],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def update_facts(self, session_id: str, new_facts: dict[str, Any]) -> dict[str, Any]:
        """Merge verified facts (e.g. order numbers, course interests) into session."""
        session = self.get_session(session_id)
        if not session:
            session = self.resolve_session(session_id)
        facts = dict(session.facts)
        facts.update(new_facts)
        self.conn.execute(
            "UPDATE conversation_sessions SET facts = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(facts), self._now_iso(), session_id),
        )
        self.conn.commit()
        return facts

    def update_summary(self, session_id: str, summary: str) -> None:
        """Store conversation summary for durable context retention."""
        session = self.get_session(session_id)
        if not session:
            self.resolve_session(session_id)
        self.conn.execute(
            "UPDATE conversation_sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
            (summary, self._now_iso(), session_id),
        )
        self.conn.commit()

    def end_session(self, session_id: str) -> None:
        """Mark session as ended."""
        self.conn.execute(
            "UPDATE conversation_sessions SET status = 'ended', ended_at = ? WHERE session_id = ?",
            (self._now_iso(), session_id),
        )
        self.conn.commit()
