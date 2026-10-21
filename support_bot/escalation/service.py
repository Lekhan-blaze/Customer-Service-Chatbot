"""SentimentEscalationService — evaluate escalation rules and persist records.

Rules (evaluated in priority order):
1. High-risk category detected (account_compromise, duplicate_payment, legal_threat, data_breach)
   → immediate escalation → on-call queue if outside business hours, else human queue
2. N consecutive messages with negative/frustrated dominant label
   → escalate → on-call if urgent+outside hours, else schedule next working day if outside hours
3. Urgent sentiment outside business hours → on-call queue
4. 15-minute unresolved-negativity timeout (job-queue based)
   → escalated with rule "unresolved_negativity_timeout"
5. Low-confidence sentiment → ask clarification, do not escalate (idempotent)

Idempotency: an open (non-resolved) escalation for the same session and rule
is never duplicated.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, SystemClock, is_business_hours, next_working_day
from support_bot.config import get_config
from support_bot.db import get_connection, run_migrations
from support_bot.jobs import JobQueue
from support_bot.masking import mask_pii
from support_bot.sentiment.analyzer import SentimentAnalyzer, SentimentResult


@dataclass
class EscalationRecord:
    """In-memory representation of an escalation_records row."""

    id: int
    session_id: str
    reason: str
    rule_name: str
    calibrated_scores: dict[str, float]
    conversation_summary: str
    message_refs: list[str]
    routing_outcome: str  # human_queue | on_call | scheduled_next_working_day
    status: str  # pending | in_progress | resolved
    created_at: str
    resolved_at: str | None = None


def _load_escalations(
    conn: sqlite3.Connection,
    session_id: str,
    status_filter: str | None = None,
) -> list[EscalationRecord]:
    if status_filter:
        rows = conn.execute(
            "SELECT * FROM escalation_records WHERE session_id = ? AND status = ? ORDER BY id DESC",
            (session_id, status_filter),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM escalation_records WHERE session_id = ? ORDER BY id DESC",
            (session_id,),
        ).fetchall()
    result = []
    for r in rows:
        result.append(
            EscalationRecord(
                id=r["id"],
                session_id=r["session_id"],
                reason=r["reason"],
                rule_name=r["rule_name"],
                calibrated_scores=json.loads(r["calibrated_scores"] or "{}"),
                conversation_summary=r["conversation_summary"] or "",
                message_refs=json.loads(r["message_refs"] or "[]"),
                routing_outcome=r["routing_outcome"],
                status=r["status"],
                created_at=r["created_at"],
                resolved_at=r["resolved_at"],
            )
        )
    return result


class SentimentEscalationService:
    """Process sentiment results and apply escalation rules."""

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        clock: Clock | None = None,
        config: dict[str, Any] | None = None,
        audit: AuditLogger | None = None,
        analyzer: SentimentAnalyzer | None = None,
        auto_migrate: bool = True,
    ):
        self.conn = conn or get_connection()
        if auto_migrate:
            run_migrations(self.conn)
        self.clock = clock or SystemClock()
        self.config = config or get_config(self.conn)
        self.audit = audit or AuditLogger(self.conn, self.clock)
        self.analyzer = analyzer
        self._jobs = JobQueue(self.conn, self.clock, self.config)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def process_message(
        self,
        session_id: str,
        text: str,
        context: dict[str, Any] | None = None,
        message_id: int | None = None,
    ) -> tuple[SentimentResult, EscalationRecord | None, str | None]:
        """Analyze message sentiment and evaluate escalation rules.

        Returns:
            (sentiment_result, escalation_record_or_None, clarification_question_or_None)
        """
        if self.analyzer is None:
            from support_bot.sentiment.analyzer import GeminiSentimentAnalyzer
            self.analyzer = GeminiSentimentAnalyzer()

        sentiment = self.analyzer.analyze(text, context=context, config=self.config)
        self._persist_sentiment(session_id, text, sentiment, message_id)

        # Low-confidence: don't escalate, return clarification
        cfg_sent = self.config.get("sentiment", {})
        low_conf_threshold = cfg_sent.get("low_confidence_threshold", 0.45)
        if sentiment.low_confidence or max(sentiment.calibrated_scores.values(), default=0.0) < low_conf_threshold:
            return sentiment, None, self._clarification_question(text)

        # Evaluate rules
        escalation = self._evaluate_rules(session_id, text, sentiment, context)
        return sentiment, escalation, None

    def process_conversation_sentiment(
        self,
        session_id: str,
        recent_texts: list[str],
        context: dict[str, Any] | None = None,
    ) -> SentimentResult:
        """Aggregate sentiment over recent messages (for timeout rule etc.)."""
        if self.analyzer is None:
            from support_bot.sentiment.analyzer import GeminiSentimentAnalyzer
            self.analyzer = GeminiSentimentAnalyzer()
        return self.analyzer.analyze_conversation(recent_texts, config=self.config)

    def resolve_escalation(self, session_id: str, rule_name: str | None = None) -> int:
        """Mark open escalations for session as resolved. Returns count resolved."""
        from zoneinfo import ZoneInfo
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        if rule_name:
            cur = self.conn.execute(
                "UPDATE escalation_records SET status='resolved', resolved_at=? "
                "WHERE session_id=? AND rule_name=? AND status!='resolved'",
                (now, session_id, rule_name),
            )
        else:
            cur = self.conn.execute(
                "UPDATE escalation_records SET status='resolved', resolved_at=? "
                "WHERE session_id=? AND status!='resolved'",
                (now, session_id),
            )
        self.conn.commit()
        return cur.rowcount

    def get_escalations(self, session_id: str) -> list[EscalationRecord]:
        """Return all escalation records for a session (latest first)."""
        return _load_escalations(self.conn, session_id)

    def cancel_timeout_job(self, session_id: str) -> None:
        """Cancel a pending unresolved-negativity timeout job for the session."""
        self.conn.execute(
            "UPDATE jobs SET status='done' WHERE job_type='unresolved_negativity_timeout' "
            "AND status='pending' AND json_extract(payload,'$.session_id')=?",
            (session_id,),
        )
        self.conn.commit()

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────

    def _persist_sentiment(
        self,
        session_id: str,
        text: str,
        sentiment: SentimentResult,
        message_id: int | None,
    ) -> None:
        from zoneinfo import ZoneInfo
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        self.conn.execute(
            "INSERT INTO message_sentiment "
            "(session_id, message_id, message_text, labels, calibrated_scores, "
            " dominant_label, risk_category, risk_score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                message_id,
                mask_pii(text),
                json.dumps(sentiment.raw_scores),
                json.dumps(sentiment.calibrated_scores),
                sentiment.dominant_label,
                sentiment.risk_category,
                sentiment.risk_score,
                now,
            ),
        )
        self.conn.commit()

    def _evaluate_rules(
        self,
        session_id: str,
        text: str,
        sentiment: SentimentResult,
        context: dict[str, Any] | None,
    ) -> EscalationRecord | None:
        cfg_esc = self.config.get("escalation", {})
        if not cfg_esc.get("enabled", True):
            return None

        in_hours = is_business_hours(self.clock, self.config)
        high_risk_cats = cfg_esc.get(
            "high_risk_categories",
            ["account_compromise", "duplicate_payment", "legal_threat", "data_breach"],
        )
        risk_threshold = cfg_esc.get("risk_threshold", 0.70)
        neg_threshold = cfg_esc.get("repeated_negative_threshold", 2)
        urgent_threshold = cfg_esc.get("urgent_threshold", 0.65)
        timeout_minutes = cfg_esc.get("unresolved_negative_timeout_minutes", 15)

        # Rule 1: High-risk category
        if (
            sentiment.risk_category in high_risk_cats
            and sentiment.risk_score >= risk_threshold
        ):
            routing = "on_call" if not in_hours else "human_queue"
            return self._escalate(
                session_id,
                text,
                sentiment,
                context,
                reason=f"High-risk category detected: {sentiment.risk_category} (score={sentiment.risk_score:.2f})",
                rule_name="high_risk_category",
                routing=routing,
            )

        # Rule 2: Urgent outside business hours
        if (
            sentiment.calibrated_scores.get("urgent", 0.0) >= urgent_threshold
            and not in_hours
        ):
            return self._escalate(
                session_id,
                text,
                sentiment,
                context,
                reason="Urgent message received outside business hours",
                rule_name="urgent_outside_hours",
                routing="on_call",
            )

        # Rule 3: Repeated negative/frustrated messages
        n_negative = self._count_recent_negative(session_id, n=neg_threshold)
        if n_negative >= neg_threshold:
            routing = "scheduled_next_working_day" if not in_hours else "human_queue"
            record = self._escalate(
                session_id,
                text,
                sentiment,
                context,
                reason=f"Repeated negative sentiment: {n_negative} consecutive messages",
                rule_name="repeated_negative",
                routing=routing,
            )
            # Schedule unresolved-negativity timeout job
            if record:
                self._schedule_timeout_job(session_id, timeout_minutes)
            return record

        # Rule 4: Outside business hours (non-urgent) — schedule next working day
        if not in_hours and sentiment.dominant_label in ("negative", "frustrated"):
            return self._escalate(
                session_id,
                text,
                sentiment,
                context,
                reason="Negative sentiment outside business hours",
                rule_name="negative_outside_hours",
                routing="scheduled_next_working_day",
            )

        return None

    def _escalate(
        self,
        session_id: str,
        text: str,
        sentiment: SentimentResult,
        context: dict[str, Any] | None,
        reason: str,
        rule_name: str,
        routing: str,
    ) -> EscalationRecord | None:
        """Idempotent — skips if open escalation for same session+rule exists."""
        existing = _load_escalations(self.conn, session_id, status_filter="pending")
        existing += _load_escalations(self.conn, session_id, status_filter="in_progress")
        for e in existing:
            if e.rule_name == rule_name:
                return e  # already escalated

        from zoneinfo import ZoneInfo
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        summary = (context or {}).get("summary", "")
        cal_json = json.dumps(sentiment.calibrated_scores)
        refs_json = json.dumps([mask_pii(text[:200])])

        cur = self.conn.execute(
            "INSERT INTO escalation_records "
            "(session_id, reason, rule_name, calibrated_scores, conversation_summary, "
            " message_refs, routing_outcome, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (session_id, reason, rule_name, cal_json, summary, refs_json, routing, now),
        )
        self.conn.commit()
        row_id = cur.lastrowid

        self.audit.log(
            "escalation_created",
            decision="escalate",
            rule=rule_name,
            references={"session_id": session_id, "routing": routing, "reason": reason},
            score=sentiment.risk_score,
        )

        return EscalationRecord(
            id=row_id,
            session_id=session_id,
            reason=reason,
            rule_name=rule_name,
            calibrated_scores=sentiment.calibrated_scores,
            conversation_summary=summary,
            message_refs=[mask_pii(text[:200])],
            routing_outcome=routing,
            status="pending",
            created_at=now,
        )

    def _count_recent_negative(self, session_id: str, n: int) -> int:
        """Count how many of the last *n* messages have negative/frustrated dominant label."""
        rows = self.conn.execute(
            "SELECT dominant_label FROM message_sentiment "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
        if len(rows) < n:
            return len([r for r in rows if r["dominant_label"] in ("negative", "frustrated")])
        return sum(1 for r in rows if r["dominant_label"] in ("negative", "frustrated"))

    def _schedule_timeout_job(self, session_id: str, timeout_minutes: int) -> None:
        """Schedule a timeout escalation job if one is not already pending."""
        existing = self.conn.execute(
            "SELECT id FROM jobs WHERE job_type='unresolved_negativity_timeout' "
            "AND status='pending' AND json_extract(payload,'$.session_id')=?",
            (session_id,),
        ).fetchone()
        if existing:
            return
        self._jobs.enqueue(
            "unresolved_negativity_timeout",
            payload={"session_id": session_id},
            delay_minutes=timeout_minutes,
        )

    def _clarification_question(self, text: str) -> str:
        """Generic low-confidence clarification prompt."""
        return (
            "I want to make sure I understand how you're feeling. "
            "Could you tell me a bit more about your concern so I can help you best?"
        )
