"""SentimentEscalationService — evaluate escalation rules and persist records.

Rules (evaluated in priority order):
1. High-risk category detected (account_compromise, duplicate_payment, legal_threat, data_breach)
   → immediate escalation → on-call queue if outside business hours, else human queue
2. Urgent sentiment outside business hours → on-call queue
3. N consecutive messages with negative/frustrated dominant label → human queue (in hours)
   or schedule next working day (outside hours)
4. Negative sentiment outside business hours → schedule next working day
5. 15-minute unresolved-negativity timeout — scheduled at the FIRST negative/frustrated
   message in a conversation (not when repeated-negative fires). Cancelled when the
   conversation is "resolved" (see _is_resolved below).

Definition of "resolved" (cancels the timeout job and any open repeated_negative escalation):
  - A later positive or neutral dominant-label message from the customer, OR
  - An explicit customer confirmation (intent "confirmed" or text matching confirmation
    keywords), OR
  - A call to resolve_escalation() from application code (e.g., agent marks it resolved).

Idempotency: an open (non-resolved) escalation for the same session and rule
is never duplicated.

Customer-facing messages (customer_message field):
  Make no promises about outcomes, refunds, or timings beyond what the config allows.
  They only acknowledge the concern and state that a team member will follow up.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, SystemClock, is_business_hours, next_working_day
from support_bot.config import get_config
from support_bot.db import get_connection, run_migrations
from support_bot.jobs import JobQueue
from support_bot.masking import mask_pii
from support_bot.sentiment.analyzer import SentimentAnalyzer, SentimentResult

# Keywords that signal customer confirmation / resolution
_CONFIRMATION_PATTERN = re.compile(
    r"\b(thank you|thanks|got it|perfect|resolved|solved|understood|sorted|okay|ok)\b",
    re.IGNORECASE,
)

# Negative qualifiers that negate confirmation
_NEGATIVE_QUALIFIERS = re.compile(
    r"\b(not working|still not|nothing|for nothing|useless|waste|terrible|broken|worst|unresolved|problem|issue|failed|cannot|can't)\b",
    re.IGNORECASE,
)

# Labels considered "negative" for the timer and repeated-negative rule
_NEGATIVE_LABELS = frozenset(["negative", "frustrated"])


def _is_resolved(
    sentiment: SentimentResult,
    text: str,
    config: dict[str, Any] | None = None,
) -> bool:
    """Return True if this message genuinely signals that the customer's concern is resolved.

    Strict Criteria:
      1. Explicit confirmation phrase is present in the text (e.g. 'thanks', 'thank you', 'got it', 'resolved', 'ok').
         A neutral or positive label alone DOES NOT resolve a conversation.
      2. No negative qualifiers present in the text (e.g. 'for nothing', 'still not working').
      3. No negative or frustrated dominant label.
      4. Sarcasm score is strictly below sarcasm_threshold (default 0.60).
      5. Negative and frustrated calibrated scores are low (< 0.45).
      6. No risk signal (risk_score < 0.30).
    """
    cfg = config or {}
    sarcasm_threshold = cfg.get("sentiment", {}).get("sarcasm_threshold", 0.60)

    # 1. Must contain explicit confirmation
    if not _CONFIRMATION_PATTERN.search(text):
        return False

    # 2. Must not contain negative qualifiers
    if _NEGATIVE_QUALIFIERS.search(text):
        return False

    # 3. Dominant label cannot be negative or frustrated
    if sentiment.dominant_label in _NEGATIVE_LABELS:
        return False

    # 4. Sarcasm score must be below threshold
    if sentiment.calibrated_scores.get("sarcastic", 0.0) >= sarcasm_threshold:
        return False

    # 5. Negative and frustrated scores must be low
    if sentiment.calibrated_scores.get("negative", 0.0) >= 0.45:
        return False
    if sentiment.calibrated_scores.get("frustrated", 0.0) >= 0.45:
        return False

    # 6. Risk score must be low
    if sentiment.risk_score >= 0.30:
        return False

    return True


@dataclass
class EscalationRecord:
    """In-memory representation of an escalation_records row."""

    id: int
    session_id: str
    reason: str
    rule_name: str
    calibrated_scores: dict[str, float]
    conversation_summary: str
    message_refs: list[int | str]
    routing_outcome: str  # human_queue | on_call | scheduled_next_working_day
    status: str           # pending | in_progress | resolved
    created_at: str
    resolved_at: str | None = None
    customer_message: str | None = None
    message_snippets: list[str] = field(default_factory=list)


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
                customer_message=r["customer_message"] if "customer_message" in r.keys() else None,
                message_snippets=json.loads(r["message_snippets"] or "[]") if "message_snippets" in r.keys() and r["message_snippets"] else [],
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

        cfg_esc = self.config.get("escalation", {})
        high_risk_cats = cfg_esc.get(
            "high_risk_categories",
            ["account_compromise", "duplicate_payment", "legal_threat", "data_breach"],
        )
        cat_thresholds = cfg_esc.get("risk_thresholds", {
            "account_compromise": 0.70,
            "duplicate_payment": 0.45,
            "legal_threat": 0.45,
            "data_breach": 0.60,
            "default": cfg_esc.get("risk_threshold", 0.60),
        })
        risk_clarification_min = cfg_esc.get("risk_clarification_min", 0.30)

        # Borderline risk: between risk_clarification_min and category threshold -> ask short clarification
        if sentiment.risk_category in high_risk_cats:
            cat_thresh = cat_thresholds.get(
                sentiment.risk_category,
                cat_thresholds.get("default", 0.60)
            )
            if risk_clarification_min <= sentiment.risk_score < cat_thresh:
                return sentiment, None, self._risk_clarification_question(sentiment.risk_category, text)

        # If this message signals resolution: cancel timer, resolve open escalations
        if _is_resolved(sentiment, text, config=self.config):
            self.cancel_timeout_job(session_id)
            self.resolve_escalation(session_id, rule_name="repeated_negative")
            self.resolve_escalation(session_id, rule_name="unresolved_negativity_timeout")

        # 15-minute timer: start at FIRST negative/frustrated message
        if sentiment.dominant_label in _NEGATIVE_LABELS:
            timeout_minutes = cfg_esc.get("unresolved_negative_timeout_minutes", 15)
            self._schedule_timeout_job(session_id, timeout_minutes)

        # Evaluate escalation rules
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

    def handle_timeout_job(self, payload: dict | None) -> EscalationRecord | None:
        """Handler for 'unresolved_negativity_timeout' job.

        Triggered when 15 minutes have passed since the first negative message
        without resolution. Idempotent: does not duplicate if repeated_negative
        or another timeout escalation is already open.
        """
        if not payload or "session_id" not in payload:
            return None
        session_id = payload["session_id"]

        existing_open = _load_escalations(self.conn, session_id, status_filter="pending")
        existing_open += _load_escalations(self.conn, session_id, status_filter="in_progress")
        if any(
            r.rule_name in ("unresolved_negativity_timeout", "repeated_negative")
            for r in existing_open
        ):
            return None

        # Check latest sentiment in database
        latest = self.conn.execute(
            "SELECT dominant_label, message_text FROM message_sentiment "
            "WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()

        if not latest or latest["dominant_label"] not in _NEGATIVE_LABELS:
            return None  # Conversation was resolved or no negative message

        # Escalate
        in_hours = is_business_hours(self.clock, self.config)
        routing = "human_queue" if in_hours else "scheduled_next_working_day"
        msg = self._customer_message(routing)

        sent_rows = self.conn.execute(
            "SELECT id, message_text FROM message_sentiment WHERE session_id = ? ORDER BY id DESC LIMIT 5",
            (session_id,),
        ).fetchall()
        msg_ids = [r["id"] for r in reversed(sent_rows)]
        if not msg_ids:
            msg_ids = [1]
        snippets = [mask_pii(r["message_text"][:200]) for r in reversed(sent_rows)] if sent_rows else [""]

        cal_scores = {"negative": 0.8, "frustrated": 0.8}
        cal_json = json.dumps(cal_scores)
        summary = "Negative sentiment unresolved for 15 minutes"
        refs_json = json.dumps(msg_ids)
        snippets_json = json.dumps(snippets)
        now = self.clock.now(ZoneInfo("UTC")).isoformat()

        cur = self.conn.execute(
            "INSERT INTO escalation_records "
            "(session_id, reason, rule_name, calibrated_scores, conversation_summary, "
            " message_refs, routing_outcome, status, created_at, customer_message, message_snippets) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
            (
                session_id,
                "Negative sentiment unresolved for 15 minutes",
                "unresolved_negativity_timeout",
                cal_json,
                summary,
                refs_json,
                routing,
                now,
                msg,
                snippets_json,
            ),
        )
        self.conn.commit()
        record_id = cur.lastrowid
        self.audit.log(
            "escalation_created",
            decision="escalate",
            rule="unresolved_negativity_timeout",
            session_id=session_id,
            routing_outcome=routing,
        )
        return EscalationRecord(
            id=record_id,
            session_id=session_id,
            reason="Negative sentiment unresolved for 15 minutes",
            rule_name="unresolved_negativity_timeout",
            calibrated_scores={"negative": 0.8, "frustrated": 0.8},
            conversation_summary=summary,
            message_refs=msg_ids,
            routing_outcome=routing,
            status="pending",
            created_at=now,
            customer_message=msg,
            message_snippets=snippets,
        )

    def process_due_jobs(self) -> list[int]:
        """Process any pending jobs that are due according to the clock."""
        handlers = {
            "unresolved_negativity_timeout": self.handle_timeout_job,
        }
        return self._jobs.tick(handlers)

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

        # Rule 1: High-risk category
        cat_thresholds = cfg_esc.get("risk_thresholds", {
            "account_compromise": 0.70,
            "duplicate_payment": 0.45,
            "legal_threat": 0.45,
            "data_breach": 0.60,
            "default": cfg_esc.get("risk_threshold", 0.60),
        })
        if sentiment.risk_category in high_risk_cats:
            threshold = cat_thresholds.get(
                sentiment.risk_category,
                cat_thresholds.get("default", cfg_esc.get("risk_threshold", 0.60))
            )
            if sentiment.risk_score >= threshold:
                routing = "on_call" if not in_hours else "human_queue"
                msg = self._customer_message(routing)
                return self._escalate(
                    session_id, text, sentiment, context,
                    reason=f"High-risk category detected: {sentiment.risk_category} (score={sentiment.risk_score:.2f})",
                    rule_name="high_risk_category",
                    routing=routing,
                    customer_message=msg,
                )

        # Rule 2: Urgent outside business hours
        if (
            sentiment.calibrated_scores.get("urgent", 0.0) >= urgent_threshold
            and not in_hours
        ):
            msg = self._customer_message("on_call")
            return self._escalate(
                session_id, text, sentiment, context,
                reason="Urgent message received outside business hours",
                rule_name="urgent_outside_hours",
                routing="on_call",
                customer_message=msg,
            )

        # Rule 3: Repeated negative/frustrated messages
        n_negative = self._count_recent_negative(session_id, n=neg_threshold)
        if n_negative >= neg_threshold:
            routing = "scheduled_next_working_day" if not in_hours else "human_queue"
            msg = self._customer_message(routing)
            return self._escalate(
                session_id, text, sentiment, context,
                reason=f"Repeated negative sentiment: {n_negative} consecutive messages",
                rule_name="repeated_negative",
                routing=routing,
                customer_message=msg,
            )

        # Rule 4: Outside business hours (non-urgent) — schedule next working day
        if not in_hours and sentiment.dominant_label in _NEGATIVE_LABELS:
            msg = self._customer_message("scheduled_next_working_day")
            return self._escalate(
                session_id, text, sentiment, context,
                reason="Negative sentiment outside business hours",
                rule_name="negative_outside_hours",
                routing="scheduled_next_working_day",
                customer_message=msg,
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
        customer_message: str | None = None,
    ) -> EscalationRecord | None:
        """Idempotent — skips if open escalation for same session+rule exists."""
        existing = _load_escalations(self.conn, session_id, status_filter="pending")
        existing += _load_escalations(self.conn, session_id, status_filter="in_progress")
        for e in existing:
            if e.rule_name == rule_name:
                return e  # already escalated under this rule
            if rule_name in ("repeated_negative", "unresolved_negativity_timeout") and e.rule_name in (
                "repeated_negative",
                "unresolved_negativity_timeout",
            ):
                return e  # already escalated under negativity

        from zoneinfo import ZoneInfo
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        summary = (context or {}).get("summary", "")
        cal_json = json.dumps(sentiment.calibrated_scores)

        # Get recent message IDs and snippets from message_sentiment
        sent_rows = self.conn.execute(
            "SELECT id, message_text FROM message_sentiment WHERE session_id = ? ORDER BY id DESC LIMIT 5",
            (session_id,),
        ).fetchall()
        msg_ids = [r["id"] for r in reversed(sent_rows)]
        if not msg_ids:
            msg_ids = [1]
        snippets = [mask_pii(r["message_text"][:200]) for r in reversed(sent_rows)] if sent_rows else [mask_pii(text[:200])]

        refs_json = json.dumps(msg_ids)
        snippets_json = json.dumps(snippets)

        cur = self.conn.execute(
            "INSERT INTO escalation_records "
            "(session_id, reason, rule_name, calibrated_scores, conversation_summary, "
            " message_refs, routing_outcome, status, created_at, customer_message, message_snippets) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
            (session_id, reason, rule_name, cal_json, summary, refs_json, routing, now, customer_message, snippets_json),
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
            message_refs=msg_ids,
            routing_outcome=routing,
            status="pending",
            created_at=now,
            customer_message=customer_message,
            message_snippets=snippets,
        )

    def _count_recent_negative(self, session_id: str, n: int) -> int:
        """Count how many of the last *n* messages have negative/frustrated dominant label."""
        rows = self.conn.execute(
            "SELECT dominant_label FROM message_sentiment "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, n),
        ).fetchall()
        if len(rows) < n:
            return len([r for r in rows if r["dominant_label"] in _NEGATIVE_LABELS])
        return sum(1 for r in rows if r["dominant_label"] in _NEGATIVE_LABELS)

    def _schedule_timeout_job(self, session_id: str, timeout_minutes: int) -> None:
        """Schedule timeout escalation job if one is not already pending.

        Called at the FIRST negative/frustrated message in a conversation.
        """
        existing = self.conn.execute(
            "SELECT id FROM jobs WHERE job_type='unresolved_negativity_timeout' "
            "AND status='pending' AND json_extract(payload,'$.session_id')=?",
            (session_id,),
        ).fetchone()
        if existing:
            return  # already scheduled — don't reschedule
        self._jobs.enqueue(
            "unresolved_negativity_timeout",
            payload={"session_id": session_id},
            delay_minutes=timeout_minutes,
        )

    def _customer_message(self, routing: str) -> str:
        """Safe customer-facing escalation message. No outcome promises, no timings."""
        msgs = {
            "on_call": (
                "We've noted your concern and a team member will look into it. "
                "We aim to follow up as quickly as we can."
            ),
            "human_queue": (
                "We've noted your concern and it has been passed to our support team. "
                "Someone will be in touch with you."
            ),
            "scheduled_next_working_day": (
                "We've noted your concern. Our team is currently outside working hours "
                "and will follow up on the next working day."
            ),
        }
        return msgs.get(
            routing,
            "We've noted your concern and will follow up with you.",
        )

    def _clarification_question(self, text: str) -> str:
        """Generic low-confidence clarification prompt."""
        return (
            "I want to make sure I understand how you're feeling. "
            "Could you tell me a bit more about your concern so I can help you best?"
        )

    def _risk_clarification_question(self, category: str, text: str) -> str:
        """Short clarification question when a potential risk is detected with borderline confidence."""
        prompts = {
            "duplicate_payment": "It sounds like you may have experienced a duplicate or extra charge. Could you please confirm if your account was charged more than once?",
            "account_compromise": "It sounds like you may be experiencing an account security issue. Could you clarify if you suspect unauthorized access or login to your account?",
            "legal_threat": "We noticed a mention of legal counsel or consumer forum. Could you please specify your concern so we can connect you with the appropriate team?",
            "data_breach": "We take data privacy very seriously. Could you please clarify if you observed any exposed or unauthorized information?",
        }
        return prompts.get(
            category,
            "Could you please provide a few more details so we can understand your concern and direct you to the right team?",
        )
