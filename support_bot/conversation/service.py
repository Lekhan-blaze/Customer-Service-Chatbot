"""ConversationService orchestrating multilingual analysis, session management, and response formatting."""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, SystemClock
from support_bot.config import get_config
from support_bot.conversation.analyzer import AnalysisResult, GeminiAnalyzer, MessageAnalyzer
from support_bot.conversation.entities import ProtectedText, protect_entities, restore_entities
from support_bot.conversation.session import Session, SessionManager
from support_bot.db import get_connection, run_migrations
from support_bot.injection import detect_injection
from support_bot.masking import mask_pii

logger = logging.getLogger("support_bot.conversation")


@dataclass
class ConversationResponse:
    """Result returned by ConversationService.handle_message."""

    session_id: str
    language: str
    languages: list[str]
    script: str
    intents: list[str]
    confidences: dict[str, float]
    normalized_query: str
    entities: dict[str, str]
    protected_text: ProtectedText
    needs_clarification: bool = False
    clarification_question: str | None = None
    is_injection: bool = False
    context: dict[str, Any] = field(default_factory=dict)


class ConversationService:
    """High-level service interface for conversation, translation, and session flows."""

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        clock: Clock | None = None,
        config: dict[str, Any] | None = None,
        audit: AuditLogger | None = None,
        analyzer: MessageAnalyzer | None = None,
        auto_migrate: bool = True,
    ):
        self.conn = conn or get_connection()
        if auto_migrate:
            run_migrations(self.conn)
        self.clock = clock or SystemClock()
        self.config = config or get_config(self.conn)
        self.audit = audit or AuditLogger(self.conn, self.clock)
        self.analyzer = analyzer or GeminiAnalyzer()
        self.session_manager = SessionManager(
            conn=self.conn,
            clock=self.clock,
            config=self.config,
            analyzer=self.analyzer,
        )

    def handle_message(
        self,
        session_id: str,
        text: str,
        user_id: str | None = None,
    ) -> ConversationResponse:
        """Process incoming user message through security, session, entity, and analysis pipelines."""
        conv_cfg = self.config.get("conversation", {})
        inj_cfg = self.config.get("injection_detection", {})

        # 1. Untrusted message check: Run prompt-injection detector
        if inj_cfg.get("enabled", True):
            threshold = inj_cfg.get("threshold", 0.6)
            inj_result = detect_injection(text, threshold=threshold)
            if inj_result.is_risky:
                self.audit.log(
                    "prompt_injection_blocked",
                    decision="rejected",
                    rule="injection_detector",
                    references={"session_id": session_id, "reasons": inj_result.reasons},
                    score=inj_result.score,
                    text=mask_pii(text),
                )
                safe_warning = (
                    "I cannot process instructions that attempt to override my security guidelines. "
                    "How may I help you with our courses or services?"
                )
                return ConversationResponse(
                    session_id=session_id,
                    language="English",
                    languages=["English"],
                    script="latin",
                    intents=["security_violation"],
                    confidences={"security_violation": 1.0},
                    normalized_query="",
                    entities={},
                    protected_text=ProtectedText(text=text),
                    needs_clarification=True,
                    clarification_question=safe_warning,
                    is_injection=True,
                )

        # 2. Resolve / restore session based on inactivity and restore windows
        session = self.session_manager.resolve_session(session_id, customer_id=user_id)

        # 3. Protect entities before normalization/translation
        protected = protect_entities(text, extra_entities=session.facts)

        # 4. Context compilation
        recent = self.session_manager.get_recent_messages(session.session_id, limit=5)
        context = {
            "session_id": session.session_id,
            "summary": session.summary,
            "facts": session.facts,
            "recent_messages": recent,
        }

        # 5. Model-based analysis: language, script, intents, corrections, normalization
        analysis = self.analyzer.analyze(protected.text, context=context, config=self.config)

        # Merge extracted entities
        merged_entities = dict(protected.entity_map)
        if analysis.entities:
            merged_entities.update(analysis.entities)

        # 6. Customer corrections handling
        if analysis.is_correction and analysis.correction_value:
            field_name = analysis.correction_field or "order_id"
            raw_val = analysis.correction_value
            if raw_val in protected.entity_map:
                raw_val = protected.entity_map[raw_val]
            self.session_manager.update_facts(
                session.session_id,
                {field_name: raw_val},
            )
            self.audit.log(
                "customer_correction",
                decision="fact_updated",
                rule="user_correction",
                references={
                    "session_id": session.session_id,
                    "field": field_name,
                    "new_value": mask_pii(str(raw_val)),
                },
            )

        # 7. Low-confidence handling: check thresholds
        lang_thresh = conv_cfg.get("language_confidence_threshold", 0.60)
        intent_thresh = conv_cfg.get("intent_confidence_threshold", 0.60)
        min_intent_conf = (
            min(analysis.intent_confidences.values())
            if analysis.intent_confidences
            else 1.0
        )

        needs_clarification = (
            analysis.language_confidence < lang_thresh
            or min_intent_conf < intent_thresh
        )

        clarification_q = None
        if needs_clarification:
            clarification_q = (
                analysis.clarification_question
                or self._build_clarification(analysis.primary_language)
            )
            # Record user message & assistant clarification
            self.session_manager.add_message(
                session.session_id,
                "user",
                text,
                language=analysis.primary_language,
                intents=analysis.intents,
            )
            self.session_manager.add_message(
                session.session_id,
                "assistant",
                clarification_q,
                language=analysis.primary_language,
                intents=["clarification_request"],
            )

        # Save user message when no clarification is needed
        if not needs_clarification:
            self.session_manager.add_message(
                session.session_id,
                "user",
                text,
                language=analysis.primary_language,
                intents=analysis.intents,
            )

        return ConversationResponse(
            session_id=session.session_id,
            language=analysis.primary_language,
            languages=analysis.languages,
            script=analysis.script,
            intents=analysis.intents,
            confidences={
                "language": analysis.language_confidence,
                **analysis.intent_confidences,
            },
            normalized_query=analysis.normalized_english,
            entities=merged_entities,
            protected_text=protected,
            needs_clarification=needs_clarification,
            clarification_question=clarification_q,
            context=context,
        )

    def format_reply(
        self,
        conv_response: ConversationResponse,
        english_answer: str,
    ) -> str:
        """Translate retrieved answer back to customer language and restore protected entities."""
        # 1. Translate if necessary
        if conv_response.language.lower() != "english":
            translated = self.analyzer.translate(
                english_answer,
                conv_response.language,
                context=conv_response.context,
            )
        else:
            translated = english_answer

        # 2. Restore protected entities verbatim
        final_answer = conv_response.protected_text.restore(translated)

        # 3. Store assistant message (with PII masked in storage)
        self.session_manager.add_message(
            conv_response.session_id,
            "assistant",
            final_answer,
            language=conv_response.language,
            intents=["answer"],
        )

        # 4. Periodically update conversation summary
        recent = self.session_manager.get_recent_messages(conv_response.session_id, limit=6)
        if len(recent) >= 4:
            new_summary = self.analyzer.summarize(
                recent,
                conv_response.context.get("summary"),
            )
            self.session_manager.update_summary(conv_response.session_id, new_summary)

        return final_answer

    def _build_clarification(self, language: str) -> str:
        """Short, non-guessing clarification question in customer's most likely language."""
        lang_lower = language.lower()
        if lang_lower == "hindi":
            return "कृपया स्पष्ट करें कि आप कोर्स विवरण चाहते हैं या फीस और भुगतान के बारे में पूछ रहे हैं?"
        elif lang_lower == "tamil":
            return "தயவுசெய்து தெளிவுபடுத்தவும்: நீங்கள் பாடப்பிரிவு விவரங்கள் கேட்கிறீர்களா அல்லது கட்டணம் பற்றியதா?"
        elif lang_lower == "kannada":
            return "ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟಪಡಿಸಿ: ನೀವು ಕೋರ್ಸ್ ವಿವರಗಳನ್ನು ಅಥವಾ ಶುಲ್ಕದ ಬಗ್ಗೆ ಕೇಳುತ್ತಿದ್ದೀರಾ?"
        return "Could you please clarify if you are asking about course details or fee payment?"

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Fetch session context including summary, facts, and recent messages."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {}
        recent = self.session_manager.get_recent_messages(session_id)
        return {
            "session_id": session.session_id,
            "customer_id": session.customer_id,
            "language": session.language,
            "summary": session.summary,
            "facts": session.facts,
            "status": session.status,
            "recent_messages": recent,
        }

    def get_summary(self, session_id: str) -> str:
        """Return the durable conversation summary."""
        session = self.session_manager.get_session(session_id)
        return session.summary if session and session.summary else ""

    def end_session(self, session_id: str) -> None:
        """Mark the session as ended."""
        self.session_manager.end_session(session_id)
