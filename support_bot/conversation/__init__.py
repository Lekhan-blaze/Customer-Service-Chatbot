"""Conversation and multilingual session management package."""

from support_bot.conversation.analyzer import (
    AnalysisResult,
    FakeAnalyzer,
    GeminiAnalyzer,
    MessageAnalyzer,
)
from support_bot.conversation.entities import (
    ProtectedText,
    protect_entities,
    restore_entities,
)
from support_bot.conversation.service import (
    ConversationResponse,
    ConversationService,
)
from support_bot.conversation.session import Session, SessionManager

__all__ = [
    "AnalysisResult",
    "FakeAnalyzer",
    "GeminiAnalyzer",
    "MessageAnalyzer",
    "ProtectedText",
    "protect_entities",
    "restore_entities",
    "Session",
    "SessionManager",
    "ConversationResponse",
    "ConversationService",
]
