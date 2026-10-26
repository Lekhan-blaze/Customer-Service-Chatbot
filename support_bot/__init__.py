"""support_bot — shared foundation for the customer-service chatbot."""

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, FakeClock, SystemClock, add_business_time, is_business_hours, next_working_day
from support_bot.config import get_config, get_override, load_defaults, set_override
from support_bot.container import Container
from support_bot.db import get_connection, run_migrations
from support_bot.injection import InjectionResult, detect_injection
from support_bot.jobs import Job, JobQueue, enqueue, enqueue_after_hours, get_job, tick
from support_bot.masking import MaskingFilter, mask_pii
from support_bot.conversation import (
    ConversationResponse,
    ConversationService,
)
from support_bot.tickets import TicketService, Ticket, TicketPriority, TicketStatus
from support_bot.files import FileService, FileUploadRecord, OcrResult, ConflictReport
from support_bot.rbac import (
    DEFAULT_ROLES,
    assign_role,
    get_user_permissions,
    has_permission,
    require_permission,
    revoke_role,
)

__all__ = [
    "AuditLogger",
    "Clock",
    "FakeClock",
    "SystemClock",
    "add_business_time",
    "is_business_hours",
    "next_working_day",
    "get_config",
    "get_override",
    "load_defaults",
    "set_override",
    "Container",
    "ConversationResponse",
    "ConversationService",
    "get_connection",
    "run_migrations",
    "InjectionResult",
    "detect_injection",
    "Job",
    "JobQueue",
    "enqueue",
    "enqueue_after_hours",
    "get_job",
    "tick",
    "MaskingFilter",
    "mask_pii",
    "DEFAULT_ROLES",
    "assign_role",
    "get_user_permissions",
    "has_permission",
    "require_permission",
    "revoke_role",
    "TicketService",
    "Ticket",
    "TicketPriority",
    "TicketStatus",
    "FileService",
    "FileUploadRecord",
    "OcrResult",
    "ConflictReport",
]
