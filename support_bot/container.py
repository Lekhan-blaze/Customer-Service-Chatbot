"""Dependency-injection container — wires up all foundation services."""

from __future__ import annotations

import sqlite3

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, SystemClock
from support_bot.config import get_config, load_defaults
from support_bot.db import get_connection, run_migrations
from support_bot.jobs import JobQueue


class Container:
    """Simple service container / factory.

    Modules receive their dependencies from this container instead of
    importing globals.  In tests, pass a ``FakeClock`` and an in-memory
    SQLite connection.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        conn: sqlite3.Connection | None = None,
        db_path: str | None = None,
        auto_migrate: bool = True,
    ):
        self.clock: Clock = clock or SystemClock()
        self.conn: sqlite3.Connection = conn or get_connection(db_path)
        self.db = self.conn
        if auto_migrate:
            run_migrations(self.conn)
        self._config_cache: dict | None = None
        self.audit = AuditLogger(self.conn, self.clock)
        self.job_queue = JobQueue(self.conn, self.clock, self.config)
        self.jobs = self.job_queue

    @property
    def config(self) -> dict:
        """Merged YAML defaults + SQLite overrides (cached per container)."""
        if self._config_cache is None:
            self._config_cache = get_config(self.conn)
        return self._config_cache

    def reload_config(self) -> dict:
        """Force-reload configuration from disk and database."""
        self._config_cache = get_config(self.conn)
        return self._config_cache

    def get_conversation_service(self, analyzer=None):
        """Return a ConversationService wired with this container's services."""
        from support_bot.conversation.service import ConversationService
        return ConversationService(
            conn=self.conn,
            clock=self.clock,
            config=self.config,
            audit=self.audit,
            analyzer=analyzer,
            auto_migrate=False,
        )

    @property
    def conversation(self):
        """Cached ConversationService instance."""
        if not hasattr(self, "_conversation_service") or self._conversation_service is None:
            self._conversation_service = self.get_conversation_service()
        return self._conversation_service

    def get_knowledge_pipeline(self, embedder=None):
        """Return KnowledgePipeline wired with container services."""
        from support_bot.knowledge.pipeline import KnowledgePipeline
        return KnowledgePipeline(
            conn=self.conn,
            clock=self.clock,
            config=self.config,
            audit=self.audit,
            job_queue=self.job_queue,
            embedder=embedder,
        )

    def get_rag_service(self, pipeline=None, claim_validator=None):
        """Return RagService wired with container services."""
        from support_bot.rag.service import RagService
        pipe = pipeline or self.get_knowledge_pipeline()
        return RagService(
            pipeline=pipe,
            clock=self.clock,
            audit=self.audit,
            claim_validator=claim_validator,
        )


    def get_ticket_service(self):
        """Return TicketService wired with container dependencies."""
        from support_bot.tickets.service import TicketService
        return TicketService(
            conn=self.conn,
            clock=self.clock,
            config=self.config,
            audit=self.audit,
            job_queue=self.job_queue,
            auto_migrate=False,
        )

    def get_file_service(self, ocr_engine=None, storage_root: str = "storage/uploads"):
        """Return FileService wired with container dependencies."""
        from support_bot.files.service import FileService
        return FileService(
            conn=self.conn,
            clock=self.clock,
            config=self.config,
            audit=self.audit,
            job_queue=self.job_queue,
            ocr_engine=ocr_engine,
            storage_root=storage_root,
            auto_migrate=False,
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

