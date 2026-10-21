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

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
