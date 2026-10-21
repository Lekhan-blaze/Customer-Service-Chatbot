"""Shared pytest fixtures for support_bot tests."""

import sqlite3
import pytest

from support_bot.clock import FakeClock
from support_bot.container import Container
from support_bot.db import run_migrations


@pytest.fixture
def fake_clock():
    """A FakeClock frozen at Mon 2025-01-06 10:00 IST (a working day)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return FakeClock(datetime(2025, 1, 6, 10, 0, 0, tzinfo=ZoneInfo("Asia/Kolkata")))


@pytest.fixture
def db_conn():
    """An in-memory SQLite connection with all migrations applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    yield conn
    conn.close()


@pytest.fixture
def container(fake_clock, db_conn):
    """A Container wired with FakeClock and an in-memory database."""
    c = Container(clock=fake_clock, conn=db_conn, auto_migrate=False)
    return c
