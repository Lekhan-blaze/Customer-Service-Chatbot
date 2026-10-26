"""Tests for support_bot.container and support_bot.db integration with temporary SQLite databases."""

import sqlite3
from support_bot.container import Container
from support_bot.db import get_connection, run_migrations


def test_container_dependency_injection(container, fake_clock):
    assert container.clock is fake_clock
    assert container.conn is not None
    assert container.db is container.conn
    assert container.audit is not None
    assert container.job_queue is not None
    assert container.jobs is not None
    assert container.config["company_timezone"] == "Asia/Kolkata"


def test_container_file_based_sqlite(tmp_path, fake_clock):
    db_file = tmp_path / "test_support_bot.db"
    
    # 1. Initialize container on a fresh file
    c1 = Container(clock=fake_clock, db_path=str(db_file), auto_migrate=True)
    c1.audit.log("test_event", decision="approved")
    c1.close()

    # 2. Re-open container on same file
    c2 = Container(clock=fake_clock, db_path=str(db_file), auto_migrate=True)
    records = c2.audit.query()
    assert len(records) == 1
    assert records[0]["event"] == "test_event"
    assert records[0]["decision"] == "approved"
    c2.close()


def test_migrations_idempotent(db_conn):
    v1 = run_migrations(db_conn)
    assert v1 >= 1
    # Running migrations again should be a no-op and return same version
    v2 = run_migrations(db_conn)
    assert v2 == v1


def test_container_services_resolution(container):
    t_svc = container.get_ticket_service()
    assert t_svc is not None
    f_svc = container.get_file_service()
    assert f_svc is not None
    kp = container.get_knowledge_pipeline()
    assert kp is not None
    rag = container.get_rag_service(pipeline=kp)
    assert rag is not None
