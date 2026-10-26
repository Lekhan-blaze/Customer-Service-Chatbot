"""Tests for migration runner and migration integrity."""

import sqlite3
from pathlib import Path
import pytest

from support_bot.db import MIGRATIONS_DIR, get_migration_scripts, run_migrations


def test_migration_versions_unique():
    """Verify all repository migration files have unique, strictly increasing version numbers."""
    scripts = get_migration_scripts()
    assert len(scripts) >= 8
    versions = [v for v, _ in scripts]
    # Check uniqueness
    assert len(versions) == len(set(versions)), f"Duplicate versions found: {versions}"
    # Check that versions are strictly sorted and start from 1
    assert versions == sorted(versions)
    assert versions[0] == 1


def test_migration_runner_fails_loudly_on_duplicates(tmp_path):
    """Verify that get_migration_scripts raises ValueError if duplicate versions exist."""
    mig_dir = tmp_path / "migrations"
    mig_dir.mkdir()
    (mig_dir / "001_initial.sql").write_text("CREATE TABLE t1 (id INT);")
    (mig_dir / "002_first.sql").write_text("CREATE TABLE t2 (id INT);")
    (mig_dir / "002_duplicate.sql").write_text("CREATE TABLE t3 (id INT);")

    with pytest.raises(ValueError, match="Duplicate migration version 002 detected"):
        get_migration_scripts(mig_dir)


def test_migrations_run_on_empty_db_creates_all_tables():
    """Verify clean execution on fresh DB produces schema_version 8 and expected tables."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    final_version = run_migrations(conn)
    assert final_version == 8

    # Query schema_version
    applied = [r[0] for r in conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()]
    assert applied == [1, 2, 3, 4, 5, 6, 7, 8]

    # Verify key tables exist
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
    }
    expected = {
        "schema_version", "audit_log", "config_overrides", "jobs", "roles", "user_roles",
        "conversation_sessions", "conversation_messages",
        "message_sentiment", "escalation_records",
        "kb_versions", "kb_documents", "kb_chunks", "kb_quarantine", "kb_metrics", "kb_audit_events",
        "tickets", "ticket_audit", "file_uploads",
    }
    assert expected.issubset(tables)
    conn.close()
