"""Database helpers — SQLite connection management and migration runner."""

import os
import sqlite3
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
DEFAULT_DB_PATH = os.environ.get("SUPPORT_BOT_DB", "support_bot.db")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and foreign keys enabled."""
    path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _current_version(conn: sqlite3.Connection) -> int:
    """Return the highest applied migration version, or 0 if none."""
    try:
        row = conn.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ).fetchone()
        return row["v"] or 0
    except sqlite3.OperationalError:
        return 0


def run_migrations(conn: sqlite3.Connection) -> int:
    """Apply all pending forward-only migration scripts. Return new version."""
    current = _current_version(conn)
    scripts = sorted(MIGRATIONS_DIR.glob("*.sql"))
    applied = 0
    for script in scripts:
        # Extract version number from filename like "001_initial.sql"
        version = int(script.stem.split("_")[0])
        if version <= current:
            continue
        sql = script.read_text()
        conn.executescript(sql)
        applied += 1
    conn.commit()
    return _current_version(conn)
