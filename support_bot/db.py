"""Database helpers — SQLite connection management and migration runner."""

import os
import re
import sqlite3
from pathlib import Path
from typing import List, Tuple

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


def get_migration_scripts(migrations_dir: Path | None = None) -> List[Tuple[int, Path]]:
    """Return list of (version, script_path) sorted by version.
    
    Raises ValueError loudly if duplicate versions are detected.
    """
    directory = migrations_dir or MIGRATIONS_DIR
    scripts = sorted(directory.glob("*.sql"))
    seen_versions: dict[int, Path] = {}
    parsed: List[Tuple[int, Path]] = []

    for script in scripts:
        match = re.match(r"^(\d+)_", script.name)
        if not match:
            continue
        version = int(match.group(1))
        if version in seen_versions:
            raise ValueError(
                f"Duplicate migration version {version:03d} detected: "
                f"'{script.name}' conflicts with '{seen_versions[version].name}'"
            )
        seen_versions[version] = script
        parsed.append((version, script))

    parsed.sort(key=lambda x: x[0])
    return parsed


def run_migrations(conn: sqlite3.Connection, migrations_dir: Path | None = None) -> int:
    """Apply all pending forward-only migration scripts. Return new version.
    
    Fails loudly on duplicate version numbers.
    """
    current = _current_version(conn)
    migration_scripts = get_migration_scripts(migrations_dir)
    
    for version, script in migration_scripts:
        if version <= current:
            continue
        sql = script.read_text()
        conn.executescript(sql)
        # Ensure schema_version is recorded even if script didn't explicitly insert
        try:
            conn.execute("INSERT OR IGNORE INTO schema_version (version) VALUES (?)", (version,))
        except sqlite3.OperationalError:
            pass
    conn.commit()
    return _current_version(conn)
