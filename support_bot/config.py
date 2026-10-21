"""Runtime configuration — YAML defaults merged with SQLite overrides."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.yaml"


def load_defaults(path: Path | str | None = None) -> dict:
    """Load the YAML defaults file and return it as a dict."""
    p = Path(path) if path else _DEFAULTS_PATH
    with open(p) as f:
        return yaml.safe_load(f) or {}


def get_override(conn: sqlite3.Connection, key: str) -> Any | None:
    """Return a single runtime override from the config_overrides table."""
    row = conn.execute(
        "SELECT value FROM config_overrides WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    raw = row["value"] if isinstance(row, sqlite3.Row) else row[0]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def set_override(conn: sqlite3.Connection, key: str, value: Any) -> None:
    """Insert or update a runtime override."""
    serialized = json.dumps(value) if not isinstance(value, str) else value
    conn.execute(
        "INSERT INTO config_overrides (key, value, updated) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated = excluded.updated",
        (key, serialized),
    )
    conn.commit()


def get_config(conn: sqlite3.Connection | None, defaults: dict | None = None) -> dict:
    """Return merged configuration: YAML defaults with SQLite overrides on top.

    If *conn* is None, only defaults are returned (useful in tests without a db).
    """
    cfg = dict(defaults) if defaults else load_defaults()
    if conn is None:
        return cfg
    try:
        rows = conn.execute("SELECT key, value FROM config_overrides").fetchall()
    except sqlite3.OperationalError:
        return cfg
    for row in rows:
        key = row["key"] if isinstance(row, sqlite3.Row) else row[0]
        raw = row["value"] if isinstance(row, sqlite3.Row) else row[1]
        try:
            cfg[key] = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            cfg[key] = raw
    return cfg
