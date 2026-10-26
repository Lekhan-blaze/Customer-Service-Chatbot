"""RBAC helper — role and permission checks with a decorator."""

from __future__ import annotations

import functools
import json
import sqlite3
from typing import Callable


# ── Built-in role definitions (also seeded in migration 001) ────────────
DEFAULT_ROLES: dict[str, list[str]] = {
    "viewer": ["read"],
    "ingestor": ["read", "ingest"],
    "approver": ["read", "ingest", "approve"],
    "activator": ["read", "ingest", "approve", "activate"],
    "admin": ["read", "ingest", "approve", "activate", "admin", "configure"],
}


def get_user_permissions(conn: sqlite3.Connection, user_id: str) -> set[str]:
    """Return the union of all permissions for *user_id*'s assigned roles."""
    rows = conn.execute(
        "SELECT r.permissions FROM user_roles ur "
        "JOIN roles r ON r.role = ur.role "
        "WHERE ur.user_id = ?",
        (user_id,),
    ).fetchall()
    perms: set[str] = set()
    for row in rows:
        raw = row["permissions"] if isinstance(row, sqlite3.Row) else row[0]
        perms.update(json.loads(raw))
    return perms


def has_permission(conn: sqlite3.Connection, user_id: str, permission: str) -> bool:
    """Check whether *user_id* holds *permission* through any assigned role."""
    return permission in get_user_permissions(conn, user_id)


def assign_role(conn: sqlite3.Connection, user_id: str, role: str) -> None:
    """Grant *role* to *user_id*. Idempotent."""
    conn.execute(
        "INSERT OR IGNORE INTO user_roles (user_id, role) VALUES (?, ?)",
        (user_id, role),
    )
    conn.commit()


def revoke_role(conn: sqlite3.Connection, user_id: str, role: str) -> None:
    """Remove *role* from *user_id*."""
    conn.execute(
        "DELETE FROM user_roles WHERE user_id = ? AND role = ?",
        (user_id, role),
    )
    conn.commit()


def require_permission(permission: str) -> Callable:
    """Decorator that checks *permission* before calling the wrapped function.

    The wrapped function must accept ``conn`` and ``user_id`` as its first
    two positional (or keyword) arguments.

    Raises ``PermissionError`` if the user lacks the required permission.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Try to find conn and user_id from args or kwargs
            conn = kwargs.get("conn", args[0] if len(args) > 0 else None)
            user_id = kwargs.get("user_id", args[1] if len(args) > 1 else None)
            if conn is None or user_id is None:
                raise TypeError(
                    f"{fn.__name__}() requires conn and user_id arguments "
                    "for permission checks"
                )
            if not has_permission(conn, user_id, permission):
                raise PermissionError(
                    f"User '{user_id}' lacks permission '{permission}'"
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


from enum import Enum


class Role(str, Enum):
    VIEWER = "viewer"
    INTERNAL = "internal"
    INGESTOR = "ingestor"
    APPROVER = "approver"
    ACTIVATOR = "activator"
    ADMIN = "admin"
    CUSTOMER = "customer"
    SYSTEM = "system"


class Permission(str, Enum):
    READ = "read"
    INGEST = "ingest"
    APPROVE = "approve"
    ACTIVATE = "activate"
    ADMIN = "admin"
    CONFIGURE = "configure"
    KB_INGEST = "ingest"
    KB_APPROVE = "approve"
    KB_ACTIVATE = "activate"
    KB_READ = "read"


def check_role_permission(role: str | Role, permission: str | Permission) -> bool:
    """Check if a role has a given permission without database lookup."""
    r_str = role.value if isinstance(role, Role) else str(role).lower()
    p_str = permission.value if isinstance(permission, Permission) else str(permission).lower()
    if r_str == "admin":
        return True
    perms = DEFAULT_ROLES.get(r_str, [])
    return p_str in perms
