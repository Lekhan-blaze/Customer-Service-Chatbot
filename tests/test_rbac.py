"""Tests for support_bot.rbac — roles, permissions, and require_permission decorator."""

import pytest
from support_bot.rbac import (
    assign_role,
    get_user_permissions,
    has_permission,
    require_permission,
    revoke_role,
)


def test_seed_roles_and_assignment(db_conn):
    # Assign viewer role
    assign_role(db_conn, "alice", "viewer")
    assert has_permission(db_conn, "alice", "read") is True
    assert has_permission(db_conn, "alice", "ingest") is False

    # Assign ingestor role
    assign_role(db_conn, "bob", "ingestor")
    assert has_permission(db_conn, "bob", "read") is True
    assert has_permission(db_conn, "bob", "ingest") is True
    assert has_permission(db_conn, "bob", "approve") is False

    # Assign approver role
    assign_role(db_conn, "carol", "approver")
    assert has_permission(db_conn, "carol", "approve") is True
    assert has_permission(db_conn, "carol", "activate") is False

    # Assign activator role
    assign_role(db_conn, "dave", "activator")
    assert has_permission(db_conn, "dave", "activate") is True
    assert has_permission(db_conn, "dave", "admin") is False

    # Assign admin role
    assign_role(db_conn, "eve", "admin")
    perms = get_user_permissions(db_conn, "eve")
    for expected in ["read", "ingest", "approve", "activate", "admin", "configure"]:
        assert expected in perms


def test_revoke_role(db_conn):
    assign_role(db_conn, "user1", "admin")
    assert has_permission(db_conn, "user1", "admin") is True

    revoke_role(db_conn, "user1", "admin")
    assert has_permission(db_conn, "user1", "admin") is False


def test_require_permission_decorator(db_conn):
    @require_permission("approve")
    def approve_document(conn, user_id, doc_id):
        return f"Document {doc_id} approved"

    assign_role(db_conn, "reviewer", "approver")
    assign_role(db_conn, "reader", "viewer")

    # Success case
    res = approve_document(db_conn, "reviewer", 42)
    assert res == "Document 42 approved"

    # Unauthorized case
    with pytest.raises(PermissionError) as exc_info:
        approve_document(db_conn, "reader", 42)
    assert "User 'reader' lacks permission 'approve'" in str(exc_info.value)

    # Keyword argument call
    res_kw = approve_document(conn=db_conn, user_id="reviewer", doc_id=42)
    assert res_kw == "Document 42 approved"
