"""Tests for support_bot.config — YAML defaults merged with SQLite overrides."""

from support_bot.config import get_config, get_override, load_defaults, set_override


def test_load_defaults():
    cfg = load_defaults()
    assert cfg["company_timezone"] == "Asia/Kolkata"
    assert cfg["business_hours"]["start"] == "09:00"
    assert cfg["business_hours"]["end"] == "18:00"
    assert "Hindi" in cfg["supported_languages"]
    assert "Tamil" in cfg["supported_languages"]
    assert "Kannada" in cfg["supported_languages"]
    assert cfg["maintenance_window"]["start"] == "02:00"
    assert cfg["maintenance_window"]["end"] == "04:00"
    assert cfg["session"]["inactivity_timeout_minutes"] == 30
    assert cfg["session"]["restore_window_hours"] == 24
    assert cfg["retention"]["audit_days"] == 90
    assert cfg["retention"]["session_days"] == 30
    assert cfg["sla"]["first_response_minutes"] == 30
    assert cfg["sla"]["resolution_hours"] == 24
    assert cfg["job_retry_backoff_minutes"] == [15, 30, 60]


def test_runtime_override(db_conn):
    # Initial state: no override
    assert get_override(db_conn, "sla.first_response_minutes") is None

    # Set override
    set_override(db_conn, "company_timezone", "America/New_York")
    assert get_override(db_conn, "company_timezone") == "America/New_York"

    # Set structured override (dict/list)
    set_override(db_conn, "job_retry_backoff_minutes", [5, 10, 20])
    assert get_override(db_conn, "job_retry_backoff_minutes") == [5, 10, 20]

    # Merged config
    cfg = get_config(db_conn)
    assert cfg["company_timezone"] == "America/New_York"
    assert cfg["job_retry_backoff_minutes"] == [5, 10, 20]
    # Unchanged keys retain defaults
    assert cfg["business_hours"]["start"] == "09:00"


def test_get_config_without_db():
    cfg = get_config(None)
    assert cfg["company_timezone"] == "Asia/Kolkata"
