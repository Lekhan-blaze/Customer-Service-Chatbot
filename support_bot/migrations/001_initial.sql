-- Migration 001: initial schema for support_bot foundation
-- Tables: schema_version, config_overrides, audit_log, jobs, roles, user_roles

CREATE TABLE IF NOT EXISTS schema_version (
    version   INTEGER PRIMARY KEY,
    applied   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS config_overrides (
    key       TEXT    PRIMARY KEY,
    value     TEXT    NOT NULL,
    updated   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    event      TEXT    NOT NULL,
    decision   TEXT,
    rule       TEXT,
    references TEXT,          -- JSON string
    meta       TEXT,          -- JSON string for extra fields
    created    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type      TEXT    NOT NULL,
    payload       TEXT,               -- JSON string
    status        TEXT    NOT NULL DEFAULT 'pending',  -- pending | running | done | failed | deferred
    scheduled_at  TEXT    NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 4,   -- 1 initial + 3 retries
    last_error    TEXT,
    created       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS roles (
    role        TEXT PRIMARY KEY,
    permissions TEXT NOT NULL        -- JSON list of permission strings
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id   TEXT NOT NULL,
    role      TEXT NOT NULL REFERENCES roles(role),
    granted   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, role)
);

-- Seed default roles
INSERT OR IGNORE INTO roles (role, permissions) VALUES
    ('viewer',    '["read"]'),
    ('ingestor',  '["read", "ingest"]'),
    ('approver',  '["read", "ingest", "approve"]'),
    ('activator', '["read", "ingest", "approve", "activate"]'),
    ('admin',     '["read", "ingest", "approve", "activate", "admin", "configure"]');

INSERT INTO schema_version (version) VALUES (1);
