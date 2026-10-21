-- Migration 002: conversation sessions and message history
-- Tables: conversation_sessions, conversation_messages

CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id   TEXT PRIMARY KEY,
    customer_id  TEXT,
    language     TEXT NOT NULL DEFAULT 'English',
    summary      TEXT,
    facts        TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'active',  -- active | expired | ended
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    ended_at     TEXT
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES conversation_sessions(session_id),
    role         TEXT NOT NULL,  -- user | assistant | system
    content      TEXT NOT NULL,
    language     TEXT,
    intents      TEXT,           -- JSON array of intent names
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_customer ON conversation_sessions(customer_id);
CREATE INDEX IF NOT EXISTS idx_messages_session ON conversation_messages(session_id);

INSERT INTO schema_version (version) VALUES (2);
