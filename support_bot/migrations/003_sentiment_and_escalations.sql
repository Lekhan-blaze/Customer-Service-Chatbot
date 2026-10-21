-- Migration 003: sentiment analysis and escalation records
-- Tables: message_sentiment, escalation_records

CREATE TABLE IF NOT EXISTS message_sentiment (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id         TEXT NOT NULL,
    message_id         INTEGER,
    message_text       TEXT NOT NULL,
    labels             TEXT NOT NULL,  -- JSON dict: {"positive": 0.1, "neutral": 0.2, "negative": 0.7, "frustrated": 0.8, "urgent": 0.4, "sarcastic": 0.1}
    calibrated_scores  TEXT NOT NULL,  -- JSON dict of calibrated scores
    dominant_label     TEXT NOT NULL,
    risk_category      TEXT,           -- account_compromise | duplicate_payment | legal_threat | null
    risk_score         REAL NOT NULL DEFAULT 0.0,
    created_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS escalation_records (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id            TEXT NOT NULL,
    reason                TEXT NOT NULL,
    rule_name             TEXT NOT NULL,
    calibrated_scores     TEXT,       -- JSON dict
    conversation_summary  TEXT,
    message_refs          TEXT,       -- JSON list of message IDs / snippets
    routing_outcome       TEXT NOT NULL, -- human_queue | on_call | scheduled_next_working_day
    status                TEXT NOT NULL DEFAULT 'pending', -- pending | in_progress | resolved
    created_at            TEXT NOT NULL,
    resolved_at           TEXT
);

CREATE INDEX IF NOT EXISTS idx_sentiment_session ON message_sentiment(session_id);
CREATE INDEX IF NOT EXISTS idx_escalation_session ON escalation_records(session_id);
CREATE INDEX IF NOT EXISTS idx_escalation_status ON escalation_records(status);

INSERT INTO schema_version (version) VALUES (3);
