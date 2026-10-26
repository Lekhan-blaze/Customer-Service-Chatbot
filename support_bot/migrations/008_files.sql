-- Migration 008: Multimodal File Uploads & OCR Evidence

CREATE TABLE IF NOT EXISTS file_uploads (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    ticket_id TEXT,
    original_filename TEXT NOT NULL,
    safe_filename TEXT NOT NULL,
    file_type TEXT NOT NULL,             -- png, jpg, pdf
    file_size_bytes INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    scan_status TEXT DEFAULT 'pending',   -- pending, clean, quarantined
    ocr_status TEXT DEFAULT 'pending',    -- pending, processing, done, failed
    ocr_text TEXT,
    ocr_confidence REAL,
    ocr_extracted_fields TEXT,           -- JSON: {order_id, amount, date, card_last4}
    extraction_conflicts TEXT,           -- JSON: list of conflict descriptions
    injection_risk_score REAL DEFAULT 0.0,
    injection_reasons TEXT DEFAULT '[]',
    expires_at TEXT,                     -- retention expiry timestamp
    purged INTEGER DEFAULT 0,            -- raw file purged but audit record kept
    purged_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
);

CREATE INDEX IF NOT EXISTS idx_file_uploads_session ON file_uploads(session_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_ticket ON file_uploads(ticket_id);
CREATE INDEX IF NOT EXISTS idx_file_uploads_expires ON file_uploads(expires_at);


INSERT INTO schema_version (version) VALUES (8);
