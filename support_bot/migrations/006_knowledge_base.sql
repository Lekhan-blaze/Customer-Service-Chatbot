-- Migration 006: Knowledge Base Document, Versioning, Quarantine, and Audit Tables

CREATE TABLE IF NOT EXISTS kb_documents (
    id TEXT PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    title TEXT NOT NULL,
    product TEXT DEFAULT 'general',
    region TEXT DEFAULT 'global',
    access_level TEXT DEFAULT 'customer',
    effective_date TEXT NOT NULL,
    expiry_date TEXT,
    source_location TEXT,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'discovered',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kb_docs_hash ON kb_documents (content_hash);
CREATE INDEX IF NOT EXISTS idx_kb_docs_status ON kb_documents (status);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks (document_id);

CREATE TABLE IF NOT EXISTS kb_versions (
    id TEXT PRIMARY KEY,
    index_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'staged',
    quality_score REAL DEFAULT 0.0,
    document_count INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    is_active INTEGER DEFAULT 0,
    activated_at TEXT,
    activated_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_quarantine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    file_path TEXT NOT NULL,
    reason TEXT NOT NULL,
    risk_score REAL DEFAULT 0.0,
    quarantined_at TEXT NOT NULL,
    actor TEXT DEFAULT 'system'
);

CREATE TABLE IF NOT EXISTS kb_audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT,
    version_id TEXT,
    from_state TEXT,
    to_state TEXT NOT NULL,
    actor TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS kb_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    tags_json TEXT,
    timestamp TEXT NOT NULL
);

INSERT INTO schema_version (version) VALUES (6);
