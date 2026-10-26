-- Migration 007: Structured Support Tickets & SLA Management

CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    customer_name TEXT NOT NULL,
    contact_info TEXT,
    order_or_enrollment_id TEXT NOT NULL,
    product_or_course TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    description TEXT NOT NULL,
    summary TEXT,
    evidence_refs TEXT, -- JSON array of message/file references
    priority TEXT NOT NULL, -- P1, P2, P3, P4
    assigned_team TEXT NOT NULL,
    status TEXT NOT NULL, -- open, in_progress, pending_customer, resolved, closed, escalated
    sla_deadline TEXT NOT NULL,
    sla_warning_at TEXT NOT NULL,
    sla_breached INTEGER DEFAULT 0,
    sla_warning_sent INTEGER DEFAULT 0,
    duplicate_of_ticket_id TEXT,
    related_ticket_ids TEXT DEFAULT '[]', -- JSON array
    metadata TEXT, -- JSON string
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (duplicate_of_ticket_id) REFERENCES tickets(id)
);

CREATE TABLE IF NOT EXISTS ticket_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    action TEXT NOT NULL, -- extraction, priority_calculated, routed, duplicate_detected, sla_warning, sla_breached, escalated, updated
    actor TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(id)
);

CREATE INDEX IF NOT EXISTS idx_tickets_session ON tickets(session_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_dup ON tickets(duplicate_of_ticket_id);
CREATE INDEX IF NOT EXISTS idx_tickets_order ON tickets(order_or_enrollment_id);
CREATE INDEX IF NOT EXISTS idx_ticket_audit_tid ON ticket_audit(ticket_id);
