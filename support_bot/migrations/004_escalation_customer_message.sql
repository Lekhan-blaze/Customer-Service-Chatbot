-- Migration 004: add customer_message column to escalation_records
-- Stores the safe customer-facing escalation message (no outcome promises beyond config)

ALTER TABLE escalation_records ADD COLUMN customer_message TEXT;

INSERT INTO schema_version (version) VALUES (4);
