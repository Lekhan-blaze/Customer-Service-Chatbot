-- Migration 005: add message_snippets column to escalation_records
-- message_refs stores JSON list of message IDs
-- message_snippets stores JSON list of masked text snippets

ALTER TABLE escalation_records ADD COLUMN message_snippets TEXT;

INSERT INTO schema_version (version) VALUES (5);
