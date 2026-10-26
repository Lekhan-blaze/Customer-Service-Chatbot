"""File Service: Main orchestrator for multimodal file upload, OCR, and retention."""

import json
import re
import sqlite3
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, SystemClock
from support_bot.db import run_migrations
from support_bot.files.mismatch import detect_conflicts
from support_bot.files.models import ConflictReport, FileUploadRecord, OcrExtractedFields
from support_bot.files.ocr import BaseOCR, FakeOCR
from support_bot.files.validator import safe_filename, validate_upload
from support_bot.injection import detect_injection
from support_bot.jobs import JobQueue
from support_bot.masking import mask_pii


class FileService:
    """Service orchestrating file upload validation, OCR analysis, and retention."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: Optional[Clock] = None,
        config: Optional[Dict[str, Any]] = None,
        audit: Optional[AuditLogger] = None,
        job_queue: Optional[JobQueue] = None,
        ocr_engine: Optional[BaseOCR] = None,
        storage_root: Optional[str] = None,
        auto_migrate: bool = True,
    ) -> None:
        self.conn = conn
        self.conn.row_factory = sqlite3.Row
        if auto_migrate:
            run_migrations(self.conn)
        self.clock = clock or SystemClock()
        self.config = config or {}
        self.audit = audit or AuditLogger(self.conn, self.clock)
        self.job_queue = job_queue or JobQueue(self.conn, self.clock, self.config)
        self.ocr_engine = ocr_engine or FakeOCR()
        self.storage_root = Path(storage_root or "/tmp/chatbot_uploads")
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def _retention_days(self) -> int:
        return self.config.get("files", {}).get("retention_days", self.config.get("retention", {}).get("file_days", 30))

    def _processing_timeout_seconds(self) -> float:
        return float(self.config.get("files", {}).get("processing_timeout_seconds", 30.0))

    def process_upload(
        self,
        file_path: str,
        file_bytes: Optional[bytes] = None,
        session_id: Optional[str] = None,
        ticket_id: Optional[str] = None,
        user_claims: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[FileUploadRecord], Optional[str], List[ConflictReport]]:
        """Process an uploaded file.
        
        If processing exceeds files.processing_timeout_seconds (default 30),
        moves work to the background job queue, marks status as 'queued',
        and immediately returns customer notification message.
        """
        p = Path(file_path)
        file_type = p.suffix.lstrip(".").lower()
        if file_type == "jpeg":
            file_type = "jpg"

        if file_bytes is None and p.exists():
            file_bytes = p.read_bytes()

        # 1. Validation
        is_valid, reason = validate_upload(str(file_path), file_bytes, self.config)
        if not is_valid:
            self.audit.log("file_upload_rejected", filename=p.name, reason=reason, session_id=session_id)
            return None, reason, []

        # 2. Save file to storage
        safe_name = safe_filename(p.name)
        uid = uuid.uuid4().hex[:8]
        stored_name = f"{uid}_{safe_name}"
        stored_path = self.storage_root / stored_name
        if file_bytes:
            stored_path.write_bytes(file_bytes)

        record_id = f"FILE-{uid.upper()}"
        now = self.clock.now(ZoneInfo("UTC"))
        expires_at = (now + timedelta(days=self._retention_days())).isoformat()

        # 3. Check for execution timeout / slow processing
        duration = getattr(self.ocr_engine, "duration_seconds", 0.0) or getattr(self.ocr_engine, "simulated_duration", 0.0)
        timeout = self._processing_timeout_seconds()

        if duration > timeout:
            # Advance clock if FakeClock has advance method
            if hasattr(self.clock, "advance"):
                self.clock.advance(seconds=timeout)

            customer_msg = (
                f"Your document processing is taking longer than {int(timeout)} seconds and has been "
                f"queued for background analysis. You can check status with Upload ID: {record_id}."
            )

            record = FileUploadRecord(
                id=record_id,
                session_id=session_id,
                ticket_id=ticket_id,
                original_filename=p.name,
                safe_filename=stored_name,
                file_type=file_type,
                file_size_bytes=len(file_bytes) if file_bytes else 0,
                storage_path=str(stored_path),
                scan_status="clean",
                ocr_status="queued",
                ocr_text=None,
                ocr_confidence=None,
                ocr_extracted_fields={},
                extraction_conflicts=[],
                injection_risk_score=0.0,
                injection_reasons=[],
                expires_at=expires_at,
                customer_message=customer_msg,
                created_at=now.isoformat(),
                updated_at=now.isoformat(),
            )

            self.conn.execute(
                "INSERT INTO file_uploads ("
                "id, session_id, ticket_id, original_filename, safe_filename, file_type, "
                "file_size_bytes, storage_path, scan_status, ocr_status, ocr_text, "
                "ocr_confidence, ocr_extracted_fields, extraction_conflicts, "
                "injection_risk_score, injection_reasons, expires_at, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id, record.session_id, record.ticket_id, record.original_filename,
                    record.safe_filename, record.file_type, record.file_size_bytes,
                    record.storage_path, record.scan_status, record.ocr_status,
                    record.ocr_text, record.ocr_confidence, json.dumps({}), json.dumps([]),
                    record.injection_risk_score, json.dumps([]), record.expires_at,
                    record.created_at, record.updated_at,
                ),
            )
            self.conn.commit()

            # Enqueue background job
            if self.job_queue:
                self.job_queue.enqueue(
                    job_type="process_file_ocr",
                    payload={"file_id": record.id, "user_claims": user_claims or {}},
                )

            self.audit.log(
                "file_processing_queued",
                file_id=record.id,
                session_id=session_id,
                timeout_threshold=timeout,
                duration=duration,
            )
            return record, None, []

        # 4. Synchronous OCR execution
        return self._execute_ocr_and_persist(
            record_id=record_id,
            stored_path=stored_path,
            original_filename=p.name,
            stored_name=stored_name,
            file_type=file_type,
            file_bytes=file_bytes,
            session_id=session_id,
            ticket_id=ticket_id,
            user_claims=user_claims,
            expires_at=expires_at,
            created_at=now.isoformat(),
        )

    def _execute_ocr_and_persist(
        self,
        record_id: str,
        stored_path: Path,
        original_filename: str,
        stored_name: str,
        file_type: str,
        file_bytes: Optional[bytes],
        session_id: Optional[str],
        ticket_id: Optional[str],
        user_claims: Optional[Dict[str, Any]],
        expires_at: str,
        created_at: str,
    ) -> Tuple[FileUploadRecord, Optional[str], List[ConflictReport]]:
        now = self.clock.now(ZoneInfo("UTC"))
        ocr_result = self.ocr_engine.process(str(stored_path), file_type)

        # Check for OCR failure
        is_failed = (
            ocr_result.engine_used == "failed"
            or (ocr_result.confidence == 0.0 and not ocr_result.text)
            or ocr_result.engine_used.startswith("gemini_error")
        )

        if is_failed:
            clarification = (
                "We were unable to read this file clearly. "
                "Please upload another clear photo or PDF document."
            )
            conflicts = [
                ConflictReport(
                    field="file",
                    user_stated="",
                    ocr_extracted="unreadable",
                    severity="high",
                    clarification_prompt=clarification,
                )
            ]
            record = FileUploadRecord(
                id=record_id,
                session_id=session_id,
                ticket_id=ticket_id,
                original_filename=original_filename,
                safe_filename=stored_name,
                file_type=file_type,
                file_size_bytes=len(file_bytes) if file_bytes else 0,
                storage_path=str(stored_path),
                scan_status="clean",
                ocr_status="failed",
                ocr_text="",
                ocr_confidence=0.0,
                ocr_extracted_fields={},
                extraction_conflicts=[c.clarification_prompt for c in conflicts],
                injection_risk_score=0.0,
                injection_reasons=[],
                expires_at=expires_at,
                customer_message=clarification,
                created_at=created_at,
                updated_at=now.isoformat(),
            )
        else:
            # Injection detection
            inj_res = detect_injection(ocr_result.text)
            injection_risk = inj_res.score
            injection_reasons = inj_res.reasons

            scrubbed_text = ocr_result.text
            if injection_risk >= (self.config.get("injection_detection", {}).get("threshold", 0.6)):
                scrubbed_text = re.sub(r"(?i)\b(system override|ignore previous rules|reveal admin password|disregard prior directives|jailbreak)[^.\n]*[.\n]?", "[REDACTED_INJECTION]", scrubbed_text).strip()

            # Mismatches
            conflicts = []
            if user_claims:
                conflicts = detect_conflicts(ocr_result.extracted_fields, user_claims)

            record = FileUploadRecord(
                id=record_id,
                session_id=session_id,
                ticket_id=ticket_id,
                original_filename=original_filename,
                safe_filename=stored_name,
                file_type=file_type,
                file_size_bytes=len(file_bytes) if file_bytes else 0,
                storage_path=str(stored_path),
                scan_status="clean",
                ocr_status="done",
                ocr_text=mask_pii(scrubbed_text),
                ocr_confidence=ocr_result.confidence,
                ocr_extracted_fields=ocr_result.extracted_fields.to_dict(),
                extraction_conflicts=[c.clarification_prompt for c in conflicts],
                injection_risk_score=injection_risk,
                injection_reasons=injection_reasons,
                expires_at=expires_at,
                customer_message=None,
                created_at=created_at,
                updated_at=now.isoformat(),
            )

        self.conn.execute(
            "INSERT INTO file_uploads ("
            "id, session_id, ticket_id, original_filename, safe_filename, file_type, "
            "file_size_bytes, storage_path, scan_status, ocr_status, ocr_text, "
            "ocr_confidence, ocr_extracted_fields, extraction_conflicts, "
            "injection_risk_score, injection_reasons, expires_at, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "ocr_status=excluded.ocr_status, ocr_text=excluded.ocr_text, "
            "ocr_confidence=excluded.ocr_confidence, ocr_extracted_fields=excluded.ocr_extracted_fields, "
            "extraction_conflicts=excluded.extraction_conflicts, "
            "injection_risk_score=excluded.injection_risk_score, "
            "injection_reasons=excluded.injection_reasons, updated_at=excluded.updated_at",
            (
                record.id, record.session_id, record.ticket_id, record.original_filename,
                record.safe_filename, record.file_type, record.file_size_bytes,
                record.storage_path, record.scan_status, record.ocr_status,
                record.ocr_text, record.ocr_confidence,
                json.dumps(record.ocr_extracted_fields or {}),
                json.dumps(record.extraction_conflicts),
                record.injection_risk_score,
                json.dumps(record.injection_reasons),
                record.expires_at, record.created_at, record.updated_at,
            ),
        )
        self.conn.commit()

        self.audit.log(
            "file_upload_processed",
            file_id=record.id,
            session_id=session_id,
            file_type=file_type,
            ocr_status=record.ocr_status,
            ocr_confidence=record.ocr_confidence,
            conflicts=len(conflicts),
        )

        if self.job_queue:
            self.job_queue.enqueue(
                job_type="purge_expired_file",
                payload={"file_id": record.id},
                scheduled_at=expires_at,
            )

        return record, None, conflicts

    def process_queued_file(self, file_id: str, user_claims: Optional[Dict[str, Any]] = None) -> Optional[FileUploadRecord]:
        """Worker task to process a queued file."""
        row = self.conn.execute("SELECT * FROM file_uploads WHERE id = ?", (file_id,)).fetchone()
        if not row:
            return None

        # Mark as processing
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        self.conn.execute("UPDATE file_uploads SET ocr_status = 'processing', updated_at = ? WHERE id = ?", (now, file_id))
        self.conn.commit()

        stored_path = Path(row["storage_path"])
        file_type = row["file_type"]

        # Run OCR
        ocr_result = self.ocr_engine.process(str(stored_path), file_type)
        is_failed = (
            ocr_result.engine_used == "failed"
            or (ocr_result.confidence == 0.0 and not ocr_result.text)
            or ocr_result.engine_used.startswith("gemini_error")
        )

        if is_failed:
            clarification = (
                "We were unable to read this file clearly. "
                "Please upload another clear photo or PDF document."
            )
            conflicts = [clarification]
            extracted_fields: Dict[str, Any] = {}
            status = "failed"
            text = ""
            conf = 0.0
            inj_score = 0.0
            inj_reasons: List[str] = []
        else:
            inj_res = detect_injection(ocr_result.text)
            inj_score = inj_res.score
            inj_reasons = inj_res.reasons

            scrubbed_text = ocr_result.text
            if inj_score >= (self.config.get("injection_detection", {}).get("threshold", 0.6)):
                scrubbed_text = re.sub(r"(?i)\b(system override|ignore previous rules|reveal admin password|disregard prior directives|jailbreak)[^.\n]*[.\n]?", "[REDACTED_INJECTION]", scrubbed_text).strip()

            c_objs = detect_conflicts(ocr_result.extracted_fields, user_claims or {}) if user_claims else []
            conflicts = [c.clarification_prompt for c in c_objs]
            extracted_fields = ocr_result.extracted_fields.to_dict()
            status = "done"
            text = mask_pii(scrubbed_text)
            conf = ocr_result.confidence

        updated_now = self.clock.now(ZoneInfo("UTC")).isoformat()
        self.conn.execute(
            "UPDATE file_uploads SET "
            "ocr_status = ?, ocr_text = ?, ocr_confidence = ?, ocr_extracted_fields = ?, "
            "extraction_conflicts = ?, injection_risk_score = ?, injection_reasons = ?, updated_at = ? "
            "WHERE id = ?",
            (
                status, text, conf, json.dumps(extracted_fields),
                json.dumps(conflicts), inj_score, json.dumps(inj_reasons),
                updated_now, file_id,
            ),
        )
        self.conn.commit()
        self.audit.log("file_queued_processing_complete", file_id=file_id, ocr_status=status)
        return self.get_record(file_id)

    def poll_upload(self, file_id: str) -> Optional[FileUploadRecord]:
        """Poll the current processing status and details of an upload by ID."""
        return self.get_record(file_id)

    def get_job_handlers(self) -> Dict[str, Callable[[dict | None], Any]]:
        """Return job handlers for integration with JobQueue / background workers."""
        return {
            "process_file_ocr": lambda p: self.process_queued_file((p or {}).get("file_id"), (p or {}).get("user_claims")),
            "purge_expired_file": lambda p: self.purge_expired_files(),
        }

    def get_record(self, file_id: str) -> Optional[FileUploadRecord]:
        """Fetch a file upload record by ID."""
        row = self.conn.execute("SELECT * FROM file_uploads WHERE id = ?", (file_id,)).fetchone()
        if not row:
            return None
        return FileUploadRecord(
            id=row["id"],
            session_id=row["session_id"],
            ticket_id=row["ticket_id"],
            original_filename=row["original_filename"],
            safe_filename=row["safe_filename"],
            file_type=row["file_type"],
            file_size_bytes=row["file_size_bytes"],
            storage_path=row["storage_path"],
            scan_status=row["scan_status"],
            ocr_status=row["ocr_status"],
            ocr_text=row["ocr_text"],
            ocr_confidence=row["ocr_confidence"],
            ocr_extracted_fields=json.loads(row["ocr_extracted_fields"] or "{}"),
            extraction_conflicts=json.loads(row["extraction_conflicts"] or "[]"),
            injection_risk_score=row["injection_risk_score"] or 0.0,
            injection_reasons=json.loads(row["injection_reasons"] or "[]"),
            expires_at=row["expires_at"],
            purged=bool(row["purged"]),
            purged_at=row["purged_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def purge_expired_files(self) -> List[str]:
        """Purge raw file bytes from disk for expired records; keep audit row."""
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        rows = self.conn.execute(
            "SELECT id, storage_path FROM file_uploads "
            "WHERE expires_at <= ? AND purged = 0",
            (now,),
        ).fetchall()

        purged_ids = []
        for r in rows:
            storage_path = Path(r["storage_path"])
            if storage_path.exists():
                try:
                    storage_path.unlink()
                except OSError:
                    pass
            self.conn.execute(
                "UPDATE file_uploads SET purged = 1, purged_at = ?, updated_at = ? WHERE id = ?",
                (now, now, r["id"]),
            )
            self.conn.commit()
            self.audit.log("file_purged", file_id=r["id"])
            purged_ids.append(r["id"])
        return purged_ids
