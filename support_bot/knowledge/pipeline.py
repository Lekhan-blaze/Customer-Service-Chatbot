"""Knowledge Base Ingestion, Versioning, Quality Gate, Activation, and Rollback Pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from support_bot.audit import AuditLogger
from support_bot.clock import Clock, is_maintenance_window
from support_bot.injection import PromptInjectionDetector
from support_bot.jobs import JobQueue
from support_bot.knowledge.embedder import BaseEmbedder, FakeEmbedder, get_default_embedder
from support_bot.knowledge.models import (
    DocumentChunk,
    DocumentMetadata,
    DocumentState,
    KnowledgeVersion,
    QualityGateResult,
    QuarantineRecord,
    validate_transition,
)
from support_bot.knowledge.quality_gate import QualityGate
from support_bot.knowledge.vector_store import VersionedVectorStore
from support_bot.masking import mask_pii
from support_bot.rbac import Permission, Role, check_role_permission as has_permission


class KnowledgePipeline:
    """Manages the full lifecycle of knowledge base documents and versioned vector indexes."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        clock: Clock,
        config: Dict[str, Any],
        audit: Optional[AuditLogger] = None,
        job_queue: Optional[JobQueue] = None,
        injection_detector: Optional[PromptInjectionDetector] = None,
        embedder: Optional[BaseEmbedder] = None,
        versions_root: Optional[Path | str] = None,
        golden_qa_path: Optional[Path | str] = None,
    ) -> None:
        self.conn = conn
        self.clock = clock
        self.config = config
        self.audit = audit or AuditLogger(conn=conn, clock=clock)
        self.job_queue = job_queue or JobQueue(conn=conn, clock=clock, cfg=config)
        self.injection_detector = injection_detector or PromptInjectionDetector(config=config)
        self.embedder = embedder or get_default_embedder(prefer_fake=True)
        self.versions_root = Path(versions_root or "faiss_index/versions")
        self.golden_qa_path = Path(
            golden_qa_path or "tests/data/golden_qa.json"
        )
        self.quality_gate = QualityGate(
            golden_qa_path=self.golden_qa_path,
            min_hit_rate=self.config.get("knowledge", {}).get("min_hit_rate", 0.65),
            min_grounding_score=self.config.get("knowledge", {}).get("min_grounding_score", 0.60),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Document Ingestion & Validation
    # ──────────────────────────────────────────────────────────────────────────

    def ingest_document_file(
        self,
        file_path: Path | str,
        actor: str = "system",
        user_role: str | Role = Role.INGESTOR,
    ) -> Optional[DocumentMetadata]:
        """Ingest, validate, safety-check, and chunk a single policy/FAQ file."""
        if not has_permission(user_role, Permission.KB_INGEST):
            raise PermissionError(f"Role '{user_role}' lacks permission '{Permission.KB_INGEST.value}'")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        # 1. File extension validation
        allowlist = self.config.get("knowledge", {}).get(
            "allowed_extensions", [".txt", ".csv", ".md", ".json", ".pdf"]
        )
        if path.suffix.lower() not in allowlist:
            reason = f"File type '{path.suffix}' not in allowlist {allowlist}"
            self._quarantine_file(path, reason=reason, risk_score=1.0, actor=actor)
            return None

        # 2. File size validation (limit 5MB)
        max_size = self.config.get("knowledge", {}).get("max_file_size_bytes", 5 * 1024 * 1024)
        if path.stat().st_size > max_size:
            reason = f"File size {path.stat().st_size} exceeds maximum {max_size} bytes"
            self._quarantine_file(path, reason=reason, risk_score=0.9, actor=actor)
            return None

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            self._quarantine_file(path, reason=f"Unreadable file encoding: {e}", risk_score=0.8, actor=actor)
            return None

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 3. Duplicate check: skip if content hash already exists
        dup = self.conn.execute(
            "SELECT id, status FROM kb_documents WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if dup:
            self.audit.log(
                "kb_duplicate_skipped",
                document_id=dup["id"],
                content_hash=content_hash,
                path=str(path),
                actor=actor,
            )
            return None

        # 4. Load metadata sidecar (.meta.json) if present
        meta_file = path.parent / f"{path.stem}.meta.json"
        metadata: Dict[str, Any] = {}
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass

        doc_id = metadata.get("id", f"doc_{path.stem}")
        version = int(metadata.get("version", 1))
        title = metadata.get("title", path.stem.replace("_", " ").title())
        product = metadata.get("product", "general")
        region = metadata.get("region", "global")
        access_level = metadata.get("access_level", "customer").lower()
        effective_date = metadata.get("effective_date", "2026-01-01")
        expiry_date = metadata.get("expiry_date")

        # 5. Prompt injection defense check on untrusted document content
        inj_res = self.injection_detector.detect(content, source_type="document")
        if inj_res.is_injection:
            reason = f"Prompt injection detected in document: {', '.join(inj_res.matched_reasons)}"
            self._quarantine_file(path, reason=reason, risk_score=inj_res.risk_score, document_id=doc_id, actor=actor)
            return None

        now = self.clock.now(ZoneInfo("UTC")).isoformat()

        # 6. Save validated document in SQLite
        self.conn.execute(
            "INSERT INTO kb_documents (id, version, title, product, region, access_level, "
            "effective_date, expiry_date, source_location, content_hash, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET version=excluded.version, title=excluded.title, "
            "content_hash=excluded.content_hash, status=excluded.status, updated_at=excluded.updated_at",
            (
                doc_id,
                version,
                title,
                product,
                region,
                access_level,
                effective_date,
                expiry_date,
                str(path),
                content_hash,
                DocumentState.VALIDATED.value,
                now,
                now,
            ),
        )

        # 7. Chunk document into semantic passages
        chunks = self._chunk_text(content, doc_id, {
            "document_id": doc_id,
            "version": version,
            "title": title,
            "product": product,
            "region": region,
            "access_level": access_level,
            "effective_date": effective_date,
            "expiry_date": expiry_date,
            "source_location": str(path),
        })

        self.conn.execute("DELETE FROM kb_chunks WHERE document_id = ?", (doc_id,))
        for c in chunks:
            self.conn.execute(
                "INSERT INTO kb_chunks (document_id, chunk_index, chunk_text, metadata_json) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, c.chunk_index, c.chunk_text, json.dumps(c.metadata)),
            )

        self.conn.commit()

        self.audit.log(
            "kb_document_ingested",
            document_id=doc_id,
            version=version,
            chunks=len(chunks),
            actor=actor,
        )

        return DocumentMetadata(
            id=doc_id,
            version=version,
            title=title,
            product=product,
            region=region,
            access_level=access_level,
            effective_date=effective_date,
            expiry_date=expiry_date,
            source_location=str(path),
            content_hash=content_hash,
            status=DocumentState.VALIDATED.value,
            created_at=now,
            updated_at=now,
        )

    def ingest_inbox(
        self,
        inbox_dir: Path | str,
        actor: str = "system",
        user_role: str | Role = Role.INGESTOR,
    ) -> List[DocumentMetadata]:
        """Ingest all eligible policy and document files in a directory."""
        folder = Path(inbox_dir)
        if not folder.exists():
            return []

        results = []
        for file in sorted(folder.iterdir()):
            if file.is_file() and not file.name.endswith(".meta.json") and not file.name.startswith("."):
                doc = self.ingest_document_file(file, actor=actor, user_role=user_role)
                if doc:
                    results.append(doc)
        return results

    def _quarantine_file(
        self,
        file_path: Path,
        reason: str,
        risk_score: float,
        document_id: Optional[str] = None,
        actor: str = "system",
    ) -> None:
        now = self.clock.now(ZoneInfo("UTC")).isoformat()
        quarantine_dir = file_path.parent / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / file_path.name
        try:
            shutil.copy2(file_path, target)
        except Exception:
            pass

        self.conn.execute(
            "INSERT INTO kb_quarantine (document_id, file_path, reason, risk_score, quarantined_at, actor) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, str(file_path), mask_pii(reason), risk_score, now, actor),
        )
        self.conn.commit()

        self.audit.log(
            "kb_document_quarantined",
            document_id=document_id,
            file_path=str(file_path),
            reason=mask_pii(reason),
            risk_score=risk_score,
            actor=actor,
        )

    def _chunk_text(
        self,
        content: str,
        doc_id: str,
        metadata: Dict[str, Any],
        max_chunk_chars: int = 500,
    ) -> List[DocumentChunk]:
        """Split text into structured chunks keeping paragraph boundaries."""
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        chunks: List[DocumentChunk] = []
        idx = 0

        for para in paragraphs:
            if len(para) <= max_chunk_chars:
                chunks.append(DocumentChunk(
                    id=None,
                    document_id=doc_id,
                    chunk_index=idx,
                    chunk_text=para,
                    metadata=dict(metadata),
                ))
                idx += 1
            else:
                # Sub-split long paragraph by sentences or periods
                lines = [l.strip() for l in para.split(". ") if l.strip()]
                buffer = ""
                for l in lines:
                    if len(buffer) + len(l) <= max_chunk_chars:
                        buffer = f"{buffer} {l}.".strip()
                    else:
                        if buffer:
                            chunks.append(DocumentChunk(
                                id=None,
                                document_id=doc_id,
                                chunk_index=idx,
                                chunk_text=buffer,
                                metadata=dict(metadata),
                            ))
                            idx += 1
                        buffer = f"{l}."
                if buffer:
                    chunks.append(DocumentChunk(
                        id=None,
                        document_id=doc_id,
                        chunk_index=idx,
                        chunk_text=buffer,
                        metadata=dict(metadata),
                    ))
                    idx += 1
        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Version Building & Quality Gate
    # ──────────────────────────────────────────────────────────────────────────

    def build_version(
        self,
        version_id: str,
        actor: str = "system",
        user_role: str | Role = Role.INGESTOR,
    ) -> Tuple[KnowledgeVersion, QualityGateResult]:
        """Build a new index version from all validated chunks, test with quality gate."""
        if not has_permission(user_role, Permission.KB_INGEST):
            raise PermissionError(f"Role '{user_role}' lacks permission '{Permission.KB_INGEST.value}'")

        # Load all validated chunks
        rows = self.conn.execute(
            "SELECT c.id, c.document_id, c.chunk_index, c.chunk_text, c.metadata_json "
            "FROM kb_chunks c JOIN kb_documents d ON c.document_id = d.id "
            "WHERE d.status = 'validated' "
            "ORDER BY c.document_id, c.chunk_index"
        ).fetchall()

        if not rows:
            raise ValueError("No validated documents available to build an index version.")

        chunks = [
            DocumentChunk(
                id=r["id"],
                document_id=r["document_id"],
                chunk_index=r["chunk_index"],
                chunk_text=r["chunk_text"],
                metadata=json.loads(r["metadata_json"]) if r["metadata_json"] else {},
            )
            for r in rows
        ]

        store = VersionedVectorStore(
            version_id=version_id,
            storage_dir=self.versions_root,
            embedder=self.embedder,
        )
        store.add_chunks(chunks)
        store.save()

        # Run Quality Gate against golden QA benchmark
        active_ver = self.get_active_version()
        active_score = active_ver.quality_score if active_ver else None

        gate_res = self.quality_gate.evaluate(store, active_version_score=active_score)
        now = self.clock.now(ZoneInfo("UTC")).isoformat()

        status = "tested" if gate_res.passed else "failed"
        if not gate_res.passed and self.job_queue:
            self.schedule_update_retry(version_id, attempt=1)

        self.conn.execute(
            "INSERT INTO kb_versions (id, index_path, status, quality_score, document_count, "
            "chunk_count, is_active, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, quality_score=excluded.quality_score, "
            "chunk_count=excluded.chunk_count",
            (
                version_id,
                str(store.storage_dir),
                status,
                gate_res.hit_rate,
                len(set(c.document_id for c in chunks)),
                len(chunks),
                now,
            ),
        )
        self.conn.commit()

        self.audit.log(
            "kb_version_built",
            version_id=version_id,
            status=status,
            quality_hit_rate=gate_res.hit_rate,
            gate_passed=gate_res.passed,
            actor=actor,
        )

        ver = KnowledgeVersion(
            id=version_id,
            index_path=str(store.storage_dir),
            status=status,
            quality_score=gate_res.hit_rate,
            document_count=len(set(c.document_id for c in chunks)),
            chunk_count=len(chunks),
            is_active=False,
            created_at=now,
        )

        return ver, gate_res


    def schedule_update_retry(self, version_id: str, attempt: int = 1) -> Optional[int]:
        """Schedule retry for failed knowledge updates at 15, 30, 60 minutes."""
        if attempt > 3 or not self.job_queue:
            return None
        retry_delays = {1: 15, 2: 30, 3: 60}
        delay_mins = retry_delays.get(attempt, 60)
        from datetime import timedelta
        run_at = self.clock.now(ZoneInfo("UTC")) + timedelta(minutes=delay_mins)
        job_id = self.job_queue.enqueue(
            job_type="retry_kb_update",
            payload={"version_id": version_id, "attempt": attempt},
            delay_minutes=delay_mins,
        )
        self.audit.log(
            "kb_update_retry_scheduled",
            version_id=version_id,
            attempt=attempt,
            delay_minutes=delay_mins,
            job_id=job_id,
        )
        return job_id

    # ──────────────────────────────────────────────────────────────────────────
    # Version Approval & Activation
    # ──────────────────────────────────────────────────────────────────────────

    def approve_version(
        self,
        version_id: str,
        actor: str = "approver",
        user_role: str | Role = Role.APPROVER,
    ) -> KnowledgeVersion:
        """Approve a tested knowledge base version for scheduling/activation."""
        if not has_permission(user_role, Permission.KB_APPROVE):
            raise PermissionError(f"Role '{user_role}' lacks permission '{Permission.KB_APPROVE.value}'")

        row = self.conn.execute("SELECT * FROM kb_versions WHERE id = ?", (version_id,)).fetchone()
        if not row:
            raise ValueError(f"Knowledge version '{version_id}' not found.")

        validate_transition(row["status"], DocumentState.APPROVED.value)

        self.conn.execute(
            "UPDATE kb_versions SET status = 'approved' WHERE id = ?", (version_id,)
        )
        self.conn.commit()

        self.audit.log(
            "kb_version_approved",
            version_id=version_id,
            actor=actor,
        )

        return self.get_version(version_id)

    def activate_version(
        self,
        version_id: str,
        actor: str = "activator",
        user_role: str | Role = Role.ACTIVATOR,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Activate an approved knowledge base version.

        Enforces maintenance window restriction (02:00-04:00) unless force=True.
        Runs post-activation health checks and automatically rolls back if health checks fail.
        """
        if not has_permission(user_role, Permission.KB_ACTIVATE):
            raise PermissionError(f"Role '{user_role}' lacks permission '{Permission.KB_ACTIVATE.value}'")

        row = self.conn.execute("SELECT * FROM kb_versions WHERE id = ?", (version_id,)).fetchone()
        if not row:
            raise ValueError(f"Knowledge version '{version_id}' not found.")

        # Check maintenance window
        in_window = is_maintenance_window(self.clock, self.config)
        if not in_window and not force:
            # Schedule activation job for next maintenance window
            self.job_queue.enqueue_after_hours(
                "activate_kb_version",
                payload={"version_id": version_id, "actor": actor},
            )
            self.conn.execute(
                "UPDATE kb_versions SET status = 'scheduled' WHERE id = ?", (version_id,)
            )
            self.conn.commit()
            self.audit.log(
                "kb_activation_scheduled",
                version_id=version_id,
                reason="Outside maintenance window (02:00-04:00); scheduled for next window",
                actor=actor,
            )
            return {
                "status": "scheduled",
                "message": "Outside maintenance window (02:00-04:00); scheduled via job queue.",
                "version_id": version_id,
            }

        # Remember previous active version for auto-rollback protection
        prev_active = self.get_active_version()
        now = self.clock.now(ZoneInfo("UTC")).isoformat()

        # Point active pointer to new version
        self.conn.execute("UPDATE kb_versions SET is_active = 0 WHERE is_active = 1")
        self.conn.execute(
            "UPDATE kb_versions SET is_active = 1, status = 'active', "
            "activated_at = ?, activated_by = ? WHERE id = ?",
            (now, actor, version_id),
        )
        self.conn.commit()

        # Post-activation health check
        health_ok = self._run_health_check(version_id)
        if not health_ok:
            # Automatic rollback within 5 minutes
            self.audit.log(
                "kb_health_check_failed",
                version_id=version_id,
                action="auto_rollback",
                actor="system",
            )
            if prev_active:
                self.rollback_to(prev_active.id, actor="system_auto_rollback", user_role=Role.ADMIN)
            else:
                self.conn.execute(
                    "UPDATE kb_versions SET is_active = 0, status = 'failed' WHERE id = ?",
                    (version_id,),
                )
                self.conn.commit()
            raise RuntimeError(f"Health check failed for version '{version_id}'; rolled back automatically.")

        self.audit.log(
            "kb_version_activated",
            version_id=version_id,
            actor=actor,
        )

        return {
            "status": "active",
            "version_id": version_id,
            "activated_at": now,
        }

    def _run_health_check(self, version_id: str) -> bool:
        """Quick health check to ensure active index loads and searches successfully."""
        try:
            store = VersionedVectorStore.load(
                version_id=version_id,
                storage_dir=self.versions_root,
                embedder=self.embedder,
            )
            res = store.search("test health check", top_k=1, score_threshold=0.0)
            return True
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Rollback (Admin Only)
    # ──────────────────────────────────────────────────────────────────────────

    def rollback_to(
        self,
        target_version_id: str,
        actor: str = "admin",
        user_role: str | Role = Role.ADMIN,
    ) -> KnowledgeVersion:
        """Roll back to a previous knowledge base version (Admin role only, strictly audited)."""
        role_enum = Role(user_role) if isinstance(user_role, str) else user_role
        if role_enum not in (Role.ADMIN, Role.SYSTEM):
            raise PermissionError(f"Role '{user_role}' denied. Only Admin can perform Knowledge Base rollback.")

        target = self.get_version(target_version_id)
        if not target:
            raise ValueError(f"Target rollback version '{target_version_id}' does not exist.")

        prev_active = self.get_active_version()
        now = self.clock.now(ZoneInfo("UTC")).isoformat()

        if prev_active:
            self.conn.execute(
                "UPDATE kb_versions SET is_active = 0, status = 'rolled_back' WHERE id = ?",
                (prev_active.id,),
            )

        self.conn.execute(
            "UPDATE kb_versions SET is_active = 1, status = 'active', "
            "activated_at = ?, activated_by = ? WHERE id = ?",
            (now, actor, target_version_id),
        )
        self.conn.commit()

        self.audit.log(
            "kb_version_rollback",
            from_version=prev_active.id if prev_active else None,
            to_version=target_version_id,
            actor=actor,
        )

        return self.get_version(target_version_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Version Accessors
    # ──────────────────────────────────────────────────────────────────────────

    def get_version(self, version_id: str) -> Optional[KnowledgeVersion]:
        row = self.conn.execute("SELECT * FROM kb_versions WHERE id = ?", (version_id,)).fetchone()
        if not row:
            return None
        return KnowledgeVersion(
            id=row["id"],
            index_path=row["index_path"],
            status=row["status"],
            quality_score=row["quality_score"],
            document_count=row["document_count"],
            chunk_count=row["chunk_count"],
            is_active=bool(row["is_active"]),
            activated_at=row["activated_at"],
            activated_by=row["activated_by"],
            created_at=row["created_at"],
        )

    def get_active_version(self) -> Optional[KnowledgeVersion]:
        row = self.conn.execute(
            "SELECT * FROM kb_versions WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return KnowledgeVersion(
            id=row["id"],
            index_path=row["index_path"],
            status=row["status"],
            quality_score=row["quality_score"],
            document_count=row["document_count"],
            chunk_count=row["chunk_count"],
            is_active=True,
            activated_at=row["activated_at"],
            activated_by=row["activated_by"],
            created_at=row["created_at"],
        )

    def load_active_store(self) -> Optional[VersionedVectorStore]:
        active_ver = self.get_active_version()
        if not active_ver:
            return None
        try:
            return VersionedVectorStore.load(
                version_id=active_ver.id,
                storage_dir=self.versions_root,
                embedder=self.embedder,
            )
        except Exception:
            return None

    def list_versions(self) -> List[KnowledgeVersion]:
        """Return all knowledge base versions sorted by created_at desc."""
        rows = self.conn.execute("SELECT * FROM kb_versions ORDER BY created_at DESC").fetchall()
        return [
            KnowledgeVersion(
                id=r["id"],
                document_count=r["document_count"],
                chunk_count=r["chunk_count"],
                storage_path=r["storage_path"],
                is_active=bool(r["is_active"]),
                state=KnowledgeVersionState(r["state"]),
                quality_score=r["quality_score"],
                activated_at=r["activated_at"],
                rolled_back_at=r["rolled_back_at"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def get_job_handlers(self) -> Dict[str, Callable[[dict | None], Any]]:
        """Return handlers for background jobs enqueued by KnowledgePipeline."""
        return {
            "activate_kb_version": lambda p: self.activate_version(
                version_id=p["version_id"], actor=p.get("actor", "job_queue"), force=True
            ),
            "retry_kb_update": lambda p: self.build_version(p["version_id"]),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return operational pipeline metrics."""
        docs_count = self.conn.execute("SELECT count(*) as c FROM kb_documents").fetchone()["c"]
        chunks_count = self.conn.execute("SELECT count(*) as c FROM kb_chunks").fetchone()["c"]
        quarantine_count = self.conn.execute("SELECT count(*) as c FROM kb_quarantine").fetchone()["c"]
        versions_count = self.conn.execute("SELECT count(*) as c FROM kb_versions").fetchone()["c"]
        active = self.get_active_version()

        return {
            "total_documents": docs_count,
            "total_chunks": chunks_count,
            "quarantined_files": quarantine_count,
            "total_versions": versions_count,
            "active_version": active.id if active else None,
            "active_quality_score": active.quality_score if active else 0.0,
        }
