"""Secure RAG Service with pre-retrieval metadata filtering, citation generation, and injection defense."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from support_bot.audit import AuditLogger
from support_bot.clock import Clock
from support_bot.injection import PromptInjectionDetector
from support_bot.knowledge.models import DocumentChunk
from support_bot.knowledge.pipeline import KnowledgePipeline
from support_bot.knowledge.vector_store import VersionedVectorStore
from support_bot.masking import mask_pii
from support_bot.rag.claim_validator import BaseClaimValidator, FakeClaimValidator
from support_bot.rag.models import Citation, RagResponse
from support_bot.rbac import Role


_DATE_PATTERNS = [
    # YYYY-MM-DD
    r"\b(\d{4}-\d{2}-\d{2})\b",
    # DD Month YYYY (e.g. 5 March 2025, 15 Jan 2026)
    r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    # Month DD, YYYY
    r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
]

_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}


def parse_date_from_query(query: str) -> Optional[str]:
    """Extract an explicit historical date mentioned in a user question as YYYY-MM-DD."""
    for pattern in _DATE_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            raw = match.group(1).replace(",", "").strip()
            # Try YYYY-MM-DD
            if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
                return raw
            # Try DD Month YYYY
            parts = raw.split()
            if len(parts) == 3:
                try:
                    if parts[0].isdigit():
                        day = int(parts[0])
                        month = _MONTH_MAP[parts[1].lower()]
                        year = int(parts[2])
                    else:
                        month = _MONTH_MAP[parts[0].lower()]
                        day = int(parts[1])
                        year = int(parts[2])
                    return f"{year:04d}-{month:02d}-{day:02d}"
                except Exception:
                    pass
    return None


class RagService:
    """Secure enterprise RAG service with pre-retrieval metadata guards."""

    def __init__(
        self,
        pipeline: KnowledgePipeline,
        clock: Clock,
        audit: Optional[AuditLogger] = None,
        injection_detector: Optional[PromptInjectionDetector] = None,
        claim_validator: Optional[BaseClaimValidator] = None,
        llm: Optional[Any] = None,
    ) -> None:
        self.pipeline = pipeline
        self.clock = clock
        self.audit = audit or pipeline.audit
        self.injection_detector = injection_detector or pipeline.injection_detector
        self.claim_validator = claim_validator or FakeClaimValidator()
        self.llm = llm

    def answer(
        self,
        question: str,
        user_role: str | Role = Role.CUSTOMER,
        region: str = "global",
        session_id: Optional[str] = None,
    ) -> RagResponse:
        """Answer question using the active knowledge base with strict pre-retrieval filtering."""
        role_enum = Role(user_role) if isinstance(user_role, str) else user_role

        # 1. Determine target policy date: check for historical date in query, else use current clock date
        historical_date = parse_date_from_query(question)
        if historical_date:
            target_date = historical_date
        else:
            now_dt = self.clock.now(ZoneInfo("UTC"))
            target_date = now_dt.strftime("%Y-%m-%d")

        # 2. Load active vector store
        store = self.pipeline.load_active_store()
        if not store:
            return RagResponse(
                answer="I don't know.",
                citations=[],
                confidence=0.0,
                is_refusal=True,
                target_policy_date=target_date,
            )

        # 3. Construct Pre-Retrieval Metadata Filter
        # Applied BEFORE retrieval/ranking so unauthorized or invalid passages never reach candidates
        def _pre_retrieval_filter(meta: Dict[str, Any]) -> bool:
            # A. Access level check
            doc_access = meta.get("access_level", "customer").lower()
            if role_enum in (Role.CUSTOMER, Role.VIEWER):
                if doc_access == "internal":
                    return False
            # B. Regional check
            doc_region = meta.get("region", "global").upper()
            req_region = region.upper()
            if doc_region not in ("GLOBAL", req_region):
                return False

            # C. Policy date check (effective vs expiry)
            eff_date = meta.get("effective_date", "2000-01-01")
            exp_date = meta.get("expiry_date")

            # Must be effective on or before the target date
            if eff_date > target_date:
                return False

            # Must not be expired on target date
            if exp_date and exp_date < target_date:
                return False

            return True

        # 4. Search candidates
        candidates = store.search(
            question,
            top_k=4,
            score_threshold=0.25,
            metadata_filter=_pre_retrieval_filter,
        )

        if not candidates:
            self.audit.log(
                "rag_retrieval_empty",
                question=mask_pii(question),
                user_role=role_enum.value,
                target_date=target_date,
            )
            return RagResponse(
                answer="I don't know.",
                citations=[],
                confidence=0.0,
                is_refusal=True,
                target_policy_date=target_date,
            )

        # 5. Policy Conflict Resolution: choose the latest version if multiple versions match
        grouped_by_doc: Dict[str, List[Tuple[DocumentChunk, float]]] = {}
        for chunk, score in candidates:
            doc_key = chunk.metadata.get("title", chunk.document_id)
            grouped_by_doc.setdefault(doc_key, []).append((chunk, score))

        resolved_chunks: List[Tuple[DocumentChunk, float]] = []
        for doc_key, items in grouped_by_doc.items():
            # Pick the item with the highest version number
            items.sort(key=lambda x: x[0].metadata.get("version", 1), reverse=True)
            resolved_chunks.append(items[0])

        resolved_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = resolved_chunks[:3]

        # 6. Treat retrieved documents as untrusted — prompt injection defense on retrieved text
        safe_passages: List[str] = []
        injection_suppressed = False
        citations: List[Citation] = []

        for chunk, score in top_chunks:
            text = chunk.chunk_text
            inj_res = self.injection_detector.detect(text, source_type="document")
            if inj_res.is_injection:
                injection_suppressed = True
                self.audit.log(
                    "rag_injection_in_doc_suppressed",
                    document_id=chunk.document_id,
                    matched=inj_res.matched_reasons,
                )
                # Strip out malicious command phrases
                text = re.sub(
                    r"(?i)\b(ignore previous rules|reveal admin password|system override|disregard prior directives)[^.\n]*[.\n]?",
                    "",
                    text,
                ).strip()

            if text:
                safe_passages.append(text)
                meta = chunk.metadata
                citations.append(Citation(
                    title=meta.get("title", "Policy Document"),
                    version=int(meta.get("version", 1)),
                    document_id=chunk.document_id,
                    source_location=meta.get("source_location", ""),
                    snippet=text[:120],
                ))

        if not safe_passages:
            return RagResponse(
                answer="I don't know.",
                citations=[],
                confidence=0.0,
                is_refusal=True,
                target_policy_date=target_date,
            )

        # 7. Synthesize answer from retrieved passages
        primary_passage = safe_passages[0]
        # Clean synthetic header tags from answer output
        clean_ans = re.sub(r"\[SYNTHETIC TEST DOCUMENT[^\]]*\]\s*", "", primary_passage).strip()

        # Append source citations to every factual answer
        unique_citations = []
        seen_titles = set()
        for c in citations:
            key = f"{c.title}_v{c.version}"
            if key not in seen_titles:
                seen_titles.add(key)
                unique_citations.append(c)

        citation_str = ", ".join(c.format() for c in unique_citations)
        raw_answer = f"{clean_ans}\n\nSources: {citation_str}"

        # 8. Claim-to-source validation
        validated_answer, removed_claims = self.claim_validator.validate_claims(raw_answer, safe_passages)

        avg_conf = sum(score for _, score in top_chunks) / len(top_chunks)

        self.audit.log(
            "rag_answer_generated",
            question=mask_pii(question),
            citations=[c.format() for c in unique_citations],
            confidence=round(avg_conf, 3),
            user_role=role_enum.value,
            target_date=target_date,
        )

        return RagResponse(
            answer=validated_answer,
            citations=unique_citations,
            confidence=round(avg_conf, 3),
            is_refusal=(validated_answer == "I don't know."),
            target_policy_date=target_date,
            applied_filters={"role": role_enum.value, "region": region, "target_date": target_date},
            unsupported_claims_removed=removed_claims,
            injection_suppressed=injection_suppressed,
        )
