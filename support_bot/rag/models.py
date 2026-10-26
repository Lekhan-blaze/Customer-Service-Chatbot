"""Data structures for Secure RAG retrieval, citations, and claim validation."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Citation:
    title: str
    version: int
    document_id: str
    source_location: str
    snippet: str

    def format(self) -> str:
        return f"[{self.title} (v{self.version})]"


@dataclass
class RagResponse:
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0
    is_refusal: bool = False
    is_clarification: bool = False
    target_policy_date: Optional[str] = None
    applied_filters: Dict[str, Any] = field(default_factory=dict)
    unsupported_claims_removed: int = 0
    injection_suppressed: bool = False
