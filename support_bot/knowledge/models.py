"""Data models, state definitions, and transitions for Knowledge Base management."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class DocumentState(str, Enum):
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    QUARANTINED = "quarantined"
    TESTED = "tested"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


LEGAL_STATE_TRANSITIONS: Dict[DocumentState, Set[DocumentState]] = {
    DocumentState.DISCOVERED: {DocumentState.VALIDATED, DocumentState.QUARANTINED, DocumentState.FAILED},
    DocumentState.VALIDATED: {DocumentState.TESTED, DocumentState.FAILED},
    DocumentState.QUARANTINED: set(),  # terminal until manual re-scan
    DocumentState.TESTED: {DocumentState.APPROVED, DocumentState.FAILED},
    DocumentState.APPROVED: {DocumentState.SCHEDULED, DocumentState.ACTIVE, DocumentState.FAILED},
    DocumentState.SCHEDULED: {DocumentState.ACTIVE, DocumentState.FAILED},
    DocumentState.ACTIVE: {DocumentState.ROLLED_BACK, DocumentState.FAILED},
    DocumentState.FAILED: {DocumentState.DISCOVERED},  # allow retry
    DocumentState.ROLLED_BACK: {DocumentState.ACTIVE},  # allow re-activation
}


def validate_transition(current: str | DocumentState, target: str | DocumentState) -> None:
    """Validate that transition from current state to target state is legally permissible."""
    curr_state = DocumentState(current) if isinstance(current, str) else current
    target_state = DocumentState(target) if isinstance(target, str) else target

    allowed = LEGAL_STATE_TRANSITIONS.get(curr_state, set())
    if target_state not in allowed:
        raise ValueError(
            f"Illegal state transition from '{curr_state.value}' to '{target_state.value}'. "
            f"Allowed next states: {[s.value for s in allowed]}"
        )


@dataclass
class DocumentMetadata:
    id: str
    version: int
    title: str
    product: str = "general"
    region: str = "global"
    access_level: str = "customer"  # public, customer, internal
    effective_date: str = "2026-01-01"
    expiry_date: Optional[str] = None
    source_location: str = ""
    content_hash: str = ""
    status: str = DocumentState.DISCOVERED.value
    created_at: str = ""
    updated_at: str = ""


@dataclass
class DocumentChunk:
    id: Optional[int]
    document_id: str
    chunk_index: int
    chunk_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeVersion:
    id: str
    index_path: str
    status: str = "staged"
    quality_score: float = 0.0
    document_count: int = 0
    chunk_count: int = 0
    is_active: bool = False
    activated_at: Optional[str] = None
    activated_by: Optional[str] = None
    created_at: str = ""


@dataclass
class QuarantineRecord:
    id: Optional[int]
    document_id: Optional[str]
    file_path: str
    reason: str
    risk_score: float
    quarantined_at: str
    actor: str = "system"


@dataclass
class QualityGateResult:
    passed: bool
    hit_rate: float
    grounding_score: float
    total_evaluations: int
    details: List[Dict[str, Any]] = field(default_factory=list)
    rejection_reason: Optional[str] = None
